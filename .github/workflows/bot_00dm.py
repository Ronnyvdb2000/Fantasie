#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00dm.py  —  DUAL MOMENTUM ENGINE v1.0
Universe → top 20% omzetgroei → top 20% relatieve sterkte
→ koop uitbraak boven 52-weeks high → verkoop bij slot onder MA50

Verwant aan Gary Antonacci Dual Momentum + IBD Big Cap 20.
Zelfde structuur, tickerbestanden en Telegram output als bot_00xxxV2.py.

Pipeline:
  Stap 1: Filter top 20% omzetgroei (revenue YoY via yfinance)
  Stap 2: Filter top 20% relatieve sterkte (RS vs. universe)
  Stap 3: Koop bij uitbraak boven 52-weekse high op volume
  Stap 4: Verkoop bij slotkoers onder MA50

Score systeem (0-5):
  1. Top 20% omzetgroei universe
  2. Top 20% RS rating universe
  3. Prijs binnen 5% van of boven 52-weekse high
  4. Uitbraak boven 52w high op verhoogd volume (≥1.5×)
  5. Prijs boven MA50 (exit filter omgekeerd)

Gebruik:
  python bot_00dm.py          # live rapport
  python bot_00dm.py backtest # backtest modus

GitHub Actions: dagelijks om 22:10 UTC
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL        = 50_000.0
MAX_POSITIONS        = 10
MIN_CASH_RATIO       = 0.10
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001

TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
MAX_HOLD_DAYS        = 120   # langere houdduur — trend following

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGES = {
    "041 Benelux":     "tickers_041x.txt",
    "042 Parijs":      "tickers_042x.txt",
    "043 Frankfurt":   "tickers_043x.txt",
    "044 Spanje/Port": "tickers_044x.txt",
    "045 Londen":      "tickers_045x.txt",
    "046 Milaan":      "tickers_046x.txt",
    "047 Toronto":     "tickers_047x.txt",
    "048 Nasdaq/NYSE": "tickers_048x.txt",
}

DM_CFG = {
    # Pipeline filters
    "top_pct":              20.0,   # top X% voor omzetgroei en RS
    # 52-weekse high
    "high52w_lookback":     252,
    "high_proximity_pct":   5.0,    # binnen 5% van 52w high
    # Uitbraak
    "breakout_vol_mult":    1.5,    # volume ≥ 1.5× gemiddelde
    "vol_ma_period":        50,
    # Exit
    "ma_exit":              50,     # verkoop bij slot onder MA50
    # ATR voor sizing
    "atr_period":           14,
    # Stop (initieel, vóór MA50 exit)
    "stop_pct":             10.0,   # hard stop 10% onder entry
    # Rapportage
    "min_score":            3,      # min 3/5 om te rapporteren
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES  (identiek aan bot_00xxxV2.py)
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")

def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# ATR SIZING  (identiek aan bot_00xxxV2.py)
# ============================================================

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period
            + valid.iloc[i] / period
        )
    return result

def bereken_positie(
    portfolio_waarde: float,
    entry_prijs:      float,
    stop_prijs:       float,
    risico_pct:       float = RISICO_PCT_PER_TRADE,
) -> Tuple[int, float]:
    risico_eur   = portfolio_waarde * risico_pct
    stop_afstand = entry_prijs - stop_prijs
    if stop_afstand <= 0:
        return 0, 0.0
    aandelen    = max(1, int(risico_eur / stop_afstand))
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, max_verlies

def sizing_tekst(ticker, prijs, stop, ma50, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    investering = round(entry * aandelen, 2)
    slip_est    = round(entry * SLIPPAGE_PCT * aandelen * 2, 2)
    kosten      = round(trade_cost(investering), 2)
    return (
        f"  📐 *Sizing:*\n"
        f"  Entry geschat : EUR{entry:.2f}  (uitbraak boven 52w high)\n"
        f"  Stop-Loss     : EUR{stop:.2f}  ({DM_CFG['stop_pct']:.0f}% hard stop)\n"
        f"  Exit trigger  : slot onder MA{DM_CFG['ma_exit']} (nu EUR{ma50:.2f})\n"
        f"  Aandelen      : {aandelen} stuks\n"
        f"  Investering   : EUR{investering:,.2f}\n"
        f"  Max verlies   : EUR{max_loss:,.2f}  (5% portfolio)\n"
        f"  Slippage est. : EUR{slip_est:.2f}\n"
        f"  Kosten        : EUR{kosten:.2f}"
    )


# ============================================================
# DATA DOWNLOAD  (identiek aan bot_00xxxV2.py)
# ============================================================

def _normalise(df_raw, ticker: str) -> Optional[pd.DataFrame]:
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return None
    df = df_raw.copy().dropna(how="all")
    if df.empty:
        return None
    if df.index.name in ("Date", "Datetime") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Date" in df.columns:
        df = df.loc[:, ~df.columns.duplicated()]
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    df["Ticker"] = ticker
    return df


def download_history(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    kwargs = dict(tickers=tickers, auto_adjust=True, group_by="ticker",
                  progress=False, threads=True, period=period)
    frames = []
    try:
        data = yf.download(**kwargs)
    except Exception as e:
        print(f"[WARN] Batch mislukt ({e}), probeer 1-voor-1...")
        data = pd.DataFrame()

    if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_level = 1
            for lvl in range(data.columns.nlevels):
                if any(t in set(data.columns.get_level_values(lvl)) for t in tickers):
                    ticker_level = lvl
                    break
            available = set(data.columns.get_level_values(ticker_level))
            for t in tickers:
                if t not in available:
                    continue
                try:
                    norm = _normalise(data.xs(t, axis=1, level=ticker_level).copy(), t)
                    if norm is not None:
                        frames.append(norm)
                except Exception as e:
                    print(f"[WARN] {t}: fout ({e})")
        else:
            norm = _normalise(data, tickers[0])
            if norm is not None:
                frames.append(norm)

    if not frames:
        for t in tickers:
            try:
                raw = yf.download(t, period=period, auto_adjust=True, progress=False)
                if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                    continue
                if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                norm = _normalise(raw, t)
                if norm is not None:
                    frames.append(norm)
                time.sleep(0.2)
            except Exception as e:
                print(f"[WARN] {t}: mislukt ({e})")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# STAP 1: OMZETGROEI RANKING
# ============================================================

def compute_revenue_growth(tickers: List[str]) -> Dict[str, float]:
    """
    Haalt jaarlijkse omzetgroei op via yfinance.
    Geeft dict terug: ticker → omzetgroei % YoY (meest recente jaar).
    Alleen tickers met positieve groei worden meegenomen.
    """
    groei: Dict[str, float] = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            t  = yf.Ticker(ticker)
            fs = t.financials  # jaarlijks income statement
            if fs is None or fs.empty:
                continue
            # Zoek omzetregel
            rev_row = None
            for lbl in ["Total Revenue", "Revenue", "Net Revenue"]:
                if lbl in fs.index:
                    rev_row = fs.loc[lbl].dropna().sort_index()
                    break
            if rev_row is None or len(rev_row) < 2:
                continue
            rev_now  = float(rev_row.iloc[-1])
            rev_prev = float(rev_row.iloc[-2])
            if rev_prev > 0:
                groei[ticker] = round((rev_now - rev_prev) / rev_prev * 100, 1)
        except Exception:
            pass
        if i % 25 == 0:
            print(f"  Omzetgroei: {i}/{len(tickers)} opgehaald...")
        time.sleep(0.25)
    return groei


def top_pct_filter(scores: Dict[str, float], pct: float = 20.0) -> List[str]:
    """Geeft tickers terug die in de top X% vallen."""
    if not scores:
        return []
    threshold = np.percentile(list(scores.values()), 100 - pct)
    return [t for t, v in scores.items() if v >= threshold]


# ============================================================
# STAP 2: RELATIEVE STERKTE RANKING
# ============================================================

def compute_rs_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """
    RS Rating: gewogen 12-maands prijsprestatie als percentiel.
    Identiek aan bot_00ms.py en bot_00cs.py.
    """
    perf: Dict[str, float] = {}
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date")
        if len(g) < 252:
            continue
        c_now = safe_float(g["Close"].iloc[-1])
        c_3m  = safe_float(g["Close"].iloc[-63])
        c_12m = safe_float(g["Close"].iloc[-252])
        if math.isnan(c_now) or math.isnan(c_12m) or c_12m <= 0 or c_3m <= 0:
            continue
        perf[ticker] = 0.4 * (c_now - c_3m) / c_3m + 0.6 * (c_now - c_12m) / c_12m

    rs_map: Dict[str, float] = {}
    if not perf:
        return rs_map
    values = sorted(perf.values())
    n = len(values)
    for ticker, p in perf.items():
        rank = sum(1 for v in values if v < p)
        rs_map[ticker] = round((rank / n) * 99, 1)
    return rs_map


# ============================================================
# DUAL MOMENTUM SIGNAAL
# ============================================================

@dataclass
class DMSignaal:
    ticker:          str
    price:           float
    score:           int           # 0-5
    score_labels:    List[str]
    # Pipeline metrics
    rev_growth:      float         # omzetgroei %
    rev_rank:        float         # percentiel in universe
    rs:              float         # RS rating 0-99
    rs_rank:         float         # percentiel in universe
    # Technisch
    high52w:         float
    pct_from_high:   float
    breakout:        bool          # prijs boven 52w high
    breakout_vol:    float         # volume ratio
    ma50:            float
    above_ma50:      bool
    # Sizing
    stop:            float
    total_score:     float


def analyse_ticker(
    ticker:       str,
    g:            pd.DataFrame,
    rs_ratings:   Dict[str, float],
    rev_growth:   Dict[str, float],
    rs_top20:     List[str],
    rev_top20:    List[str],
) -> Optional[DMSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < DM_CFG["ma_exit"] + 10:
            return None

        close  = g["Close"]
        high   = g["High"]
        volume = g["Volume"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        # ── Indicatoren ───────────────────────────────────────────
        ma50    = safe_float(close.rolling(DM_CFG["ma_exit"]).mean().iloc[-1])
        vol_ma  = volume.rolling(DM_CFG["vol_ma_period"]).mean()
        vol_now = safe_float(volume.iloc[-1])
        vol_avg = safe_float(vol_ma.iloc[-1])
        vol_ratio = (vol_now / vol_avg) if vol_avg > 0 else 0.0

        lb   = min(DM_CFG["high52w_lookback"], len(close))
        h52w = float(close.iloc[-lb:].max())

        pct_from_high = ((h52w - current_price) / h52w * 100) if h52w > 0 else 100.0
        breakout      = current_price > h52w or pct_from_high <= 0.5
        above_ma50    = not math.isnan(ma50) and current_price > ma50

        # Stop: 10% onder entry
        stop = current_price * (1 - DM_CFG["stop_pct"] / 100)

        # Pipeline metrics
        rv       = rev_growth.get(ticker, float("nan"))
        rs       = rs_ratings.get(ticker, 0.0)
        in_rev20 = ticker in rev_top20
        in_rs20  = ticker in rs_top20

        # Percentiel positie
        all_rs  = list(rs_ratings.values())
        all_rv  = [v for v in rev_growth.values() if not math.isnan(v)]
        rs_pct  = (sum(1 for x in all_rs if x < rs) / len(all_rs) * 100) if all_rs else 0.0
        rv_pct  = (sum(1 for x in all_rv if x < rv) / len(all_rv) * 100) if all_rv and not math.isnan(rv) else 0.0

        # ── Score 0-5 ─────────────────────────────────────────────
        score  = 0
        labels = []

        def chk(ok: bool, ok_msg: str, fail_msg: str):
            nonlocal score
            if ok:
                score += 1
                labels.append(f"✓ {ok_msg}")
            else:
                labels.append(f"✗ {fail_msg}")

        # 1. Top 20% omzetgroei
        chk(in_rev20,
            f"omzetgroei +{rv:.1f}% (top {DM_CFG['top_pct']:.0f}% universe)",
            f"omzetgroei {rv:.1f}% (niet top {DM_CFG['top_pct']:.0f}%)" if not math.isnan(rv) else "geen omzetdata")

        # 2. Top 20% RS
        chk(in_rs20,
            f"RS={rs:.0f} — top {DM_CFG['top_pct']:.0f}% universe ({rs_pct:.0f}e percentiel)",
            f"RS={rs:.0f} — niet top {DM_CFG['top_pct']:.0f}% ({rs_pct:.0f}e percentiel)")

        # 3. Nabij of boven 52-weekse high
        chk(pct_from_high <= DM_CFG["high_proximity_pct"],
            f"{pct_from_high:.1f}% van 52w high ({h52w:.2f})",
            f"{pct_from_high:.1f}% onder 52w high (>{DM_CFG['high_proximity_pct']:.0f}%)")

        # 4. Uitbraak boven 52w high op volume
        chk(breakout and vol_ratio >= DM_CFG["breakout_vol_mult"],
            f"uitbraak boven {h52w:.2f} op {vol_ratio:.1f}× volume",
            f"geen uitbraak (vol={vol_ratio:.1f}×, high={h52w:.2f})" if not breakout
            else f"uitbraak maar volume zwak ({vol_ratio:.1f}×)")

        # 5. Boven MA50 (exit filter)
        chk(above_ma50,
            f"boven MA{DM_CFG['ma_exit']} ({ma50:.2f}) — geen exit",
            f"onder MA{DM_CFG['ma_exit']} ({ma50:.2f}) — EXIT SIGNAAL")

        if score < DM_CFG["min_score"]:
            return None

        # Ranking: pipeline kwaliteit + breakout wegen zwaarder
        total_score = (
            score * 10
            + (rs * 0.3 if in_rs20 else 0)
            + (rv * 0.1 if in_rev20 and not math.isnan(rv) else 0)
            + (15 if breakout else 0)
            + (5 if vol_ratio >= DM_CFG["breakout_vol_mult"] else 0)
        )

        return DMSignaal(
            ticker=ticker,
            price=round(current_price, 2),
            score=score,
            score_labels=labels,
            rev_growth=round(rv, 1) if not math.isnan(rv) else 0.0,
            rev_rank=round(rv_pct, 1),
            rs=round(rs, 1),
            rs_rank=round(rs_pct, 1),
            high52w=round(h52w, 2),
            pct_from_high=round(pct_from_high, 1),
            breakout=breakout,
            breakout_vol=round(vol_ratio, 2),
            ma50=round(ma50, 2) if not math.isnan(ma50) else 0.0,
            above_ma50=above_ma50,
            stop=round(stop, 2),
            total_score=round(total_score, 1),
        )

    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# EXIT SIGNALEN
# ============================================================

def check_exits(
    positions: Dict[str, Dict],
    df_ex:     pd.DataFrame,
) -> List[Dict]:
    """
    Controleert open posities op exit: slot onder MA50.
    """
    exits = []
    for ticker, pos in positions.items():
        g = df_ex[df_ex["Ticker"] == ticker].sort_values("Date")
        if g.empty:
            continue
        close = g["Close"]
        ma50  = close.rolling(DM_CFG["ma_exit"]).mean()
        c_now = safe_float(close.iloc[-1])
        m_now = safe_float(ma50.iloc[-1])
        if math.isnan(c_now) or math.isnan(m_now):
            continue
        if c_now < m_now:
            exits.append({
                "ticker": ticker,
                "price":  round(c_now, 2),
                "ma50":   round(m_now, 2),
                "reason": f"Slot {c_now:.2f} < MA{DM_CFG['ma_exit']} {m_now:.2f}",
                **pos,
            })
    return exits


# ============================================================
# TELEGRAM OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 5) -> str:
    filled = "█" * score
    empty  = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def format_dm_per_exchange(
    exchange_name:    str,
    signalen:         List[DMSignaal],
    exits:            List[Dict],
    portfolio_waarde: float,
) -> Tuple[str, str]:
    nu = today_str()

    def koop_blok(sigs: List[DMSignaal]) -> str:
        if not sigs:
            return "_Geen kandidaten_"
        lines = []
        for s in sigs:
            lines.append(
                f"• `{s.ticker}` | Score: {_score_bar(s.score)} | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                f"  Omzet: +{s.rev_growth:.1f}% ({s.rev_rank:.0f}e pct) | "
                f"RS:{s.rs:.0f} ({s.rs_rank:.0f}e pct)\n"
                f"  52w high: EUR{s.high52w:.2f} ({s.pct_from_high:.1f}% eronder) | "
                f"Vol: {s.breakout_vol:.1f}×\n"
                + "\n".join(f"  {lbl}" for lbl in s.score_labels) + "\n"
                + sizing_tekst(s.ticker, s.price, s.stop, s.ma50, portfolio_waarde)
            )
        return "\n\n".join(lines)

    def exit_blok(ex: List[Dict]) -> str:
        if not ex:
            return "_Geen exit signalen_"
        lines = []
        for e in ex:
            lines.append(
                f"• `{e['ticker']}` 🔴 *VERKOOP*\n"
                f"  {e['reason']}\n"
                f"  Entry was: EUR{e.get('entry_price', 0):.2f} | "
                f"Commando: `/sell {e['ticker']} {e['price']:.2f}`"
            )
        return "\n\n".join(lines)

    top2   = signalen[:2]
    score5 = [s for s in signalen if s.score == 5]
    score4 = [s for s in signalen if s.score == 4]
    score3 = [s for s in signalen if s.score == 3]

    deel1 = "\n\n".join([
        f"🚀 *DUAL MOMENTUM — {exchange_name}*",
        f"_{nu} | Top 20% omzet × top 20% RS → uitbraak 52w high_",
        f"_Exit: slot onder MA{DM_CFG['ma_exit']}_",
        "─────────────────────────────",
        f"🏆 *TOP 2 HOOGSTE POTENTIEEL:*",
        koop_blok(top2) if top2 else "_Geen kandidaten vandaag_",
        "─────────────────────────────",
        f"🔴 *EXIT SIGNALEN (slot < MA{DM_CFG['ma_exit']}):*",
        exit_blok(exits),
        "─────────────────────────────",
        f"⭐ *PERFECTE SCORE (5/5):*",
        koop_blok(score5) if score5 else "_Geen_",
    ])

    deel2_parts = [
        f"🚀 *DUAL MOMENTUM — {exchange_name} (2/2)*",
        "",
        f"⚡ *STERK (4/5):*",
        koop_blok(score4) if score4 else "_Geen_",
        "",
        f"📊 *WATCHLIST (3/5):*",
        koop_blok(score3) if score3 else "_Geen_",
        "",
        "─────────────────────────────",
        f"📊 *SAMENVATTING:*",
        f"  Kandidaten (score≥{DM_CFG['min_score']}) : {len(signalen)}",
        f"  Uitbraken                 : {sum(s.breakout for s in signalen)}",
        f"  Exit signalen             : {len(exits)}",
        f"  Score 5/5                 : {len(score5)}",
        f"  Score 4/5                 : {len(score4)}",
        f"  Score 3/5                 : {len(score3)}",
        "",
        "⚙️ *DUAL MOMENTUM PARAMETERS:*",
        f"_Stap 1: top {DM_CFG['top_pct']:.0f}% omzetgroei YoY_",
        f"_Stap 2: top {DM_CFG['top_pct']:.0f}% relatieve sterkte (RS)_",
        f"_Stap 3: uitbraak boven 52w high op ≥{DM_CFG['breakout_vol_mult']}× volume_",
        f"_Stap 4: verkoop bij slot onder MA{DM_CFG['ma_exit']}_",
        f"_Hard stop: {DM_CFG['stop_pct']:.0f}% | Risico: 5% portfolio_",
    ]

    return deel1, "\n".join(deel2_parts)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"DUAL MOMENTUM — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        if tlist:
            exchange_tickers[ex_name] = tlist
            all_tickers.extend(tlist)
            print(f"  {ex_name}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print("Koersdata downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return
    print(f"Data geladen: {df['Ticker'].nunique()} tickers")

    # ── Stap 1: Omzetgroei ophalen ───────────────────────────────
    print(f"\nStap 1: Omzetgroei ophalen ({len(all_tickers)} tickers)...")
    rev_growth = compute_revenue_growth(all_tickers)
    print(f"  Omzetgroei beschikbaar: {len(rev_growth)} tickers")

    # ── Stap 2: RS berekenen ─────────────────────────────────────
    print("Stap 2: RS ratings berekenen...")
    rs_ratings = compute_rs_ratings(df)
    print(f"  RS ratings: {len(rs_ratings)} tickers")

    # ── Top 20% filters ──────────────────────────────────────────
    rev_top20 = top_pct_filter(rev_growth, DM_CFG["top_pct"])
    rs_top20  = top_pct_filter(rs_ratings, DM_CFG["top_pct"])
    print(f"  Top {DM_CFG['top_pct']:.0f}% omzetgroei : {len(rev_top20)} tickers")
    print(f"  Top {DM_CFG['top_pct']:.0f}% RS         : {len(rs_top20)} tickers")

    # Overlap = kandidaten voor technische analyse
    kandidaten = sorted(set(rev_top20) & set(rs_top20))
    print(f"  Overlap (beide top 20%) : {len(kandidaten)} tickers")

    portfolio_waarde = START_CAPITAL

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name}...")
        tlist_set = set(tlist)
        df_ex     = df[df["Ticker"].isin(tlist_set)].copy()

        # Kandidaten die ook in deze exchange zitten
        ex_kandidaten = [t for t in kandidaten if t in tlist_set]
        print(f"  Pipeline kandidaten in exchange: {len(ex_kandidaten)}")

        signalen: List[DMSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(
                ticker, group, rs_ratings, rev_growth, rs_top20, rev_top20
            )
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: {sig.score}/5 | "
                    f"omzet+{sig.rev_growth:.1f}% | RS={sig.rs:.0f} | "
                    f"{'BREAKOUT' if sig.breakout else 'setup'}"
                )

        signalen.sort(key=lambda s: s.total_score, reverse=True)

        # Exit check op lege posities (geen live portfolio in deze versie)
        exits: List[Dict] = []

        print(f"  → {len(signalen)} Dual Momentum kandidaten")

        deel1, deel2 = format_dm_per_exchange(ex_name, signalen, exits, portfolio_waarde)
        send_telegram_message(deel1)
        time.sleep(1)
        send_telegram_message(deel2)

        if signalen:
            _log_csv(signalen, ex_name)

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# CSV LOGGING
# ============================================================

def _log_csv(signalen: List[DMSignaal], exchange: str):
    fname  = f"dm_signalen_{exchange.split()[0]}_{today_str()}.csv"
    header = ["datum","exchange","ticker","score","price","rev_growth","rev_rank",
              "rs","rs_rank","high52w","pct_from_high","breakout","breakout_vol",
              "ma50","above_ma50","stop","total_score"]
    ensure_csv_header(fname, header)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in signalen:
            w.writerow([
                today_str(), exchange, s.ticker, s.score, s.price,
                s.rev_growth, s.rev_rank, s.rs, s.rs_rank,
                s.high52w, s.pct_from_high, s.breakout, s.breakout_vol,
                s.ma50, s.above_ma50, s.stop, s.total_score,
            ])
    print(f"  CSV: {fname}")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"DUAL MOMENTUM BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"NB: omzetgroei backtest gebruikt technische proxy (12m prijsprestatie)")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for path in EXCHANGES.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    all_dates = sorted(df["Date"].dt.date.unique())
    rs_ratings = compute_rs_ratings(df)
    rs_top20   = top_pct_filter(rs_ratings, DM_CFG["top_pct"])

    # Proxy voor omzetgroei in backtest: 12m prijsprestatie top 20%
    perf_proxy: Dict[str, float] = {}
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date")
        if len(g) < 252:
            continue
        c_now  = safe_float(g["Close"].iloc[-1])
        c_12m  = safe_float(g["Close"].iloc[-252])
        if c_12m > 0:
            perf_proxy[ticker] = (c_now - c_12m) / c_12m * 100
    rev_top20_proxy = top_pct_filter(perf_proxy, DM_CFG["top_pct"])

    cash      = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []

    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(scan_date)].copy()

        price_map: Dict[str, float] = {}
        for _, row in day_df.iterrows():
            t = row.get("Ticker")
            c = safe_float(row.get("Close"))
            if t and not math.isnan(c):
                price_map[t] = c

        # Exits: slot onder MA50
        for ticker, pos in list(positions.items()):
            pos["days"] += 1
            if ticker not in price_map:
                continue
            g     = df_hist[df_hist["Ticker"] == ticker].sort_values("Date")
            close = g["Close"]
            ma50  = safe_float(close.rolling(DM_CFG["ma_exit"]).mean().iloc[-1])
            c_now = price_map[ticker]
            reason = None

            if not math.isnan(ma50) and c_now < ma50:
                reason = f"Slot<MA{DM_CFG['ma_exit']} ({c_now:.2f}<{ma50:.2f})"
            elif c_now <= pos["stop"]:
                reason = f"Hard SL ({pos['stop']:.2f})"
            elif pos["days"] >= MAX_HOLD_DAYS:
                reason = f"Time ({pos['days']}d)"

            if reason:
                exit_slip = c_now * (1 - SLIPPAGE_PCT)
                gross     = exit_slip * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax       = pnl * TAX_RATE if pnl > 0 else 0.0
                cash     += gross - cost - tax
                trades.append({
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   scan_date.isoformat(),
                    "ticker":      ticker,
                    "score":       pos["score"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  round(exit_slip, 4),
                    "size":        pos["size"],
                    "pnl":         round(pnl, 2),
                    "tax":         round(tax, 2),
                    "net":         round(pnl - tax, 2),
                    "reason":      reason,
                    "days":        pos["days"],
                })
                del positions[ticker]

        # Entries: pipeline filter + uitbraak
        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            if ticker not in rs_top20 or ticker not in rev_top20_proxy:
                continue
            sig = analyse_ticker(
                ticker, group, rs_ratings, perf_proxy, rs_top20, rev_top20_proxy
            )
            if not sig or not sig.breakout:
                continue

            entry      = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, _ = bereken_positie(cash, entry, sig.stop)
            if aandelen <= 0:
                continue
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue

            cash -= investering
            positions[ticker] = {
                "entry_date":  scan_date,
                "entry_price": round(entry, 4),
                "size":        aandelen,
                "stop":        sig.stop,
                "score":       sig.score,
                "days":        0,
                "cost":        trade_cost(investering),
            }

    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("dm_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final_val = cash + sum(
            price_map.get(t, p["entry_price"]) * p["size"]
            for t, p in positions.items()
        )
        print(f"\n{'='*60}")
        print(f"Startkapitaal    : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal     : EUR{final_val:>12,.2f}")
        print(f"Totaal rendement : {(final_val-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
        print(f"Opgeslagen       : dm_backtest_trades.csv")
        print(f"{'='*60}")
    else:
        print("Geen trades gegenereerd.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
