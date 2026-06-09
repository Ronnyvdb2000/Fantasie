#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00vcp.py  —  VCP ENGINE v1.0
Volatility Contraction Pattern — Minervini's 'Narrowing' techniek.
Zelfde structuur, tickerbestanden en Telegram output als bot_00xxxV2.py.

Hoe VCP werkt:
  Na een stijging consolideert een aandeel in steeds smallere correcties:
    Correctie 1: -15% op hoog volume  (1e contractie)
    Correctie 2: -10% op lager volume (2e contractie)
    Correctie 3:  -6% op laag volume  (3e contractie)
    Correctie 4:  -3% op minimaal vol (4e contractie — ideaal)
    → BREAKOUT boven pivot op 2-3× normaal volume

Score systeem (0-8):
  1. Minimum 2 VCP contracties gedetecteerd
  2. Elke correctie kleiner dan vorige (%)
  3. Elke correctie korter in tijd dan vorige
  4. Volume daalt bij elke correctie
  5. Laatste contractie ≤ 10% diep
  6. Prijs binnen 10% van pivot high
  7. Stage 2 trend (Close > MA50 > MA150 > MA200)
  8. Breakout boven pivot op verhoogd volume

Gebruik:
  python bot_00vcp.py          # live rapport
  python bot_00vcp.py backtest # backtest modus

GitHub Actions: dagelijks om 22:05 UTC
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
from dataclasses import dataclass, field
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
MAX_HOLD_DAYS        = 60

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

# VCP parameters
VCP_CFG = {
    # Contractie detectie
    "min_contracties":       2,      # minimum aantal VCP contracties
    "max_contracties":       5,      # maximum te zoeken contracties
    "min_correctie_pct":     3.0,    # minimale correctie om te tellen (%)
    "max_correctie_pct":     50.0,   # maximale correctie (anders geen VCP)
    "contractie_ratio":      0.80,   # elke correctie max 80% van vorige
    "tijd_ratio":            0.90,   # elke correctie max 90% van vorige duur
    # Laatste contractie
    "laatste_max_pct":       10.0,   # laatste correctie max 10% diep
    "pivot_proximity_pct":   10.0,   # prijs binnen 10% van pivot
    # Volume
    "vol_ma_period":         50,     # volume MA periode
    "vol_droogval_ratio":    0.80,   # volume bij contractie max 80% van vorige
    "breakout_vol_mult":     1.5,    # breakout volume ≥ 1.5× gemiddelde
    # Trend (Stage 2)
    "ma_fast":               50,
    "ma_mid":                150,
    "ma_slow":               200,
    # ATR
    "atr_period":            14,
    # Rapportage
    "min_score":             4,      # min score om te rapporteren (0-8)
    # Lookback voor VCP detectie
    "lookback_days":         120,    # zoek VCP in laatste 120 handelsdagen
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
# TECHNISCHE INDICATOREN
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

def sizing_tekst(ticker, prijs, stop, pivot, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    investering = round(entry * aandelen, 2)
    slip_est    = round(entry * SLIPPAGE_PCT * aandelen * 2, 2)
    kosten      = round(trade_cost(investering), 2)
    rr          = ((pivot - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  📐 *Sizing:*\n"
        f"  Entry geschat : EUR{entry:.2f}  (boven pivot)\n"
        f"  Stop-Loss     : EUR{stop:.2f}  (onder laatste low)\n"
        f"  Pivot (TP)    : EUR{pivot:.2f}\n"
        f"  R/R ratio     : {rr:.1f}:1\n"
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
# VCP KERN: CONTRACTIE DETECTIE
# ============================================================

@dataclass
class Contractie:
    nummer:      int
    high:        float    # top van de correctie
    low:         float    # bodem van de correctie
    pct:         float    # diepte in %
    duur:        int      # aantal handelsdagen
    vol_gem:     float    # gemiddeld volume tijdens correctie
    start_idx:   int
    end_idx:     int


@dataclass
class VCPResultaat:
    contracties:       List[Contractie]
    n_contracties:     int
    pct_krimpt:        bool   # elke correctie kleiner dan vorige
    tijd_krimpt:       bool   # elke correctie korter dan vorige
    vol_krimpt:        bool   # volume daalt bij elke correctie
    laatste_pct:       float  # diepte laatste contractie
    pivot:             float  # weerstand / hoogste punt voor VCP
    laatste_low:       float  # laagste punt laatste contractie (= stop)
    breakout:          bool   # prijs boven pivot vandaag
    breakout_vol:      float  # volume ratio bij breakout
    near_pivot:        bool   # prijs binnen 10% van pivot


def detect_vcp(
    close:  pd.Series,
    high:   pd.Series,
    low:    pd.Series,
    volume: pd.Series,
) -> Optional[VCPResultaat]:
    """
    Detecteert VCP door pieken en dalen te identificeren
    in het lookback window en contracties te meten.
    """
    n = len(close)
    if n < VCP_CFG["lookback_days"] + 20:
        return None

    # Werk met lookback window
    lb      = VCP_CFG["lookback_days"]
    c       = close.values[-lb:]
    h       = high.values[-lb:]
    l       = low.values[-lb:]
    v       = volume.values[-lb:]
    n_lb    = len(c)

    vol_ma  = pd.Series(v).rolling(VCP_CFG["vol_ma_period"]).mean().values

    # ── Stap 1: Vind lokale pieken (highs) en dalen (lows) ──────
    # Gebruik een vereenvoudigde swing-detectie:
    # Een piek = punt hoger dan de 5 vorige en 5 volgende punten
    # Een dal  = punt lager  dan de 5 vorige en 5 volgende punten
    swing = 5

    pieken = []
    dalen  = []

    for i in range(swing, n_lb - swing):
        if all(h[i] >= h[i-j] for j in range(1, swing+1)) and \
           all(h[i] >= h[i+j] for j in range(1, swing+1)):
            pieken.append(i)
        if all(l[i] <= l[i-j] for j in range(1, swing+1)) and \
           all(l[i] <= l[i+j] for j in range(1, swing+1)):
            dalen.append(i)

    if len(pieken) < 1 or len(dalen) < 1:
        return None

    # Pivot = hoogste piek in het window
    pivot_idx = max(pieken, key=lambda i: h[i])
    pivot     = float(h[pivot_idx])

    # Alleen dalen NA de pivot analyseren
    dalen_na_pivot = [d for d in dalen if d > pivot_idx]
    pieken_na_pivot = [p for p in pieken if p > pivot_idx]

    if len(dalen_na_pivot) < VCP_CFG["min_contracties"]:
        return None

    # ── Stap 2: Bouw contracties op ─────────────────────────────
    # Elke contractie = van een lokale piek naar het volgende lokale dal
    contracties: List[Contractie] = []

    # Combineer pieken en dalen na pivot, gesorteerd op tijd
    events = sorted(
        [(i, "piek") for i in pieken_na_pivot] +
        [(i, "dal")  for i in dalen_na_pivot]
    )

    # Zoek piek→dal paren
    i = 0
    while i < len(events) - 1 and len(contracties) < VCP_CFG["max_contracties"]:
        idx_e, type_e = events[i]
        if type_e == "piek":
            # Zoek volgend dal
            for j in range(i + 1, len(events)):
                idx_d, type_d = events[j]
                if type_d == "dal":
                    top_val  = float(h[idx_e])
                    bot_val  = float(l[idx_d])
                    corr_pct = (top_val - bot_val) / top_val * 100

                    if VCP_CFG["min_correctie_pct"] <= corr_pct <= VCP_CFG["max_correctie_pct"]:
                        duur     = idx_d - idx_e
                        vol_gem  = float(np.mean(v[idx_e:idx_d+1]))
                        contracties.append(Contractie(
                            nummer=len(contracties) + 1,
                            high=round(top_val, 4),
                            low=round(bot_val, 4),
                            pct=round(corr_pct, 2),
                            duur=duur,
                            vol_gem=round(vol_gem, 0),
                            start_idx=idx_e,
                            end_idx=idx_d,
                        ))
                    break
        i += 1

    if len(contracties) < VCP_CFG["min_contracties"]:
        return None

    # ── Stap 3: Valideer contractie-eigenschappen ────────────────
    pct_krimpt  = all(
        contracties[i].pct <= contracties[i-1].pct * VCP_CFG["contractie_ratio"]
        for i in range(1, len(contracties))
    )
    tijd_krimpt = all(
        contracties[i].duur <= contracties[i-1].duur * VCP_CFG["tijd_ratio"]
        for i in range(1, len(contracties))
    )
    vol_krimpt  = all(
        contracties[i].vol_gem <= contracties[i-1].vol_gem * VCP_CFG["vol_droogval_ratio"]
        for i in range(1, len(contracties))
    )

    laatste         = contracties[-1]
    laatste_pct     = laatste.pct
    laatste_low     = laatste.low

    # ── Stap 4: Breakout check ───────────────────────────────────
    current_price   = float(c[-1])
    current_vol     = float(v[-1])
    vol_ma_now      = safe_float(vol_ma[-1], 1.0)

    # Pivot voor breakout = hoogste high na de pivot maar voor de contracties
    # = high van eerste contractie (top van VCP)
    vcp_pivot       = contracties[0].high
    breakout        = current_price > vcp_pivot
    breakout_vol    = (current_vol / vol_ma_now) if vol_ma_now > 0 else 0.0
    near_pivot      = ((vcp_pivot - current_price) / vcp_pivot * 100) <= VCP_CFG["pivot_proximity_pct"]

    return VCPResultaat(
        contracties=contracties,
        n_contracties=len(contracties),
        pct_krimpt=pct_krimpt,
        tijd_krimpt=tijd_krimpt,
        vol_krimpt=vol_krimpt,
        laatste_pct=round(laatste_pct, 2),
        pivot=round(vcp_pivot, 4),
        laatste_low=round(laatste_low, 4),
        breakout=breakout,
        breakout_vol=round(breakout_vol, 2),
        near_pivot=near_pivot,
    )


# ============================================================
# STAGE 2 TREND CHECK
# ============================================================

def check_stage2(g: pd.DataFrame) -> Tuple[bool, str]:
    close = g["Close"]
    ma50  = close.rolling(VCP_CFG["ma_fast"]).mean()
    ma150 = close.rolling(VCP_CFG["ma_mid"]).mean()
    ma200 = close.rolling(VCP_CFG["ma_slow"]).mean()

    c   = safe_float(close.iloc[-1])
    m50 = safe_float(ma50.iloc[-1])
    m150= safe_float(ma150.iloc[-1])
    m200= safe_float(ma200.iloc[-1])

    if any(math.isnan(x) for x in [c, m50, m150, m200]):
        return False, "onvoldoende data"

    ok = c > m50 > m150 > m200
    if ok:
        return True, f"✓ Close>{VCP_CFG['ma_fast']}>{VCP_CFG['ma_mid']}>{VCP_CFG['ma_slow']} MA"
    else:
        return False, f"✗ Stage 2 vereist Close>MA{VCP_CFG['ma_fast']}>MA{VCP_CFG['ma_mid']}>MA{VCP_CFG['ma_slow']}"


# ============================================================
# VCP SIGNAAL
# ============================================================

@dataclass
class VCPSignaal:
    ticker:        str
    price:         float
    score:         int          # 0-8
    score_labels:  List[str]
    vcp:           VCPResultaat
    stage2:        bool
    stage2_label:  str
    atr:           float
    stop:          float
    total_score:   float


def analyse_ticker(ticker: str, g: pd.DataFrame) -> Optional[VCPSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < VCP_CFG["ma_slow"] + VCP_CFG["lookback_days"]:
            return None

        close  = g["Close"]
        high   = g["High"]
        low    = g["Low"]
        volume = g["Volume"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        # ── ATR voor stop ────────────────────────────────────────
        hl  = high - close.shift()
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([high - low, hcp, lcp], axis=1).max(axis=1)
        atr = safe_float(_wilder_smooth(tr, VCP_CFG["atr_period"]).iloc[-1],
                         current_price * 0.02)

        # ── VCP detectie ─────────────────────────────────────────
        vcp = detect_vcp(close, high, low, volume)
        if vcp is None:
            return None

        # ── Stage 2 check ─────────────────────────────────────────
        stage2, stage2_label = check_stage2(g)

        # ── Stop berekening ───────────────────────────────────────
        # Stop = laagste punt van laatste contractie - 0.5× ATR
        stop = vcp.laatste_low - (0.5 * atr)

        # ── Score 0-8 ────────────────────────────────────────────
        score  = 0
        labels = []

        def chk(ok: bool, ok_msg: str, fail_msg: str):
            nonlocal score
            if ok:
                score += 1
                labels.append(f"✓ {ok_msg}")
            else:
                labels.append(f"✗ {fail_msg}")

        # 1. Minimum contracties
        chk(vcp.n_contracties >= VCP_CFG["min_contracties"],
            f"{vcp.n_contracties} contracties gedetecteerd (min {VCP_CFG['min_contracties']})",
            f"slechts {vcp.n_contracties} contractie(s)")

        # 2. Correcties krimpen in %
        chk(vcp.pct_krimpt,
            "correcties krimpen in % ✓",
            "correcties krimpen NIET in %")

        # 3. Correcties krimpen in tijd
        chk(vcp.tijd_krimpt,
            "correcties krimpen in tijd ✓",
            "correcties krimpen NIET in tijd")

        # 4. Volume droogvalt
        chk(vcp.vol_krimpt,
            "volume daalt bij elke contractie ✓",
            "volume daalt NIET consistent")

        # 5. Laatste contractie ≤ 10%
        chk(vcp.laatste_pct <= VCP_CFG["laatste_max_pct"],
            f"laatste contractie {vcp.laatste_pct:.1f}% (≤{VCP_CFG['laatste_max_pct']:.0f}%)",
            f"laatste contractie {vcp.laatste_pct:.1f}% (>{VCP_CFG['laatste_max_pct']:.0f}%)")

        # 6. Prijs nabij pivot
        chk(vcp.near_pivot or vcp.breakout,
            f"prijs nabij pivot ({vcp.pivot:.2f})",
            f"prijs ver van pivot ({vcp.pivot:.2f})")

        # 7. Stage 2 trend
        chk(stage2, stage2_label, stage2_label)

        # 8. Breakout op volume
        chk(vcp.breakout and vcp.breakout_vol >= VCP_CFG["breakout_vol_mult"],
            f"BREAKOUT boven {vcp.pivot:.2f} op {vcp.breakout_vol:.1f}× volume",
            f"geen breakout (vol={vcp.breakout_vol:.1f}×, pivot={vcp.pivot:.2f})")

        if score < VCP_CFG["min_score"]:
            return None

        # Ranking: meer contracties + stage2 + breakout = hogere score
        total_score = (
            score * 10
            + vcp.n_contracties * 5
            + (20 if vcp.breakout else 0)
            + (10 if stage2 else 0)
            + (5 if vcp.vol_krimpt else 0)
        )

        return VCPSignaal(
            ticker=ticker,
            price=round(current_price, 2),
            score=score,
            score_labels=labels,
            vcp=vcp,
            stage2=stage2,
            stage2_label=stage2_label,
            atr=round(atr, 4),
            stop=round(stop, 2),
            total_score=round(total_score, 1),
        )

    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM OUTPUT  (zelfde stijl als bot_00xxxV2.py)
# ============================================================

def _score_bar(score: int, max_score: int = 8) -> str:
    filled = "█" * score
    empty  = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def _vcp_diagram(vcp: VCPResultaat) -> str:
    """Tekstueel diagram van de VCP contracties."""
    lines = ["  📉 *VCP Contracties:*"]
    for c in vcp.contracties:
        bar_len = max(1, int(c.pct / 2))
        bar     = "▓" * bar_len
        lines.append(
            f"  C{c.nummer}: {bar} -{c.pct:.1f}% | {c.duur}d | "
            f"high:{c.high:.2f} low:{c.low:.2f}"
        )
    lines.append(f"  Pivot: EUR{vcp.pivot:.2f}")
    if vcp.breakout:
        lines.append(f"  🚀 BREAKOUT op {vcp.breakout_vol:.1f}× volume!")
    return "\n".join(lines)


def format_vcp_per_exchange(
    exchange_name:    str,
    signalen:         List[VCPSignaal],
    portfolio_waarde: float,
) -> Tuple[str, str]:
    nu = today_str()

    def detail_blok(sigs: List[VCPSignaal]) -> str:
        if not sigs:
            return "_Geen kandidaten_"
        lines = []
        for s in sigs:
            lines.append(
                f"• `{s.ticker}` | Score: {_score_bar(s.score)} | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                + _vcp_diagram(s.vcp) + "\n"
                + "\n".join(f"  {lbl}" for lbl in s.score_labels) + "\n"
                + sizing_tekst(s.ticker, s.price, s.stop, s.vcp.pivot, portfolio_waarde)
            )
        return "\n\n".join(lines)

    top2       = signalen[:2]
    score8     = [s for s in signalen if s.score == 8]
    score7     = [s for s in signalen if s.score == 7]
    score6     = [s for s in signalen if s.score == 6]
    score45    = [s for s in signalen if s.score in (4, 5)]
    breakouts  = [s for s in signalen if s.vcp.breakout]

    deel1 = "\n\n".join([
        f"🔻 *VCP — {exchange_name}*",
        f"_{nu} | Volatility Contraction Pattern | Min score: {VCP_CFG['min_score']}/8_",
        f"_Narrowing techniek: correcties krimpen in %, tijd én volume_",
        "─────────────────────────────",
        f"🏆 *TOP 2 HOOGSTE POTENTIEEL:*",
        detail_blok(top2) if top2 else "_Geen kandidaten vandaag_",
        "─────────────────────────────",
        f"🔥 *PERFECTE VCP (8/8):*",
        detail_blok(score8) if score8 else "_Geen_",
        f"⭐ *UITSTEKEND (7/8):*",
        detail_blok(score7) if score7 else "_Geen_",
    ])

    deel2_parts = [
        f"🔻 *VCP — {exchange_name} (2/2)*",
        "",
        f"⚡ *STERK (6/8):*",
        detail_blok(score6) if score6 else "_Geen_",
        "",
        f"📊 *WATCHLIST (4-5/8):*",
        detail_blok(score45) if score45 else "_Geen_",
        "",
        "─────────────────────────────",
        f"🚀 *ACTIEVE BREAKOUTS:* {len(breakouts)}",
    ]
    for s in breakouts[:5]:
        deel2_parts.append(
            f"  • `{s.ticker}` boven EUR{s.vcp.pivot:.2f} | "
            f"{s.vcp.n_contracties} contracties | "
            f"vol {s.vcp.breakout_vol:.1f}× | score {s.score}/8"
        )
    deel2_parts += [
        "",
        "─────────────────────────────",
        f"📊 *SAMENVATTING:*",
        f"  Kandidaten (score≥{VCP_CFG['min_score']}) : {len(signalen)}",
        f"  Actieve breakouts         : {len(breakouts)}",
        f"  Score 8/8                 : {len(score8)}",
        f"  Score 7/8                 : {len(score7)}",
        f"  Score 6/8                 : {len(score6)}",
        f"  Score 4-5/8               : {len(score45)}",
        "",
        "⚙️ *VCP PARAMETERS:*",
        f"_Min {VCP_CFG['min_contracties']} contracties | Contractie ratio: {VCP_CFG['contractie_ratio']}×_",
        f"_Laatste correctie ≤{VCP_CFG['laatste_max_pct']:.0f}% | Pivot proximiteit ≤{VCP_CFG['pivot_proximity_pct']:.0f}%_",
        f"_Volume droogval: {VCP_CFG['vol_droogval_ratio']}× | Breakout vol: {VCP_CFG['breakout_vol_mult']}×_",
        f"_Stage 2: Close>MA{VCP_CFG['ma_fast']}>MA{VCP_CFG['ma_mid']}>MA{VCP_CFG['ma_slow']}_",
        f"_Stop: onder laatste contractie low - 0.5×ATR_",
    ]

    return deel1, "\n".join(deel2_parts)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"VCP ENGINE — LIVE  {today_str()}")
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
    print("Data downloaden (2 jaar)...")

    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers")
    portfolio_waarde = START_CAPITAL

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        signalen: List[VCPSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group)
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: {sig.score}/8 | "
                    f"{sig.vcp.n_contracties} contracties | "
                    f"laatste -{sig.vcp.laatste_pct:.1f}% | "
                    f"{'BREAKOUT' if sig.vcp.breakout else 'setup'}"
                )

        signalen.sort(key=lambda s: s.total_score, reverse=True)
        print(f"  → {len(signalen)} VCP kandidaten | {sum(s.vcp.breakout for s in signalen)} breakouts")

        deel1, deel2 = format_vcp_per_exchange(ex_name, signalen, portfolio_waarde)
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

def _log_csv(signalen: List[VCPSignaal], exchange: str):
    fname  = f"vcp_signalen_{exchange.split()[0]}_{today_str()}.csv"
    header = ["datum","exchange","ticker","score","price","n_contracties",
              "laatste_pct","pivot","stop","breakout","breakout_vol",
              "stage2","pct_krimpt","tijd_krimpt","vol_krimpt","total_score"]
    ensure_csv_header(fname, header)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in signalen:
            w.writerow([
                today_str(), exchange, s.ticker, s.score, s.price,
                s.vcp.n_contracties, s.vcp.laatste_pct, s.vcp.pivot, s.stop,
                s.vcp.breakout, s.vcp.breakout_vol, s.stage2,
                s.vcp.pct_krimpt, s.vcp.tijd_krimpt, s.vcp.vol_krimpt,
                s.total_score,
            ])
    print(f"  CSV: {fname}")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"VCP BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
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
    print(f"Handelsdagen: {len(all_dates)}")

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

        # Exits
        for ticker, pos in list(positions.items()):
            pos["days"] += 1
            if ticker not in price_map:
                continue
            close  = price_map[ticker]
            reason = None
            if close <= pos["stop"]:
                reason = f"SL ({pos['stop']:.2f})"
            elif close >= pos["tp"]:
                reason = f"TP ({pos['tp']:.2f})"
            elif pos["days"] >= MAX_HOLD_DAYS:
                reason = f"Time ({pos['days']}d)"

            if reason:
                exit_slip = close * (1 - SLIPPAGE_PCT)
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
                    "n_contracties": pos["n_contracties"],
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

        # Entries — alleen bij breakout
        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig or not sig.vcp.breakout:
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
                "entry_date":    scan_date,
                "entry_price":   round(entry, 4),
                "size":          aandelen,
                "stop":          sig.stop,
                "tp":            sig.vcp.pivot * 1.20,  # TP = 20% boven pivot
                "score":         sig.score,
                "n_contracties": sig.vcp.n_contracties,
                "days":          0,
                "cost":          trade_cost(investering),
            }

    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("vcp_backtest_trades.csv", index=False)
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
        print(f"\n{'Contracties':<14} {'#':>4} {'Win%':>6} {'Net':>10}")
        for nc, g in tdf.groupby("n_contracties"):
            wr = (g["net"] > 0).sum() / len(g) * 100
            print(f"{nc} contracties  {len(g):>4} {wr:>5.1f}% {g['net'].sum():>+10.2f}")
        print(f"{'='*60}")
        print(f"Opgeslagen: vcp_backtest_trades.csv")
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
