#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00vcp.py  —  VCP ENGINE v1.0
Volatility Contraction Pattern — Minervini's 'Narrowing' techniek.

Score systeem (0-8):
  1. Minimum 2 VCP contracties gedetecteerd
  2. Elke correctie kleiner dan vorige (%)
  3. Elke correctie korter in tijd dan vorige
  4. Volume daalt bij elke correctie
  5. Laatste contractie <= 10% diep
  6. Prijs binnen 10% van pivot high
  7. Stage 2 trend (Close > MA50 > MA150 > MA200)
  8. Breakout boven pivot op verhoogd volume

Gebruik:
  python bot_00vcp.py live     # live rapport
  python bot_00vcp.py backtest # backtest modus

GitHub Actions: dagelijks om 22:05 UTC

TradingAgents-koppeling:
  Na elke live run wordt vcp_signals.json weggeschreven met alle tickers
  die de min_score-filter doorstonden (over alle beurzen heen), zodat
  tradingagents_bridge.py deze kan oppikken als artifact in de workflow.
"""

import os
import sys
import math
import json
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
MAX_HOLD_DAYS        = 60

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

EXCHANGES = {
    "041 Benelux":     "tickers_041x.txt",
    "042 Parijs":      "tickers_042x.txt",
    "043 Frankfurt":   "tickers_043x.txt",
    "044 Spanje/Port": "tickers_044x.txt",
    "045 Londen":      "tickers_045x.txt",
    "046 Milaan":      "tickers_046x.txt",
    "047 Toronto":     "tickers_047x.txt",
    "048 Nasdaq/NYSE": "tickers_048x.txt",
    "049 Stockholm":   "tickers_049x.txt",
    "050 Zurich":      "tickers_050x.txt",
    "051 Warschau":    "tickers_051x.txt",
    "052 Oslo":        "tickers_052x.txt",
    "053 Kopenhagen":  "tickers_053x.txt",
    "054 Helsinki":    "tickers_054x.txt",
}

VCP_CFG = {
    "min_contracties":       2,
    "max_contracties":       5,
    "min_correctie_pct":     3.0,
    "max_correctie_pct":     50.0,
    "contractie_ratio":      0.80,
    "tijd_ratio":            0.90,
    "laatste_max_pct":       10.0,
    "pivot_proximity_pct":   10.0,
    "vol_ma_period":         50,
    "vol_droogval_ratio":    0.80,
    "breakout_vol_mult":     1.5,
    "ma_fast":               50,
    "ma_mid":                150,
    "ma_slow":               200,
    "atr_period":            14,
    "min_score":             4,
    "lookback_days":         120,
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()

# Bestandsnaam voor de TradingAgents-koppeling
VCP_SIGNALS_PATH = "vcp_signals.json"


# ============================================================
# HULPFUNCTIES
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
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")

def send_email(subject: str, body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        clean = body.replace("*", "").replace("`", "").replace("•", "-").replace("_", "")
        msg.attach(MIMEText(clean, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email fout: {e}")

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"

def sla_vcp_signals_op(tickers: List[str]) -> None:
    """Schrijft de kandidatenlijst weg zodat tradingagents_bridge.py deze kan oppikken."""
    try:
        with open(VCP_SIGNALS_PATH, "w", encoding="utf-8") as f:
            json.dump({"tickers": sorted(set(tickers))}, f)
        print(f"\n→ {VCP_SIGNALS_PATH} geschreven ({len(set(tickers))} tickers)")
    except Exception as e:
        print(f"[WARN] Kon {VCP_SIGNALS_PATH} niet schrijven: {e}")


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
    kosten      = round(trade_cost(investering), 2)
    rr          = ((pivot - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | Pivot: EUR{pivot:.2f}\n"
        f"  R/R: {rr:.1f}:1 | {aandelen} stuks | EUR{investering:,.2f} | Max verlies: EUR{max_loss:,.2f}"
    )


# ============================================================
# DATA DOWNLOAD
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
    nummer:    int
    high:      float
    low:       float
    pct:       float
    duur:      int
    vol_gem:   float
    start_idx: int
    end_idx:   int

@dataclass
class VCPResultaat:
    contracties:   List[Contractie]
    n_contracties: int
    pct_krimpt:    bool
    tijd_krimpt:   bool
    vol_krimpt:    bool
    laatste_pct:   float
    pivot:         float
    laatste_low:   float
    breakout:      bool
    breakout_vol:  float
    near_pivot:    bool

def detect_vcp(
    close:  pd.Series,
    high:   pd.Series,
    low:    pd.Series,
    volume: pd.Series,
) -> Optional[VCPResultaat]:
    n = len(close)
    if n < VCP_CFG["lookback_days"] + 20:
        return None

    lb   = VCP_CFG["lookback_days"]
    c    = close.values[-lb:]
    h    = high.values[-lb:]
    l    = low.values[-lb:]
    v    = volume.values[-lb:]
    n_lb = len(c)

    vol_ma = pd.Series(v).rolling(VCP_CFG["vol_ma_period"]).mean().values

    swing  = 5
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

    pivot_idx       = max(pieken, key=lambda i: h[i])
    pivot           = float(h[pivot_idx])
    dalen_na_pivot  = [d for d in dalen if d > pivot_idx]
    pieken_na_pivot = [p for p in pieken if p > pivot_idx]

    if len(dalen_na_pivot) < VCP_CFG["min_contracties"]:
        return None

    contracties: List[Contractie] = []
    events = sorted(
        [(i, "piek") for i in pieken_na_pivot] +
        [(i, "dal")  for i in dalen_na_pivot]
    )

    i = 0
    while i < len(events) - 1 and len(contracties) < VCP_CFG["max_contracties"]:
        idx_e, type_e = events[i]
        if type_e == "piek":
            for j in range(i + 1, len(events)):
                idx_d, type_d = events[j]
                if type_d == "dal":
                    top_val  = float(h[idx_e])
                    bot_val  = float(l[idx_d])
                    corr_pct = (top_val - bot_val) / top_val * 100
                    if VCP_CFG["min_correctie_pct"] <= corr_pct <= VCP_CFG["max_correctie_pct"]:
                        duur    = idx_d - idx_e
                        vol_gem = float(np.mean(v[idx_e:idx_d+1]))
                        contracties.append(Contractie(
                            nummer=len(contracties) + 1,
                            high=round(top_val, 4), low=round(bot_val, 4),
                            pct=round(corr_pct, 2), duur=duur,
                            vol_gem=round(vol_gem, 0),
                            start_idx=idx_e, end_idx=idx_d,
                        ))
                    break
        i += 1

    if len(contracties) < VCP_CFG["min_contracties"]:
        return None

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

    laatste      = contracties[-1]
    laatste_pct  = laatste.pct
    laatste_low  = laatste.low

    current_price = float(c[-1])
    current_vol   = float(v[-1])
    vol_ma_now    = safe_float(vol_ma[-1], 1.0)

    vcp_pivot    = contracties[0].high
    breakout     = current_price > vcp_pivot
    breakout_vol = (current_vol / vol_ma_now) if vol_ma_now > 0 else 0.0
    near_pivot   = ((vcp_pivot - current_price) / vcp_pivot * 100) <= VCP_CFG["pivot_proximity_pct"]

    return VCPResultaat(
        contracties=contracties, n_contracties=len(contracties),
        pct_krimpt=pct_krimpt, tijd_krimpt=tijd_krimpt, vol_krimpt=vol_krimpt,
        laatste_pct=round(laatste_pct, 2), pivot=round(vcp_pivot, 4),
        laatste_low=round(laatste_low, 4), breakout=breakout,
        breakout_vol=round(breakout_vol, 2), near_pivot=near_pivot,
    )


# ============================================================
# STAGE 2 TREND CHECK
# ============================================================

def check_stage2(g: pd.DataFrame) -> Tuple[bool, str]:
    close = g["Close"]
    ma50  = close.rolling(VCP_CFG["ma_fast"]).mean()
    ma150 = close.rolling(VCP_CFG["ma_mid"]).mean()
    ma200 = close.rolling(VCP_CFG["ma_slow"]).mean()

    c    = safe_float(close.iloc[-1])
    m50  = safe_float(ma50.iloc[-1])
    m150 = safe_float(ma150.iloc[-1])
    m200 = safe_float(ma200.iloc[-1])

    if any(math.isnan(x) for x in [c, m50, m150, m200]):
        return False, "onvoldoende data"

    ok = c > m50 > m150 > m200
    if ok:
        return True, f"✓ Close>MA{VCP_CFG['ma_fast']}>MA{VCP_CFG['ma_mid']}>MA{VCP_CFG['ma_slow']}"
    else:
        return False, f"✗ Stage 2 vereist Close>MA{VCP_CFG['ma_fast']}>MA{VCP_CFG['ma_mid']}>MA{VCP_CFG['ma_slow']}"


# ============================================================
# VCP SIGNAAL
# ============================================================

@dataclass
class VCPSignaal:
    ticker:       str
    price:        float
    score:        int
    score_labels: List[str]
    vcp:          VCPResultaat
    stage2:       bool
    stage2_label: str
    atr:          float
    stop:         float
    total_score:  float

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

        hl  = high - close.shift()
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([high - low, hcp, lcp], axis=1).max(axis=1)
        atr = safe_float(_wilder_smooth(tr, VCP_CFG["atr_period"]).iloc[-1],
                         current_price * 0.02)

        vcp = detect_vcp(close, high, low, volume)
        if vcp is None:
            return None

        stage2, stage2_label = check_stage2(g)
        stop = vcp.laatste_low - (0.5 * atr)

        score  = 0
        labels = []

        def chk(ok: bool, ok_msg: str, fail_msg: str):
            nonlocal score
            if ok:
                score += 1
                labels.append(f"✓ {ok_msg}")
            else:
                labels.append(f"✗ {fail_msg}")

        chk(vcp.n_contracties >= VCP_CFG["min_contracties"],
            f"{vcp.n_contracties} contracties (min {VCP_CFG['min_contracties']})",
            f"slechts {vcp.n_contracties} contractie(s)")
        chk(vcp.pct_krimpt,
            "correcties krimpen in %",
            "correcties krimpen NIET in %")
        chk(vcp.tijd_krimpt,
            "correcties krimpen in tijd",
            "correcties krimpen NIET in tijd")
        chk(vcp.vol_krimpt,
            "volume daalt bij elke contractie",
            "volume daalt NIET consistent")
        chk(vcp.laatste_pct <= VCP_CFG["laatste_max_pct"],
            f"laatste contractie {vcp.laatste_pct:.1f}% (<={VCP_CFG['laatste_max_pct']:.0f}%)",
            f"laatste contractie {vcp.laatste_pct:.1f}% (>{VCP_CFG['laatste_max_pct']:.0f}%)")
        chk(vcp.near_pivot or vcp.breakout,
            f"prijs nabij pivot ({vcp.pivot:.2f})",
            f"prijs ver van pivot ({vcp.pivot:.2f})")
        chk(stage2, stage2_label, stage2_label)
        chk(vcp.breakout and vcp.breakout_vol >= VCP_CFG["breakout_vol_mult"],
            f"BREAKOUT boven {vcp.pivot:.2f} op {vcp.breakout_vol:.1f}x volume",
            f"geen breakout (vol={vcp.breakout_vol:.1f}x)")

        if score < VCP_CFG["min_score"]:
            return None

        total_score = (
            score * 10
            + vcp.n_contracties * 5
            + (20 if vcp.breakout else 0)
            + (10 if stage2 else 0)
            + (5 if vcp.vol_krimpt else 0)
        )

        return VCPSignaal(
            ticker=ticker, price=round(current_price, 2),
            score=score, score_labels=labels,
            vcp=vcp, stage2=stage2, stage2_label=stage2_label,
            atr=round(atr, 4), stop=round(stop, 2),
            total_score=round(total_score, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 8) -> str:
    return "█" * score + "░" * (max_score - score) + f" {score}/{max_score}"

def format_bericht(
    exchange_name:    str,
    signalen:         List[VCPSignaal],
    portfolio_waarde: float,
) -> Optional[str]:
    if not signalen:
        return None

    nu        = today_str()
    max_score = max(s.score for s in signalen)
    toon      = [s for s in signalen if s.score == max_score]
    top2      = signalen[:2]
    lbl       = {8: "🔥 PERFECT (8/8)", 7: "⭐ UITSTEKEND (7/8)",
                 6: "⚡ STERK (6/8)", 5: "📊 GOED (5/8)",
                 4: "📊 WATCHLIST (4/8)"}.get(max_score, "📊")

    delen = [
        f"🔻 *VCP — {exchange_name}*",
        f"_{nu} | {len(signalen)} kandidaten | {sum(s.vcp.breakout for s in signalen)} breakouts_",
        "─────────────────────────────",
        "🏆 *TOP 2:*",
    ]
    for s in top2:
        delen.append(
            f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} {_yahoo_link(s.ticker)}\n"
            f"  {s.vcp.n_contracties} contracties | laatste -{s.vcp.laatste_pct:.1f}% | "
            f"{'BREAKOUT' if s.vcp.breakout else 'setup'}\n"
            + sizing_tekst(s.ticker, s.price, s.stop, s.vcp.pivot, portfolio_waarde)
        )

    delen += ["─────────────────────────────", f"*{lbl}:*"]
    extra = [s for s in toon if s not in top2]
    if extra:
        for s in extra:
            delen.append(
                f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} | "
                f"{s.vcp.n_contracties} contracties | "
                f"{'BREAKOUT' if s.vcp.breakout else 'setup'}"
            )
    else:
        delen.append("_Zie top 2 hierboven_")

    delen.append(
        f"⚙️ _Min {VCP_CFG['min_contracties']} contracties | "
        f"Breakout {VCP_CFG['breakout_vol_mult']}x vol | Risico 5%_"
    )
    return "\n\n".join(delen)


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
    email_delen: List[str] = []
    alle_kandidaten: List[str] = []  # verzamelt tickers voor vcp_signals.json (TradingAgents)

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

        alle_kandidaten.extend(s.ticker for s in signalen)

        bericht = format_bericht(ex_name, signalen, portfolio_waarde)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Overgeslagen (geen signalen): {ex_name}")

    if email_delen:
        send_email(
            subject=f"VCP rapport {today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    # Schrijf kandidatenlijst weg voor de TradingAgents-bridge
    sla_vcp_signals_op(alle_kandidaten)

    print(f"\n{'='*60}")
    print("Klaar.")


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

    all_dates  = sorted(df["Date"].dt.date.unique())
    cash       = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []
    price_map: Dict[str, float] = {}

    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(scan_date)].copy()

        price_map = {}
        for _, row in day_df.iterrows():
            t = row.get("Ticker")
            c = safe_float(row.get("Close"))
            if t and not math.isnan(c):
                price_map[t] = c

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
                    "entry_date":    pos["entry_date"].isoformat(),
                    "exit_date":     scan_date.isoformat(),
                    "ticker":        ticker,
                    "score":         pos["score"],
                    "n_contracties": pos["n_contracties"],
                    "entry_price":   pos["entry_price"],
                    "exit_price":    round(exit_slip, 4),
                    "size":          pos["size"],
                    "pnl":           round(pnl, 2),
                    "tax":           round(tax, 2),
                    "net":           round(pnl - tax, 2),
                    "reason":        reason,
                    "days":          pos["days"],
                })
                del positions[ticker]

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig or not sig.vcp.breakout:
                continue
            entry       = sig.price * (1 + SLIPPAGE_PCT)
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
                "tp":            sig.vcp.pivot * 1.20,
                "score":         sig.score,
                "n_contracties": sig.vcp.n_contracties,
                "days":          0,
                "cost":          trade_cost(investering),
            }

    if trades:
        tdf  = pd.DataFrame(trades)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final_val = cash + sum(
            price_map.get(t, p["entry_price"]) * p["size"]
            for t, p in positions.items())
        print(f"\n{'='*60}")
        print(f"Startkapitaal    : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal     : EUR{final_val:>12,.2f}")
        print(f"Totaal rendement : {(final_val-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
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
