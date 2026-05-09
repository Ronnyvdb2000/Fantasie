#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRA SIGNAL ENGINE v5.2
- Sequential per ticker
- auto_adjust=False (correcte IBS, BB, ATR, ADX)
- Mixed tickerlijst input
- Versoepelde filters (OPTIE A)
- Multi-index FIX voor yfinance 1.2.0
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# TELEGRAM
# -----------------------------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def telegram_send(msg: str):
    """Split automatisch bij >3500 chars."""
    if not TOKEN or not CHAT_ID:
        return

    chunks = [msg[i:i+3500] for i in range(0, len(msg), 3500)]
    for c in chunks:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": c,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                },
                timeout=20
            )
            time.sleep(1)
        except Exception as e:
            logger.error(f"Telegram fout: {e}")

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
MRA_BB_STD      = 2.2
MRA_IBS_MAX     = 0.30
MRA_SNEL_WINST  = 1.12
MRA_SNEL_MA     = 5
MRA_TRAAG_WINST = 1.25
MRA_TRAAG_MA    = 10
MRA_TRAAG_HOLD  = 5

INZET = 2500.0
KOSTEN = 15.0 + (INZET * 0.0035)

# -----------------------------------------------------------------------------
# DOWNLOAD ENGINE + MULTI-INDEX FIX
# -----------------------------------------------------------------------------
def download_ticker(ticker: str):
    """Yahoo-proof download, sequential, auto_adjust=False + multi-index fix."""
    try:
        df = yf.download(
            ticker,
            period="5y",
            auto_adjust=False,
            progress=False
        )

        if df is None or len(df) < 250:
            return None

        # -------------------------------
        # MULTI-INDEX FIX
        # -------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Forceer kolommen naar Series
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in df.columns and isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        return df.dropna()

    except Exception as e:
        logger.error(f"Download fout {ticker}: {e}")
        return None

# -----------------------------------------------------------------------------
# INDICATOREN
# -----------------------------------------------------------------------------
def indicators(df: pd.DataFrame):
    """Volledig vectorized indicator-engine."""
    p = df["Close"].ffill()
    h = df["High"].ffill()
    l = df["Low"].ffill()
    v = df["Volume"].ffill()

    ema200 = p.ewm(span=200, adjust=False).mean()
    ma20   = p.rolling(20).mean()
    ma5    = p.rolling(5).mean()

    std20 = p.rolling(20).std()
    lower_bb = ma20 - (MRA_BB_STD * std20)

    ibs = (p - l) / (h - l + 1e-10)

    delta = p.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

    tr = pd.concat([
        h - l,
        (h - p.shift()).abs(),
        (l - p.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    up = h.diff().clip(lower=0)
    dn = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (dn.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean() * 100

    return p, ema200, rsi, atr, adx, ibs, lower_bb, ma5

# -----------------------------------------------------------------------------
# STRATEGIE-ENGINE
# -----------------------------------------------------------------------------
def run_strategy(df: pd.DataFrame, ticker: str):
    p, ema200, rsi, atr, adx, ibs, lower_bb, ma5 = indicators(df)

    res = {"T":0, "S":0, "HT":0, "HS":0, "MRAS":0, "MRAT":0}
    trades = {"T":0, "S":0, "HT":0, "HS":0, "MRAS":0, "MRAT":0}
    sig = {"T":[], "S":[], "HT":[], "HS":[], "MRAS":[], "MRAT":[]}

    f50  = p.ewm(span=50, adjust=False).mean()
    s200 = p.ewm(span=200, adjust=False).mean()
    f20  = p.ewm(span=20, adjust=False).mean()
    s50  = p.ewm(span=50, adjust=False).mean()

    f9  = p.ewm(span=9, adjust=False).mean()
    s21 = p.ewm(span=21, adjust=False).mean()

    # -------------------------------
    # MA STRATEGIE (T, S, HT, HS)
    # -------------------------------
    def run_ma_strategy(fast, slow, use_trend, key):
        pr = 0.0
        pos = False
        ins = 0.0
        hi  = 0.0

        vol_ma20 = df["Volume"].rolling(20).mean()

        for i in range(1, len(p)):
            cp = p.iloc[i]

            if not pos:
                if fast.iloc[i] > slow.iloc[i] and fast.iloc[i-1] <= slow.iloc[i-1]:

                    # EMA200-filter alleen voor T en S
                    if use_trend and key in ["T", "S"] and cp <= ema200.iloc[i]:
                        continue

                    # ADX versoepeld
                    if adx.iloc[i] < 10:
                        continue

                    # Volume versoepeld
                    if df["Volume"].iloc[i] < vol_ma20.iloc[i] * 0.3:
                        continue

                    pos = True
                    ins = cp
                    hi  = cp
                    pr -= KOSTEN
                    trades[key] += 1

            else:
                hi = max(hi, cp)

                # Trailing stop 1.5 ATR
                if cp < hi - 1.5 * atr.iloc[i] or fast.iloc[i] < slow.iloc[i]:
                    pr += (INZET * (cp / ins) - INZET) - KOSTEN
                    pos = False

        if pos:
            pr += (INZET * (p.iloc[-1] / ins) - INZET) - KOSTEN

        res[key] += pr

        # Signalen
        if fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]:
            sig[key].append(
                f"• `{ticker}`: 🟢 *KOOP* | €{p.iloc[-1]:.2f} | ATR {atr.iloc[-1]:.2f}"
            )
        elif fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]:
            sig[key].append(
                f"• `{ticker}`: 🔴 *VERKOOP* | €{p.iloc[-1]:.2f} | ATR {atr.iloc[-1]:.2f}"
            )

    run_ma_strategy(f50, s200, True,  "T")
    run_ma_strategy(f20, s50,  True,  "S")
    run_ma_strategy(f9,  s21,  False, "HT")
    run_ma_strategy(f9,  s21,  False, "HS")

    # -------------------------------
    # MRA SNEL
    # -------------------------------
    pr = 0.0
    pos = False
    ins = 0.0
    pb = p.iloc[-252:]
    lbb = lower_bb.iloc[-252:]
    ibs_s = ibs.iloc[-252:]
    ma5_s = ma5.iloc[-252:]

    for i in range(1, len(pb)):
        cp = pb.iloc[i]
        if not pos:
            if cp < lbb.iloc[i] and ibs_s.iloc[i] < MRA_IBS_MAX:
                pos = True
                ins = cp
                pr -= KOSTEN
                trades["MRAS"] += 1
        else:
            if cp > ma5_s.iloc[i] or cp > ins * MRA_SNEL_WINST:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        pr += (INZET * (pb.iloc[-1] / ins) - INZET) - KOSTEN

    res["MRAS"] += pr

    if p.iloc[-1] < lower_bb.iloc[-1] and ibs.iloc[-1] < MRA_IBS_MAX:
        sig["MRAS"].append(f"• `{ticker}`: 🛡️ MRA SNEL")

    # -------------------------------
    # MRA TRAAG
    # -------------------------------
    pr = 0.0
    pos = False
    ins = 0.0
    hold = 0
    ma10 = p.rolling(MRA_TRAAG_MA).mean().iloc[-252:]

    for i in range(1, len(pb)):
        cp = pb.iloc[i]
        if not pos:
            if cp < lbb.iloc[i] and ibs_s.iloc[i] < MRA_IBS_MAX:
                pos = True
                ins = cp
                hold = 0
                pr -= KOSTEN
                trades["MRAT"] += 1
        else:
            hold += 1
            if hold >= MRA_TRAAG_HOLD:
                if cp > ma10.iloc[i] or cp > ins * MRA_TRAAG_WINST:
                    pr += (INZET * (cp / ins) - INZET) - KOSTEN
                    pos = False

    if pos:
        pr += (INZET * (pb.iloc[-1] / ins) - INZET) - KOSTEN

    res["MRAT"] += pr

    if p.iloc[-1] < lower_bb.iloc[-1] and ibs.iloc[-1] < MRA_IBS_MAX:
        sig["MRAT"].append(f"• `{ticker}`: 🐢 MRA TRAAG")

    return res, trades, sig

# -----------------------------------------------------------------------------
# RAPPORTAGE
# -----------------------------------------------------------------------------
def rapport(label, naam, nu, res, trades, sig):
    def fmt(n): return f"€{100000 + n:,.0f}"
    def block(lst): return "\n".join(lst) if lst else "Geen actie"

    deel1 = [
        f"📊 *{label} {naam} RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 Traag (50/200): {fmt(res['T'])} ({trades['T']} trades)",
        f"⚡ Snel (20/50): {fmt(res['S'])} ({trades['S']} trades)",
        f"🚀 Hyper Trend: {fmt(res['HT'])} ({trades['HT']} trades)",
        f"🔥 Hyper Scalp: {fmt(res['HS'])} ({trades['HS']} trades)",
        f"🛡️ MRA Snel: {fmt(res['MRAS'])} ({trades['MRAS']} trades)",
        f"🐢 MRA Traag: {fmt(res['MRAT'])} ({trades['MRAT']} trades)",
        "",
        "🛡️ SIGNALEN TRAAG:",
        block(sig["T"]),
        "",
        "🎯 SIGNALEN SNEL:",
        block(sig["S"]),
    ]

    deel2 = [
        f"📊 *{label} {naam} (2/2)*",
        "",
        "📈 SIGNALEN HYPER TREND:",
        block(sig["HT"]),
        "",
        "⚡ SIGNALEN HYPER SCALP:",
        block(sig["HS"]),
        "",
        "🛡️ SIGNALEN MRA SNEL:",
        block(sig["MRAS"]),
        "",
        "🐢 SIGNALEN MRA TRAAG:",
        block(sig["MRAT"]),
        "",
        "⚙️ PARAMETERS:",
        "_ADX>10 | Vol>0.3×MA20 | EMA200 alleen T/S | SL=1.5×ATR_",
    ]

    telegram_send("\n".join(deel1))
    telegram_send("\n".join(deel2))

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    sectoren = {
        "041": "Benelux",
        "042": "Parijs",
        "043": "Frankfurt",
        "044": "Spanje/Portugal",
        "045": "Londen",
        "046": "Milaan",
        "047": "Toronto",
        "048": "Nasdaq/NYSE",
    }

    for i in range(41, 61):
        nr = f"0{i}"
        fname = f"tickers_{nr}x.txt"
        naam = sectoren.get(nr, f"Lijst {nr}")

        if not os.path.exists(fname):
            continue

        with open(fname, "r") as f:
            raw = f.read().replace("\n", ",").replace("$", "")
            tickers = sorted(set(t.strip().upper() for t in raw.split(",") if t.strip()))

        if not tickers:
            continue

        logger.info(f"Start analyse {nr} ({naam}) met {len(tickers)} tickers.")
        nu = datetime.now().strftime("%d/%m/%Y %H:%M")

        res = {"T":0, "S":0, "HT":0, "HS":0, "MRAS":0, "MRAT":0}
        trades = {"T":0, "S":0, "HT":0, "HS":0, "MRAS":0, "MRAT":0}
        sig = {"T":[], "S":[], "HT":[], "HS":[], "MRAS":[], "MRAT":[]}

        for t in tickers:
            df = download_ticker(t)
            if df is None:
                continue

            r, tr, sg = run_strategy(df, t)

            for k in res:
                res[k] += r[k]
                trades[k] += tr[k]
                sig[k].extend(sg[k])

            time.sleep(0.2)

        rapport(nr, naam, nu, res, trades, sig)
        time.sleep(2)

if __name__ == "__main__":
    main()
