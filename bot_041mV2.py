"""
bot_041m_diagnose.py
====================
Diagnostische tool: toont per ticker WAAROM hij de filter niet haalt.
Geeft statistieken per filter zodat je ziet welke parameter het meeste
aandelen weggooit.

Gebruik lokaal:   python bot_041m_diagnose.py
Gebruik via CI:   DIAGNOSE_LIJST=041 python bot_041m_diagnose.py

Output: diagnose_GETAL_DATUM.txt  (+ Telegram samenvatting per lijst)

Via GitHub Actions wordt elke lijst als aparte stap uitgevoerd
met een pauze ertussen (zie bot_041m_diagnose.yml).
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import date

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       tekst[:4096],
                "parse_mode": "Markdown",
            },
            timeout=15,
        ).raise_for_status()
    except Exception as e:
        print(f"Telegram fout: {e}")


# ── Beursconfiguratie ─────────────────────────────────────────────────────────
BEURS_CONFIG = {
    "041": {"naam": "Benelux",         "suffixen": [".AS", ".BR", ".LU"]},
    "042": {"naam": "Parijs",          "suffixen": [".PA"]},
    "043": {"naam": "Frankfurt",       "suffixen": [".DE"]},
    "044": {"naam": "Spanje/Portugal", "suffixen": [".MC", ".LS"]},
    "045": {"naam": "Londen",          "suffixen": [".L"]},
    "046": {"naam": "Milaan",          "suffixen": [".MI"]},
    "047": {"naam": "Toronto",         "suffixen": [".TO", ".V"]},
    "048": {"naam": "Nasdaq/NYSE",     "suffixen": [""]},
}

# ── Criteria per beurstype ────────────────────────────────────────────────────
EUROPA_BEURZEN = {"041", "042", "043", "044", "045", "046"}

CRITERIA = {
    "europa": {
        "ROE_MIN":      0.07,
        "DEBT_MAX":     130.0,
        "MARGE_MIN":    0.04,
        "VOL_MIN":      0.18,
        "VOL_MAX":      0.65,
        "MIN_DAGOMZET": 150_000,
    },
    "noordamerika": {
        "ROE_MIN":      0.08,
        "DEBT_MAX":     120.0,
        "MARGE_MIN":    0.07,
        "VOL_MIN":      0.22,
        "VOL_MAX":      0.70,
        "MIN_DAGOMZET": 500_000,
    },
}

def get_criteria(getal: str) -> dict:
    return CRITERIA["europa"] if getal in EUROPA_BEURZEN else CRITERIA["noordamerika"]


# ── Tickers laden ─────────────────────────────────────────────────────────────
def laad_tickers(getal: str) -> list:
    pad = f"tickers_{getal}a.txt"
    if not os.path.exists(pad):
        print(f"  ⚠️  {pad} niet gevonden — overgeslagen")
        return []
    with open(pad, "r", encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace(";", ",").replace("$", "")
    tickers = sorted(set(t.strip().upper() for t in inhoud.split(",") if t.strip()))
    print(f"  📋 {len(tickers)} tickers geladen uit {pad}")
    return tickers


# ── Batch OHLCV download ──────────────────────────────────────────────────────
def batch_download(tickers: list) -> dict:
    resultaat = {}
    batches   = [tickers[i:i+50] for i in range(0, len(tickers), 50)]
    print(f"  📥 OHLCV download: {len(tickers)} tickers in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        print(f"     Batch {i+1}/{len(batches)}...", end="", flush=True)
        try:
            raw = yf.download(
                batch, period="1y", progress=False,
                auto_adjust=True, multi_level_index=True,
            )
            for t in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns[0], tuple):
                            df.columns = [c[0] for c in df.columns]
                    else:
                        df = raw.xs(t, axis=1, level=1).dropna(how="all")
                    resultaat[t] = df if len(df) >= 50 else None
                except Exception:
                    resultaat[t] = None
            print(" ✅")
        except Exception as e:
            print(f" ❌ {e}")
            for t in batch:
                resultaat[t] = None

        if i < len(batches) - 1:
            time.sleep(3)

    return resultaat


# ── Filter checks ─────────────────────────────────────────────────────────────
def check_fundamenteel(ticker: str, crit: dict) -> tuple:
    try:
        info = yf.Ticker(ticker).info
        if not info or "returnOnEquity" not in info:
            return None, None, None, ["geen_fundamentele_data"]

        roe   = info.get("returnOnEquity", 0)    or 0
        debt  = info.get("debtToEquity",   9999) or 9999
        marge = info.get("profitMargins",  0)    or 0
        if 0 < debt < 2:
            debt *= 100

        falen = []
        if roe   < crit["ROE_MIN"]:   falen.append(f"ROE={roe:.1%}<{crit['ROE_MIN']:.0%}")
        if debt  > crit["DEBT_MAX"]:  falen.append(f"Debt={debt:.0f}>{crit['DEBT_MAX']:.0f}")
        if marge < crit["MARGE_MIN"]: falen.append(f"Marge={marge:.1%}<{crit['MARGE_MIN']:.0%}")
        return roe, debt, marge, falen

    except Exception as e:
        return None, None, None, [f"api_fout:{e}"]


def check_vol_liq(ticker: str, ohlcv: dict, crit: dict) -> tuple:
    df = ohlcv.get(ticker)
    if df is None or len(df) < 50:
        return None, None, ["te_weinig_koersdata"]
    try:
        vol   = float(df["Close"].ffill().pct_change().dropna().std() * np.sqrt(252))
        omzet = float((df["Close"] * df["Volume"]).mean()) if "Volume" in df.columns else 0.0

        falen = []
        if vol < crit["VOL_MIN"]:
            falen.append(f"Vol={vol:.0%}<{crit['VOL_MIN']:.0%}")
        elif vol > crit["VOL_MAX"]:
            falen.append(f"Vol={vol:.0%}>{crit['VOL_MAX']:.0%}")
        if omzet < crit["MIN_DAGOMZET"]:
            falen.append(f"Omzet=€{omzet:,.0f}<€{crit['MIN_DAGOMZET']:,.0f}")
        return vol, omzet, falen

    except Exception as e:
        return None, None, [f"vol_fout:{e}"]


# ── Diagnose één lijst ────────────────────────────────────────────────────────
def diagnose_lijst(getal: str) -> dict:
    vandaag = date.today().strftime("%d/%m/%Y")
    config  = BEURS_CONFIG.get(getal, {"naam": f"Lijst {getal}"})
    naam    = config["naam"]
    crit    = get_criteria(getal)

    print(f"\n{'='*65}")
    print(f"  🔍 DIAGNOSE {getal} — {naam}")
    print(f"  Criteria: ROE>{crit['ROE_MIN']:.0%} | Debt<{crit['DEBT_MAX']:.0f} | "
          f"Marge>{crit['MARGE_MIN']:.0%} | "
          f"Vol {crit['VOL_MIN']:.0%}-{crit['VOL_MAX']:.0%} | "
          f"Omzet>€{crit['MIN_DAGOMZET']:,.0f}")
    print(f"{'='*65}")

    tickers = laad_tickers(getal)
    if not tickers:
        return {}

    ohlcv = batch_download(tickers)

    stats = {
        "totaal": len(tickers), "geslaagd": 0, "geen_data": 0,
        "roe_fail": 0, "debt_fail": 0, "marge_fail": 0,
        "vol_laag": 0, "vol_hoog": 0, "liq_fail": 0, "meerdere": 0,
    }

    regels   = []
    geslaagd = []

    for ticker in tickers:
        roe, debt, marge, fund_falen = check_fundamenteel(ticker, crit)
        time.sleep(0.3)

        if any("geen_fundamentele_data" in f or "api_fout" in f for f in fund_falen):
            stats["geen_data"] += 1
            regels.append(f"  {ticker:<16} ❓ Geen fundamentele data")
            continue

        vol, omzet, vol_falen = check_vol_liq(ticker, ohlcv, crit)
        alle_falen = (fund_falen or []) + (vol_falen or [])

        if not alle_falen:
            stats["geslaagd"] += 1
            geslaagd.append(ticker)
            regels.append(
                f"  {ticker:<16} ✅ "
                f"ROE={roe:.0%} Debt={debt:.0f} Marge={marge:.0%} "
                f"Vol={vol:.0%} Omzet=€{omzet:,.0f}"
            )
            continue

        if len(alle_falen) > 1:
            stats["meerdere"] += 1
        for f in alle_falen:
            if "ROE"   in f: stats["roe_fail"]   += 1
            if "Debt"  in f: stats["debt_fail"]  += 1
            if "Marge" in f: stats["marge_fail"] += 1
            if "Vol="  in f and "<" in f: stats["vol_laag"] += 1
            if "Vol="  in f and ">" in f: stats["vol_hoog"] += 1
            if "Omzet" in f: stats["liq_fail"]   += 1

        regels.append(f"  {ticker:<16} ❌ {' | '.join(alle_falen)}")

    # Samenvatting
    n   = stats["totaal"]
    ok  = stats["geslaagd"]
    pct = ok / n * 100 if n else 0

    samenvatting = [
        "",
        f"  ── SAMENVATTING {getal} — {naam} ──",
        f"  Totaal gescand  : {n}",
        f"  ✅ Door filter   : {ok} ({pct:.0f}%)",
        f"  ❓ Geen data     : {stats['geen_data']}",
        f"  ❌ ROE te laag   : {stats['roe_fail']}",
        f"  ❌ Debt te hoog  : {stats['debt_fail']}",
        f"  ❌ Marge te laag : {stats['marge_fail']}",
        f"  ❌ Vol te laag   : {stats['vol_laag']}",
        f"  ❌ Vol te hoog   : {stats['vol_hoog']}",
        f"  ❌ Illiquide     : {stats['liq_fail']}",
        f"  ⚠️  Meerdere fail : {stats['meerdere']} (tellen dubbel hierboven)",
    ]
    if geslaagd:
        samenvatting.append(f"\n  ✅ Geslaagd: {', '.join(geslaagd)}")

    print("\n".join(samenvatting))

    # Schrijf naar bestand
    uitvoer = (
        [f"DIAGNOSE {getal} — {naam} | {vandaag}", "=" * 65,
         f"Criteria: ROE>{crit['ROE_MIN']:.0%} | Debt<{crit['DEBT_MAX']:.0f} | "
         f"Marge>{crit['MARGE_MIN']:.0%} | "
         f"Vol {crit['VOL_MIN']:.0%}-{crit['VOL_MAX']:.0%} | "
         f"Omzet>€{crit['MIN_DAGOMZET']:,.0f}", ""]
        + regels
        + samenvatting
    )
    pad = f"diagnose_{getal}_{date.today().isoformat()}.txt"
    with open(pad, "w", encoding="utf-8") as f:
        f.write("\n".join(uitvoer))
    print(f"\n  💾 Opgeslagen: {pad}")

    # Telegram
    tg = (
        f"🔬 *Diagnose {getal} — {naam}*\n_{vandaag}_\n\n"
        f"Totaal: {n} | ✅ *{ok} ({pct:.0f}%)*\n\n"
        f"❌ Marge te laag : *{stats['marge_fail']}*\n"
        f"❌ Vol te laag   : *{stats['vol_laag']}*\n"
        f"❌ ROE te laag   : *{stats['roe_fail']}*\n"
        f"❌ Debt te hoog  : *{stats['debt_fail']}*\n"
        f"❌ Illiquide     : *{stats['liq_fail']}*\n"
        f"❓ Geen data     : *{stats['geen_data']}*\n"
    )
    if geslaagd:
        tg += f"\n✅ _Door filter: {', '.join(geslaagd)}_"

    send_telegram(tg)
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # GitHub Actions modus: DIAGNOSE_LIJST=041 → scan alleen lijst 041
    # Lokale modus: geen variabele → scan alle beschikbare lijsten
    lijst_env = os.environ.get("DIAGNOSE_LIJST", "").strip()

    if lijst_env:
        diagnose_lijst(lijst_env)
    else:
        for nr in range(41, 61):
            getal = f"0{nr}"
            if os.path.exists(f"tickers_{getal}a.txt"):
                diagnose_lijst(getal)
                print("\n  ⏳ Pauze 60s voor volgende lijst...")
                time.sleep(60)
