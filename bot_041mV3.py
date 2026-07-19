#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRA FILTER BOT v4
- 1 script
- Yahoo Finance fallback
- Absolute criteria
- Verbeterde suffix-correctie
- Minimale sleeps (maar veilig voor Yahoo)
- Batch OHLCV download
- Compatibel met jouw masterlijst-structuur
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM (optioneel)
# ─────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=15
        )
    except:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# BEURS CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BEURS_CONFIG = {
    "041": {"naam": "Benelux",         "suffixen": [".AS", ".BR", ".LU"]},
    "042": {"naam": "Parijs",          "suffixen": [".PA"]},
    "043": {"naam": "Frankfurt",       "suffixen": [".DE"]},
    "044": {"naam": "Spanje/Portugal", "suffixen": [".MC", ".LS"]},
    "045": {"naam": "Londen",          "suffixen": [".L"]},
    "046": {"naam": "Milaan",          "suffixen": [".MI"]},
    "047": {"naam": "Toronto",         "suffixen": [".TO", ".V"]},
    "048": {"naam": "Nasdaq/NYSE",     "suffixen": [""]},
    "049": {"naam": "Stockholm",       "suffixen": [".ST"]},
    "050": {"naam": "Zurich",          "suffixen": [".SW"]},
    "051": {"naam": "Warschau",        "suffixen": [".WA"]},
    "052": {"naam": "Oslo",            "suffixen": [".OL"]},
    "053": {"naam": "Kopenhagen",      "suffixen": [".CO"]},
    "054": {"naam": "Helsinki",        "suffixen": [".HE"]},
}

EUROPA = {"041", "042", "043", "044", "045", "046", "049", "050", "051", "052", "053", "054"}

CRITERIA = {
    "europa": {
        "ROE_MIN":      0.07,
        "DEBT_MAX":     130,
        "MARGE_MIN":    0.04,
        "VOL_MIN":      0.18,
        "VOL_MAX":      0.65,
        "MIN_DAGOMZET": 150_000,
    },
    "noordamerika": {
        "ROE_MIN":      0.08,
        "DEBT_MAX":     120,
        "MARGE_MIN":    0.07,
        "VOL_MIN":      0.22,
        "VOL_MAX":      0.70,
        "MIN_DAGOMZET": 500_000,
    },
}

def get_criteria(g):
    return CRITERIA["europa"] if g in EUROPA else CRITERIA["noordamerika"]

# ─────────────────────────────────────────────────────────────────────────────
# BESTANDSLOCATIES
# ─────────────────────────────────────────────────────────────────────────────
def pad_bron(g):     return f"tickers_{g}a.txt"
def pad_master(g):   return f"tickers_{g}m.txt"
def pad_export(g):   return f"tickers_{g}x.txt"
def pad_delisted(g): return f"tickers_{g}d.txt"

# ─────────────────────────────────────────────────────────────────────────────
# SUFFIX CORRECTIE (v4)
# ─────────────────────────────────────────────────────────────────────────────
ALLE_SUFFIXEN = {s for cfg in BEURS_CONFIG.values() for s in cfg["suffixen"] if s}
SUFFIX_CACHE = {}

def strip_suffix(t):
    for s in sorted(ALLE_SUFFIXEN, key=len, reverse=True):
        if t.endswith(s):
            return t[:-len(s)]
    return t

def corrigeer_suffix(ticker, suffixen):
    """Snelle suffix-correctie met caching en minimale sleeps."""
    if ticker in SUFFIX_CACHE:
        return SUFFIX_CACHE[ticker]

    basis = strip_suffix(ticker)

    # Amerikaanse tickers
    if suffixen == [""]:
        try:
            fi = yf.Ticker(basis).fast_info
            if getattr(fi, "last_price", None):
                SUFFIX_CACHE[ticker] = (basis, basis != ticker, "")
                return SUFFIX_CACHE[ticker]
        except:
            pass
        SUFFIX_CACHE[ticker] = (ticker, False, "niet gevonden")
        return SUFFIX_CACHE[ticker]

    # Europese tickers
    for s in suffixen:
        k = basis + s
        try:
            fi = yf.Ticker(k).fast_info
            if getattr(fi, "last_price", None):
                SUFFIX_CACHE[ticker] = (k, k != ticker, "")
                return SUFFIX_CACHE[ticker]
        except:
            pass
        time.sleep(0.05)

    SUFFIX_CACHE[ticker] = (ticker, False, "niet gevonden")
    return SUFFIX_CACHE[ticker]

# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS (Yahoo fallback)
# ─────────────────────────────────────────────────────────────────────────────
def haal_fundamenteel(ticker):
    try:
        info = yf.Ticker(ticker).info
        roe   = info.get("returnOnEquity")
        debt  = info.get("debtToEquity")
        marge = info.get("profitMargins")

        if roe is None or debt is None or marge is None:
            return None

        if 0 < debt < 2:
            debt *= 100

        return {
            "ROE": float(roe),
            "Debt": float(debt),
            "Marge": float(marge),
        }
    except:
        return None
    finally:
        time.sleep(0.10)

# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITEIT + LIQUIDITEIT
# ─────────────────────────────────────────────────────────────────────────────
def analyse_ohlcv(df):
    if df is None or len(df) < 50:
        return None, None
    try:
        returns = df["Close"].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)

        dv = (df["Close"] * df["Volume"]).rolling(20).median().dropna()
        omzet = float(dv.median()) if len(dv) else None

        return float(vol), omzet
    except:
        return None, None

# ─────────────────────────────────────────────────────────────────────────────
# MASTERLIJST
# ─────────────────────────────────────────────────────────────────────────────
def laad_master(g):
    master = {}
    p = pad_master(g)
    if not os.path.exists(p):
        return master

    with open(p, encoding="utf-8") as f:
        for r in f:
            r = r.strip()
            if not r or r.startswith("#"):
                continue
            delen = [d.strip() for d in r.split("|")]
            t = delen[0].strip().upper()
            entry = {"ticker": t}
            for d in delen[1:]:
                if ":" in d:
                    k, v = d.split(":", 1)
                    entry[k.strip()] = v.strip()
                else:
                    entry["status"] = d.strip()
            entry["weken_buiten"] = int(entry.get("weken_buiten", 0))
            master[t] = entry
    return master

def sla_master_op(g, master, naam):
    vandaag = date.today().strftime("%d/%m/%Y")
    crit = get_criteria(g)

    regels = [
        f"# MASTERLIJST {g} — {naam}",
        f"# Laatste update: {vandaag}",
        f"# Criteria: ROE>{crit['ROE_MIN']:.0%} | Debt<{crit['DEBT_MAX']} | Marge>{crit['MARGE_MIN']:.0%} | Vol {crit['VOL_MIN']:.0%}-{crit['VOL_MAX']:.0%} | Omzet>{crit['MIN_DAGOMZET']:,}",
        "# " + "-"*80,
    ]

    volgorde = {"actief": 0, "zwakker": 1, "verwijderd": 2}
    gesorteerd = sorted(master.values(), key=lambda e: (volgorde.get(e.get("status","verwijderd"),3), e["ticker"]))

    for e in gesorteerd:
        t = e["ticker"]
        status = e.get("status","?")
        opname = e.get("opname","?")

        regel = f"{t:<16} | opname:{opname} | ROE:{e.get('ROE','?')} | Debt:{e.get('Debt','?')} | Marge:{e.get('Marge','?')} | Vol:{e.get('Vol','?')} | Omzet:{e.get('Omzet','?')} | {status}"
        if status == "zwakker":
            regel += f" | weken_buiten:{e.get('weken_buiten',0)}"
        if status == "verwijderd":
            regel += f" | verwijderd:{e.get('verwijderd', vandaag)}"
        regels.append(regel)

    with open(pad_master(g), "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")

def sla_export_op(g, master):
    export = sorted(t for t,e in master.items() if e.get("status") in ("actief","zwakker"))
    with open(pad_export(g), "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export

# ─────────────────────────────────────────────────────────────────────────────
# UPDATE MASTER ENTRY
# ─────────────────────────────────────────────────────────────────────────────
MAX_WEKEN_BUITEN = 3

def update_master(master, ticker, door_filter, metrics):
    vandaag = date.today().isoformat()

    if ticker not in master:
        if door_filter:
            master[ticker] = {
                "ticker": ticker,
                "status": "actief",
                "opname": vandaag,
                "weken_buiten": 0,
                **metrics,
            }
            return "nieuw"
        return "onbekend"

    e = master[ticker]

    for k,v in metrics.items():
        if v is not None:
            e[k] = v

    if door_filter:
        e["status"] = "actief"
        e["weken_buiten"] = 0
        return "actief"

    if e.get("status") == "verwijderd":
        return "verwijderd"

    e["weken_buiten"] += 1
    if e["weken_buiten"] >= MAX_WEKEN_BUITEN:
        e["status"] = "verwijderd"
        e["verwijderd"] = vandaag
        return "verwijderd"

    e["status"] = "zwakker"
    return "zwakker"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH OHLCV DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def batch_download(tickers):
    resultaat = {}
    batches = [tickers[i:i+50] for i in range(0, len(tickers), 50)]

    for i, batch in enumerate(batches):
        try:
            raw = yf.download(batch, period="1y", auto_adjust=True, progress=False)
            for t in batch:
                try:
                    df = raw.xs(t, axis=1, level=1).dropna(how="all")
                    resultaat[t] = df if len(df) >= 50 else None
                except:
                    resultaat[t] = None
        except:
            for t in batch:
                resultaat[t] = None

        if i < len(batches) - 1:
            time.sleep(0.5)

    return resultaat

# ─────────────────────────────────────────────────────────────────────────────
# SCAN LIJST
# ─────────────────────────────────────────────────────────────────────────────
def scan_lijst(g):
    cfg = BEURS_CONFIG[g]
    naam = cfg["naam"]
    suffixen = cfg["suffixen"]
    crit = get_criteria(g)

    print(f"\n=== LIJST {g} — {naam} ===")

    # bronlijst
    with open(pad_bron(g), encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace(";", ",").replace("$","")
    ruwe = sorted(set(t.strip().upper() for t in inhoud.split(",") if t.strip()))

    tickers = []
    for t in ruwe:
        nieuw, gew, reden = corrigeer_suffix(t, suffixen)
        if reden:
            continue
        tickers.append(nieuw)

    master = laad_master(g)

    print("Download OHLCV…")
    ohlcv = batch_download(tickers)

    tellers = {"nieuw":[], "actief":[], "zwakker":[], "verwijderd":[], "geen_data":[]}

    for t in tickers:
        print(f"{t:<12} ", end="")

        df = ohlcv.get(t)
        if df is None:
            print("❌ geen data")
            tellers["geen_data"].append(t)
            continue

        fund = haal_fundamenteel(t)
        if not fund:
            print("❓ geen fundamentals")
            tellers["geen_data"].append(t)
            continue

        vol, omzet = analyse_ohlcv(df)
        if vol is None or omzet is None:
            print("❌ vol/liquiditeit fout")
            tellers["geen_data"].append(t)
            continue

        door = (
            fund["ROE"] >= crit["ROE_MIN"] and
            fund["Debt"] <= crit["DEBT_MAX"] and
            fund["Marge"] >= crit["MARGE_MIN"] and
            crit["VOL_MIN"] <= vol <= crit["VOL_MAX"] and
            omzet >= crit["MIN_DAGOMZET"]
        )

        metrics = {
            "ROE": f"{fund['ROE']:.1%}",
            "Debt": f"{fund['Debt']:.1f}",
            "Marge": f"{fund['Marge']:.1%}",
            "Vol": f"{vol:.1%}",
            "Omzet": f"{omzet:,.0f}",
        }

        status = update_master(master, t, door, metrics)
        tellers.get(status, tellers["geen_data"]).append(t)

        if door:
            print(f"✅ ROE:{metrics['ROE']} Debt:{metrics['Debt']} Marge:{metrics['Marge']} Vol:{metrics['Vol']} Omzet:{metrics['Omzet']}")
        else:
            print("❌ faalt criteria")

    sla_master_op(g, master, naam)
    export = sla_export_op(g, master)

    print(f"Export: {len(export)} tickers")
    return tellers, export

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def scan_alle():
    print("MRA FILTER BOT v4 — START")
    for nr in range(41, 61):
        g = f"0{nr}"
        if os.path.exists(pad_bron(g)):
            scan_lijst(g)
            time.sleep(1)

if __name__ == "__main__":
    scan_alle()
