"""
MRA Filter Bot — bot_041m.py  (v3)
====================================
Wekelijkse evaluatie (zondag) van alle bronlijsten 041a t/m 060a.

FLOW PER LIJST:
  1. Laad bronlijst (tickers_041a.txt)
  2. Suffix-correctie per beurs
  3. Batch OHLCV download (1 jaar)
  4. Per ticker: Munger fundamentelen + Volatiliteit + Liquiditeit
  5. Masterlijst (m.txt) bijwerken:
       - Nieuwe tickers die filter halen → toevoegen met datum + parameters
       - Bestaande actieve tickers → parameters refreshen
       - Tickers die filter niet meer halen → weken_buiten teller ophogen
       - Na MAX_WEKEN_BUITEN weken → status "verwijderd"
  6. Exportlijst (x.txt) herschrijven:
       - Alle tickers met status actief of zwakker
       - Gesorteerd alfabetisch

BESTANDSNAMEN:
  tickers_041a.txt  → bronlijst (alle tickers van de beurs)
  tickers_041m.txt  → masterlijst (history + parameters per ticker)
  tickers_041x.txt  → exportlijst (actieve selectie voor triplex bot)
  tickers_041d.txt  → delisted cache

CRITERIA (gedifferentieerd per beurstype):
  Europa (041-046): ROE>7% | Debt<130 | Marge>4% | Vol 18%-65% | Omzet>150k
  Noord-Amerika (047-048): ROE>8% | Debt<120 | Marge>7% | Vol 22%-70% | Omzet>500k
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠️  Telegram niet geconfigureerd")
        return
    for i in range(0, len(tekst), 4096):
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst[i:i+4096],
                      "parse_mode": "Markdown"},
                timeout=15,
            ).raise_for_status()
            if i + 4096 < len(tekst):
                time.sleep(1)
        except Exception as e:
            print(f"  ⚠️  Telegram fout: {e}")

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

ALLE_SUFFIXEN = set()
for _cfg in BEURS_CONFIG.values():
    for _s in _cfg["suffixen"]:
        if _s:
            ALLE_SUFFIXEN.add(_s)

# ── Criteria per beurstype ────────────────────────────────────────────────────
EUROPA_BEURZEN = {"041", "042", "043", "044", "045", "046"}

CRITERIA = {
    "europa": {
        "ROE_MIN":      0.07,   # 7%
        "DEBT_MAX":     130.0,
        "MARGE_MIN":    0.04,   # 4%
        "VOL_MIN":      0.18,   # 18%
        "VOL_MAX":      0.65,   # 65%
        "MIN_DAGOMZET": 150_000,
    },
    "noordamerika": {
        "ROE_MIN":      0.08,   # 8%
        "DEBT_MAX":     120.0,
        "MARGE_MIN":    0.07,   # 7%
        "VOL_MIN":      0.22,   # 22%
        "VOL_MAX":      0.70,   # 70%
        "MIN_DAGOMZET": 500_000,
    },
}

def get_criteria(getal: str) -> dict:
    return CRITERIA["europa"] if getal in EUROPA_BEURZEN else CRITERIA["noordamerika"]

# ── Constanten ────────────────────────────────────────────────────────────────
MAX_WEKEN_BUITEN = 3    # weken buiten filter voor verwijdering
BATCH_SIZE       = 50
SLEEP_BATCH      = 2.0
SLEEP_INFO       = 0.3
REEKS_START      = 41
REEKS_EINDE      = 60

# ── Bestandsnamen ─────────────────────────────────────────────────────────────
def pad_bron(g):     return f"tickers_{g}a.txt"
def pad_master(g):   return f"tickers_{g}m.txt"
def pad_export(g):   return f"tickers_{g}x.txt"
def pad_delisted(g): return f"tickers_{g}d.txt"

# ── Suffix-correctie ──────────────────────────────────────────────────────────
def heeft_geldig_suffix(ticker: str, suffixen: list) -> bool:
    if suffixen == [""]:
        return not any(ticker.endswith(s) for s in ALLE_SUFFIXEN if s)
    return any(ticker.endswith(s) for s in suffixen)

def strip_suffix(ticker: str) -> str:
    for s in sorted(ALLE_SUFFIXEN, key=len, reverse=True):
        if ticker.endswith(s):
            return ticker[:-len(s)]
    return ticker

def corrigeer_suffix(ticker: str, suffixen: list) -> tuple:
    """Geeft (gecorrigeerde_ticker, was_gewijzigd, reden) terug."""
    if heeft_geldig_suffix(ticker, suffixen):
        return ticker, False, ""
    basis = strip_suffix(ticker)
    if suffixen == [""]:
        try:
            fi = yf.Ticker(basis).fast_info
            if getattr(fi, "last_price", None):
                return basis, ticker != basis, ""
        except Exception:
            pass
        return ticker, False, "niet gevonden Nasdaq/NYSE"
    for suffix in suffixen:
        kandidaat = basis + suffix
        try:
            fi = yf.Ticker(kandidaat).fast_info
            if getattr(fi, "last_price", None):
                return kandidaat, True, ""
        except Exception:
            pass
        time.sleep(0.2)
    return ticker, False, f"niet gevonden met {suffixen}"

# ── Delisted cache ────────────────────────────────────────────────────────────
def laad_delisted(g: str) -> set:
    pad = pad_delisted(g)
    if not os.path.exists(pad):
        return set()
    with open(pad, encoding="utf-8") as f:
        return set(t.strip().upper() for t in f.read().split(",") if t.strip())

def sla_delisted_op(g: str, delisted: set) -> None:
    with open(pad_delisted(g), "w", encoding="utf-8") as f:
        f.write(", ".join(sorted(delisted)))

# ── Batch OHLCV download ──────────────────────────────────────────────────────
def batch_download(tickers: list) -> dict:
    resultaat = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"  📥 OHLCV: {len(tickers)} tickers in {len(batches)} batches...")
    for i, batch in enumerate(batches):
        print(f"     Batch {i+1}/{len(batches)}...", end="", flush=True)
        try:
            raw = yf.download(batch, period="1y", progress=False,
                              auto_adjust=True, multi_level_index=True)
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
            time.sleep(SLEEP_BATCH)
    return resultaat

# ── Fundamentele check ────────────────────────────────────────────────────────
def check_fundamenteel(ticker: str, crit: dict) -> tuple:
    """Geeft (geslaagd, metrics_dict, reden_string) terug."""
    try:
        info = yf.Ticker(ticker).info
        if not info or "returnOnEquity" not in info:
            return False, {}, "geen_fundamentele_data"
        roe   = info.get("returnOnEquity", 0)    or 0
        debt  = info.get("debtToEquity",   9999) or 9999
        marge = info.get("profitMargins",  0)    or 0
        if 0 < debt < 2:
            debt *= 100

        metrics = {
            "ROE":   f"{roe:.1%}",
            "Debt":  f"{debt:.1f}",
            "Marge": f"{marge:.1%}",
        }
        falen = []
        if roe   < crit["ROE_MIN"]:   falen.append(f"ROE {roe:.1%}<{crit['ROE_MIN']:.0%}")
        if debt  > crit["DEBT_MAX"]:  falen.append(f"Debt {debt:.0f}>{crit['DEBT_MAX']:.0f}")
        if marge < crit["MARGE_MIN"]: falen.append(f"Marge {marge:.1%}<{crit['MARGE_MIN']:.0%}")
        return (not falen), metrics, " | ".join(falen)
    except Exception as e:
        return False, {}, f"api_fout:{e}"

# ── Volatiliteit + liquiditeit check ─────────────────────────────────────────
def check_vol_liq(ticker: str, ohlcv: dict, crit: dict) -> tuple:
    """Geeft (geslaagd, vol_str, omzet_str, reden_string) terug."""
    df = ohlcv.get(ticker)
    if df is None or len(df) < 50:
        return False, "?", "?", "te_weinig_data"
    try:
        vol   = float(df["Close"].ffill().pct_change().dropna().std() * np.sqrt(252))
        omzet = float((df["Close"] * df["Volume"]).mean()) if "Volume" in df.columns else 0.0
        vol_str   = f"{vol:.1%}"
        omzet_str = f"{omzet:,.0f}"

        falen = []
        if vol < crit["VOL_MIN"]:
            falen.append(f"Vol {vol:.1%}<{crit['VOL_MIN']:.0%}")
        elif vol > crit["VOL_MAX"]:
            falen.append(f"Vol {vol:.1%}>{crit['VOL_MAX']:.0%}")
        if omzet < crit["MIN_DAGOMZET"]:
            falen.append(f"Omzet {omzet:,.0f}<{crit['MIN_DAGOMZET']:,.0f}")
        return (not falen), vol_str, omzet_str, " | ".join(falen)
    except Exception as e:
        return False, "?", "?", f"vol_fout:{e}"

# ── Masterlijst lezen ─────────────────────────────────────────────────────────
def laad_master(g: str) -> dict:
    """
    Leest m.txt en geeft dict terug: {ticker: entry_dict}
    Elke entry bevat: ticker, status, opname, weken_buiten,
                      ROE, Debt, Marge, Vol, Omzet, [verwijderd]
    """
    master = {}
    pad = pad_master(g)
    if not os.path.exists(pad):
        return master
    with open(pad, encoding="utf-8") as f:
        for regel in f:
            regel = regel.strip()
            if not regel or regel.startswith("#"):
                continue
            delen = [d.strip() for d in regel.split("|")]
            if not delen:
                continue
            ticker = delen[0].strip().upper()
            entry  = {"ticker": ticker}
            for deel in delen[1:]:
                if ":" in deel:
                    k, v = deel.split(":", 1)
                    entry[k.strip()] = v.strip()
                else:
                    entry["status"] = deel.strip()
            entry["weken_buiten"] = int(entry.get("weken_buiten", 0))
            master[ticker] = entry
    return master

# ── Masterlijst schrijven ─────────────────────────────────────────────────────
def sla_master_op(g: str, master: dict, naam: str) -> None:
    vandaag  = date.today().strftime("%d/%m/%Y")
    crit     = get_criteria(g)
    regels   = [
        f"# MASTERLIJST {g} — {naam}",
        f"# Laatste update: {vandaag}",
        f"# Criteria: ROE>{crit['ROE_MIN']:.0%} | Debt<{crit['DEBT_MAX']:.0f} | "
        f"Marge>{crit['MARGE_MIN']:.0%} | "
        f"Vol {crit['VOL_MIN']:.0%}-{crit['VOL_MAX']:.0%} | "
        f"Omzet>€{crit['MIN_DAGOMZET']:,.0f}",
        f"# Kolommen: ticker | opname | ROE | Debt | Marge | Vol | Omzet | "
        f"status [| weken_buiten] [| verwijderd]",
        "# " + "-" * 80,
    ]

    volgorde   = {"actief": 0, "zwakker": 1, "verwijderd": 2}
    gesorteerd = sorted(
        master.values(),
        key=lambda e: (volgorde.get(e.get("status", "verwijderd"), 3), e["ticker"]),
    )

    for e in gesorteerd:
        t      = e["ticker"]
        status = e.get("status", "?")
        opname = e.get("opname", "?")

        if status == "verwijderd":
            verw = e.get("verwijderd", date.today().isoformat())
            regels.append(
                f"{t:<16} | opname:{opname} | "
                f"ROE:{e.get('ROE','?')} | Debt:{e.get('Debt','?')} | "
                f"Marge:{e.get('Marge','?')} | Vol:{e.get('Vol','?')} | "
                f"Omzet:{e.get('Omzet','?')} | verwijderd | verwijderd:{verw}"
            )
        else:
            regel = (
                f"{t:<16} | opname:{opname} | "
                f"ROE:{e.get('ROE','?')} | Debt:{e.get('Debt','?')} | "
                f"Marge:{e.get('Marge','?')} | Vol:{e.get('Vol','?')} | "
                f"Omzet:{e.get('Omzet','?')} | {status}"
            )
            if status == "zwakker":
                regel += f" | weken_buiten:{e.get('weken_buiten', 1)}"
            regels.append(regel)

    with open(pad_master(g), "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")

# ── Exportlijst schrijven ─────────────────────────────────────────────────────
def sla_export_op(g: str, master: dict) -> list:
    """
    Schrijft x.txt met alle actieve + zwakkere tickers, gesorteerd.
    Geeft de lijst terug.
    """
    export = sorted(
        t for t, e in master.items()
        if e.get("status") in ("actief", "zwakker")
    )
    with open(pad_export(g), "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export

# ── Master entry bijwerken ────────────────────────────────────────────────────
def update_master(master: dict, ticker: str, door_filter: bool,
                  metrics: dict) -> str:
    """
    Past de masterlijst aan voor één ticker.
    Geeft de nieuwe status terug: nieuw | actief | zwakker | verwijderd
    """
    vandaag = date.today().isoformat()

    if ticker not in master:
        if door_filter:
            # Nieuw door filter → toevoegen
            master[ticker] = {
                "ticker":       ticker,
                "status":       "actief",
                "opname":       vandaag,
                "weken_buiten": 0,
                **metrics,
            }
            return "nieuw"
        # Niet door filter én niet bekend → negeren
        return "onbekend"

    entry = master[ticker]

    # Parameters altijd refreshen (ook bij zwakker)
    for k, v in metrics.items():
        if v and v != "?":
            entry[k] = v

    if door_filter:
        entry["status"]       = "actief"
        entry["weken_buiten"] = 0
        return "actief"
    else:
        # Verwijderde tickers niet opnieuw activeren
        if entry.get("status") == "verwijderd":
            return "verwijderd"
        entry["weken_buiten"] = entry.get("weken_buiten", 0) + 1
        if entry["weken_buiten"] >= MAX_WEKEN_BUITEN:
            entry["status"]     = "verwijderd"
            entry["verwijderd"] = vandaag
            return "verwijderd"
        entry["status"] = "zwakker"
        return "zwakker"

# ── Scan één lijst ────────────────────────────────────────────────────────────
def scan_lijst(getal: str) -> dict:
    config   = BEURS_CONFIG.get(getal, {"naam": f"Lijst {getal}", "suffixen": []})
    naam     = config["naam"]
    suffixen = config["suffixen"]
    crit     = get_criteria(getal)
    bron     = pad_bron(getal)

    print(f"\n{'='*65}")
    print(f"  📋 LIJST {getal} — {naam}")
    print(f"  Criteria: ROE>{crit['ROE_MIN']:.0%} | Debt<{crit['DEBT_MAX']:.0f} | "
          f"Marge>{crit['MARGE_MIN']:.0%} | "
          f"Vol {crit['VOL_MIN']:.0%}-{crit['VOL_MAX']:.0%} | "
          f"Omzet>€{crit['MIN_DAGOMZET']:,.0f}")
    print(f"{'='*65}")

    # ── 1. Bronlijst laden ────────────────────────────────────────────────────
    with open(bron, encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace(";", ",").replace("$", "")
    ruwe_tickers = sorted(set(t.strip().upper() for t in inhoud.split(",") if t.strip()))
    print(f"  📂 {len(ruwe_tickers)} tickers in bronlijst")

    # ── 2. Suffix-correctie ───────────────────────────────────────────────────
    tickers     = []
    correcties  = []
    niet_gevonden = []

    for ticker in ruwe_tickers:
        if heeft_geldig_suffix(ticker, suffixen):
            tickers.append(ticker)
        else:
            gecorr, gewijzigd, reden = corrigeer_suffix(ticker, suffixen)
            if gewijzigd:
                tickers.append(gecorr)
                correcties.append((ticker, gecorr))
            elif not reden:
                tickers.append(gecorr)
            else:
                niet_gevonden.append((ticker, reden))

    if correcties:
        print(f"  ✏️  {len(correcties)} suffix-correcties")
    if niet_gevonden:
        print(f"  ❌ {len(niet_gevonden)} tickers niet te corrigeren")

    # ── 3. Delisted cache ─────────────────────────────────────────────────────
    delisted       = laad_delisted(getal)
    te_scannen     = [t for t in tickers if t not in delisted]
    print(f"  ⚡ {len(delisted)} delisted overgeslagen | {len(te_scannen)} te scannen")

    # ── 4. Master laden ───────────────────────────────────────────────────────
    master = laad_master(getal)
    print(f"  📁 {len(master)} tickers gekend in master")

    # ── 5. OHLCV download ─────────────────────────────────────────────────────
    ohlcv = batch_download(te_scannen)

    # Detecteer nieuwe delisted
    nieuw_delisted = {t for t, df in ohlcv.items() if df is None}
    delisted.update(nieuw_delisted)
    sla_delisted_op(getal, delisted)

    actief = [t for t in te_scannen if t not in nieuw_delisted]
    print(f"  ✅ {len(actief)} met data | ❌ {len(nieuw_delisted)} nieuw delisted\n")

    # ── 6. Evaluatie per ticker ───────────────────────────────────────────────
    tellers = {
        "nieuw": [], "actief": [], "zwakker": [],
        "verwijderd": [], "geen_data": [],
    }

    for ticker in actief:
        print(f"  {ticker:<16} ", end="", flush=True)

        # Fundamentelen
        fund_ok, fund_metrics, fund_reden = check_fundamenteel(ticker, crit)
        time.sleep(SLEEP_INFO)

        if "geen_fundamentele_data" in fund_reden or "api_fout" in fund_reden:
            print(f"❓ {fund_reden}")
            tellers["geen_data"].append(ticker)
            continue

        # Volatiliteit + liquiditeit
        vol_ok, vol_str, omzet_str, vol_reden = check_vol_liq(ticker, ohlcv, crit)

        door_filter = fund_ok and vol_ok
        metrics     = {**fund_metrics, "Vol": vol_str, "Omzet": omzet_str}
        status      = update_master(master, ticker, door_filter, metrics)

        if door_filter:
            print(f"✅ {fund_metrics.get('ROE','')} Debt:{fund_metrics.get('Debt','')} "
                  f"Marge:{fund_metrics.get('Marge','')} Vol:{vol_str} "
                  f"Omzet:€{omzet_str} → {status.upper()}")
        else:
            reden = fund_reden or vol_reden
            print(f"❌ {reden} → {status}")

        tellers.get(status, tellers["geen_data"]).append(ticker)

    # ── 7. Opslaan ────────────────────────────────────────────────────────────
    sla_master_op(getal, master, naam)
    export = sla_export_op(getal, master)

    print(f"\n  💾 Master: {pad_master(getal)} | Export: {pad_export(getal)}")
    print(f"  📊 Exportlijst: {len(export)} tickers")

    return {
        "getal":   getal,
        "naam":    naam,
        "tellers": tellers,
        "export":  export,
        "correcties":   correcties,
        "niet_gevonden": niet_gevonden,
    }

# ── Hoofd scan ────────────────────────────────────────────────────────────────
def scan_alle() -> None:
    start     = time.time()
    vandaag   = date.today().strftime("%d/%m/%Y")
    verwerkt  = []
    overgesl  = []
    resultaten = {}

    print(f"\n{'='*65}")
    print(f"  🤖 MRA FILTER BOT v3 — WEKELIJKSE SCAN")
    print(f"  📅 {vandaag}")
    print(f"{'='*65}")

    for nr in range(REEKS_START, REEKS_EINDE + 1):
        getal = f"0{nr}"
        bron  = pad_bron(getal)
        if not os.path.exists(bron):
            overgesl.append(getal)
            continue
        verwerkt.append(getal)
        resultaten[getal] = scan_lijst(getal)
        time.sleep(2)

    # ── Samenvatting ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    m, s    = int(elapsed // 60), int(elapsed % 60)

    print(f"\n{'='*65}")
    print(f"  ✅ KLAAR in {m}m {s}s | {len(verwerkt)} lijsten verwerkt")
    print(f"{'='*65}")

    tg = (
        f"🤖 *MRA Filter Bot v3*\n_{vandaag}_\n"
        f"⏱ {m}m {s}s | {len(verwerkt)} lijsten\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for getal, res in resultaten.items():
        t      = res["tellers"]
        export = res["export"]
        naam   = res["naam"]

        # Console
        print(f"  {getal} {naam}: "
              f"🆕{len(t['nieuw'])} ✅{len(t['actief'])} "
              f"⚠️{len(t['zwakker'])} ❌{len(t['verwijderd'])} "
              f"→ {len(export)} export")

        # Telegram
        tg += f"*{getal} — {naam}* → {len(export)} tickers\n"
        if t["nieuw"]:
            tg += f"  🆕 {', '.join(f'`{x}`' for x in t['nieuw'])}\n"
        if t["verwijderd"]:
            tg += f"  🗑 {', '.join(f'`{x}`' for x in t['verwijderd'])}\n"
        if t["zwakker"]:
            tg += f"  ⚠️ {', '.join(f'`{x}`' for x in t['zwakker'])}\n"
        if res["correcties"]:
            tg += f"  ✏️ {len(res['correcties'])} suffix-correcties\n"
        if res["niet_gevonden"]:
            tg += f"  ❓ {len(res['niet_gevonden'])} niet gevonden\n"
        if not any([t["nieuw"], t["verwijderd"], t["zwakker"]]):
            tg += f"  ✅ Geen wijzigingen\n"
        tg += "\n"

    tg += f"_Volgende run: zondag_"
    send_telegram(tg)

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scan_alle()
