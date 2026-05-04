"""
MRA Filter Bot — bot_041m.py
=============================
Verwerkt tickerlijsten per beurs met automatische suffix-correctie:
  041 → Benelux          (.AS / .BR / .LU)
  042 → Parijs           (.PA)
  043 → Frankfurt        (.DE)
  044 → Spanje/Portugal  (.MC / .LS)
  045 → Londen           (.L)
  046 → Milaan           (.MI)
  047 → Toronto          (.TO / .V)
  048 → Nasdaq/NYSE      (geen suffix)

Suffix-correctie: als een ticker het verkeerde of geen suffix heeft,
wordt automatisch het correcte suffix voor de betreffende beurs geprobeerd.

Filtert op:
  1. Suffix check + automatische correctie
  2. Munger kwaliteitscriteria (ROE>10%, Debt<100, Marge>7%)
  3. MRA-geschikte volatiliteit (18% - 65% jaarlijks)
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# BEURS CONFIGURATIE
# Per lijstnummer: naam + geldige suffixen (volgorde = prioriteit bij correctie)
# Lege string "" = geen suffix (Nasdaq/NYSE)
# ---------------------------------------------------------------------------
BEURS_CONFIG = {
    "041": {
        "naam":     "Benelux",
        "suffixen": [".AS", ".BR", ".LU"],
    },
    "042": {
        "naam":     "Parijs",
        "suffixen": [".PA"],
    },
    "043": {
        "naam":     "Frankfurt",
        "suffixen": [".DE"],
    },
    "044": {
        "naam":     "Spanje/Portugal",
        "suffixen": [".MC", ".LS"],
    },
    "045": {
        "naam":     "Londen",
        "suffixen": [".L"],
    },
    "046": {
        "naam":     "Milaan",
        "suffixen": [".MI"],
    },
    "047": {
        "naam":     "Toronto",
        "suffixen": [".TO", ".V"],
    },
    "048": {
        "naam":     "Nasdaq/NYSE",
        "suffixen": [""],
    },
}

# Alle geldige suffixen gecombineerd
ALLE_SUFFIXEN = set()
for cfg in BEURS_CONFIG.values():
    for s in cfg["suffixen"]:
        if s:
            ALLE_SUFFIXEN.add(s)

# ---------------------------------------------------------------------------
# CONFIG — criteria
# ---------------------------------------------------------------------------
ROE_MIN          = 0.10
DEBT_MAX         = 100.0
MARGE_MIN        = 0.07
VOL_MIN          = 0.18
VOL_MAX          = 0.65
MAX_WEKEN_BUITEN = 2

BATCH_SIZE       = 50
SLEEP_BATCH      = 2.0
SLEEP_INFO       = 0.3

REEKS_START      = 41
REEKS_EINDE      = 60


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram(bericht: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram niet geconfigureerd.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       bericht,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️  Telegram fout: {e}")


# ---------------------------------------------------------------------------
# BESTANDSNAMEN
# ---------------------------------------------------------------------------
def bestand_bron(getal):     return f"tickers_{getal}a.txt"
def bestand_master(getal):   return f"tickers_{getal}m.txt"
def bestand_export(getal):   return f"tickers_{getal}x.txt"
def bestand_delisted(getal): return f"tickers_{getal}d.txt"


# ---------------------------------------------------------------------------
# SUFFIX CORRECTIE
# ---------------------------------------------------------------------------
def heeft_geldig_suffix(ticker: str, suffixen: list) -> bool:
    """Controleert of ticker al een geldig suffix voor deze beurs heeft."""
    if suffixen == [""]:
        # Nasdaq/NYSE: ticker mag geen Europees/.TO suffix hebben
        return not any(ticker.endswith(s) for s in ALLE_SUFFIXEN if s)
    return any(ticker.endswith(s) for s in suffixen)


def strip_suffix(ticker: str) -> str:
    """Verwijdert elk bekend suffix van een ticker."""
    for s in sorted(ALLE_SUFFIXEN, key=len, reverse=True):
        if ticker.endswith(s):
            return ticker[:-len(s)]
    return ticker


def corrigeer_suffix(ticker: str, suffixen: list) -> tuple:
    """
    Probeert de ticker te corrigeren naar het juiste suffix voor deze beurs.
    Geeft (gecorrigeerde_ticker, was_gecorrigeerd, reden) terug.
    """
    # Al correct?
    if heeft_geldig_suffix(ticker, suffixen):
        return ticker, False, ""

    basis = strip_suffix(ticker)

    # Geen suffix nodig (Nasdaq/NYSE)
    if suffixen == [""]:
        kandidaat = basis
        try:
            fi = yf.Ticker(kandidaat).fast_info
            prijs = getattr(fi, "last_price", None)
            if prijs and prijs > 0:
                was_gewijzigd = ticker != kandidaat
                return kandidaat, was_gewijzigd, ""
        except Exception:
            pass
        return ticker, False, "niet gevonden op Nasdaq/NYSE"

    # Probeer elk geldig suffix voor deze beurs
    for suffix in suffixen:
        kandidaat = basis + suffix
        try:
            fi = yf.Ticker(kandidaat).fast_info
            prijs = getattr(fi, "last_price", None)
            if prijs and prijs > 0:
                return kandidaat, True, ""
        except Exception:
            pass
        time.sleep(0.2)

    return ticker, False, f"niet gevonden met {suffixen}"


# ---------------------------------------------------------------------------
# DELISTED CACHE
# ---------------------------------------------------------------------------
def laad_delisted(getal: str) -> set:
    pad = bestand_delisted(getal)
    if not os.path.exists(pad):
        return set()
    with open(pad, "r", encoding="utf-8") as f:
        return set(t.strip().upper() for t in f.read().split(",") if t.strip())


def sla_delisted_op(getal: str, delisted: set) -> None:
    with open(bestand_delisted(getal), "w", encoding="utf-8") as f:
        f.write(", ".join(sorted(delisted)))


# ---------------------------------------------------------------------------
# BATCH OHLCV DOWNLOAD
# ---------------------------------------------------------------------------
def batch_download_ohlcv(tickers: list) -> dict:
    resultaat = {}
    if not tickers:
        return resultaat

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(f"\n  📥 OHLCV batch download: {len(tickers)} tickers "
          f"in {len(batches)} batches van {BATCH_SIZE}...")

    for i, batch in enumerate(batches):
        print(f"     Batch {i+1}/{len(batches)} ({len(batch)} tickers)...", end="", flush=True)
        try:
            raw = yf.download(
                batch,
                period="1y",
                progress=False,
                auto_adjust=True,
                multi_level_index=True,
            )
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns[0], tuple):
                            df.columns = [c[0] for c in df.columns]
                    else:
                        df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
                    resultaat[ticker] = df if len(df) >= 50 else None
                except Exception:
                    resultaat[ticker] = None
            print(f" ✅")
        except Exception as e:
            print(f" ❌ {e}")
            for ticker in batch:
                resultaat[ticker] = None

        if i < len(batches) - 1:
            time.sleep(SLEEP_BATCH)

    return resultaat


# ---------------------------------------------------------------------------
# MASTERLIJST — lezen
# ---------------------------------------------------------------------------
def laad_master(getal: str) -> dict:
    master = {}
    pad    = bestand_master(getal)
    if not os.path.exists(pad):
        return master
    with open(pad, "r", encoding="utf-8") as f:
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
                    sleutel, waarde = deel.split(":", 1)
                    entry[sleutel.strip()] = waarde.strip()
                else:
                    entry["status"] = deel.strip()
            entry["weken_buiten"] = int(entry.get("weken_buiten", 0))
            master[ticker] = entry
    return master


# ---------------------------------------------------------------------------
# MASTERLIJST — schrijven
# ---------------------------------------------------------------------------
def sla_master_op(getal: str, master: dict, beurs_naam: str) -> None:
    vandaag = date.today().strftime("%d/%m/%Y")
    regels  = [
        f"# MASTERLIJST — {beurs_naam} | bron: {bestand_bron(getal)}",
        f"# Laatste update: {vandaag}",
        f"# Criteria (soepel): ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | "
        f"Marge>{MARGE_MIN:.0%} | Vol {VOL_MIN:.0%}-{VOL_MAX:.0%}",
        f"# Status: actief | zwakker (max {MAX_WEKEN_BUITEN} weken) | verwijderd",
        "# " + "-" * 70,
    ]

    volgorde   = {"nieuw": 0, "actief": 0, "zwakker": 1, "verwijderd": 2}
    gesorteerd = sorted(
        master.values(),
        key=lambda e: (volgorde.get(e.get("status", "verwijderd"), 3), e["ticker"]),
    )

    for entry in gesorteerd:
        t      = entry["ticker"]
        status = entry.get("status", "?")
        opname = entry.get("opname", "?")

        if status == "verwijderd":
            verw = entry.get("verwijderd", date.today().isoformat())
            regels.append(f"{t:<16} | opname:{opname} | verwijderd:{verw}")
        else:
            roe   = entry.get("ROE",   "?")
            debt  = entry.get("Debt",  "?")
            marge = entry.get("Marge", "?")
            vol   = entry.get("Vol",   "?")
            regel = (
                f"{t:<16} | opname:{opname} | "
                f"ROE:{roe} | Debt:{debt} | Marge:{marge} | Vol:{vol} | {status}"
            )
            if status == "zwakker":
                regel += f" | weken_buiten:{entry.get('weken_buiten', 1)}"
            regels.append(regel)

    with open(bestand_master(getal), "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")


# ---------------------------------------------------------------------------
# EXPORT — schrijven
# ---------------------------------------------------------------------------
def sla_export_op(getal: str, master: dict) -> list:
    export = sorted(
        t for t, e in master.items()
        if e.get("status") in ("nieuw", "actief", "zwakker")
    )
    with open(bestand_export(getal), "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export


# ---------------------------------------------------------------------------
# LAAG 2 — MUNGER KWALITEITSFILTER
# ---------------------------------------------------------------------------
def check_munger(ticker: str) -> tuple:
    try:
        info  = yf.Ticker(ticker).info
        if not info or "returnOnEquity" not in info:
            return False, {}, "geen fundamentele data"

        roe   = info.get("returnOnEquity", 0)    or 0
        debt  = info.get("debtToEquity",   9999) or 9999
        marge = info.get("profitMargins",  0)    or 0

        if 0 < debt < 2:
            debt = debt * 100

        metrics = {
            "ROE":   f"{roe:.0%}",
            "Debt":  f"{debt:.1f}",
            "Marge": f"{marge:.0%}",
        }

        if roe >= ROE_MIN and debt <= DEBT_MAX and marge >= MARGE_MIN:
            return True, metrics, ""

        redenen = []
        if roe   < ROE_MIN:   redenen.append(f"ROE {roe:.1%}<{ROE_MIN:.0%}")
        if debt  > DEBT_MAX:  redenen.append(f"Debt {debt:.0f}>{DEBT_MAX:.0f}")
        if marge < MARGE_MIN: redenen.append(f"Marge {marge:.1%}<{MARGE_MIN:.0%}")
        return False, metrics, " | ".join(redenen)

    except Exception as e:
        return False, {}, str(e)


# ---------------------------------------------------------------------------
# LAAG 3 — VOLATILITEITSFILTER
# ---------------------------------------------------------------------------
def check_volatiliteit(ticker: str, ohlcv_cache: dict) -> tuple:
    df = ohlcv_cache.get(ticker)
    if df is None or len(df) < 50:
        return False, 0.0, "te weinig data"
    try:
        vol = float(df["Close"].pct_change().dropna().std() * np.sqrt(252))
        if vol < VOL_MIN:
            return False, vol, f"te laag ({vol:.0%}<{VOL_MIN:.0%})"
        if vol > VOL_MAX:
            return False, vol, f"te hoog ({vol:.0%}>{VOL_MAX:.0%})"
        return True, vol, ""
    except Exception as e:
        return False, 0.0, str(e)


# ---------------------------------------------------------------------------
# MASTER BIJWERKEN
# ---------------------------------------------------------------------------
def update_entry(master: dict, ticker: str, door_filter: bool, metrics: dict) -> str:
    vandaag = date.today().isoformat()

    if ticker not in master:
        if door_filter:
            master[ticker] = {
                "ticker":       ticker,
                "status":       "nieuw",
                "opname":       vandaag,
                "weken_buiten": 0,
                **metrics,
            }
            return "nieuw"
        return "onbekend"

    entry = master[ticker]
    entry.update({k: v for k, v in metrics.items()})

    if door_filter:
        entry["status"]       = "actief"
        entry["weken_buiten"] = 0
        return "actief"
    else:
        entry["weken_buiten"] = entry.get("weken_buiten", 0) + 1
        if entry["weken_buiten"] >= MAX_WEKEN_BUITEN:
            entry["status"]     = "verwijderd"
            entry["verwijderd"] = vandaag
            return "verwijderd"
        else:
            entry["status"] = "zwakker"
            return "zwakker"


# ---------------------------------------------------------------------------
# SCAN ÉÉN LIJST
# ---------------------------------------------------------------------------
def scan_lijst(getal: str) -> dict:
    config     = BEURS_CONFIG.get(getal, {"naam": f"Lijst {getal}", "suffixen": []})
    beurs_naam = config["naam"]
    suffixen   = config["suffixen"]
    bron       = bestand_bron(getal)

    print(f"\n{'='*60}")
    print(f"  🔍 LIJST {getal} — {beurs_naam}")
    print(f"  📋 Bron     : {bron}")
    print(f"  📁 Master   : {bestand_master(getal)}")
    print(f"  📤 Export   : {bestand_export(getal)}")
    suffix_label = suffixen if suffixen != [""] else ["geen (Nasdaq/NYSE)"]
    print(f"  🏦 Suffixen : {suffix_label}")
    print(f"{'='*60}")

    # Bronbestand laden
    with open(bron, "r", encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace(";", ",").replace("$", "")
    ruwe_tickers = sorted(set(
        t.strip().upper() for t in inhoud.split(",") if t.strip()
    ))

    print(f"  📊 {len(ruwe_tickers)} tickers geladen uit bronbestand")

    # --- SUFFIX CORRECTIE ---
    gecorrigeerde_tickers = []
    correcties            = []
    niet_gevonden         = []

    print(f"\n  🔧 Suffix-correctie...")
    for ticker in ruwe_tickers:
        if heeft_geldig_suffix(ticker, suffixen):
            gecorrigeerde_tickers.append(ticker)
        else:
            gecorrigeerd, was_gewijzigd, reden = corrigeer_suffix(ticker, suffixen)
            if was_gewijzigd:
                gecorrigeerde_tickers.append(gecorrigeerd)
                correcties.append((ticker, gecorrigeerd))
                print(f"     ✏️  {ticker} → {gecorrigeerd}")
            elif not reden:
                # Geen suffix nodig of al correct
                gecorrigeerde_tickers.append(gecorrigeerd)
            else:
                niet_gevonden.append((ticker, reden))
                print(f"     ❌ {ticker}: {reden}")

    if correcties:
        print(f"  ✏️  {len(correcties)} tickers gecorrigeerd")
    if niet_gevonden:
        print(f"  ❌ {len(niet_gevonden)} tickers niet te corrigeren")

    tickers = gecorrigeerde_tickers

    # Delisted cache
    delisted_cache     = laad_delisted(getal)
    tickers_te_scannen = [t for t in tickers if t not in delisted_cache]
    if delisted_cache:
        print(f"  ⚡ {len(delisted_cache)} gedenoteerde tickers overgeslagen (cache)")
    print(f"  📊 {len(tickers_te_scannen)} tickers te scannen")

    # Master laden
    master     = laad_master(getal)
    eerste_run = len(master) == 0
    print(f"  {'Eerste run' if eerste_run else f'{len(master)} tickers gekend in master'}")

    # Batch OHLCV download
    ohlcv_cache = batch_download_ohlcv(tickers_te_scannen)

    # Detecteer nieuw gedenoteerde tickers
    nieuw_delisted = {t for t, df in ohlcv_cache.items() if df is None or len(df) == 0}
    delisted_cache.update(nieuw_delisted)
    sla_delisted_op(getal, delisted_cache)

    tickers_actief = [t for t in tickers_te_scannen if t not in nieuw_delisted]
    print(f"\n  ✅ {len(tickers_actief)} tickers met data | "
          f"❌ {len(nieuw_delisted)} nieuw gedenoteerd\n")

    # Munger + Volatiliteit
    print(f"{'='*60}")
    tellers = {
        "nieuw":       [],
        "actief":      [],
        "zwakker":     [],
        "verwijderd":  [],
        "munger_fail": [],
        "vol_fail":    [],
    }

    for ticker in tickers_actief:
        print(f"  {ticker:<16} ", end="", flush=True)

        munger_ok, munger_metrics, munger_reden = check_munger(ticker)
        if not munger_ok:
            print(f"❌ Munger: {munger_reden}")
            update_entry(master, ticker, False, munger_metrics)
            tellers["munger_fail"].append(ticker)
            time.sleep(SLEEP_INFO)
            continue

        print(
            f"✅ ROE:{munger_metrics['ROE']} "
            f"Debt:{munger_metrics['Debt']} "
            f"Marge:{munger_metrics['Marge']}  ",
            end="", flush=True,
        )
        time.sleep(SLEEP_INFO)

        vol_ok, vol, vol_reden = check_volatiliteit(ticker, ohlcv_cache)
        vol_str = f"{vol:.0%}"

        if not vol_ok:
            print(f"❌ Vol:{vol_str} {vol_reden}")
            update_entry(master, ticker, False, {**munger_metrics, "Vol": vol_str})
            tellers["vol_fail"].append(ticker)
            continue

        print(f"✅ Vol:{vol_str}")
        metrics = {**munger_metrics, "Vol": vol_str}
        status  = update_entry(master, ticker, True, metrics)
        tellers[status].append(ticker)

    # Opslaan
    sla_master_op(getal, master, beurs_naam)
    export_lijst = sla_export_op(getal, master)

    print(f"\n  ✅ {getal} ({beurs_naam}) klaar: "
          f"{len(export_lijst)} tickers → {bestand_export(getal)}")

    return {
        "getal":         getal,
        "naam":          beurs_naam,
        "tellers":       tellers,
        "master":        master,
        "export":        export_lijst,
        "correcties":    correcties,
        "niet_gevonden": niet_gevonden,
    }


# ---------------------------------------------------------------------------
# HOOFD SCAN
# ---------------------------------------------------------------------------
def scan_alle() -> None:
    start_tijd   = time.time()
    nu           = date.today().strftime("%d/%m/%Y")
    verwerkt     = []
    overgeslagen = []

    print(f"\n{'='*60}")
    print(f"  🔍 MRA FILTER BOT — ALLE LIJSTEN 041-060")
    print(f"  📅 Datum  : {nu}")
    print(f"  ⚙️  Criteria: ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | "
          f"Marge>{MARGE_MIN:.0%} | Vol {VOL_MIN:.0%}-{VOL_MAX:.0%}")
    print(f"{'='*60}")

    alle_resultaten = {}

    for nr in range(REEKS_START, REEKS_EINDE + 1):
        getal = f"0{nr}"
        bron  = bestand_bron(getal)

        if not os.path.exists(bron):
            print(f"\n  ⏭️  {bron} niet gevonden — overgeslagen")
            overgeslagen.append(getal)
            continue

        verwerkt.append(getal)
        resultaat = scan_lijst(getal)
        alle_resultaten[getal] = resultaat
        time.sleep(1)

    # Timing
    elapsed  = time.time() - start_tijd
    minuten  = int(elapsed // 60)
    seconden = int(elapsed % 60)

    # Console samenvatting
    print(f"\n{'='*60}")
    print(f"  ✅ ALLE SCANS VOLTOOID in {minuten}m {seconden}s")
    print(f"  📋 Verwerkt    : {len(verwerkt)} lijsten ({', '.join(verwerkt)})")
    print(f"  ⏭️  Overgeslagen: {len(overgeslagen)} lijsten")
    for getal, res in alle_resultaten.items():
        t = res["tellers"]
        print(f"     {getal} ({res['naam']}): "
              f"🆕{len(t['nieuw'])} ✅{len(t['actief'])} "
              f"⚠️{len(t['zwakker'])} ❌{len(t['verwijderd'])} "
              f"→ {len(res['export'])} export")
    print(f"{'='*60}\n")

    # Telegram rapport
    rapport  = f"📊 *MRA Filter — Alle lijsten*\n_{nu}_\n"
    rapport += f"⏱️ _Looptijd: {minuten}m {seconden}s_\n"
    rapport += f"⚙️ ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | Marge>{MARGE_MIN:.0%}\n"
    rapport += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    for getal, res in alle_resultaten.items():
        t      = res["tellers"]
        export = res["export"]
        naam   = res["naam"]
        rapport += f"*{getal} — {naam}* ({len(export)} tickers)\n"

        if res.get("correcties"):
            rapport += f"  ✏️ {len(res['correcties'])} suffix-correcties\n"
        if res.get("niet_gevonden"):
            rapport += f"  ⚠️ {len(res['niet_gevonden'])} niet te corrigeren\n"
        if t["nieuw"]:
            rapport += f"  🆕 {', '.join(f'`{x}`' for x in t['nieuw'][:10])}"
            if len(t["nieuw"]) > 10:
                rapport += f" +{len(t['nieuw'])-10}"
            rapport += "\n"
        if t["verwijderd"]:
            rapport += f"  ❌ {', '.join(f'`{x}`' for x in t['verwijderd'])}\n"
        if t["zwakker"]:
            rapport += f"  ⚠️ {', '.join(f'`{x}`' for x in t['zwakker'])}\n"
        if not t["nieuw"] and not t["verwijderd"] and not t["zwakker"]:
            rapport += f"  ✅ Geen wijzigingen\n"
        rapport += "\n"

    rapport += "_Volgende run: volgende zondag_"

    if len(rapport) <= 4096:
        send_telegram(rapport)
    else:
        send_telegram(rapport[:4000] + "\n_...zie master voor volledig overzicht_")
        time.sleep(1)
        send_telegram("_(vervolg)_\n" + rapport[4000:])


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scan_alle()
