"""
MRA Filter Bot — bot_041m.py
=============================
Filtert tickers_041a.txt op:
  1. Geldig Europees suffix + yfinance data beschikbaar
  2. Munger kwaliteitscriteria (ROE, Debt, Marge) — soepel niveau
  3. MRA-geschikte volatiliteit (18% - 65% jaarlijks)

Output:
  tickers_041m.txt  — masterlijst met datum + parameters
  tickers_041x.txt  — schone tickerlijst voor MRA-bot

Snelheidsoptimalisaties:
  - Batch download van alle OHLCV-data in één yfinance-aanroep
  - Minder sleep tussen calls
  - Gedenoteerde tickers worden gecached en overgeslagen

Scheduling: elke zondag (minst druk op Yahoo Finance API)
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
# LOGGING — onderdruk lelijke yfinance meldingen
# ---------------------------------------------------------------------------
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# BESTANDEN
# ---------------------------------------------------------------------------
BRON_BESTAND     = "tickers_041a.txt"
MASTER_BESTAND   = "tickers_041m.txt"
EXPORT_BESTAND   = "tickers_041x.txt"
DELISTED_BESTAND = "tickers_041d.txt"   # gecachede gedenoteerde tickers

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# CONFIG — soepel niveau
# ---------------------------------------------------------------------------
GELDIGE_SUFFIXEN = [
    ".AS", ".PA", ".BR", ".DE", ".MC",
    ".L",  ".SW", ".MI", ".OL", ".ST",
    ".CO", ".HE", ".LS", ".IR",
]

ROE_MIN          = 0.10   # ROE > 10%
DEBT_MAX         = 100.0  # Debt/Equity < 100
MARGE_MIN        = 0.07   # Winstmarge > 7%
VOL_MIN          = 0.18   # Jaarlijkse volatiliteit > 18%
VOL_MAX          = 0.65   # Jaarlijkse volatiliteit < 65%
MAX_WEKEN_BUITEN = 2      # Weken buiten filter voor verwijdering

BATCH_SIZE       = 50     # tickers per batch OHLCV download
SLEEP_BATCH      = 2.0    # seconden tussen batches
SLEEP_INFO       = 0.3    # seconden tussen Munger info-calls


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
# DELISTED CACHE
# ---------------------------------------------------------------------------
def laad_delisted() -> set:
    if not os.path.exists(DELISTED_BESTAND):
        return set()
    with open(DELISTED_BESTAND, "r", encoding="utf-8") as f:
        return set(t.strip().upper() for t in f.read().split(",") if t.strip())


def sla_delisted_op(delisted: set) -> None:
    with open(DELISTED_BESTAND, "w", encoding="utf-8") as f:
        f.write(", ".join(sorted(delisted)))


# ---------------------------------------------------------------------------
# BATCH OHLCV DOWNLOAD
# ---------------------------------------------------------------------------
def batch_download_ohlcv(tickers: list) -> dict:
    """
    Download 1 jaar OHLCV voor alle tickers in batches van BATCH_SIZE.
    Veel sneller dan ticker-per-ticker downloaden.
    Geeft dict terug: {ticker: DataFrame of None}
    """
    resultaat = {}
    batches   = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

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
def laad_master() -> dict:
    master = {}
    if not os.path.exists(MASTER_BESTAND):
        return master
    with open(MASTER_BESTAND, "r", encoding="utf-8") as f:
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
def sla_master_op(master: dict) -> None:
    vandaag = date.today().strftime("%d/%m/%Y")
    regels  = [
        f"# MASTERLIJST — bron: {BRON_BESTAND}",
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
            regels.append(f"{t:<14} | opname:{opname} | verwijderd:{verw}")
        else:
            roe   = entry.get("ROE",   "?")
            debt  = entry.get("Debt",  "?")
            marge = entry.get("Marge", "?")
            vol   = entry.get("Vol",   "?")
            regel = (
                f"{t:<14} | opname:{opname} | "
                f"ROE:{roe} | Debt:{debt} | Marge:{marge} | Vol:{vol} | {status}"
            )
            if status == "zwakker":
                regel += f" | weken_buiten:{entry.get('weken_buiten', 1)}"
            regels.append(regel)

    with open(MASTER_BESTAND, "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")


# ---------------------------------------------------------------------------
# EXPORT — schrijven
# ---------------------------------------------------------------------------
def sla_export_op(master: dict) -> list:
    export = sorted(
        t for t, e in master.items()
        if e.get("status") in ("nieuw", "actief", "zwakker")
    )
    with open(EXPORT_BESTAND, "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export


# ---------------------------------------------------------------------------
# LAAG 1 — SUFFIX CHECK
# ---------------------------------------------------------------------------
def check_suffix(ticker: str) -> bool:
    return any(ticker.endswith(s) for s in GELDIGE_SUFFIXEN)


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
# LAAG 3 — VOLATILITEITSFILTER (uit batch cache)
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
# HOOFD SCAN
# ---------------------------------------------------------------------------
def scan() -> None:
    start_tijd = time.time()

    print(f"\n{'='*60}")
    print(f"  🔍 MRA FILTER BOT — soepel niveau")
    print(f"  📋 Bron   : {BRON_BESTAND}")
    print(f"  📁 Master : {MASTER_BESTAND}")
    print(f"  📤 Export : {EXPORT_BESTAND}")
    print(f"  📅 Datum  : {date.today().strftime('%d/%m/%Y')}")
    print(f"  ⚙️  Criteria: ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | "
          f"Marge>{MARGE_MIN:.0%} | Vol {VOL_MIN:.0%}-{VOL_MAX:.0%}")
    print(f"{'='*60}")

    # --- Bronbestand laden ---
    if not os.path.exists(BRON_BESTAND):
        msg = f"❌ Bronbestand {BRON_BESTAND} niet gevonden."
        print(msg)
        send_telegram(msg)
        return

    with open(BRON_BESTAND, "r", encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace("$", "")
    alle_tickers = sorted(set(
        t.strip().upper() for t in inhoud.split(",") if t.strip()
    ))

    # --- Suffix filter ---
    tickers        = [t for t in alle_tickers if check_suffix(t)]
    afgew_suffix   = [t for t in alle_tickers if not check_suffix(t)]
    if afgew_suffix:
        print(f"\n  🚫 {len(afgew_suffix)} tickers zonder geldig suffix overgeslagen")

    # --- Delisted cache ---
    delisted_cache       = laad_delisted()
    tickers_te_scannen   = [t for t in tickers if t not in delisted_cache]
    if delisted_cache:
        print(f"  ⚡ {len(delisted_cache)} gedenoteerde tickers overgeslagen (cache)")
    print(f"  📊 {len(tickers_te_scannen)} tickers te scannen")

    # --- Master laden ---
    master     = laad_master()
    eerste_run = len(master) == 0
    print(f"  {'Eerste run — master wordt aangemaakt' if eerste_run else f'{len(master)} tickers gekend in master'}")

    # --- STAP 1: Batch OHLCV download ---
    ohlcv_cache = batch_download_ohlcv(tickers_te_scannen)

    # Detecteer nieuw gedenoteerde tickers
    nieuw_delisted = {t for t, df in ohlcv_cache.items() if df is None or len(df) == 0}
    delisted_cache.update(nieuw_delisted)
    sla_delisted_op(delisted_cache)

    tickers_actief = [t for t in tickers_te_scannen if t not in nieuw_delisted]
    print(f"\n  ✅ {len(tickers_actief)} tickers met data | "
          f"❌ {len(nieuw_delisted)} nieuw gedenoteerd\n")

    # --- STAP 2: Munger + Volatiliteit per ticker ---
    print(f"{'='*60}")
    print(f"  📊 ANALYSE")
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
        print(f"  {ticker:<14} ", end="", flush=True)

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

    # --- Opslaan ---
    sla_master_op(master)
    export_lijst = sla_export_op(master)

    # --- Timing ---
    elapsed  = time.time() - start_tijd
    minuten  = int(elapsed // 60)
    seconden = int(elapsed % 60)

    # --- Console samenvatting ---
    print(f"\n{'='*60}")
    print(f"  ✅ SCAN VOLTOOID in {minuten}m {seconden}s")
    print(f"  🆕 Nieuw       : {len(tellers['nieuw'])}")
    print(f"  ✅ Actief      : {len(tellers['actief'])}")
    print(f"  ⚠️  Zwakker     : {len(tellers['zwakker'])}")
    print(f"  ❌ Verwijderd  : {len(tellers['verwijderd'])}")
    print(f"  🚫 Munger fail : {len(tellers['munger_fail'])}")
    print(f"  📉 Vol fail    : {len(tellers['vol_fail'])}")
    print(f"  📤 Export      : {len(export_lijst)} tickers → {EXPORT_BESTAND}")
    print(f"{'='*60}\n")

    # --- Telegram rapport ---
    nu      = date.today().strftime("%d/%m/%Y")
    rapport = f"📊 *MRA Filter — {BRON_BESTAND}*\n_{nu}_\n"
    rapport += f"⏱️ _Looptijd: {minuten}m {seconden}s_\n"
    rapport += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    rapport += f"⚙️ ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | Marge>{MARGE_MIN:.0%}\n\n"

    if tellers["nieuw"]:
        rapport += f"🆕 *Nieuw ({len(tellers['nieuw'])}):*\n"
        for t in tellers["nieuw"]:
            e = master[t]
            rapport += (
                f"• `{t}` — ROE:{e.get('ROE','?')} | "
                f"Debt:{e.get('Debt','?')} | "
                f"Marge:{e.get('Marge','?')} | "
                f"Vol:{e.get('Vol','?')}\n"
            )

    if tellers["verwijderd"]:
        rapport += f"\n❌ *Verwijderd ({len(tellers['verwijderd'])}):*\n"
        for t in tellers["verwijderd"]:
            rapport += f"• `{t}`\n"

    if tellers["zwakker"]:
        rapport += f"\n⚠️ *Verzwakt ({len(tellers['zwakker'])}):*\n"
        for t in tellers["zwakker"]:
            weken = master[t].get("weken_buiten", 1)
            rapport += f"• `{t}` ({weken}/{MAX_WEKEN_BUITEN} weken)\n"

    if not tellers["nieuw"] and not tellers["verwijderd"] and not tellers["zwakker"]:
        rapport += "✅ Geen wijzigingen deze week\n"

    actief_totaal = sorted(
        t for t, e in master.items()
        if e.get("status") in ("nieuw", "actief", "zwakker")
    )
    rapport += f"\n📤 *Export: {len(actief_totaal)} tickers*\n"
    if actief_totaal:
        rapport += f"`{', '.join(actief_totaal)}`\n"
    rapport += "\n_Volgende run: volgende zondag_"

    if len(rapport) <= 4096:
        send_telegram(rapport)
    else:
        send_telegram(rapport[:4000] + "\n_...zie master voor volledig overzicht_")


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scan()
