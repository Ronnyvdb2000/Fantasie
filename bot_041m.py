"""
MRA Filter Bot
==============
Filtert een bronlijst van tickers op:
  1. Geldig Europees suffix + yfinance data beschikbaar
  2. Munger kwaliteitscriteria (ROE, Debt, Marge)
  3. MRA-geschikte volatiliteit (18% - 65% jaarlijks)

Gebruik:
  python mra_filter_bot.py --bron tickers_041a.txt --master tickers_041m.txt --export tickers_041x.txt

Bij eerste run: maakt master en export aan.
Bij volgende runs: werkt master bij en herschrijft export.
"""

import argparse
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — pas hier de criteria aan
# ---------------------------------------------------------------------------
GELDIGE_SUFFIXEN = [
    ".AS", ".PA", ".BR", ".DE", ".MC",
    ".L",  ".SW", ".MI", ".OL", ".ST",
    ".CO", ".HE", ".LS", ".IR",
]

ROE_MIN          = 0.15    # ROE > 15%
DEBT_MAX         = 60.0    # Debt/Equity < 60%
MARGE_MIN        = 0.10    # Winstmarge > 10%
VOL_MIN          = 0.18    # Jaarlijkse volatiliteit > 18%
VOL_MAX          = 0.65    # Jaarlijkse volatiliteit < 65%
MAX_WEKEN_BUITEN = 2       # Weken buiten filter voor verwijdering
SLEEP_SEC        = 0.8     # Pauze tussen API-calls

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram(bericht: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       bericht,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
    except Exception as e:
        print(f"⚠️  Telegram fout: {e}")


# ---------------------------------------------------------------------------
# MASTERLIJST — lezen en schrijven
# ---------------------------------------------------------------------------
def laad_master(master_bestand: str) -> dict:
    """
    Leest tickers_041m.txt in als dict.
    Formaat per regel:
      ASML.AS | opname:2025-01-15 | ROE:42% | Debt:12 | Marge:28% | Vol:31% | actief
      ING.AS  | opname:2025-02-10 | ROE:16% | Debt:58 | Marge:11% | Vol:38% | zwakker | weken_buiten:1
      ABN.AS  | opname:2025-01-20 | verwijderd:2025-04-28
    """
    master = {}
    if not os.path.exists(master_bestand):
        return master

    with open(master_bestand, "r", encoding="utf-8") as f:
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
                    # status zonder dubbele punt (actief/zwakker/verwijderd)
                    entry["status"] = deel.strip()

            # Zorg dat weken_buiten altijd een int is
            entry["weken_buiten"] = int(entry.get("weken_buiten", 0))
            master[ticker] = entry

    return master


def sla_master_op(master: dict, master_bestand: str, bron_bestand: str) -> None:
    """Schrijft de masterlijst terug naar tickers_041m.txt."""
    vandaag = date.today().strftime("%d/%m/%Y")
    regels  = [
        f"# MASTERLIJST — bron: {bron_bestand}",
        f"# Laatste update: {vandaag}",
        f"# Criteria: ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | "
        f"Marge>{MARGE_MIN:.0%} | Vol {VOL_MIN:.0%}-{VOL_MAX:.0%}",
        f"# Status: actief | zwakker (max {MAX_WEKEN_BUITEN} weken) | verwijderd",
        "# " + "-" * 70,
    ]

    # Sorteer: actief eerst, dan zwakker, dan verwijderd
    volgorde = {"actief": 0, "zwakker": 1, "nieuw": 0, "verwijderd": 2}
    gesorteerd = sorted(
        master.values(),
        key=lambda e: (volgorde.get(e.get("status", "verwijderd"), 3), e["ticker"]),
    )

    for entry in gesorteerd:
        t      = entry["ticker"]
        status = entry.get("status", "?")
        opname = entry.get("opname", "?")

        if status == "verwijderd":
            verw_datum = entry.get("verwijderd", date.today().isoformat())
            regels.append(f"{t:<12} | opname:{opname} | verwijderd:{verw_datum}")
        else:
            roe   = entry.get("ROE",   "?")
            debt  = entry.get("Debt",  "?")
            marge = entry.get("Marge", "?")
            vol   = entry.get("Vol",   "?")

            regel = (
                f"{t:<12} | opname:{opname} | "
                f"ROE:{roe} | Debt:{debt} | Marge:{marge} | Vol:{vol} | {status}"
            )
            if status == "zwakker":
                weken = entry.get("weken_buiten", 1)
                regel += f" | weken_buiten:{weken}"
            regels.append(regel)

    with open(master_bestand, "w", encoding="utf-8") as f:
        f.write("\n".join(regels) + "\n")


def sla_export_op(master: dict, export_bestand: str) -> list:
    """
    Schrijft tickers_041x.txt — alleen actieve en zwakkere tickers.
    Geeft de lijst terug.
    """
    export = sorted(
        t for t, e in master.items()
        if e.get("status") in ("actief", "nieuw", "zwakker")
    )
    with open(export_bestand, "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export


# ---------------------------------------------------------------------------
# LAAG 1 — SUFFIX + DATA CHECK
# ---------------------------------------------------------------------------
def check_suffix(ticker: str) -> tuple:
    if not any(ticker.endswith(s) for s in GELDIGE_SUFFIXEN):
        return False, "geen geldig suffix"
    return True, "OK"


def check_data(ticker: str) -> tuple:
    try:
        fi    = yf.Ticker(ticker).fast_info
        prijs = getattr(fi, "last_price", None)
        if prijs and prijs > 0:
            return True, f"€{prijs:.2f}"
        return False, "geen koers"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# LAAG 2 — MUNGER KWALITEITSFILTER
# ---------------------------------------------------------------------------
def check_munger(ticker: str) -> tuple:
    try:
        info = yf.Ticker(ticker).info
        if not info or "returnOnEquity" not in info:
            return False, {}, "geen fundamentele data"

        roe   = info.get("returnOnEquity", 0)   or 0
        debt  = info.get("debtToEquity",   9999) or 9999
        marge = info.get("profitMargins",  0)   or 0

        # Normaliseer debt (0.60 → 60 of 60 blijft 60)
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
def check_volatiliteit(ticker: str) -> tuple:
    try:
        data = yf.download(
            ticker, period="1y",
            progress=False, auto_adjust=True,
            multi_level_index=False,
        )
        if len(data) < 50:
            return False, 0.0, "te weinig data"

        vol = float(data["Close"].pct_change().dropna().std() * np.sqrt(252))

        if vol < VOL_MIN:
            return False, vol, f"te laag ({vol:.0%}<{VOL_MIN:.0%})"
        if vol > VOL_MAX:
            return False, vol, f"te hoog ({vol:.0%}>{VOL_MAX:.0%})"

        return True, vol, ""

    except Exception as e:
        return False, 0.0, str(e)


# ---------------------------------------------------------------------------
# MASTER BIJWERKEN — status per ticker
# ---------------------------------------------------------------------------
def update_entry(master: dict, ticker: str, door_filter: bool, metrics: dict) -> str:
    """
    Beheert de status van een ticker.
    Geeft de nieuwe status terug.
    """
    vandaag = date.today().isoformat()

    if ticker not in master:
        if door_filter:
            master[ticker] = {
                "ticker":        ticker,
                "status":        "nieuw",
                "opname":        vandaag,
                "weken_buiten":  0,
                **metrics,
            }
            return "nieuw"
        return "onbekend"  # niet door filter, nog niet in master → negeer

    entry = master[ticker]
    # Metrics altijd bijwerken (kunnen veranderd zijn)
    entry.update(metrics)

    if door_filter:
        entry["status"]       = "actief"
        entry["weken_buiten"] = 0
        return "actief"
    else:
        entry["weken_buiten"] = entry.get("weken_buiten", 0) + 1
        if entry["weken_buiten"] >= MAX_WEKEN_BUITEN:
            entry["status"]    = "verwijderd"
            entry["verwijderd"] = vandaag
            return "verwijderd"
        else:
            entry["status"] = "zwakker"
            return "zwakker"


# ---------------------------------------------------------------------------
# HOOFD SCAN
# ---------------------------------------------------------------------------
def scan(bron: str, master_bestand: str, export_bestand: str) -> None:
    print(f"\n{'='*60}")
    print(f"  🔍 MRA FILTER BOT")
    print(f"  📋 Bron   : {bron}")
    print(f"  📁 Master : {master_bestand}")
    print(f"  📤 Export : {export_bestand}")
    print(f"  📅 Datum  : {date.today().strftime('%d/%m/%Y')}")
    print(f"{'='*60}\n")

    # Bronbestand inladen
    if not os.path.exists(bron):
        msg = f"❌ Bronbestand {bron} niet gevonden."
        print(msg)
        send_telegram(msg)
        return

    with open(bron, "r", encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace("$", "")
    tickers = sorted(set(
        t.strip().upper() for t in inhoud.split(",") if t.strip()
    ))

    # Master inladen (leeg bij eerste run)
    master = laad_master(master_bestand)
    eerste_run = len(master) == 0

    print(f"  {len(tickers)} tickers te verwerken")
    print(f"  {'Eerste run — master wordt aangemaakt' if eerste_run else f'{len(master)} tickers al gekend in master'}\n")

    # Tellers voor rapport
    tellers = {"nieuw": [], "actief": [], "zwakker": [], "verwijderd": [], "afgewezen": []}

    for ticker in tickers:
        print(f"  {ticker:<12} ", end="", flush=True)

        # --- Laag 1a: Suffix ---
        suffix_ok, suffix_reden = check_suffix(ticker)
        if not suffix_ok:
            print(f"❌ {suffix_reden}")
            tellers["afgewezen"].append((ticker, suffix_reden))
            time.sleep(0.1)
            continue

        # --- Laag 1b: Data beschikbaar ---
        data_ok, prijs = check_data(ticker)
        if not data_ok:
            print(f"❌ Geen data: {prijs}")
            tellers["afgewezen"].append((ticker, f"geen data: {prijs}"))
            time.sleep(SLEEP_SEC)
            continue

        print(f"✅ {prijs:<10} ", end="", flush=True)
        time.sleep(SLEEP_SEC)

        # --- Laag 2: Munger ---
        munger_ok, munger_metrics, munger_reden = check_munger(ticker)
        if not munger_ok:
            print(f"❌ Munger: {munger_reden}")
            update_entry(master, ticker, False, munger_metrics)
            time.sleep(SLEEP_SEC)
            continue

        print(
            f"✅ ROE:{munger_metrics['ROE']} Debt:{munger_metrics['Debt']} "
            f"Marge:{munger_metrics['Marge']}  ",
            end="", flush=True,
        )
        time.sleep(SLEEP_SEC)

        # --- Laag 3: Volatiliteit ---
        vol_ok, vol, vol_reden = check_volatiliteit(ticker)
        vol_str = f"{vol:.0%}"

        if not vol_ok:
            print(f"❌ Vol:{vol_str} {vol_reden}")
            update_entry(master, ticker, False, {**munger_metrics, "Vol": vol_str})
            time.sleep(SLEEP_SEC)
            continue

        print(f"✅ Vol:{vol_str}")

        # --- Alle lagen OK ---
        metrics = {**munger_metrics, "Vol": vol_str}
        status  = update_entry(master, ticker, True, metrics)
        tellers[status].append(ticker)
        time.sleep(SLEEP_SEC)

    # --- Opslaan ---
    sla_master_op(master, master_bestand, bron)
    export_lijst = sla_export_op(master, export_bestand)

    # --- Samenvatting console ---
    print(f"\n{'='*60}")
    print(f"  ✅ SCAN VOLTOOID")
    print(f"  🆕 Nieuw     : {len(tellers['nieuw'])}")
    print(f"  ✅ Actief    : {len(tellers['actief'])}")
    print(f"  ⚠️  Zwakker   : {len(tellers['zwakker'])}")
    print(f"  ❌ Verwijderd: {len(tellers['verwijderd'])}")
    print(f"  🚫 Afgewezen : {len(tellers['afgewezen'])}")
    print(f"  📤 Export    : {len(export_lijst)} tickers → {export_bestand}")
    print(f"{'='*60}\n")

    # --- Telegram rapport ---
    nu      = date.today().strftime("%d/%m/%Y")
    rapport = f"📊 *MRA Filter — {bron}*\n_{nu}_\n"
    rapport += "━━━━━━━━━━━━━━━━━━━━━━━━\n"

    if tellers["nieuw"]:
        rapport += f"\n🆕 *Nieuw toegevoegd ({len(tellers['nieuw'])}):*\n"
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
        rapport += f"\n⚠️ *Verzwakt — nog in lijst ({len(tellers['zwakker'])}):*\n"
        for t in tellers["zwakker"]:
            weken = master[t].get("weken_buiten", 1)
            rapport += f"• `{t}` ({weken}/{MAX_WEKEN_BUITEN} weken buiten filter)\n"

    if not tellers["nieuw"] and not tellers["verwijderd"] and not tellers["zwakker"]:
        rapport += "\n✅ Geen wijzigingen deze week\n"

    actief_totaal = [
        t for t, e in master.items()
        if e.get("status") in ("actief", "nieuw", "zwakker")
    ]
    rapport += f"\n📤 *Export: {len(actief_totaal)} tickers*\n"
    rapport += f"`{', '.join(sorted(actief_totaal))}`\n"
    rapport += "\n_Volgende run: volgende maandag_"

    send_telegram(rapport)


# ---------------------------------------------------------------------------
# COMMAND LINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRA Filter Bot")
    parser.add_argument(
        "--bron",   required=True,
        help="Bronbestand met alle tickers (bv. tickers_041a.txt)",
    )
    parser.add_argument(
        "--master", required=True,
        help="Masterbestand met geschiedenis (bv. tickers_041m.txt)",
    )
    parser.add_argument(
        "--export", required=True,
        help="Exportbestand voor MRA-bot (bv. tickers_041x.txt)",
    )
    args = parser.parse_args()

    scan(
        bron=args.bron,
        master_bestand=args.master,
        export_bestand=args.export,
    )
