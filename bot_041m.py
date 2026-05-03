"""
MRA Filter Bot — bot_041m.py
=============================
Filtert tickers_041a.txt op:
  1. Geldig Europees suffix + yfinance data beschikbaar
  2. Munger kwaliteitscriteria (ROE, Debt, Marge)
  3. MRA-geschikte volatiliteit (18% - 65% jaarlijks)

Output:
  tickers_041m.txt  — masterlijst met datum + parameters
  tickers_041x.txt  — schone tickerlijst voor MRA-bot

Eerste run: maakt master en export aan.
Volgende runs: werkt master bij en herschrijft export.
"""

import os
import time
import requests
import numpy as np
import yfinance as yf
from datetime import date

# ---------------------------------------------------------------------------
# BESTANDEN — pas hier aan indien nodig
# ---------------------------------------------------------------------------
BRON_BESTAND   = "tickers_041a.txt"
MASTER_BESTAND = "tickers_041m.txt"
EXPORT_BESTAND = "tickers_041x.txt"

# ---------------------------------------------------------------------------
# TELEGRAM — via omgevingsvariabelen (GitHub Secrets of .env export)
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# CONFIG — criteria
# ---------------------------------------------------------------------------
GELDIGE_SUFFIXEN = [
    ".AS", ".PA", ".BR", ".DE", ".MC",
    ".L",  ".SW", ".MI", ".OL", ".ST",
    ".CO", ".HE", ".LS", ".IR",
]

ROE_MIN          = 0.15   # ROE > 15%
DEBT_MAX         = 60.0   # Debt/Equity < 60%
MARGE_MIN        = 0.10   # Winstmarge > 10%
VOL_MIN          = 0.18   # Jaarlijkse volatiliteit > 18%
VOL_MAX          = 0.65   # Jaarlijkse volatiliteit < 65%
MAX_WEKEN_BUITEN = 2      # Weken buiten filter voor verwijdering
SLEEP_SEC        = 0.8    # Pauze tussen API-calls


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram(bericht: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram niet geconfigureerd — bericht overgeslagen.")
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
# MASTERLIJST — lezen
# ---------------------------------------------------------------------------
def laad_master(master_bestand: str) -> dict:
    """
    Leest de masterlijst in als dict.
    Formaat per regel:
      ASML.AS      | opname:2025-01-15 | ROE:42% | Debt:12.0 | Marge:28% | Vol:31% | actief
      ING.AS       | opname:2025-02-10 | ROE:16% | Debt:58.0 | Marge:11% | Vol:38% | zwakker | weken_buiten:1
      ABN.AS       | opname:2025-01-20 | verwijderd:2025-04-28
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
                    entry["status"] = deel.strip()

            entry["weken_buiten"] = int(entry.get("weken_buiten", 0))
            master[ticker] = entry

    return master


# ---------------------------------------------------------------------------
# MASTERLIJST — schrijven
# ---------------------------------------------------------------------------
def sla_master_op(master: dict) -> None:
    """Schrijft de masterlijst terug naar MASTER_BESTAND."""
    vandaag = date.today().strftime("%d/%m/%Y")
    regels  = [
        f"# MASTERLIJST — bron: {BRON_BESTAND}",
        f"# Laatste update: {vandaag}",
        f"# Criteria: ROE>{ROE_MIN:.0%} | Debt<{DEBT_MAX:.0f} | "
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
    """Schrijft EXPORT_BESTAND — alleen actieve en zwakkere tickers."""
    export = sorted(
        t for t, e in master.items()
        if e.get("status") in ("nieuw", "actief", "zwakker")
    )
    with open(EXPORT_BESTAND, "w", encoding="utf-8") as f:
        f.write(", ".join(export))
    return export


# ---------------------------------------------------------------------------
# LAAG 1A — SUFFIX CHECK
# ---------------------------------------------------------------------------
def check_suffix(ticker: str) -> tuple:
    if not any(ticker.endswith(s) for s in GELDIGE_SUFFIXEN):
        return False, "geen geldig suffix"
    return True, "OK"


# ---------------------------------------------------------------------------
# LAAG 1B — DATA CHECK
# ---------------------------------------------------------------------------
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

        roe   = info.get("returnOnEquity", 0)    or 0
        debt  = info.get("debtToEquity",   9999) or 9999
        marge = info.get("profitMargins",  0)    or 0

        # Normaliseer debt: sommige APIs geven 0.60, andere 60
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
    print(f"\n{'='*60}")
    print(f"  🔍 MRA FILTER BOT")
    print(f"  📋 Bron   : {BRON_BESTAND}")
    print(f"  📁 Master : {MASTER_BESTAND}")
    print(f"  📤 Export : {EXPORT_BESTAND}")
    print(f"  📅 Datum  : {date.today().strftime('%d/%m/%Y')}")
    print(f"{'='*60}\n")

    # --- Bronbestand laden ---
    if not os.path.exists(BRON_BESTAND):
        msg = f"❌ Bronbestand {BRON_BESTAND} niet gevonden."
        print(msg)
        send_telegram(msg)
        return

    with open(BRON_BESTAND, "r", encoding="utf-8") as f:
        inhoud = f.read().replace("\n", ",").replace("$", "")
    tickers = sorted(set(
        t.strip().upper() for t in inhoud.split(",") if t.strip()
    ))

    # --- Master laden ---
    master     = laad_master(MASTER_BESTAND)
    eerste_run = len(master) == 0

    print(f"  {len(tickers)} tickers te verwerken")
    if eerste_run:
        print("  Eerste run — master wordt aangemaakt\n")
    else:
        print(f"  {len(master)} tickers gekend in master\n")

    tellers = {
        "nieuw":      [],
        "actief":     [],
        "zwakker":    [],
        "verwijderd": [],
        "afgewezen":  [],
    }

    for ticker in tickers:
        print(f"  {ticker:<14} ", end="", flush=True)

        # Laag 1a: Suffix
        suffix_ok, suffix_reden = check_suffix(ticker)
        if not suffix_ok:
            print(f"❌ {suffix_reden}")
            tellers["afgewezen"].append((ticker, suffix_reden))
            time.sleep(0.1)
            continue

        # Laag 1b: Data beschikbaar
        data_ok, prijs = check_data(ticker)
        if not data_ok:
            print(f"❌ Geen data: {prijs}")
            tellers["afgewezen"].append((ticker, f"geen data"))
            time.sleep(SLEEP_SEC)
            continue

        print(f"✅ {prijs:<10} ", end="", flush=True)
        time.sleep(SLEEP_SEC)

        # Laag 2: Munger
        munger_ok, munger_metrics, munger_reden = check_munger(ticker)
        if not munger_ok:
            print(f"❌ Munger: {munger_reden}")
            update_entry(master, ticker, False, munger_metrics)
            time.sleep(SLEEP_SEC)
            continue

        print(
            f"✅ ROE:{munger_metrics['ROE']} "
            f"Debt:{munger_metrics['Debt']} "
            f"Marge:{munger_metrics['Marge']}  ",
            end="", flush=True,
        )
        time.sleep(SLEEP_SEC)

        # Laag 3: Volatiliteit
        vol_ok, vol, vol_reden = check_volatiliteit(ticker)
        vol_str = f"{vol:.0%}"

        if not vol_ok:
            print(f"❌ Vol:{vol_str} {vol_reden}")
            update_entry(master, ticker, False, {**munger_metrics, "Vol": vol_str})
            time.sleep(SLEEP_SEC)
            continue

        print(f"✅ Vol:{vol_str}")

        # Alle lagen OK
        metrics = {**munger_metrics, "Vol": vol_str}
        status  = update_entry(master, ticker, True, metrics)
        tellers[status].append(ticker)
        time.sleep(SLEEP_SEC)

    # --- Opslaan ---
    sla_master_op(master)
    export_lijst = sla_export_op(master)

    # --- Console samenvatting ---
    print(f"\n{'='*60}")
    print(f"  ✅ SCAN VOLTOOID")
    print(f"  🆕 Nieuw     : {len(tellers['nieuw'])}")
    print(f"  ✅ Actief    : {len(tellers['actief'])}")
    print(f"  ⚠️  Zwakker   : {len(tellers['zwakker'])}")
    print(f"  ❌ Verwijderd: {len(tellers['verwijderd'])}")
    print(f"  🚫 Afgewezen : {len(tellers['afgewezen'])}")
    print(f"  📤 Export    : {len(export_lijst)} tickers → {EXPORT_BESTAND}")
    print(f"{'='*60}\n")

    # --- Telegram rapport ---
    nu      = date.today().strftime("%d/%m/%Y")
    rapport = f"📊 *MRA Filter — {BRON_BESTAND}*\n_{nu}_\n"
    rapport += "━━━━━━━━━━━━━━━━━━━━━━━━\n"

    if tellers["nieuw"]:
        rapport += f"\n🆕 *Nieuw ({len(tellers['nieuw'])}):*\n"
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
            rapport += f"• `{t}` ({weken}/{MAX_WEKEN_BUITEN} weken)\n"

    if not tellers["nieuw"] and not tellers["verwijderd"] and not tellers["zwakker"]:
        rapport += "\n✅ Geen wijzigingen deze week\n"

    actief_totaal = sorted(
        t for t, e in master.items()
        if e.get("status") in ("nieuw", "actief", "zwakker")
    )
    rapport += f"\n📤 *Export: {len(actief_totaal)} tickers*\n"
    rapport += f"`{', '.join(actief_totaal)}`\n"
    rapport += "\n_Volgende run: volgende maandag_"

    # Splits indien te lang voor Telegram (max 4096 tekens)
    if len(rapport) <= 4096:
        send_telegram(rapport)
    else:
        send_telegram(rapport[:4000] + "\n_...zie master voor volledig overzicht_")


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scan()
