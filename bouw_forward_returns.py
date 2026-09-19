#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_forward_returns.py  —  FEATURE STORE: selecties -> forward-rendement  v1.0

DOEL
====
De `selecties`-tabel bevat al, per dag/aandeel/strategie, een volledige set
indicatorwaarden (RSI, ATR%, VWAP-afstand, kwaliteitsscores, overlap-
tellingen, ...). Wat ontbreekt is het LABEL: wat deed dat aandeel nadien
werkelijk? Dit script vult die kloof op een incrementele, idempotente
manier -- geen jaren wachten per losse strategie, maar de duizenden reeds
gelogde rijen in `selecties` retroactief van een label voorzien zodra er
genoeg tijd verstreken is.

Voor elke (ticker, datum, strategie)-combinatie in `selecties` waarvoor nog
geen (volledig) label bestaat in `forward_returns`, wordt via yfinance het
koersverloop na `datum` opgehaald en worden 3 horizons berekend:
  - fwd_ret_5d, fwd_ret_10d, fwd_ret_20d  (handelsdagen, niet kalenderdagen)
t.o.v. de reeds gelogde `koers` (entry-koers op selectiemoment) -- niet
t.o.v. een nieuw opgehaalde slotkoers op diezelfde dag, om consistent te
blijven met wat de bots zelf als instapkoers rapporteerden.

INCREMENTEEL EN IDEMPOTENT
===========================
- Een horizon wordt pas ingevuld zodra er ECHT genoeg toekomstige
  handelsdagen bestaan (geen NaN/placeholder-vulling voor wat nog niet
  gebeurd is). Een rij kan dus na een eerste run enkel fwd_ret_5d hebben,
  en een latere run vult fwd_ret_10d/20d aan -- vandaar de upsert
  (INSERT ... ON CONFLICT ... DO UPDATE) i.p.v. gewone INSERT.
- Tickers worden gegroepeerd: één yfinance-download per ticker dekt ALLE
  openstaande (datum, strategie)-rijen van die ticker in dit run, i.p.v.
  een aparte call per rij.
- Enkel tickers met effectief nog onvolledige forward_returns-rijen (of nog
  helemaal geen rij) worden opnieuw bevraagd -- reeds volledig ingevulde
  combinaties worden overgeslagen.

GEBRUIK
=======
  python bouw_forward_returns.py build

Env vars:
  SUPABASE_DB_URL     - Postgres connectiestring
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  - optioneel, stuurt een korte
                        samenvatting (hoeveel rijen bijgewerkt/nog
                        wachtend); als afwezig wordt enkel naar stdout
                        geprint
  MAX_TICKERS_PER_RUN - veiligheidslimiet tegen te lange/rate-limited
                        runs (default 400)
  HORIZONS            - komma-gescheiden lijst handelsdagen (default
                        "5,10,20")
"""

import os
import sys
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import yfinance as yf
import pandas as pd
import requests

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_TICKERS_PER_RUN = int(os.environ.get("MAX_TICKERS_PER_RUN", "400"))
# LET OP: als je dit wijzigt, moet forward_returns's schema mee-veranderen
# (migratie_forward_returns.sql heeft vaste kolommen fwd_close_5d/10d/20d en
# fwd_ret_5d/10d/20d) -- de kolomnamen hier worden dynamisch opgebouwd, maar
# de tabel zelf niet automatisch.
HORIZONS = sorted(int(h) for h in os.environ.get("HORIZONS", "5,10,20").split(","))


def vandaag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")


# --------------------------------------------------------------------------
# Stap 1: welke (ticker, datum, strategie)-rijen hebben nog een onvolledig
# of ontbrekend label, en zijn oud genoeg om de langste horizon te vullen?
# --------------------------------------------------------------------------
def haal_openstaande_rijen(conn) -> List[dict]:
    max_horizon = HORIZONS[-1]
    kolom_max = f"fwd_ret_{max_horizon}d"
    # ~1.5x buffer voor weekends/feestdagen om max_horizon handelsdagen te dekken
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(max_horizon * 1.6) + 3)).strftime("%Y-%m-%d")

    query = f"""
        SELECT s.ticker, s.datum, s.strategie, s.beurs, s.koers
        FROM selecties s
        LEFT JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        WHERE s.datum <= %s
          AND s.koers IS NOT NULL
          AND (f.ticker IS NULL OR f.{kolom_max} IS NULL)
        ORDER BY s.ticker, s.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (cutoff,))
        return cur.fetchall()


# --------------------------------------------------------------------------
# Stap 2: per ticker het koersverloop ophalen en de horizons berekenen
# --------------------------------------------------------------------------
def bereken_labels_voor_ticker(ticker: str, rijen: List[dict]) -> List[dict]:
    """rijen = alle openstaande (datum, strategie, beurs, koers)-combinaties
    voor deze ene ticker. Geeft een lijst dicts terug, klaar voor upsert."""
    vroegste = min(r["datum"] for r in rijen)
    try:
        hist = yf.Ticker(ticker).history(start=vroegste, auto_adjust=True)
    except Exception as e:
        print(f"  [WARN] {ticker}: download mislukt ({e})")
        return []

    if hist is None or hist.empty or "Close" not in hist.columns:
        print(f"  [WARN] {ticker}: geen koersdata")
        return []

    closes = hist["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    datums = closes.index

    resultaten = []
    for r in rijen:
        try:
            doel = pd.Timestamp(r["datum"])
            pos_kandidaten = datums.searchsorted(doel)
            if pos_kandidaten >= len(datums):
                continue  # datum ligt na alle beschikbare koersdata, niets te doen
            entry_koers = float(r["koers"])
            if not entry_koers or math.isnan(entry_koers) or entry_koers <= 0:
                continue

            rij = {
                "ticker": ticker, "datum": r["datum"], "strategie": r["strategie"],
                "beurs": r["beurs"], "entry_koers": entry_koers,
            }
            for h in HORIZONS:
                idx = pos_kandidaten + h
                if idx < len(closes):
                    fwd_close = float(closes.iloc[idx])
                    rij[f"fwd_close_{h}d"] = fwd_close
                    rij[f"fwd_ret_{h}d"] = round((fwd_close / entry_koers - 1) * 100, 3)
                else:
                    rij[f"fwd_close_{h}d"] = None
                    rij[f"fwd_ret_{h}d"] = None
            resultaten.append(rij)
        except Exception as e:
            print(f"  [WARN] {ticker} {r['datum']}/{r['strategie']}: {e}")
            continue

    return resultaten


# --------------------------------------------------------------------------
# Stap 3: upsert naar forward_returns
# --------------------------------------------------------------------------
def upsert_labels(conn, rijen: List[dict]) -> int:
    if not rijen:
        return 0
    kolommen = ["ticker", "datum", "strategie", "beurs", "entry_koers"]
    for h in HORIZONS:
        kolommen += [f"fwd_close_{h}d", f"fwd_ret_{h}d"]

    kolom_lijst = ", ".join(kolommen)
    placeholders = ", ".join(f"%({k})s" for k in kolommen)
    update_lijst = ", ".join(f"{k} = EXCLUDED.{k}" for k in kolommen if k not in ("ticker", "datum", "strategie"))
    update_lijst += ", bijgewerkt_op = now()"

    query = f"""
        INSERT INTO forward_returns ({kolom_lijst})
        VALUES ({placeholders})
        ON CONFLICT (ticker, datum, strategie)
        DO UPDATE SET {update_lijst};
    """
    with conn.cursor() as cur:
        for rij in rijen:
            volledige_rij = {k: rij.get(k) for k in kolommen}
            cur.execute(query, volledige_rij)
    conn.commit()
    return len(rijen)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_build():
    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        open_rijen = haal_openstaande_rijen(conn)
        print(f"{len(open_rijen)} (ticker, datum, strategie)-rijen wachten op een (bijgewerkt) label.")

        per_ticker: Dict[str, List[dict]] = {}
        for r in open_rijen:
            per_ticker.setdefault(r["ticker"], []).append(r)

        tickers = sorted(per_ticker.keys())[:MAX_TICKERS_PER_RUN]
        overgeslagen = len(per_ticker) - len(tickers)
        print(f"{len(tickers)} unieke tickers te verwerken dit run"
              + (f" ({overgeslagen} tickers volgen in een volgend run, MAX_TICKERS_PER_RUN bereikt)" if overgeslagen > 0 else ""))

        totaal_bijgewerkt = 0
        totaal_compleet = 0
        for i, ticker in enumerate(tickers, start=1):
            labels = bereken_labels_voor_ticker(ticker, per_ticker[ticker])
            aantal = upsert_labels(conn, labels)
            totaal_bijgewerkt += aantal
            totaal_compleet += sum(1 for l in labels if l.get(f"fwd_ret_{HORIZONS[-1]}d") is not None)
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} tickers verwerkt...")
            time.sleep(0.1)

        nog_wachtend = len(open_rijen) - totaal_bijgewerkt

        bericht = (
            f"📊 *Forward-Returns Feature Store — {vandaag()}*\n\n"
            f"{totaal_bijgewerkt} rijen bijgewerkt in forward_returns "
            f"({totaal_compleet} daarvan nu volledig, alle {HORIZONS[-1]} handelsdagen ingevuld)\n"
            f"{len(tickers)} unieke tickers verwerkt dit run"
            + (f" ({overgeslagen} tickers nog te gaan)" if overgeslagen > 0 else "")
        )
        send_telegram(bericht)
        print(f"\nKlaar. {totaal_bijgewerkt} rijen bijgewerkt.")
    finally:
        conn.close()


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "build"
    run_build()
