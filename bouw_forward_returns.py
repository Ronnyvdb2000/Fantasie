#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_forward_returns.py  —  FEATURE STORE: selecties -> forward-rendement  v1.2

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
koersverloop na `datum` opgehaald en worden de horizons uit HORIZONS
berekend (bv. 5/10/20/30/60 handelsdagen) t.o.v. de reeds gelogde `koers`
(entry-koers op selectiemoment) -- niet t.o.v. een nieuw opgehaalde
slotkoers op diezelfde dag, om consistent te blijven met wat de bots zelf
als instapkoers rapporteerden.

INCREMENTEEL EN IDEMPOTENT
===========================
- Een horizon wordt pas ingevuld zodra er ECHT genoeg toekomstige
  handelsdagen bestaan (geen NaN/placeholder-vulling voor wat nog niet
  gebeurd is). Een rij kan dus na een eerste run enkel fwd_ret_5d hebben,
  en een latere run vult fwd_ret_10d/20d/30d/60d aan -- vandaar de upsert
  (INSERT ... ON CONFLICT ... DO UPDATE) i.p.v. gewone INSERT.
- Tickers worden gegroepeerd: één yfinance-download per ticker dekt ALLE
  openstaande (datum, strategie)-rijen van die ticker in dit run, i.p.v.
  een aparte call per rij.
- Enkel tickers met effectief nog onvolledige forward_returns-rijen (of nog
  helemaal geen rij) worden opnieuw bevraagd -- reeds volledig ingevulde
  combinaties worden overgeslagen.
- PER-HORIZON cutoff (v1.1, 2026-09-26): een rij wordt al opgepikt zodra
  ÉÉN horizon oud genoeg is EN nog een ontbrekend label heeft voor die
  horizon -- niet pas wanneer de LANGSTE horizon (bv. 60d) matuur is. Zo
  hoeft fwd_ret_30d niet nodeloos ~48 dagen langer te wachten dan nodig,
  enkel omdat er ook een 60d-horizon in de lijst staat.
- ONBRUIKBARE KOERS = ONTBREKEND (v1.2): een forward-koers die NaN/inf is of
  <= 0 wordt als None (NULL) weggeschreven. In v1.1 liep zo'n waarde
  ongehinderd door: NaN kwam in forward_returns terecht, en omdat de
  selectie op `IS NULL` filtert (NaN is geen NULL) werd zo'n rij daarna nooit
  meer opgepikt. Negatieve koersen (bv. MOL.WA) gaven bovendien onmogelijke
  rendementen (-490% .. -543%).

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
# (fwd_close_<h>d/fwd_ret_<h>d-kolommen moeten al bestaan voor elke horizon
# in deze lijst) -- de kolomnamen hier worden dynamisch opgebouwd, maar
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
# of ontbrekend label, en zijn oud genoeg voor MINSTENS ÉÉN horizon?
# --------------------------------------------------------------------------
def haal_openstaande_rijen(conn) -> List[dict]:
    nu = datetime.now(timezone.utc)
    # Per horizon een eigen cutoff -- een rij wordt al opgepikt zodra ÉÉN
    # horizon oud genoeg is en nog een ontbrekend label heeft, niet pas
    # wanneer de LANGSTE horizon (bv. 60d) matuur is. Zo hoeft fwd_ret_30d
    # niet nodeloos ~48 dagen te wachten tot de rij ook oud genoeg is voor 60d.
    voorwaarden = []
    params: Dict[str, str] = {}
    for h in HORIZONS:
        kolom = f"fwd_ret_{h}d"
        param_naam = f"cutoff_{h}"
        # ~1.5x buffer voor weekends/feestdagen om h handelsdagen te dekken
        params[param_naam] = (nu - timedelta(days=int(h * 1.6) + 3)).strftime("%Y-%m-%d")
        voorwaarden.append(
            f"(s.datum <= %({param_naam})s AND (f.ticker IS NULL OR f.{kolom} IS NULL))"
        )
    horizon_filter = " OR ".join(voorwaarden)

    # ticker NOT LIKE '%/%' sluit bot_01cointegr.py's samengestelde
    # "TICKER_A/TICKER_B"-paarstrings uit (zie bouw_generieke_technicals.py
    # voor dezelfde fix en de uitleg) -- kunnen nooit via yfinance opgelost
    # worden, en koers is voor die rijen sowieso een spread-ratio, geen
    # echte prijs, dus een forward-rendement erop zou toch niet zinvol zijn.
    # De %% (i.p.v. losse %) is nodig omdat psycopg2 een bare % anders
    # verwart met zijn eigen %(naam)s-placeholder-syntax.
    query = f"""
        SELECT s.ticker, s.datum, s.strategie, s.beurs, s.koers
        FROM selecties s
        LEFT JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        WHERE s.koers IS NOT NULL
          AND s.ticker NOT LIKE '%%/%%'
          AND ({horizon_filter})
        ORDER BY s.ticker, s.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        return cur.fetchall()


# --------------------------------------------------------------------------
# Stap 2: per ticker het koersverloop ophalen en de horizons berekenen
# --------------------------------------------------------------------------
def bereken_labels_voor_ticker(ticker: str, rijen: List[dict]) -> List[dict]:
    """rijen = alle openstaande (datum, strategie, beurs, koers)-combinaties
    voor deze ene ticker. Geeft een lijst dicts terug, klaar voor upsert.

    v1.2: een forward-koers die NaN/inf is of <= 0 wordt behandeld als
    ONTBREKEND (None -> NULL). Zo komt er nooit NaN in forward_returns
    (NaN is geen NULL, dus die rijen werden nooit meer opgepikt) en
    schrijven we geen onmogelijke rendementen (bv. MOL.WA) weg."""
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

                    # v1.2: onbruikbare koers (NaN/inf of <= 0) = ontbrekend (NULL).
                    if not math.isfinite(fwd_close) or fwd_close <= 0:
                        rij[f"fwd_close_{h}d"] = None
