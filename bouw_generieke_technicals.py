#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_generieke_technicals.py  —  GEDEELDE TECHNISCHE INDICATOREN  v1.0

DOEL
====
Lost het "n_strategieen=1"-probleem van analyse_parameter_correlatie.py op:
een technische indicator (ATR%, RSI, MA-afstand, volumeratio, 52w-hoogte)
hangt niet af van WELKE strategie een aandeel koos, enkel van de koers-
/volumegeschiedenis op die datum. Door deze centraal en strategie-
onafhankelijk te berekenen -- één rij per (ticker, datum), niet per
(ticker, datum, strategie) -- krijgt ELKE selectie uit ELKE bot dezelfde
kolommen ingevuld, i.p.v. enkel de rijen van de ene bot die toevallig zijn
eigen ATR berekent.

GEEN LOOK-AHEAD BIAS: enkel indicatoren die uit koers-/volumedata TOT EN MET
`datum` berekenbaar zijn (RSI/ATR/MA/volumeratio/52w-hoogte). Fundamentele
data (ROE, P/E, FCF-yield, Piotroski, sector) wordt hier BEWUST NIET
opgenomen -- yfinance geeft daarvan enkel de HUIDIGE momentopname, geen
historisch puntmoment, dus retroactief invullen zou toekomstige informatie
in het verleden plaatsen. Vandaar de aparte, bewust kleinere scope van dit
script t.o.v. wat voor fundamentals ooit zou kunnen (nooit met yfinance).

INCREMENTEEL, GEEN WACHTTIJD NODIG (in tegenstelling tot forward_returns):
een technische indicator op datum X is METEEN berekenbaar zodra X voorbij
is, geen 5/10/20 handelsdagen wachten zoals bij forward-rendement. Elke
run vult dus zowel de historische achterstand als de nieuwste selecties
van gisteren/vandaag aan.

Zelfde patronen als bouw_forward_returns.py: bulk yfinance-download per
ticker (dekt alle openstaande datums van die ticker in 1 call), upsert
(ON CONFLICT DO UPDATE), MAX_TICKERS_PER_RUN als veiligheidslimiet,
prioriteit op oudste openstaande datum i.p.v. alfabetisch (zelfde
fairness-fix als daar).

GEBRUIK
=======
  python bouw_generieke_technicals.py build

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel), MAX_TICKERS_PER_RUN (default 400)
"""

import os
import sys
import math
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
import yfinance as yf
import pandas as pd
import requests

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_TICKERS_PER_RUN = int(os.environ.get("MAX_TICKERS_PER_RUN", "400"))

RSI_PERIOD = 14
ATR_PERIOD = 14
MA_KORT = 50
MA_LANG = 200
# genoeg extra historiek vóór de vroegste benodigde datum om MA200/52w-hoogte
# op DIE datum al zinvol te kunnen berekenen (geen look-ahead: enkel data
# tot en met de datum zelf wordt gebruikt, dit is puur de opstartbuffer)
LOOKBACK_BUFFER_DAGEN = 380


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
# Wilder-smoothing, zelfde conventie als bot_00mr.py / bot_01volhunter.py
# --------------------------------------------------------------------------
def _wilder(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period
            + valid.iloc[i] / period
        )
    return result


def bereken_indicatoren(hist: pd.DataFrame) -> pd.DataFrame:
    g = hist.copy()
    close, high, low, volume = g["Close"], g["High"], g["Low"], g["Volume"]

    g["IBS"] = (close - low) / (high - low + 1e-9)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = _wilder(gain, RSI_PERIOD) / (_wilder(loss, RSI_PERIOD) + 1e-9)
    g["RSI14"] = 100 - (100 / (1 + rs))

    hl = high - low
    hcp = (high - close.shift()).abs()
    lcp = (low - close.shift()).abs()
    tr = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    g["ATR14"] = _wilder(tr, ATR_PERIOD)
    g["ATR14_PCT"] = g["ATR14"] / close * 100

    g["MA50"] = close.rolling(MA_KORT).mean()
    g["MA200"] = close.rolling(MA_LANG).mean()
    g["PCT_FROM_MA50"] = (close / g["MA50"] - 1) * 100
    g["PCT_FROM_MA200"] = (close / g["MA200"] - 1) * 100

    basis_vol = volume.rolling(20).mean()
    g["VOL_RATIO_20D"] = volume / basis_vol

    g["HIGH52W"] = high.rolling(252, min_periods=50).max()
    g["PCT_FROM_HIGH52W"] = (close / g["HIGH52W"] - 1) * 100

    return g


# --------------------------------------------------------------------------
# Stap 1: welke (ticker, datum)-paren ontbreken nog in generieke_technicals?
# --------------------------------------------------------------------------
def haal_openstaande_paren(conn) -> List[dict]:
    query = """
        SELECT DISTINCT s.ticker, s.datum
        FROM selecties s
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE g.ticker IS NULL
        ORDER BY s.ticker, s.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()


# --------------------------------------------------------------------------
# Stap 2: per ticker het koersverloop ophalen en indicatoren berekenen
# --------------------------------------------------------------------------
def bereken_voor_ticker(ticker: str, datums: List[str]) -> List[dict]:
    vroegste = min(datums)
    start = (pd.Timestamp(vroegste) - pd.Timedelta(days=LOOKBACK_BUFFER_DAGEN)).strftime("%Y-%m-%d")
    try:
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    except Exception as e:
        print(f"  [WARN] {ticker}: download mislukt ({e})")
        return []

    if hist is None or hist.empty or "Close" not in hist.columns:
        print(f"  [WARN] {ticker}: geen koersdata")
        return []

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    g = bereken_indicatoren(hist)

    resultaten = []
    for datum in datums:
        try:
            doel = pd.Timestamp(datum)
            pos = g.index.searchsorted(doel)
            if pos >= len(g):
                continue
            # exacte handelsdag pakken indien aanwezig, anders de eerstvolgende
            rij = g.iloc[pos]

            def veilig(waarde):
                try:
                    f = float(waarde)
                    return None if math.isnan(f) else round(f, 4)
                except Exception:
                    return None

            resultaten.append({
                "ticker": ticker, "datum": datum,
                "atr14": veilig(rij["ATR14"]), "atr14_pct": veilig(rij["ATR14_PCT"]),
                "rsi14": veilig(rij["RSI14"]), "ibs": veilig(rij["IBS"]),
                "ma50": veilig(rij["MA50"]), "ma200": veilig(rij["MA200"]),
                "pct_from_ma50": veilig(rij["PCT_FROM_MA50"]),
                "pct_from_ma200": veilig(rij["PCT_FROM_MA200"]),
                "vol_ratio_20d": veilig(rij["VOL_RATIO_20D"]),
                "high52w": veilig(rij["HIGH52W"]), "pct_from_high52w": veilig(rij["PCT_FROM_HIGH52W"]),
            })
        except Exception as e:
            print(f"  [WARN] {ticker} {datum}: {e}")
            continue

    return resultaten


# --------------------------------------------------------------------------
# Stap 3: upsert
# --------------------------------------------------------------------------
KOLOMMEN = ["ticker", "datum", "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
            "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w"]


def upsert_rijen(conn, rijen: List[dict]) -> int:
    if not rijen:
        return 0
    kolom_lijst = ", ".join(KOLOMMEN)
    placeholders = ", ".join(f"%({k})s" for k in KOLOMMEN)
    update_lijst = ", ".join(f"{k} = EXCLUDED.{k}" for k in KOLOMMEN if k not in ("ticker", "datum"))
    update_lijst += ", bijgewerkt_op = now()"

    query = f"""
        INSERT INTO generieke_technicals ({kolom_lijst})
        VALUES ({placeholders})
        ON CONFLICT (ticker, datum)
        DO UPDATE SET {update_lijst};
    """
    with conn.cursor() as cur:
        for rij in rijen:
            cur.execute(query, {k: rij.get(k) for k in KOLOMMEN})
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
        open_paren = haal_openstaande_paren(conn)
        print(f"{len(open_paren)} (ticker, datum)-paren nog niet aanwezig in generieke_technicals.")

        per_ticker: Dict[str, List[str]] = {}
        for r in open_paren:
            per_ticker.setdefault(r["ticker"], []).append(r["datum"])

        # zelfde fairness-fix als bouw_forward_returns.py: oudste eerst
        oudste_per_ticker = {t: min(datums) for t, datums in per_ticker.items()}
        alle_tickers_gesorteerd = sorted(per_ticker.keys(), key=lambda t: oudste_per_ticker[t])
        tickers = alle_tickers_gesorteerd[:MAX_TICKERS_PER_RUN]
        overgeslagen = len(per_ticker) - len(tickers)
        print(f"{len(tickers)} unieke tickers te verwerken dit run"
              + (f" ({overgeslagen} tickers volgen in een volgend run)" if overgeslagen > 0 else ""))

        totaal_bijgewerkt = 0
        for i, ticker in enumerate(tickers, start=1):
            rijen = bereken_voor_ticker(ticker, per_ticker[ticker])
            totaal_bijgewerkt += upsert_rijen(conn, rijen)
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} tickers verwerkt...")
            time.sleep(0.1)

        send_telegram(
            f"📊 *Generieke Technicals — {vandaag()}*\n\n"
            f"{totaal_bijgewerkt} (ticker, datum)-rijen bijgewerkt\n"
            f"{len(tickers)} unieke tickers verwerkt dit run"
            + (f" ({overgeslagen} tickers nog te gaan)" if overgeslagen > 0 else "")
        )
        print(f"\nKlaar. {totaal_bijgewerkt} rijen bijgewerkt.")
    finally:
        conn.close()


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "build"
    run_build()
