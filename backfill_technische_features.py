#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_technische_features.py
================================
Vult de nieuwe technische features (macd, macd_signaal, macd_hist,
bb_breedte, bb_percent_b, stoch_k, stoch_d, adx14, rel_sterkte_20d, hv20,
hv60, dagen_sinds_low52w, vol_ratio_50d, ema8, ema20, pct_from_ema8,
pct_from_ema20, ema8_minus_ema20) aan voor bestaande datums waar ze
nu NULL zijn in generieke_technicals.

Methode:
  1. Lees alle (ticker, datum) uit generieke_technicals waar minstens één
     van de nieuwe features NULL is
  2. Per ticker: haal OHLCV op via yfinance (zelfde aanroep als de live
     builder: Ticker().history(auto_adjust=True), start = vroegste datum
     minus LOOKBACK_BUFFER_DAGEN)
  3. Bereken ALLE indicatoren met de LIVE-functie bereken_indicatoren()
     uit bouw_generieke_technicals.py, eenmaal per ticker
  4. Per datum: lees de rij van de laatste handelsdag OP OF VÓÓR de datum
     (as-of, geen look-ahead) en UPDATE alleen de NULL-velden

Waarom de live-berekening hergebruiken: elke eigen implementatie in dit
script gaf andere waarden dan de live builder (bb_breedte x100, stochastics
zonder extra smoothing, Wilder-ADX, relatieve sterkte t.o.v. index,
vol_ratio_50d inclusief vandaag, dagen_sinds_low52w in handelsdagen). Eén
kolom met twee definities maakt elk model dat erop traint onbetrouwbaar.

Point-in-time: alle indicatoren in bereken_indicatoren() zijn causaal
(rolling / ewm / Wilder), dus de waarde op datum D hangt alleen af van data
tot en met D. Eenmaal berekenen over de hele historie en de rij van D
aflezen geeft hetzelfde als herberekenen met data tot D.

Veilig:
  - Alleen UPDATE, nooit INSERT
  - Alleen NULL-velden worden gevuld (COALESCE + WHERE ... IS NULL)
  - DRY RUN staat standaard AAN; alleen BACKFILL_DRY_RUN=0/false/nee
    zet schrijven aan. Elke andere (onduidelijke) waarde = dry-run.

Env:
  SUPABASE_DB_URL        (verplicht)
  BACKFILL_DRY_RUN       "1"/"true"/"ja" = dry-run, "0"/"false"/"nee" =
                         schrijven. Leeg of ontbrekend = dry-run.
  BACKFILL_BATCH_SIZE    max. aantal tickers per run (leeg/0 = alles)
  BACKFILL_TICKER_FILTER komma-gescheiden tickers, bv. "AAPL,ASML.AS"
                         (leeg = alle tickers met openstaande rijen)
  BACKFILL_SLEEP         seconden wachttijd tussen tickers (default 2.0)

WIJZIGINGEN NA REVIEW (2026-10-05):
A. Berekening hergebruikt uit bouw_generieke_technicals.py (zie boven).
B. DRY-RUN-FIX: de workflow geeft "true"/"false" door, het oude script
   vergeleek met "1". Daardoor was dry_run=true NIET actief en werd er
   toch geschreven. Nu worden beide notaties herkend en is dry-run de
   veilige default.
C. BATCH_SIZE en TICKER_FILTER werken nu. Een lege BACKFILL_BATCH_SIZE
   (workflow-default) gaf voorheen int("") -> crash bij opstarten.
D. Bij een batch wordt willekeurig geschud, zodat tickers die nooit
   gevuld kunnen worden (te korte historie) de volgende batches niet
   blijven blokkeren.
E. UPDATE-voorwaarde "... AND (kolom IS NULL OR ...)" zodat rowcount echt
   het aantal rijen is waar iets gevuld werd.
G. DRY-RUN CONTROLEERT DE UPDATE-WHERE: in dry-run draait nu dezelfde WHERE
   als SELECT (geen schrijfactie), zodat typefouten zoals "text = timestamp"
   (datum is een tekstkolom) al in de dry-run zichtbaar worden. De waarde
   van datum wordt ongewijzigd uit de database teruggegeven, niet omgezet
   naar een Timestamp.
F. Controle: ma50 en rsi14 (reeds in de tabel) worden herberekend en
   vergeleken met de opgeslagen waarde; dit toont of de koersdata/as-of
   uitlijning klopt (informatief, geen blokkade).

WIJZIGINGEN v2 (2026-10-10):
H. EMA8, EMA20, pct_from_ema8, pct_from_ema20 en ema8_minus_ema20
   toegevoegd aan KOLOM_NAAR_G. De rest van het script werkt automatisch
   met de nieuwe kolommen (loopt over de dict).
"""

import os
import sys
import math
import time
import random
import warnings
import datetime as dt

import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf

# Hergebruik de LIVE-berekening, zodat backfill en live identieke definities
# hebben. Dit script moet in dezelfde map staan als bouw_generieke_technicals.py.
from bouw_generieke_technicals import (
    bereken_indicatoren,
    download_index_returns,
    LOOKBACK_BUFFER_DAGEN,
    MAX_BAR_AFSTAND_DAGEN,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Config (robuust tegen lege strings)
# --------------------------------------------------------------------------
def _env_tekst(naam: str) -> str:
    return (os.environ.get(naam) or "").strip()


def _env_int(naam: str, standaard=None):
    w = _env_tekst(naam)
    if not w:
        return standaard
    try:
        return int(w)
    except ValueError:
        print(f"[WARN] {naam}='{w}' is geen geheel getal; gebruik {standaard}")
        return standaard


def _env_float(naam: str, standaard: float) -> float:
    w = _env_tekst(naam)
    if not w:
        return standaard
    try:
        return float(w)
    except ValueError:
        print(f"[WARN] {naam}='{w}' is geen getal; gebruik {standaard}")
        return standaard


def _env_dry_run() -> bool:
    """True = dry-run. Alleen een duidelijke 'nee' zet schrijven aan."""
    w = _env_tekst("BACKFILL_DRY_RUN").lower()
    if w in ("0", "false", "nee", "no", "n"):
        return False
    if w in ("", "1", "true", "ja", "yes", "y"):
        return True
    print(f"[WARN] BACKFILL_DRY_RUN='{w}' onduidelijk; dry-run blijft AAN")
    return True


DB_URL = os.environ.get("SUPABASE_DB_URL")
DRY_RUN = _env_dry_run()
BATCH_SIZE = _env_int("BACKFILL_BATCH_SIZE", None)
if BATCH_SIZE is not None and BATCH_SIZE <= 0:
    BATCH_SIZE = None
SLEEP_SEC = _env_float("BACKFILL_SLEEP", 2.0)
TICKER_FILTER = {
    t.strip().upper()
    for t in _env_tekst("BACKFILL_TICKER_FILTER").split(",")
    if t.strip()
}

# kolomnaam in generieke_technicals -> kolomnaam in het resultaat van
# bereken_indicatoren()
KOLOM_NAAR_G = {
    "macd": "MACD",
    "macd_signaal": "MACD_SIGNAAL",
    "macd_hist": "MACD_HIST",
    "bb_breedte": "BB_BREEDTE",
    "bb_percent_b": "BB_PERCENT_B",
    "stoch_k": "STOCH_K",
    "stoch_d": "STOCH_D",
    "adx14": "ADX14",
    "rel_sterkte_20d": "REL_STERKTE_20D",
    "hv20": "HV20",
    "hv60": "HV60",
    "dagen_sinds_low52w": "DAGEN_SINDS_LOW52W",
    "vol_ratio_50d": "VOL_RATIO_50D",
    # NIEUW (v2):
    "ema8": "EMA8",
    "ema20": "EMA20",
    "pct_from_ema8": "PCT_FROM_EMA8",
    "pct_from_ema20": "PCT_FROM_EMA20",
    "ema8_minus_ema20": "EMA8_MINUS_EMA20",
}
DOEL_FEATURES = list(KOLOM_NAAR_G.keys())
INT_FEATURES = {"dagen_sinds_low52w"}

# Controle tegen reeds opgeslagen waarden:
# (kolom in DB, kolom in g, type verschil, tolerantie)
CONTROLE_KOLOMMEN = [
    ("ma50", "MA50", "rel", 0.02),   # max 2% relatief verschil
    ("rsi14", "RSI14", "abs", 3.0),  # max 3 RSI-punten verschil
]

MAX_VOORBEELDEN = 3   # aantal tickers waarvan een voorbeeldrij wordt getoond


# --------------------------------------------------------------------------
# Hulpfuncties
# --------------------------------------------------------------------------
def _veilig(waarde, is_int: bool = False):
    """Zelfde afronding als de live builder: 4 decimalen, NaN -> None."""
    try:
        f = float(waarde)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(round(f)) if is_int else round(f, 4)
    except Exception:
        return None


def _sql_datum(d):
    """Zet Timestamp/datetime om naar date voor gebruik in SQL."""
    if isinstance(d, (pd.Timestamp, dt.datetime)):
        return d.date()
    return d


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------
def haal_te_vullen_rijen(conn) -> pd.DataFrame:
    """(ticker, datum) waar minstens één doel-feature NULL is."""
    where = " OR ".join(f"{f} IS NULL" for f in DOEL_FEATURES)
    # position('/' in ticker) = 0: sluit pairs-tickers uit (zoals de live
    # builder), zonder %-teken in de query.
    query = f"""
        SELECT ticker, datum
        FROM generieke_technicals
        WHERE ({where})
          AND position('/' in ticker) = 0
        ORDER BY ticker, datum
    """
    return pd.read_sql(query, conn)


def update_features(conn, ticker, datum, waarden: dict) -> int:
    """
    UPDATE alleen de NULL-velden. waarden = {feature: waarde}.
    Retourneert het aantal bijgewerkte rijen (0 of 1). In dry-run: het aantal
    rijen dat bijgewerkt ZOU worden (SELECT, er wordt niets geschreven).
    """
    te_setten = {
        k: v for k, v in waarden.items()
        if k in DOEL_FEATURES and v is not None
    }
    if not te_setten:
        return 0

    null_voorwaarde = " OR ".join(f"{k} IS NULL" for k in te_setten)

    if DRY_RUN:
        # Zelfde WHERE als de echte UPDATE, maar als SELECT: er wordt niets
        # geschreven, maar het script controleert wel dat de rij gevonden
        # wordt en dat het datumtype klopt.
        query = (
            "SELECT count(*) FROM generieke_technicals "
            "WHERE ticker = %s AND datum = %s "
            f"AND ({null_voorwaarde})"
        )
        with conn.cursor() as cur:
            cur.execute(query, [ticker, datum])
            return int(cur.fetchone()[0])

    set_delen = [f"{k} = COALESCE({k}, %s)" for k in te_setten]
    params = list(te_setten.values()) + [ticker, datum]
    query = f"""
        UPDATE generieke_technicals
        SET {', '.join(set_delen)}
        WHERE ticker = %s AND datum = %s
          AND ({null_voorwaarde})
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.rowcount


def controleer_tegen_db(conn, ticker, controle_rijen):
    """
    Vergelijkt herberekende ma50/rsi14 met de opgeslagen waarden.
    controle_rijen = lijst van (datum_sql, {db_kolom: herberekende waarde}).
    Retourneert (n_vergeleken, n_afwijkend). Informatief.
    """
    if not controle_rijen:
        return 0, 0
    kolommen = ", ".join(k[0] for k in CONTROLE_KOLOMMEN)
    query = (f"SELECT datum, {kolommen} FROM generieke_technicals "
             f"WHERE ticker = %s AND datum = ANY(%s)")
    datums = [d for d, _ in controle_rijen]
    with conn.cursor() as cur:
        cur.execute(query, (ticker, datums))
        db_rijen = {_sql_datum(r[0]): r[1:] for r in cur.fetchall()}

    n_vergeleken = 0
    n_afwijkend = 0
    for datum, nieuw in controle_rijen:
        opgeslagen = db_rijen.get(datum)
        if opgeslagen is None:
            continue
        for (db_kol, _, soort, tol), oud in zip(CONTROLE_KOLOMMEN, opgeslagen):
            nw = nieuw.get(db_kol)
            if oud is None or nw is None:
                continue
            oud = float(oud)
            n_vergeleken += 1
            if soort == "rel":
                verschil = abs(nw - oud) / abs(oud) if oud != 0 else abs(nw)
            else:
                verschil = abs(nw - oud)
            if verschil > tol:
                n_afwijkend += 1
    return n_vergeleken, n_afwijkend


# --------------------------------------------------------------------------
# Verwerking per ticker
# --------------------------------------------------------------------------
def verwerk_ticker(conn, ticker, datums, index_ret) -> dict:
    stat = {
        "status": "ok", "fout": None,
        "n_rijen": 0, "n_bijgewerkt": 0,
        "n_onvolledig": 0, "n_geen_bar": 0,
        "n_controle": 0, "n_afwijkend": 0,
        "voorbeeld": None,
    }

    vroegste = min(pd.Timestamp(d) for d in datums)
    start = (vroegste - pd.Timedelta(days=LOOKBACK_BUFFER_DAGEN)).strftime("%Y-%m-%d")

    try:
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    except Exception as e:
        stat["status"] = "download_fout"
        stat["fout"] = str(e)
        return stat

    if hist is None or hist.empty or "Close" not in hist.columns:
        stat["status"] = "geen_data"
        return stat

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    g = bereken_indicatoren(hist, index_ret)

    controle_rijen = []
    for datum_db in datums:
        doel = pd.Timestamp(datum_db)

        # As-of: laatste handelsdag OP OF VÓÓR de datum (zoals de live
        # builder v2.1). Geen look-ahead.
        pos = g.index.searchsorted(doel, side="right") - 1
        if pos < 0 or (doel.normalize() - g.index[pos]).days > MAX_BAR_AFSTAND_DAGEN:
            stat["n_geen_bar"] += 1
            continue
        rij = g.iloc[pos]

        # Onvolledige koersbalk: niet invullen (de datums zijn historisch,
        # er komt geen betere balk meer bij).
        if pd.isna(rij["Close"]) or pd.isna(rij["High"]) or pd.isna(rij["Low"]):
            stat["n_onvolledig"] += 1
            continue

        datum_sql = _sql_datum(datum_db)

        waarden = {}
        for kolom, g_kolom in KOLOM_NAAR_G.items():
            if g_kolom not in g.columns:
                continue
            v = _veilig(rij[g_kolom], is_int=(kolom in INT_FEATURES))
            if v is not None:
                waarden[kolom] = v

        # Voor de controle tegen reeds opgeslagen ma50 / rsi14
        controle_rijen.append((
            datum_sql,
            {db_kol: _veilig(rij[g_kol]) for db_kol, g_kol, _, _ in CONTROLE_KOLOMMEN},
        ))

        if not waarden:
            continue

        stat["n_rijen"] += 1
        if stat["voorbeeld"] is None:
            stat["voorbeeld"] = (datum_sql, waarden)
        stat["n_bijgewerkt"] += update_features(conn, ticker, datum_sql, waarden)

    if not DRY_RUN:
        conn.commit()

    # Controle na de commit, zodat een eventuele SQL-fout hier geen
    # reeds uitgevoerde updates terugdraait.
    try:
        n_v, n_a = controleer_tegen_db(conn, ticker, controle_rijen)
        stat["n_controle"], stat["n_afwijkend"] = n_v, n_a
    except Exception as e:
        conn.rollback()
        print(f"  [INFO] controle tegen DB overgeslagen: {e}")

    return stat


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not DB_URL:
        sys.exit("SUPABASE_DB_URL ontbreekt")

    print("Backfill technische features")
    print(f"  dry_run       = {DRY_RUN}")
    print(f"  batch_size    = {BATCH_SIZE if BATCH_SIZE else '(alles)'}")
    print(f"  ticker_filter = {sorted(TICKER_FILTER) if TICKER_FILTER else '(geen)'}")
    print(f"  sleep_sec     = {SLEEP_SEC}")
    print()
    if DRY_RUN:
        print("DRY RUN: er wordt NIETS naar de database geschreven.")
        print()

    conn = psycopg2.connect(DB_URL)
    try:
        te_vullen = haal_te_vullen_rijen(conn)
        print(f"{len(te_vullen):,} rijen met minstens één NULL-feature")
        if te_vullen.empty:
            print("Niets te doen.")
            return

        per_ticker = te_vullen.groupby("ticker")["datum"].apply(list).to_dict()
        tickers = sorted(per_ticker.keys())
        print(f"{len(tickers):,} unieke tickers met openstaande rijen")

        if TICKER_FILTER:
            tickers = [t for t in tickers if t.upper() in TICKER_FILTER]
            print(f"{len(tickers):,} tickers na ticker_filter")
            ontbrekend = TICKER_FILTER - {t.upper() for t in tickers}
            if ontbrekend:
                print(f"  [INFO] geen openstaande rijen voor: {sorted(ontbrekend)}")

        if BATCH_SIZE and len(tickers) > BATCH_SIZE:
            random.shuffle(tickers)
            tickers = sorted(tickers[:BATCH_SIZE])
            print(f"Batch: {len(tickers)} tickers in deze run (willekeurige selectie)")

        if not tickers:
            print("Geen tickers om te verwerken.")
            return
        print()

        # Referentie-index één keer ophalen (voor rel_sterkte_20d)
        vroegste_alle = min(pd.Timestamp(d) for t in tickers for d in per_ticker[t])
        index_start = (vroegste_alle - pd.Timedelta(days=LOOKBACK_BUFFER_DAGEN)).strftime("%Y-%m-%d")
        index_ret = download_index_returns(index_start)
        if index_ret is None:
            print("[WARN] Referentie-index niet beschikbaar: rel_sterkte_20d "
                  "blijft NULL in deze run (de rest wordt wel gevuld).")
        print()

        totaal = {
            "n_rijen": 0, "n_bijgewerkt": 0, "n_onvolledig": 0,
            "n_geen_bar": 0, "n_controle": 0, "n_afwijkend": 0,
        }
        n_geen_data = 0
        n_download_fout = 0
        fouten = []
        voorbeelden_getoond = 0

        for i, ticker in enumerate(tickers, 1):
            datums = per_ticker[ticker]
            print(f"[{i}/{len(tickers)}] {ticker} ({len(datums)} datums)")

            try:
                stat = verwerk_ticker(conn, ticker, datums, index_ret)
            except Exception as e:
                fouten.append((ticker, str(e)))
                print(f"  → FOUT: {e}")
                # Ook in dry-run: een mislukte SELECT laat de transactie
                # anders in 'aborted' staan en laat alle volgende tickers falen.
                conn.rollback()
                time.sleep(SLEEP_SEC)
                continue

            if stat["status"] == "geen_data":
                n_geen_data += 1
                print("  → geen data")
            elif stat["status"] == "download_fout":
                n_download_fout += 1
                fouten.append((ticker, stat["fout"]))
                print(f"  → download mislukt: {stat['fout']}")
            else:
                for k in totaal:
                    totaal[k] += stat[k]
                if stat["n_afwijkend"]:
                    print(f"  [INFO] {stat['n_afwijkend']} van {stat['n_controle']} "
                          f"controlewaarden (ma50/rsi14) wijken af van de opgeslagen waarde")
                if stat["voorbeeld"] and voorbeelden_getoond < MAX_VOORBEELDEN:
                    d, w = stat["voorbeeld"]
                    voorbeeld_txt = ", ".join(f"{k}={v}" for k, v in list(w.items())[:8])
                    print(f"  voorbeeld {d}: {voorbeeld_txt} ...")
                    voorbeelden_getoond += 1

            time.sleep(SLEEP_SEC)

        # Samenvatting
        print()
        print("=" * 60)
        print("SAMENVATTING" + ("  (DRY RUN, niets geschreven)" if DRY_RUN else ""))
        print("=" * 60)
        print(f"Tickers verwerkt      : {len(tickers)}")
        print(f"Rijen met waarden     : {totaal['n_rijen']}")
        label = "Rijen die bijgewerkt zouden worden" if DRY_RUN else "Rijen bijgewerkt"
        print(f"{label:<22}: {totaal['n_bijgewerkt']}")
        print(f"Geen data (ticker)    : {n_geen_data}")
        print(f"Download mislukt      : {n_download_fout}")
        print(f"Geen koersbalk (datum): {totaal['n_geen_bar']}")
        print(f"Onvolledige balk      : {totaal['n_onvolledig']}")
        if totaal["n_controle"]:
            pct = 100 * totaal["n_afwijkend"] / totaal["n_controle"]
            print(f"Controle ma50/rsi14   : {totaal['n_controle']} vergeleken, "
                  f"{totaal['n_afwijkend']} afwijkend ({pct:.1f}%)")

        if fouten:
            print(f"\nFouten ({len(fouten)}):")
            for t, e in fouten[:20]:
                print(f"  {t}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
