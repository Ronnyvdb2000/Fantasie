#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_technische_features.py
================================
Vult de nieuwe technische features (macd_hist, stoch_k, adx14, hv20,
rel_sterkte_20d, bb_percent_b, bb_breedte, stoch_d, macd, macd_signaal,
hv60, dagen_sinds_low52w, vol_ratio_50d) aan voor bestaande datums waar
ze nu NULL zijn.

Methode:
  1. Lees alle (ticker, datum) uit generieke_technicals waar minstens
     één van de nieuwe features NULL is
  2. Haal per unieke ticker de OHLCV op via yfinance (bulk-download)
  3. Bereken indicatoren POINT-IN-TIME (alleen data tot datum D)
  4. UPDATE alleen de NULL-velden

Veilig:
  - Alleen UPDATE, nooit INSERT
  - Alleen NULL-velden worden gevuld (bestaande waarden blijven)
  - Dry-run optie via env var BACKFILL_DRY_RUN=1

Env:
  SUPABASE_DB_URL
  BACKFILL_DRY_RUN   (default 0)
  BACKFILL_BATCH_SIZE (default 50)
  BACKFILL_SLEEP     (default 2.0)
"""

import os
import sys
import time
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DB_URL = os.environ.get("SUPABASE_DB_URL")
DRY_RUN = os.environ.get("BACKFILL_DRY_RUN", "0") == "1"
BATCH_SIZE = int(os.environ.get("BACKFILL_BATCH_SIZE", "50"))
SLEEP_SEC = float(os.environ.get("BACKFILL_SLEEP", "2.0"))

# Features om aan te vullen
DOEL_FEATURES = [
    "macd", "macd_signaal", "macd_hist",
    "bb_breedte", "bb_percent_b",
    "stoch_k", "stoch_d", "adx14",
    "rel_sterkte_20d", "hv20", "hv60",
    "dagen_sinds_low52w", "vol_ratio_50d",
]

# Hoeveel jaar historie ophalen? 2 jaar is genoeg voor alle indicatoren
HISTORIE_JAREN = 2


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------
def haal_te_vullen_rijen(conn):
    """Geeft DataFrame met (ticker, datum) waar minstens één feature NULL is."""
    features_str = ", ".join(DOEL_FEATURES)
    where = " OR ".join(f"{f} IS NULL" for f in DOEL_FEATURES)
    query = f"""
        SELECT ticker, datum
        FROM generieke_technicals
        WHERE {where}
        ORDER BY ticker, datum
    """
    return pd.read_sql(query, conn)


def haal_bestaande_features(conn, ticker, datum):
    """Haalt bestaande waarden op voor verificatie."""
    query = f"""
        SELECT {', '.join(DOEL_FEATURES)}
        FROM generieke_technicals
        WHERE ticker = %s AND datum = %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (ticker, datum))
        return cur.fetchone()


def update_features(conn, ticker, datum, waarden):
    """UPDATE alleen de NULL-velden. waarden is dict {feature: value}."""
    if not waarden:
        return 0

    # Alleen features met niet-NaN waarde
    te_setten = {
        k: v for k, v in waarden.items()
        if k in DOEL_FEATURES and v is not None and not pd.isna(v)
    }
    if not te_setten:
        return 0

    # UPDATE met COALESCE: alleen als bestaande waarde NULL is
    set_delen = []
    params = []
    for k, v in te_setten.items():
        set_delen.append(f"{k} = COALESCE({k}, %s)")
        params.append(float(v))

    params.extend([ticker, datum])
    query = f"""
        UPDATE generieke_technicals
        SET {', '.join(set_delen)}
        WHERE ticker = %s AND datum = %s
    """

    if DRY_RUN:
        return len(te_setten)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.rowcount


# --------------------------------------------------------------------------
# Indicator-berekening (point-in-time)
# --------------------------------------------------------------------------
def bereken_indicatoren_voor_datum(df_ohlcv, datum):
    """
    Berekent alle indicatoren voor één specifieke datum, met alleen data
    tot en met die datum. df_ohlcv moet kolommen hebben: open, high, low,
    close, volume; index = DatetimeIndex.
    """
    if df_ohlcv is None or df_ohlcv.empty:
        return {}

    # Filter op datum
    df = df_ohlcv[df_ohlcv.index <= datum].copy()
    if len(df) < 60:  # minimale historie voor indicatoren
        return {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    resultaat = {}

    # MACD (12, 26, 9)
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signaal = macd.ewm(span=9, adjust=False).mean()
        resultaat["macd"] = float(macd.iloc[-1])
        resultaat["macd_signaal"] = float(signaal.iloc[-1])
        resultaat["macd_hist"] = float(macd.iloc[-1] - signaal.iloc[-1])

    # Bollinger Bands (20, 2)
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        bb_breedte = (upper - lower) / ma20
        bb_pct_b = (close - lower) / (upper - lower)
        resultaat["bb_breedte"] = float(bb_breedte.iloc[-1]) if not pd.isna(bb_breedte.iloc[-1]) else None
        resultaat["bb_percent_b"] = float(bb_pct_b.iloc[-1]) if not pd.isna(bb_pct_b.iloc[-1]) else None

    # Stochastics (14, 3, 3)
    if len(df) >= 14:
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k_ruw = 100 * (close - low14) / (high14 - low14)
        k = k_ruw.rolling(3).mean()
        d = k.rolling(3).mean()
        resultaat["stoch_k"] = float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else None
        resultaat["stoch_d"] = float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else None

    # ADX (14)
    if len(df) >= 28:
        try:
            high_diff = high.diff()
            low_diff = -low.diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
            min_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr
            min_di = 100 * pd.Series(min_dm, index=df.index).rolling(14).mean() / atr
            dx = 100 * (plus_di - min_di).abs() / (plus_di + min_di)
            adx = dx.rolling(14).mean()
            if not pd.isna(adx.iloc[-1]):
                resultaat["adx14"] = float(adx.iloc[-1])
        except Exception:
            pass

    # Relatieve sterkte (20d) = rendement over 20 dagen
    if len(close) >= 21:
        rs = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        resultaat["rel_sterkte_20d"] = float(rs) if not pd.isna(rs) else None

    # HV20 en HV60 (historical volatility = std van log returns)
    log_ret = np.log(close / close.shift(1))
    if len(log_ret.dropna()) >= 20:
        hv20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
        resultaat["hv20"] = float(hv20.iloc[-1]) if not pd.isna(hv20.iloc[-1]) else None
    if len(log_ret.dropna()) >= 60:
        hv60 = log_ret.rolling(60).std() * np.sqrt(252) * 100
        resultaat["hv60"] = float(hv60.iloc[-1]) if not pd.isna(hv60.iloc[-1]) else None

    # Dagen sinds 52w low
    if len(df) >= 252:
        recent = df.tail(252)
        idx_low = recent["Low"].idxmin()
        dagen = (df.index[-1] - idx_low).days
        resultaat["dagen_sinds_low52w"] = int(dagen)

    # Volume ratio 50d
    if len(volume) >= 51:
        vol_avg_50 = volume.rolling(50).mean()
        if vol_avg_50.iloc[-2] > 0:
            ratio = volume.iloc[-1] / vol_avg_50.iloc[-2]
            resultaat["vol_ratio_50d"] = float(ratio) if not pd.isna(ratio) else None

    return resultaat


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not DB_URL:
        sys.exit("SUPABASE_DB_URL ontbreekt")

    print(f"Backfill technische features — dry_run={DRY_RUN}")
    print()

    with psycopg2.connect(DB_URL) as conn:
        te_vullen = haal_te_vullen_rijen(conn)
        print(f"{len(te_vullen):,} rijen met minstens één NULL-feature")

        unieke_tickers = sorted(te_vullen["ticker"].unique().tolist())
        print(f"{len(unieke_tickers):,} unieke tickers")
        print()

        if DRY_RUN:
            print("DRY RUN — geen database-writes")
            print()

        # Groepeer rijen per ticker voor efficiënte verwerking
        per_ticker = te_vullen.groupby("ticker")["datum"].apply(list).to_dict()

        n_verwerkt = 0
        n_geupdatet = 0
        n_geen_data = 0
        n_geen_historie = 0
        fouten = []

        periode_start = dt.date.today() - dt.timedelta(days=HISTORIE_JAREN * 365)

        for i, ticker in enumerate(unieke_tickers, 1):
            datums = per_ticker[ticker]
            print(f"[{i}/{len(unieke_tickers)}] {ticker} ({len(datums)} datums)")

            try:
                # yfinance bulk-download voor deze ticker
                df = yf.download(
                    ticker,
                    start=periode_start.isoformat(),
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if df is None or df.empty:
                    n_geen_data += 1
                    print(f"  → geen data")
                    time.sleep(SLEEP_SEC)
                    continue

                # yfinance geeft soms MultiIndex kolommen terug
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.index = pd.to_datetime(df.index)

                # Verwerk elke datum
                for datum_str in datums:
                    try:
                        datum = pd.to_datetime(datum_str)
                    except Exception:
                        continue

                    waarden = bereken_indicatoren_voor_datum(df, datum)
                    if not waarden:
                        n_geen_historie += 1
                        continue

                    n_verwerkt += 1
                    rows = update_features(conn, ticker, datum, waarden)
                    n_geupdatet += rows

                if not DRY_RUN:
                    conn.commit()

                time.sleep(SLEEP_SEC)

            except Exception as e:
                fouten.append((ticker, str(e)))
                print(f"  → FOUT: {e}")
                if not DRY_RUN:
                    conn.rollback()
                time.sleep(SLEEP_SEC)

        # Samenvatting
        print()
        print("=" * 60)
        print("SAMENVATTING")
        print("=" * 60)
        print(f"Tickers verwerkt : {len(unieke_tickers)}")
        print(f"Rijen verwerkt   : {n_verwerkt}")
        print(f"Features gevuld  : {n_geupdatet}")
        print(f"Geen data        : {n_geen_data}")
        print(f"Geen historie    : {n_geen_historie}")

        if fouten:
            print(f"\nFouten ({len(fouten)}):")
            for t, e in fouten[:20]:
                print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
