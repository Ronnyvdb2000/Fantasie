#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_generieke_technicals.py
=============================

Bouwt per ticker/datum een rij met technische indicatoren in de tabel
generieke_technicals.

Scope: alle tickers uit de tickers_*.txt bestanden (het brede universum).

Features (kolommen):
  ATR14, ATR14_pct, RSI14, IBS,
  MA50, MA200, pct_from_ma50, pct_from_ma200,
  vol_ratio_20d, high52w, pct_from_high52w,
  MACD, MACD_signaal, MACD_hist,
  BB_breedte, BB_percent_b,
  STOCH_K, STOCH_D, ADX14,
  REL_STERKTE_20D, HV20, HV60,
  DAGEN_SINDS_LOW52W, VOL_RATIO_50D,
  EMA8, EMA20, PCT_FROM_EMA8, PCT_FROM_EMA20, EMA8_MINUS_EMA20

Env:
  SUPABASE_DB_URL    verplicht
  BACKFILL_DATUM     optioneel; default = vandaag (YYYY-MM-DD)
"""

import os
import sys
import math
import time
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf

warnings.filterwarnings("ignore")

DB_URL = os.environ.get("SUPABASE_DB_URL")

# Buffer voor indicatoren met lange lookback (MA200, 52w-high, HV60)
LOOKBACK_BUFFER_DAGEN = 400

# Maximale afstand tussen gevraagde datum en laatste bar
MAX_BAR_AFSTAND_DAGEN = 7

# Index voor relatieve sterkte (referentie voor REL_STERKTE_20D)
INDEX_TICKER = "^GSPC"


# --------------------------------------------------------------------------
# Bestandslijst tickers
# --------------------------------------------------------------------------
def bouw_bestandslijst():
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]


def load_tickers_from_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(set(result))


def alle_tickers():
    alle = set()
    for f in bouw_bestandslijst():
        alle.update(load_tickers_from_file(f))
    return sorted(alle)


# --------------------------------------------------------------------------
# Indicatoren
# --------------------------------------------------------------------------
def _veilig(waarde, is_int=False):
    try:
        f = float(waarde)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(round(f)) if is_int else round(f, 4)
    except Exception:
        return None


def download_index_returns(start_datum):
    try:
        hist = yf.Ticker(INDEX_TICKER).history(
            start=start_datum, auto_adjust=True
        )
        if hist is None or hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        ret = hist["Close"].pct_change()
        return ret
    except Exception as e:
        print(f"[index] download mislukt: {e}")
        return None


def bereken_indicatoren(hist, index_ret=None):
    """
    Berekent alle indicatoren voor één ticker.
    `hist` = OHLCV DataFrame met kolommen Open, High, Low, Close, Volume.
    `index_ret` = Series met index-rendementen (voor rel_sterkte_20d).
    Retourneert DataFrame met alle indicatoren per datum.
    """
    df = hist.copy()
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out = pd.DataFrame(index=df.index)

    # ATR14 (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out["ATR14"] = atr14
    out["ATR14_PCT"] = (atr14 / close) * 100

    # RSI14 (Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    out["RSI14"] = rsi14

    # IBS = Internal Bar Strength
    ibs = (close - low) / (high - low).replace(0, np.nan)
    out["IBS"] = ibs

    # MA50, MA200
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()
    out["PCT_FROM_MA50"] = ((close - out["MA50"]) / out["MA50"]) * 100
    out["PCT_FROM_MA200"] = ((close - out["MA200"]) / out["MA200"]) * 100

    # Volume ratio 20d
    vol_avg_20 = volume.rolling(20).mean()
    out["VOL_RATIO_20D"] = volume / vol_avg_20

    # 52w high
    high52w = high.rolling(252).max()
    out["HIGH52W"] = high52w
    out["PCT_FROM_HIGH52W"] = ((close - high52w) / high52w) * 100

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out["MACD"] = macd
    out["MACD_SIGNAAL"] = macd_sig
    out["MACD_HIST"] = macd - macd_sig

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    out["BB_BREEDTE"] = ((bb_upper - bb_lower) / ma20) * 100
    out["BB_PERCENT_B"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Stochastics (14, 3, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k_ruw = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    k = k_ruw.rolling(3).mean()
    d = k.rolling(3).mean()
    out["STOCH_K"] = k
    out["STOCH_D"] = d

    # ADX14 (Wilder)
    high_diff = high.diff()
    low_diff = -low.diff()
    plus_dm = pd.Series(
        np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0),
        index=df.index,
    )
    min_dm = pd.Series(
        np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0),
        index=df.index,
    )
    atr_wilder = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_wilder
    min_di = 100 * min_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_wilder
    dx = 100 * (plus_di - min_di).abs() / (plus_di + min_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    out["ADX14"] = adx

    # Relatieve sterkte 20d (t.o.v. index)
    if index_ret is not None:
        aandeel_ret_20 = close.pct_change(20)
        index_ret_20 = index_ret.reindex(df.index).pct_change(20).fillna(0)
        out["REL_STERKTE_20D"] = (aandeel_ret_20 - index_ret_20) * 100
    else:
        out["REL_STERKTE_20D"] = None

    # HV20, HV60 (geannualiseerd)
    log_ret = np.log(close / close.shift(1))
    out["HV20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    out["HV60"] = log_ret.rolling(60).std() * np.sqrt(252) * 100

    # Dagen sinds 52w low
    low52w = low.rolling(252).min()
    dagen_low = []
    for i in range(len(df)):
        if i < 252:
            dagen_low.append(None)
            continue
        venster = low.iloc[i-251:i+1]
        idx_low = venster.idxmin()
        dagen = (df.index[i] - idx_low).days
        dagen_low.append(dagen)
    out["DAGEN_SINDS_LOW52W"] = dagen_low

    # Volume ratio 50d
    vol_avg_50 = volume.rolling(50).mean()
    out["VOL_RATIO_50D"] = volume / vol_avg_50

    # --- EMA8, EMA20 en afgeleiden -------------------------------------
    try:
        ema8_reeks = close.ewm(span=8, adjust=False).mean()
        ema20_reeks = close.ewm(span=20, adjust=False).mean()

        out["EMA8"] = ema8_reeks
        out["EMA20"] = ema20_reeks
        out["PCT_FROM_EMA8"] = ((close - ema8_reeks) / ema8_reeks) * 100
        out["PCT_FROM_EMA20"] = ((close - ema20_reeks) / ema20_reeks) * 100
        out["EMA8_MINUS_EMA20"] = ema8_reeks - ema20_reeks
    except Exception:
        out["EMA8"] = None
        out["EMA20"] = None
        out["PCT_FROM_EMA8"] = None
        out["PCT_FROM_EMA20"] = None
        out["EMA8_MINUS_EMA20"] = None

    return out


# --------------------------------------------------------------------------
# DB-write
# --------------------------------------------------------------------------
INSERT_QUERY = """
INSERT INTO generieke_technicals (
    ticker, datum,
    atr14, atr14_pct, rsi14, ibs,
    ma50, ma200, pct_from_ma50, pct_from_ma200,
    vol_ratio_20d, high52w, pct_from_high52w,
    macd, macd_signaal, macd_hist,
    bb_breedte, bb_percent_b,
    stoch_k, stoch_d, adx14,
    rel_sterkte_20d, hv20, hv60,
    dagen_sinds_low52w, vol_ratio_50d,
    ema8, ema20, pct_from_ema8, pct_from_ema20, ema8_minus_ema20
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (ticker, datum) DO UPDATE SET
    atr14 = EXCLUDED.atr14,
    atr14_pct = EXCLUDED.atr14_pct,
    rsi14 = EXCLUDED.rsi14,
    ibs = EXCLUDED.ibs,
    ma50 = EXCLUDED.ma50,
    ma200 = EXCLUDED.ma200,
    pct_from_ma50 = EXCLUDED.pct_from_ma50,
    pct_from_ma200 = EXCLUDED.pct_from_ma200,
    vol_ratio_20d = EXCLUDED.vol_ratio_20d,
    high52w = EXCLUDED.high52w,
    pct_from_high52w = EXCLUDED.pct_from_high52w,
    macd = EXCLUDED.macd,
    macd_signaal = EXCLUDED.macd_signaal,
    macd_hist = EXCLUDED.macd_hist,
    bb_breedte = EXCLUDED.bb_breedte,
    bb_percent_b = EXCLUDED.bb_percent_b,
    stoch_k = EXCLUDED.stoch_k,
    stoch_d = EXCLUDED.stoch_d,
    adx14 = EXCLUDED.adx14,
    rel_sterkte_20d = EXCLUDED.rel_sterkte_20d,
    hv20 = EXCLUDED.hv20,
    hv60 = EXCLUDED.hv60,
    dagen_sinds_low52w = EXCLUDED.dagen_sinds_low52w,
    vol_ratio_50d = EXCLUDED.vol_ratio_50d,
    ema8 = EXCLUDED.ema8,
    ema20 = EXCLUDED.ema20,
    pct_from_ema8 = EXCLUDED.pct_from_ema8,
    pct_from_ema20 = EXCLUDED.pct_from_ema20,
    ema8_minus_ema20 = EXCLUDED.ema8_minus_ema20;
"""


def schrijf_rij(conn, ticker, datum, rij):
    waarden = [
        ticker, datum,
        _veilig(rij.get("ATR14")),
        _veilig(rij.get("ATR14_PCT")),
        _veilig(rij.get("RSI14")),
        _veilig(rij.get("IBS")),
        _veilig(rij.get("MA50")),
        _veilig(rij.get("MA200")),
        _veilig(rij.get("PCT_FROM_MA50")),
        _veilig(rij.get("PCT_FROM_MA200")),
        _veilig(rij.get("VOL_RATIO_20D")),
        _veilig(rij.get("HIGH52W")),
        _veilig(rij.get("PCT_FROM_HIGH52W")),
        _veilig(rij.get("MACD")),
        _veilig(rij.get("MACD_SIGNAAL")),
        _veilig(rij.get("MACD_HIST")),
        _veilig(rij.get("BB_BREEDTE")),
        _veilig(rij.get("BB_PERCENT_B")),
        _veilig(rij.get("STOCH_K")),
        _veilig(rij.get("STOCH_D")),
        _veilig(rij.get("ADX14")),
        _veilig(rij.get("REL_STERKTE_20D")),
        _veilig(rij.get("HV20")),
        _veilig(rij.get("HV60")),
        _veilig(rij.get("DAGEN_SINDS_LOW52W"), is_int=True),
        _veilig(rij.get("VOL_RATIO_50D")),
        _veilig(rij.get("EMA8")),
        _veilig(rij.get("EMA20")),
        _veilig(rij.get("PCT_FROM_EMA8")),
        _veilig(rij.get("PCT_FROM_EMA20")),
        _veilig(rij.get("EMA8_MINUS_EMA20")),
    ]
    with conn.cursor() as cur:
        cur.execute(INSERT_QUERY, waarden)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not DB_URL:
        sys.exit("SUPABASE_DB_URL ontbreekt")

    datum_doel = os.environ.get("BACKFILL_DATUM") or dt.date.today().isoformat()
    datum_doel_ts = pd.Timestamp(datum_doel).normalize()

    tickers = alle_tickers()
    if not tickers:
        sys.exit("Geen tickers gevonden in tickers_*.txt")

    print(f"Tickers: {len(tickers)}")
    print(f"Doeldatum: {datum_doel_ts.date()}")

    start = (datum_doel_ts - pd.Timedelta(days=LOOKBACK_BUFFER_DAGEN)).strftime("%Y-%m-%d")
    index_ret = download_index_returns(start)

    conn = psycopg2.connect(DB_URL)
    n_ok = 0
    n_skip = 0
    fouten = []

    try:
        for i, ticker in enumerate(tickers, 1):
            if "/" in ticker:
                continue
            try:
                hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
                if hist is None or hist.empty:
                    n_skip += 1
                    continue

                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                g = bereken_indicatoren(hist, index_ret)
                if g.empty:
                    n_skip += 1
                    continue

                # Zoek bar op of vóór doeldatum
                pos = g.index.searchsorted(datum_doel_ts, side="right") - 1
                if pos < 0:
                    n_skip += 1
                    continue
                if (datum_doel_ts - g.index[pos]).days > MAX_BAR_AFSTAND_DAGEN:
                    n_skip += 1
                    continue

                rij = g.iloc[pos]
                if pd.isna(rij.get("Close")) if "Close" in rij else False:
                    n_skip += 1
                    continue

                schrijf_rij(conn, ticker, datum_doel_ts.date(), rij)
                n_ok += 1

                if i % 100 == 0:
                    conn.commit()
                    print(f"  [{i}/{len(tickers)}] {n_ok} geschreven")
                time.sleep(0.15)

            except Exception as e:
                fouten.append((ticker, str(e)))

        conn.commit()
    finally:
        conn.close()

    print()
    print("=" * 60)
    print("SAMENVATTING")
    print("=" * 60)
    print(f"Tickers verwerkt : {len(tickers)}")
    print(f"Rijen geschreven : {n_ok}")
    print(f"Overgeslagen     : {n_skip}")
    if fouten:
        print(f"Fouten           : {len(fouten)}")
        for t, e in fouten[:10]:
            print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
