#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_parameter_ic.py
========================
Bereken voor ELKE parameter de voorspellende waarde voor ELKE horizon
(5d, 10d, 20d, 30d, 60d).

Methode (standaard factor-onderzoek):
  1. Per datum: Spearman rank-correlatie tussen parameter en fwd_ret_Xd
  2. Middel over alle datums -> IC (Information Coefficient)
  3. IC-IR = IC / std(IC) -> hoe consistent werkt de parameter
  4. Hit rate = % datums met het juiste teken

Interpretatie:
  |IC| > 0.03 en |IC-IR| > 0.5   -> bruikbaar
  |IC| > 0.05 en |IC-IR| > 1.0   -> sterk
  Teken: + = hoge waarde goed, - = lage waarde goed

Env:
  SUPABASE_DB_URL
Output:
  results/parameter_ic_matrix.csv   (parameters x horizons)
  Console: top-10 per horizon
"""

import os
import sys
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr


HORIZONS = ["5d", "10d", "20d", "30d", "60d"]
TARGET_KOLOM = {h: f"fwd_ret_{h}" for h in HORIZONS}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TICKERS_PER_DATUM = 20
MIN_DATUMS = 10

NIET_FACTOR = {
    "ticker", "datum", "beurs", "strategie", "bijgewerkt_op", "id",
    "entry_koers", "high52w", "ma50", "ma200", "market_cap",
    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d", "fwd_ret_60d",
    "fwd_close_5d", "fwd_close_10d", "fwd_close_20d",
    "fwd_close_30d", "fwd_close_60d",
}


def get_data(conn) -> pd.DataFrame:
    query = """
    WITH tech AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, atr14, atr14_pct, rsi14, ibs,
            ma50, ma200, pct_from_ma50, pct_from_ma200,
            vol_ratio_20d, high52w, pct_from_high52w,
            macd, macd_signaal, macd_hist, bb_breedte, bb_percent_b,
            stoch_k, stoch_d, adx14, rel_sterkte_20d, hv20, hv60,
            dagen_sinds_low52w, vol_ratio_50d
        FROM generieke_technicals
        ORDER BY ticker, datum
    ),
    fund AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, market_cap, trailing_pe, price_to_book,
            dividend_yield, current_ratio, revenue_growth_pct, fcf_yield,
            net_debt_ebitda, payout_pct, analisten_count,
            piotroski_score, eps_growth_pct, eps_cagr_pct, peg_ratio
        FROM generieke_fundamentals
        ORDER BY ticker, datum DESC
    ),
    fr AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum,
            fwd_ret_5d, fwd_ret_10d, fwd_ret_20d, fwd_ret_30d, fwd_ret_60d
        FROM forward_returns
        ORDER BY ticker, datum
    )
    SELECT
        t.ticker, t.datum,
        t.atr14, t.atr14_pct, t.rsi14, t.ibs,
        t.ma50, t.ma200, t.pct_from_ma50, t.pct_from_ma200,
        t.vol_ratio_20d, t.high52w, t.pct_from_high52w,
        t.macd, t.macd_signaal, t.macd_hist, t.bb_breedte, t.bb_percent_b,
        t.stoch_k, t.stoch_d, t.adx14, t.rel_sterkte_20d, t.hv20, t.hv60,
        t.dagen_sinds_low52w, t.vol_ratio_50d,
        f.market_cap, f.trailing_pe, f.price_to_book, f.dividend_yield,
        f.current_ratio, f.revenue_growth_pct, f.fcf_yield,
        f.net_debt_ebitda, f.payout_pct, f.analisten_count,
        f.piotroski_score, f.eps_growth_pct, f.eps_cagr_pct, f.peg_ratio,
        fr.fwd_ret_5d, fr.fwd_ret_10d, fr.fwd_ret_20d,
        fr.fwd_ret_30d, fr.fwd_ret_60d
    FROM tech t
    LEFT JOIN fund f ON f.ticker = t.ticker AND f.datum = t.datum
    LEFT JOIN fr ON fr.ticker = t.ticker AND fr.datum = t.datum;
    """
    return pd.read_sql(query, conn)


def ic_voor_parameter(df: pd.DataFrame, parameter: str, target: str) -> dict:
    """Bereken IC/IC-IR/hitrate voor 1 parameter op 1 horizon."""
    sub = df[["datum", parameter, target]].dropna()
    if len(sub) < MIN_TICKERS_PER_DATUM:
        return None

    ics = []
    for _, groep in sub.groupby("datum"):
        if len(groep) < MIN_TICKERS_PER_DATUM:
            continue
        rho, _ = spearmanr(groep[parameter], groep[target])
        if not np.isnan(rho):
            ics.append(rho)

    if len(ics) < MIN_DATUMS:
        return None

    ics = np.array(ics)
    ic_gem = float(np.mean(ics))
    ic_std = float(np.std(ics))
    ic_ir = ic_gem / ic_std if ic_std > 1e-9 else 0.0
    hit = float(np.mean(ics > 0)) if ic_gem > 0 else float(np.mean(ics < 0))

    return {
        "ic": round(ic_gem, 4),
        "ic_ir": round(ic_ir, 3),
        "hit": round(hit, 3),
        "n_datums": len(ics),
        "n_obs": len(sub),
    }


def main():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        sys.exit("SUPABASE_DB_URL ontbreekt")

    print("Data ophalen...")
    with psycopg2.connect(db_url) as conn:
        df = get_data(conn)
    print(f"{len(df):,} rijen | {df['datum'].nunique()} datums | "
          f"{df['ticker'].nunique()} tickers\n")

    # Alle numerieke kolommen behalve identifiers
    params = [
        c for c in df.columns
        if c not in NIET_FACTOR and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Matrix: rijen = parameters, kolommen = horizons
    rijen = []
    for p in params:
        rij = {"parameter": p}
        for h in HORIZONS:
            r = ic_voor_parameter(df, p, TARGET_KOLOM[h])
            rij[f"ic_{h}"] = r["ic"] if r else np.nan
            rij[f"ir_{h}"] = r["ic_ir"] if r else np.nan
        rijen.append(rij)

    matrix = pd.DataFrame(rijen)

    # Sorteer op absolute IC op 10d (of eerste horizon met data)
    sort_kolom = "ic_10d"
    matrix["abs_sort"] = matrix[sort_kolom].abs()
    matrix = matrix.sort_values("abs_sort", ascending=False).drop(columns="abs_sort")

    # Opslaan
    out = os.path.join(RESULTS_DIR, "parameter_ic_matrix.csv")
    matrix.to_csv(out, index=False)
    print(f"Opgeslagen: {out}\n")

    # Console: top 15 per horizon
    for h in HORIZONS:
        kolom = f"ic_{h}"
        if kolom not in matrix.columns or matrix[kolom].isna().all():
            print(f"[{h}] geen data\n")
            continue

        sub = matrix.dropna(subset=[kolom]).copy()
        sub["abs"] = sub[kolom].abs()
        top = sub.sort_values("abs", ascending=False).head(15)

        print("=" * 78)
        print(f"TOP 15 PARAMETERS — horizon {h}")
        print("=" * 78)
        print(f"{'Parameter':<22} {'IC':>8} {'IC-IR':>7} {'hit_10d':>8}")
        print("-" * 50)
        for _, r in top.iterrows():
            print(
                f"{r['parameter']:<22} {r[kolom]:>8.4f} "
                f"{r.get(f'ir_{h}', 0):>7.3f} "
                f"{r.get('ic_10d', np.nan):>8.4f}"
            )
        print()

    # Samenvatting: parameters die op ALLE horizons werken
    print("=" * 78)
    print("PARAMETERS MET CONSISTENTE VOORSPELLENDE WAARDE (|IC|>0.02 op alle horizons)")
    print("=" * 78)
    ic_kolommen = [f"ic_{h}" for h in HORIZONS]
    mask = (matrix[ic_kolommen].abs() > 0.02).all(axis=1)
    consistent = matrix[mask]
    if len(consistent) == 0:
        print("  Geen enkele parameter werkt consistent op alle horizons.\n")
    else:
        for _, r in consistent.iterrows():
            waarden = " | ".join(f"{h}={r[f'ic_{h}']:+.3f}" for h in HORIZONS)
            print(f"  {r['parameter']:<22} {waarden}")


if __name__ == "__main__":
    main()
