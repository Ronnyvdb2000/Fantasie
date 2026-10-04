#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_meta_model_score.py — berekent meta-model scores voor de laatste
selectie-datum (of een opgegeven datum) en schrijft ze naar
meta_model_scores.

Gebruikt het nieuwste model uit meta_model_models. Past dezelfde
cross-sectionele ranking en scaler toe als tijdens training.

Env vars: SUPABASE_DB_URL (verplicht)
"""

import os
import sys
import argparse
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

from features import MODEL_HORIZON

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

GENERIEKE_TECHNICALS = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]


def haal_laatste_model(conn):
    query = """
        SELECT versie, horizon_dagen, features, coefs, scaler_mean, scaler_std, ridge_alpha
        FROM meta_model_models
        ORDER BY getraind_op DESC
        LIMIT 1;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        row = cur.fetchone()
    return row


def haal_selecties(conn, features, datum=None):
    """Haal selecties op voor een specifieke datum, of de laatste datum
    waarvoor selecties bestaan zonder score."""
    s = [f for f in features if f not in GENERIEKE_TECHNICALS]
    g = [f for f in features if f in GENERIEKE_TECHNICALS]
    s_lijst = ", ".join(f"s.{k}" for k in s)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g)) if g else ""

    if datum is None:
        # Laatste selectie-datum
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(datum) FROM selecties;")
            datum = cur.fetchone()[0]
    print(f"Scoren voor datum: {datum}")

    query = f"""
        SELECT s.ticker, s.datum, s.strategie, {s_lijst}{g_lijst}
        FROM selecties s
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE s.datum = %s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (datum,))
        return pd.DataFrame(cur.fetchall()), datum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datum", type=str, default=None,
                        help="Specifieke datum (YYYY-MM-DD). Default: laatste selectiedatum.")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        model_row = haal_laatste_model(conn)
        if model_row is None:
            print("Geen model gevonden. Draai eerst a_meta_model_train.py.")
            sys.exit(1)

        versie = model_row["versie"]
        features = list(model_row["features"])
        coefs = np.array(model_row["coefs"], dtype=float)
        mean = np.array(model_row["scaler_mean"], dtype=float)
        std = np.array(model_row["scaler_std"], dtype=float)

        print(f"Model: {versie}")
        print(f"Horizon: {model_row['horizon_dagen']}d, features: {len(features)}")

        df, datum = haal_selecties(conn, features, args.datum)
        print(f"{len(df)} selecties opgehaald.")
        if df.empty:
            print("Geen selecties voor die datum.")
            return

        # Cross-sectionele rank per datum (maar hier is het 1 datum)
        for k in features:
            df[k] = df[k].rank(pct=True)

        df = df.dropna(subset=features)
        print(f"{len(df)} rijen met complete features.")

        X = (df[features].values - mean) / std
        scores = X @ coefs

        df["score"] = scores

        insert = """
            INSERT INTO meta_model_scores
                (datum, ticker, strategie, score, model_versie, horizon_dagen)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (datum, ticker, strategie, model_versie, horizon_dagen)
            DO UPDATE SET score = EXCLUDED.score, created_at = now();
        """
        rijen = [
            (datum, r["ticker"], r["strategie"], float(r["score"]), versie, MODEL_HORIZON)
            for _, r in df.iterrows()
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert, rijen, page_size=500)
        conn.commit()
        print(f"{len(rijen)} scores opgeslagen in meta_model_scores.")

        # Toon top 10 als sanity check
        top = df.nlargest(10, "score")[["ticker", "strategie", "score"]]
        print("\nTop 10 scores:")
        print(top.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
