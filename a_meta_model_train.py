#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_meta_model_train.py — traint het meta-model op ALLE beschikbare data en
schrijft scaler + coëfficiënten naar Supabase (tabel meta_model_models).

Dit is het productie-model. Anders dan meta_model.py (walk-forward
evaluatie) wordt hier geen fold-split gedaan: alle rijen met een
beschikbare target gaan in de training. De uitkomst is een versie-string
die scoring gebruikt om consistent te blijven.

Env vars: SUPABASE_DB_URL (verplicht)
"""

import os
import sys
import argparse
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import MODEL_FEATURES, MODEL_TARGET, MODEL_HORIZON

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

GENERIEKE_TECHNICALS = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]


def haal_training_data(conn, features):
    s = [f for f in features if f not in GENERIEKE_TECHNICALS]
    g = [f for f in features if f in GENERIEKE_TECHNICALS]
    s_lijst = ", ".join(f"s.{k}" for k in s)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g)) if g else ""

    query = f"""
        SELECT s.datum, {s_lijst}{g_lijst},
               f.{MODEL_TARGET} AS target
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE f.{MODEL_TARGET} IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return pd.DataFrame(cur.fetchall())


def cross_sectionele_rank(df, kolommen):
    out = df.copy()
    for k in kolommen:
        out[k] = out.groupby("datum")[k].rank(pct=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularisatie")
    parser.add_argument("--notities", type=str, default="",
                        help="Vrije tekst om aan de model-versie te hangen")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    features = list(MODEL_FEATURES)
    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = haal_training_data(conn, features)
        print(f"{len(df)} rijen opgehaald.")
        if df.empty:
            print("Geen training-data.")
            return

        df = cross_sectionele_rank(df, features).dropna(subset=features + ["target"])
        print(f"{len(df)} rijen na dropna, {df['datum'].nunique()} unieke datums.")

        X = df[features].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=args.alpha)
        model.fit(X_scaled, y)

        versie = f"ridge_a{args.alpha}_h{MODEL_HORIZON}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

        coefs = [float(c) for c in model.coef_]
        mean = [float(m) for m in scaler.mean_]
        std = [float(s) for s in scaler.scale_]

        insert = """
            INSERT INTO meta_model_models
                (versie, horizon_dagen, features, coefs, scaler_mean, scaler_std,
                 ridge_alpha, n_train_rijen, n_train_datums, getraind_op, notities)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with conn.cursor() as cur:
            cur.execute(insert, (
                versie, MODEL_HORIZON, features, coefs, mean, std,
                args.alpha, len(df), int(df["datum"].nunique()),
                datetime.now(timezone.utc), args.notities or None,
            ))
        conn.commit()

        print(f"\nModel opgeslagen: {versie}")
        print(f"Features: {', '.join(features)}")
        print("Coëfficiënten:")
        for f, c in zip(features, coefs):
            print(f"  {f:<24} {c:+.4f}")
        print(f"Intercept: {model.intercept_:+.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
