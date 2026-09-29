#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lightgbm_predictor.py
=====================

Actuele voorspeller voor LightGBM V1.

GEBRUIK
-------
Laadt:

    results/lightgbmV1_10d_model.pkl
    results/lightgbmV1_30d_model.pkl
    results/lightgbmV1_60d_model.pkl

en gebruikt de meest recente technische gegevens uit:

    generieke_technicals

voor ALLE beschikbare tickers.

OUTPUT
------
results/lightgbmV1_current_predictions.csv

Kolommen:

    ticker
    datum
    prob_10d
    prob_30d
    prob_60d
    avg_probability
    score
    rank

INTERPRETATIE
-------------
prob_10d = kans dat 10d forward return > 0
prob_30d = kans dat 30d forward return > 0
prob_60d = kans dat 60d forward return > 0

avg_probability =
    gemiddelde van de drie modelkansen

score =
    gemiddelde kans * 100

Dit is GEEN gegarandeerd rendement.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = "results"

MODEL_VERSIE = "lightgbmV1"

HORIZONS = [
    "10d",
    "30d",
    "60d",
]


FEATURE_COLUMNS = [
    "atr14",
    "atr14_pct",
    "rsi14",
    "ibs",
    "ma50",
    "ma200",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "high52w",
    "pct_from_high52w",
]


OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "lightgbmV1_current_predictions.csv",
)


# ============================================================
# DATABASE QUERY
# ============================================================

CURRENT_DATA_QUERY = """
SELECT
    ticker,
    datum,

    atr14,
    atr14_pct,
    rsi14,
    ibs,

    ma50,
    ma200,

    pct_from_ma50,
    pct_from_ma200,

    vol_ratio_20d,

    high52w,
    pct_from_high52w

FROM generieke_technicals

WHERE datum IS NOT NULL

ORDER BY
    ticker,
    datum DESC;
"""


# ============================================================
# DATA OPHALEN
# ============================================================

def load_current_data(conn):

    print()
    print("=" * 70)
    print("ACTUELE TECHNISCHE DATA OPHALEN")
    print("=" * 70)

    df = pd.read_sql(
        CURRENT_DATA_QUERY,
        conn,
    )

    if df.empty:

        raise RuntimeError(
            "Geen gegevens gevonden in "
            "generieke_technicals."
        )

    df["datum"] = pd.to_datetime(
        df["datum"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["datum"]
    )

    # Omdat de query per ticker aflopend is:
    # eerste regel = meest recente.
    df = (
        df.sort_values(
            [
                "ticker",
                "datum",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["ticker"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Historische regels gelezen : "
        f"{len(df):,}"
    )

    print(
        f"Actuele tickers             : "
        f"{df['ticker'].nunique():,}"
    )

    print(
        f"Laatste datum                : "
        f"{df['datum'].max()}"
    )

    return df


# ============================================================
# MODELLEN LADEN
# ============================================================

def load_models():

    models = {}

    print()
    print("=" * 70)
    print("LIGHTGBM-MODELLEN LADEN")
    print("=" * 70)

    for horizon in HORIZONS:

        filename = os.path.join(
            RESULTS_DIR,
            f"{MODEL_VERSIE}_{horizon}_model.pkl",
        )

        if not os.path.isfile(
            filename
        ):

            raise FileNotFoundError(
                f"Model ontbreekt: "
                f"{filename}"
            )

        model = joblib.load(
            filename
        )

        models[horizon] = model

        print(
            f"OK {horizon}: "
            f"{filename}"
        )

    return models


# ============================================================
# FEATURES CONTROLEREN
# ============================================================

def prepare_features(df):

    work = df.copy()

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in work.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Ontbrekende featurekolommen: "
            + ", ".join(
                missing_columns
            )
        )

    for feature in FEATURE_COLUMNS:

        work[feature] = pd.to_numeric(
            work[feature],
            errors="coerce",
        )

    return work


# ============================================================
# VOORSPELLEN
# ============================================================

def predict_models(
    df,
    models,
):

    work = prepare_features(
        df
    )

    print()
    print("=" * 70)
    print("VOORSPELLINGEN MAKEN")
    print("=" * 70)

    valid_mask = (
        work[
            FEATURE_COLUMNS
        ]
        .notna()
        .all(axis=1)
    )

    valid = work[
        valid_mask
    ].copy()

    invalid = work[
        ~valid_mask
    ].copy()

    print(
        f"Tickers met complete data : "
        f"{len(valid):,}"
    )

    print(
        f"Tickers overgeslagen       : "
        f"{len(invalid):,}"
    )

    if valid.empty:

        raise RuntimeError(
            "Geen enkele ticker heeft "
            "alle benodigde features."
        )

    X = valid[
        FEATURE_COLUMNS
    ]

    # --------------------------------------------------------
    # MODEL PER HORIZON
    # --------------------------------------------------------

    for horizon in HORIZONS:

        model = models[
            horizon
        ]

        probabilities = (
            model.predict_proba(
                X
            )[:, 1]
        )

        valid[
            f"prob_{horizon}"
        ] = probabilities

    # --------------------------------------------------------
    # GEMIDDELDE
    # --------------------------------------------------------

    probability_columns = [
        f"prob_{horizon}"
        for horizon in HORIZONS
    ]

    valid[
        "avg_probability"
    ] = valid[
        probability_columns
    ].mean(
        axis=1
    )

    valid[
        "score"
    ] = (
        valid[
            "avg_probability"
        ] * 100
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    valid = valid.sort_values(
        [
            "avg_probability",
            "prob_30d",
            "prob_10d",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    valid[
        "rank"
    ] = (
        valid.index + 1
    )

    return (
        valid,
        invalid,
    )


# ============================================================
# OUTPUT OPSLAAN
# ============================================================

def save_predictions(
    predictions,
    invalid,
):

    output_columns = [
        "rank",
        "ticker",
        "datum",

        "prob_10d",
        "prob_30d",
        "prob_60d",

        "avg_probability",
        "score",

        "rsi14",
        "ibs",
        "atr14_pct",

        "pct_from_ma50",
        "pct_from_ma200",

        "vol_ratio_20d",

        "pct_from_high52w",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in predictions.columns
    ]

    predictions[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Voorspellingen opgeslagen:"
        f"\n  {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # OVERGESLAGEN TICKERS
    # --------------------------------------------------------

    if not invalid.empty:

        skipped_file = os.path.join(
            RESULTS_DIR,
            "lightgbmV1_skipped_tickers.csv",
        )

        invalid[
            [
                "ticker",
                "datum",
            ]
        ].to_csv(
            skipped_file,
            index=False,
        )

        print(
            f"Overgeslagen tickers:"
            f"\n  {skipped_file}"
        )


# ============================================================
# TOP 25 PRINTEN
# ============================================================

def print_top_predictions(
    predictions
):

    print()
    print("=" * 90)
    print("TOP 25 LIGHTGBM VOORSPELLINGEN")
    print("=" * 90)

    top = predictions.head(
        25
    )

    print(
        f"{'RANK':<6}"
        f"{'TICKER':<16}"
        f"{'10D':>9}"
        f"{'30D':>9}"
        f"{'60D':>9}"
        f"{'GEM.':>9}"
        f"{'SCORE':>9}"
    )

    print(
        "-" * 90
    )

    for _, row in top.iterrows():

        print(
            f"{int(row['rank']):<6}"
            f"{str(row['ticker']):<16}"
            f"{row['prob_10d'] * 100:>8.2f}%"
            f"{row['prob_30d'] * 100:>8.2f}%"
            f"{row['prob_60d'] * 100:>8.2f}%"
            f"{row['avg_probability'] * 100:>8.2f}%"
            f"{row['score']:>8.2f}"
        )


# ============================================================
# SUMMARY
# ============================================================

def save_summary(
    predictions,
    invalid,
):

    filename = os.path.join(
        RESULTS_DIR,
        "lightgbmV1_predictor_summary.txt",
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "LIGHTGBMV1 CURRENT PREDICTIONS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Aantal voorspellingen: "
            f"{len(predictions)}\n"
        )

        f.write(
            f"Overgeslagen: "
            f"{len(invalid)}\n"
        )

        if not predictions.empty:

            f.write(
                f"Laatste technische datum: "
                f"{predictions['datum'].max()}\n\n"
            )

            f.write(
                "TOP 25\n"
            )

            f.write(
                "-" * 70 + "\n"
            )

            for _, row in predictions.head(
                25
            ).iterrows():

                f.write(
                    f"{int(row['rank']):3d} "
                    f"{row['ticker']:<15} "
                    f"10d={row['prob_10d'] * 100:6.2f}% "
                    f"30d={row['prob_30d'] * 100:6.2f}% "
                    f"60d={row['prob_60d'] * 100:6.2f}% "
                    f"gem={row['avg_probability'] * 100:6.2f}%\n"
                )

    print(
        f"Predictor summary opgeslagen:"
        f"\n  {filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "LIGHTGBMV1 — ACTUELE VOORSPELLER"
    )
    print("=" * 70)

    db_url = os.environ.get(
        "SUPABASE_DB_URL"
    )

    if not db_url:

        raise RuntimeError(
            "SUPABASE_DB_URL ontbreekt."
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # MODELLEN
    # --------------------------------------------------------

    models = load_models()

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    print()
    print(
        "Verbinding maken met Supabase..."
    )

    conn = psycopg2.connect(
        db_url
    )

    try:

        df = load_current_data(
            conn
        )

    finally:

        conn.close()

    # --------------------------------------------------------
    # PREDICTIES
    # --------------------------------------------------------

    (
        predictions,
        invalid,
    ) = predict_models(
        df,
        models,
    )

    # --------------------------------------------------------
    # OPSLAAN
    # --------------------------------------------------------

    save_predictions(
        predictions,
        invalid,
    )

    save_summary(
        predictions,
        invalid,
    )

    # --------------------------------------------------------
    # TOP 25
    # --------------------------------------------------------

    print_top_predictions(
        predictions
    )

    # --------------------------------------------------------
    # VALIDATIE
    # --------------------------------------------------------

    if predictions.empty:

        raise RuntimeError(
            "Predictor heeft geen "
            "voorspellingen geproduceerd."
        )

    print()
    print("=" * 70)
    print(
        "LIGHTGBMV1 PREDICTOR VOLTOOID"
    )
    print("=" * 70)

    print(
        f"Voorspelde tickers : "
        f"{len(predictions):,}"
    )

    print(
        f"Overgeslagen       : "
        f"{len(invalid):,}"
    )

    print(
        f"CSV                : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()
