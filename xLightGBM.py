#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lightgbmV1.py
=============

LightGBM trainingsengine gebaseerd op de methodiek van xgboostV3.

DOEL
----
Leert uit ALLE historische regels uit:

    generieke_technicals
    +
    forward_returns

Er wordt NIET vooraf gefilterd op Nitro-aandelen.

Per horizon wordt een afzonderlijk LightGBM-classificatiemodel
getraind:

    lightgbmV1_10d_model.pkl
    lightgbmV1_30d_model.pkl
    lightgbmV1_60d_model.pkl

TARGET
------
1 = forward return > 0
0 = forward return <= 0

EVALUATIE
---------
- chronologische 80/20 split
- Accuracy
- AUC
- top 20% volgens model
- top 20% volgens baseline pct_from_ma50
- gemiddelde volledige testset
- schaalvrij controlemodel
- Pearson correlatie
- Spearman correlatie
- LightGBM feature importance

BELANGRIJK
----------
De correlatieanalyse gebruikt alleen de TRAINING-set.

FEATURES
--------
Volledig model:

    atr14
    atr14_pct
    rsi14
    ibs
    ma50
    ma200
    pct_from_ma50
    pct_from_ma200
    vol_ratio_20d
    high52w
    pct_from_high52w

Controlemodel:

    atr14_pct
    rsi14
    ibs
    pct_from_ma50
    pct_from_ma200
    vol_ratio_20d
    pct_from_high52w

DATABASE
--------
Environment variable:

    SUPABASE_DB_URL

OUTPUT
------
results/
    lightgbmV1_10d_model.pkl
    lightgbmV1_30d_model.pkl
    lightgbmV1_60d_model.pkl

    lightgbmV1_10d_feature_importance.csv
    lightgbmV1_30d_feature_importance.csv
    lightgbmV1_60d_feature_importance.csv

    lightgbmV1_10d_correlations.csv
    lightgbmV1_30d_correlations.csv
    lightgbmV1_60d_correlations.csv

    lightgbmV1_10d_test_predictions.csv
    lightgbmV1_30d_test_predictions.csv
    lightgbmV1_60d_test_predictions.csv

    lightgbmV1_summary.txt
"""

import os
import warnings
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import psycopg2
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATIE
# ============================================================

MODEL_VERSIE = "lightgbmV1"

HORIZONS = [
    "10d",
    "30d",
    "60d",
]

TOP_N_FRACTIE = 0.20

MIN_RIJEN_TRAINING = 100
MIN_RIJEN_TEST = 30

RANDOM_STATE = 42

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# FEATURES
# ============================================================

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


# Schaalvrije controlefeatures.
#
# Absolute prijsniveaus worden hier verwijderd.
#
# Hierdoor kunnen we controleren of het volledige model
# sterk afhankelijk is van absolute prijsinformatie.

FEATURE_COLUMNS_RELATIEF = [
    "atr14_pct",
    "rsi14",
    "ibs",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "pct_from_high52w",
]


BASELINE_KOLOM = "pct_from_ma50"

AUC_VERSCHIL_WAARSCHUWING = 0.10


# ============================================================
# DATABASE QUERY
# ============================================================

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker,
        datum,
        fwd_ret_10d,
        fwd_ret_30d,
        fwd_ret_60d
    FROM forward_returns
    ORDER BY ticker, datum
)

SELECT
    gt.ticker,
    gt.datum,

    gt.atr14,
    gt.atr14_pct,
    gt.rsi14,
    gt.ibs,

    gt.ma50,
    gt.ma200,

    gt.pct_from_ma50,
    gt.pct_from_ma200,

    gt.vol_ratio_20d,

    gt.high52w,
    gt.pct_from_high52w,

    fr_dedup.fwd_ret_10d,
    fr_dedup.fwd_ret_30d,
    fr_dedup.fwd_ret_60d

FROM generieke_technicals gt

JOIN fr_dedup
    ON fr_dedup.ticker = gt.ticker
    AND fr_dedup.datum = gt.datum

ORDER BY
    gt.datum,
    gt.ticker;
"""


# ============================================================
# HULPFUNCTIES
# ============================================================

def is_nan(value):

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def fmt(value, decimals=3):

    if is_nan(value):
        return "n.v.t."

    return f"{float(value):.{decimals}f}"


def pct(value, decimals=2):

    if is_nan(value):
        return "n.v.t."

    return f"{float(value) * 100:.{decimals}f}%"


def safe_float(value):

    if is_nan(value):
        return None

    return float(value)


# ============================================================
# DATA OPHALEN
# ============================================================

def get_training_data(conn):

    print()
    print("=" * 70)
    print("DATASET OPHALEN")
    print("=" * 70)

    df = pd.read_sql(
        JOIN_QUERY,
        conn,
    )

    if df.empty:

        raise RuntimeError(
            "Supabase gaf 0 rijen terug uit "
            "generieke_technicals + forward_returns."
        )

    df["datum"] = pd.to_datetime(
        df["datum"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["datum"]
    )

    print(
        f"Rijen opgehaald : {len(df):,}"
    )

    print(
        f"Tickers         : "
        f"{df['ticker'].nunique():,}"
    )

    print(
        f"Van             : "
        f"{df['datum'].min()}"
    )

    print(
        f"Tot             : "
        f"{df['datum'].max()}"
    )

    return df


# ============================================================
# DATA VOORBEREIDEN
# ============================================================

def prepare_horizon_data(
    df,
    horizon,
):

    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:

        raise RuntimeError(
            f"Doelkolom ontbreekt: {target_column}"
        )

    work = df.copy()

    work["datum"] = pd.to_datetime(
        work["datum"],
        errors="coerce",
    )

    # Eén observatie per ticker/datum.
    work = work.drop_duplicates(
        subset=[
            "ticker",
            "datum",
        ],
        keep="first",
    )

    # Forward return numeriek maken.
    work[target_column] = pd.to_numeric(
        work[target_column],
        errors="coerce",
    )

    # Eerst target verwijderen indien ontbreekt.
    work = work.dropna(
        subset=[
            target_column,
            "datum",
        ]
    )

    # Target:
    #
    # positief toekomstig rendement = 1
    # niet-positief = 0
    work["is_profitable"] = (
        work[target_column] > 0
    ).astype(int)

    # Features numeriek maken.
    for feature in FEATURE_COLUMNS:

        work[feature] = pd.to_numeric(
            work[feature],
            errors="coerce",
        )

    # Ontbrekende features verwijderen.
    work = work.dropna(
        subset=FEATURE_COLUMNS
    )

    work = work.sort_values(
        [
            "datum",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    return work


# ============================================================
# CHRONOLOGISCHE SPLIT
# ============================================================

def time_split(df):

    if len(df) < (
        MIN_RIJEN_TRAINING
        + MIN_RIJEN_TEST
    ):

        raise RuntimeError(
            f"Te weinig bruikbare rijen: "
            f"{len(df)}. "
            f"Minimaal "
            f"{MIN_RIJEN_TRAINING + MIN_RIJEN_TEST} "
            f"vereist."
        )

    dates = (
        df["datum"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < 2:

        raise RuntimeError(
            "Er zijn onvoldoende "
            "verschillende handelsdatums."
        )

    split_index = max(
        0,
        int(len(dates) * 0.80) - 1,
    )

    split_date = dates.iloc[
        split_index
    ]

    train_df = df[
        df["datum"] <= split_date
    ].copy()

    test_df = df[
        df["datum"] > split_date
    ].copy()

    if len(train_df) < MIN_RIJEN_TRAINING:

        raise RuntimeError(
            f"Trainingsset te klein: "
            f"{len(train_df)}."
        )

    if len(test_df) < MIN_RIJEN_TEST:

        raise RuntimeError(
            f"Testset te klein: "
            f"{len(test_df)}."
        )

    if train_df["is_profitable"].nunique() < 2:

        raise RuntimeError(
            "Trainingsset bevat slechts "
            "één klasse."
        )

    return (
        train_df,
        test_df,
        split_date,
    )


# ============================================================
# CORRELATIE
# ============================================================

def calculate_correlations(
    train_df,
    target_column,
    horizon,
):

    print()
    print(
        f"[{horizon}] "
        "Correlatieanalyse op TRAINING-set..."
    )

    rows = []

    target = pd.to_numeric(
        train_df[target_column],
        errors="coerce",
    )

    for feature in FEATURE_COLUMNS:

        x = pd.to_numeric(
            train_df[feature],
            errors="coerce",
        )

        valid = pd.concat(
            [
                x,
                target,
            ],
            axis=1,
        ).dropna()

        if len(valid) < 10:

            pearson = np.nan
            spearman = np.nan

        else:

            pearson = valid.iloc[:, 0].corr(
                valid.iloc[:, 1],
                method="pearson",
            )

            spearman = valid.iloc[:, 0].corr(
                valid.iloc[:, 1],
                method="spearman",
            )

        rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": (
                    abs(pearson)
                    if not pd.isna(pearson)
                    else np.nan
                ),
                "abs_spearman": (
                    abs(spearman)
                    if not pd.isna(spearman)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "abs_spearman",
        ascending=False,
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"{MODEL_VERSIE}_{horizon}_correlations.csv",
    )

    result.to_csv(
        filename,
        index=False,
    )

    print()
    print(
        f"[{horizon}] Sterkste correlaties:"
    )

    for _, row in result.head(10).iterrows():

        print(
            f"  {row['feature']:<22} "
            f"Pearson {fmt(row['pearson'])} | "
            f"Spearman {fmt(row['spearman'])}"
        )

    return result


# ============================================================
# LIGHTGBM MODEL
# ============================================================

def create_lightgbm_model():

    return lgb.LGBMClassifier(

        objective="binary",

        boosting_type="gbdt",

        n_estimators=400,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=6,

        min_child_samples=40,

        subsample=0.80,

        subsample_freq=1,

        colsample_bytree=0.80,

        reg_alpha=0.10,

        reg_lambda=0.50,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1,

        importance_type="gain",
    )


# ============================================================
# MODEL TRAINEN
# ============================================================

def train_model(
    train_df,
    test_df,
    features,
    target_column,
):

    model = create_lightgbm_model()

    X_train = train_df[
        features
    ]

    y_train = train_df[
        "is_profitable"
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        "is_profitable"
    ]

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    if y_test.nunique() >= 2:

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

    else:

        auc = np.nan

    result = test_df.copy()

    result[
        "model_probability"
    ] = probabilities

    result[
        "model_prediction"
    ] = predictions

    return (
        model,
        result,
        accuracy,
        auc,
    )


# ============================================================
# TOP N GEMIDDELD RENDEMENT
# ============================================================

def top_n_average(
    df,
    sort_column,
    target_column,
    n_top,
    descending=True,
):

    ordered = df.sort_values(
        sort_column,
        ascending=not descending,
    )

    selected = ordered.head(
        n_top
    )

    if selected.empty:

        return np.nan

    return float(
        selected[
            target_column
        ].mean()
    )


# ============================================================
# FEATURE IMPORTANCE OPSLAAN
# ============================================================

def save_feature_importance(
    model,
    features,
    horizon,
):

    importance = pd.DataFrame(
        {
            "feature": features,

            "importance_gain":
                model.booster_.feature_importance(
                    importance_type="gain"
                ),

            "importance_split":
                model.booster_.feature_importance(
                    importance_type="split"
                ),
        }
    )

    importance = importance.sort_values(
        "importance_gain",
        ascending=False,
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"{MODEL_VERSIE}_{horizon}_feature_importance.csv",
    )

    importance.to_csv(
        filename,
        index=False,
    )

    print()
    print(
        f"[{horizon}] Feature importance:"
    )

    for _, row in importance.iterrows():

        print(
            f"  {row['feature']:<22} "
            f"gain={row['importance_gain']:.2f} "
            f"split={int(row['importance_split'])}"
        )

    return importance


# ============================================================
# ÉÉN HORIZON TRAINEN
# ============================================================

def train_horizon(
    df,
    horizon,
):

    print()
    print()
    print("=" * 70)
    print(
        f"LIGHTGBM — HORIZON {horizon}"
    )
    print("=" * 70)

    target_column = (
        f"fwd_ret_{horizon}"
    )

    work = prepare_horizon_data(
        df,
        horizon,
    )

    print(
        f"[{horizon}] "
        f"Bruikbare rijen: "
        f"{len(work):,}"
    )

    print(
        f"[{horizon}] "
        f"Tickers: "
        f"{work['ticker'].nunique():,}"
    )

    train_df, test_df, split_date = (
        time_split(work)
    )

    print(
        f"[{horizon}] TRAIN: "
        f"{len(train_df):,} rijen"
    )

    print(
        f"[{horizon}] TEST : "
        f"{len(test_df):,} rijen"
    )

    print(
        f"[{horizon}] Splitdatum: "
        f"{split_date}"
    )

    # --------------------------------------------------------
    # CORRELATIE
    # --------------------------------------------------------

    calculate_correlations(
        train_df,
        target_column,
        horizon,
    )

    # --------------------------------------------------------
    # VOLLEDIG MODEL
    # --------------------------------------------------------

    print(
        f"\n[{horizon}] "
        "Volledig LightGBM-model trainen..."
    )

    (
        model,
        test_predictions,
        accuracy,
        auc,
    ) = train_model(
        train_df,
        test_df,
        FEATURE_COLUMNS,
        target_column,
    )

    # --------------------------------------------------------
    # TOP 20%
    # --------------------------------------------------------

    n_top = max(
        1,
        int(
            len(test_predictions)
            * TOP_N_FRACTIE
        ),
    )

    top_model = top_n_average(
        test_predictions,
        "model_probability",
        target_column,
        n_top,
        descending=True,
    )

    # --------------------------------------------------------
    # BASELINE
    #
    # De laagste pct_from_ma50 wordt geselecteerd.
    # Dit volgt de XGBoostV3-baseline.
    # --------------------------------------------------------

    top_baseline = top_n_average(
        test_predictions,
        BASELINE_KOLOM,
        target_column,
        n_top,
        descending=False,
    )

    test_average = float(
        test_predictions[
            target_column
        ].mean()
    )

    # --------------------------------------------------------
    # CONTROLEMODEL
    # --------------------------------------------------------

    print(
        f"\n[{horizon}] "
        "Controlemodel trainen..."
    )

    (
        control_model,
        control_predictions,
        control_accuracy,
        control_auc,
    ) = train_model(
        train_df,
        test_df,
        FEATURE_COLUMNS_RELATIEF,
        target_column,
    )

    top_control = top_n_average(
        control_predictions,
        "model_probability",
        target_column,
        n_top,
        descending=True,
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    save_feature_importance(
        model,
        FEATURE_COLUMNS,
        horizon,
    )

    # --------------------------------------------------------
    # MODEL OPSLAAN
    # --------------------------------------------------------

    model_filename = os.path.join(
        RESULTS_DIR,
        f"{MODEL_VERSIE}_{horizon}_model.pkl",
    )

    joblib.dump(
        model,
        model_filename,
    )

    # --------------------------------------------------------
    # TESTPREDICTIES OPSLAAN
    # --------------------------------------------------------

    predictions_filename = os.path.join(
        RESULTS_DIR,
        f"{MODEL_VERSIE}_{horizon}_test_predictions.csv",
    )

    output_columns = [
        "ticker",
        "datum",
        target_column,
        "model_probability",
        "model_prediction",
        BASELINE_KOLOM,
    ]

    test_predictions[
        output_columns
    ].sort_values(
        "model_probability",
        ascending=False,
    ).to_csv(
        predictions_filename,
        index=False,
    )

    # --------------------------------------------------------
    # AUC VERSCHIL
    # --------------------------------------------------------

    if (
        not pd.isna(auc)
        and not pd.isna(control_auc)
    ):

        auc_delta = (
            auc - control_auc
        )

    else:

        auc_delta = np.nan

    # --------------------------------------------------------
    # RESULTAAT PRINTEN
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(
        f"[{horizon}] RESULTAAT"
    )
    print("-" * 70)

    print(
        f"Train              : "
        f"{len(train_df):,}"
    )

    print(
        f"Test               : "
        f"{len(test_df):,}"
    )

    print(
        f"Tickers            : "
        f"{work['ticker'].nunique():,}"
    )

    print(
        f"Accuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"AUC                : "
        f"{fmt(auc)}"
    )

    print(
        f"Controle AUC       : "
        f"{fmt(control_auc)}"
    )

    print(
        f"AUC verschil       : "
        f"{fmt(auc_delta)}"
    )

    print(
        f"Top {n_top} model       : "
        f"{pct(top_model)}"
    )

    print(
        f"Top {n_top} baseline   : "
        f"{pct(top_baseline)}"
    )

    print(
        f"Hele testset       : "
        f"{pct(test_average)}"
    )

    print(
        f"Top {n_top} controle : "
        f"{pct(top_control)}"
    )

    print(
        f"\nModel opgeslagen:"
        f"\n  {model_filename}"
    )

    return {
        "horizon": horizon,
        "status": "getraind",
        "n_total": len(work),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_tickers": work["ticker"].nunique(),
        "split_date": str(split_date),
        "accuracy": accuracy,
        "auc": auc,
        "control_auc": control_auc,
        "auc_delta": auc_delta,
        "top_n": n_top,
        "top_model": top_model,
        "top_baseline": top_baseline,
        "top_control": top_control,
        "test_average": test_average,
        "model_file": model_filename,
    }


# ============================================================
# SUMMARY
# ============================================================

def write_summary(results):

    filename = os.path.join(
        RESULTS_DIR,
        "lightgbmV1_summary.txt",
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "LIGHTGBMV1 TRAINING SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Run: "
            f"{dt.datetime.now().isoformat()}\n\n"
        )

        for result in results:

            f.write(
                f"HORIZON: "
                f"{result['horizon']}\n"
            )

            f.write(
                f"Status: "
                f"{result['status']}\n"
            )

            f.write(
                f"Total: "
                f"{result['n_total']}\n"
            )

            f.write(
                f"Train: "
                f"{result['n_train']}\n"
            )

            f.write(
                f"Test: "
                f"{result['n_test']}\n"
            )

            f.write(
                f"Tickers: "
                f"{result['n_tickers']}\n"
            )

            f.write(
                f"Split: "
                f"{result['split_date']}\n"
            )

            f.write(
                f"Accuracy: "
                f"{safe_float(result['accuracy'])}\n"
            )

            f.write(
                f"AUC: "
                f"{safe_float(result['auc'])}\n"
            )

            f.write(
                f"Control AUC: "
                f"{safe_float(result['control_auc'])}\n"
            )

            f.write(
                f"AUC delta: "
                f"{safe_float(result['auc_delta'])}\n"
            )

            f.write(
                f"Top N: "
                f"{result['top_n']}\n"
            )

            f.write(
                f"Top model return: "
                f"{safe_float(result['top_model'])}\n"
            )

            f.write(
                f"Top baseline return: "
                f"{safe_float(result['top_baseline'])}\n"
            )

            f.write(
                f"Top control return: "
                f"{safe_float(result['top_control'])}\n"
            )

            f.write(
                f"Test average return: "
                f"{safe_float(result['test_average'])}\n"
            )

            f.write(
                f"Model: "
                f"{result['model_file']}\n\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "LIGHTGBMV1 — HISTORISCHE TRAINING"
    )
    print("=" * 70)

    print(
        "Bron: ALLE historische "
        "generieke_technicals-regels"
    )

    print(
        "Horizon: 10d / 30d / 60d"
    )

    print(
        f"Top selectie: "
        f"{TOP_N_FRACTIE * 100:.0f}%"
    )

    # --------------------------------------------------------
    # SUPABASE
    # --------------------------------------------------------

    db_url = os.environ.get(
        "SUPABASE_DB_URL"
    )

    if not db_url:

        raise RuntimeError(
            "SUPABASE_DB_URL ontbreekt."
        )

    print(
        "\nVerbinding maken met Supabase..."
    )

    conn = psycopg2.connect(
        db_url
    )

    try:

        df = get_training_data(
            conn
        )

    finally:

        conn.close()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    results = []

    for horizon in HORIZONS:

        try:

            result = train_horizon(
                df,
                horizon,
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"ERROR [{horizon}]: "
                f"{e}"
            )

            results.append(
                {
                    "horizon": horizon,
                    "status": "fout",
                    "n_total": 0,
                    "n_train": 0,
                    "n_test": 0,
                    "n_tickers": 0,
                    "split_date": "",
                    "accuracy": np.nan,
                    "auc": np.nan,
                    "control_auc": np.nan,
                    "auc_delta": np.nan,
                    "top_n": 0,
                    "top_model": np.nan,
                    "top_baseline": np.nan,
                    "top_control": np.nan,
                    "test_average": np.nan,
                    "model_file": "",
                }
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    write_summary(
        results
    )

    # --------------------------------------------------------
    # CONTROLEREN OF ALLE MODELLEN BESTAAN
    # --------------------------------------------------------

    print()
    print(
        "Controleren of alle modellen "
        "correct zijn opgeslagen..."
    )

    required_models = [
        os.path.join(
            RESULTS_DIR,
            f"{MODEL_VERSIE}_10d_model.pkl",
        ),
        os.path.join(
            RESULTS_DIR,
            f"{MODEL_VERSIE}_30d_model.pkl",
        ),
        os.path.join(
            RESULTS_DIR,
            f"{MODEL_VERSIE}_60d_model.pkl",
        ),
    ]

    missing_models = [
        path
        for path in required_models
        if not os.path.isfile(path)
    ]

    if missing_models:

        print(
            "\nOntbrekende modellen:"
        )

        for path in missing_models:
            print(
                f"  {path}"
            )

        raise RuntimeError(
            "Niet alle drie LightGBM-modellen "
            "zijn aangemaakt."
        )

    print(
        "\nAlle drie modellen zijn aanwezig."
    )

    print()
    print("=" * 70)
    print(
        "LIGHTGBMV1 RUN VOLTOOID"
    )
    print("=" * 70)

    for result in results:

        print(
            f"{result['horizon']:>4} | "
            f"{result['status']:<10} | "
            f"AUC={fmt(result['auc'])} | "
            f"control={fmt(result['control_auc'])}"
        )

    print(
        f"\nOutput: {RESULTS_DIR}/"
    )


if __name__ == "__main__":

    main()
