```python
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

Er wordt NIET eerst gefilterd op de Nitro-aandelenlijst.

Per horizon wordt een afzonderlijk LightGBM-classificatiemodel getraind:

    lightgbmV1_10d_model.pkl
    lightgbmV1_30d_model.pkl
    lightgbmV1_60d_model.pkl

Het model voorspelt:

    1 = forward return > 0
    0 = forward return <= 0

EVALUATIE
---------
- chronologische 80/20 split
- AUC
- accuratesse
- top 20% volgens model
- top 20% volgens baseline pct_from_ma50
- gemiddelde volledige test-set
- controlemodel met enkel schaalvrije features
- Pearson + Spearman correlaties
- LightGBM feature importance

BELANGRIJK
----------
De correlatieanalyse wordt uitsluitend op de TRAINING-set uitgevoerd.
Daarmee voorkomen we dat informatie uit de testperiode de featureselectie
of analyse beïnvloedt.

DATA
----
SUPABASE_DB_URL moet als environment variable aanwezig zijn.

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

    lightgbmV1_summary.txt

De modellen zijn joblib-bestanden zodat ze later door een aparte
voorspellingsbot geladen kunnen worden.
"""

import os
import math
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

HORIZONS = ["10d", "30d", "60d"]

TOP_N_FRACTIE = 0.20

MIN_RIJEN_TRAINING = 100
MIN_RIJEN_TEST = 30

RANDOM_STATE = 42


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


# Controlemodel.
#
# Absolute prijsniveaus worden bewust verwijderd:
#
#   ma50
#   ma200
#   high52w
#   atr14
#
# Hierdoor kunnen we controleren of het volledige model mogelijk
# gedeeltelijk een ticker/prijsklasse herkent.
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
# OUTPUT
# ============================================================

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


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

ORDER BY gt.datum, gt.ticker;
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

    df = pd.read_sql(JOIN_QUERY, conn)

    if df.empty:
        raise RuntimeError(
            "Supabase gaf 0 rijen terug uit "
            "generieke_technicals + forward_returns."
        )

    print(f"Rijen opgehaald : {len(df):,}")
    print(f"Tickers         : {df['ticker'].nunique():,}")

    datum_min = pd.to_datetime(df["datum"]).min()
    datum_max = pd.to_datetime(df["datum"]).max()

    print(f"Van             : {datum_min}")
    print(f"Tot             : {datum_max}")

    return df


# ============================================================
# DATA OPSCHONEN
# ============================================================

def prepare_horizon_data(df, horizon):

    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(
            f"Doelkolom ontbreekt: {target_column}"
        )

    work = df.copy()

    work["datum"] = pd.to_datetime(work["datum"])

    # Heel belangrijk:
    # ticker + datum moet maximaal één observatie bevatten.
    work = work.drop_duplicates(
        subset=["ticker", "datum"],
        keep="first",
    )

    # Binary classification target.
    work["is_profitable"] = (
        pd.to_numeric(
            work[target_column],
            errors="coerce",
        ) > 0
    ).astype("float")

    # Eerst ontbrekende target verwijderen.
    work = work.dropna(
        subset=[target_column, "datum"]
    )

    # Daarna features controleren.
    required = FEATURE_COLUMNS

    work = work.dropna(
        subset=required
    )

    work = work.sort_values(
        ["datum", "ticker"]
    ).reset_index(drop=True)

    work["is_profitable"] = work["is_profitable"].astype(int)

    return work


# ============================================================
# TIJDSGEBASEERDE SPLIT
# ============================================================

def time_split(df):

    if len(df) < MIN_RIJEN_TRAINING + MIN_RIJEN_TEST:
        raise RuntimeError(
            f"Te weinig bruikbare rijen: {len(df)}. "
            f"Minimaal {MIN_RIJEN_TRAINING + MIN_RIJEN_TEST} vereist."
        )

    # Unieke datums bepalen.
    #
    # We splitsen op DATUM, niet op willekeurige rijen.
    # Daardoor komen observaties van dezelfde handelsdag niet
    # gedeeltelijk in train en gedeeltelijk in test terecht.
    dates = (
        df["datum"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < 2:
        raise RuntimeError(
            "Er zijn onvoldoende verschillende handelsdatums."
        )

    split_date = dates.iloc[
        max(0, int(len(dates) * 0.80) - 1)
    ]

    train_df = df[
        df["datum"] <= split_date
    ].copy()

    test_df = df[
        df["datum"] > split_date
    ].copy()

    if len(test_df) < MIN_RIJEN_TEST:
        raise RuntimeError(
            f"Testset te klein: {len(test_df)} "
            f"(minimum {MIN_RIJEN_TEST})."
        )

    if train_df["is_profitable"].nunique() < 2:
        raise RuntimeError(
            "Trainingsset bevat slechts één klasse."
        )

    if test_df["is_profitable"].nunique() < 2:
        print(
            "WAARSCHUWING: testset bevat slechts één klasse. "
            "AUC wordt n.v.t."
        )

    return train_df, test_df, split_date


# ============================================================
# CORRELATIE
# ============================================================

def calculate_correlations(train_df, target_column, horizon):

    print()
    print(f"[{horizon}] Correlatieanalyse op TRAINING-set...")

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
            [x, target],
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

        rows.append({
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
        })

    result = pd.DataFrame(rows)

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
    print(f"[{horizon}] Sterkste correlaties:")

    for _, row in result.head(10).iterrows():

        print(
            f"  {row['feature']:<22} "
            f"Pearson {fmt(row['pearson'])} | "
            f"Spearman {fmt(row['spearman'])}"
        )

    print(
        f"[{horizon}] Correlaties opgeslagen: {filename}"
    )

    return result


# ============================================================
# LIGHTGBM MODEL
# ============================================================

def create_lightgbm_model():

    return lgb.LGBMClassifier(

        objective="binary",

        boosting_type="gbdt",

        # Rustiger dan de standaard LightGBM-instellingen.
        # Dit is bewust gekozen om overfitting bij financiële
        # datasets te beperken.
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

    X_train = train_df[features]
    y_train = train_df["is_profitable"]

    X_test = test_df[features]
    y_test = test_df["is_profitable"]

    # Geen testset gebruiken om het model te trainen.
    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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

    result["model_probability"] = probabilities
    result["model_prediction"] = predictions

    return (
        model,
        result,
        accuracy,
        auc,
    )


# ============================================================
# TOP-N
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

    selected = ordered.head(n_top)

    if selected.empty:
        return np.nan

    return float(
        selected[target_column].mean()
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    features,
    horizon,
):

    importance = pd.DataFrame({
        "feature": features,
        "importance_gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "importance_split": model.booster_.feature_importance(
            importance_type="split"
        ),
    })

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
    print(f"[{horizon}] Feature importance:")

    for _, row in importance.iterrows():

        print(
            f"  {row['feature']:<22} "
            f"gain={row['importance_gain']:.2f} "
            f"split={int(row['importance_split'])}"
        )

    print(
        f"[{horizon}] Feature importance opgeslagen: "
        f"{filename}"
    )

    return importance


# ============================================================
# ÉÉN HORIZON
# ============================================================

def train_horizon(df, horizon):

    print()
    print()
    print("=" * 70)
    print(f"LIGHTGBM — HORIZON {horizon}")
    print("=" * 70)

    target_column = f"fwd_ret_{horizon}"

    work = prepare_horizon_data(
        df,
        horizon,
    )

    print(
        f"[{horizon}] Bruikbare rijen: "
        f"{len(work):,}"
    )

    print(
        f"[{horizon}] Aantal tickers: "
        f"{work['ticker'].nunique():,}"
    )

    train_df, test_df, split_date = time_split(
        work
    )

    print(
        f"[{horizon}] TRAIN: "
        f"{len(train_df):,} rijen "
        f"tot {split_date}"
    )

    print(
        f"[{horizon}] TEST : "
        f"{len(test_df):,} rijen "
        f"na {split_date}"
    )

    # --------------------------------------------------------
    # CORRELATIE
    # --------------------------------------------------------

    correlations = calculate_correlations(
        train_df,
        target_column,
        horizon,
    )

    # --------------------------------------------------------
    # VOLLEDIG MODEL
    # --------------------------------------------------------

    print()
    print(
        f"[{horizon}] Volledig LightGBM-model trainen..."
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
        int(len(test_predictions) * TOP_N_FRACTIE),
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
    # Laagste pct_from_ma50 wordt eerst geselecteerd.
    # Dit volgt exact de baseline-logica van XGBoostV3.
    # --------------------------------------------------------

    top_baseline = top_n_average(
        test_predictions,
        BASELINE_KOLOM,
        target_column,
        n_top,
        descending=False,
    )

    test_average = float(
        test_predictions[target_column].mean()
    )

    # --------------------------------------------------------
    # CONTROLEMODEL
    # --------------------------------------------------------

    print()
    print(
        f"[{horizon}] Controlemodel trainen "
        f"(schaalvrije features)..."
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

    control_predictions["model_probability"] = (
        control_predictions["model_probability"]
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

    importance = save_feature_importance(
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
    # RESULTAAT
    # --------------------------------------------------------

    auc_delta = np.nan

    if (
        not pd.isna(auc)
        and not pd.isna(control_auc)
    ):
        auc_delta = auc - control_auc

    print()
    print("-" * 70)
    print(f"[{horizon}] RESULTAAT")
    print("-" * 70)

    print(
        f"Train              : {len(train_df):,}"
    )

    print(
        f"Test               : {len(test_df):,}"
    )

    print(
        f"Tickers            : "
        f"{work['ticker'].nunique():,}"
    )

    print(
        f"Split              : {split_date}"
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

    print()
    print(
        f"Model opgeslagen als:"
        f"\n  {model_filename}"
    )

    print(
        f"Testvoorspellingen:"
        f"\n  {predictions_filename}"
    )

    # --------------------------------------------------------
    # WAARSCHUWINGEN
    # --------------------------------------------------------

    warnings_list = []

    if not pd.isna(auc) and auc < 0.55:

        warnings_list.append(
            "AUC ligt dicht bij 0.50."
        )

    if (
        not pd.isna(top_model)
        and not pd.isna(top_baseline)
        and top_model <= top_baseline
    ):

        warnings_list.append(
            "Model presteert in topselectie niet "
            "beter dan de baseline."
        )

    if (
        not pd.isna(auc_delta)
        and auc_delta > AUC_VERSCHIL_WAARSCHUWING
    ):

        warnings_list.append(
            "AUC valt sterk terug bij het "
            "schaalvrije controlemodel."
        )

    if warnings_list:

        print()
        print("WAARSCHUWINGEN:")

        for warning in warnings_list:
            print(f"  ⚠ {warning}")

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
# SAMENVATTING
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
            f"Run: "
            f"{dt.datetime.now().isoformat()}\n\n"
        )

        for result in results:

            f.write(
                f"HORIZON: {result['horizon']}\n"
            )

            f.write(
                f"Status: {result['status']}\n"
            )

            f.write(
                f"Total: {result['n_total']}\n"
            )

            f.write(
                f"Train: {result['n_train']}\n"
            )

            f.write(
                f"Test: {result['n_test']}\n"
            )

            f.write(
                f"Tickers: {result['n_tickers']}\n"
            )

            f.write(
                f"Split: {result['split_date']}\n"
            )

            f.write(
                f"Accuracy: "
                f"{result['accuracy']:.6f}\n"
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
                f"Top N: {result['top_n']}\n"
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
                f"Model: {result['model_file']}\n\n"
            )

    print()
    print(
        f"Samenvatting opgeslagen: {filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LIGHTGBMV1 — HISTORISCHE TRAINING")
    print("=" * 70)

    print(
        "Doel: leren uit ALLE historische "
        "generieke_technicals-regels."
    )

    print(
        "Horizon: 10d / 30d / 60d"
    )

    print(
        f"Top-selectie: {TOP_N_FRACTIE * 100:.0f}%"
    )

    db_url = os.environ.get(
        "SUPABASE_DB_URL"
    )

    if not db_url:

        raise RuntimeError(
            "SUPABASE_DB_URL ontbreekt."
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    print()
    print("Verbinding maken met Supabase...")

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
                f"ERROR [{horizon}]: {e}"
            )

            results.append({
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
            })

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    write_summary(
        results
    )

    print()
    print("=" * 70)
    print("LIGHTGBMV1 RUN VOLTOOID")
    print("=" * 70)

    for result in results:

        print(
            f"{result['horizon']:>4} | "
            f"{result['status']:<10} | "
            f"AUC={fmt(result['auc'])} | "
            f"control={fmt(result['control_auc'])}"
        )

    print()
    print(
        f"Alle output staat in: {RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()
