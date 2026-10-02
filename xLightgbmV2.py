#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xLightgbmV2.py
==============

LightGBM trainingsengine, opvolger van lightgbmV1 (methodiek xgboostV3).

WIJZIGINGEN T.O.V. V1
---------------------
1. EMBARGO: trainrijen waarvan het label (horizon) in de testperiode reikt
   worden weggelaten (voorkomt label-overlap tussen train en test).
   Uitzetten kan met --geen-embargo (= gedrag van V1).
2. TARGET: standaard marktneutraal ("excess"):
       1 = forward return > mediaan van dezelfde datum
   Met --target abs krijg je het V1-target (forward return > 0).
3. EVALUATIE PER DATUM: gemiddelde AUC per datum en gemiddelde Spearman IC
   per datum, naast de gepoolde AUC.
4. UITSCHIETERS: top-20% rapporteert gemiddelde, mediaan EN getrimd gemiddelde.
5. STRATEGIE-FILTER: --exclude-strategie sluit selecties van die strategie(en)
   uit (alleen als forward_returns een kolom `strategie` heeft).
6. EARLY STOPPING op een tijdsgebonden validatieset (laatste 15% van de
   trainingsdatums, met embargo).
7. OPSLAG: per horizon een pkl-bundel met:
       model        -> getraind op TRAIN (voor evaluatie)
       model_full   -> hertraind op ALLE rijen (voor productie)
8. AUC_VERSCHIL_WAARSCHUWING wordt nu echt gebruikt.
9. PREFLIGHT: per horizon wordt vooraf gerapporteerd of er genoeg data is.
   Een horizon zonder genoeg data krijgt status "overgeslagen".
   Met --strict faalt de run wel als niet alles lukt.
10. NOTIFICATIES: aan het einde wordt een rapport gestuurd naar Telegram
    (HTML, emojis) en e-mail (HTML). Zelfde secrets als a_trade.py:
        TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
        EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER
    Optioneel: RUN_URL, ARTIFACT_URL voor links in het bericht.
    Notificaties falen nooit hard -- een falend kanaal blokkeert de
    trainingsrun niet. Uitzetten kan met --geen-notify.

GEBRUIK
-------
    python xLightgbmV2.py
    python xLightgbmV2.py --horizons 10d
    python xLightgbmV2.py --target abs
    python xLightgbmV2.py --exclude-strategie bot_00db
    python xLightgbmV2.py --strict
    python xLightgbmV2.py --geen-notify

DATABASE
--------
Environment variable: SUPABASE_DB_URL

OUTPUT (results/)
-----------------
    lightgbmV2_<h>_model.pkl
    lightgbmV2_<h>_feature_importance.csv
    lightgbmV2_<h>_correlations.csv
    lightgbmV2_<h>_test_predictions.csv
    lightgbmV2_summary.txt
"""

import argparse
import datetime as dt
import html
import os
import smtplib
import sys
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
import requests
from scipy.stats import trim_mean
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATIE
# ============================================================

MODEL_VERSIE = "lightgbmV2"

HORIZONS_DEFAULT = "10d,30d,60d"

TOP_N_FRACTIE = 0.20
TRIM_FRACTIE = 0.10

TRAIN_FRACTIE = 0.80          # chronologische split op unieke datums
VALIDATIE_FRACTIE = 0.15      # laatste deel van de trainingsdatums

MIN_RIJEN_TRAINING = 100
MIN_RIJEN_TEST = 30
MIN_RIJEN_VALIDATIE = 50
MIN_RIJEN_PER_DATUM = 10
MIN_DATUMS = 20
DATUMS_WAARSCHUWING = 60

MAX_BOMEN = 1000
BOMEN_ZONDER_VALIDATIE = 400
EARLY_STOPPING_RONDES = 50

RANDOM_STATE = 42

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Notificatie-drempels
AUC_ALARM = 0.50
AUC_GOED = 0.55


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


def horizon_dagen(horizon):
    return int(str(horizon).lower().replace("d", ""))


def embargo_kalenderdagen(horizon):
    return int(np.ceil(horizon_dagen(horizon) * 7 / 5)) + 2


# ============================================================
# DATABASE
# ============================================================

def heeft_kolom(conn, tabel, kolom):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s LIMIT 1",
            (tabel, kolom),
        )
        return cur.fetchone() is not None


def bouw_query(met_strategie, uitsluiten):
    where = ""
    params = None
    order = "ORDER BY ticker, datum"

    if met_strategie:
        order = "ORDER BY ticker, datum, strategie"
        if uitsluiten:
            where = "WHERE strategie <> ALL(%(uitsluiten)s)"
            params = {"uitsluiten": list(uitsluiten)}

    query = f"""
    WITH fr_dedup AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker,
            datum,
            fwd_ret_10d,
            fwd_ret_30d,
            fwd_ret_60d
        FROM forward_returns
        {where}
        {order}
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
    ORDER BY gt.datum, gt.ticker
    """

    return query, params


def get_training_data(conn, uitsluiten):
    print()
    print("=" * 70)
    print("DATASET OPHALEN")
    print("=" * 70)

    met_strategie = heeft_kolom(conn, "forward_returns", "strategie")

    if uitsluiten and not met_strategie:
        print(
            "WAARSCHUWING: forward_returns heeft geen kolom 'strategie'; "
            "--exclude-strategie wordt genegeerd."
        )
        uitsluiten = []

    print(f"forward_returns.strategie aanwezig : {met_strategie}")
    print(f"Uitgesloten strategieen            : {uitsluiten or '-'}")

    query, params = bouw_query(met_strategie, uitsluiten)
    df = pd.read_sql(query, conn, params=params)

    if df.empty:
        raise RuntimeError(
            "Supabase gaf 0 rijen terug uit "
            "generieke_technicals + forward_returns."
        )

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])

    print(f"Rijen opgehaald : {len(df):,}")
    print(f"Tickers         : {df['ticker'].nunique():,}")
    print(f"Datums          : {df['datum'].nunique():,}")
    print(f"Van             : {df['datum'].min()}")
    print(f"Tot             : {df['datum'].max()}")

    for h in ("10d", "30d", "60d"):
        kolom = f"fwd_ret_{h}"
        if kolom in df.columns:
            print(f"Rijen met {kolom:<11}: {df[kolom].notna().sum():,}")

    return df


# ============================================================
# PREFLIGHT
# ============================================================

def preflight_report(df, horizons):
    rapport = {}

    print()
    print("=" * 70)
    print("PREFLIGHT — DATA-BESCHIKBAARHEID PER HORIZON")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'Rijen':>9} | {'Datums':>7} | Status")
    print("-" * 70)

    min_rijen = MIN_RIJEN_TRAINING + MIN_RIJEN_TEST

    for horizon in horizons:
        kolom = f"fwd_ret_{horizon}"

        if kolom not in df.columns:
            rapport[horizon] = (False, f"kolom {kolom} ontbreekt", 0, 0)
            print(f"{horizon:>8} | {'-':>9} | {'-':>7} | KOLOM ONTBREEKT")
            continue

        mask = df[kolom].notna()
        n_rijen = int(mask.sum())
        n_datums = int(df.loc[mask, "datum"].nunique()) if n_rijen else 0

        if n_rijen < min_rijen:
            status = f"TE WEINIG RIJEN (min {min_rijen})"
            reden = f"te weinig bruikbare rijen: {n_rijen}"
        elif n_datums < MIN_DATUMS:
            status = f"TE WEINIG DATUMS (min {MIN_DATUMS})"
            reden = f"slechts {n_datums} verschillende datums"
        else:
            status = "OK"
            reden = ""

        rapport[horizon] = (status == "OK", reden, n_rijen, n_datums)
        print(f"{horizon:>8} | {n_rijen:>9,} | {n_datums:>7,} | {status}")

    print("=" * 70)
    return rapport


# ============================================================
# DATA VOORBEREIDEN
# ============================================================

def prepare_horizon_data(df, horizon, target_mode):
    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(f"Doelkolom ontbreekt: {target_column}")

    work = df.copy()
    work["datum"] = pd.to_datetime(work["datum"], errors="coerce")
    work = work.drop_duplicates(subset=["ticker", "datum"], keep="first")

    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.dropna(subset=[target_column, "datum"])

    for feature in FEATURE_COLUMNS:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    work = work.dropna(subset=FEATURE_COLUMNS)

    work["is_profitable"] = (work[target_column] > 0).astype(int)

    if target_mode == "excess":
        aantal = work.groupby("datum")[target_column].transform("size")
        work = work[aantal >= MIN_RIJEN_PER_DATUM].copy()

        mediaan = work.groupby("datum")[target_column].transform("median")
        work["target"] = (work[target_column] > mediaan).astype(int)
    else:
        work["target"] = work["is_profitable"]

    work = work.sort_values(["datum", "ticker"]).reset_index(drop=True)

    return work


# ============================================================
# CHRONOLOGISCHE SPLIT MET EMBARGO
# ============================================================

def time_split(df, horizon, embargo=True):
    if len(df) < (MIN_RIJEN_TRAINING + MIN_RIJEN_TEST):
        raise RuntimeError(
            f"Te weinig bruikbare rijen: {len(df)}. "
            f"Minimaal {MIN_RIJEN_TRAINING + MIN_RIJEN_TEST} vereist."
        )

    dates = (
        df["datum"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    if len(dates) < MIN_DATUMS:
        raise RuntimeError(
            f"Slechts {len(dates)} verschillende datums; minimaal "
            f"{MIN_DATUMS} nodig voor een zinvolle tijdssplit."
        )

    if len(dates) < DATUMS_WAARSCHUWING:
        print(
            f"[{horizon}] WAARSCHUWING: maar {len(dates)} datums. "
            f"De testset beslaat dan een heel korte periode (een of twee "
            f"marktregimes); interpreteer AUC/top-20% met grote voorzichtigheid."
        )

    split_index = max(0, int(len(dates) * TRAIN_FRACTIE) - 1)
    split_date = dates.iloc[split_index]

    if embargo:
        train_einde = split_date - pd.Timedelta(
            days=embargo_kalenderdagen(horizon)
        )
    else:
        train_einde = split_date

    train_df = df[df["datum"] <= train_einde].copy()
    test_df = df[df["datum"] > split_date].copy()

    if len(train_df) < MIN_RIJEN_TRAINING:
        raise RuntimeError(f"Trainingsset te klein na embargo: {len(train_df)}.")

    if len(test_df) < MIN_RIJEN_TEST:
        raise RuntimeError(f"Testset te klein: {len(test_df)}.")

    if train_df["target"].nunique() < 2:
        raise RuntimeError("Trainingsset bevat slechts één klasse.")

    return train_df, test_df, split_date, train_einde


def split_validatie(train_df, horizon, embargo=True):
    dates = (
        train_df["datum"].drop_duplicates().sort_values().reset_index(drop=True)
    )

    if len(dates) < 10:
        return train_df, None

    idx = int(len(dates) * (1 - VALIDATIE_FRACTIE))
    val_start = dates.iloc[min(idx, len(dates) - 1)]

    if embargo:
        fit_einde = val_start - pd.Timedelta(days=embargo_kalenderdagen(horizon))
    else:
        fit_einde = val_start

    fit_df = train_df[train_df["datum"] <= fit_einde]
    val_df = train_df[train_df["datum"] >= val_start]

    if (
        len(fit_df) < MIN_RIJEN_TRAINING
        or len(val_df) < MIN_RIJEN_VALIDATIE
        or fit_df["target"].nunique() < 2
        or val_df["target"].nunique() < 2
    ):
        return train_df, None

    return fit_df, val_df


# ============================================================
# CORRELATIE
# ============================================================

def calculate_correlations(train_df, target_column, horizon):
    print()
    print(f"[{horizon}] Correlatieanalyse op TRAINING-set...")

    rows = []
    target = pd.to_numeric(train_df[target_column], errors="coerce")

    for feature in FEATURE_COLUMNS:
        x = pd.to_numeric(train_df[feature], errors="coerce")
        valid = pd.concat([x, target], axis=1).dropna()

        if len(valid) < 10:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson")
            spearman = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")

        rows.append(
            {
                "feature": feature,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson) if not pd.isna(pearson) else np.nan,
                "abs_spearman": abs(spearman) if not pd.isna(spearman) else np.nan,
            }
        )

    result = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)

    filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_correlations.csv"
    )
    result.to_csv(filename, index=False)

    for _, row in result.head(10).iterrows():
        print(
            f"  {row['feature']:<22} "
            f"Pearson {fmt(row['pearson'])} | Spearman {fmt(row['spearman'])}"
        )

    return result


# ============================================================
# MODEL
# ============================================================

def create_lightgbm_model(n_estimators):
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
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


def per_datum_metrics(df, prob_col, ret_col, target_col):
    aucs = []
    ics = []

    for _, grp in df.groupby("datum"):
        if len(grp) < MIN_RIJEN_PER_DATUM:
            continue

        if grp[target_col].nunique() == 2:
            aucs.append(roc_auc_score(grp[target_col], grp[prob_col]))

        ic = grp[prob_col].corr(grp[ret_col], method="spearman")
        if not pd.isna(ic):
            ics.append(ic)

    return (
        float(np.mean(aucs)) if aucs else np.nan,
        float(np.mean(ics)) if ics else np.nan,
        len(aucs),
    )


def train_model(train_df, test_df, features, target_column, horizon, embargo):
    fit_df, val_df = split_validatie(train_df, horizon, embargo)

    if val_df is not None:
        model = create_lightgbm_model(MAX_BOMEN)
        model.fit(
            fit_df[features],
            fit_df["target"],
            eval_set=[(val_df[features], val_df["target"])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_RONDES, verbose=False)
            ],
        )
        best_iteration = int(getattr(model, "best_iteration_", 0) or MAX_BOMEN)
    else:
        model = create_lightgbm_model(BOMEN_ZONDER_VALIDATIE)
        model.fit(train_df[features], train_df["target"])
        best_iteration = BOMEN_ZONDER_VALIDATIE

    X_test = test_df[features]
    y_test = test_df["target"]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    auc = (
        roc_auc_score(y_test, probabilities) if y_test.nunique() >= 2 else np.nan
    )

    auc_abs = (
        roc_auc_score(test_df["is_profitable"], probabilities)
        if test_df["is_profitable"].nunique() >= 2
        else np.nan
    )

    result = test_df.copy()
    result["model_probability"] = probabilities
    result["model_prediction"] = predictions

    auc_datum, ic_datum, n_datums = per_datum_metrics(
        result, "model_probability", target_column, "target"
    )

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "auc_abs": auc_abs,
        "auc_datum": auc_datum,
        "ic_datum": ic_datum,
        "n_datums_eval": n_datums,
        "used_validation": val_df is not None,
    }

    return model, result, metrics, best_iteration


def top_n_stats(df, sort_column, target_column, n_top, descending=True):
    ordered = df.sort_values(sort_column, ascending=not descending)
    selected = ordered.head(n_top)[target_column]

    if selected.empty:
        return {"gem": np.nan, "med": np.nan, "trim": np.nan}

    return {
        "gem": float(selected.mean()),
        "med": float(selected.median()),
        "trim": float(trim_mean(selected.values, TRIM_FRACTIE)),
    }


def save_feature_importance(model, features, horizon):
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
            "importance_split": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("importance_gain", ascending=False)

    filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_feature_importance.csv"
    )
    importance.to_csv(filename, index=False)

    print()
    print(f"[{horizon}] Feature importance:")
    for _, row in importance.iterrows():
        print(
            f"  {row['feature']:<22} "
            f"gain={row['importance_gain']:.2f} "
            f"split={int(row['importance_split'])}"
        )

    return importance


# ============================================================
# ÉÉN HORIZON
# ============================================================

def train_horizon(df, horizon, target_mode, embargo, uitgesloten):
    print()
    print()
    print("=" * 70)
    print(f"LIGHTGBM V2 — HORIZON {horizon} — target={target_mode}")
    print("=" * 70)

    target_column = f"fwd_ret_{horizon}"

    work = prepare_horizon_data(df, horizon, target_mode)

    print(f"[{horizon}] Bruikbare rijen: {len(work):,}")
    print(f"[{horizon}] Tickers: {work['ticker'].nunique():,}")
    print(f"[{horizon}] Datums : {work['datum'].nunique():,}")

    train_df, test_df, split_date, train_einde = time_split(
        work, horizon, embargo
    )

    print(f"[{horizon}] TRAIN: {len(train_df):,} rijen (t/m {train_einde.date()})")
    print(f"[{horizon}] TEST : {len(test_df):,} rijen (na {split_date.date()})")
    print(
        f"[{horizon}] Embargo: "
        f"{embargo_kalenderdagen(horizon) if embargo else 0} kalenderdagen"
    )
    print(
        f"[{horizon}] Basisrate target: "
        f"train {train_df['target'].mean():.3f} | "
        f"test {test_df['target'].mean():.3f}"
    )

    calculate_correlations(train_df, target_column, horizon)

    print(f"\n[{horizon}] Volledig LightGBM-model trainen...")

    model, test_predictions, metrics, best_iteration = train_model(
        train_df, test_df, FEATURE_COLUMNS, target_column, horizon, embargo
    )

    n_top = max(1, int(len(test_predictions) * TOP_N_FRACTIE))

    top_model = top_n_stats(
        test_predictions, "model_probability", target_column, n_top, True
    )

    top_baseline = top_n_stats(
        test_predictions, BASELINE_KOLOM, target_column, n_top, False
    )

    test_stats = {
        "gem": float(test_predictions[target_column].mean()),
        "med": float(test_predictions[target_column].median()),
        "trim": float(
            trim_mean(test_predictions[target_column].values, TRIM_FRACTIE)
        ),
    }

    print(f"\n[{horizon}] Controlemodel trainen...")

    control_model, control_predictions, control_metrics, _ = train_model(
        train_df,
        test_df,
        FEATURE_COLUMNS_RELATIEF,
        target_column,
        horizon,
        embargo,
    )

    top_control = top_n_stats(
        control_predictions, "model_probability", target_column, n_top, True
    )

    save_feature_importance(model, FEATURE_COLUMNS, horizon)

    n_full = max(50, int(best_iteration * 1.10))
    model_full = create_lightgbm_model(n_full)
    model_full.fit(work[FEATURE_COLUMNS], work["target"])

    bundle = {
        "versie": MODEL_VERSIE,
        "horizon": horizon,
        "target_mode": target_mode,
        "features": list(FEATURE_COLUMNS),
        "split_date": str(split_date),
        "embargo": bool(embargo),
        "best_iteration": best_iteration,
        "uitgesloten_strategieen": list(uitgesloten),
        "model": model,
        "model_full": model_full,
    }

    model_filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_model.pkl"
    )
    joblib.dump(bundle, model_filename)

    predictions_filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_test_predictions.csv"
    )

    output_columns = [
        "ticker",
        "datum",
        target_column,
        "target",
        "model_probability",
        "model_prediction",
        BASELINE_KOLOM,
    ]

    test_predictions[output_columns].sort_values(
        "model_probability", ascending=False
    ).to_csv(predictions_filename, index=False)

    if not pd.isna(metrics["auc"]) and not pd.isna(control_metrics["auc"]):
        auc_delta = metrics["auc"] - control_metrics["auc"]
    else:
        auc_delta = np.nan

    auc_waarschuwing = (
        (not pd.isna(auc_delta)) and abs(auc_delta) > AUC_VERSCHIL_WAARSCHUWING
    )

    print()
    print("-" * 70)
    print(f"[{horizon}] RESULTAAT")
    print("-" * 70)
    print(f"Train / Test         : {len(train_df):,} / {len(test_df):,}")
    print(f"Tickers              : {work['ticker'].nunique():,}")
    print(f"Early stopping       : {metrics['used_validation']} "
          f"(best_iteration={best_iteration})")
    print(f"Accuracy             : {metrics['accuracy'] * 100:.2f}%")
    print(f"AUC (target)         : {fmt(metrics['auc'])}")
    print(f"AUC (fwd_ret > 0)    : {fmt(metrics['auc_abs'])}")
    print(f"AUC per datum (gem.) : {fmt(metrics['auc_datum'])} "
          f"over {metrics['n_datums_eval']} datums")
    print(f"IC per datum (gem.)  : {fmt(metrics['ic_datum'])}")
    print(f"Controle AUC         : {fmt(control_metrics['auc'])}")
    print(f"Controle AUC/datum   : {fmt(control_metrics['auc_datum'])}")
    print(f"AUC verschil         : {fmt(auc_delta)}")

    if auc_waarschuwing:
        print(
            f"  !! WAARSCHUWING: |AUC verschil| > {AUC_VERSCHIL_WAARSCHUWING:.2f}: "
            f"het volledige model leunt mogelijk op absolute prijsniveaus."
        )

    print()
    print(f"Top {n_top} (20%)        gem      med      getrimd")
    for naam, st in (
        ("model   ", top_model),
        ("baseline", top_baseline),
        ("controle", top_control),
        ("testset ", test_stats),
    ):
        print(
            f"  {naam}           {pct(st['gem']):>8} "
            f"{pct(st['med']):>8} {pct(st['trim']):>8}"
        )

    print(f"\nModel opgeslagen: {model_filename}")

    return {
        "horizon": horizon,
        "status": "getraind",
        "reden": "",
        "target_mode": target_mode,
        "n_total": len(work),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_tickers": work["ticker"].nunique(),
        "split_date": str(split_date),
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "auc_abs": metrics["auc_abs"],
        "auc_datum": metrics["auc_datum"],
        "ic_datum": metrics["ic_datum"],
        "control_auc": control_metrics["auc"],
        "auc_delta": auc_delta,
        "auc_waarschuwing": auc_waarschuwing,
        "top_n": n_top,
        "top_model_gem": top_model["gem"],
        "top_model_med": top_model["med"],
        "top_model_trim": top_model["trim"],
        "top_baseline_gem": top_baseline["gem"],
        "top_baseline_med": top_baseline["med"],
        "top_baseline_trim": top_baseline["trim"],
        "top_control_gem": top_control["gem"],
        "test_gem": test_stats["gem"],
        "test_med": test_stats["med"],
        "test_trim": test_stats["trim"],
        "model_file": model_filename,
        "fout": "",
    }


def fout_resultaat(horizon, target_mode, fout):
    reden = str(fout)
    leeg = {
        "horizon": horizon,
        "status": "overgeslagen",
        "reden": reden,
        "target_mode": target_mode,
        "n_total": 0,
        "n_train": 0,
        "n_test": 0,
        "n_tickers": 0,
        "split_date": "",
        "auc_waarschuwing": False,
        "top_n": 0,
        "model_file": "",
        "fout": reden,
    }
    for sleutel in (
        "accuracy", "auc", "auc_abs", "auc_datum", "ic_datum",
        "control_auc", "auc_delta",
        "top_model_gem", "top_model_med", "top_model_trim",
        "top_baseline_gem", "top_baseline_med", "top_baseline_trim",
        "top_control_gem", "test_gem", "test_med", "test_trim",
    ):
        leeg[sleutel] = np.nan
    return leeg


# ============================================================
# SUMMARY FILE
# ============================================================

def write_summary(results, args):
    filename = os.path.join(RESULTS_DIR, f"{MODEL_VERSIE}_summary.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("LIGHTGBMV2 TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run: {dt.datetime.now().isoformat()}\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Embargo: {not args.geen_embargo}\n")
        f.write(f"Uitgesloten strategieen: {args.exclude_strategie or '-'}\n\n")

        for result in results:
            f.write(f"HORIZON: {result['horizon']}\n")
            for sleutel, waarde in result.items():
                if sleutel == "horizon":
                    continue
                if isinstance(waarde, (float, np.floating)):
                    waarde = safe_float(waarde)
                f.write(f"  {sleutel}: {waarde}\n")
            f.write("\n")

    return filename


# ============================================================
# NOTIFICATIES — TELEGRAM
# ============================================================

def _esc(s):
    return html.escape(str(s))


def _status_emoji(result):
    if result["status"] != "getraind":
        return "⏭️", "overgeslagen"
    auc = result.get("auc")
    if is_nan(auc):
        return "⚠️", "getraind (AUC onbekend)"
    auc = float(auc)
    if auc >= AUC_GOED:
        return "🟢", "getraind"
    if auc >= AUC_ALARM:
        return "🟡", "getraind (zwak)"
    return "🔴", "getraind (AUC < 0.50)"


def maak_telegram_bericht(results, args):
    vandaag = dt.datetime.now().strftime("%Y-%m-%d")
    n_ok = sum(1 for r in results if r["status"] == "getraind")
    n_tot = len(results)

    run_url = os.environ.get("RUN_URL", "")
    artifact_url = os.environ.get("ARTIFACT_URL", "")

    lijnen = [
        "🤖🤖🤖 <b>LIGHTGBM V2 — TRAININGSRAPPORT</b> 🤖🤖🤖",
        f"<i>{vandaag}</i>",
        "",
        f"✅ <b>{n_ok}/{n_tot}</b> horizons getraind  "
        f"<i>(target={_esc(args.target)})</i>",
    ]

    for r in results:
        naam = _esc(r["horizon"])
        emoji, kort = _status_emoji(r)

        lijnen.append("")
        lijnen.append(f"{emoji} <b>{naam}</b> — {_esc(kort)}")

        if r["status"] == "getraind":
            auc = fmt(r.get("auc"))
            auc_d = fmt(r.get("auc_datum"))
            ic_d = fmt(r.get("ic_datum"))
            ctrl = fmt(r.get("control_auc"))

            lijnen.append(f"  📊 AUC: <b>{_esc(auc)}</b>  (per datum: {_esc(auc_d)})")
            lijnen.append(f"  📈 IC/datum: <b>{_esc(ic_d)}</b>")
            lijnen.append(f"  🎯 Controle AUC: {_esc(ctrl)}")

            top_gem = pct(r.get("top_model_gem"))
            base_gem = pct(r.get("top_baseline_gem"))
            test_gem = pct(r.get("test_gem"))
            lijnen.append(
                f"  🥇 Top-20%: <b>{_esc(top_gem)}</b>  "
                f"(baseline: {_esc(base_gem)}, testset: {_esc(test_gem)})"
            )

            if r.get("auc_waarschuwing"):
                lijnen.append(
                    "  ⚠️ <i>AUC-verschil t.o.v. controlemodel groot — "
                    "model leunt mogelijk op prijsniveau i.p.v. signaal</i>"
                )

            test_gem_f = r.get("test_gem")
            if not is_nan(test_gem_f) and float(test_gem_f) < 0:
                lijnen.append("  🔻 <i>Testset gem. rendement negatief</i>")
        else:
            reden = _esc(r.get("reden", ""))
            if len(reden) > 100:
                reden = reden[:97] + "..."
            lijnen.append(f"  <i>{reden}</i>")

    if run_url:
        lijnen.append("")
        lijnen.append(f'🔗 <a href="{_esc(run_url)}">Bekijk volledige run</a>')

    if artifact_url:
        lijnen.append(f'📦 <a href="{_esc(artifact_url)}">Download artifact</a>')

    return "\n".join(lijnen)


def stuur_telegram(tekst):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram-secrets ontbreken, overslaan.", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": tekst,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if not resp.ok:
            print(
                f"Telegram-fout: {resp.status_code} {resp.text}",
                file=sys.stderr,
            )
            return False
        print("Telegram verzonden.")
        return True
    except Exception as e:
        print(f"Telegram-exception: {e}", file=sys.stderr)
        return False


# ============================================================
# NOTIFICATIES — E-MAIL
# ============================================================

def _horizon_blok_html(r):
    naam = r["horizon"]
    emoji, kort = _status_emoji(r)

    if r["status"] != "getraind":
        reden = r.get("reden", "onbekende reden")
        return f"""
        <div style="background:#f5f5f5; border-left:6px solid #999;
                    border-radius:6px; padding:14px; margin-bottom:12px;">
          <h3 style="margin:0 0 6px 0;">{emoji} {naam} — {kort}</h3>
          <p style="margin:0; color:#555;">{reden}</p>
        </div>
        """

    auc = r.get("auc")
    if is_nan(auc):
        kleur = "#999"
    elif float(auc) >= AUC_GOED:
        kleur = "#2e7d32"
    elif float(auc) >= AUC_ALARM:
        kleur = "#f9a825"
    else:
        kleur = "#c62828"

    rijen = [
        ("AUC (target)", fmt(r.get("auc"))),
        ("AUC per datum", fmt(r.get("auc_datum"))),
        ("IC per datum", fmt(r.get("ic_datum"))),
        ("Controle AUC", fmt(r.get("control_auc"))),
        ("AUC-verschil", fmt(r.get("auc_delta"))),
        ("Accuracy", pct(r.get("accuracy"), 1)),
        ("Train / Test", f"{r.get('n_train')} / {r.get('n_test')}"),
        ("Tickers", r.get("n_tickers")),
        ("Split-datum", r.get("split_date")),
        ("Top-20% gem.", pct(r.get("top_model_gem"))),
        ("Top-20% mediaan", pct(r.get("top_model_med"))),
        ("Baseline gem.", pct(r.get("top_baseline_gem"))),
        ("Testset gem.", pct(r.get("test_gem"))),
    ]

    rijen_html = "".join(
        f"<tr><td style='padding:4px 10px; color:#555;'>{k}</td>"
        f"<td style='padding:4px 10px;'><b>{v}</b></td></tr>"
        for k, v in rijen
    )

    waarschuwing_html = ""
    if not is_nan(auc) and float(auc) < AUC_ALARM:
        waarschuwing_html = (
            "<p style='color:#c62828; margin:8px 0 0 0;'>"
            "⚠️ AUC onder 0.50 — model presteert slechter dan willekeurig. "
            "Controleer features, target-definitie en embargo."
            "</p>"
        )
    if r.get("auc_waarschuwing"):
        waarschuwing_html += (
            "<p style='color:#e65100; margin:8px 0 0 0;'>"
            "⚠️ Groot AUC-verschil met controlemodel — mogelijk "
            "overfit op absolute prijsniveaus."
            "</p>"
        )

    return f"""
    <div style="background:#fff; border-left:6px solid {kleur};
                border-radius:6px; padding:16px; margin-bottom:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <h3 style="margin:0 0 10px 0;">{emoji} {naam} — {kort}</h3>
      <table style="border-collapse:collapse; font-family:Arial,sans-serif;
                    font-size:14px;">{rijen_html}</table>
      {waarschuwing_html}
    </div>
    """


def maak_email_html(results, args):
    vandaag = dt.datetime.now().strftime("%Y-%m-%d")
    n_ok = sum(1 for r in results if r["status"] == "getraind")
    n_tot = len(results)

    run_url = os.environ.get("RUN_URL", "")
    artifact_url = os.environ.get("ARTIFACT_URL", "")

    blokken = "".join(_horizon_blok_html(r) for r in results)

    run_html = (
        f'<p><a href="{run_url}">🔗 Bekijk run op GitHub</a></p>'
        if run_url else ""
    )
    artifact_html = (
        f'<p><a href="{artifact_url}">📦 Download artifact</a></p>'
        if artifact_url else ""
    )

    return f"""
    <html>
      <body style="font-family:Arial,sans-serif; background:#fafafa; padding:20px;">
        <div style="max-width:680px; margin:0 auto;">
          <div style="background:#e3f2fd; border:3px solid #1976d2;
                      border-radius:10px; padding:20px; margin-bottom:20px;">
            <h1 style="color:#0d47a1; margin-top:0;">🤖 LIGHTGBM V2</h1>
            <p style="color:#555; margin:0;">{vandaag} — target={args.target}</p>
            <p style="font-size:18px; margin:10px 0 0 0;">
              ✅ <b>{n_ok}/{n_tot}</b> horizons getraind
            </p>
          </div>
          {blokken}
          {run_html}
          {artifact_html}
        </div>
      </body>
    </html>
    """


def stuur_email(html_body, heeft_modellen):
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    receiver = os.environ.get("EMAIL_RECEIVER")

    if not user or not password or not receiver:
        print("Email-secrets ontbreken, overslaan.", file=sys.stderr)
        return False

    onderwerp = (
        "🤖 LightGBM V2 — trainingsrapport"
        if heeft_modellen
        else "⚠️ LightGBM V2 — geen resultaten"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = onderwerp
    msg["From"] = user
    msg["To"] = receiver
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, receiver, msg.as_string())
        print(f"E-mail verzonden naar {receiver}.")
        return True
    except Exception as e:
        print(f"Email-exception: {e}", file=sys.stderr)
        return False


def stuur_notificaties(results, args):
    """Stuurt Telegram + e-mail. Faalt nooit hard."""
    if getattr(args, "geen_notify", False):
        print("Notificaties uitgeschakeld via --geen-notify.")
        return

    try:
        telegram_tekst = maak_telegram_bericht(results, args)
        ok_tg = stuur_telegram(telegram_tekst)
    except Exception as e:
        print(f"Telegram-voorbereiding faalde: {e}", file=sys.stderr)
        ok_tg = False

    try:
        email_html = maak_email_html(results, args)
        n_ok = sum(1 for r in results if r["status"] == "getraind")
        ok_mail = stuur_email(email_html, heeft_modellen=(n_ok > 0))
    except Exception as e:
        print(f"E-mail-voorbereiding faalde: {e}", file=sys.stderr)
        ok_mail = False

    print(
        f"Notificaties: Telegram={'OK' if ok_tg else 'skip/fout'}, "
        f"mail={'OK' if ok_mail else 'skip/fout'}."
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="LightGBM V2 training")
    ap.add_argument(
        "--horizons",
        default=HORIZONS_DEFAULT,
        help="komma-gescheiden, bv. 10d,30d,60d",
    )
    ap.add_argument(
        "--target",
        choices=["excess", "abs"],
        default="excess",
        help="excess = boven mediaan van dezelfde datum (standaard); "
             "abs = forward return > 0 (zoals V1/xgboostV3)",
    )
    ap.add_argument(
        "--exclude-strategie",
        default="",
        help="komma-gescheiden strategieen om uit te sluiten, bv. bot_00db",
    )
    ap.add_argument(
        "--geen-embargo",
        action="store_true",
        help="zet de embargo uit (gedrag van V1)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit met code 1 als niet ALLE opgegeven horizons getraind zijn",
    )
    ap.add_argument(
        "--geen-notify",
        action="store_true",
        help="stuur geen Telegram/e-mail notificatie",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    uitsluiten = [s.strip() for s in args.exclude_strategie.split(",") if s.strip()]
    embargo = not args.geen_embargo

    print()
    print("=" * 70)
    print("LIGHTGBMV2 — HISTORISCHE TRAINING")
    print("=" * 70)
    print(f"Horizons : {', '.join(horizons)}")
    print(f"Target   : {args.target}")
    print(f"Embargo  : {embargo}")
    print(f"Top      : {TOP_N_FRACTIE * 100:.0f}%")
    print(f"Strict   : {args.strict}")
    print(f"Notify   : {not args.geen_notify}")

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt.")

    print("\nVerbinding maken met Supabase...")
    conn = psycopg2.connect(db_url)

    try:
        df = get_training_data(conn, uitsluiten)
    finally:
        conn.close()

    rapport = preflight_report(df, horizons)

    results = []

    for horizon in horizons:
        kan_trainen, reden, _, _ = rapport[horizon]
        if not kan_trainen:
            print()
            print(f"OVERGESLAGEN [{horizon}]: {reden}")
            results.append(fout_resultaat(horizon, args.target, reden))
            continue

        try:
            results.append(
                train_horizon(df, horizon, args.target, embargo, uitsluiten)
            )
        except Exception as e:
            print()
            print(f"ERROR [{horizon}]: {e}")
            results.append(fout_resultaat(horizon, args.target, e))

    write_summary(results, args)

    getraind = [r for r in results if r["status"] == "getraind"]
    n_getraind = len(getraind)
    n_totaal = len(results)

    print()
    print("=" * 78)
    print("LIGHTGBMV2 RUN VOLTOOID")
    print("=" * 78)
    print(
        f"{'Hor':>4} | {'Status':<12} | {'AUC':>6} | {'AUC/d':>6} | "
        f"{'IC/d':>7} | {'Ctrl':>6} | Reden"
    )
    print("-" * 78)

    for r in results:
        if r["status"] == "getraind":
            print(
                f"{r['horizon']:>4} | getraind     | "
                f"{fmt(r['auc']):>6} | {fmt(r['auc_datum']):>6} | "
                f"{fmt(r['ic_datum']):>7} | {fmt(r['control_auc']):>6} | -"
            )
        else:
            reden = str(r.get("reden", r.get("fout", "")))
            if len(reden) > 40:
                reden = reden[:37] + "..."
            print(
                f"{r['horizon']:>4} | overgeslagen | "
                f"{'-':>6} | {'-':>6} | {'-':>7} | {'-':>6} | {reden}"
            )

    print("-" * 78)
    print(f"Modellen : {n_getraind}/{n_totaal} getraind")
    print(f"Output   : {RESULTS_DIR}/")

    # Notificaties sturen (tenzij uitgeschakeld)
    stuur_notificaties(results, args)

    if args.strict and n_getraind < n_totaal:
        raise SystemExit(
            f"STRIKT: {n_totaal - n_getraind} van {n_totaal} horizons "
            f"niet getraind."
        )

    if n_getraind == 0:
        raise SystemExit("Geen enkele horizon kon getraind worden.")


if __name__ == "__main__":
    main()
