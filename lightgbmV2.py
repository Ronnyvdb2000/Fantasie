#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lightgbmV2.py
=============

LightGBM trainingsengine, opvolger van lightgbmV1 (methodiek xgboostV3).

WIJZIGINGEN T.O.V. V1
---------------------
1. EMBARGO: trainrijen waarvan het label (horizon) in de testperiode reikt
   worden weggelaten (voorkomt label-overlap tussen train en test).
   Uitzetten kan met --geen-embargo (= gedrag van V1).
2. TARGET: standaard marktneutraal ("excess"):
       1 = forward return > mediaan van dezelfde datum
   Met --target abs krijg je het V1-target (forward return > 0), zodat je
   eerlijk kunt vergelijken met xgboostV3.
3. EVALUATIE PER DATUM: gemiddelde AUC per datum en gemiddelde Spearman IC
   per datum, naast de gepoolde AUC.
4. UITSCHIETERS: top-20% rapporteert gemiddelde, mediaan EN getrimd gemiddelde.
5. STRATEGIE-FILTER: --exclude-strategie bot_00db sluit selecties van die
   strategie(en) uit (alleen als forward_returns een kolom `strategie` heeft;
   dat wordt bij het opstarten gecontroleerd in information_schema).
6. EARLY STOPPING op een tijdsgebonden validatieset (laatste 15% van de
   trainingsdatums, met embargo).
7. OPSLAG: per horizon een pkl-bundel met:
       model        -> getraind op TRAIN (voor evaluatie)
       model_full   -> hertraind op ALLE rijen (voor productie)
       features, horizon, target_mode, split_date, best_iteration, ...
8. AUC_VERSCHIL_WAARSCHUWING wordt nu echt gebruikt.
9. Een horizon zonder genoeg data geeft status "fout" in de summary maar
   laat de run niet meer crashen; de run faalt alleen als GEEN enkele
   horizon getraind kon worden.

GEBRUIK
-------
    python lightgbmV2.py
    python lightgbmV2.py --horizons 10d
    python lightgbmV2.py --target abs
    python lightgbmV2.py --exclude-strategie bot_00db
    python lightgbmV2.py --exclude-strategie bot_00db,bot_00vcp

DATABASE
--------
Environment variable: SUPABASE_DB_URL

OUTPUT (results/)
-----------------
    lightgbmV2_<h>_model.pkl              (bundel, zie hierboven)
    lightgbmV2_<h>_feature_importance.csv
    lightgbmV2_<h>_correlations.csv
    lightgbmV2_<h>_test_predictions.csv
    lightgbmV2_summary.txt
"""

import argparse
import datetime as dt
import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
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
MIN_RIJEN_PER_DATUM = 10      # voor median-target en per-datum metrics
MIN_DATUMS = 20               # minder verschillende datums = geen split
DATUMS_WAARSCHUWING = 60      # minder dan dit = waarschuwing, geen fout

MAX_BOMEN = 1000              # met early stopping
BOMEN_ZONDER_VALIDATIE = 400  # V1-waarde, fallback
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

# Schaalvrij controlemodel: zonder absolute prijsniveaus.
FEATURE_COLUMNS_RELATIEF = [
    "atr14_pct",
    "rsi14",
    "ibs",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "pct_from_high52w",
]

BASELINE_KOLOM = "pct_from_ma50"     # laagste waarde = geselecteerd

AUC_VERSCHIL_WAARSCHUWING = 0.10


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
    """'30d' -> 30 (handelsdagen)."""
    return int(str(horizon).lower().replace("d", ""))


def embargo_kalenderdagen(horizon):
    """Handelsdagen -> kalenderdagen (x 7/5) + buffer voor feestdagen."""
    return int(np.ceil(horizon_dagen(horizon) * 7 / 5)) + 2


# ============================================================
# DATABASE
# ============================================================

def heeft_kolom(conn, tabel, kolom):
    """Controleert via information_schema of een kolom bestaat."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s LIMIT 1",
            (tabel, kolom),
        )
        return cur.fetchone() is not None


def bouw_query(met_strategie, uitsluiten):
    """
    Bouwt de JOIN-query.

    met_strategie: forward_returns heeft een kolom `strategie`
        -> deterministische dedup (ORDER BY ... strategie) en optionele
           uitsluiting van strategieen VOOR de dedup.
    """
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
# DATA VOORBEREIDEN
# ============================================================

def prepare_horizon_data(df, horizon, target_mode):
    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(f"Doelkolom ontbreekt: {target_column}")

    work = df.copy()
    work["datum"] = pd.to_datetime(work["datum"], errors="coerce")

    # Eén observatie per ticker/datum.
    work = work.drop_duplicates(subset=["ticker", "datum"], keep="first")

    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.dropna(subset=[target_column, "datum"])

    for feature in FEATURE_COLUMNS:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    work = work.dropna(subset=FEATURE_COLUMNS)

    # Absoluut label (V1-target). Blijft altijd beschikbaar als diagnose.
    work["is_profitable"] = (work[target_column] > 0).astype(int)

    if target_mode == "excess":
        # Marktneutraal: boven de mediaan van dezelfde datum.
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
            f"Minimaal {MIN_RIJEN_TRAINING + MIN_RIJEN_TEST} vereist "
            f"(waarschijnlijk is {horizon} nog niet genoeg gevuld in "
            f"forward_returns)."
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

    # V1 (zonder embargo):
    # train_df = df[df["datum"] <= split_date].copy()
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
    """
    Splitst de trainingsset chronologisch in fit + validatie (voor early
    stopping). Geeft (train_df, None) terug als een degelijke validatieset
    niet mogelijk is; dan wordt met een vast aantal bomen getraind.
    """
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
# CORRELATIE (alleen TRAINING)
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

    # Pearson is gevoelig voor uitschieters (bv. split-artefacten);
    # Spearman is hier de betrouwbaardere maat.
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
        num_leaves=31,        # eventueel lager (15) bij weinig data
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
    """
    Gemiddelde AUC en gemiddelde Spearman IC (prob vs forward return),
    berekend PER DATUM en daarna gemiddeld. Dit meet cross-sectionele
    selectiekracht, los van de richting van de markt.
    """
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
    """Traint op TRAIN (met early stopping indien mogelijk), evalueert op TEST."""

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
    """Gemiddelde, mediaan en getrimd gemiddelde van de top-n."""
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

    # --------------------------------------------------------
    # VOLLEDIG MODEL
    # --------------------------------------------------------
    print(f"\n[{horizon}] Volledig LightGBM-model trainen...")

    model, test_predictions, metrics, best_iteration = train_model(
        train_df, test_df, FEATURE_COLUMNS, target_column, horizon, embargo
    )

    n_top = max(1, int(len(test_predictions) * TOP_N_FRACTIE))

    top_model = top_n_stats(
        test_predictions, "model_probability", target_column, n_top, True
    )

    # Baseline: laagste pct_from_ma50 (mean-reversion, zoals xgboostV3).
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

    # --------------------------------------------------------
    # CONTROLEMODEL (schaalvrij)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # HERTRAINEN OP ALLE DATA (productiemodel)
    # --------------------------------------------------------
    n_full = max(50, int(best_iteration * 1.10))
    model_full = create_lightgbm_model(n_full)
    model_full.fit(work[FEATURE_COLUMNS], work["target"])

    # --------------------------------------------------------
    # OPSLAAN
    # --------------------------------------------------------
    bundle = {
        "versie": MODEL_VERSIE,
        "horizon": horizon,
        "target_mode": target_mode,
        "features": list(FEATURE_COLUMNS),
        "split_date": str(split_date),
        "embargo": bool(embargo),
        "best_iteration": best_iteration,
        "uitgesloten_strategieen": list(uitgesloten),
        "model": model,              # getraind op TRAIN
        "model_full": model_full,    # getraind op alle rijen
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

    # --------------------------------------------------------
    # AUC-VERSCHIL
    # --------------------------------------------------------
    if not pd.isna(metrics["auc"]) and not pd.isna(control_metrics["auc"]):
        auc_delta = metrics["auc"] - control_metrics["auc"]
    else:
        auc_delta = np.nan

    auc_waarschuwing = (
        (not pd.isna(auc_delta)) and abs(auc_delta) > AUC_VERSCHIL_WAARSCHUWING
    )

    # --------------------------------------------------------
    # PRINTEN
    # --------------------------------------------------------
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
            f"het volledige model leunt mogelijk op absolute prijsniveaus "
            f"(ticker-identiteit) in plaats van op echte signalen."
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
    """Resultaat-dict voor een horizon die niet getraind kon worden."""
    leeg = {
        "horizon": horizon,
        "status": "fout",
        "target_mode": target_mode,
        "n_total": 0,
        "n_train": 0,
        "n_test": 0,
        "n_tickers": 0,
        "split_date": "",
        "auc_waarschuwing": False,
        "top_n": 0,
        "model_file": "",
        "fout": str(fout),
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
# SUMMARY
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

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt.")

    print("\nVerbinding maken met Supabase...")
    conn = psycopg2.connect(db_url)

    try:
        df = get_training_data(conn, uitsluiten)
    finally:
        conn.close()

    results = []

    for horizon in horizons:
        try:
            results.append(
                train_horizon(df, horizon, args.target, embargo, uitsluiten)
            )
        except Exception as e:
            print()
            print(f"ERROR [{horizon}]: {e}")
            results.append(fout_resultaat(horizon, args.target, e))

    write_summary(results, args)

    # V1 eiste dat ALLE drie modellen bestonden en gooide anders een
    # RuntimeError. Dat laten we los: een horizon die nog niet genoeg
    # gelabelde data heeft (bv. 60d) mag de rest niet blokkeren.
    #
    # required_models = [... 10d, 30d, 60d ...]
    # missing_models = [p for p in required_models if not os.path.isfile(p)]
    # if missing_models:
    #     raise RuntimeError("Niet alle drie LightGBM-modellen zijn aangemaakt.")

    getraind = [r for r in results if r["status"] == "getraind"]

    print()
    print("=" * 70)
    print("LIGHTGBMV2 RUN VOLTOOID")
    print("=" * 70)

    for r in results:
        if r["status"] == "getraind":
            print(
                f"{r['horizon']:>4} | getraind | "
                f"AUC={fmt(r['auc'])} | "
                f"AUC/datum={fmt(r['auc_datum'])} | "
                f"IC/datum={fmt(r['ic_datum'])} | "
                f"control={fmt(r['control_auc'])}"
            )
        else:
            print(f"{r['horizon']:>4} | fout     | {r['fout']}")

    print(f"\nOutput: {RESULTS_DIR}/")

    if not getraind:
        raise SystemExit("Geen enkele horizon kon getraind worden.")


if __name__ == "__main__":
    main()
