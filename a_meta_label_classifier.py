#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_meta_label_classifier.py — META-LABELING CLASSIFIER v1.8 (Fantasie-versie)

Doel: per historisch signaal leren voorspellen of de trade slaagt
(forward return > +threshold) of faalt (< -threshold). De voorspelde
kans wordt gebruikt als filter.

BELANGRIJK: dit script voorspelt NIETS. Het rapporteert eerlijk of
het filter iets toevoegt t.o.v. de baseline.

v1.8 wijziging t.o.v. v1.7:
  - nieuwe CLI-arg --train-tot YYYY-MM-DD: harde grens op de
    trainingsset. Testset = alle datums > cutoff. Zonder deze arg
    werkt het script exact als v1.7 (rolling walk-forward).
  - versie-string bevat 'oos<YYYYMMDD>' als cutoff gezet is.
  - per-fold markering '(OOS)' om verwarring te voorkomen.

v1.7 wijziging t.o.v. v1.6:
  - aparte walk-forward per strategie (strat_*-dummies eruit)
  - rapport: GECOMBINEERD + PER STRATEGIE + SAMENVATTING

v1.6 wijziging t.o.v. v1.5:
  - maak_features() sluit 'score' uit base_features (bug: score 2x)
  - vangnet: dedupliceer X-kolommen

v1.5 wijziging t.o.v. v1.4:
  - maak_features() prefix alle X-kolommen met 'F_'

v1.4 wijzigingen t.o.v. v1.3:
  - max_dd in PROCENTPUNTEN via cumsum
  - ruwe + gewinsoriseerde cijfers
  - per-fold: gewogen + ongewogen AUC, folds n<20 gemarkeerd
  - duplicaat-rapport, feature-importance debug

v1.3 wijzigingen t.o.v. v1.2:
  - per-fold AUC/IC + folds met AUC>0.5
  - feature importance per fold
  - walk_forward retourneert (resultaten_df, coef_df)

v1.2 wijzigingen t.o.v. v1.0:
  - 'datum' tz-naive Timestamp
  - walk_forward conversie bovenaan
  - schrijf_scores Timestamp -> date

Env:
  SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID,
  EMAIL_USER/EMAIL_PASS/EMAIL_RECEIVER (optioneel).
"""

import os
import sys
import argparse
import uuid
import html
import smtplib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from features import MODEL_FEATURES

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")

GENERIEKE_TECHNICALS = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]

DEFAULT_HORIZON = 10
DEFAULT_THRESHOLD = 3.0
DEFAULT_EMBARGO_EXTRA = 5
DEFAULT_MIN_TRAIN = 200
DEFAULT_MIN_TEST = 20
DEFAULT_DREMPELS = [0.55, 0.60, 0.65]
DEFAULT_WINSOR_PCT = 30.0
KLEINE_FOLD_DREMPEL = 20


# ============================================================
# LOKALE HULPFUNCTIES
# ============================================================

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Secrets ontbreken, overslaan.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        print("[Telegram] Bericht verzonden.")
    except Exception as e:
        print(f"[Telegram] Fout: {e}")

def send_email(subject: str, body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        print("[Email] Secrets ontbreken, overslaan.")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print("[Email] Bericht verzonden.")
    except Exception as e:
        print(f"[Email] Fout: {e}")


# ============================================================
# DATA
# ============================================================

def haal_data(conn, features: List[str], horizon: int) -> pd.DataFrame:
    s_feats = [f for f in features if f not in GENERIEKE_TECHNICALS]
    g_feats = [f for f in features if f in GENERIEKE_TECHNICALS]
    s_lijst = ", ".join(f"s.{k}" for k in s_feats)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g_feats)) if g_feats else ""
    target = f"fwd_ret_{horizon}d"
    query = f"""
        SELECT s.ticker, s.datum, s.strategie, s.score,
               {s_lijst}{g_lijst},
               f.{target} AS fwd_ret
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE f.{target} IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def bouw_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out = out[out["fwd_ret"].notna()].copy()
    out["label"] = np.nan
    out.loc[out["fwd_ret"] >  threshold, "label"] = 1.0
    out.loc[out["fwd_ret"] < -threshold, "label"] = 0.0
    return out.dropna(subset=["label"]).copy()


def cross_sectionele_rank(df: pd.DataFrame, kolommen: List[str]) -> pd.DataFrame:
    out = df.copy()
    for k in kolommen:
        out[k] = out.groupby("datum")[k].rank(pct=True)
    return out


# ============================================================
# MODEL
# ============================================================

def maak_features(df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    out = df.copy()
    strat_dummies = pd.get_dummies(out["strategie"], prefix="strat", dtype=float)
    bf = [f for f in base_features if f != "score"]
    X = pd.concat([
        out[bf].reset_index(drop=True),
        strat_dummies.reset_index(drop=True),
        out["score"].fillna(0.0).reset_index(drop=True),
    ], axis=1)
    X.columns = [f"F_{c}" for c in X.columns]
    if not X.columns.is_unique:
        dup = X.columns[X.columns.duplicated()].unique().tolist()
        print(f"[WARN] maak_features: dubbele kolommen {dup} — wordt gededupliceerd")
        X = X.loc[:, ~X.columns.duplicated()]
    return X


def walk_forward(df: pd.DataFrame, feature_cols: List[str],
                 embargo_dagen: int, min_train: int, min_test: int,
                 model_type: str,
                 train_tot: Optional[pd.Timestamp] = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)
    df["datum"] = df["datum"].dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["datum"]).reset_index(drop=True)

    df = df.sort_values("datum").reset_index(drop=True)
    datums = sorted(df["datum"].unique())

    # v1.8: welke datums worden getest?
    if train_tot is not None:
        test_datums = [d for d in datums if d > train_tot]
    else:
        test_datums = datums  # rolling: alles testen

    resultaten = []
    coef_lijst = []
    eerste_coef_poging = False

    for test_datum in test_datums:
        test_datum = pd.Timestamp(test_datum)
        # v1.8: alleen historische datums vóór test_datum, embargo
        train_datums = [
            d for d in datums
            if d < test_datum and (test_datum - d).days > embargo_dagen
        ]
        # v1.8: harde grens op trainingsset indien opgegeven
        if train_tot is not None:
            train_datums = [d for d in train_datums if d <= train_tot]
        if not train_datums:
            continue

        train = df[df["datum"].isin(train_datums)].dropna(subset=feature_cols + ["label"])
        test = df[df["datum"] == test_datum].dropna(subset=feature_cols + ["label"])

        if len(train) < min_train or len(test) < min_test:
            continue
        if train["label"].nunique() < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[feature_cols].values)
        X_te = scaler.transform(test[feature_cols].values)

        if model_type == "xgb":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                objective="binary:logistic", eval_metric="auc", n_jobs=-1,
            )
            model.fit(X_tr, train["label"].values)
        else:
            model = LogisticRegression(C=1.0, max_iter=2000,
                                       class_weight="balanced", solver="lbfgs")
            model.fit(X_tr, train["label"].values)

        proba = model.predict_proba(X_te)[:, 1]
        if len(np.unique(proba)) < 2:
            continue

        try:
            if model_type == "xgb":
                coefs = np.asarray(model.feature_importances_, dtype=float)
            else:
                coefs = np.asarray(
                    model.coef_[0] if model.coef_.ndim > 1 else model.coef_,
                    dtype=float,
                )
            if len(coefs) == len(feature_cols):
                coef_lijst.append(pd.Series(coefs, index=feature_cols, name=test_datum))
            else:
                if not eerste_coef_poging:
                    print(f"[WARN] fold {test_datum.date()}: "
                          f"len(coefs)={len(coefs)} != len(feature_cols)={len(feature_cols)}")
                    eerste_coef_poging = True
        except Exception as e:
            if not eerste_coef_poging:
                print(f"[WARN] fold {test_datum.date()}: coef-extractie faalde: "
                      f"{type(e).__name__}: {e}")
                eerste_coef_poging = True

        try:
            ic = float(stats.spearmanr(proba, test["fwd_ret"].values).correlation)
        except Exception:
            ic = float("nan")
        try:
            auc = float(roc_auc_score(test["label"].values, proba))
        except Exception:
            auc = float("nan")

        test = test.assign(proba=proba, fold_datum=test_datum, fold_ic=ic, fold_auc=auc)
        resultaten.append(test)

    res_df = pd.concat(resultaten, ignore_index=True) if resultaten else pd.DataFrame()
    coef_df = pd.DataFrame(coef_lijst) if coef_lijst else pd.DataFrame()
    return res_df, coef_df


def walk_forward_per_strategie(df: pd.DataFrame, feature_cols: List[str],
                                embargo_dagen: int, min_train: int, min_test: int,
                                model_type: str,
                                train_tot: Optional[pd.Timestamp] = None
                                ) -> Tuple[Dict[str, pd.DataFrame],
                                          Dict[str, pd.DataFrame]]:
    """v1.7: aparte walk-forward per strategie, strat_*-dummies eruit."""
    resultaten: Dict[str, pd.DataFrame] = {}
    coefs: Dict[str, pd.DataFrame] = {}
    sf = [f for f in feature_cols if not f.startswith("F_strat_")]
    for strat in sorted(df["strategie"].unique()):
        sub = df[df["strategie"] == strat].copy()
        print(f"\n  --- strategie {strat} (n={len(sub)}) ---")
        res, cf = walk_forward(sub, sf, embargo_dagen, min_train, min_test,
                               model_type, train_tot)
        if res.empty:
            print(f"  {strat}: geen folds met resultaat.")
        else:
            print(f"  {strat}: {len(res)} voorspellingen over "
                  f"{res['fold_datum'].nunique()} folds, "
                  f"AUC={res['fold_auc'].mean():+.4f}, "
                  f"IC={res['fold_ic'].mean():+.4f}")
        resultaten[strat] = res
        coefs[strat] = cf
    return resultaten, coefs


# ============================================================
# EVALUATIE
# ============================================================

def evalueer(df: pd.DataFrame, drempel: float,
             winsor_pct: float = DEFAULT_WINSOR_PCT) -> Dict:
    if df.empty:
        return {}

    def _stats(arr, label):
        if len(arr) < 2:
            return {"n": len(arr), "label": label,
                    "mean": None, "sharpe": None,
                    "hit": None, "max_dd_pp": None, "pf": None}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        sharpe = mean / std if std > 1e-9 else None
        hit = float(np.mean(arr > 0))
        eq = np.cumsum(arr)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd_pp = float(dd.min())
        gains = arr[arr > 0].sum()
        losses = -arr[arr < 0].sum()
        pf = float(gains / losses) if losses > 1e-9 else None
        return {"n": len(arr), "label": label, "mean": mean, "sharpe": sharpe,
                "hit": hit, "max_dd_pp": max_dd_pp, "pf": pf}

    baseline = df["fwd_ret"].values
    gefilterd = df.loc[df["proba"] >= drempel, "fwd_ret"].values
    baseline_w = np.clip(baseline, -winsor_pct, winsor_pct)
    gefilterd_w = np.clip(gefilterd, -winsor_pct, winsor_pct)

    return {
        "baseline":   _stats(baseline,    "baseline (ruw)"),
        "baseline_w": _stats(baseline_w,  f"baseline (winsor ±{winsor_pct:.0f}%)"),
        "filter":     _stats(gefilterd,   "filter   (ruw)"),
        "filter_w":   _stats(gefilterd_w, f"filter   (winsor ±{winsor_pct:.0f}%)"),
    }


def _print_stats(s: Dict) -> None:
    if s["mean"] is None:
        print(f"    {s['label']:<30} n={s['n']:>4}  te weinig data")
        return
    print(f"    {s['label']:<30} n={s['n']:>4}  "
          f"mean={s['mean']:+.2f}%  sharpe={s['sharpe']:+.3f}  "
          f"hit={s['hit']:.1%}  max_dd={s['max_dd_pp']:+.1f}pp  pf={s['pf']:.2f}")


# ============================================================
# DIAGNOSTIEK
# ============================================================

def print_per_fold(res: pd.DataFrame, klein_drempel: int = KLEINE_FOLD_DREMPEL,
                   oos: bool = False) -> None:
    print("\nPer-fold AUC/IC:")
    folds_df = (
        res.groupby("fold_datum")
           .agg(n=("label", "size"),
                auc=("fold_auc", "first"),
                ic=("fold_ic", "first"))
           .sort_index()
    )
    for datum, row in folds_df.iterrows():
        mark = "  (klein)" if row["n"] < klein_drempel else ""
        oos_mark = "  (OOS)" if oos else ""
        print(f"  {datum.date()}  n={int(row['n']):>4}  "
              f"AUC={row['auc']:+.3f}  IC={row['ic']:+.4f}{oos_mark}{mark}")

    geldig = folds_df[folds_df["n"] >= klein_drempel]
    n_tot = len(geldig)
    if n_tot == 0:
        print(f"\n  Geen folds met n>={klein_drempel}.")
        return
    n_pos = int((geldig["auc"] > 0.5).sum())
    ong = float(geldig["auc"].mean())
    gew = float((geldig["auc"] * geldig["n"]).sum() / geldig["n"].sum())
    print(f"\n  Folds met AUC > 0.5 (n>={klein_drempel}): "
          f"{n_pos}/{n_tot} ({100*n_pos/n_tot:.1f}%)")
    print(f"  Gemiddelde AUC  (ongewogen, n>={klein_drempel}): {ong:+.4f}")
    print(f"  Gemiddelde AUC  (gewogen naar n):              {gew:+.4f}")


def print_feature_importance(coef_df: pd.DataFrame, model_type: str) -> None:
    if coef_df.empty:
        print("\nFeature importance: geen coëfficiënten verzameld.")
        return

    print("\nFeature importance (gemiddeld over folds):")
    if model_type == "xgb":
        gem = coef_df.mean().sort_values(ascending=False)
        for feat, val in gem.head(15).items():
            print(f"  {feat:<28} gain={val:.4f}")
        return

    gem_abs = coef_df.abs().mean().sort_values(ascending=False)
    print("  Top 10 op |coef|:")
    for feat, val in gem_abs.head(10).items():
        print(f"  {feat:<28} |coef|={val:.4f}")

    gem_signed = coef_df.mean()
    order = gem_signed.abs().sort_values(ascending=False).index
    print("\n  Richting (gemiddelde coef, + = voorspelt 'wint'):")
    for feat in order[:10]:
        print(f"  {feat:<28} coef={gem_signed[feat]:+.4f}")


# ============================================================
# DATABASE WRITE
# ============================================================

def schrijf_scores(conn, df: pd.DataFrame, versie: str, horizon: int):
    insert = """
        INSERT INTO meta_label_scores
            (datum, ticker, strategie, horizon_dagen, model_versie, proba)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (datum, ticker, strategie, horizon_dagen, model_versie)
        DO UPDATE SET proba = EXCLUDED.proba, created_at = now();
    """
    rijen = [
        (
            r["datum"].date() if hasattr(r["datum"], "date") else r["datum"],
            r["ticker"], r["strategie"], horizon, versie, float(r["proba"])
        )
        for _, r in df.iterrows()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, insert, rijen, page_size=500)
    conn.commit()
    print(f"{len(rijen)} scores opgeslagen in meta_label_scores.")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--embargo-extra", type=int, default=DEFAULT_EMBARGO_EXTRA)
    ap.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    ap.add_argument("--min-test", type=int, default=DEFAULT_MIN_TEST)
    ap.add_argument("--model", choices=["logreg", "xgb"], default="logreg")
    ap.add_argument("--strategieen", type=str,
                    default="bot_00xxxV2,bot_00vcp,bot_00mr,bot_00dm,bot_00ms")
    ap.add_argument("--winsor-pct", type=float, default=DEFAULT_WINSOR_PCT,
                    help="Cap op fwd_ret in %% voor gewinsoriseerde evaluatie")
    ap.add_argument("--train-tot", type=str, default=None,
                    help="Train alleen op datums <= YYYY-MM-DD (out-of-sample test). "
                         "Zonder deze arg: rolling walk-forward.")
    ap.add_argument("--geen-db-write", action="store_true")
    ap.add_argument("--geen-notify", action="store_true")
    args = ap.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    # v1.8: parse cutoff
    train_tot: Optional[pd.Timestamp] = None
    if args.train_tot:
        try:
            train_tot = pd.Timestamp(args.train_tot)
            if train_tot.tzinfo is not None:
                train_tot = train_tot.tz_convert(None)
        except Exception as e:
            print(f"FOUT: --train-tot '{args.train_tot}' is geen geldige datum: {e}",
                  file=sys.stderr)
            sys.exit(1)

    strategieen = [s.strip() for s in args.strategieen.split(",") if s.strip()]
    embargo = args.horizon + args.embargo_extra
    oos_tag = f"_oos{train_tot.strftime('%Y%m%d')}" if train_tot is not None else ""
    versie = f"meta_label_{args.model}_h{args.horizon}_t{args.threshold}{oos_tag}_" \
             f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"

    print(f"Versie: {versie}")
    print(f"Strategieën: {strategieen}")
    print(f"Horizon: {args.horizon}d, threshold: ±{args.threshold}%, embargo: {embargo}d, "
          f"winsor: ±{args.winsor_pct:.0f}%")
    if train_tot is not None:
        print(f"OOS-modus: train_tot={train_tot.date()} "
              f"(test = alle datums > {train_tot.date()})")
    else:
        print("Modus: rolling walk-forward (geen OOS-cutoff)")

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = haal_data(conn, list(MODEL_FEATURES), args.horizon)
        print(f"{len(df)} rijen opgehaald.")
        if df.empty:
            print("Geen data.")
            return

        df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)
        df["datum"] = df["datum"].dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=["datum"]).reset_index(drop=True)

        n_rijen = len(df)
        n_uniek = df[["ticker", "datum"]].drop_duplicates().shape[0]
        if n_uniek > 0:
            print(f"Unieke (ticker, datum): {n_uniek} "
                  f"(rijen: {n_rijen}, factor {n_rijen/n_uniek:.2f}x)")

        df = df[df["strategie"].isin(strategieen)].copy()
        print(f"{len(df)} rijen na strategie-filter.")

        df = bouw_labels(df, args.threshold)
        print(f"{len(df)} rijen met label (middenzone uitgesloten).")
        if df.empty:
            print("Geen labels.")
            return

        base_features = [f for f in MODEL_FEATURES if f in df.columns]
        df = cross_sectionele_rank(df, base_features)
        X = maak_features(df, base_features)
        feature_cols = list(X.columns)
        df = pd.concat([df.reset_index(drop=True), X], axis=1)

        print(f"Features: {len(feature_cols)} kolommen.")

        oos_label = "OOS" if train_tot is not None else "rolling"

        # =========================================================
        # GECOMBINEERD
        # =========================================================
        print("\n============================================================")
        print(f"GECOMBINEERD ({oos_label})")
        print("============================================================")
        res_all, coefs_all = walk_forward(
            df, feature_cols, embargo, args.min_train, args.min_test,
            args.model, train_tot,
        )
        if res_all.empty:
            print("Geen folds met resultaat (gecombineerd).")
        else:
            print(f"\n{len(res_all)} voorspellingen over "
                  f"{res_all['fold_datum'].nunique()} folds.")
            print(f"Gemiddelde IC:  {res_all['fold_ic'].mean():+.4f}")
            print(f"Gemiddelde AUC: {res_all['fold_auc'].mean():+.4f}")
            print_per_fold(res_all, oos=(train_tot is not None))
            print_feature_importance(coefs_all, args.model)
            print(f"\nEvaluatie per drempel (gecombineerd, {oos_label}):")
            for d in DEFAULT_DREMPELS:
                ev = evalueer(res_all, d, args.winsor_pct)
                print(f"\n  drempel {d:.2f}:")
                for key in ("baseline", "baseline_w", "filter", "filter_w"):
                    _print_stats(ev[key])

        # =========================================================
        # PER STRATEGIE
        # =========================================================
        print("\n\n============================================================")
        print(f"PER STRATEGIE ({oos_label}, strat-dummies verwijderd)")
        print("============================================================")
        res_per_strat, coefs_per_strat = walk_forward_per_strategie(
            df, feature_cols, embargo, args.min_train, args.min_test,
            args.model, train_tot,
        )

        geldige_strats = [s for s, r in res_per_strat.items() if not r.empty]
        if not geldige_strats:
            print("\nGeen enkele strategie leverde folds op.")
        else:
            for strat in geldige_strats:
                res = res_per_strat[strat]
                coefs = coefs_per_strat[strat]
                print(f"\n\n--- {strat} — detail ---")
                print(f"Gemiddelde IC:  {res['fold_ic'].mean():+.4f}")
                print(f"Gemiddelde AUC: {res['fold_auc'].mean():+.4f}")
                print_per_fold(res, oos=(train_tot is not None))
                print_feature_importance(coefs, args.model)
                print(f"\nEvaluatie per drempel ({strat}, {oos_label}):")
                for d in DEFAULT_DREMPELS:
                    ev = evalueer(res, d, args.winsor_pct)
                    print(f"\n  drempel {d:.2f}:")
                    for key in ("baseline", "baseline_w", "filter", "filter_w"):
                        _print_stats(ev[key])

            # Samenvattingstabel per strategie
            print("\n\n=== SAMENVATTING PER STRATEGIE ===")
            print(f"{'strategie':<20} {'n':>6} {'folds':>6} {'AUC':>8} {'IC':>8} "
                  f"{'AUC>0.5':>10}")
            for strat in geldige_strats:
                res = res_per_strat[strat]
                folds = (
                    res.groupby("fold_datum")
                       .agg(n=("label", "size"), auc=("fold_auc", "first"))
                )
                g = folds[folds["n"] >= KLEINE_FOLD_DREMPEL]
                n_pos = int((g["auc"] > 0.5).sum())
                n_tot = len(g)
                print(f"{strat:<20} {len(res):>6} {res['fold_datum'].nunique():>6} "
                      f"{res['fold_auc'].mean():>+8.4f} "
                      f"{res['fold_ic'].mean():>+8.4f} "
                      f"{n_pos:>4}/{n_tot:<4}")

        # DB-write: gecombineerd
        if not res_all.empty and not args.geen_db_write:
            schrijf_scores(conn, res_all, versie, args.horizon)

        if not args.geen_notify:
            samenvatting = (
                f"Meta-labeling {versie}\n"
                f"Strategieën: {strategieen}\n"
                f"Modus: {oos_label}\n"
            )
            if not res_all.empty:
                samenvatting += (
                    f"Gecombineerd: AUC={res_all['fold_auc'].mean():.3f}, "
                    f"IC={res_all['fold_ic'].mean():+.3f}\n"
                )
            for strat in geldige_strats:
                res = res_per_strat[strat]
                samenvatting += (
                    f"{strat}: AUC={res['fold_auc'].mean():.3f}, "
                    f"IC={res['fold_ic'].mean():+.3f}\n"
                )
            send_telegram(samenvatting)
            send_email(f"Meta-labeling {versie}", samenvatting)

        print("\nKlaar.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
