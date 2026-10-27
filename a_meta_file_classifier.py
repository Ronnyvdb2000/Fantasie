#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_meta_label_classifier.py — META-LABELING CLASSIFIER v1.1 (Fantasie-versie)

Doel: per historisch signaal leren voorspellen of de trade slaagt
(forward return > +threshold) of faalt (< -threshold). De voorspelde
kans wordt gebruikt als filter: alleen signalen met proba >= drempel
worden doorgelaten.

BELANGRIJK: dit script voorspelt NIETS. Het bouwt een model, past het
walk-forward toe, en rapporteert eerlijk of het filter iets toevoegt
t.o.v. de baseline. Het kan ook slechter presteren.

Deze versie is standalone (geen common/ package nodig) en werkt direct
in de originele Fantasie-repo. Het gebruikt features.py voor MODEL_FEATURES.

v1.1 (2026-10-27): fix voor TypeError bij walk_forward. De kolom
`datum` wordt nu meteen na het ophalen geconverteerd naar datetime.
Bij het wegschrijven naar meta_label_scores wordt hij teruggezet
naar date.

Env:
  SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID,
  EMAIL_USER/EMAIL_PASS/EMAIL_RECEIVER (optioneel).
"""

import os
import sys
import argparse
import uuid
import smtplib
from datetime import datetime, timezone
from typing import List, Optional, Dict

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

# features.py bestaat in de originele repo
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


# ============================================================
# LOKALE HULPFUNCTIES (vervangen common/)
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
    out.loc[out["fwd_ret"] > threshold, "label"] = 1.0
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
    X = pd.concat([
        out[base_features].reset_index(drop=True),
        strat_dummies.reset_index(drop=True),
        out["score"].fillna(0.0).reset_index(drop=True),
    ], axis=1)
    return X


def walk_forward(df: pd.DataFrame, feature_cols: List[str],
                 embargo_dagen: int, min_train: int, min_test: int,
                 model_type: str) -> pd.DataFrame:
    df = df.sort_values("datum").reset_index(drop=True)
    datums = sorted(df["datum"].unique())
    resultaten = []

    for i in range(len(datums)):
        test_datum = datums[i]
        train_datums = [d for d in datums[:i] if (test_datum - d).days > embargo_dagen]
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
            model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
            model.fit(X_tr, train["label"].values)

        proba = model.predict_proba(X_te)[:, 1]
        if len(np.unique(proba)) < 2:
            continue

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

    return pd.concat(resultaten, ignore_index=True) if resultaten else pd.DataFrame()


# ============================================================
# EVALUATIE: BASELINE VS FILTER
# ============================================================

def evalueer(df: pd.DataFrame, drempel: float) -> Dict:
    if df.empty:
        return {}

    baseline = df["fwd_ret"].values
    gefilterd = df.loc[df["proba"] >= drempel, "fwd_ret"].values

    def _stats(arr):
        if len(arr) < 2:
            return {"n": len(arr), "mean": None, "sharpe": None, "hit": None, "max_dd": None, "pf": None}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        sharpe = mean / std if std > 1e-9 else None
        hit = float(np.mean(arr > 0))
        eq = np.cumsum(arr)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak == 0, 1, peak)
        max_dd = float(dd.min())
        gains = arr[arr > 0].sum()
        losses = -arr[arr < 0].sum()
        pf = float(gains / losses) if losses > 1e-9 else None
        return {"n": len(arr), "mean": mean, "sharpe": sharpe, "hit": hit, "max_dd": max_dd, "pf": pf}

    return {"baseline": _stats(baseline), "filter": _stats(gefilterd)}


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
            r["ticker"],
            r["strategie"],
            horizon,
            versie,
            float(r["proba"]),
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
    ap.add_argument("--geen-db-write", action="store_true")
    ap.add_argument("--geen-notify", action="store_true")
    args = ap.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    strategieen = [s.strip() for s in args.strategieen.split(",") if s.strip()]
    embargo = args.horizon + args.embargo_extra
    versie = (
        f"meta_label_{args.model}_h{args.horizon}_t{args.threshold}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    )

    print(f"Versie: {versie}")
    print(f"Strategieën: {strategieen}")
    print(f"Horizon: {args.horizon}d, threshold: ±{args.threshold}%, embargo: {embargo}d")

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = haal_data(conn, list(MODEL_FEATURES), args.horizon)
        print(f"{len(df)} rijen opgehaald.")
        if df.empty:
            print("Geen data.")
            return

        # FIX: converteer datum van string naar datetime
        df["datum"] = pd.to_datetime(df["datum"]).dt.tz_localize(None).dt.normalize()

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

        res = walk_forward(df, feature_cols, embargo, args.min_train, args.min_test, args.model)
        if res.empty:
            print("Geen folds met resultaat.")
            return

        print(f"\n{len(res)} voorspellingen over {res['fold_datum'].nunique()} folds.")
        print(f"Gemiddelde IC: {res['fold_ic'].mean():+.4f}")
        print(f"Gemiddelde AUC: {res['fold_auc'].mean():+.4f}")

        print("\nEvaluatie per drempel:")
        for d in DEFAULT_DREMPELS:
            ev = evalueer(res, d)
            b, f = ev["baseline"], ev["filter"]
            print(f"\n  drempel {d:.2f}:")
            print(
                f"    baseline n={b['n']}, mean={b['mean']:+.2f}%, "
                f"sharpe={b['sharpe']:+.3f}, hit={b['hit']:.1%}, "
                f"max_dd={b['max_dd']:.1%}, pf={b['pf']:.2f}"
            )
            print(
                f"    filter   n={f['n']}, mean={f['mean']:+.2f}%, "
                f"sharpe={f['sharpe']:+.3f}, hit={f['hit']:.1%}, "
                f"max_dd={f['max_dd']:.1%}, pf={f['pf']:.2f}"
            )

        if not args.geen_db_write:
            schrijf_scores(conn, res, versie, args.horizon)

        if not args.geen_notify:
            samenvatting = (
                f"Meta-labeling {versie}\n"
                f"Folds: {res['fold_datum'].nunique()}\n"
                f"Gem. AUC: {res['fold_auc'].mean():.3f}\n"
                f"Gem. IC: {res['fold_ic'].mean():+.3f}\n\n"
                "Zie console voor baseline vs filter."
            )
            send_telegram(samenvatting)
            send_email(f"Meta-labeling {versie}", samenvatting)

        print("\nKlaar.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
