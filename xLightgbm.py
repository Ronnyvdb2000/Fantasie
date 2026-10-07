#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lightgbmV1.py
=============

LightGBM trainingsengine gebaseerd op de methodiek van xgboostV3.

DOEL
----
Leert uit ALLE historische regels uit:
    generieke_technicals + forward_returns

Per horizon wordt een afzonderlijk LightGBM-classificatiemodel getraind.

TARGET
------
1 = forward return > 0
0 = forward return <= 0

WIJZIGINGEN (2026-10-07)
------------------------
A. main(): de controle "alle drie modellen aanwezig" is vervangen door
   een tolerante variant. Een horizon zonder genoeg data (bv. 60d) mag
   de rest niet blokkeren. De run faalt alleen als GEEN enkel model
   gemaakt kon worden.
B. Telegram + e-mail notificatie toegevoegd (HTML-opmaak).

ENV
---
SUPABASE_DB_URL (verplicht)
TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (optioneel)
EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER (optioneel)
"""

import os
import sys
import smtplib
import html
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
import psycopg2
import lightgbm as lgb
import requests

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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Notificatie-drempels
AUC_ALARM = 0.50
AUC_GOED = 0.60


# ============================================================
# OMGEVING
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip()
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "atr14", "atr14_pct", "rsi14", "ibs",
    "ma50", "ma200", "pct_from_ma50", "pct_from_ma200",
    "vol_ratio_20d", "high52w", "pct_from_high52w",
]

FEATURE_COLUMNS_RELATIEF = [
    "atr14_pct", "rsi14", "ibs",
    "pct_from_ma50", "pct_from_ma200",
    "vol_ratio_20d", "pct_from_high52w",
]

BASELINE_KOLOM = "pct_from_ma50"
AUC_VERSCHIL_WAARSCHUWING = 0.10


# ============================================================
# DATABASE QUERY
# ============================================================

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker, datum,
        fwd_ret_10d, fwd_ret_30d, fwd_ret_60d
    FROM forward_returns
    ORDER BY ticker, datum
)
SELECT
    gt.ticker, gt.datum,
    gt.atr14, gt.atr14_pct, gt.rsi14, gt.ibs,
    gt.ma50, gt.ma200,
    gt.pct_from_ma50, gt.pct_from_ma200,
    gt.vol_ratio_20d, gt.high52w, gt.pct_from_high52w,
    fr_dedup.fwd_ret_10d, fr_dedup.fwd_ret_30d, fr_dedup.fwd_ret_60d
FROM generieke_technicals gt
JOIN fr_dedup
    ON fr_dedup.ticker = gt.ticker AND fr_dedup.datum = gt.datum
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


def _esc(s):
    return html.escape(str(s))


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

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])

    print(f"Rijen opgehaald : {len(df):,}")
    print(f"Tickers         : {df['ticker'].nunique():,}")
    print(f"Van             : {df['datum'].min()}")
    print(f"Tot             : {df['datum'].max()}")
    return df


# ============================================================
# DATA VOORBEREIDEN
# ============================================================

def prepare_horizon_data(df, horizon):
    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        raise RuntimeError(f"Doelkolom ontbreekt: {target_column}")

    work = df.copy()
    work["datum"] = pd.to_datetime(work["datum"], errors="coerce")
    work = work.drop_duplicates(subset=["ticker", "datum"], keep="first")

    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work.dropna(subset=[target_column, "datum"])

    work["is_profitable"] = (work[target_column] > 0).astype(int)

    for feature in FEATURE_COLUMNS:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")

    work = work.dropna(subset=FEATURE_COLUMNS)
    work = work.sort_values(["datum", "ticker"]).reset_index(drop=True)

    return work


# ============================================================
# CHRONOLOGISCHE SPLIT
# ============================================================

def time_split(df):
    if len(df) < (MIN_RIJEN_TRAINING + MIN_RIJEN_TEST):
        raise RuntimeError(
            f"Te weinig bruikbare rijen: {len(df)}. "
            f"Minimaal {MIN_RIJEN_TRAINING + MIN_RIJEN_TEST} vereist."
        )

    dates = df["datum"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < 2:
        raise RuntimeError("Onvoldoende verschillende handelsdatums.")

    split_index = max(0, int(len(dates) * 0.80) - 1)
    split_date = dates.iloc[split_index]

    train_df = df[df["datum"] <= split_date].copy()
    test_df = df[df["datum"] > split_date].copy()

    if len(train_df) < MIN_RIJEN_TRAINING:
        raise RuntimeError(f"Trainingsset te klein: {len(train_df)}.")
    if len(test_df) < MIN_RIJEN_TEST:
        raise RuntimeError(f"Testset te klein: {len(test_df)}.")
    if train_df["is_profitable"].nunique() < 2:
        raise RuntimeError("Trainingsset bevat slechts één klasse.")

    return train_df, test_df, split_date


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

        rows.append({
            "feature": feature,
            "pearson": pearson,
            "spearman": spearman,
            "abs_pearson": abs(pearson) if not pd.isna(pearson) else np.nan,
            "abs_spearman": abs(spearman) if not pd.isna(spearman) else np.nan,
        })

    result = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)

    filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_correlations.csv"
    )
    result.to_csv(filename, index=False)

    print()
    print(f"[{horizon}] Sterkste correlaties:")
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

def train_model(train_df, test_df, features, target_column):
    model = create_lightgbm_model()

    X_train = train_df[features]
    y_train = train_df["is_profitable"]
    X_test = test_df[features]
    y_test = test_df["is_profitable"]

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities) if y_test.nunique() >= 2 else np.nan

    result = test_df.copy()
    result["model_probability"] = probabilities
    result["model_prediction"] = predictions

    return model, result, accuracy, auc


def top_n_average(df, sort_column, target_column, n_top, descending=True):
    ordered = df.sort_values(sort_column, ascending=not descending)
    selected = ordered.head(n_top)
    if selected.empty:
        return np.nan
    return float(selected[target_column].mean())


def save_feature_importance(model, features, horizon):
    importance = pd.DataFrame({
        "feature": features,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

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
# ÉÉN HORIZON TRAINEN
# ============================================================

def train_horizon(df, horizon):
    print()
    print()
    print("=" * 70)
    print(f"LIGHTGBM — HORIZON {horizon}")
    print("=" * 70)

    target_column = f"fwd_ret_{horizon}"
    work = prepare_horizon_data(df, horizon)

    print(f"[{horizon}] Bruikbare rijen: {len(work):,}")
    print(f"[{horizon}] Tickers: {work['ticker'].nunique():,}")

    train_df, test_df, split_date = time_split(work)

    print(f"[{horizon}] TRAIN: {len(train_df):,} rijen")
    print(f"[{horizon}] TEST : {len(test_df):,} rijen")
    print(f"[{horizon}] Splitdatum: {split_date}")

    calculate_correlations(train_df, target_column, horizon)

    print(f"\n[{horizon}] Volledig LightGBM-model trainen...")
    model, test_predictions, accuracy, auc = train_model(
        train_df, test_df, FEATURE_COLUMNS, target_column,
    )

    n_top = max(1, int(len(test_predictions) * TOP_N_FRACTIE))

    top_model = top_n_average(
        test_predictions, "model_probability", target_column, n_top, True,
    )
    top_baseline = top_n_average(
        test_predictions, BASELINE_KOLOM, target_column, n_top, False,
    )
    test_average = float(test_predictions[target_column].mean())

    print(f"\n[{horizon}] Controlemodel trainen...")
    control_model, control_predictions, control_accuracy, control_auc = train_model(
        train_df, test_df, FEATURE_COLUMNS_RELATIEF, target_column,
    )

    top_control = top_n_average(
        control_predictions, "model_probability", target_column, n_top, True,
    )

    save_feature_importance(model, FEATURE_COLUMNS, horizon)

    model_filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_model.pkl"
    )
    joblib.dump(model, model_filename)

    predictions_filename = os.path.join(
        RESULTS_DIR, f"{MODEL_VERSIE}_{horizon}_test_predictions.csv"
    )
    output_columns = [
        "ticker", "datum", target_column,
        "model_probability", "model_prediction", BASELINE_KOLOM,
    ]
    test_predictions[output_columns].sort_values(
        "model_probability", ascending=False,
    ).to_csv(predictions_filename, index=False)

    if not pd.isna(auc) and not pd.isna(control_auc):
        auc_delta = auc - control_auc
    else:
        auc_delta = np.nan

    print()
    print("-" * 70)
    print(f"[{horizon}] RESULTAAT")
    print("-" * 70)
    print(f"Train              : {len(train_df):,}")
    print(f"Test               : {len(test_df):,}")
    print(f"Tickers            : {work['ticker'].nunique():,}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")
    print(f"AUC                : {fmt(auc)}")
    print(f"Controle AUC       : {fmt(control_auc)}")
    print(f"AUC verschil       : {fmt(auc_delta)}")
    print(f"Top {n_top} model       : {pct(top_model)}")
    print(f"Top {n_top} baseline   : {pct(top_baseline)}")
    print(f"Hele testset       : {pct(test_average)}")
    print(f"Top {n_top} controle : {pct(top_control)}")
    print(f"\nModel opgeslagen:\n  {model_filename}")

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
        "reden": "",
    }


def fout_resultaat(horizon, reden):
    return {
        "horizon": horizon,
        "status": "overgeslagen",
        "n_total": 0, "n_train": 0, "n_test": 0, "n_tickers": 0,
        "split_date": "", "accuracy": np.nan, "auc": np.nan,
        "control_auc": np.nan, "auc_delta": np.nan, "top_n": 0,
        "top_model": np.nan, "top_baseline": np.nan,
        "top_control": np.nan, "test_average": np.nan,
        "model_file": "", "reden": str(reden),
    }


# ============================================================
# SUMMARY
# ============================================================

def write_summary(results):
    filename = os.path.join(RESULTS_DIR, f"{MODEL_VERSIE}_summary.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("LIGHTGBMV1 TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Run: {dt.datetime.now().isoformat()}\n\n")
        for result in results:
            f.write(f"HORIZON: {result['horizon']}\n")
            f.write(f"Status: {result['status']}\n")
            for k in (
                "n_total", "n_train", "n_test", "n_tickers", "split_date",
                "accuracy", "auc", "control_auc", "auc_delta",
                "top_n", "top_model", "top_baseline", "top_control",
                "test_average", "model_file",
            ):
                v = result.get(k)
                if isinstance(v, (float, np.floating)):
                    v = safe_float(v)
                f.write(f"{k}: {v}\n")
            f.write("\n")


# ============================================================
# NOTIFICATIES — TELEGRAM
# ============================================================

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


def maak_telegram_bericht(results):
    vandaag = dt.datetime.now().strftime("%Y-%m-%d")
    n_ok = sum(1 for r in results if r["status"] == "getraind")
    n_tot = len(results)

    lijnen = [
        "🤖🤖🤖 <b>LIGHTGBM V1 — TRAININGSRAPPORT</b> 🤖🤖🤖",
        f"<i>{vandaag}</i>",
        "",
        f"✅ <b>{n_ok}/{n_tot}</b> horizons getraind",
    ]

    for r in results:
        naam = _esc(r["horizon"])
        emoji, kort = _status_emoji(r)

        lijnen.append("")
        lijnen.append(f"{emoji} <b>{naam}</b> — {_esc(kort)}")

        if r["status"] == "getraind":
            auc = fmt(r.get("auc"))
            ctrl = fmt(r.get("control_auc"))
            lijnen.append(f"  📊 AUC: <b>{_esc(auc)}</b> | controle: {_esc(ctrl)}")
            lijnen.append(
                f"  🎯 Train/Test: {r.get('n_train')}/{r.get('n_test')} "
                f"(split {r.get('split_date')})"
            )
            top_model = pct(r.get("top_model"))
            base = pct(r.get("top_baseline"))
            lijnen.append(
                f"  🥇 Top-{r.get('top_n')}: model=<b>{_esc(top_model)}</b> "
                f"| baseline={_esc(base)}"
            )
        else:
            reden = _esc(r.get("reden", ""))[:100]
            lijnen.append(f"  <i>{reden}</i>")

    return "\n".join(lijnen)


def stuur_telegram(tekst):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Niet geconfigureerd — overgeslagen.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": tekst,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"[Telegram] status {r.status_code}: {r.text[:200]}")
            return False
        print("✅ Telegrambericht verzonden.")
        return True
    except Exception as e:
        print(f"[Telegram] fout: {e}")
        return False


# ============================================================
# NOTIFICATIES — E-MAIL
# ============================================================

def maak_email_html(results):
    vandaag = dt.datetime.now().strftime("%Y-%m-%d")
    n_ok = sum(1 for r in results if r["status"] == "getraind")
    n_tot = len(results)

    rijen_html = ""
    for r in results:
        emoji, kort = _status_emoji(r)
        if r["status"] == "getraind":
            rijen_html += (
                f"<div style='border-left:6px solid #2e7d32;padding:12px;"
                f"margin-bottom:10px;background:#fff;'>"
                f"<h3 style='margin:0;'>{emoji} {r['horizon']} — {kort}</h3>"
                f"<p style='margin:6px 0 0 0;'>"
                f"AUC: <b>{fmt(r.get('auc'))}</b> | "
                f"controle: {fmt(r.get('control_auc'))}<br>"
                f"Train/Test: {r.get('n_train')}/{r.get('n_test')} "
                f"(split {r.get('split_date')})<br>"
                f"Top-{r.get('top_n')}: model=<b>{pct(r.get('top_model'))}</b> "
                f"| baseline={pct(r.get('top_baseline'))}"
                f"</p></div>"
            )
        else:
            rijen_html += (
                f"<div style='border-left:6px solid #999;padding:12px;"
                f"margin-bottom:10px;background:#f5f5f5;'>"
                f"<h3 style='margin:0;'>{emoji} {r['horizon']} — {kort}</h3>"
                f"<p style='margin:6px 0 0 0;color:#555;'>"
                f"{r.get('reden','')}</p></div>"
            )

    return f"""
    <html><body style="font-family:Arial,sans-serif;padding:20px;">
      <div style="max-width:680px;margin:0 auto;">
        <div style="background:#e3f2fd;border:3px solid #1976d2;
                    border-radius:10px;padding:20px;margin-bottom:20px;">
          <h1 style="color:#0d47a1;margin-top:0;">🤖 LIGHTGBM V1</h1>
          <p style="color:#555;margin:0;">{vandaag}</p>
          <p style="font-size:18px;margin:10px 0 0 0;">
            ✅ <b>{n_ok}/{n_tot}</b> horizons getraind
          </p>
        </div>
        {rijen_html}
      </div>
    </body></html>
    """


def stuur_email(html_body, heeft_modellen):
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        print("[E-mail] Niet geconfigureerd — overgeslagen.")
        return False

    onderwerp = (
        "🤖 LightGBM V1 — trainingsrapport"
        if heeft_modellen else
        "⚠️ LightGBM V1 — geen resultaten"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = onderwerp
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        print(f"✅ E-mail verzonden naar {EMAIL_RECEIVER}.")
        return True
    except Exception as e:
        print(f"[E-mail] fout: {e}")
        return False


def stuur_notificaties(results):
    try:
        tekst = maak_telegram_bericht(results)
        ok_tg = stuur_telegram(tekst)
    except Exception as e:
        print(f"Telegram-voorbereiding faalde: {e}", file=sys.stderr)
        ok_tg = False

    try:
        html_body = maak_email_html(results)
        n_ok = sum(1 for r in results if r["status"] == "getraind")
        ok_mail = stuur_email(html_body, heeft_modellen=(n_ok > 0))
    except Exception as e:
        print(f"E-mail-voorbereiding faalde: {e}", file=sys.stderr)
        ok_mail = False

    print(
        f"Notificaties: Telegram={'OK' if ok_tg else 'skip'}, "
        f"mail={'OK' if ok_mail else 'skip'}."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("LIGHTGBMV1 — HISTORISCHE TRAINING")
    print("=" * 70)
    print("Bron: ALLE historische generieke_technicals-regels")
    print(f"Horizon: {' / '.join(HORIZONS)}")
    print(f"Top selectie: {TOP_N_FRACTIE * 100:.0f}%")

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt.")

    print("\nVerbinding maken met Supabase...")
    conn = psycopg2.connect(db_url)
    try:
        df = get_training_data(conn)
    finally:
        conn.close()

    # --------------------------------------------------------
    # TRAINING PER HORIZON — tolerant
    # --------------------------------------------------------
    results = []
    for horizon in HORIZONS:
        try:
            result = train_horizon(df, horizon)
            results.append(result)
        except Exception as e:
            print(f"\nERROR [{horizon}]: {e}")
            results.append(fout_resultaat(horizon, e))

    write_summary(results)

    # --------------------------------------------------------
    # TOLERANTE MODEL-CONTROLE
    # --------------------------------------------------------
    print()
    print("Controleren of modellen correct zijn opgeslagen...")

    geldige_modellen = [
        r for r in results
        if r["status"] == "getraind"
        and os.path.isfile(r["model_file"])
        and os.path.getsize(r["model_file"]) > 1000
    ]

    for r in results:
        pad = r.get("model_file") or f"(geen bestand voor {r['horizon']})"
        if r["status"] == "getraind" and os.path.isfile(pad):
            print(f"  ✅ {pad} ({os.path.getsize(pad):,} bytes)")
        else:
            print(f"  ⏳ {r['horizon']}: geen geldig modelbestand")

    n_geldig = len(geldige_modellen)
    print(f"\nModellen aangemaakt: {n_geldig}/{len(results)}")

    # --------------------------------------------------------
    # NOTIFICATIES
    # --------------------------------------------------------
    stuur_notificaties(results)

    # --------------------------------------------------------
    # SAMENVATTING
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("LIGHTGBMV1 RUN VOLTOOID")
    print("=" * 70)
    for result in results:
        if result["status"] == "getraind":
            print(
                f"{result['horizon']:>4} | getraind     | "
                f"AUC={fmt(result['auc'])} | "
                f"control={fmt(result['control_auc'])}"
            )
        else:
            reden = str(result.get("reden", ""))[:60]
            print(f"{result['horizon']:>4} | overgeslagen | {reden}")

    print(f"\nOutput: {RESULTS_DIR}/")

    # Alleen falen als GEEN enkel model gemaakt is
    if n_geldig == 0:
        raise RuntimeError("Geen enkel LightGBM-model kon worden aangemaakt.")


if __name__ == "__main__":
    main()
