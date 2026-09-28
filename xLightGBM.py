#!/usr/bin/env python3
# xLightGBM.py — ML-bot met LightGBM + Telegram + e-mail + Supabase
# Score-optie 1: score = LightGBM-predictie (kans op positieve return)

import os
import sys
import time
import json
import smtplib
import traceback
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import requests

# ------------------------------------------------------------
# Config uit environment (zelfde stijl als je Darvas-bot)
# ------------------------------------------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_TO = os.getenv("MAIL_TO", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "xlightgbm_signals")

HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")

# Tickers (zoals in je log: 8 stuks)
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]

# ------------------------------------------------------------
# Helper: veilige download met proxy fallback
# ------------------------------------------------------------

def download_with_proxy_fallback(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Probeer eerst met proxy (als gezet), dan zonder proxy.
    Lost typische SPY/QQQ timezone/no data issues vaak op.
    """
    proxies = {}
    if HTTP_PROXY:
        proxies["http"] = HTTP_PROXY
    if HTTPS_PROXY:
        proxies["https"] = HTTPS_PROXY

    # 1) met proxy (indien aanwezig)
    try:
        if proxies:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                proxy=HTTP_PROXY or HTTPS_PROXY,
            )
        else:
            data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
    except Exception as e:
        print(f"Download met proxy faalde voor {ticker}: {e}")

    # 2) zonder proxy
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data
    except Exception as e:
        print(f"Download zonder proxy faalde voor {ticker}: {e}")

    print(f"Geen bruikbare data voor {ticker} na proxy fallback.")
    return pd.DataFrame()

# ------------------------------------------------------------
# Feature & label bouw
# ------------------------------------------------------------

def build_features_and_labels(df: pd.DataFrame, horizon: int = 10):
    """
    Bouw simpele features + label:
    - label = 1 als close(t+horizon) > close(t), anders 0
    - features: returns, rolling vol, moving averages, RSI-achtig
    """
    df = df.copy()
    df["return_1d"] = df["Adj Close"].pct_change()
    df["return_5d"] = df["Adj Close"].pct_change(5)
    df["return_10d"] = df["Adj Close"].pct_change(10)

    df["vol_10d"] = df["return_1d"].rolling(10).std()
    df["vol_20d"] = df["return_1d"].rolling(20).std()

    df["ma_10"] = df["Adj Close"].rolling(10).mean()
    df["ma_20"] = df["Adj Close"].rolling(20).mean()
    df["ma_ratio_10_20"] = df["ma_10"] / df["ma_20"]

    # simpele momentum / RSI-achtig
    df["up_move"] = np.where(df["return_1d"] > 0, df["return_1d"], 0.0)
    df["down_move"] = np.where(df["return_1d"] < 0, -df["return_1d"], 0.0)
    df["avg_up"] = df["up_move"].rolling(14).mean()
    df["avg_down"] = df["down_move"].rolling(14).mean()
    df["rsi"] = df["avg_up"] / (df["avg_up"] + df["avg_down"] + 1e-9)

    # label: horizon forward return
    df["future_price"] = df["Adj Close"].shift(-horizon)
    df["future_return"] = (df["future_price"] - df["Adj Close"]) / df["Adj Close"]
    df["label"] = np.where(df["future_return"] > 0, 1, 0)

    df = df.dropna()

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_10d",
        "vol_10d",
        "vol_20d",
        "ma_10",
        "ma_20",
        "ma_ratio_10_20",
        "rsi",
    ]

    X = df[feature_cols].values
    y = df["label"].values

    return X, y, df

# ------------------------------------------------------------
# Train LightGBM model (score = predictie)
# ------------------------------------------------------------

def train_lightgbm(X: np.ndarray, y: np.ndarray):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # AUC voor info
    y_val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"LightGBM AUC (valid): {auc:.4f}")

    return model

# ------------------------------------------------------------
# Telegram, e-mail, Supabase
# ------------------------------------------------------------

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram-config ontbreekt, bericht niet verzonden.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"Telegram-fout: {r.status_code} {r.text}")
        else:
            print("Telegram-bericht verzonden.")
    except Exception as e:
        print(f"Telegram-exceptie: {e}")

def send_email(subject: str, body: str):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS or not MAIL_TO:
        print("SMTP-config ontbreekt, e-mail niet verzonden.")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = MAIL_TO

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("E-mail verzonden.")
    except Exception as e:
        print(f"E-mail-exceptie: {e}")

def log_to_supabase(records):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase-config ontbreekt, logging niet verzonden.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(records), timeout=10)
        if r.status_code not in (200, 201):
            print(f"Supabase-fout: {r.status_code} {r.text}")
        else:
            print("Supabase-logging verzonden.")
    except Exception as e:
        print(f"Supabase-exceptie: {e}")

# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def main():
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    print("\nStart xLightGBM run")
    print(f"RUN_ID: {run_id}")

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365 * 3)  # 3 jaar historie
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    all_X = []
    all_y = []
    per_ticker_df = {}

    print("Download data:")
    for ticker in tqdm(TICKERS):
        try:
            df = download_with_proxy_fallback(ticker, start_str, end_str)
            if df.empty:
                print(f"Geen data voor {ticker}, skip.")
                continue
            df = df.rename(columns=str)  # defensief
            X, y, df_feat = build_features_and_labels(df)
            if len(X) < 200:
                print(f"Te weinig data voor {ticker}, skip.")
                continue

            all_X.append(X)
            all_y.append(y)
            per_ticker_df[ticker] = df_feat
        except Exception as e:
            print(f"Fout bij ticker {ticker}: {e}")
            traceback.print_exc()

    if not all_X:
        print("Geen bruikbare data voor enig ticker — geen model, geen signalen.")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    print("Train LightGBM-model...")
    model = train_lightgbm(X_all, y_all)

    # --------------------------------------------------------
    # Score berekening (optie 1: score = predictie)
    # --------------------------------------------------------
    signals = []

    for ticker, df_feat in per_ticker_df.items():
        # laatste rij = huidige dag
        last_row = df_feat.iloc[-1]
        feature_cols = [
            "return_1d",
            "return_5d",
            "return_10d",
            "vol_10d",
            "vol_20d",
            "ma_10",
            "ma_20",
            "ma_ratio_10_20",
            "rsi",
        ]
        x = last_row[feature_cols].values.reshape(1, -1)
        prob = float(model.predict(x)[0])  # kans op label=1
        score = prob  # 0–1
        total_score = score * 100.0

        signals.append({
            "ticker": ticker,
            "date": end_date.isoformat(),
            "score": round(score, 4),
            "total_score": round(total_score, 2),
            "close": float(last_row["Adj Close"]),
            "run_id": run_id,
        })

    # sorteer op score
    signals_sorted = sorted(signals, key=lambda s: s["score"], reverse=True)
    top_n = signals_sorted[:2]

    # --------------------------------------------------------
    # Bericht opbouwen (Telegram + e-mail)
    # --------------------------------------------------------
    date_str = end_date.strftime("%Y-%m-%d")
    lines = []
    lines.append(f"📊 xLightGBM-signalen — {date_str}")
    lines.append(f"RUN_ID: {run_id}")
    lines.append("")
    lines.append("Top 2 tickers (score = LightGBM-kans op positieve 10d-return):")
    for s in top_n:
        lines.append(
            f"- {s['ticker']}: score={s['score']:.4f} (total_score={s['total_score']:.2f}), close={s['close']:.2f}"
        )
    lines.append("")
    lines.append("Alle scores:")
    for s in signals_sorted:
        lines.append(
            f"{s['ticker']}: score={s['score']:.4f}, total_score={s['total_score']:.2f}, close={s['close']:.2f}"
        )
    lines.append("")
    lines.append("Deze bot gebruikt LightGBM op 3 jaar Yahoo Finance data met proxy fallback.")
    message_text = "\n".join(lines)

    # Telegram
    send_telegram_message(message_text)

    # E-mail
    send_email(subject=f"xLightGBM-signalen {date_str}", body=message_text)

    # Supabase logging
    supabase_records = []
    for s in signals_sorted:
        supabase_records.append({
            "ticker": s["ticker"],
            "date": s["date"],
            "score": s["score"],
            "total_score": s["total_score"],
            "close": s["close"],
            "run_id": s["run_id"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        })
    log_to_supabase(supabase_records)

    print("xLightGBM run klaar.")

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Onverwachte fout in xLightGBM: {e}")
        traceback.print_exc()
        sys.exit(1)
