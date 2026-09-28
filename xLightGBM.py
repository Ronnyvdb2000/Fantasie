import os
import time
import uuid
from datetime import datetime, timedelta

import yfinance as yf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import requests


# -----------------------------
# Config
# -----------------------------

TICKERS = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
]

HORIZONS = [10, 30, 60]  # dagen vooruit

START_DATE = "2015-01-01"
END_DATE = None  # None = tot vandaag

TOP_N = 20  # voor eenvoudige top-N backtest

# Supabase config (vul zelf in)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://YOUR_PROJECT.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "YOUR_SERVICE_ROLE_KEY")
SUPABASE_TABLE = "signals_lightgbm"

RUN_ID = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


# -----------------------------
# Data download
# -----------------------------

def download_data(tickers, start, end=None):
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    all_data = {}
    for t in tqdm(tickers, desc="Download data"):
        df = yf.download(t, start=start, end=end, auto_adjust=False)
        if df.empty:
            continue
        df = df.rename(columns=str.lower)
        df["ticker"] = t
        all_data[t] = df

    return all_data


# -----------------------------
# Feature engineering
# -----------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)

    df["vol_10d"] = df["return_1d"].rolling(10).std()
    df["vol_20d"] = df["return_1d"].rolling(20).std()

    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    df["ma_10_rel"] = df["ma_10"] / df["close"] - 1.0
    df["ma_20_rel"] = df["ma_20"] / df["close"] - 1.0
    df["ma_50_rel"] = df["ma_50"] / df["close"] - 1.0

    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std()
    )

    df = df.dropna()
    return df


# -----------------------------
# Labels per horizon
# -----------------------------

def add_labels_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    future_price = df["close"].shift(-horizon)
    future_return = (future_price / df["close"]) - 1.0
    df[f"label_{horizon}d"] = (future_return > 0.0).astype(int)
    df[f"target_ret_{horizon}d"] = future_return
    df = df.dropna()
    return df


# -----------------------------
# Model training
# -----------------------------

def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    return model


# -----------------------------
# Eenvoudige top-N backtest
# -----------------------------

def simple_topN_backtest(df: pd.DataFrame, horizon: int, prob_col: str) -> dict:
    """
    Heel eenvoudige backtest:
    - per datum: sorteer tickers op prob_up
    - neem top-N
    - gebruik echte future return (target_ret_horizon)
    - bereken gemiddelde R/R en hitrate
    """
    df = df.copy()
    label_col = f"label_{horizon}d"
    ret_col = f"target_ret_{horizon}d"

    # we doen het per datum
    results = []
    for date, group in df.groupby(df.index.date):
        group = group.sort_values(prob_col, ascending=False)
        top = group.head(TOP_N)
        if top.empty:
            continue

        avg_ret = top[ret_col].mean()
        hitrate = (top[ret_col] > 0.0).mean()
        results.append((date, avg_ret, hitrate))

    if not results:
        return {"rr": None, "hitrate": None}

    rr = np.mean([r[1] for r in results])
    hitrate = np.mean([r[2] for r in results])

    return {"rr": rr, "hitrate": hitrate}


# -----------------------------
# Supabase helper
# -----------------------------

def push_signals_to_supabase(rows):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase config ontbreekt, skip push.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    resp = requests.post(url, headers=headers, json=rows, timeout=30)
    if resp.status_code >= 300:
        print(f"Supabase error: {resp.status_code} {resp.text}")
    else:
        print(f"Supabase OK, {len(rows)} rows ingevoegd.")


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    print("Start xLightGBM run")
    print(f"RUN_ID: {RUN_ID}")

    data = download_data(TICKERS, START_DATE, END_DATE)

    all_rows_for_supabase = []

    for horizon in HORIZONS:
        print(f"\n=== Horizon {horizon} dagen ===")

        frames = []
        for t, df in data.items():
            df_feat = add_features(df)
            df_lab = add_labels_for_horizon(df_feat, horizon)
            df_lab["ticker"] = t
            frames.append(df_lab)

        if not frames:
            print(f"Geen data voor horizon {horizon}")
            continue

        full = pd.concat(frames, axis=0)
        full = full.sort_index()

        feature_cols = [
            "return_1d", "return_5d", "return_10d",
            "vol_10d", "vol_20d",
            "ma_10_rel", "ma_20_rel", "ma_50_rel",
            "high_low_range", "volume_zscore",
        ]
        label_col = f"label_{horizon}d"

        X = full[feature_cols].values
        y = full[label_col].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = train_lightgbm(X_train, y_train, X_val, y_val)

        # AUC
        y_val_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_val_pred)
        print(f"AUC horizon {horizon}: {auc:.3f}")

        # Backtest
        full["prob_up"] = model.predict(full[feature_cols].values)
        bt = simple_topN_backtest(full, horizon, "prob_up")
        rr = bt["rr"]
        hitrate = bt["hitrate"]
        print(f"Backtest horizon {horizon}:")
        if rr is not None:
            print(f"  R/R: {rr:.3%}, hitrate: {hitrate:.1%}")
        else:
            print("  Onvoldoende data voor backtest.")

        # Laatste datum per ticker als signaal naar Supabase
        latest_date = full.index.max()
        latest = full[full.index == latest_date].copy()
        latest = latest.sort_values("prob_up", ascending=False)
        latest["rank"] = np.arange(1, len(latest) + 1)

        for _, row in latest.iterrows():
            all_rows_for_supabase.append(
                {
                    "run_id": RUN_ID,
                    "horizon_days": int(horizon),
                    "ticker": str(row["ticker"]),
                    "prob_up": float(row["prob_up"]),
                    "expected_return": float(row.get(f"target_ret_{horizon}d", np.nan)),
                    "rank": int(row["rank"]),
                    "rr": float(rr) if rr is not None else None,
                    "meta": {
                        "auc": float(auc),
                        "hitrate": float(hitrate) if hitrate is not None else None,
                        "latest_date": latest_date.isoformat(),
                    },
                }
            )

    # Push naar Supabase (bot doet daarna telegram + mail)
    if all_rows_for_supabase:
        push_signals_to_supabase(all_rows_for_supabase)
    else:
        print("Geen rows om naar Supabase te sturen.")

    print("xLightGBM run klaar.")


if __name__ == "__main__":
    main()
