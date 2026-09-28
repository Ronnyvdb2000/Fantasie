#!/usr/bin/env python3
"""
Single-file pipeline:
- Download daily OHLCV from Yahoo Finance
- Build features
- Create 10 / 30 / 60 day labels
- Train LightGBM models
- Evaluate AUC + top-N returns
"""

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# -----------------------------
# CONFIG
# -----------------------------
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "NFLX",
    "INTC", "AMD", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "QCOM",
    "AVGO", "TXN", "SHOP", "SQ"
]  # voorbeeld; vervang door jouw universum

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

HORIZONS = [10, 30, 60]  # dagen
TOP_N_PCT = 0.20         # top 20% voor rendement-evaluatie
TEST_SIZE = 0.3          # time-based split doen we zelf, deze is alleen fallback

# -----------------------------
# DATA DOWNLOAD
# -----------------------------
def download_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False
    )
    # data is MultiIndex: (ticker, field)
    return data


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_features_for_ticker(df):
    """
    df: DataFrame met kolommen ['Open','High','Low','Close','Adj Close','Volume']
    index: Date
    """
    df = df.copy()

    # Basis returns
    df["ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5d"] = df["ret_1d"].rolling(5).sum()
    df["ret_10d"] = df["ret_1d"].rolling(10).sum()
    df["ret_20d"] = df["ret_1d"].rolling(20).sum()

    # Volatility
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # SMA's
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["sma_10_50_ratio"] = df["sma_10"] / df["sma_50"]
    df["sma_20_50_ratio"] = df["sma_20"] / df["sma_50"]

    # Volume features
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_zscore_20"] = (df["Volume"] - df["vol_sma_20"]) / (df["Volume"].rolling(20).std())
    df["vol_spike"] = df["Volume"] / df["vol_sma_20"]

    # RSI 14 (eenvoudige implementatie)
    delta = df["Close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


defHier is een **single‑file, end‑to‑end Python‑script** (Yahoo Finance → features → labels 10/30/60 → LightGBM‑modellen → eenvoudige top‑N backtest).

```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# ---------- CONFIG ----------
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]  # vervang door jouw universum
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
HORIZONS = [10, 30, 60]
TOP_FRACTION = 0.2  # top 20%

# ---------- DATA DOWNLOAD ----------
def download_data(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker"
    )
    return data

# ---------- FEATURE ENGINEERING ----------
def compute_features_for_ticker(df):
    df = df.copy()
    df["Return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Return_5d"] = df["Return_1d"].rolling(5).sum()
    df["Return_10d"] = df["Return_1d"].rolling(10).sum()
    df["Return_20d"] = df["Return_1d"].rolling(20).sum()

    df["RetMean_10"] = df["Return_1d"].rolling(10).mean()
    df["RetMean_20"] = df["Return_1d"].rolling(20).mean()
    df["RetStd_10"] = df["Return_1d"].rolling(10).std()
    df["RetStd_20"] = df["Return_1d"].rolling(20).std()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_10_50"] = df["SMA_10"] / df["SMA_50"]
    df["SMA_20_50"] = df["SMA_20"] / df["SMA_50"]

    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Vol_Z"] = (df["Volume"] - df["Vol_SMA_20"]) / (df["Vol_SMA_20"] + 1e-9)
    df["Vol_Spike"] = df["Volume"] / (df["Vol_SMA_20"] + 1e-9)

    # RSI 14
    delta = df["Close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

def build_feature_frame(raw_data):
    frames = []
    for ticker in TICKERS:
        df = raw_data[ticker].dropna().copy()
        df = compute_features_for_ticker(df)
        df["Ticker"] = ticker
        frames.append(df)
    full = pd.concat(frames)
    full = full.reset_index().rename(columns={"Date": "Date"})
    return full

# ---------- LABELS ----------
def add_labels(df, horizons):
    df = df.sort_values(["Ticker", "Date"]).copy()
    for H in horizons:
        df[f"R_{H}"] = df.groupby("Ticker")["Close"].shift(-H)
        df[f"R_{H}"] = (df[f"R_{H}"] - df["Close"]) / df["Close"]
        df[f"Y_{H}"] = (df[f"R_{H}"] > 0).astype(int)
    return df

# ---------- TRAIN / TEST SPLIT ----------
def time_split(df, test_fraction=0.3):
    df = df.sort_values("Date")
    dates = df["Date"].unique()
    split_idx = int(len(dates) * (1 - test_fraction))
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    train = df[df["Date"].isin(train_dates)].copy()
    test = df[df["Date"].isin(test_dates)].copy()
    return train, test

# ---------- MODEL TRAINING ----------
def train_lgbm(X_train, y_train):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "min_child_samples": 40,
        "metric": "auc"
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

# ---------- BACKTEST TOP-N ----------
def backtest_topN(test_df, model, horizon, feature_cols, top_fraction=0.2):
    df = test_df.dropna(subset(feature_cols + [f"R_{horizon}", f"Y_{horizon}"])).copy()
    X = df[feature_cols]
    y = df[f"Y_{horizon}"]
    preds = model.predict_proba(X)[:, 1]
    df["Score"] = preds

    auc = roc_auc_score(y, preds)

    results = []
    for date, group in df.groupby("Date"):
        group = group.sort_values("Score", ascending=False)
        N = max(1, int(len(group) * top_fraction))
        top = group.head(N)
        ret = top[f"R_{horizon}"].mean()
        results.append({"Date": date, "Ret": ret})

    res_df = pd.DataFrame(results)
    avg_ret = res_df["Ret"].mean()
    return auc, avg_ret, res_df

# ---------- MAIN ----------
def main():
    print("Downloading data...")
    raw = download_data(TICKERS, START_DATE, END_DATE)

    print("Building features...")
    full = build_feature_frame(raw)
    full = add_labels(full, HORIZONS)

    # kies feature columns
    feature_cols = [
        "Return_1d", "Return_5d", "Return_10d", "Return_20d",
        "RetMean_10", "RetMean_20", "RetStd_10", "RetStd_20",
        "SMA_10", "SMA_20", "SMA_50", "SMA_10_50", "SMA_20_50",
        "Vol_SMA_20", "Vol_Z", "Vol_Spike",
        "RSI_14", "MACD", "MACD_Signal"
    ]

    train_df, test_df = time_split(full, test_fraction=0.3)

    for H in HORIZONS:
        print(f"\n=== Horizon {H} dagen ===")
        train_h = train_df.dropna(subset=feature_cols + [f"Y_{H}"]).copy()
        X_train = train_h[feature_cols]
        y_train = train_h[f"Y_{H}"]

        print("Training LightGBM...")
        model = train_lgbm(X_train, y_train)

        print("Backtesting top-N...")
        auc, avg_ret, res_df = backtest_topN(test_df, model, H, feature_cols, TOP_FRACTION)

        print(f"AUC: {auc:.3f}")
        print(f"Gemiddeld top-{int(TOP_FRACTION*100)}% rendement over {H} dagen: {avg_ret*100:.2f}%")

if __name__ == "__main__":
    main()
