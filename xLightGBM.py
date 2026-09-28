#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import random
import socket
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------
#  PROXY FALLBACK
# ---------------------------------------------------------

PROXIES = [
    None,
    {"http": "http://1.1.1.1:8080", "https": "http://1.1.1.1:8080"},
    {"http": "http://8.8.8.8:3128", "https": "http://8.8.8.8:3128"},
]

def safe_download(ticker):
    """Download ticker with proxy fallback + retry."""
    for proxy in PROXIES:
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    period="5y",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    proxy=proxy
                )
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                time.sleep(0.5)
    return None

# ---------------------------------------------------------
#  FEATURE ENGINE
# ---------------------------------------------------------

def make_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df

# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

def main():
    print("\nStart xLightGBM run")
    print("RUN_ID:", RUN_ID)
    print("Download data:\n")

    tickers = ["SPY", "QQQ", "DIA", "IWM", "XLF", "XLK", "XLE", "XLY"]

    all_data = []

    for t in tqdm(tickers):
        df = safe_download(t)
        if df is None:
            print(f"FAILED: {t}")
            continue
        df = make_features(df)
        df["Ticker"] = t
        all_data.append(df)

    if len(all_data) == 0:
        print("NO DATA DOWNLOADED — STOPPING")
        return

    data = pd.concat(all_data)
    X = data[["Return", "MA10", "MA50", "Volatility"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    out_file = f"{RESULT_DIR}/{RUN_ID}_results.txt"
    with open(out_file, "w") as f:
        f.write(f"RUN_ID: {RUN_ID}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Samples: {len(data)}\n")

    print("\nDONE — results saved:", out_file)


if __name__ == "__main__":
    main()
