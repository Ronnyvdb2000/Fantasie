# xLightGBM – Optie 1: Volledige versie met Proxy + RapidAPI + CSV fallback
import os
import json
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "VIX", "GLD", "TLT", "XLF"]

# ---------------------------
# 1. Yahoo Finance Download
# ---------------------------
def download_yahoo(ticker):
    try:
        df = yf.download(ticker, period="5y", auto_adjust=False)
        if df is None or df.empty:
            raise Exception("Empty dataframe")
        return df
    except Exception as e:
        return None

# ---------------------------
# 2. Proxy Fallback
# ---------------------------
def download_proxy(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        params = {
            "range": "5y",
            "interval": "1d",
            "events": "history"
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        df = pd.read_csv(pd.compat.StringIO(r.text))
        return df
    except:
        return None

# ---------------------------
# 3. RapidAPI Fallback
# ---------------------------
def download_rapidapi(ticker):
    try:
        url = "https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/" + ticker
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY", ""),
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "body" not in data:
            return None
        df = pd.DataFrame(data["body"])
        return df
    except:
        return None

# ---------------------------
# 4. CSV Backup Fallback
# ---------------------------
def download_csv_backup(ticker):
    path = f"backup/{ticker}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ---------------------------
# 5. Unified Download Handler
# ---------------------------
def get_data(ticker):
    print(f"\nDownloading {ticker}...")

    methods = [
        ("Yahoo", download_yahoo),
        ("Proxy", download_proxy),
        ("RapidAPI", download_rapidapi),
        ("CSV Backup", download_csv_backup)
    ]

    for name, func in methods:
        df = func(ticker)
        if df is not None and not df.empty:
            print(f"✔ {ticker} via {name}")
            return df

    print(f"✖ FAILED: {ticker}")
    return None

# ---------------------------
# MAIN
# ---------------------------
print("\nStart xLightGBM run")
print("RUN_ID:", RUN_ID)
print("Download data:\n")

all_data = {}

for ticker in tqdm(TICKERS):
    df = get_data(ticker)
    if df is None:
        print(f"Ticker {ticker} FAILED completely.")
    else:
        all_data[ticker] = df
        df.to_csv(f"{RESULT_DIR}/{ticker}_{RUN_ID}.csv")

print("\nDownload complete.\n")

# ---------------------------
# MODEL (simple example)
# ---------------------------
if "SPY" not in all_data:
    print("SPY missing → model cannot run.")
    exit(0)

df = all_data["SPY"].copy()
df["Return"] = df["Close"].pct_change()
df["Target"] = (df["Return"] > 0).astype(int)
df = df.dropna()

X = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

with open(f"{RESULT_DIR}/model_result_{RUN_ID}.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")

print(f"Model accuracy: {acc}")
print("Done.")
