#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xLightGBM.py — ML SIGNAL ENGINE v1.0
LightGBM + samengestelde Score C (0–8) + Telegram + e-mail + Supabase logging.

Gebruik:
  python xLightGBM.py live     # dagelijkse run (GitHub Actions)
  python xLightGBM.py backtest # optioneel later

Deze bot:
  - Haalt OHLCV-data op via Yahoo Finance
  - Bouwt features
  - Traint LightGBM (10/30/60 dagen horizon)
  - Berekent samengestelde Score C (lgbm + momentum + volatility + trend)
  - Selecteert top-kandidaten
  - Stuurt Telegram + e-mail
  - Logt alle signalen in Supabase tabel xlightgbm_signalen
"""

import os
import math
import time
import warnings
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from supabase import create_client

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL        = 50_000.0
TOP_FRACTION         = 0.20   # top 20% selectie
HORIZONS             = [10, 30, 60]

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# Tickers — vervang door jouw universum of laad uit bestanden
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "NFLX", "INTC", "AMD",
]

START_DATE = "2015-01-01"
END_DATE   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES
# ============================================================

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 4096):
        chunk = text[i:i + 4096]
        try:
            r = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                requests.post(
                    url,
                    json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                          "disable_web_page_preview": True},
                    timeout=10,
                )
        except Exception as e:
            print(f"Telegram fout: {e}")
        if i + 4096 < len(text):
            time.sleep(1)


def send_email(subject: str, body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        clean = body.replace("*", "").replace("`", "").replace("•", "-").replace("_", "")
        msg.attach(MIMEText(clean, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email fout: {e}")


def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


def _score_bar(score: float, max_score: int = 8) -> str:
    s_int = max(0, min(max_score, int(round(score))))
    return "█" * s_int + "░" * (max_score - s_int) + f" {score:.1f}/{max_score}"


def insert_supabase_record(data: dict) -> None:
    if supabase is None:
        return
    try:
        supabase.table("xlightgbm_signalen").insert(data).execute()
    except Exception as e:
        print(f"[WARN] Supabase insert fout: {e}")


# ============================================================
# DATA & FEATURES
# ============================================================

def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    frames = []
    for t in tickers:
        try:
            df = data[t].dropna().copy()
            df["Ticker"] = t
            df = df.reset_index().rename(columns={"Date": "Date"})
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full.sort_values(["Ticker", "Date"], inplace=True)
    return full


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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

    delta = df["Close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for t, g in raw.groupby("Ticker"):
        g = compute_features(g)
        frames.append(g)
    full = pd.concat(frames, ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full.sort_values(["Ticker", "Date"], inplace=True)
    return full


def add_labels(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()
    for H in horizons:
        future = df.groupby("Ticker")["Close"].shift(-H)
        df[f"R_{H}"] = (future - df["Close"]) / df["Close"]
        df[f"Y_{H}"] = (df[f"R_{H}"] > 0).astype(int)
    return df


def time_split(df: pd.DataFrame, test_fraction: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date")
    dates = df["Date"].unique()
    split_idx = int(len(dates) * (1 - test_fraction))
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    train = df[df["Date"].isin(train_dates)].copy()
    test = df[df["Date"].isin(test_dates)].copy()
    return train, test


# ============================================================
# MODEL & SCORE C
# ============================================================

FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    "RetMean_10", "RetMean_20", "RetStd_10", "RetStd_20",
    "SMA_10", "SMA_20", "SMA_50", "SMA_10_50", "SMA_20_50",
    "Vol_SMA_20", "Vol_Z", "Vol_Spike",
    "RSI_14", "MACD", "MACD_Signal",
]


def train_lgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
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
        "metric": "auc",
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def compute_score_components(row: pd.Series, lgbm_prob: float) -> Tuple[float, float, float, float, float]:
    """
    Score C:
      - score_lgbm: direct uit predictie (0–1 → 0–3)
      - score_momentum: op basis van Return_10d, RSI_14
      - score_volatility: op basis van RetStd_20, Vol_Z
      - score_trend: op basis van SMA_10_50, SMA_20_50, MACD
      - total_score: som, begrensd op 0–8
    """
    # LightGBM component (0–3)
    score_lgbm = max(0.0, min(3.0, lgbm_prob * 3.0))

    # Momentum (0–2)
    r10 = safe_float(row.get("Return_10d"), 0.0)
    rsi = safe_float(row.get("RSI_14"), 50.0)
    score_momentum = 0.0
    if r10 > 0:
        score_momentum += 1.0
    if 50 <= rsi <= 70:
        score_momentum += 1.0

    # Volatility (0–1.5)
    vol_std = abs(safe_float(row.get("RetStd_20"), 0.0))
    vol_z   = abs(safe_float(row.get("Vol_Z"), 0.0))
    score_volatility = 1.5
    if vol_std > 0.05 or vol_z > 3.0:
        score_volatility -= 0.5

    # Trend (0–1.5)
    sma_ratio_10_50 = safe_float(row.get("SMA_10_50"), 1.0)
    sma_ratio_20_50 = safe_float(row.get("SMA_20_50"), 1.0)
    macd = safe_float(row.get("MACD"), 0.0)
    score_trend = 0.0
    if sma_ratio_10_50 > 1.0 and sma_ratio_20_50 > 1.0:
        score_trend += 1.0
    if macd > 0:
        score_trend += 0.5

    total = score_lgbm + score_momentum + score_volatility + score_trend
    total = max(0.0, min(8.0, total))

    return score_lgbm, score_momentum, score_volatility, score_trend, total


# ============================================================
# TELEGRAM / EMAIL FORMAT
# ============================================================

def format_bericht(
    horizon: int,
    df_signals: pd.DataFrame,
    portfolio_waarde: float,
) -> Optional[str]:
    if df_signals.empty:
        return None

    nu = today_str()
    top2 = df_signals.head(2)
    max_score = df_signals["total_score"].max()

    lbl = {
        8: "🔥 PERFECT (8/8)",
        7: "⭐ UITSTEKEND (7/8)",
        6: "⚡ STERK (6/8)",
        5: "📊 GOED (5/8)",
        4: "📊 WATCHLIST (4/8)",
    }.get(int(round(max_score)), "📊")

    delen = [
        f"🤖 *xLightGBM — Horizon {horizon} dagen*",
        f"_{nu} | {len(df_signals)} kandidaten | top-{int(TOP_FRACTION*100)}% selectie_",
        "─────────────────────────────",
        "🏆 *TOP 2:*",
    ]

    for _, s in top2.iterrows():
        rr = safe_float(s.get("rr_ratio"), 0.0)
        delen.append(
            f"• `{s['ticker']}` {_score_bar(s['total_score'])} EUR{s['prijs']:.2f} {_yahoo_link(s['ticker'])}\n"
            f"  LGBM={s['score_lgbm']:.2f} | Mom={s['score_momentum']:.2f} | Vol={s['score_volatility']:.2f} | Trend={s['score_trend']:.2f}\n"
            f"  R/R: {rr:.1
