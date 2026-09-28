#!/usr/bin/env python3
# xLightGBM.py
#
# Optie 1: score = LightGBM-predictie (kans / verwachte return)
# Met:
# - Yahoo Finance download met per-ticker fallback
# - Altijd Telegram + e-mail + Supabase logging, ook bij fouten

import os
import sys
import time
import uuid
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from lightgbm import LGBMRegressor
import requests

# -----------------------
# CONFIG
# -----------------------

TICKERS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"]
LOOKBACK_DAYS = 365
PRED_HORIZON_DAYS = 5
TOP_N = 2

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Email
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "xlightgbm_signals")

# -----------------------
# LOGGING
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# -----------------------
# DATA DOWNLOAD (met fallback)
# -----------------------

def download_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV data voor één ticker met fallback.
    - Eerste poging: yf.download
    - Tweede poging: yf.Ticker(ticker).history
    Bij mislukking: return None, maar script blijft verder lopen.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is not None and not df.empty:
            df["Ticker"] = ticker
            return df
        logging.warning(f"Empty dataframe via yf.download voor {ticker}, probeer fallback.")
    except Exception as e:
        logging.warning(f"Failed to get ticker '{ticker}' via yf.download, reason: {e}")

    # Fallback: Ticker().history
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end)
        if df is not None and not df.empty:
            df["Ticker"] = ticker
            return df
        logging.warning(f"Empty dataframe via Ticker().history voor {ticker}.")
    except Exception as e:
        logging.warning(f"Failed to get ticker '{ticker}' via Ticker().history, reason: {e}")

    logging.error(f"Geen bruikbare data voor {ticker} (Yahoo faalt of delisted).")
    return None

def download_all_data(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)
