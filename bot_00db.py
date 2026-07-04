#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00vcp.py  —  VCP ENGINE v1.0
Volatility Contraction Pattern — Minervini's 'Narrowing' techniek.
Zelfde structuur, tickerbestanden en Telegram output als bot_00xxxV2.py.

Hoe VCP werkt:
  Na een stijging consolideert een aandeel in steeds smallere correcties:
    Correctie 1: -15% op hoog volume  (1e contractie)
    Correctie 2: -10% op lager volume (2e contractie)
    Correctie 3:  -6% op laag volume  (3e contractie)
    Correctie 4:  -3% op minimaal vol (4e contractie — ideaal)
    → BREAKOUT boven pivot op 2-3× normaal volume

Score systeem (0-8):
  1. Minimum 2 VCP contracties gedetecteerd
  2. Elke correctie kleiner dan vorige (%)
  3. Elke correctie korter in tijd dan vorige
  4. Volume daalt bij elke correctie
  5. Laatste contractie ≤ 10% diep
  6. Prijs binnen 10% van pivot high
  7. Stage 2 trend (Close > MA50 > MA150 > MA200)
  8. Breakout boven pivot op verhoogd volume
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL        = 50_000.0
MAX_POSITIONS        = 10
MIN_CASH_RATIO       = 0.10
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001

TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
MAX_HOLD_DAYS        = 60

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

EXCHANGES = {
    "041 Benelux":     "tickers_041x.txt",
    "042 Parijs":      "tickers_042x.txt",
    "043 Frankfurt":   "tickers_043x.txt",
    "044 Spanje/Port": "tickers_044x.txt",
    "045 Londen":      "tickers_045x.txt",
    "046 Milaan":      "tickers_046x.txt",
    "047 Toronto":     "tickers_047x.txt",
    "048 Nasdaq/NYSE": "tickers_048x.txt",
}

VCP_CFG = {
    "min_contracties":       2,      
    "max_contracties":       5,      
    "min_correctie_pct":     3.0,    
    "max_correctie_pct":     50.0,   
    "contractie_ratio":      0.80,   
    "tijd_ratio":            0.90,   
    "laatste_max_pct":       10.0,   
    "pivot_proximity_pct":   10.0,   
    "vol_ma_period":         50,     
    "vol_droogval_ratio":    0.80,   
    "breakout_vol_mult":     1.5,    
    "ma_fast":               50,
    "ma_mid":                150,
    "ma_slow":               200,
    "atr_period":            14,
    "min_score":             4,      
    "lookback_days":         120,    
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()

# ============================================================
# HULPFUNCTIES
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        if res.status_code != 200:
            print(f"[Telegram Error] {res.text}")
    except Exception as e:
        print(f"[Telegram Exception] {e}")

# ============================================================
# VCP CORE ENGINE
# ============================================================

def find_contractions(df: pd.DataFrame, idx: int, lookback: int) -> List[Dict]:
    """
    Zoekt opeenvolgende golven (highs en lows) om contracties te meten.
    """
    contractions = []
    start_idx = max(0, idx - lookback)
    sub_df = df.iloc[start_idx:idx+1]
    
    if len(sub_df) < 20:
        return []
        
    prices = sub_df['Close'].values
    volumes = sub_df['Volume'].values
    
    # Eenvoudige lokale minima/maxima detectie via rolling window
    peaks = []
    troughs = []
    
    for i in range(2, len(prices) - 2):
        if prices[i] == max(prices[i-2:i+3]):
            peaks.append((i, prices[i], volumes[i]))
        if prices[i] == min(prices[i-2:i+3]):
            troughs.append((i, prices[i], volumes[i]))
            
    # Combineer peaks en troughs tot opeenvolgende correctiegolven
    pairs = []
    for p_idx, p_val, p_vol in peaks:
        # Zoek de eerstvolgende diepe trough na deze peak
        valid_troughs = [t for t in troughs if t[0] > p_idx]
        if valid_troughs:
            t_idx, t_val,
