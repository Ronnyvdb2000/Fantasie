import yfinance as yf
import pandas as pd
import os
import requests
import time
from datetime import datetime

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
INZET_PER_TRADE = 1000.0
COMMISSIE = 1.0
BEURSTAKS = 0.0035
BELASTING = 0.10

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: 
        print(f"Telegram configuratie mist. Bericht: {bericht}")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: 
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
        if res.status_code != 200:
            print(f"Telegram fout: {res.text}")
    except Exception as e: 
        print(f"Fout bij verzenden: {e}")

def bereken_indicatoren(df, ticker):
    try:
        # Fix voor yfinance Multi-Index (v0.2.x +)
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].ffill().astype(float)
        else:
            p = df['Close'].ffill().astype(float)
            
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / (loss + 1e-10)
        rsi2 = 100 - (100 / (1 +
