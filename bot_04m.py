import yfinance as yf
import os
import time
import requests

# --- CONFIG ---
SOURCE_FILE = "tickers_04a.txt"
MASTER_FILE = "tickers_04xx.txt"
CURRENT_RUN_FILE = "tickers_04m.txt"

# Telegram Config (Zorg dat deze variabelen in je GitHub Secrets staan of vul ze hier in)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"⚠️ Telegram fout: {e}")

def munger_keuring(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        roe = info.get('returnOnEquity', 0)
        debt = info.get('debtToEquity', 100)
        margin = info.get('profitMargins', 0)
        
        # Munger Criteria: ROE > 15%, Schuld < 60, Marge > 10%
        if roe > 0.15 and debt < 60 and margin > 0.10:
            return True, roe, debt
        return False, 0, 0
    except:
        return False, 0, 0

def main():
    print("--- MASTER LIST UPDATER + TELEGRAM (04m -> 04xx) ---")
    
    # 1. Vind bronbestand
    bestanden = os.listdir('.')
    actueel_bron = next((f for f in bestanden if f.lower() == SOURCE_FILE.lower()), None)
    if not actueel_bron:
        print(f"❌ Bronbestand {SOURCE_FILE} niet gevonden.")
        return

    # 2. Lees tickers
    with open(actueel_bron, 'r') as f:
        data = f.read().replace('\n', ',')
        scan_tickers = [t.strip().upper() for t in data.
