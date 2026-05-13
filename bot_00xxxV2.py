import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# ==============================================================================
# 1. ORIGINELE TICKER LIJSTEN EN IMPORT LOGICA
# ==============================================================================
try:
    from ticker_lijsten import (
        AEX_TICKERS, AMX_TICKERS, ASCX_TICKERS, 
        BEL20_TICKERS, DAX_TICKERS, EUROSTOXX50_TICKERS, 
        DOW_JONES_TICKERS, NASDAQ_100_TICKERS, S_AND_P_500_TICKERS
    )
    TICKER_GROUPS = {
        "AEX": AEX_TICKERS,
        "AMX": AMX_TICKERS,
        "ASCX": ASCX_TICKERS,
        "BEL20": BEL20_TICKERS,
        "DAX": DAX_TICKERS,
        "EUROSTOXX50": EUROSTOXX50_TICKERS,
        "DOW_JONES": DOW_JONES_TICKERS,
        "NASDAQ_100": NASDAQ_100_TICKERS,
        "S&P 500": S_AND_P_500_TICKERS
    }
except ImportError:
    # Als het bestand ticker_lijsten.py ontbreekt in de map
    TICKER_GROUPS = {"ERROR": []}
    print("WAARSCHUWING: ticker_lijsten.py niet gevonden!")

# ==============================================================================
# 2. CONFIGURATIE & TELEGRAM (JOUW ORIGINELE FUNCTIES)
# ==============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# ==============================================================================
# 3. DE WISKUNDIGE INDICATOREN (HET HART VAN DE 1100 REGELS)
# ==============================================================================
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MA's
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # ATR (Voor ADX berekening)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # IBS
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
    
    return df

# ==============================================================================
# 4. DE MAIN LOOP (EEN-OP-EEN JOUW STRUCTUUR)
# ==============================================================================
def run_trading_bot():
    print(f"Sessie gestart op {datetime.now()}")
    totaal_rapport = []

    for group_name, tickers in TICKER_GROUPS.items():
        group_results = []
        print(f"Verwerken van groep: {group_name}")
        
        for ticker in tickers:
            try:
                # DOWNLOAD DATA
                df = yf.download(ticker, period="1y", interval="1d", progress=False)
                
                if df.empty or len(df) < 200:
                    continue

                # --- DE ESSENTIËLE FIX VOOR DE CRASH ---
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # --------------------------------------

                df = calculate_indicators(df)
                last_row = df.iloc[-1]
                
                # JOUW ORIGINELE LOGICA VOOR SIGNALEN
                rsi = last_row['RSI']
                price = last_row['Close']
                ma200 = last_row['MA200']
                ibs = last_row['IBS']
                
                status = ""
                if rsi < 30 and price > ma200:
                    status = "🟢 *KOOP (Dip)*"
                elif rsi < 25:
                    status = "🔥 *STERK KOOP*"
                elif rsi > 75:
                    status = "🔴 *VERKOOP*"
                
                if status:
                    group_results.append(f"`{ticker:<9}`: €{price:>7.2f} | RSI: {rsi:>4.1f} | {status}")

            except Exception as e:
                print(f"Fout bij {ticker}: {str(e)}")

        if group_results:
            header = f"📊 *INDEX: {group_name}*"
            totaal_rapport.append(header + "\n" + "\n".join(group_results))

    # VERSTUREN PER INDEX (Zoals in je originele code)
    if totaal_rapport:
        for rapport_deel in totaal_rapport:
            send_telegram_message(rapport_deel)
            time.sleep(1) # Voorkom Telegram spam block
    else:
        send_telegram_message("Beursscan voltooid: Geen bijzondere signalen vandaag.")

if __name__ == "__main__":
    run_trading_bot()
