import yfinance as yf
import pandas as pd
import requests
import os
import time
from datetime import datetime

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Fout bij versturen Telegram: {e}")

def get_signal(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 200:
            return None

        # --- ORIGINELE INDICATOREN ---
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI berekening
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        cp = df['Close'].iloc[-1]
        s50 = df['SMA50'].iloc[-1]
        s200 = df['SMA200'].iloc[-1]
        rsi = df['RSI'].iloc[-1]

        # --- ORIGINELE REGIMES LOGICA ---
        # Regime 1: BUY (Bullish & Oversold)
        if cp > s50 > s200 and rsi < 40:
            return f"🟢 *BUY*: {ticker} (RSI: {rsi:.1f})"
        
        # Regime 2: HOLD (Trend is sterk)
        elif cp > s50 > s200:
            return f"🔵 *HOLD*: {ticker}"
        
        # Regime 3: SELL (Bearish trend bevestigd)
        elif cp < s50 < s200:
            return f"🔴 *SELL*: {ticker}"
        
        # Regime 4: WAIT (Onzeker / Transitie)
        else:
            return f"🟡 *WAIT*: {ticker}"

    except:
        return None

def voer_lijst_uit(bestandsnaam, rapport_naam, rapport_nr):
    if not os.path.exists(bestandsnaam):
        return

    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    resultaten = []
    for ticker in tickers:
        res = get_signal(ticker)
        if res:
            resultaten.append(res)
        time.sleep(0.1)

    # Rapport opmaken met de juiste naam
    bericht = f"📊 *{rapport_nr} {rapport_naam} RAPPORT*\n"
    bericht += f"Datum: {datetime.now().strftime('%d-%m %H:%M')}\n"
    bericht += "----------------------------------\n"
    
    if resultaten:
        bericht += "\n".join(resultaten)
    else:
        bericht += "Geen data beschikbaar voor deze lijst."

    send_telegram_message(bericht)

def main():
    # De originele indeling
    rapporten = {
        "01": "Hoogland",
        "02": "Macrotrends",
        "03": "Beursbrink",
        "04": "Benelux",
        "05": "Parijs",
        "06": "Power & AI",
        "07": "Metalen",
        "08": "Defensie",
        "09": "Varia"
    }

    for nr, naam in rapporten.items():
        bestandsnaam = f"tickers_{nr}.txt"
        voer_lijst_uit(bestandsnaam, naam, int(nr))
        time.sleep(3) # Rustpauze tussen rapporten

if __name__ == "__main__":
    main()
