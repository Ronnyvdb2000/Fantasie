import yfinance as yf
import pandas as pd
import requests
import os
import time
import numpy as np
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

def get_signals(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 200:
            return None

        # Berekeningen (v29 logica)
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        current_price = df['Close'].iloc[-1]
        sma50 = df['SMA50'].iloc[-1]
        sma200 = df['SMA200'].iloc[-1]
        
        # Simpele Trend Check
        is_bullish = current_price > sma50 > sma200
        
        if is_bullish:
            return f"✅ {ticker}: Bullish Trend (Prijs > SMA50 > SMA200)"
        return None
    except:
        return None

def voer_lijst_uit(bestandsnaam, rapport_naam, rapport_nr):
    if not os.path.exists(bestandsnaam):
        print(f"Bestand {bestandsnaam} niet gevonden. Overslaan...")
        return

    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    print(f"Start scan: {rapport_naam} ({len(tickers)} tickers)")
    gevonden_signalen = []

    for ticker in tickers:
        signaal = get_signals(ticker)
        if signaal:
            gevonden_signalen.append(signaal)
        time.sleep(0.2) # Voorkom Yahoo Finance blokkades

    # Rapport opmaken
    header = f"📊 *{rapport_nr} {rapport_naam} RAPPORT*\n"
    header += f"Datum: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n"
    header += "----------------------------------\n"

    if gevonden_signalen:
        bericht = header + "\n".join(gevonden_signalen)
    else:
        bericht = header + "Geen specifieke signalen gevonden in deze lijst."

    send_telegram_message(bericht)

def main():
    # De 9 rapporten koppelen aan hun bestanden
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

    send_telegram_message("🚀 *Global Scanner Start:* Controleren van alle 9 sectoren...")

    for nr, naam in rapporten.items():
        bestandsnaam = f"tickers_{nr}.txt"
        voer_lijst_uit(bestandsnaam, naam, int(nr))
        time.sleep(2) # Korte pauze tussen rapporten

    send_telegram_message("🏁 *Global Scanner Voltooid.*")

if __name__ == "__main__":
    main()
