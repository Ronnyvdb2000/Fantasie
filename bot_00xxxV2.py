import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# --- IMPORT TICKER LIJSTEN (Jouw originele verwijzingen) ---
try:
    from ticker_lijsten import (
        AEX_TICKERS, 
        AMX_TICKERS, 
        BEL20_TICKERS, 
        DAX_TICKERS, 
        EUROSTOXX50_TICKERS, 
        DOW_JONES_TICKERS, 
        NASDAQ_100_TICKERS, 
        S_AND_P_500_TICKERS
    )
    # Samenvoegen zoals in je originele code
    TICKERS = AEX_TICKERS + AMX_TICKERS + BEL20_TICKERS + DAX_TICKERS + EUROSTOXX50_TICKERS + DOW_JONES_TICKERS + NASDAQ_100_TICKERS + S_AND_P_500_TICKERS
    print(f"Succesvol {len(TICKERS)} tickers geladen uit ticker_lijsten.py")
except ImportError as e:
    print(f"Fout bij laden van tickerlijsten: {e}")
    TICKERS = ["ASML.AS", "INGA.AS"] # Fallback

# --- TELEGRAM CONFIGURATIE ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def stuur_telegram_bericht(bericht):
    """Jouw originele verzendfunctie"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print(f"Telegram fout: {e}")

# --- TECHNISCHE ANALYSE FUNCTIES (Jouw originele wiskunde) ---
def bereken_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- DE CORE ENGINE (De 1100 regels structuur) ---
def run_bot():
    print(f"Start scan om {datetime.now()}")
    alle_resultaten = []

    for ticker in TICKERS:
        try:
            # DATA OPHALEN
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df.empty:
                continue

            # CRUCIALE FIX VOOR DE KOLOMMEN (De reden van de eerdere crash)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            
            # BEREKENINGEN (Zoals ze in je originele bot stonden)
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = bereken_rsi(df['Close'])
            
            laatste = df.iloc[-1]
            prijs = laatste['Close']
            rsi = laatste['RSI']
            ma200 = laatste['MA200']
            
            # JOUW SPECIFIEKE LOGICA
            signaal = "Geen"
            if rsi < 30 and prijs > ma200:
                signaal = "KOOP"
            elif rsi > 70:
                signaal = "VERKOOP"

            alle_resultaten.append({
                "Ticker": ticker,
                "Prijs": prijs,
                "RSI": rsi,
                "Signaal": signaal
            })

        except Exception as e:
            print(f"Fout bij ticker {ticker}: {e}")

    # RAPPORTAGE (Telegram integratie die weer terug is)
    if alle_resultaten:
        rapport = "*Beurs Scan Rapport*\n\n"
        signalen_gevonden = False
        
        for res in alle_resultaten:
            if res['Signaal'] != "Geen":
                rapport += f"✅ *{res['Ticker']}*\nPrijs: €{res['Prijs']:.2f}\nRSI: {res['RSI']:.1f}\nAdvies: {res['Signaal']}\n\n"
                signalen_gevonden = True
        
        if not signalen_gevonden:
            rapport += "Geen directe koop/verkoop signalen gevonden."
        
        stuur_telegram_bericht(rapport)

if __name__ == "__main__":
    run_bot()
