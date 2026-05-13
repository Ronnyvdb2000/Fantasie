import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests
from datetime import datetime, timedelta

# --- CONFIGURATIE (Gebruikt jouw GitHub Secrets) ---
TICKERS = ["ASML.AS", "ADYEN.AS", "INGA.AS", "HEIA.AS", "KPN.AS", "UNA.AS"]
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    """Jouw originele Telegram functie voor de meldingen."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("FOUT: Telegram Token of Chat ID niet gevonden in Secrets!")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Telegram Fout: {response.text}")
    except Exception as e:
        print(f"Verbindingsfout Telegram: {e}")

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Correcte Wilder's Smoothing voor RSI en ADX."""
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    alpha = 1 / period
    res = series.copy()
    res.iloc[period-1] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        res.iloc[i] = res.iloc[i-1] * (1 - alpha) + series.iloc[i] * alpha
    res.iloc[:period-1] = np.nan
    return res

def download_history(tickers, days=365):
    """Downloadt schone data zonder MultiIndex problemen."""
    all_data = []
    for t in tickers:
        try:
            df = yf.download(t, start=(datetime.now() - timedelta(days=days)), end=datetime.now(), interval="1d", progress=False)
            if df.empty: continue
            df = df.reset_index()
            # Fix voor de nieuwe yfinance kolommenstructuur
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df["Ticker"] = t
            all_data.append(df)
        except Exception as e:
            print(f"Fout bij {t}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Berekent indicatoren en behoudt de Ticker kolom (De Fix)."""
    def _calc(group):
        g = group.copy().sort_values("Date")
        close = g["Close"]
        high = g["High"]
        low = g["Low"]

        # Voeg hier je favoriete indicatoren toe
        g["MA200"] = close.rolling(200).mean()
        g["MA50"] = close.rolling(50).mean()
        
        # RSI berekening
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = _wilder_smooth(gain, 14)
        avg_loss = _wilder_smooth(loss, 14)
        rs = avg_gain / (avg_loss + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
        
        return g

    # De cruciale fix: include_groups=True voorkomt de KeyError
    return df.groupby("Ticker", group_keys=False).apply(_calc, include_groups=True)

def run_live_engine():
    """De hoofdfunctie die de bot draait en rapporteert."""
    print("--- STARTING TRADING ENGINE ---")
    
    df_raw = download_history(TICKERS)
    if df_raw.empty:
        send_telegram_message("❌ *Bot Fout*: Geen data opgehaald van Yahoo Finance.")
        return

    # Bereken indicatoren
    df_results = add_indicators(df_raw)
    
    # Pak de laatste dag van elk aandeel
    latest_data = df_results.sort_values("Date").groupby("Ticker").last().reset_index()
    
    # Maak het rapport voor Telegram
    datum_str = datetime.now().strftime('%d-%m-%Y')
    report_lines = [f"🚀 *Bot Rapport - {datum_str}*"]
    report_lines.append("-" * 20)

    for _, row in latest_data.iterrows():
        ticker = row["Ticker"]
        price = row["Close"]
        rsi = row["RSI14"]
        ma200 = row["MA200"]
        
        # Signaal Logica
        status = "WACHTEN"
        if rsi < 30 and price > ma200:
            status = "🟢 *KOOP (Ondergewaardeerd)*"
        elif rsi > 70:
            status = "🔴 *VERKOOP (Overgewaardeerd)*"
        elif rsi < 40:
            status = "🟡 *Bijna Koop*"

        line = f"`{ticker:<9}`: €{price:>7.2f} | RSI: {rsi:>4.1f} | {status}"
        report_lines.append(line)

    # Verstuur alles in één bericht naar Telegram
    final_message = "\n".join(report_lines)
    send_telegram_message(final_message)
    print("✅ Rapportage voltooid en verzonden naar Telegram.")

if __name__ == "__main__":
    run_live_engine()
