import yfinance as yf
import pandas as pd
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv

# --- 1. CONFIGURATIE EN SETUP ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def stuur_telegram(bericht):
    """Verstuurt een geformatteerd bericht naar de opgegeven Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Fout: Telegram credentials missen!")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code != 200:
            print(f"Telegram API Fout: {response.text}")
    except Exception as e:
        print(f"Verbindingsfout Telegram: {e}")

def get_sentiment(ticker):
    """Haalt de laatste 5 nieuwsberichten op en berekent de gemiddelde score."""
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)
        if not articles['articles']:
            return 0
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in articles['articles']]
        return sum(scores) / len(scores)
    except Exception:
        return 0

def analyseer_aandeel(ticker):
    """Berekent indicatoren en checkt op koop/verkoop signalen."""
    print(f"Bezig met analyseren van: {ticker}...")
    
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if df.empty or len(df) < 200:
        print(f"Slaan {ticker} over: onvoldoende data.")
        return None

    close_prices = df['Close'].squeeze()
    volumes = df['Volume'].squeeze()

    sma50 = close_prices.rolling(window=50).mean()
    sma200 = close_prices.rolling(window=200).mean()
    avg_volume = volumes.rolling(window=20).mean()

    # Gebruik float() om zeker te weten dat we een getal hebben voor de f-string
    huidige_koers = float(close_prices.iloc[-1])
    vorige_koers = float(close_prices.iloc[-2])
    
    sma50_nu = float(sma50.iloc[-1])
    sma200_nu = float(sma200.iloc[-1])
    sma50_gisteren = float(sma50.iloc[-2])
    sma200_gisteren = float(sma200.iloc[-2])
    
    huidig_volume = float(volumes.iloc[-1])
    gem_volume = float(avg_volume.iloc[-1])
    volume_factor = huidig_volume / gem_volume

    # --- LOGICA ---
    # 1. KOOP: Golden Cross + Volume
    if sma50_nu > sma200_nu and sma50_gisteren <= sma200_gisteren:
        if volume_factor >= 1.2:
            sentiment = get_sentiment(ticker)
            if sentiment > 0.05:
                return (f"🚀 *KOOP SIGNAAL: {ticker}*\n"
                        f"• Trend: Golden Cross ✅\n"
                        f"• Volume: {volume_factor:.1f}x gem. ✅\n"
                        f"• Sentiment: {sentiment:.2f} ✅\n"
                        f"• Prijs: ${huidige_koers:.2f}")

    # 2. VERKOOP: Death Cross
    if sma50_nu < sma200_nu and sma50_gisteren >= sma200_gisteren:
        return f"📉 *VERKOOP: {ticker}*\nTrendbreuk (Death Cross). Koers: ${huidige_koers:.2f}"

    # 3. STOP-LOSS: 5% daling
    if huidige_koers < vorige_koers * 0.95:
        return f"⚠️ *SHARP DROP: {ticker}*\nKoers >5% gedaald! Prijs: ${huidige_koers:.2f}"

    return None

def main():
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        print("aandelen.txt niet gevonden!"); return

    signalen = []
    for t in tickers:
        res = analyseer(t)
        if res:
            signalen.append(res)
    
    # OUTPUT LOGICA
    if signalen:
        # Stuur alle koop/verkoop signalen
        stuur_telegram("🚨 *Nieuwe Signalen!*\n\n" + "\n\n".join(signalen))
    else:
        # Alleen een korte bevestiging dat de bot gedraaid heeft
        stuur_telegram("🤖 *Bot Status:* Scan voltooid. Geen nieuwe signalen gevonden.")

if __name__ == "__main__":
    main()
