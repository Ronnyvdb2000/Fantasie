import yfinance as yf
import pandas as pd
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv

# --- 1. SETUP ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
NEWS_KEY = os.getenv('NEWS_API_KEY')

newsapi = NewsApiClient(api_key=NEWS_KEY)
analyzer = SentimentIntensityAnalyzer()

def stuur_telegram(bericht):
    """Verstuurt een bericht naar Telegram."""
    if not TOKEN or not CHAT_ID or not bericht:
        return
    url = f"https://api.telegram.org/bot{TOKEN.strip()}/sendMessage"
    payload = {"chat_id": str(CHAT_ID).strip(), "text": bericht, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram fout: {e}")

def haal_sentiment(ticker):
    """Berekent sentiment op basis van nieuws."""
    try:
        articles = newsapi.get_everything(q=ticker, language='en', page_size=3)
        if not articles['articles']: return 0
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in articles['articles']]
        return sum(scores) / len(scores)
    except:
        return 0

def analyseer_aandeel(ticker):
    """De kern-analyse functie (voorheen analyseer)."""
    print(f"Bezig met: {ticker}")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if df.empty or len(df) < 200: 
        return None
    
    close = df['Close'].squeeze()
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()
    
    s50_nu, s50_oud = float(sma50.iloc[-1]), float(sma50.iloc[-2])
    s200_nu, s200_oud = float(sma200.iloc[-1]), float(sma200.iloc[-2])
    koers = float(close.iloc[-1])

    # KOOP LOGICA
    if s50_nu > s200_nu and s50_oud <= s200_oud:
        sentiment = haal_sentiment(ticker)
        if sentiment > 0.05:
            return f"🚀 *KOOP: {ticker}*\nTrend: Golden Cross ✅\nKoers: ${koers:.2f}"

    # VERKOOP LOGICA
    if s50_nu < s200_nu and s50_oud >= s200_oud:
        return f"📉 *VERKOOP: {ticker}*\nTrend: Death Cross ⚠️\nKoers: ${koers:.2f}"

    return None

def main():
    """Hoofdprogramma."""
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("aandelen.txt niet gevonden!")
        return

    gevonden_signalen = []
    for t in tickers:
        # Hier gebruiken we de exacte naam van de functie hierboven
        res = analyseer_aandeel(t)
        if res:
            gevonden_signalen.append(res)
    
    # Rapportage aan Telegram
    if gevonden_signalen:
        stuur_telegram("🚨 *Nieuwe Signalen!*\n\n" + "\n\n".join(gevonden_signalen))
    
    # ALTIJD vermelden dat de bot gelopen heeft
    stuur_telegram("🤖 *Bot Status:* Scan voltooid. Geen nieuwe acties nodig op dit moment.")

if __name__ == "__main__":
    main()
