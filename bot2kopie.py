import yfinance as yf
import pandas as pd
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
# Zorg dat deze namen EXACT zo in GitHub Secrets staan (Settings -> Secrets)
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
NEWS_KEY = os.getenv('NEWS_API_KEY')

newsapi = NewsApiClient(api_key=NEWS_KEY)
analyzer = SentimentIntensityAnalyzer()

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID:
        print("Fout: Token of Chat ID mist!")
        return
    url = f"https://api.telegram.org/bot{TOKEN.strip()}/sendMessage"
    payload = {"chat_id": str(CHAT_ID).strip(), "text": bericht, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"Telegram Fout: {r.text}")
    except Exception as e:
        print(f"Verbinding mislukt: {e}")

def haal_sentiment(ticker):
    """Vervangt de foute 'get_news_analysis' functie"""
    try:
        articles = newsapi.get_everything(q=ticker, language='en', page_size=3)
        if not articles['articles']: return 0
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in articles['articles']]
        return sum(scores) / len(scores)
    except:
        return 0

def analyseer(ticker):
    print(f"Analyse: {ticker}")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty or len(df) < 200: return None
    
    close = df['Close'].squeeze()
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    koers = float(close.iloc[-1])
    
    if sma50 > sma200:
        sent = haal_sentiment(ticker)
        return f"🚀 *{ticker}* is positief! Koers: ${koers:.2f} (Sentiment: {sent:.2f})"
    return None

def main():
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        print("aandelen.txt niet gevonden!"); return

    resultaten = []
    for t in tickers:
        res = analyseer(t)
        if res: resultaten.append(res)
    
    if resultaten:
        stuur_telegram("\n".join(resultaten))
    else:
        stuur_telegram("Bot scan voltooid: Geen actie nodig.")

if __name__ == "__main__":
    main()
