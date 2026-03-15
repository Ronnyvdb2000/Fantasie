import yfinance as yf
import pandas as pd
import requests
import os
import time
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
    """Verstuurt een bericht naar Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Fout: {e}")

def get_sentiment(ticker):
    """Berekent het gemiddelde sentiment van de laatste 5 nieuwsberichten."""
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)
        if not articles['articles']:
            return 0
        scores = [analyzer.polarity_scores(a['title'])['compound'] for a in articles['articles']]
        return sum(scores) / len(scores)
    except:
        return 0

def analyseer_aandeel(ticker):
    """Swing-analyse logica (EMA 13/48 + RSI + Volume)."""
    print(f"Bezig met analyseren van: {ticker}...")
    
    # Haal 6 maanden aan data op voor betrouwbare EMA en RSI
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    
    if df.empty or len(df) < 50:
        return None

    # Gebruik .squeeze() om zeker te zijn van Series formaat
    close_prices = df['Close'].squeeze()
    volumes = df['Volume'].squeeze()

    # 1. EMA Berekeningen (Exponential Moving Average)
    ema13 = close_prices.ewm(span=13, adjust=False).mean()
    ema48 = close_prices.ewm(span=48, adjust=False).mean()

    # 2. RSI Berekening (14 dagen)
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # 3. Waarden van vandaag en gisteren ophalen
    huidige_prijs = close_prices.iloc[-1]
    vorige_prijs = close_prices.iloc[-2]
    ema13_nu = ema13.iloc[-1]
    ema48_nu = ema48.iloc[-1]
    ema13_gisteren = ema13.iloc[-2]
    ema48_gisteren = ema48.iloc[-2]
    rsi_nu = rsi.iloc[-1]
    
    huidig_volume = volumes.iloc[-1]
    gem_volume = volumes.rolling(window=20).mean().iloc[-1]
    vol_factor = huidig_volume / gem_volume

    # --- SWING TRADING LOGICA ---

    # A. KOOP SIGNAAL: EMA Cross naar boven + RSI niet oververhit + Hoog volume
    if ema13_nu > ema48_nu and ema13_gisteren <= ema48_gisteren:
        if rsi_nu < 65 and vol_factor > 1.2:
            sentiment = get_sentiment(ticker)
            return (f"🚀 *SWING KOOP: {ticker}*\n"
                    f"• Prijs: ${huidige_prijs:.2f}\n"
                    f"• Trend: EMA 13/48 Cross ✅\n"
                    f"• RSI: {rsi_nu:.1f} ✅\n"
                    f"• Volume: {vol_factor:.1f}x ✅\n"
                    f"• Sentiment: {sentiment:.2f}")

    # B. VERKOOP SIGNAAL: EMA Cross naar beneden
    if ema13_nu < ema48_nu and ema13_gisteren >= ema48_gisteren:
        return f"📉 *SWING VERKOOP: {ticker}*\nTrendbreuk gedetecteerd (EMA Bearish Cross). Prijs: ${huidige_prijs:.2f}"

    # C. WINST NEMEN: RSI is extreem hoog
    if rsi_nu > 75:
        return f"💰 *WINST NEMEN? {ticker}*\nAandeel is oververhit (RSI: {rsi_nu:.1f})."

    # D. STOP-LOSS: 4% daling in één dag
    if huidige_prijs < vorige_prijs * 0.96:
        return f"⚠️ *SHARP DROP: {ticker}*\nKoers is meer dan 4% gedaald! Prijs: ${huidige_prijs:.2f}"

    return None

def main():
    if not os.path.exists('aandelen.txt'):
        print("Fout: aandelen.txt niet gevonden!")
        return

    with open('aandelen.txt', 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    gevonden_signalen = []
    totaal_overzicht = []

    for t in tickers:
        # Prijzen ophalen voor de status update
        try:
            temp_df = yf.download(t, period="5d", progress=False)
            if not temp_df.empty:
                laatste_prijs = temp_df['Close'].iloc[-1].item()
                totaal_overzicht.append(f"{t}: ${laatste_prijs:.2f}")
        except:
            pass

        # Analyse uitvoeren
        resultaat = analyseer_aandeel(t)
        if resultaat:
            gevonden_signalen.append(resultaat)
        
        time.sleep(1) # API limit respecteren

    # --- BERICHTGEVING ---

    # 1. Signalen sturen (indien aanwezig)
    if gevonden_signalen:
        signaal_bericht = "🔔 *SWING ALERTS:*\n\n" + "\n\n".join(gevonden_signalen)
        stuur_telegram(signaal_bericht)
    
    # 2. Dagelijkse Status Update
    status_tekst = "🤖 *Ronny Bot Status Update*\n"
    status_tekst += "Marktscan voltooid.\n\n"
    status_tekst += "*Huidige koersen:*\n" + "\n".join(totaal_overzicht)
    
    if not gevonden_signalen:
        stuur_telegram(status_tekst)
    else:
        stuur_telegram("✅ Overige aandelen in de lijst zijn gecontroleerd.")

if __name__ == "__main__":
    main()