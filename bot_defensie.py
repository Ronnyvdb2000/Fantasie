import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_strategie(df, inzet, snelle_ma, trage_ma):
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    df = df.copy()
    df['Fast'] = df['Close'].rolling(window=snelle_ma).mean()
    df['Slow'] = df['Close'].rolling(window=trage_ma).mean()
    test_data = df.iloc[-252:].copy()
    
    positie = False
    koop_prijs = 0
    huidig_saldo = inzet
    
    for i in range(1, len(test_data)):
        f_nu, f_oud = test_data['Fast'].iloc[i], test_data['Fast'].iloc[i-1]
        s_nu, s_oud = test_data['Slow'].iloc[i], test_data['Slow'].iloc[i-1]
        prijs = float(test_data['Close'].iloc[i])

        if not positie and f_nu > s_nu and f_oud <= s_oud:
            koop_prijs = prijs
            positie = True
        elif positie and f_nu < s_nu and f_oud >= s_oud:
            rendement = prijs / koop_prijs
            bruto = inzet * rendement
            huidig_saldo = bruto - (VASTE_KOST * 2) - (inzet * BEURSTAKS_PCT * 2)
            positie = False

    if positie:
        huidig_saldo = (inzet * (float(test_data['Close'].iloc[-1]) / koop_prijs)) - 35
    return huidig_saldo

def main():
    stuur_telegram("🦈 *START: DEFENSIE SCAN (INCL. KLEINE VISSEN)*")
    
    try:
        with open('tickers_defensie.txt', 'r') as f:
            tickers = [t.strip() for t in f.read().split(',') if t.strip()]
    except:
        tickers = ['LMT', 'AVAV', 'PLTR', 'RHM.DE']

    resultaten = []

    for t in tickers:
        print(f"Scannen: {t}")
        df = yf.download(t, period="2y", progress=False)
        if df.empty or len(df) < 200: continue
        
        r1 = bereken_strategie(df, 2500, 50, 200)
        r2 = bereken_strategie(df, 2500, 20, 50)
        
        # Welke bot wint voor DIT specifiek aandeel?
        fav = "🤖1" if r1 > r2 else "🤖2"
        resultaten.append(f"`{t.split('.')[0]:<6}`: {fav} (Best: €{max(r1, r2):.0f})")

    # Splits resultaten in blokken van 10 voor leesbaarheid in Telegram
    for i in range(0, len(resultaten), 10):
        bericht = "📊 *Check per aandeel:*\n" + "\n".join(resultaten[i:i+10])
        stuur_telegram(bericht)

if __name__ == "__main__":
    main()
