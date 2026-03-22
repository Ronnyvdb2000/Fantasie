import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings
from datetime import datetime

# Instellingen
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except:
        print("Telegram verzenden mislukt.")

def check_live_signaal(df, snelle_ma, trage_ma, ticker):
    """Checkt op kruising en genereert een directe analyse-link"""
    df = df.copy()
    df['Fast'] = df['Close'].rolling(window=snelle_ma).mean()
    df['Slow'] = df['Close'].rolling(window=trage_ma).mean()
    
    nu_f, oud_f = df['Fast'].iloc[-1], df['Fast'].iloc[-2]
    nu_s, oud_s = df['Slow'].iloc[-1], df['Slow'].iloc[-2]
    
    # Maak Yahoo Finance link (handig voor Bolero check)
    link = f"https://finance.yahoo.com/quote/{ticker}"
    
    if nu_f > nu_s and oud_f <= oud_s:
        return f"🚀 *KOOP* ([Grafiek]({link}))"
    elif nu_f < nu_s and oud_f >= oud_s:
        return f"💀 *VERKOOP* ([Grafiek]({link}))"
    return None

def bereken_backtest(df, inzet, snelle_ma, trage_ma):
    VASTE_KOST = 15.00 # Bolero gemiddelde
    BEURSTAKS_PCT = 0.0035
    
    test_data = df.iloc[-252:].copy()
    test_data['Fast'] = test_data['Close'].rolling(window=snelle_ma).mean()
    test_data['Slow'] = test_data['Close'].rolling(window=trage_ma).mean()
    
    positie, koop_prijs, saldo = False, 0, inzet
    for i in range(1, len(test_data)):
        f_nu, f_oud = test_data['Fast'].iloc[i], test_data['Fast'].iloc[i-1]
        s_nu, s_oud = test_data['Slow'].iloc[i], test_data['Slow'].iloc[i-1]
        prijs = float(test_data['Close'].iloc[i])

        if not positie and f_nu > s_nu and f_oud <= s_oud:
            koop_prijs, positie = prijs, True
        elif positie and f_nu < s_nu and f_oud >= s_oud:
            bruto = inzet * (prijs / koop_prijs)
            kosten = (VASTE_KOST * 2) + (inzet * BEURSTAKS_PCT) + (bruto * BEURSTAKS_PCT)
            saldo = bruto - kosten
            positie = False
    if positie:
        saldo = (inzet * (float(test_data['Close'].iloc[-1]) / koop_prijs)) - 35
    return saldo

def main():
    stuur_telegram("🛡️ *START ANALYSE: DEFENSIE (BOLERO READY)*")
    
    start_kapitaal = 100000
    inzet = 2500
    with open('tickers_defensie.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    b1_totaal, b2_totaal = start_kapitaal - (len(tickers)*inzet), start_kapitaal - (len(tickers)*inzet)
    live_meldingen = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            
            b1_totaal += bereken_backtest(df, inzet, 50, 200)
            b2_totaal += bereken_backtest(df, inzet, 20, 50)
            
            # Live Signalen met link
            sig1 = check_live_signaal(df, 50, 200, t)
            sig2 = check_live_signaal(df, 20, 50, t)
            if sig1: live_meldingen.append(f"• `{t}` (Bot 1): {sig1}")
            if sig2: live_meldingen.append(f"• `{t}` (Bot 2): {sig2}")
        except: pass

    rapport = f"🛡️ *EINDRAPPORT DEFENSIE*\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.0f}\n"
    rapport += f"----------------------------------\n"
    rapport += f"🤖 *Bot 1 (50/200):* €{b1_totaal:,.0f}\n"
    rapport += f"🤖 *Bot 2 (20/50):* €{b2_totaal:,.0f}\n"
    rapport += f"----------------------------------\n"
    
    if live_meldingen:
        rapport += "🎯 *SIGNALEER & CHECK OP BOLERO:*\n" + "\n".join(live_meldingen)
    else:
        rapport += "😴 *Geen actie vereist op dit moment.*"

    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
