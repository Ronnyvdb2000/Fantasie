import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings
from datetime import datetime

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
        print("Telegram mislukt.")

def check_live_signaal(df, snelle_ma, trage_ma):
    df = df.copy()
    df['Fast'] = df['Close'].rolling(window=snelle_ma).mean()
    df['Slow'] = df['Close'].rolling(window=trage_ma).mean()
    if len(df) < 2: return None
    nu_f, oud_f = df['Fast'].iloc[-1], df['Fast'].iloc[-2]
    nu_s, oud_s = df['Slow'].iloc[-1], df['Slow'].iloc[-2]
    
    if nu_f > nu_s and oud_f <= oud_s: return "🚀 KOOP"
    if nu_f < nu_s and oud_f >= oud_s: return "💀 VERKOOP"
    return None

def bereken_backtest(df, inzet, snelle_ma, trage_ma):
    # Kosten voor kleinere posities (vaak meer trades bij small caps)
    VASTE_KOST = 10.00 
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
            saldo -= VASTE_KOST
        elif positie and f_nu < s_nu and f_oud >= s_oud:
            saldo = (saldo * (prijs / koop_prijs)) - VASTE_KOST
            positie = False
    return saldo

def main():
    stuur_telegram("⚛️🤖 *START: POWER & AI (INCL. SMALL CAPS)*\n_Lijst van 25 tickers wordt geanalyseerd..._")
    
    start_kapitaal = 100000
    inzet = 2500
    with open('tickers_power.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    bot1_totaal, bot2_totaal = 0, 0
    live_signals = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            
            bot1_totaal += bereken_backtest(df, inzet, 50, 200)
            bot2_totaal += bereken_backtest(df, inzet, 20, 50)
            
            s1 = check_live_signaal(df, 50, 200)
            s2 = check_live_signaal(df, 20, 50)
            if s1: live_signals.append(f"• `{t}` (B1): {s1}")
            if s2: live_signals.append(f"• `{t}` (B2): {s2}")
        except: pass

    rapport = f"⚛️🤖 *EINDRAPPORT POWER SECTOR*\n"
    rapport += f"📊 Analyse van {len(tickers)} bedrijven\n"
    rapport += f"----------------------------------\n"
    rapport += f"🤖 *Bot 1 (50/200):* €{bot1_totaal:,.0f}\n"
    rapport += f"🤖 *Bot 2 (20/50):* €{bot2_totaal:,.0f}\n"
    rapport += f"----------------------------------\n"
    
    if live_signals:
        rapport += "🎯 *LIVE SIGNALEN (ACTIE VEREIST):*\n" + "\n".join(live_signals)
    else:
        rapport += "😴 *Geen kruisingen gedetecteerd.*"
        
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
