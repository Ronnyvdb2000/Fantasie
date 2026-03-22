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
        print("Telegram fout.")

def check_live_signaal(df, s, t, ticker):
    df = df.copy()
    df['F'] = df['Close'].rolling(window=s).mean()
    df['S'] = df['Close'].rolling(window=t).mean()
    
    link = f"https://finance.yahoo.com/quote/{ticker}"
    
    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]:
        return f"🚀 *KOOP* ([Grafiek]({link}))"
    elif df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]:
        return f"💀 *VERKOOP* ([Grafiek]({link}))"
    return None

def bereken_bt(df, inzet, s, t):
    # Bolero tarieven voor Euronext (Brussel/Amsterdam)
    VASTE_KOST = 7.50 
    BEURSTAKS_BE = 0.0135 # TOB voor Belgische aandelen
    
    data = df.iloc[-252:].copy()
    data['F'], data['S'] = data['Close'].rolling(s).mean(), data['Close'].rolling(t).mean()
    
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        if not pos and data['F'].iloc[i] > data['S'].iloc[i] and data['F'].iloc[i-1] <= data['S'].iloc[i-1]:
            k, pos = float(data['Close'].iloc[i]), True
        elif pos and data['F'].iloc[i] < data['S'].iloc[i] and data['F'].iloc[i-1] >= data['S'].iloc[i-1]:
            bruto = inzet * (float(data['Close'].iloc[i]) / k)
            # Kosten bij aan- en verkoop + taksen
            saldo = bruto - (VASTE_KOST * 2) - (inzet * BEURSTAKS_BE) - (bruto * BEURSTAKS_BE)
            pos = False
    return saldo

def main():
    stuur_telegram("🇧🇪🇳🇱 *START ANALYSE: BENELUX BLUE CHIPS*")
    
    start_kapitaal = 100000
    inzet = 2500
    with open('tickers_benelux.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    b1_t, b2_t = start_kapitaal - (len(tickers)*inzet), start_kapitaal - (len(tickers)*inzet)
    live = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            
            b1_t += bereken_bt(df, inzet, 50, 200)
            b2_t += bereken_bt(df, inzet, 20, 50)
            
            s1 = check_live_signaal(df, 50, 200, t)
            s2 = check_live_signaal(df, 20, 50, t)
            if s1: live.append(f"• `{t}` (B1): {s1}")
            if s2: live.append(f"• `{t}` (B2): {s2}")
        except: pass

    rapport = f"🇧🇪🇳🇱 *EINDRAPPORT BENELUX*\n"
    rapport += f"💰 Portefeuille-simulatie: €{start_kapitaal:,.0f}\n"
    rapport += f"----------------------------------\n"
    rapport += f"🤖 *Bot 1 (50/200):* €{b1_t:,.0f}\n"
    rapport += f"🤖 *Bot 2 (20/50):* €{b2_t:,.0f}\n"
    rapport += f"----------------------------------\n"
    
    if live:
        rapport += "🎯 *SIGNALEER & CHECK OP BOLERO:*\n" + "\n".join(live)
    else:
        rapport += "😴 *Geen actie vereist voor de Benelux.*"

    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
