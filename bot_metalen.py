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
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def check_live_signaal(df, s, t):
    df = df.copy()
    df['F'] = df['Close'].rolling(window=s).mean()
    df['S'] = df['Close'].rolling(window=t).mean()
    if len(df) < 2: return None
    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]: return "🚀 KOOP"
    if df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]: return "💀 VERKOOP"
    return None

def bereken_bt(df, inzet, s, t):
    data = df.iloc[-252:].copy()
    data['F'], data['S'] = data['Close'].rolling(s).mean(), data['Close'].rolling(t).mean()
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        if not pos and data['F'].iloc[i] > data['S'].iloc[i] and data['F'].iloc[i-1] <= data['S'].iloc[i-1]:
            k, pos = float(data['Close'].iloc[i]), True
        elif pos and data['F'].iloc[i] < data['S'].iloc[i] and data['F'].iloc[i-1] >= data['S'].iloc[i-1]:
            saldo = (inzet * (float(data['Close'].iloc[i]) / k)) - 35
            pos = False
    return saldo

def main():
    stuur_telegram("🥇🥈🥉 *START: METALEN ANALYSE (GOUD/ZILVER/KOPER)*")
    
    with open('tickers_metalen.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    start_kapitaal = 100000
    inzet = 2500
    b1_t, b2_t = start_kapitaal - (len(tickers) * inzet), start_kapitaal - (len(tickers) * inzet)
    live = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            b1_t += bereken_bt(df, inzet, 50, 200)
            b2_t += bereken_bt(df, inzet, 20, 50)
            
            s1, s2 = check_live_signaal(df, 50, 200), check_live_signaal(df, 20, 50)
            if s1: live.append(f"• `{t}` (B1): {s1}")
            if s2: live.append(f"• `{t}` (B2): {s2}")
        except: pass

    rapport = f"🥇🥈🥉 *EINDRAPPORT METALEN*\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.0f}\n"
    rapport += f"----------------------------------\n"
    rapport += f"🤖 *Bot 1 (50/200):* €{b1_t:,.0f}\n"
    rapport += f"🤖 *Bot 2 (20/50):* €{b2_t:,.0f}\n"
    rapport += f"----------------------------------\n"
    
    if live:
        rapport += "🎯 *LIVE SIGNALEN:* \n" + "\n".join(live)
    else:
        rapport += "😴 *Geen nieuwe signalen.*"
        
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
