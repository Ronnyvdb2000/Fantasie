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

def bereken_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def check_live_signaal(df, s, t, ticker):
    df = df.copy()
    df['F'] = df['Close'].rolling(window=s).mean()
    df['S'] = df['Close'].rolling(window=t).mean()
    df['RSI'] = bereken_rsi(df['Close'])
    
    rsi_nu = round(df['RSI'].iloc[-1], 1)
    link = f"https://finance.yahoo.com/quote/{ticker}"
    
    # Status bepalen op basis van RSI
    status = "🔥 OVERVERHIT" if rsi_nu > 70 else "✅ GEZOND"
    if rsi_nu < 35: status = "💎 KOOPJE"

    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]:
        return f"🚀 *KOOP* | RSI: {rsi_nu} ({status}) | [Grafiek]({link})"
    elif df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]:
        return f"💀 *VERKOOP* | RSI: {rsi_nu} | [Grafiek]({link})"
    return None

def bereken_bt(df, inzet, s, t):
    VASTE_KOST, BEURSTAKS = 15.00, 0.0035
    data = df.iloc[-252:].copy()
    data['F'], data['S'] = data['Close'].rolling(s).mean(), data['Close'].rolling(t).mean()
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        if not pos and data['F'].iloc[i] > data['S'].iloc[i] and data['F'].iloc[i-1] <= data['S'].iloc[i-1]:
            k, pos = float(data['Close'].iloc[i]), True
        elif pos and data['F'].iloc[i] < data['S'].iloc[i] and data['F'].iloc[i-1] >= data['S'].iloc[i-1]:
            bruto = inzet * (float(data['Close'].iloc[i]) / k)
            saldo = bruto - (VASTE_KOST * 2) - (inzet * BEURSTAKS) - (bruto * BEURSTAKS)
            pos = False
    return saldo

def main():
    stuur_telegram("⚛️🤖 *POWER & AI: SCAN INCLUSIEF RSI-FILTER*")
    with open('tickers_power.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    start_kap = 100000
    inzet = 2500
    b1, b2, live = start_kap, start_kap, []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 200: continue
            b1 += (bereken_bt(df, inzet, 50, 200) - inzet)
            b2 += (bereken_bt(df, inzet, 20, 50) - inzet)
            s1, s2 = check_live_signaal(df, 50, 200, t), check_live_signaal(df, 20, 50, t)
            if s1: live.append(f"• `{t}` (B1): {s1}")
            if s2: live.append(f"• `{t}` (B2): {s2}")
        except: pass

    rapport = f"⚛️🤖 *RENDEMENTSRAPPORT POWER & AI*\n----------------------------------\n"
    rapport += f"🤖 *Bot 1 (50/200):* €{b1:,.0f}\n🤖 *Bot 2 (20/50):* €{b2:,.0f}\n"
    rapport += "\n🎯 *LIVE SIGNALEN:*\n" + ("\n".join(live) if live else "😴 Geen actie.")
    stuur_telegram(rapport)

if __name__ == "__main__": main()
