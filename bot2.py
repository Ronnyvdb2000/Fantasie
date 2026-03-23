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
    rs = ema_up / (ema_down + 1e-10)
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
    
    # Momentum-specifieke RSI labels
    if rsi_nu > 75: status = "🔥 EXTREEM OVERVERHIT"
    elif rsi_nu > 60: status = "⚡ STERK MOMENTUM"
    elif rsi_nu < 35: status = "💎 ONDERGEWAARDEERD"
    else: status = "✅ GEZOND"

    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]:
        return f"🚀 *MOMENTUM KOOP* | RSI: {rsi_nu} ({status}) | [Grafiek]({link})"
    elif df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]:
        return f"💀 *MOMENTUM EXIT* | RSI: {rsi_nu} | [Grafiek]({link})"
    return None

def bereken_bt(df, inzet, s, t):
    # Bolero Kosten (Gemiddelde voor snelle trades)
    VASTE_KOST = 15.00 
    BEURSTAKS = 0.0035 
    
    data = df.iloc[-252:].copy()
    data['F'], data['S'] = data['Close'].rolling(s).mean(), data['Close'].rolling(t).mean()
    
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        f_nu, f_oud = data['F'].iloc[i], data['F'].iloc[i-1]
        s_nu, s_oud = data['S'].iloc[i], data['S'].iloc[i-1]
        prijs = float(data['Close'].iloc[i])

        if not pos and f_nu > s_nu and f_oud <= s_oud:
            k, pos = prijs, True
            saldo -= (VASTE_KOST + (inzet * BEURSTAKS))
        elif pos and f_nu < s_nu and f_oud >= s_oud:
            bruto = inzet * (prijs / k)
            saldo += (bruto - inzet) - (VASTE_KOST + (bruto * BEURSTAKS))
            pos = False
    return saldo

def main():
    stuur_telegram("⚡ *BOT 2: SNELLE MOMENTUM SCAN (20/50)*")
    
    ticker_file = 'aandelen.txt'
    if not os.path.exists(ticker_file):
        stuur_telegram("❌ `tickers.txt` mist.")
        return

    with open(ticker_file, 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    start_kap, inzet = 100000, 2500
    b2_rendement, live = start_kap, []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False)
            if df.empty or len(df) < 60: continue
            
            b2_rendement += (bereken_bt(df, inzet, 20, 50) - inzet)
            
            sig = check_live_signaal(df, 20, 50, t)
            if sig: live.append(f"• `{t}`: {sig}")
        except: pass

    rapport = f"⚡ *BOT 2 MOMENTUM RAPPORT*\n----------------------------------\n"
    rapport += f"💰 Netto Rendement (20/50): €{b2_rendement:,.0f}\n"
    rapport += "\n🎯 *SNELLE SIGNALEN:*\n"
    rapport += "\n".join(live) if live else "😴 De markten zijn rustig."
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
