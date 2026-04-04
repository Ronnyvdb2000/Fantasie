import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

# Configuratie via GitHub Secrets of .env bestand
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
    if not TOKEN or not CHAT_ID: 
        print("Telegram configuratie ontbreekt.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"Telegram fout: {e}")

def check_live_signaal(df, s, t, ticker, is_ema=False):
    df = df.copy()
    if is_ema:
        df['F'] = df['Close'].ewm(span=s, adjust=False).mean()
        df['S'] = df['Close'].ewm(span=t, adjust=False).mean()
    else:
        df['F'] = df['Close'].rolling(window=s).mean()
        df['S'] = df['Close'].rolling(window=t).mean()
    
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = bereken_rsi(df['Close'])
    
    rsi_nu = round(float(df['RSI'].iloc[-1]), 1)
    prijs_nu = round(float(df['Close'].iloc[-1]), 2)
    ema200_nu = round(float(df['EMA200'].iloc[-1]), 2)
    link = f"https://finance.yahoo.com/quote/{ticker}"
    
    status = "🔥 OVERVERHIT" if rsi_nu > 72 else "✅ TREND GEZOND"
    if rsi_nu < 35: status = "💎 ONDERGEWAARDEERD"
    elif rsi_nu > 60 and rsi_nu <= 72: status = "⚡ STERK MOMENTUM"

    details = f"€{prijs_nu} | RSI: {rsi_nu} | EMA200: €{ema200_nu} ({status}) [Grafiek]({link})"

    if df['F'].iloc[-1] > df['S'].iloc[-1] and df['F'].iloc[-2] <= df['S'].iloc[-2]:
        return f"🚀 *KOOP* | {details}"
    elif df['F'].iloc[-1] < df['S'].iloc[-1] and df['F'].iloc[-2] >= df['S'].iloc[-2]:
        return f"💀 *VERKOOP* | {details}"
    return None

def bereken_bt(df, inzet, s, t, is_ema=False):
    VASTE_KOST = 15.00 
    BEURSTAKS = 0.0035 
    
    data = df.iloc[-252:].copy()
    if is_ema:
        data['F'] = data['Close'].ewm(span=s, adjust=False).mean()
        data['S'] = data['Close'].ewm(span=t, adjust=False).mean()
    else:
        data['F'] = data['Close'].rolling(s).mean()
        data['S'] = data['Close'].rolling(t).mean()
    
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
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    stuur_telegram(f"🛡️ *HOOGLAND SCANNER V2*\n_Gestart op {nu}_")
    
    ticker_file = 'tickers_01.txt'
    if not os.path.exists(ticker_file):
        stuur_telegram(f"❌ Fout: `{ticker_file}` niet gevonden.")
        return

    with open(ticker_file, 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    start_kap,
