import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_mr_apart(p, inzet):
    # Handmatige RSI2 berekening om NaN te voorkomen
    delta = p.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=2).mean()
    avg_loss = loss.rolling(window=2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    
    ma5 = p.rolling(window=5).mean()
    
    # Gebruik alleen het laatste jaar voor backtest
    p_bt = p.iloc[-252:]
    r_bt = rsi2.iloc[-252:]
    m_bt = ma5.iloc[-252:]
    
    profit, pos, instap = 0.0, False, 0.0
    kosten = 15.0 + (inzet * 0.0035)

    for i in range(1, len(p_bt)):
        cp = float(p_bt.iloc[i])
        cr = float(r_bt.iloc[i])
        
        if not pos:
            # We verlagen de drempel naar 30 om trades te FORCEREN
            if cr < 30 and cr > 0: 
                instap = cp
                profit -= kosten
                pos = True
        else:
            if cp > m_bt.iloc[i] or i == len(p_bt) - 1:
                w = (inzet * (cp / instap) - inzet) - kosten
                if w > 0: w *= 0.9
                profit += w
                pos = False
    return profit

def bereken_trend(ticker, inzet, s, t, use_filter=False):
    try:
        df = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0, 0
        p = df['Close'].ffill().astype(float)
        if isinstance(p, pd.DataFrame): p = p.iloc[:, 0]
        
        # Trend strategieën
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        p_bt = p.iloc[-252:]
        f_bt = f_line.iloc[-252:]
        s_bt = s_line.iloc[-252:]
        e_bt = ema200.iloc[-252:]
        
        profit, pos, instap = 0, False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if not use_filter or cp > e_bt.iloc[i]:
                        instap, pos = cp, True
                        profit -= kosten
            else:
                if f_bt.iloc[i] < s_bt.iloc[i]:
                    w = (inzet * (cp / instap) - inzet) - kosten
                    if w > 0: w *= 0.9
                    profit += w
                    pos = False
        
        # Mean Reversion apart berekenen
        mr_profit = bereken_mr_apart(p, inzet)
        
        return profit, mr_profit
    except: return 0, 0

def voer_lijst_uit(bestandsnaam):
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}

    for t in tickers:
        # We halen de resultaten op. Merk op: we berekenen MR per ticker maar 1 keer
        p_t, _ = bereken_trend(t, inzet, 50, 200, True)
        p_s, _ = bereken_trend(t, inzet, 20, 50, True)
        p_ht, _ = bereken_trend(t, inzet, 9, 21, True)
        p_hs, p_mr = bereken_trend(t, inzet, 9, 21, False)
        
        res["T"] += p_t
        res["S"] += p_s
        res["HT"] += p_ht
        res["HS"] += p_hs
        res["MR"] += p_mr

    rapport = (
        f"📊 *RESULTATEN*\n"
        f"🐢 Traag: €{100000 + res['T']:,.0f}\n"
        f"⚡ Snel: €{100000 + res['S']:,.0f}\n"
        f"🚀 Hyper T: €{100000 + res['HT']:,.0f}\n"
        f"🔥 Hyper S: €{100000 + res['HS']:,.0f}\n"
        f"📉 Mean Rev: €{100000 + res['MR']:,.0f}"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    voer_lijst_uit("tickers_01.txt")
