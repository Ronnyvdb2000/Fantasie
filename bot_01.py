import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

# --- SETUP ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False, is_hyper=False, use_mean_rev=False):
    try:
        df = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0, None
        
        # Data extractie
        p = df['Close'].ffill().astype(float)
        if isinstance(p, pd.DataFrame): p = p.iloc[:, 0]
        h = df['High'].ffill().astype(float)
        if isinstance(h, pd.DataFrame): h = h.iloc[:, 0]
        l = df['Low'].ffill().astype(float)
        if isinstance(l, pd.DataFrame): l = l.iloc[:, 0]

        # INDICATOREN
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        ma5 = p.rolling(window=5).mean()
        
        # RSI2 (Robuuste versie)
        delta = p.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=2).mean()
        avg_loss = loss.rolling(window=2).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi2 = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # BACKTEST (Laatste 252 dagen)
        # We pakken de index van de laatste 252 dagen
        idx = p.index[-252:]
        profit, pos, instap, sl_val = 0.0, False, 0.0, 0.0
        kosten = 15.0 + (inzet * 0.0035)

        for d in idx:
            cp = float(p.loc[d])
            cr2 = float(rsi2.loc[d])
            cma5 = float(ma5.loc[d])
            catr = float(atr.loc[d])
            cema = float(ema200.loc[d])
            
            if not pos:
                if use_mean_rev:
                    # Koop dip onder 25 (Connors)
                    if cr2 < 25:
                        instap, sl_val, pos = cp, cp - (2 * catr), True
                        profit -= kosten
                else:
                    # Hyper/Trend logica
                    if f_line.loc[d] > s_line.loc[d]:
                        if not use_trend_filter or cp > cema:
                            instap, sl_val, pos = cp, cp - (2 * catr), True
                            profit -= kosten
            else:
                # Verkoop logica
                sell = False
                if use_mean_rev:
                    if cp > cma5 or cp < sl_val: sell = True
                else:
                    if f_line.loc[d] < s_line.loc[d] or cp < sl_val: sell = True
                
                if sell:
                    winst = (inzet * (cp / instap) - inzet) - kosten
                    if winst > 0: winst *= 0.90
                    profit += winst
                    pos = False
                    
        return profit, None
    except: return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}

    for t in tickers:
        print(f"Check {t}...")
        for k, prm in [("T",(50,200,True,False,False)), ("S",(20,50,True,False,False)), ("HT",(9,21,True,True,False)), ("HS",(9,21,False,True,False)), ("MR",(0,0,False,False,True))]:
            p, _ = bereken_alles(t, inzet, prm[0], prm[1], prm[2], is_hyper=prm[3], use_mean_rev=prm[4])
            res[k] += p

    rapport = (
        f"📊 *{label} {naam_sector}*\n"
        f"----------------------------------\n"
        f"🐢 Traag: €{100000 + res['T']:,.0f}\n"
        f"⚡ Snel: €{100000 + res['S']:,.0f}\n"
        f"🚀 Hyper T: €{100000 + res['HT']:,.0f}\n"
        f"🔥 Hyper S: €{100000 + res['HS']:,.0f}\n"
        f"📉 Mean Rev: €{100000 + res['MR']:,.0f}"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    voer_lijst_uit("tickers_01.txt", "01", "Hoogland")
