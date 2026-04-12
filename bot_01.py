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
    
    if len(bericht) > 4000:
        parts = [bericht[i:i+4000] for i in range(0, len(bericht), 4000)]
    else:
        parts = [bericht]

    for part in parts:
        try:
            response = requests.post(url, data={
                "chat_id": CHAT_ID, 
                "text": part, 
                "parse_mode": "Markdown", 
                "disable_web_page_preview": True
            }, timeout=20)
            
            if response.status_code != 200:
                requests.post(url, data={
                    "chat_id": CHAT_ID, 
                    "text": part, 
                    "disable_web_page_preview": True
                }, timeout=20)
            time.sleep(0.5)
        except:
            pass

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False, is_hyper=False, use_macd=False):
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0, None
        
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        vol_ma = v.rolling(window=20).mean()
        
        exp1 = p.ewm(span=12, adjust=False).mean()
        exp2 = p.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        if is_hyper:
            rsi3_gain = (delta.where(delta > 0, 0)).rolling(3).mean()
            rsi3_loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
            rsi3 = 100 - (100 / (1 + (rsi3_gain / (rsi3_loss + 1e-10))))
            streak = pd.Series(0, index=p.index)
            for i in range(1, len(p)):
                if p.iloc[i] > p.iloc[i-1]: streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] > 0 else 1
                elif p.iloc[i] < p.iloc[i-1]: streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] < 0 else -1
            s_delta = streak.diff()
            s_gain = (s_delta.where(s_delta > 0, 0)).rolling(2).mean()
            s_loss = (-s_delta.where(s_delta < 0, 0)).rolling(2).mean()
            streak_rsi = 100 - (100 / (1 + (s_gain / (s_loss + 1e-10))))
            p_rank = delta.rolling(100).apply(lambda x: (x < x.iloc[-1]).sum() / 99.0 * 100, raw=False)
            rsi_val = (rsi3 + streak_rsi + p_rank) / 3

        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

        p_bt = p.iloc[-252:]
        f_bt, s_bt, e_bt = f_line.iloc[-252:], s_line.iloc[-252:], ema200.iloc[-252:]
        m_bt, sig_bt = macd_line.iloc[-252:], signal_line.iloc[-252:]
        v_bt, v_ma_bt = v.iloc[-252:], vol_ma.iloc[-252:]
        atr_bt = atr_series.iloc[-252:]
        
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for
