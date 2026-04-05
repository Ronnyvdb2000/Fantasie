import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
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
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_bt_elite(ticker, inzet, s, t, use_trend_filter=False):
    """
    DE KERN-LOGICA: 
    - S/T Crossover
    - ADX > 15
    - Volume > 60%
    - EMA200 Trend Filter (indien True)
    - ATR Trailing Stop Loss
    - Periode: Laatste 252 handelsdagen
    """
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        
        # Stap 1: Data isoleren (voorkomt Multi-index errors)
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # Stap 2: Indicatoren (SMA voor traag/snel, EMA voor hyper)
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        # ATR & ADX
        tr = pd.concat([h - l, abs(h - p.shift()), abs(l - p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        vol_ma = v.rolling(window=20).mean()

        # Stap 3: Filter op laatste jaar
        p_act, f_act, s_act = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        ema_act, v_ma_act, v_act = ema200.iloc[-252:], vol_ma.iloc[-252:], v.iloc[-252:]
        atr_act, adx_act = atr.iloc[-252:], adx.iloc[-252:]
            
        saldo_delta = 0
        pos, instap, high_p, sl = False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_act)):
            curr_p = p_act.iloc[i]
            if not pos:
                # KOOP CHECK
                if (f_act.iloc[i] > s_act.iloc[i] and f_act.iloc[i-1] <= s_act.iloc[i-1] and 
                    adx_act.iloc[i] > 15 and v_act.iloc[i] > (v_ma_act.iloc[i] * 0.6)):
                    if not use_trend_filter or curr_p > ema_act.iloc[i]:
                        instap, high_p, sl, pos = curr_p, curr_p, curr_p - (2 * atr_act.iloc[i]), True
                        saldo_delta -= kosten
            else:
                # TRAILING STOP CHECK
                high_p = max(high_p, curr_p)
                sl = max(sl, high_p - (2 * atr_act.iloc[i]))
                if curr_p < sl or f_act.iloc[i] < s_act.iloc[i]:
                    saldo_delta += (inzet * (curr_p / instap) - inzet) - kosten
                    pos = False
        return saldo_delta
    except:
        return 0

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'): return
    with open('tickers_01.txt', 'r') as f:
        raw = f.read().replace('\n', ',').replace('$', '')
        tickers = list(set([t.strip().upper() for t in raw.split(',') if t.strip()]))

    inzet = 2500.0
    # Hardcoded startwaarden
    r_traag, r_snel, r_ht, r_hs = 0.0, 0.0, 0.0, 0.0

    for t in tickers:
        r_traag += bereken_bt_elite(t, inzet, 50, 200, use_trend_filter=True)
        r_snel  += bereken_bt_elite(t, inzet, 20, 50,  use_trend_filter=True)
        r_ht    += bereken_bt_elite(t, inzet, 9, 21,   use_trend_filter=True)
        r_hs    += bereken_bt_elite(t, inzet, 9, 21,   use_trend_filter=False)

    rapport = [
        "✅ *DEFINITIEF ELITE RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200 SMA):* €{100000 + r_traag:,.0f}",
        f"⚡ *Snel (20/50 SMA):* €{100000 + r_snel:,.0f}",
        f"🚀 *Hyper Trend (9/21 EMA):* €{100000 + r_ht:,.0f}",
        f"🔥 *Hyper Scalp (9/21 EMA):* €{100000 + r_hs:,.0f}",
        "",
        "⚙️ *GARANTIE:* Alle bots gebruiken nu:",
        "• ADX > 15 | Vol > 60% | ATR Trailing Stop",
        "• Periode: Laatste 252 dagen"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
