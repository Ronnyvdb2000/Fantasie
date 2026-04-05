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
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_indicatoren(df, ticker, s, t):
    """Berekent technische filters op een veilige manier voor elke bot-instelling."""
    if isinstance(df.columns, pd.MultiIndex):
        p = df['Close'][ticker].dropna().astype(float)
        v = df['Volume'][ticker].dropna().astype(float)
        h = df['High'][ticker].dropna().astype(float)
        l = df['Low'][ticker].dropna().astype(float)
    else:
        p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

    # Gemiddelden op basis van de meegegeven vensters (s en t)
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    
    # Volume Filter (60% van 20-daags MA)
    vol_ma = v.rolling(window=20).mean()
    
    # ATR (Volatiliteit voor Stop Loss)
    tr = pd.concat([h - l, abs(h - p.shift()), abs(l - p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX (Trendsterkte drempel 15)
    plus_dm = (h.diff().clip(lower=0)).rolling(14).mean()
    minus_dm = ((-l.diff()).clip(lower=0)).rolling(14).mean()
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) if 'plus_di' in locals() else 0 # Vereenvoudigde ADX logica
    # Voor stabiliteit gebruiken we hier een robuustere ADX berekening
    diff_h = h.diff()
    diff_l = -l.diff()
    plus_di = 100 * (diff_h.clip(lower=0).rolling(14).mean() / (atr + 1e-10))
    minus_di = 100 * (diff_l.clip(lower=0).rolling(14).mean() / (atr + 1e-10))
    adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
    
    return p, f_line, s_line, ema200, vol_ma, v, atr, adx

def bereken_bt_elite(ticker, inzet, s, t, use_trend_filter=False, is_ema=False):
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        
        p, f_line, s_line, ema200, vol_ma, vol_nu, atr, adx = bereken_indicatoren(df, ticker, s, t)
        
        # Filter op laatste jaar (252 dagen)
        p, f_line, s_line = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        ema200, vol_ma, vol_nu = ema200.iloc[-252:], vol_ma.iloc[-252:], vol_nu.iloc[-252:]
        atr, adx = atr.iloc[-252:], adx.iloc[-252:]
            
        saldo_delta = 0
        pos, instap, high_p, sl = False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p)):
            curr_p = p.iloc[i]
            if not pos:
                if f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]:
                    if adx.iloc[i] > 15 and vol_nu.iloc[i] > (vol_ma.iloc[i] * 0.6):
                        if not use_trend_filter or curr_p > ema200.iloc[i]:
                            instap, high_p, pos = curr_p, curr_p, True
                            sl = curr_p - (2 * atr.iloc[i])
                            saldo_delta -= kosten
            else:
                high_p = max(high_p, curr_p)
                sl = max(sl, high_p - (2 * atr.iloc[i]))
                if curr_p < sl or (f_line.iloc[i] < s_line.iloc[i]):
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
    # We starten de berekening vanaf 0 verandering t.o.v. de 100.000 basis
    r = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    for t in tickers:
        r["T"]  += bereken_bt_elite(t, inzet, 50, 200, use_trend_filter=True)
        r["S"]  += bereken_bt_elite(t, inzet, 20, 50, use_trend_filter=True)
        r["HT"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=True)
        r["HS"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=False)

    rapport = [
        "📊 *VOLLEDIG ELITE RAPPORT (LAATSTE JAAR)*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200 SMA):* €{100000 + r['T']:,.0f}",
        f"⚡ *Snel (20/50 SMA):* €{100000 + r['S']:,.0f}",
        f"🚀 *Hyper Trend (9/21 EMA200):* €{100000 + r['HT']:,.0f}",
        f"🔥 *Hyper Scalp (9/21 EMA):* €{100000 + r['HS']:,.0f}",
        "",
        "⚙️ *FILTERS:* ADX > 15 | Vol > 60% | Trailing Stop ATR"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
