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
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"Telegram Fout: {e}")

def bereken_bt_elite(ticker, inzet, s, t, use_trend_filter=False):
    """Backtest met alle filters: ADX 15, Vol 60%, ATR Trailing Stop, 1 Jaar."""
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        
        # Data extractie
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # 1. Gemiddelden
        f_line = p.ewm(span=s, adjust=False).mean() if s < 20 else p.rolling(window=s).mean()
        s_line = p.ewm(span=t, adjust=False).mean() if t < 50 else p.rolling(window=t).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        # 2. Volume Filter (60% van 20-daags MA)
        vol_ma = v.rolling(window=20).mean()
        
        # 3. ATR (Volatiliteit voor Stop Loss)
        tr = pd.concat([h - l, abs(h - p.shift()), abs(l - p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # 4. ADX (Trendsterkte drempel 15)
        up = h.diff().clip(lower=0)
        down = (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        
        # Filter op laatste jaar (252 dagen)
        p, f_line, s_line = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        ema200, vol_ma, v_nu = ema200.iloc[-252:], vol_ma.iloc[-252:], v.iloc[-252:]
        atr, adx = atr.iloc[-252:], adx.iloc[-252:]
            
        saldo_delta = 0
        pos, instap, high_p, sl = False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p)):
            curr_p = p.iloc[i]
            if not pos:
                # KOOP CONDITIES
                crossover = f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]
                voldoet_adx = adx.iloc[i] > 15
                voldoet_vol = v_nu.iloc[i] > (vol_ma.iloc[i] * 0.6)
                voldoet_trend = curr_p > ema200.iloc[i] if use_trend_filter else True
                
                if crossover and voldoet_adx and voldoet_vol and voldoet_trend:
                    instap, high_p, pos = curr_p, curr_p, True
                    sl = curr_p - (2 * atr.iloc[i])
                    saldo_delta -= kosten
            else:
                # TRAILING STOP LOGICA
                high_p = max(high_p, curr_p)
                sl = max(sl, high_p - (2 * atr.iloc[i]))
                
                # VERKOOP (Stop loss OF cross-back)
                if curr_p < sl or (f_line.iloc[i] < s_line.iloc[i]):
                    saldo_delta += (inzet * (curr_p / instap) - inzet) - kosten
                    pos = False
        return saldo_delta
    except Exception as e:
        print(f"Fout bij {ticker}: {e}")
        return 0

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'):
        stuur_telegram("❌ Fout: `tickers_01.txt` niet gevonden.")
        return

    with open('tickers_01.txt', 'r') as f:
        raw = f.read().replace('\n', ',').replace('$', '')
        tickers = list(set([t.strip().upper() for t in raw.split(',') if t.strip()]))

    inzet = 2500.0
    r = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    try:
        for t in tickers:
            r["T"]  += bereken_bt_elite(t, inzet, 50, 200, use_trend_filter=True)
            r["S"]  += bereken_bt_elite(t, inzet, 20, 50, use_trend_filter=True)
            r["HT"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=True)
            r["HS"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=False)

        rapport = [
            "📊 *VOLLEDIG ELITE RAPPORT (v14)*",
            f"_{nu}_",
            "----------------------------------",
            f"🐢 *Traag (50/200):* €{100000 + r['T']:,.0f}",
            f"⚡ *Snel (20/50):* €{100000 + r['S']:,.0f}",
            f"🚀 *Hyper Trend (9/21 + 200):* €{100000 + r['HT']:,.0f}",
            f"🔥 *Hyper Scalp (9/21):* €{100000 + r['HS']:,.0f}",
            "",
            "⚙️ *FILTERS:* ADX > 15 | Vol > 60% | Trailing Stop ATR | 1 Jaar"
        ]
        stuur_telegram("\n".join(rapport))
    except Exception as e:
        stuur_telegram(f"❌ Er is een fout opgetreden in de main loop: {e}")

if __name__ == "__main__":
    main()
