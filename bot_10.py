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
    except Exception as e:
        print(f"Telegram Fout: {e}")

def bereken_bt_elite(ticker, inzet, s, t, use_trend_filter=False):
    """De volledige Elite-logica inclusief ATR Trailing Stop, ADX 15 en Volume 60%."""
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        
        # Data extractie & Multi-Index Fix
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # 1. Indicatoren
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        # ATR & ADX Berekening
        tr = pd.concat([h - l, abs(h - p.shift()), abs(l - p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        up = h.diff().clip(lower=0)
        down = (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
        vol_ma = v.rolling(window=20).mean()

        # Filter op laatste jaar (252 handelsdagen)
        p, f_line, s_line = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        ema200, vol_ma, v_nu = ema200.iloc[-252:], vol_ma.iloc[-252:], v.iloc[-252:]
        atr, adx = atr.iloc[-252:], adx.iloc[-252:]
            
        saldo_delta = 0
        pos, instap, high_p, sl = False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p)):
            curr_p = p.iloc[i]
            if not pos:
                # KOOP CONDITIES (Elite Filters)
                crossover = f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]
                if crossover and adx.iloc[i] > 15 and v_nu.iloc[i] > (vol_ma.iloc[i] * 0.6):
                    if not use_trend_filter or curr_p > ema200.iloc[i]:
                        instap, high_p, sl, pos = curr_p, curr_p, curr_p - (2 * atr.iloc[i]), True
                        saldo_delta -= kosten
            else:
                # TRAILING STOP & EXIT
                high_p = max(high_p, curr_p)
                sl = max(sl, high_p - (2 * atr.iloc[i]))
                if curr_p < sl or f_line.iloc[i] < s_line.iloc[i]:
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
    # Initialiseer alle resultaten op 0
    r = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0}

    for t in tickers:
        r["T"]  += bereken_bt_elite(t, inzet, 50, 200, use_trend_filter=True)
        r["S"]  += bereken_bt_elite(t, inzet, 20, 50, use_trend_filter=True)
        r["HT"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=True)
        r["HS"] += bereken_bt_elite(t, inzet, 9, 21, use_trend_filter=False)

    rapport = [
        "🏆 *OFFICIEEL ELITE RAPPORT (v15)*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200 SMA):* €{100000 + r['T']:,.0f}",
        f"⚡ *Snel (20/50 SMA):* €{100000 + r['S']:,.0f}",
        f"🚀 *Hyper Trend (9/21 EMA):* €{100000 + r['HT']:,.0f}",
        f"🔥 *Hyper Scalp (9/21 EMA):* €{100000 + r['HS']:,.0f}",
        "",
        "🛠️ *CONFIGURATIE:*",
        "• Periode: Laatste 12 maanden",
        "• Filters: ADX > 15 | Vol > 60%",
        "• Risico: ATR Trailing Stop Loss",
        f"• Tickers geanalyseerd: {len(tickers)}"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
