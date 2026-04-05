import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False):
    """Berekent zowel de historische winst als het signaal voor vandaag."""
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0, None
        
        # Data extractie
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # Indicatoren
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        vol_ma = v.rolling(window=20).mean()
        
        # ATR & ADX
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

        # --- DEEL 1: BACKTEST (Laatste jaar) ---
        p_bt, f_bt, s_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        e_bt, v_bt, v_ma_bt = ema200.iloc[-252:], v.iloc[-252:], vol_ma.iloc[-252:]
        atr_bt, adx_bt = atr.iloc[-252:], adx.iloc[-252:]
        
        profit, pos, instap, high_p, sl = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = p_bt.iloc[i]
            if not pos:
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6):
                        if not use_trend_filter or cp > e_bt.iloc[i]:
                            instap, high_p, sl, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                            profit -= kosten
            else:
                high_p = max(high_p, cp)
                sl = max(sl, high_p - (2 * atr_bt.iloc[i]))
                if cp < sl or f_bt.iloc[i] < s_bt.iloc[i]:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False

        # --- DEEL 2: SIGNAAL VANDAAG ---
        signaal = None
        if f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
            if adx.iloc[-1] > 15 and v.iloc[-1] > (vol_ma.iloc[-1] * 0.6):
                if not use_trend_filter or p.iloc[-1] > ema200.iloc[-1]:
                    signaal = f"🟢 *KOOP* | €{p.iloc[-1]:.2f}"
        elif f_line.iloc[-1] < s_line.iloc[-1] and f_line.iloc[-2] >= s_line.iloc[-2]:
            signaal = f"🔴 *VERKOOP* | €{p.iloc[-1]:.2f}"

        return profit, signaal
    except:
        return 0, None

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open('tickers_01.txt', 'r') as f:
        tickers = list(set([t.strip().upper() for t in f.read().replace('\n', ',').replace('$', '').split(',') if t.strip()]))

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": []}

    for t in tickers:
        # Berekening per bot
        for key, params in [("T", (50,200,True)), ("S", (20,50,True)), ("HT", (9,21,True)), ("HS", (9,21,False))]:
            p, s = bereken_alles(t, inzet, params[0], params[1], params[2])
            res[key] += p
            if s: sig[key].append(f"• `{t}`: {s}")

    def get_sig(lst): return "\n".join(lst) if lst else "Geen actie"

    rapport = [
        "📊 *ELITE TRADING MASTER RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend (9/21+200):* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp (9/21):* €{100000 + res['HS']:,.0f}",
        "",
        "🛡️ *SIGNALEN TRAAG:*", get_sig(sig["T"]),
        "",
        "🎯 *SIGNALEN SNEL:*", get_sig(sig["S"]),
        "",
        "📈 *SIGNALEN HYPER TREND:*", get_sig(sig["HT"]),
        "",
        "⚡ *SIGNALEN HYPER SCALP:*", get_sig(sig["HS"]),
        "",
        "⚙️ _ADX > 15 | Vol > 60% | Trailing Stop actief_"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
