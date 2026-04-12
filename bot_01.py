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
        
        # Consistent data extractie (zoals de allereerste versie)
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # INDICATOREN
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        ma5 = p.rolling(window=5).mean()
        
        # RSI2 voor Mean Reversion
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        # ATR voor Stop Loss
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # BACKTEST
        p_bt, f_bt, s_bt, e200_bt, r2_bt, ma5_bt, atr_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:], ema200.iloc[-252:], rsi2.iloc[-252:], ma5.iloc[-252:], atr.iloc[-252:]
        
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                # KOOP LOGICA
                if use_mean_rev:
                    buy = r2_bt.iloc[i] < 25 # Geen extra filters voor MR nu
                else:
                    buy = f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]
                    if use_trend_filter: buy = buy and cp > e200_bt.iloc[i]
                
                if buy:
                    instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                    profit -= kosten
            else:
                # VERKOOP LOGICA
                high_p = max(high_p, cp)
                sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                
                if use_mean_rev:
                    sell = cp > ma5_bt.iloc[i] or cp < sl_val
                else:
                    sell = f_bt.iloc[i] < s_bt.iloc[i] or cp < sl_val
                
                if sell:
                    w = (inzet * (cp / instap) - inzet) - kosten
                    if w > 0: w *= 0.9
                    profit += w
                    pos = False

        signaal = None
        if use_mean_rev and rsi2.iloc[-1] < 25:
            signaal = f"📉 *MR* | €{p.iloc[-1]:.2f}"
        elif not use_mean_rev and f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
            signaal = f"🟢 *KOOP* | €{p.iloc[-1]:.2f}"
            
        return profit, signaal
    except: return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}
    sig = {"T":[], "S":[], "HT":[], "HS":[], "MR":[]}

    for t in tickers:
        # Hier staan de exacte succesvolle parameters van je eerste goede run
        params = [
            ("T", (50, 200, True, False, False)),
            ("S", (20, 50, True, False, False)),
            ("HT", (9, 21, True, True, False)), # EMA 9/21 met Trend Filter
            ("HS", (9, 21, False, True, False)), # EMA 9/21 zonder Trend Filter
            ("MR", (0, 0, False, False, True)) # Mean Reversion
        ]
        for k, prm in params:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2], is_hyper=prm[3], use_mean_rev=prm[4])
            res[k] += p
            if s: sig[k].append(f"• `{t}`: {s}")

    rapport = [
        f"📊 *{label} {naam_sector}* - {nu}",
        "----------------------------------",
        f"🐢 *Traag:* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel:* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        f"📉 *Mean Reversion:* €{100000 + res['MR']:,.0f}"
    ]
    stuur_telegram("\n".join(rapport))

def main():
    sectoren = {"01":"Hoogland"}
    for nr, naam in sectoren.items():
        try: voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    main()
