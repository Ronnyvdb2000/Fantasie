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
        
        # Harde fix voor data kolommen
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].ffill().astype(float)
            h = df['High'][ticker].ffill().astype(float)
            l = df['Low'][ticker].ffill().astype(float)
        else:
            p, h, l = df['Close'].ffill().astype(float), df['High'].ffill().astype(float), df['Low'].ffill().astype(float)

        # INDICATOREN
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        ma5 = p.rolling(window=5).mean()
        
        # Eenvoudige RSI2 berekening die ALTIJD werkt
        diff = p.diff()
        up = diff.clip(lower=0)
        down = -1 * diff.clip(upper=0)
        ma_up = up.rolling(window=2).mean()
        ma_down = down.rolling(window=2).mean()
        rs = ma_up / (ma_down + 1e-10)
        rsi2 = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # BACKTEST DATA
        p_bt = p.iloc[-252:]
        f_bt = f_line.iloc[-252:]
        s_bt = s_line.iloc[-252:]
        e_bt = ema200.iloc[-252:]
        r2_bt = rsi2.iloc[-252:]
        m5_bt = ma5.iloc[-252:]
        a_bt = atr.iloc[-252:]
        
        profit, pos, instap, sl_val = 0, False, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(2, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                # MEAN REVERSION: RSI2 < 25 (zeer ruim)
                if use_mean_rev:
                    if r2_bt.iloc[i] < 25:
                        instap, sl_val, pos = cp, cp - (2 * a_bt.iloc[i]), True
                        profit -= kosten
                # TREND/HYPER
                elif f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if not use_trend_filter or cp > e_bt.iloc[i]:
                        instap, sl_val, pos = cp, cp - (2 * a_bt.iloc[i]), True
                        profit -= kosten
            else:
                # EXIT
                if use_mean_rev:
                    sell = cp > m5_bt.iloc[i] or cp < sl_val
                else:
                    sell = f_bt.iloc[i] < s_bt.iloc[i] or cp < sl_val
                
                if sell:
                    w = (inzet * (cp / instap) - inzet) - kosten
                    if w > 0: w *= 0.9
                    profit += w
                    pos = False

        # Signaal vandaag
        signaal = None
        if use_mean_rev and rsi2.iloc[-1] < 25:
            signaal = f"📉 *MR DIP* | €{p.iloc[-1]:.2f} (RSI2: {rsi2.iloc[-1]:.1f})"
        elif not use_mean_rev and f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
            signaal = f"🟢 *TREND* | €{p.iloc[-1]:.2f}"
            
        return profit, signaal
    except: return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}
    sig = {"MR": []}

    for t in tickers:
        for k, prm in [("T",(50,200,True,False,False)), ("S",(20,50,True,False,False)), ("HT",(9,21,True,True,False)), ("HS",(9,21,False,True,False)), ("MR",(0,0,False,False,True))]:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2], is_hyper=prm[3], use_mean_rev=prm[4])
            res[k] += p
            if k == "MR" and s: sig["MR"].append(f"• `{t}`: {s}")

    rapport = [
        f"📊 *{label} {naam_sector}* - {nu}",
        "----------------------------------",
        f"🐢 *Traag:* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel:* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        f"📉 *Mean Reversion:* €{100000 + res['MR']:,.0f}",
        "", "📉 *DIP SIGNALEN:*", "\n".join(sig["MR"]) or "Geen actie"
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
