import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=10)
    except: pass

def analyseer_ticker(ticker, inzet):
    """Voert alle 4 de strategieën uit voor één specifieke ticker."""
    resultaten = {}
    signalen = {}
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return None
        
        p = df['Close'].dropna().astype(float)
        v, h, l = df['Volume'], df['High'], df['Low']
        
        # Bereken basis indicatoren eenmalig
        ema200 = p.ewm(span=200, adjust=False).mean()
        vol_ma = v.rolling(window=20).mean()
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr_s = tr.rolling(14).mean()
        
        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        # Strategie loop
        for k, (s, t, filt) in [("T",(50,200,True)), ("S",(20,50,True)), ("HT",(9,21,True)), ("HS",(9,21,False))]:
            f_l = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
            s_l = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
            
            # Snelle Backtest (Vectorized)
            p_bt = p.iloc[-252:]
            # (Voor de snelheid houden we hier de loop, maar we beperken de data)
            prof, pos, instap, sl_v = 0, False, 0, 0
            kos = 15.0 + (inzet * 0.0035)
            
            for i in range(1, len(p_bt)):
                cp = p_bt.iloc[i]
                if not pos:
                    if f_l.iloc[i-252+i] > s_l.iloc[i-252+i] and f_l.iloc[i-252+i-1] <= s_l.iloc[i-252+i-1]:
                        instap, sl_v, pos = cp, cp - (2 * atr_s.iloc[i-252+i]), True
                        prof -= kos
                elif cp < sl_v or f_l.iloc[i-252+i] < s_l.iloc[i-252+i]:
                    prof += (inzet * (cp / instap) - inzet) - kos
                    pos = False
            
            resultaten[k] = prof
            
            # Signaal vandaag
            sig = None
            cp, catr, crsi = p.iloc[-1], atr_s.iloc[-1], rsi.iloc[-1]
            if f_l.iloc[-1] > s_l.iloc[-1] and f_l.iloc[-2] <= s_l.iloc[-2]:
                if not filt or cp > ema200.iloc[-1]:
                    ap = (catr/cp)*100
                    y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    sig = f"• `{ticker}`: 🟢 €{cp:.2f} | ⚡{ap:.1f}% | 🧠{crsi:.0f} | 🛡️€{cp-(2*catr):.2f} | {y_l}"
            elif f_l.iloc[-1] < s_l.iloc[-1] and f_l.iloc[-2] >= s_l.iloc[-2]:
                sig = f"• `{ticker}`: 🔴 €{cp:.2f} | {y_l}"
            
            signalen[k] = sig
            
        return {"res": resultaten, "sig": signalen}
    except: return None

def main():
    nu = datetime.now().strftime("%H:%M")
    with open('tickers_01.txt', 'r') as f:
        tickers = list(set([t.strip().upper() for t in f.read().replace('\n',',').split(',') if t.strip()]))

    inzet = 2500.0
    tot_res = {"T":0, "S":0, "HT":0, "HS":0}
    tot_sig = {"T":[], "S":[], "HT":[], "HS":[]}

    # TURBO: Gebruik ThreadPool voor parallelle verwerking
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyseer_ticker, t, inzet) for t in tickers]
        for f in futures:
            r = f.result()
            if r:
                for k in tot_res: tot_res[k] += r['res'][k]
                for k in tot_sig: 
                    if r['sig'][k]: tot_sig[k].append(r['sig'][k])

    def fmt(l): return "\n".join(l) if l else "Geen actie"
    
    rapport = [
        f"🚀 *HOOGLAND TURBO v23* ({nu})",
        "----------------------------------",
        f"🐢 T: €{100000+tot_res['T']:,.0f} | ⚡ S: €{100000+tot_res['S']:,.0f}",
        f"🚀 HT: €{100000+tot_res['HT']:,.0f} | 🔥 HS: €{100000+tot_res['HS']:,.0f}",
        "", "*SIGNALLY:*",
        "📈 *Trend:*", fmt(tot_sig["HT"]),
        "🎯 *Snel:*", fmt(tot_sig["S"])
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
