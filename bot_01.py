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
            requests.post(url, data={"chat_id": CHAT_ID, "text": part, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=20)
            time.sleep(0.5)
        except: pass

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False, is_hyper=False, use_mean_rev=False):
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

        # Indicatoren
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        ma5 = p.rolling(window=5).mean()
        vol_ma = v.rolling(window=20).mean()
        
        # Volume Price Trend (VPT) - Krachtig volume filter
        vpt = (v * p.pct_change()).cumsum()
        vpt_stijgend = vpt.diff() > 0

        # RSI-14 en RSI-2 (voor Mean Reversion)
        def calc_rsi(ser, window):
            delta = ser.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            return 100 - (100 / (1 + (gain / (loss + 1e-10))))

        rsi14 = calc_rsi(p, 14)
        rsi2 = calc_rsi(p, 2)

        # CRSI (voor Hyper)
        if is_hyper:
            rsi3 = calc_rsi(p, 3)
            streak = pd.Series(0, index=p.index)
            for i in range(1, len(p)):
                if p.iloc[i] > p.iloc[i-1]: streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] > 0 else 1
                elif p.iloc[i] < p.iloc[i-1]: streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] < 0 else -1
            streak_rsi = calc_rsi(streak, 2)
            p_rank = p.diff().rolling(100).apply(lambda x: (x < x.iloc[-1]).sum() / 99.0 * 100, raw=False)
            rsi_val = (rsi3 + streak_rsi + p_rank) / 3
        else:
            rsi_val = rsi14

        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        adx = (100 * abs( (h.diff().clip(lower=0)).rolling(14).sum() - (-l.diff().clip(lower=0)).rolling(14).sum() ) / (tr.rolling(14).sum() + 1e-10)).rolling(14).mean()

        # BACKTEST
        p_bt, f_bt, s_bt, e_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:], ema200.iloc[-252:]
        r2_bt, ma5_bt = rsi2.iloc[-252:], ma5.iloc[-252:]
        vpt_ok = vpt_stijgend.iloc[-252:]
        atr_bt = atr.iloc[-252:]
        
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                # Koop condities
                if use_mean_rev:
                    buy = r2_bt.iloc[i] < 10 and cp > e_bt.iloc[i] # Dip in uptrend
                else:
                    buy = f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]
                    if is_hyper: buy = buy and vpt_ok.iloc[i] # Extra Volume filter
                
                if buy:
                    instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                    profit -= kosten
            else:
                high_p = max(high_p, cp)
                sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                # Verkoop condities
                sell = cp < sl_val or (use_mean_rev and cp > ma5_bt.iloc[i]) or (not use_mean_rev and f_bt.iloc[i] < s_bt.iloc[i])
                if sell:
                    w = (inzet * (cp / instap) - inzet) - kosten
                    if w > 0: w *= 0.9
                    profit += w
                    pos = False

        # SIGNAAL VANDAAG
        signaal = None
        cp, crsi2, crsi_main = p.iloc[-1], rsi2.iloc[-1], rsi_val.iloc[-1]
        y_l = f"[Chart](https://finance.yahoo.com/quote/{ticker})"
        
        if use_mean_rev:
            if crsi2 < 10 and cp > ema200.iloc[-1]:
                signaal = f"🔵 *MEAN REV* | €{cp:.2f} | RSI2: {crsi2:.1f} | 🛡️ SL: €{cp-(2*atr.iloc[-1]):.2f} | {y_l}"
        else:
            if f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
                if vpt_stijgend.iloc[-1] or not is_hyper:
                    signaal = f"🟢 *KOOP* | €{cp:.2f} | RSI: {crsi_main:.1f} | 🛡️ SL: €{cp-(2*atr.iloc[-1]):.2f} | {y_l}"
        
        return profit, signaal
    except: return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MR": []}

    for t in tickers:
        for k, prm in [("T",(50,200,True,False,False)), ("S",(20,50,True,False,False)), ("HT",(9,21,True,True,False)), ("HS",(9,21,False,True,False)), ("MR",(0,0,True,False,True))]:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2], is_hyper=prm[3], use_mean_rev=prm[4])
            res[k] += p
            if s: sig[k].append(f"• `{t}`: {s}")

    rapport = [
        f"📊 *{label} {naam_sector} RAPPORT*",
        f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend (VPT):* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp (VPT):* €{100000 + res['HS']:,.0f}",
        f"📉 *Mean Reversion (RSI2):* €{100000 + res['MR']:,.0f}",
        "", "🛡️ *SIGNAAL TRAAG:*", "\n".join(sig["T"]) or "Geen actie",
        "", "🎯 *SIGNAAL SNEL:*", "\n".join(sig["S"]) or "Geen actie",
        "", "📈 *SIGNAAL HYPER TREND:*", "\n".join(sig["HT"]) or "Geen actie",
        "", "⚡ *SIGNAAL HYPER SCALP:*", "\n".join(sig["HS"]) or "Geen actie",
        "", "📉 *SIGNAAL MEAN REVERSION:*", "\n".join(sig["MR"]) or "Geen actie"
    ]
    stuur_telegram("\n".join(rapport))

def main():
    sectoren = {"01":"Hoogland"}
    for nr, naam in sectoren.items():
        try: voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        except: pass
        time.sleep(2)

if __name__ == "__main__":
    main()
