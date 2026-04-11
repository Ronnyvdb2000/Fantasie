import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=20)
        time.sleep(1) 
    except: pass

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False, is_hyper=False):
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
        
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        # CRSI Logica toevoeging
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

        p_bt, f_bt, s_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        e_bt, v_bt, v_ma_bt = ema200.iloc[-252:], v.iloc[-252:], vol_ma.iloc[-252:]
        atr_bt = atr_series.iloc[-252:]
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = p_bt.iloc[i]
            if not pos:
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if adx.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6):
                        if not use_trend_filter or cp > e_bt.iloc[i]:
                            instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                            profit -= kosten
            else:
                high_p = max(high_p, cp)
                sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                if cp < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False

        signaal = None
        cp, catr, crsi = p.iloc[-1], atr_series.iloc[-1], rsi_val.iloc[-1]
        ap = (catr / cp) * 100
        y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
        label_rsi = "💎 CRSI" if is_hyper else "📊 RSI"

        if f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
            if adx.iloc[-1] > 15 and v.iloc[-1] > (vol_ma.iloc[-1] * 0.6):
                if not use_trend_filter or cp > ema200.iloc[-1]:
                    signaal = f"🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} ({ap:.1f}%) | {label_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"
        elif f_line.iloc[-1] < s_line.iloc[-1] and f_line.iloc[-2] >= s_line.iloc[-2]:
            signaal = f"🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} ({ap:.1f}%) | {label_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"

        return profit, signaal
    except: return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return ""
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": []}

    for t in tickers:
        # Hier wordt bepaald of we RSI of CRSI gebruiken
        for k, prm in [("T",(50,200,True,False)), ("S",(20,50,True,False)), ("HT",(9,21,True,True)), ("HS",(9,21,False,True))]:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2], is_hyper=prm[3])
            res[k] += p
            if s: sig[k].append(f"• `{t}`: {s}")

    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"

    rapport_lijst = [
        f"📊 *{label} {naam_sector} RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        "",
        "🛡️ *SIGNALEN TRAAG (RSI):*", get_s(sig["T"]),
        "",
        "🎯 *SIGNALEN SNEL (RSI):*", get_s(sig["S"]),
        "",
        "📈 *SIGNALEN HYPER TREND (CRSI):*", get_s(sig["HT"]),
        "",
        "⚡ *SIGNALEN HYPER SCALP (CRSI):*", get_s(sig["HS"]),
        "",
        "💡 _ATR %: <2% laag, >5% hoog. RSI: >70 overbought, <30 oversold. CRSI: >90 overbought, <10 oversold_"
    ]
    rapport_tekst = "\n".join(rapport_lijst)
    stuur_telegram(rapport_tekst)
    return rapport_tekst + "\n\n"

def main():
    sectoren = {"01":"Hoogland","02":"Macrotrends","03":"Beursbrink","04":"Benelux","05":"Parijs","06":"Power & AI","07":"Metalen","08":"Defensie","09":"Varia"}
    for nr, naam in sectoren.items():
        try: voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        except: pass
        time.sleep(2)

if __name__ == "__main__":
    main()
