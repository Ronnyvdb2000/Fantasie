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
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True})
    except:
        pass

# --- STRATEGIE 1-4: JOUW ORIGINELE VISIE (VOLLEDIG ONGEWIJZIGD) ---
def bereken_alles(ticker, inzet, s, t, use_trend_filter=False):
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
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

        p_bt, f_bt, s_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        e_bt, v_bt, v_ma_bt = ema200.iloc[-252:], v.iloc[-252:], vol_ma.iloc[-252:]
        atr_bt, adx_bt = atr_series.iloc[-252:], adx.iloc[-252:]
        
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_bt)):
            cp = p_bt.iloc[i]
            if not pos:
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6):
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
        curr_p = p.iloc[-1]
        if f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
            if adx.iloc[-1] > 15 and v.iloc[-1] > (vol_ma.iloc[-1] * 0.6):
                if not use_trend_filter or curr_p > ema200.iloc[-1]:
                    signaal = f"🟢 KOOP | €{curr_p:.2f} | RSI: {rsi.iloc[-1]:.0f}"
        elif f_line.iloc[-1] < s_line.iloc[-1] and f_line.iloc[-2] >= s_line.iloc[-2]:
            signaal = f"🔴 VERKOOP | €{curr_p:.2f}"

        return profit, signaal
    except:
        return 0, None

# --- STRATEGIE 5: POWER REVERSION ALPHA (DE AANGEPASTE VERSIE) ---
def bereken_mean_reversion_alpha(ticker, inzet):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 200: return 0, None
        
        p = df['Close'][ticker].dropna().astype(float) if isinstance(df.columns, pd.MultiIndex) else df['Close']
        
        # Indicatoren
        ma20 = p.rolling(window=20).mean()
        std20 = p.rolling(window=20).std()
        upper_band = ma20 + (2.0 * std20)
        lower_band = ma20 - (2.2 * std20) 
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        # Filter: EMA200 moet stijgend zijn over de laatste 5 dagen
        ema200_stijgend = ema200.diff(5) > 0

        # RSI 2 (Agressieve entry timing)
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
        rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        p_bt = p.iloc[-252:]
        profit, pos, instap, max_p = 0, False, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(20, len(p_bt)):
            cp = p_bt.iloc[i]
            idx = i + (len(p) - 252)
            
            if not pos:
                # ENTRY: Prijs onder band + RSI2 extreem laag + Kwaliteit (stijgende EMA200)
                if cp < lower_band.iloc[idx] and rsi2.iloc[idx] < 10 and ema200_stijgend.iloc[idx]:
                    instap, max_p, pos = cp, cp, True
                    profit -= kosten
            else:
                max_p = max(max_p, cp)
                # EXIT: 8% Target of Upper Band of 3% Trailing Stop
                if cp > upper_band.iloc[idx] or cp > (instap * 1.08) or cp < (max_p * 0.97):
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False
        
        signaal = None
        if p.iloc[-1] < lower_band.iloc[-1] and rsi2.iloc[-1] < 15 and ema200_stijgend.iloc[-1]:
            signaal = f"🚀 POWER DIP | €{p.iloc[-1]:.2f} (RSI2: {rsi2.iloc[-1]:.0f})"

        return profit, signaal
    except:
        return 0, None

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open('tickers_06xx.txt', 'r') as f:
        tickers = list(set([t.strip().upper() for t in f.read().replace('\n', ',').replace('$', '').split(',') if t.strip()]))

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MRA": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MRA": []}

    for t in tickers:
        # De 4 ongewijzigde strategieën
        for k, prm in [("T", (50,200,True)), ("S", (20,50,True)), ("HT", (9,21,True)), ("HS", (9,21,False))]:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2])
            res[k] += p
            if s: sig[k].append(f"• `{t}`: {s}")
        
        # De aangepaste 5de strategie
        p_mra, s_mra = bereken_mean_reversion_alpha(t, inzet)
        res["MRA"] += p_mra
        if s_mra: sig["MRA"].append(f"• `{t}`: {s_mra}")

    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"

    rapport = [
        "📊 *Power & AI MULTI-STRAT REPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        f"💎 *Power Mean Rev:* €{100000 + res['MRA']:,.0f}",
        "----------------------------------",
        "*SIGNALEN TRAAG:*", get_s(sig["T"]),
        "\n*SIGNALEN SNEL:*", get_s(sig["S"]),
        "\n*SIGNALEN POWER REV:*", get_s(sig["MRA"])
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
