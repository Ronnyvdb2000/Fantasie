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

# --- STRATEGIE 1-4: JOUW ORIGINELE VISIE (ONGESCHONDEN) ---
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

        # Indicatoren
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        vol_ma = v.rolling(window=20).mean()
        
        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        # ATR & ADX
        tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
        atr_series = tr.rolling(14).mean()
        up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * (up.rolling(14).sum() / (tr14 + 1e-10))
        minus_di = 100 * (down.rolling(14).sum() / (tr14 + 1e-10))
        adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

        # BACKTEST (Laatste 252 dagen)
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

        # SIGNAAL VANDAAG
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

# --- STRATEGIE 5: NIEUWE MEAN REVERSION ALPHA (HOOG RENDEMENT) ---
def bereken_mean_reversion_alpha(ticker, inzet):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 200: return 0, None
        p = df['Close'][ticker] if isinstance(df.columns, pd.MultiIndex) else df['Close']
        
        ma20 = p.rolling(window=20).mean()
        std20 = p.rolling(window=20).std()
        lower_band = ma20 - (2 * std20)
        upper_band = ma20 + (2 * std20)
        ema200 = p.ewm(span=200, adjust=False).mean()

        # Backtest
        p_bt = p.iloc[-252:]
        profit, pos, instap = 0, False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(20, len(p_bt)):
            cp = p_bt.iloc[i]
            # KOOP: Dip onder Bollinger Band maar in stijgende EMA200 trend
            if not pos:
                if cp < lower_band.iloc[-252+i] and cp > ema200.iloc[-252+i]:
                    instap, pos = cp, True
                    profit -= kosten
            # VERKOOP: Terugkeer naar Upper Band (overshoot)
            else:
                if cp > upper_band.iloc[-252+i] or cp > instap * 1.07:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False
        
        signaal = None
        if p.iloc[-1] < lower_band.iloc[-1] and p.iloc[-1] > ema200.iloc[-1]:
            signaal = f"💎 MEAN REVERSION | €{p.iloc[-1]:.2f} (Target: €{ma20.iloc[-1]:.2f})"

        return profit, signaal
    except:
        return 0, None

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    # Zorg dat 'tickers_04xx.txt' in de juiste map staat
    with open('tickers_06xx.txt', 'r') as f:
        tickers = list(set([t.strip().upper() for t in f.read().replace('\n', ',').replace('$', '').split(',') if t.strip()]))

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MRA": 0}
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MRA": []}

    for t in tickers:
        # Originele 4
        for k, prm in [("T", (50,200,True)), ("S", (20,50,True)), ("HT", (9,21,True)), ("HS", (9,21,False))]:
            p, s = bereken_alles(t, inzet, prm[0], prm[1], prm[2])
            res[k] += p
            if s: sig[k].append(f"• `{t}`: {s}")
        
        # Nieuwe Alpha
        p_mra, s_mra = bereken_mean_reversion_alpha(t, inzet)
        res["MRA"] += p_mra
        if s_mra: sig["MRA"].append(f"• `{t}`: {s_mra}")

    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"

    rapport = [
        "📊 *Benelux xx *",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        f"💎 *Mean Rev Alpha:* €{100000 + res['MRA']:,.0f}",
        "----------------------------------",
        "*SIGNALEN TRAAG:*", get_s(sig["T"]),
        "\n*SIGNALEN SNEL:*", get_s(sig["S"]),
        "\n*SIGNALEN HYPER:*", get_s(sig["HT"]),
        "\n*SIGNALEN MEAN REV:*", get_s(sig["MRA"])
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
