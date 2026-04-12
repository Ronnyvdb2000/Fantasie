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
        
        # Kolom extractie
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].dropna().astype(float)
            v = df['Volume'][ticker].dropna().astype(float)
            h = df['High'][ticker].dropna().astype(float)
            l = df['Low'][ticker].dropna().astype(float)
        else:
            p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

        # Gemiddeldes
        f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
        s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        ma5 = p.rolling(window=5).mean()
        
        # Volume Price Trend (VPT)
        vpt = (v * p.pct_change()).cumsum()
        vpt_stijgend = vpt.diff() > 0

        # RSI Berekening
        def calc_rsi(ser, window):
            delta = ser.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=window-1, adjust=False).mean()
            ema_down = down.ewm(com=window-1, adjust=False).mean()
            rs = ema_up / (ema_down + 1e-10)
            return 100 - (100 / (1 + rs))

        rsi2 = calc_rsi(p, 2)
        
        # CRSI Logica (Hyper)
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
            rsi_val = calc_rsi(p, 14)

        atr = (h - l).rolling(14).mean()

        # --- BACKTEST ---
        p_bt, f_bt, s_bt, e200_bt = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:], ema200.iloc[-252:]
        r2_bt, ma5_bt, vpt_ok = rsi2.iloc[-252:], ma5.iloc[-252:], vpt_stijgend.iloc[-252:]
        atr_bt = atr.iloc[-252:]
        
        profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(2, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                if use_mean_rev:
                    # GEEN VPT filter voor MR, alleen RSI2 < 25
                    if r2_bt.iloc[i] < 25 and cp > e200_bt.iloc[i]:
                        instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                        profit -= kosten
                else:
                    if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                        if not is_hyper or vpt_ok.iloc[i]:
                            instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr_bt.iloc[i]), True
                            profit -= kosten
            else:
                high_p = max(high_p, cp)
                sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))
                sell = cp > ma5_bt.iloc[i] if use_mean_rev else f_bt.iloc[i] < s_bt.iloc[i]
                if sell or cp < sl_val:
                    w = (inzet * (cp / instap) - inzet) - kosten
                    if w > 0: w *= 0.9
                    profit += w
                    pos = False

        # --- SIGNAAL VANDAAG ---
        signaal = None
        curr_p, curr_r2 = p.iloc[-1], rsi2.iloc[-1]
        y_l = f"[Chart](https://finance.yahoo.com/quote/{ticker})"
        
        if use_mean_rev:
            if curr_r2 < 25 and curr_p > ema200.iloc[-1]:
                signaal = f"📉 *DIP KOOP* | €{curr_p:.2f} | RSI2: {curr_r2:.1f} | {y_l}"
        else:
            if f_line.iloc[-1] > s_line.iloc[-1] and f_line.iloc[-2] <= s_line.iloc[-2]:
                if not is_hyper or vpt_stijgend.iloc[-1]:
                    signaal = f"🟢 *TREND KOOP* | €{curr_p:.2f} | {y_l}"
        
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
        print(f"Analyseer {t}...")
        params = [
            ("T", (50, 200, True, False, False)),
            ("S", (20, 50, True, False, False)),
            ("HT", (9, 21, True, True, False)),
            ("HS", (9, 21, False, True, False)),
            ("MR", (0, 0, True, False, True))
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
        f"📉 *Mean Reversion:* €{100000 + res['MR']:,.0f}",
        "", "🛡️ *TRAAG:*", "\n".join(sig["T"]) or "Geen actie",
        "", "🎯 *SNEL:*", "\n".join(sig["S"]) or "Geen actie",
        "", "📉 *DIP KOOP:*", "\n".join(sig["MR"]) or "Geen actie"
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
