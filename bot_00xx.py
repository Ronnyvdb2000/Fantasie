from __future__ import annotations

import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import logging
import time

# ---------------------------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()
        time.sleep(1)
        return True
    except: return False

# ---------------------------------------------------------------------------
# INDICATOREN (Wilder Smoothing & Robust Streaks)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(df: pd.DataFrame, s: int, t: int, use_trend_filter: bool, is_hyper: bool) -> tuple:
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    
    # RSI met Wilder Smoothing (Standaard)
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Strat 5 Indicatoren
    rsi2_gain = delta.where(delta > 0, 0.0).ewm(alpha=1/2, adjust=False).mean()
    rsi2_loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/2, adjust=False).mean()
    rsi2 = 100 - (100 / (1 + rsi2_gain / (rsi2_loss + 1e-10)))
    ma5 = p.rolling(window=5).mean()
    ma2 = p.rolling(window=2).mean()
    std20 = p.rolling(window=20).std()
    lower_b = s_line - (2.0 * std20)

    if is_hyper:
        # Robuuste Streak RSI berekening tegen NaN
        change = np.sign(delta).fillna(0)
        streak = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta = streak.diff().fillna(0)
        s_gain = s_delta.where(s_delta > 0, 0.0).rolling(2).mean()
        s_loss = (-s_delta.where(s_delta < 0, 0.0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + s_gain / (s_loss + 1e-10))).fillna(50)
        
        rsi3_gain = delta.where(delta > 0, 0.0).ewm(alpha=1/3, adjust=False).mean()
        rsi3_loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/3, adjust=False).mean()
        rsi3 = 100 - (100 / (1 + rsi3_gain / (rsi3_loss + 1e-10)))
        
        p_rank = delta.rolling(100).apply(lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100 if len(x) > 0 else 50, raw=True)
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3

    # Correcte ADX met Wilder Smoothing
    tr = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v, rsi2, ma5, ma2, lower_b

# ---------------------------------------------------------------------------
# SECTOR VERWERKING
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> None:
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")

    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except: return

    inzet = 2500.0
    res = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRA": 0.0}
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MRA": []}
    STRATS = [("T", 50, 200, True, False), ("S", 20, 50, True, False), ("HT", 9, 21, True, True), ("HS", 9, 21, False, True)]

    for ticker in tickers:
        try:
            t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all') if len(tickers) > 1 else raw_df.dropna(how='all')
            if len(t_data) < 250: continue

            # --- BACKTEST OVER DE VOLLEDIGE 5 JAAR (minus warm-up) ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                p, f, sl, e200, v_ma, rsi, atr, adx, vol, rsi2, ma5, ma2, l_b = bereken_indicatoren_vectorized(t_data, s_p, t_p, utr, ihyp)
                
                # Start na 200 dagen opwarming
                p_bt = p.iloc[200:]; f_bt = f.iloc[200:]; s_bt = sl.iloc[200:]
                e_bt = e200.iloc[200:]; v_bt = vol.iloc[200:]; v_ma_bt = v_ma.iloc[200:]
                atr_bt = atr.iloc[200:]; adx_bt = adx.iloc[200:]
                
                profit = 0.0; pos = False; instap = 0.0; high_p = 0.0; kosten = 15.0 + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp = p_bt.iloc[i]
                    if not pos:
                        if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1] and adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i]*0.6) and ((not utr) or cp > e_bt.iloc[i]):
                            instap, high_p, pos = cp, cp, True
                            profit -= kosten
                    else:
                        high_p = max(high_p, cp)
                        if cp < (high_p - 2*atr_bt.iloc[i]) or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet*(cp/instap)-inzet)-kosten
                            pos = False
                
                # PUNT 3: Open positie verrekenen (Mark-to-Market)
                if pos: profit += (inzet*(p_bt.iloc[-1]/instap)-inzet)-kosten
                res[skey] += profit

                # Signaal logic (Alleen laatste dag)
                if skey == "T":
                    cp = p.iloc[-1]
                    y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    if f.iloc[-1] > sl.iloc[-1] and f.iloc[-2] <= sl.iloc[-2] and adx.iloc[-1] > 15 and vol.iloc[-1] > (v_ma.iloc[-1]*0.6) and ((not utr) or cp > e200.iloc[-1]):
                        sig[skey].append(f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | {y_l}")
                    elif f.iloc[-1] < sl.iloc[-1] and f.iloc[-2] >= sl.iloc[-2]:
                        sig[skey].append(f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | {y_l}")

            # --- STRAT 5 BACKTEST (5 JAAR) ---
            p, f, sl, e200, v_ma, rsi, atr, adx, vol, rsi2, ma5, ma2, l_b = bereken_indicatoren_vectorized(t_data, 50, 200, True, False)
            pb, r2b, m5b, m2b, e_st = p.iloc[200:], rsi2.iloc[200:], ma5.iloc[200:], ma2.iloc[200:], e200.diff(5).iloc[200:]
            pr5, pos5, ins5 = 0.0, False, 0.0
            
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos5:
                    if r2b.iloc[i] < 5 and cp < (m5b.iloc[i]*0.95) and e_st.iloc[i] > 0:
                        ins5, pos5 = cp, True
                        pr5 -= kosten
                else:
                    if cp > m2b.iloc[i] or cp > (ins5 * 1.05):
                        pr5 += (inzet*(cp/ins5)-inzet)-kosten; pos5 = False
            
            if pos5: pr5 += (inzet*(pb.iloc[-1]/ins5)-inzet)-kosten
            res["MRA"] += pr5

            if rsi2.iloc[-1] < 10 and p.iloc[-1] < (ma5.iloc[-1]*0.95) and e200.diff(5).iloc[-1] > 0:
                sig["MRA"].append(f"• `{ticker}`: 💎 *POWER BOUNCE* | €{p.iloc[-1]:.2f}")

        except: continue

    def fmt(n): return f"€{100000 + n:,.0f}"
    rapport = [
        f"📊 *{label} {naam_sector} RAPPORT*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* {fmt(res['T'])}",
        f"⚡ *Snel (20/50):* {fmt(res['S'])}",
        f"🚀 *Hyper Trend:* {fmt(res['HT'])}",
        f"🔥 *Hyper Scalp:* {fmt(res['HS'])}",
        f"💎 *Power Mean Rev:* {fmt(res['MRA'])}",
        "", "🛡️ *SIGNALEN TRAAG:*", get_s(sig["T"]),
        "", "💎 *SIGNALEN POWER MEAN REV:*", get_s(sig["MRA"]),
    ]
    stuur_telegram("\n".join(rapport))

def main():
    sectoren = {"01":"Hoogland", "02":"Macrotrends", "03":"Beursbrink", "04":"Benelux", "05":"Parijs", "06":"Power & AI", "07":"Metalen", "08":"Defensie", "09":"Varia"}
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)

def get_s(lst: list) -> str: return "\n".join(lst) if lst else "Geen actie"

if __name__ == "__main__": main()
