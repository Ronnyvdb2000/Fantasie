import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

# ---------------------------------------------------------------------------
# CONFIG & TELEGRAM
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# INDICATOREN (VECTORIZED)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(df, s, t, is_hyper):
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    # Moving Averages
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    # RSI / CRSI Logica
    delta = p.diff()
    if is_hyper:
        rsi3 = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(3).mean() / (-delta.where(delta < 0, 0).rolling(3).mean() + 1e-10))))
        change = np.sign(delta)
        streak = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta = streak.diff()
        streak_rsi = 100 - (100 / (1 + (s_delta.where(s_delta > 0, 0).rolling(2).mean() / (-s_delta.where(s_delta < 0, 0).rolling(2).mean() + 1e-10))))
        p_rank = delta.rolling(100).apply(lambda x: (x < x.iloc[-1]).sum() / 99.0 * 100, raw=False)
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3
    else:
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    # ATR & ADX
    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (h.diff().clip(lower=0).rolling(14).sum() / (tr14 + 1e-10))
    minus_di = 100 * ((-l.diff()).clip(lower=0).rolling(14).sum() / (tr14 + 1e-10))
    adx = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()

    # MRA Specifieke Indicatoren
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    lower_b3 = ma20 - (2.2 * std20)
    ibs = (p - l) / (h - l + 1e-10)
    ma5 = p.rolling(5).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v, ibs, lower_b3, ma5

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    
    inzet = 2500.0
    res = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRA": 0.0}
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MRA": []}
    STRATS = [("T", 50, 200, True, False), ("S", 20, 50, True, False), ("HT", 9, 21, True, True), ("HS", 9, 21, False, True)]

    for t in tickers:
        try:
            t_data = raw_df.xs(t, axis=1, level=1) if len(tickers) > 1 else raw_df
            kosten = 15.0 + (inzet * 0.0035)

            # --- STRAT 1-4: TREND & HYPER ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                p, f, sl, e200, v_ma, rsi, atr, adx, vol, _, _, _ = bereken_indicatoren_vectorized(t_data, s_p, t_p, ihyp)
                
                # Backtest (Trailing Stop Logica)
                p_bt = p.iloc[-252:]; f_bt = f.iloc[-252:]; s_bt = sl.iloc[-252:]
                e_bt = e200.iloc[-252:]; v_bt = vol.iloc[-252:]; vma_bt = v_ma.iloc[-252:]
                atr_bt = atr.iloc[-252:]; adx_bt = adx.iloc[-252:]
                
                profit, pos, instap, high_p, sl_val = 0,
