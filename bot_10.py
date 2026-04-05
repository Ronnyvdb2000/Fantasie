import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()

TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_indicatoren(df, ticker):
    """Berekent alle technische filters op een veilige manier."""
    if isinstance(df.columns, pd.MultiIndex):
        p = df['Close'][ticker].dropna().astype(float)
        v = df['Volume'][ticker].dropna().astype(float)
        h = df['High'][ticker].dropna().astype(float)
        l = df['Low'][ticker].dropna().astype(float)
    else:
        p, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']

    # Gemiddelden
    ema9 = p.ewm(span=9, adjust=False).mean()
    ema21 = p.ewm(span=21, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    
    # Volume Filter (60%)
    vol_ma = v.rolling(window=20).mean()
    
    # ATR (Volatiliteit)
    tr = pd.concat([h - l, abs(h - p.shift()), abs(l - p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX (Trendsterkte op 15)
    plus_dm = (h.diff().clip(lower=0)).rolling(14).mean()
    minus_dm = ((-l.diff()).clip(lower=0)).rolling(14).mean()
    dx = 100 * abs(plus_dm - minus_dm) / (plus_dm + minus_dm + 1e-10)
    adx = dx.rolling(14).mean()
    
    return p, ema9, ema21, ema200, vol_ma, v, atr, adx

def bereken_bt_elite(ticker, inzet, use_trend_filter=False):
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        
        p, f_line, s_line, ema200, vol_ma, vol_nu, atr, adx = bereken_indicatoren(df, ticker)
        
        # Laatste jaar (252 dagen)
        p, f_line, s_line = p.iloc[-252:], f_line.iloc[-252:], s_line.iloc[-252:]
        ema200, vol_ma, vol_nu = ema200.iloc[-252:], vol_ma.iloc[-252:], vol_nu.iloc[-252:]
        atr, adx = atr.iloc[-252:], adx.iloc[-252:]
            
        saldo_delta = 0
        pos, instap, high_p, sl = False, 0, 0, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p)):
            curr_p = p.iloc[i]
            # KOOP CONDITIES
            if not pos:
                crossover = f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]
                voldoet_adx = adx.iloc[i] > 15
                voldoet_vol = vol_nu.iloc[i] > (vol_ma.iloc[i] * 0.6)
                voldoet_trend = curr_p > ema200.iloc[i] if use_trend_filter else True
                
                if crossover and voldoet_adx and voldoet_vol and voldoet_trend:
                    instap = curr_p
                    high_p = curr_p
                    sl = curr_p - (2 * atr.iloc[i]) # Initiële Stop Loss
                    pos = True
                    saldo_delta -= kosten
            
            # POSITIE BEHEER (Trailing Stop & Exit)
            else:
                high_p = max(high_p, curr_p)
                sl = max(sl, high_p - (2 * atr.iloc[i])) # Trailing Stop schuift mee omhoog
                
                # EXIT (Stop loss geraakt OF cross-back)
                if curr_p < sl or (f_line.iloc[i] < s_line.iloc[i]):
                    saldo_delta += (inzet * (curr_p / instap) - inzet) - kosten
                    pos = False
                    
        return saldo_delta
    except:
        return 0

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'): return

    with open('tickers_01.txt', 'r') as f:
        raw = f.read().replace('\n', ',').replace('$', '')
        tickers = list(set([t.strip().upper() for t in raw.split(',') if t.strip()]))

    inzet = 2500.0
    scores = {"HT": 0, "HS": 0}

    for t in tickers:
        # Hyper Trend (Met EMA200 filter)
        scores["HT"] += bereken_bt_elite(t, inzet, use_trend_filter=True)
        # Hyper Scalp (Zonder EMA200 filter)
        scores["HS"] += bereken_bt_elite(t, inzet, use_trend_filter=False)

    rapport = [
        "🛡️ *ELITE HYPER RAPPORT (v12)*",
        f"_{nu}_",
        "----------------------------------",
        f"🚀 *Hyper Trend (9/21 + EMA200):* €{100000 + scores['HT']:,.0f}",
        f"🔥 *Hyper Scalp (9/21):* €{100000 + scores['HS']:,.0f}",
        "",
        "⚙️ *FILTERS ACTIEF:*",
        "• ADX > 15 (Trendbevestiging)",
        "• Volume > 60% (Liquiditeitscheck)",
        "• ATR Trailing Stop (Risicobeheer)",
        "• Periode: Laatste jaar (252 dagen)"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
