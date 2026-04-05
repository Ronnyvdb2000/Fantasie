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

def bereken_adx(df, n=14):
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (plus_dm.rolling(n).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(n).mean() / (atr + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.rolling(n).mean()

def bereken_atr(df, n=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def check_elite_signaal(df, s, t, ticker, is_ema=False, use_ema200=True):
    df = df.copy()
    if is_ema:
        df['F'] = df['Close'].ewm(span=s, adjust=False).mean()
        df['S'] = df['Close'].ewm(span=t, adjust=False).mean()
    else:
        df['F'] = df['Close'].rolling(window=s).mean()
        df['S'] = df['Close'].rolling(window=t).mean()
    
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['ADX'] = bereken_adx(df)
    df['ATR'] = bereken_atr(df)
    df['Vol_MA'] = df['Volume'].rolling(20).mean()

    c_nu = df['Close'].iloc[-1]
    f_nu, f_oud = df['F'].iloc[-1], df['F'].iloc[-2]
    s_nu, s_oud = df['S'].iloc[-1], df['S'].iloc[-2]
    adx_nu = df['ADX'].iloc[-1]
    vol_nu, vol_ma = df['Volume'].iloc[-1], df['Vol_MA'].iloc[-1]
    atr_nu = df['ATR'].iloc[-1]
    ema200_nu = df['EMA200'].iloc[-1]

    # VERSOEPELDE FILTERS
    is_trending = adx_nu > 15 
    vol_confirm = vol_nu > (vol_ma * 0.6) # Volume eis op 60%
    trend_filter = c_nu > ema200_nu if use_ema200 else True

    if f_nu > s_nu and f_oud <= s_oud:
        if is_trending and vol_confirm and trend_filter:
            sl = c_nu - (2 * atr_nu)
            return f"🚀 *KOOP* | €{c_nu:.2f} | SL: €{sl:.2f} (ADX: {adx_nu:.1f})"
    elif f_nu < s_nu and f_oud >= s_oud:
        return f"💀 *VERKOOP* | €{c_nu:.2f}"
    return None

def bereken_bt_elite(df, inzet, s, t, is_ema=False, use_ema200=True):
    VASTE_KOST, BEURSTAKS = 15.00, 0.0035
    # BACKTEST NAAR 2 JAAR (504 dagen)
    data = df.iloc[-504:].copy()
    
    if is_ema:
        data['F'] = data['Close'].ewm(span=s, adjust=False).mean()
        data['S'] = data['Close'].ewm(span=t, adjust=False).mean()
    else:
        data['F'] = data['Close'].rolling(s).mean()
        data['S'] = data['Close'].rolling(t).mean()
    
    data['ADX'] = bereken_adx(df).reindex(data.index)
    data['ATR'] = bereken_atr(df).reindex(data.index)
    data['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean().reindex(data.index)

    pos, k, saldo, high_p, sl = False, 0, inzet, 0, 0

    for i in range(1, len(data)):
        p = float(data['Close'].iloc[i])
        f_nu, f_oud = data['F'].iloc[i], data['F'].iloc[i-1]
        s_nu, s_oud = data['S'].iloc[i], data['S'].iloc[i-1]
        
        if not pos:
            # ADX 15 check voor de backtest
            if f_nu > s_nu and f_oud <= s_oud and data['ADX'].iloc[i] > 15:
                if not use_ema200 or p > data['EMA200'].iloc[i]:
                    k, pos, high_p = p, True, p
                    sl = p - (2 * data['ATR'].iloc[i])
                    saldo -= (VASTE_KOST + (inzet * BEURSTAKS))
        else:
            high_p = max(high_p, p)
            sl = max(sl, high_p - (2 * data['ATR'].iloc[i])) # Trailing Stop

            if p < sl or (f_nu < s_nu and f_oud >= s_oud):
                bruto = inzet * (p / k)
                saldo += (bruto - inzet) - (VASTE_KOST + (bruto * BEURSTAKS))
                pos = False
    return saldo

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'): return

    with open('tickers_01.txt', 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]

    inzet = 2500.0
    scores = {"T": 100000.0, "S": 100000.0, "HT": 100000.0, "HS": 100000.0}
    signals = {"T": [], "S": [], "HT": [], "HS": []}

    for t in tickers:
        try:
            df = yf.download(t, period="5y", progress=False)
            if df.empty or len(df) < 200: continue

            scores["T"] += (bereken_bt_elite(df, inzet, 50, 200, False, True) - inzet)
            scores["S"] += (bereken_bt_elite(df, inzet, 20, 50, False, True) - inzet)
            scores["HT"] += (bereken_bt_elite(df, inzet, 9, 21, True, True) - inzet)
            scores["HS"] += (bereken_bt_elite(df, inzet, 9, 21, True, False) - inzet)

            st = check_elite_signaal(df, 50, 200, t, False, True)
            ss = check_elite_signaal(df, 20, 50, t, False, True)
            sht = check_elite_signaal(df, 9, 21, t, True, True)
            shs = check_elite_signaal(df, 9, 21, t, True, False)

            if st: signals["T"].append(f"• `{t}`: {st}")
            if ss: signals["S"].append(f"• `{t}`: {ss}")
            if sht: signals["HT"].append(f"• `{t}`: {sht}")
            if shs: signals["HS"].append(f"• `{t}`: {shs}")

        except Exception as e: print(f"Fout {t}: {e}")

    def clean(lst): 
        res = [s for s in lst if s is not None]
        return "\n".join(res) if res else "_Geen actie_"

    rapport = [
        "📊 *ELITE TRADING RAPPORT (OPTIMAL)*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200 SMA):* €{scores['T']:,.0f}",
        f"⚡ *Snel (20/50 SMA):* €{scores['S']:,.0f}",
        f"🚀 *Hyper Trend (9/21 EMA200):* €{scores['HT']:,.0f}",
        f"🔥 *Hyper Scalp (9/21):* €{scores['HS']:,.0f}",
        "",
        "🛡️ *SIGNALEN TRAAG:*", clean(signals["T"]),
        "",
        "🎯 *SIGNALEN SNEL:*", clean(signals["S"]),
        "",
        "📈 *SIGNALEN HYPER TREND:*", clean(signals["HT"]),
        "",
        "⚡ *SIGNALEN HYPER SCALP:*", clean(signals["HS"])
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
