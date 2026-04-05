import yfinance as yf
import pandas as pd
import os
import requests
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

def check_flexibel_signaal(df, s, t, ticker, is_ema=False, use_ema200=True):
    df = df.copy()
    if is_ema:
        df['F'] = df['Close'].ewm(span=s, adjust=False).mean()
        df['S'] = df['Close'].ewm(span=t, adjust=False).mean()
    else:
        df['F'] = df['Close'].rolling(window=s).mean()
        df['S'] = df['Close'].rolling(window=t).mean()
    
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['ADX'] = bereken_adx(df)
    
    c_nu = df['Close'].iloc[-1]
    f_nu, f_oud = df['F'].iloc[-1], df['F'].iloc[-2]
    s_nu, s_oud = df['S'].iloc[-1], df['S'].iloc[-2]
    adx_nu = df['ADX'].iloc[-1]
    ema200_nu = df['EMA200'].iloc[-1]

    # Check voor de kruising
    if f_nu > s_nu and f_oud <= s_oud:
        # Filter check
        trend_ok = adx_nu > 15
        ema_ok = c_nu > ema200_nu if use_ema200 else True
        
        if trend_ok and ema_ok:
            return f"🌟 *GOUD KOOP* | €{c_nu:.2f} (Sterke trend ADX:{adx_nu:.1f})"
        else:
            return f"🥈 *ZILVER KOOP* | €{c_nu:.2f} (Zwakke trend/lage EMA)"
            
    elif f_nu < s_nu and f_oud >= s_oud:
        return f"💀 *VERKOOP* | €{c_nu:.2f}"
    return None

def bereken_bt_flex(df, inzet, s, t, is_ema=False, use_ema200=True):
    # Backtest zonder ADX filter om de historie te vullen
    VASTE_KOST, BEURSTAKS = 15.00, 0.0035
    data = df.iloc[-504:].copy() # 2 jaar
    if is_ema:
        data['F'] = data['Close'].ewm(span=s, adjust=False).mean()
        data['S'] = data['Close'].ewm(span=t, adjust=False).mean()
    else:
        data['F'] = data['Close'].rolling(s).mean()
        data['S'] = data['Close'].rolling(t).mean()
    
    pos, k, saldo = False, 0, inzet
    for i in range(1, len(data)):
        p = float(data['Close'].iloc[i])
        if not pos and data['F'].iloc[i] > data['S'].iloc[i] and data['F'].iloc[i-1] <= data['S'].iloc[i-1]:
            k, pos, saldo = p, True, saldo - (VASTE_KOST + (inzet * BEURSTAKS))
        elif pos and data['F'].iloc[i] < data['S'].iloc[i] and data['F'].iloc[i-1] >= data['S'].iloc[i-1]:
            saldo += (inzet * (p / k) - inzet) - (VASTE_KOST + (inzet * (p / k) * BEURSTAKS))
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
            
            scores["T"] += (bereken_bt_flex(df, inzet, 50, 200, False, True) - inzet)
            scores["S"] += (bereken_bt_flex(df, inzet, 20, 50, False, True) - inzet)
            scores["HT"] += (bereken_bt_flex(df, inzet, 9, 21, True, True) - inzet)
            scores["HS"] += (bereken_bt_flex(df, inzet, 9, 21, True, False) - inzet)

            # Gebruik de flexibele check die ook "Zilver" geeft
            signals["T"].append(f"• `{t}`: {check_flexibel_signaal(df, 50, 200, t, False, True)}")
            signals["S"].append(f"• `{t}`: {check_flexibel_signaal(df, 20, 50, t, False, True)}")
            signals["HT"].append(f"• `{t}`: {check_flexibel_signaal(df, 9, 21, t, True, True)}")
            signals["HS"].append(f"• `{t}`: {check_flexibel_signaal(df, 9, 21, t, True, False)}")
        except Exception as e: print(f"Fout {t}: {e}")

    def clean(lst): 
        res = [s for s in lst if s is not None]
        return "\n".join(res) if res else "_Geen kruising vandaag_"

    rapport = [
        "📊 *RAPPORT MET FLEXIBELE FILTERS*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag:* €{scores['T']:,.0f} | ⚡ *Snel:* €{scores['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{scores['HT']:,.0f} | 🔥 *Hyper Scalp:* €{scores['HS']:,.0f}",
        "",
        "🔍 *SIGNALEN VAN VANDAAG:*",
        "*Hyper Scalp (Meest actief):*", clean(signals["HS"]),
        "*Hyper Trend:*", clean(signals["HT"])
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
