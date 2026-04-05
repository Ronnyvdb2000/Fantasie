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

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_bt_final(ticker, inzet, s, t, is_ema=False):
    """Robuuste backtest voor grote lijsten en crypto data."""
    try:
        # Download data
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 250: return 0
        
        # Selecteer de juiste kolom (Fix voor BTC-USD multi-index bug)
        if isinstance(df.columns, pd.MultiIndex):
            prices = df['Close'][ticker].dropna().astype(float)
        else:
            prices = df['Close'].dropna().astype(float)
            
        if is_ema:
            f_line = prices.ewm(span=s, adjust=False).mean()
            s_line = prices.ewm(span=t, adjust=False).mean()
        else:
            f_line = prices.rolling(window=s).mean()
            s_line = prices.rolling(window=t).mean()
            
        # Laatste 2 jaar (504 dagen)
        f_act = f_line.iloc[-504:]
        s_act = s_line.iloc[-504:]
        p_act = prices.iloc[-504:]
            
        saldo_delta = 0
        pos, instap = False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_act)):
            # KOOP
            if not pos and f_act.iloc[i] > s_act.iloc[i] and f_act.iloc[i-1] <= s_act.iloc[i-1]:
                instap = p_act.iloc[i]
                pos = True
                saldo_delta -= kosten
            # VERKOOP
            elif pos and f_act.iloc[i] < s_act.iloc[i] and f_act.iloc[i-1] >= s_act.iloc[i-1]:
                saldo_delta += (inzet * (p_act.iloc[i] / instap) - inzet) - kosten
                pos = False
        return saldo_delta
    except:
        return 0

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'): return

    with open('tickers_01.txt', 'r') as f:
        raw = f.read().replace('\n', ',')
        tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]

    inzet = 2500.0
    scores = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    print(f"Start analyse voor {len(tickers)} tickers...")
    
    for t in tickers:
        scores["T"] += bereken_bt_final(t, inzet, 50, 200, False)
        scores["S"] += bereken_bt_final(t, inzet, 20, 50, False)
        scores["HT"] += bereken_bt_final(t, inzet, 9, 21, True)
        scores["HS"] += bereken_bt_final(t, inzet, 9, 21, True)

    rapport = [
        "🏆 *HOOGLAND PORTFOLIO PERFORMANCE*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + scores['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + scores['S']:,.0f}",
        f"📈 *Hyper Trend:* €{100000 + scores['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + scores['HS']:,.0f}",
        "",
        f"✅ _Succesvol berekend voor {len(tickers)} tickers._"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
