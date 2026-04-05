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
    """Backtest voor het LAATSTE JAAR (252 dagen)."""
    try:
        # Download data (5y nodig voor stabiele gemiddelden, we filteren later op 1y)
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 252: return 0
        
        # Fix voor BTC-USD en Multi-Index data
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
            
        # FILTER: Alleen het laatste jaar (252 handelsdagen)
        f_act = f_line.iloc[-252:]
        s_act = s_line.iloc[-252:]
        p_act = prices.iloc[-252:]
            
        saldo_delta = 0
        pos, instap = False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(p_act)):
            # KOOP (Cross-over omhoog)
            if not pos and f_act.iloc[i] > s_act.iloc[i] and f_act.iloc[i-1] <= s_act.iloc[i-1]:
                instap = p_act.iloc[i]
                pos = True
                saldo_delta -= kosten
            # VERKOOP (Cross-over omlaag)
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
        # Tickers inlezen en schoonmaken
        raw = f.read().replace('\n', ',').replace('$', '')
        tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
        tickers = list(set(tickers)) # Ontdubbelen

    inzet = 2500.0
    # Startkapitaal per bot is 100.000
    scores = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    print(f"Start analyse voor {len(tickers)} tickers over het laatste jaar...")
    
    for t in tickers:
        scores["T"] += bereken_bt_final(t, inzet, 50, 200, False)
        scores["S"] += bereken_bt_final(t, inzet, 20, 50, False)
        scores["HT"] += bereken_bt_final(t, inzet, 9, 21, True)
        scores["HS"] += bereken_bt_final(t, inzet, 9, 21, True)

    rapport = [
        "📊 *RAPPORT LAATSTE JAAR (252 DAGEN)*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Bot Traag (50/200 SMA):* €{100000 + scores['T']:,.0f}",
        f"⚡ *Bot Snel (20/50 SMA):* €{100000 + scores['S']:,.0f}",
        f"🚀 *Bot Hyper Trend (9/21 EMA):* €{100000 + scores['HT']:,.0f}",
        f"🔥 *Bot Hyper Scalp (9/21 EMA):* €{100000 + scores['HS']:,.0f}",
        "",
        f"✅ _Analyse voltooid voor {len(tickers)} tickers._",
        "⚠️ _Rendement inclusief €15 kosten & 0.35% taks per trade._"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
