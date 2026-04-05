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

def bereken_bt_safe(ticker, inzet, s, t, is_ema=False):
    """Backtest die de 'multiple columns' bug van Yahoo Finance omzeilt."""
    try:
        # Download data (auto_adjust voorkomt extra kolommen)
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        
        if df.empty or len(df) < 250:
            return 0
        
        # ESSENTIËLE FIX: Forceer 1 kolom voor Close als Yahoo Multi-index stuurt
        if isinstance(df.columns, pd.MultiIndex):
            close_data = df['Close'][ticker]
        else:
            close_data = df['Close']
            
        # Maak een schone serie van de sluitingsprijzen
        prices = close_data.dropna().astype(float)
        
        # Bereken gemiddelden op de serie (voorkomt kolom-fouten)
        if is_ema:
            f_line = prices.ewm(span=s, adjust=False).mean()
            s_line = prices.ewm(span=t, adjust=False).mean()
        else:
            f_line = prices.rolling(window=s).mean()
            s_line = prices.rolling(window=t).mean()
            
        # Alleen laatste 2 jaar voor de backtest
        f_line = f_line.iloc[-504:]
        s_line = s_line.iloc[-504:]
        prices_subset = prices.iloc[-504:]
            
        saldo_delta = 0
        pos, instap = False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(1, len(prices_subset)):
            # Koop
            if not pos and f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]:
                instap = prices_subset.iloc[i]
                pos = True
                saldo_delta -= kosten
            # Verkoop
            elif pos and f_line.iloc[i] < s_line.iloc[i] and f_line.iloc[i-1] >= s_line.iloc[i-1]:
                saldo_delta += (inzet * (prices_subset.iloc[i] / instap) - inzet) - kosten
                pos = False
                
        return saldo_delta
    except Exception as e:
        print(f"Fout bij {ticker}: {e}")
        return 0

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not os.path.exists('tickers_01.txt'):
        return

    with open('tickers_01.txt', 'r') as f:
        # Tickers opschonen (verwijder $ en vreemde tekens)
        raw_content = f.read().replace('$', '').replace('\n', ',')
        tickers = [t.strip().upper() for t in raw_content.split(',') if t.strip()]

    inzet = 2500.0
    scores = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    # Verwerk tickers één voor één om crashes te isoleren
    for t in set(tickers): # set() verwijdert dubbele tickers
        scores["T"] += bereken_bt_safe(t, inzet, 50, 200, False)
        scores["S"] += bereken_bt_safe(t, inzet, 20, 50, False)
        scores["HT"] += bereken_bt_safe(t, inzet, 9, 21, True)
        scores["HS"] += bereken_bt_safe(t, inzet, 9, 21, True)

    rapport = [
        "✅ *HOOGLAND REPAIRED RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + scores['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + scores['S']:,.0f}",
        f"📈 *Hyper Trend:* €{100000 + scores['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + scores['HS']:,.0f}",
        "",
        "⚠️ _Sommige tickers (UAU, PROX) gaven geen data en zijn overgeslagen._"
    ]
    
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    main()
