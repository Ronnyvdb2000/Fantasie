import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})
    except: pass

def bereken_trend(ticker, inzet, s, t, use_filter=False):
    try:
        df = yf.download(ticker, period="3y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 260: return 0
        p = df['Close'].ffill().astype(float)
        
        # Indicatoren
        f_line = p.ewm(span=s, adjust=False).mean()
        s_line = p.ewm(span=t, adjust=False).mean()
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        p_bt = p.iloc[-252:]
        f_bt = f_line.iloc[-252:]
        s_bt = s_line.iloc[-252:]
        e_bt = ema200.iloc[-252:]
        
        profit, pos, instap = 0, False, 0
        kosten = 15.0 # De oude standaardkost uit jouw eerste versie

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            if not pos:
                # Signaal: Cross-over naar boven
                if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    if not use_filter or cp > e_bt.iloc[i]:
                        instap, pos = cp, True
                        profit -= kosten
            else:
                # Signaal: Cross-over naar beneden
                if f_bt.iloc[i] < s_bt.iloc[i]:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False
        return profit
    except: return 0

def main():
    with open('tickers_01.txt', 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    for t in tickers:
        res["T"] += bereken_trend(t, inzet, 50, 200, True)   # Traag
        res["S"] += bereken_trend(t, inzet, 20, 50, True)    # Snel
        res["HT"] += bereken_trend(t, inzet, 9, 21, True)    # Hyper T
        res["HS"] += bereken_trend(t, inzet, 9, 21, False)   # Hyper S

    rapport = (
        f"📊 *BOT 01 RESULTATEN*\n"
        f"🐢 Traag: €{100000 + res['T']:,.0f}\n"
        f"⚡ Snel: €{100000 + res['S']:,.0f}\n"
        f"🚀 Hyper T: €{100000 + res['HT']:,.0f}\n"
        f"🔥 Hyper S: €{100000 + res['HS']:,.0f}"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
