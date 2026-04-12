import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def main():
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    with open('tickers_01.txt', 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    # DOWNLOAD ALLES IN ÉÉN KEER (Voorkomt blokkade door Yahoo)
    data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
    
    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}
    
    # We lopen door de tickers en berekenen exact jouw Bot 2 logica
    for t in tickers:
        try:
            p = data['Close'][t].ffill()
            h = data['High'][t].ffill()
            l = data['Low'][t].ffill()
            v = data['Volume'][t].ffill()

            for k, (s, t_val, use_filter) in [("T",(50,200,True)), ("S",(20,50,True)), ("HT",(9,21,True)), ("HS",(9,21,False))]:
                # INDICATOREN (Exact Bot 2)
                f_line = p.rolling(s).mean() if s >= 20 else p.ewm(span=s).mean()
                s_line = p.rolling(t_val).mean() if t_val >= 50 else p.ewm(span=t_val).mean()
                ema200 = p.ewm(span=200).mean()
                
                tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                
                # BACKTEST LOOP
                profit, pos, instap, high_p, sl_val = 0, False, 0, 0, 0
                # De laatste 252 dagen
                for i in range(len(p)-252, len(p)):
                    cp = p.iloc[i]
                    if not pos:
                        if f_line.iloc[i] > s_line.iloc[i] and f_line.iloc[i-1] <= s_line.iloc[i-1]:
                            if not use_filter or cp > ema200.iloc[i]:
                                instap, high_p, sl_val, pos = cp, cp, cp - (2 * atr.iloc[i]), True
                                profit -= 15.0 # Bot 2 kosten
                    else:
                        high_p = max(high_p, cp)
                        sl_val = max(sl_val, high_p - (2 * atr.iloc[i]))
                        if cp < sl_val or f_line.iloc[i] < s_line.iloc[i]:
                            w = (inzet * (cp / instap) - inzet) - 15.0
                            if w > 0: w *= 0.9 # Opt-in
                            profit += w
                            pos = False
                res[k] += profit
        except: continue

    rapport = (
        f"📊 *Bot 2 Logica op Tickers 01*\n"
        f"_{nu}_\n"
        f"----------------------------------\n"
        f"🐢 Traag: €{100000 + res['T']:,.0f}\n"
        f"⚡ Snel: €{100000 + res['S']:,.0f}\n"
        f"🚀 Hyper T: €{100000 + res['HT']:,.0f}\n"
        f"🔥 Hyper S: €{100000 + res['HS']:,.0f}"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
