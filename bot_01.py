import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time

# --- SETUP ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_alles(ticker, inzet, s, t, use_trend_filter=False, is_hyper=False, use_mean_rev=False):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 100: return 0, None
        
        p = df['Close'].dropna().astype(float)
        if isinstance(p, pd.DataFrame): p = p.iloc[:, 0] # Fix voor MultiIndex
        
        # Indicatoren
        f_line = p.rolling(window=s).mean() if s > 0 else p
        s_line = p.rolling(window=t).mean() if t > 0 else p
        ma5 = p.rolling(window=5).mean()
        
        # Eenvoudige RSI2
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / (loss + 1e-10)
        rsi2 = 100 - (100 / (1 + rs))

        # Backtest parameters
        p_bt = p.iloc[-252:]
        r2_bt = rsi2.iloc[-252:]
        ma5_bt = ma5.iloc[-252:]
        f_bt = f_line.iloc[-252:]
        s_bt = s_line.iloc[-252:]
        
        profit = 0.0
        pos = False
        instap = 0.0
        kosten = 10.0 # Vaste lage kosten voor test

        for i in range(1, len(p_bt)):
            cp = float(p_bt.iloc[i])
            
            if not pos:
                # MEAN REVERSION TEST: RSI2 < 30 (zeer ruim)
                if use_mean_rev:
                    if r2_bt.iloc[i] < 30:
                        instap = cp
                        profit -= kosten
                        pos = True
                # TREND TEST
                elif f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                    instap = cp
                    profit -= kosten
                    pos = True
            else:
                # EXIT
                sell = False
                if use_mean_rev:
                    if cp > ma5_bt.iloc[i]: sell = True
                else:
                    if f_bt.iloc[i] < s_bt.iloc[i]: sell = True
                
                if sell:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False

        signaal = None
        if use_mean_rev and rsi2.iloc[-1] < 30:
            signaal = f"🔵 *DIP* | {ticker} | €{p.iloc[-1]:.2f}"
        elif not use_mean_rev and f_line.iloc[-1] > s_line.iloc[-1]:
            signaal = f"🟢 *TREND* | {ticker} | €{p.iloc[-1]:.2f}"

        return profit, signaal
    except Exception as e:
        print(f"Fout bij {ticker}: {e}")
        return 0, None

def voer_lijst_uit(bestandsnaam, label, naam_sector):
    if not os.path.exists(bestandsnaam): return
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = [t.strip().upper() for t in content.split(',') if t.strip()]

    inzet = 2500.0
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0, "MR": 0}
    sig = []

    for t in tickers:
        # We testen nu alleen MR en Snel om te zien of MR eindelijk beweegt
        p_mr, s_mr = bereken_alles(t, inzet, 0, 0, False, False, True)
        p_s, s_s = bereken_alles(t, inzet, 20, 50, True, False, False)
        
        res["MR"] += p_mr
        res["S"] += p_s
        if s_mr: sig.append(s_mr)

    rapport = (
        f"🧪 *TEST RAPPORT {naam_sector}*\n"
        f"⚡ Snel: €{100000 + res['S']:,.0f}\n"
        f"📉 Mean Rev: €{100000 + res['MR']:,.0f}\n\n"
        f"Signalen:\n" + "\n".join(sig[:5])
    )
    stuur_telegram(rapport)

def main():
    sectoren = {"01":"Hoogland"}
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)

if __name__ == "__main__":
    main()
