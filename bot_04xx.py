import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_indicatoren(df, s, t, is_hyper):
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    
    # Voortschrijdende gemiddelden
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()

    # RSI / CRSI Logica
    delta = p.diff()
    if is_hyper:
        rsi3_gain = (delta.where(delta > 0, 0)).rolling(3).mean()
        rsi3_loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
        rsi3 = 100 - (100 / (1 + (rsi3_gain / (rsi3_loss + 1e-10))))
        
        change = np.sign(delta)
        streak = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta = streak.diff()
        s_gain = (s_delta.where(s_delta > 0, 0)).rolling(2).mean()
        s_loss = (-s_delta.where(s_delta < 0, 0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + (s_gain / (s_loss + 1e-10))))
        
        p_rank = delta.rolling(100).apply(lambda x: (x < x.iloc[-1]).sum() / 99.0 * 100, raw=False)
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3
    else:
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, f_line, s_line, ema200, rsi_val, atr

def voer_lijst_uit(tickers):
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    inzet = 2500.0
    fee = 15.0 + (inzet * 0.0035) 
    
    res = {"T": 0, "S": 0, "HT": 0, "HS": 0}

    for t in tickers:
        try:
            # We gebruiken auto_adjust=True voor zuivere koersen
            data = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(data) < 200: continue

            # De 4 Originele Strategieën
            strat_config = [
                ("T", 50, 200, True, False),   # Traag
                ("S", 20, 50, True, False),    # Snel
                ("HT", 9, 21, True, True),     # Hyper Trend
                ("HS", 9, 21, False, True)     # Hyper Scalp
            ]

            for key, s, t_per, use_trend, is_hyp in strat_config:
                p, f, s_line, e200, rsi, atr = bereken_indicatoren(data, s, t_per, is_hyp)
                
                pos, instap, sl, profit = False, 0, 0, 0
                
                # Backtest venster: Laatste 252 dagen
                for i in range(len(p)-252, len(p)):
                    cp = p.iloc[i]
                    if not pos:
                        # Koop conditie: Crossover
                        if f.iloc[i] > s_line.iloc[i] and f.iloc[i-1] <= s_line.iloc[i-1]:
                            # Trend filter (EMA200) alleen indien gevraagd
                            if not use_trend or cp > e200.iloc[i]:
                                instap = cp
                                sl = cp - (2.5 * atr.iloc[i])
                                pos = True
                                profit -= fee
                    else:
                        # Verkoop conditie: Crossover terug of Stop Loss
                        if cp < sl or f.iloc[i] < s_line.iloc[i]:
                            profit += (inzet * (cp / instap) - inzet) - fee
                            pos = False
                
                res[key] += profit

        except Exception as e:
            print(f"Fout bij {t}: {e}")

    rapport = [
        f"📊 *BOT V5.2 - ORIGINELE VISIE*",
        f"📅 {nu}",
        "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        "----------------------------------",
        "_De backtest resultaten zijn nu weer zichtbaar._"
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
    voer_lijst_uit(benelux_30)
