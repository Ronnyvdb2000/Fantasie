import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import time

# ... (stuur_telegram en check_munger_kwaliteit blijven identiek) ...

def bereken_indicatoren_expert(df, s, t, is_hyper):
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    # Moving Averages
    f_line = p.rolling(window=s).mean()
    s_line = p.rolling(window=t).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    
    # RSI (14)
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))

    # ATR voor Stop Loss
    tr = pd.concat([h-l, abs(h-p.shift()), abs(l-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, f_line, s_line, ema200, rsi, atr

def voer_lijst_uit_geoptimaliseerd(tickers, naam_sector):
    res = {"T": 0, "S": 0}
    sig = {"T": [], "S": []}
    inzet = 2500.0
    bolero_min_fee = 15.0 # Jouw Bolero kostenstructuur

    for t in tickers:
        try:
            t_obj = yf.Ticker(t)
            is_ok, roic, debt = check_munger_kwaliteit(t_obj)
            if not is_ok: continue

            data = t_obj.history(period="5y")
            if len(data) < 200: continue

            # Focus op Traag (50/200) en Snel (20/50)
            for strat_key, (s_per, t_per) in [("T",(50,200)), ("S",(20,50))]:
                p, f, s_line, e200, rsi, atr = bereken_indicatoren_expert(data, s_per, t_per, False)
                
                # Backtest laatste 252 dagen
                p_bt, f_bt, s_bt, e_bt, r_bt, a_bt = p.iloc[-252:], f.iloc[-252:], s_line.iloc[-252:], e200.iloc[-252:], rsi.iloc[-252:], atr.iloc[-252:]
                
                profit, pos, instap, sl_val = 0, False, 0, 0
                fee = bolero_min_fee + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp = p_bt.iloc[i]
                    # KOOP LOGICA MET MEAN REVERSION FILTER
                    if not pos:
                        if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1]:
                            # FILTER: Alleen kopen als RSI niet 'overbought' is en koers boven EMA200
                            if r_bt.iloc[i] < 60 and cp > e_bt.iloc[i] * 1.01:
                                instap = cp
                                sl_val = cp - (3 * a_bt.iloc[i]) # Ruime SL voor kwaliteit
                                pos = True
                                profit -= fee
                    # VERKOOP LOGICA
                    else:
                        if cp < sl_val or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet * (cp / instap) - inzet) - fee
                            pos = False
                
                res[strat_key] += profit

                # Actueel Signaal
                if f.iloc[-1] > s_line.iloc[-1] and f.iloc[-2] <= s_line.iloc[-2]:
                    if rsi.iloc[-1] < 65:
                        sig[strat_key].append(f"• `{t}`: 🟢 KOOP (RSI OK)")
                elif f.iloc[-1] < s_line.iloc[-1] and f.iloc[-2] >= s_line.iloc[-2]:
                    sig[strat_key].append(f"• `{t}`: 🔴 VERKOOP")

        except: continue

    # (Telegram rapportage logica hier invoegen...)
