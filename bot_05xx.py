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

def check_munger_kwaliteit(ticker_obj):
    try:
        info = ticker_obj.info
        roic = info.get('returnOnCapitalEmployed') or info.get('returnOnAssets', 0)
        debt_to_equity = info.get('debtToEquity', 150) / 100.0 # Iets ruimer filter
        if roic > 0.08 and debt_to_equity < 1.2: # Realistischere 2026 criteria
            return True, roic
    except: pass
    return False, 0

def bereken_indicatoren_pullback(df):
    p = df['Close'].ffill()
    v = df['Volume'].ffill()
    
    # Gemiddelden
    sma50 = p.rolling(window=50).mean()
    sma200 = p.rolling(window=200).mean()
    vol_ma = v.rolling(window=20).mean()
    
    # RSI (14)
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR voor Stop Loss
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-p.shift()), abs(df['Low']-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, sma50, sma200, rsi, atr, v, vol_ma

def voer_strategie_uit(tickers, label):
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    inzet = 2500.0
    bolero_fee = 15.0 + (inzet * 0.0035)
    
    total_profit = 0
    signalen = []

    for t in tickers:
        try:
            t_obj = yf.Ticker(t)
            is_ok, roic = check_munger_kwaliteit(t_obj)
            if not is_ok: continue

            data = t_obj.history(period="2y")
            if len(data) < 200: continue

            p, s50, s200, rsi, atr, vol, v_ma = bereken_indicatoren_pullback(data)
            
            # Backtest (252 dagen)
            profit, pos, instap, sl = 0, False, 0, 0
            for i in range(150, len(p)): # Start na indicator opbouw
                cp = p.iloc[i]
                
                # KOOP: Kwaliteit op een dip (RSI < 50) boven de SMA200
                if not pos:
                    if cp > s200.iloc[i] and rsi.iloc[i] < 50 and rsi.iloc[i] > 35:
                        if vol.iloc[i] > v_ma.iloc[i] * 0.8: # Bevestiging van volume
                            instap, sl, pos = cp, cp - (2.5 * atr.iloc[i]), True
                            profit -= bolero_fee
                # VERKOOP: Bij herstel naar RSI > 70 of Trendbreuk
                else:
                    if cp < sl or rsi.iloc[i] > 70 or cp < s200.iloc[i]:
                        profit += (inzet * (cp / instap) - inzet) - bolero_fee
                        pos = False
            
            total_profit += profit

            # Actueel Signaal
            curr_cp, curr_rsi, curr_s200 = p.iloc[-1], rsi.iloc[-1], s200.iloc[-1]
            if curr_cp > curr_s200 and 35 < curr_rsi < 52:
                signalen.append(f"• `{t}`: 🔵 *DIP-KOOP* | €{curr_cp:.2f} (RSI: {curr_rsi:.1f})")
            elif pos and curr_rsi > 68:
                signalen.append(f"• `{t}`: 🟠 *WINSTNEMING* | €{curr_cp:.2f}")

        except: continue

    rapport = [
        f"🏦 *{label} PULLBACK REPORT*",
        f"_{nu}_",
        f"💰 *Netto Profit (Backtest):* €{total_profit:,.2f}",
        "----------------------------------",
        "*SIGNALEN:*",
        "\n".join(signalen) if signalen else "Geen actuele kansen."
    ]
    stuur_telegram("\n".join(rapport))

# Gebruik de tickers van de 30 Benelux aandelen
benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]

voer_strategie_uit(benelux_30, "BENELUX")
