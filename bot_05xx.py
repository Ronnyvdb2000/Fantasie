import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import time

# --- SETUP ---
VERSION = "2.0 - Bolero Quality Compounder"
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_kwaliteit_indicatoren(df):
    """
    Expert indicatoren voor kwaliteitsaandelen.
    Focus op Mean Reversion (RSI) en Trend (EMA).
    """
    close = df['Close'].ffill()
    
    # Trend filters
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # RSI (14) voor instapmomenten
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR voor dynamische Trailing Stop Loss
    high, low = df['High'], df['Low']
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return close, ema20, ema50, ema200, rsi, atr

def voer_backtest_v2(tickers, label):
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    inzet = 2500.0
    # Bolero Kosten: €15 min + 0.35% per transactie
    fee = 15.0 + (inzet * 0.0035)
    
    totaal_winst = 0
    signalen = []
    totaal_trades = 0

    for t in tickers:
        try:
            # We laden 2 jaar data voor een zuivere SMA200
            data = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(data) < 200: continue
            
            p, e20, e50, e200, rsi, atr = bereken_kwaliteit_indicatoren(data)
            
            # Start backtest venster (laatste 252 dagen)
            pos, instap, high_p, sl = False, 0, 0, 0
            ticker_profit = 0
            
            for i in range(len(p)-252, len(p)):
                cp = p.iloc[i]
                
                if not pos:
                    # KOOP CONDITIES (V2):
                    # 1. Prijs boven EMA200 (Lange termijn trend)
                    # 2. EMA20 > EMA50 (Korte termijn momentum)
                    # 3. RSI < 60 (Niet overgekocht)
                    if cp > e200.iloc[i] and e20.iloc[i] > e50.iloc[i] and rsi.iloc[i] < 60:
                        instap = cp
                        high_p = cp
                        sl = cp - (2.5 * atr.iloc[i]) # Ruime stop loss
                        pos = True
                        ticker_profit -= fee
                        totaal_trades += 1
                else:
                    # TRAILING STOP LOSS LOGICA
                    high_p = max(high_p, cp)
                    sl = max(sl, high_p - (2.5 * atr.iloc[i]))
                    
                    # VERKOOP CONDITIES:
                    # 1. Stop loss geraakt OF 2. Trendbreuk (Prijs onder EMA50)
                    if cp < sl or cp < e50.iloc[i]:
                        ticker_profit += (inzet * (cp / instap) - inzet) - fee
                        pos = False
            
            totaal_winst += ticker_profit
            
            # Actueel signaal genereren
            if cp > e200.iloc[-1] and e20.iloc[-1] > e50.iloc[-1] and rsi.iloc[-1] < 55:
                signalen.append(f"• `{t}`: 🟢 *KOOP* (€{cp:.2f})")
            elif pos and cp < e50.iloc[-1]:
                signalen.append(f"• `{t}`: 🔴 *EXIT* (€{cp:.2f})")

        except: continue

    # Rapportage
    rapport = [
        f"🤖 *BOT VERSIE: {VERSION}*",
        f"📅 {nu} | Sector: {label}",
        "----------------------------------",
        f"💰 *Netto Resultaat:* €{totaal_winst:,.2f}",
        f"📈 *Aantal Trades:* {totaal_trades}",
        f"🏦 *Kapitaal:* €{100000 + totaal_winst:,.2f}",
        "----------------------------------",
        "*ACTUELE SIGNALEN:*",
        "\n".join(signalen) if signalen else "Geen actie vereist."
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
    voer_backtest_v2(benelux_30, "BENELUX")
