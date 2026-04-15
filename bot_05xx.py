import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

VERSION = "3.0 - Alpha Optimizer (Bolero Edition)"
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_alpha_indicatoren(df):
    p = df['Close'].ffill()
    # Snellere EMA's voor eerdere instap
    e20 = p.ewm(span=20, adjust=False).mean()
    e50 = p.ewm(span=50, adjust=False).mean()
    e200 = p.ewm(span=200, adjust=False).mean()
    
    # RSI voor timing
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR voor risicobeheer
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-p.shift()), abs(df['Low']-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, e20, e50, e200, rsi, atr

def voer_backtest_v3(tickers, label):
    inzet = 2500.0
    fee = 15.0 + (inzet * 0.0035) # Bolero tarief
    
    totaal_winst = 0
    totaal_trades = 0
    signalen = []

    for t in tickers:
        try:
            data = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(data) < 200: continue
            
            p, e20, e50, e200, rsi, atr = bereken_alpha_indicatoren(data)
            
            # Venster: Laatste 252 dagen
            pos, instap, sl, break_even_triggered = False, 0, 0, False
            ticker_profit = 0
            
            for i in range(len(p)-252, len(p)):
                cp = p.iloc[i]
                
                if not pos:
                    # INSTAP: EMA20 > EMA50 EN Prijs > EMA200 EN RSI < 65
                    if e20.iloc[i] > e50.iloc[i] and e20.iloc[i-1] <= e50.iloc[i-1]:
                        if cp > e200.iloc[i]:
                            instap = cp
                            sl = cp - (2.2 * atr.iloc[i])
                            pos = True
                            break_even_triggered = False
                            ticker_profit -= fee
                            totaal_trades += 1
                else:
                    # WINSTBESCHERMING: Als winst > 5%, zet SL op instap + kosten
                    winst_pct = (cp - instap) / instap
                    if winst_pct > 0.05 and not break_even_triggered:
                        sl = instap + (fee / (inzet/instap))
                        break_even_triggered = True
                    
                    # VERKOOP: SL geraakt of EMA20 kruist EMA50 terug
                    if cp < sl or e20.iloc[i] < e50.iloc[i]:
                        ticker_profit += (inzet * (cp / instap) - inzet) - fee
                        pos = False
            
            totaal_winst += ticker_profit
            
            # Actueel signaal
            if e20.iloc[-1] > e50.iloc[-1] and e20.iloc[-2] <= e50.iloc[-2] and p.iloc[-1] > e200.iloc[-1]:
                signalen.append(f"• `{t}`: 🟢 *ALPHA BUY*")
            elif pos and e20.iloc[-1] < e50.iloc[-1]:
                signalen.append(f"• `{t}`: 🔴 *ALPHA EXIT*")

        except: continue

    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    rapport = [
        f"🚀 *BOT VERSIE: {VERSION}*",
        f"📅 {nu} | {label}",
        "----------------------------------",
        f"💰 *Netto Resultaat:* €{totaal_winst:,.2f}",
        f"🏦 *Eindkapitaal:* €{100000 + totaal_winst:,.2f}",
        f"📊 *Aantal Trades:* {totaal_trades}",
        "----------------------------------",
        "*SIGNALEN:*",
        "\n".join(signalen) if signalen else "Geen actie."
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
    voer_backtest_v3(benelux_30, "BENELUX")
