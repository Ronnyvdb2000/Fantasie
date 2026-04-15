import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

VERSION = "5.0 - Mean Reversion King (Bolero Optimised)"
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=20)
    except: pass

def bereken_v5_indicatoren(df):
    p = df['Close'].ffill()
    ema20 = p.ewm(span=20, adjust=False).mean()
    ema50 = p.ewm(span=50, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    
    # RSI voor Mean Reversion
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR voor risicobeheer
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-p.shift()), abs(df['Low']-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, ema20, ema50, ema200, rsi, atr

def voer_backtest_v5(tickers, label):
    inzet = 2500.0
    fee = 15.0 + (inzet * 0.0035) 
    
    totaal_winst = 0
    totaal_trades = 0
    signalen = []

    for t in tickers:
        try:
            data = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(data) < 200: continue
            
            p, e20, e50, e200, rsi, atr = bereken_v5_indicatoren(data)
            
            pos, instap, sl = False, 0, 0
            ticker_profit = 0
            
            # Start backtest (252 dagen)
            for i in range(len(p)-252, len(p)):
                cp = p.iloc[i]
                
                if not pos:
                    # INSTAP V5: MEAN REVERSION
                    # Prijs moet onder EMA20 liggen (dip) maar BOVEN EMA200 (lange trend)
                    # RSI moet aangeven dat het 'oversold' is (< 40)
                    if cp > e200.iloc[i] and cp < e20.iloc[i] and rsi.iloc[i] < 40:
                        instap = cp
                        sl = cp - (3 * atr.iloc[i])
                        pos = True
                        ticker_profit -= fee
                        totaal_trades += 1
                else:
                    # VERKOOP: Zodra we weer boven het gemiddelde (EMA50) komen = Winst genomen
                    # Of de stop loss wordt geraakt
                    if cp > e50.iloc[i] or cp < sl:
                        ticker_profit += (inzet * (cp / instap) - inzet) - fee
                        pos = False
            
            totaal_winst += ticker_profit
            
            # Actueel signaal
            if p.iloc[-1] > e200.iloc[-1] and p.iloc[-1] < e20.iloc[-1] and rsi.iloc[-1] < 42:
                signalen.append(f"• `{t}`: 💎 *KWALITEIT DIP* (RSI: {rsi.iloc[-1]:.1f})")

        except: continue

    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    rapport = [
        f"👑 *BOT VERSIE: {VERSION}*",
        f"📅 {nu} | {label}",
        "----------------------------------",
        f"💰 *Netto Winst:* €{totaal_winst:,.2f}",
        f"🏦 *Totaal:* €{100000 + totaal_winst:,.2f}",
        f"📊 *Trades:* {totaal_trades}",
        "----------------------------------",
        "*KOOPKANSEN (DIPS):*",
        "\n".join(signalen) if signalen else "Wachten op een dip in kwaliteit."
    ]
    stuur_telegram("\n".join(rapport))

if __name__ == "__main__":
    benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
    voer_backtest_v5(benelux_30, "BENELUX")
