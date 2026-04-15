import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATIE VOOR BOLERO ---
INZET = 2500.0
KOSTEN = 15.0 + (INZET * 0.0035)

def bereken_indicatoren_flexibel(df):
    p = df['Close'].ffill()
    # We gebruiken kortere gemiddelden om sneller op trends te reageren
    sma20 = p.rolling(window=20).mean()
    sma50 = p.rolling(window=50).mean()
    sma200 = p.rolling(window=200).mean()
    
    # RSI voor oversold situaties
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    
    # ATR voor een dynamische Stop Loss
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-p.shift()), abs(df['Low']-p.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    return p, sma20, sma50, sma200, rsi, atr

def backtest_snelle_kwaliteit(tickers):
    totaal_resultaat = 0
    trades_count = 0
    
    # Download data in bulk om snelheid te verhogen
    data_bulk = yf.download(tickers, period="2y", interval="1d", progress=False)['Close']
    
    for t in tickers:
        try:
            # Haal data per ticker
            ticker_data = yf.download(t, period="2y", progress=False)
            if len(ticker_data) < 200: continue
            
            p, s20, s50, s200, rsi, atr = bereken_indicatoren_flexibel(ticker_data)
            
            # We kijken naar de laatste 252 handelsdagen
            pos, instap, sl = False, 0, 0
            ticker_profit = 0
            
            for i in range(len(p)-252, len(p)):
                cp = p.iloc[i]
                
                # KOOP: SMA20 kruist SMA50 (Snel momentum) EN koers > SMA200
                if not pos:
                    if s20.iloc[i] > s50.iloc[i] and s20.iloc[i-1] <= s50.iloc[i-1]:
                        if cp > s200.iloc[i] * 0.98: # Iets meer speling t.o.v. de 200-lijn
                            instap, pos = cp, True
                            sl = cp - (2.5 * atr.iloc[i])
                            ticker_profit -= KOSTEN
                            trades_count += 1
                
                # VERKOOP: Harde Stop Loss of SMA20 kruist terug
                else:
                    if cp < sl or s20.iloc[i] < s50.iloc[i]:
                        ticker_profit += (INZET * (cp / instap) - INZET) - KOSTEN
                        pos = False
            
            totaal_resultaat += ticker_profit
        except: continue
        
    print(f"--- BACKTEST RESULTAAT ---")
    print(f"Totaal Profit/Verlies: €{totaal_resultaat:.2f}")
    print(f"Aantal Trades: {trades_count}")
    print(f"Netto Eindkapitaal: €{100000 + totaal_resultaat:.2f}")

# Start de test
benelux_30 = ["ASML.AS", "ADYEN.AS", "WKL.AS", "LOTB.BR", "ARGX.BR", "REN.AS", "DSFIR.AS", "IMCD.AS", "AZE.BR", "MELE.BR", "SOF.BR", "ACKB.BR", "KINE.BR", "UCB.BR", "DIE.BR", "BFIT.AS", "VGP.BR", "WDP.BR", "AD.AS", "HEIA.AS", "BESI.AS", "ALFEN.AS", "GLPG.AS", "EURN.BR", "ELI.BR", "BAR.BR", "ENX.AS", "NN.AS", "AGS.BR", "RAND.AS"]
backtest_snelle_kwaliteit(benelux_30)
