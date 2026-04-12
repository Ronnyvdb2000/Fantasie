import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
START_KAPITAAL = 10000.0
INZET_PER_TRADE = 1000.0
COMMISSIE = 1.0       # Interactive Brokers schatting
BEURSTAKS = 0.0035    # 0,35% TOV België
BELASTING = 0.10      # Nieuwe Belgische Meerwaardebelasting 2026

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def bereken_indicatoren(df):
    p = df['Close'].ffill().astype(float)
    # RSI2
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / (loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    # MA's & EMA's
    ma5 = p.rolling(5).mean()
    ema9 = p.ewm(span=9, adjust=False).mean()
    ema21 = p.ewm(span=21, adjust=False).mean()
    return pd.DataFrame({'p': p, 'rsi2': rsi2, 'ma5': ma5, 'ema9': ema9, 'ema21': ema21})

def run_simulatie(tickers):
    pots = {"MR": 4000.0, "HS": 4000.0, "RESERVE": 2000.0}
    winsten = {"MR": 0.0, "HS": 0.0}
    portefeuille = {} 
    live_signalen = []

    # 1. Data ophalen
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(df) > 260: data[t] = bereken_indicatoren(df)
        except: continue

    if not data: return "Geen data."

    # 2. Backtest over 1 jaar (252 handelsdagen)
    dagen = data[next(iter(data))].index[-252:]

    for d in dagen:
        for t, df in data.items():
            if d not in df.index: continue
            row = df.loc[d]
            cp = float(row['p'])

            # CHECK VERKOOP
            if t in portefeuille:
                pos = portefeuille[t]
                sell = (pos['type']=="MR" and cp > row['ma5']) or (pos['type']=="HS" and row['ema9'] < row['ema21'])
                
                if sell or d == dagen[-1]:
                    bruto_winst = (INZET_PER_TRADE * (cp / pos['prijs'])) - INZET_PER_TRADE
                    exit_kosten = COMMISSIE + (INZET_PER_TRADE * (cp / pos['prijs']) * BEURSTAKS)
                    netto = bruto_winst - pos['taks_entry'] - exit_kosten
                    
                    # 10% belasting op winst
                    if netto > 0: netto *= (1 - BELASTING)
                    
                    pots[pos['type']] += (INZET_PER_TRADE + netto)
                    winsten[pos['type']] += netto
                    del portefeuille[t]

            # CHECK KOOP (Alleen als niet in portefeuille)
            else:
                entry_type = None
                if row['rsi2'] < 30 and pots["MR"] >= INZET_PER_TRADE:
                    entry_type = "MR"
                elif row['ema9'] > row['ema21'] and pots["HS"] >= INZET_PER_TRADE:
                    # Alleen kopen als MR niet al getriggerd is voor dit aandeel
                    entry_type = "HS"

                if entry_type:
                    taks_entry = COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
                    portefeuille[t] = {'prijs': cp, 'type': entry_type, 'taks_entry': taks_entry}
                    pots[entry_type] -= INZET_PER_TRADE

    # 3. Actuele signalen check
    for t, df in data.items():
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['rsi2'] < 30: 
            live_signalen.append(f"📉 `{t}`: MR Dip (RSI2: {last['rsi2']:.1f})")
        if last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21']:
            live_signalen.append(f"🚀 `{t}`: HS Cross")

    return pots, winsten, live_signalen

if __name__ == "__main__":
    with open("tickers_01.txt", 'r') as f:
        lijst = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]
    
    p, w, sig = run_simulatie(lijst)
    totaal = p['MR'] + p['HS'] + p['RESERVE']
    
    rapport = (
        f"🏦 *THEORETISCH JAAROVERZICHT*\n"
        f"Start: €10.000 -> *Eind: €{totaal:,.2f}*\n"
        f"--------------------------\n"
        f"📉 MR Pot (€4k): €{p['MR']:,.2f} (Winst: €{w['MR']:,.2f})\n"
        f"🚀 HS Pot (€4k): €{p['HS']:,.2f} (Winst: €{w['HS']:,.2f})\n"
        f"💰 Reserve: €{p['RESERVE']:,.0f}\n\n"
        f"*LIVE SIGNALEN VANDAAG:*\n" + ("\n".join(sig) if sig else "Geen actie.")
    )
    stuur_telegram(rapport)
