import yfinance as yf
import pandas as pd
import os
import requests
import time
from datetime import datetime

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
START_KAPITAAL = 10000.0
INZET_PER_TRADE = 2000.0  # Verhoogd naar 2000 euro
COMMISSIE = 1.25          # IBKR Tiered tarief
BEURSTAKS = 0.0035        # Belgische TOB
BELASTING = 0.10          # 10% op netto winst

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_indicatoren(df, ticker):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            p = df['Close'][ticker].ffill().astype(float)
        else:
            p = df['Close'].ffill().astype(float)
        
        ema200 = p.ewm(span=200, adjust=False).mean()
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))
        
        return pd.DataFrame({
            'p': p, 'rsi2': rsi2, 'ema200': ema200,
            'ma5': p.rolling(5).mean(), 
            'ema9': p.ewm(span=9, adjust=False).mean(), 
            'ema21': p.ewm(span=21, adjust=False).mean()
        }, index=df.index)
    except: return None

def laad_alle_tickers():
    alle = set()
    for i in range(1, 10):
        fname = f"tickers_0{i}.txt"
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                for t in f.read().replace('\n', ',').replace(';', ',').split(','):
                    ticker = t.strip().upper()
                    if ticker: alle.add(ticker)
    return list(alle)

def run_multi_lijst_simulatie():
    tickers = laad_alle_tickers()
    if not tickers: return

    cash = START_KAPITAAL
    portefeuille = {} 
    stats = {"MR_winst": 0.0, "HS_winst": 0.0, "MR_trades": 0, "HS_trades": 0}

    try:
        # Download data voor alle tickers
        raw_data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        data = {t: bereken_indicatoren(raw_data, t) for t in tickers if bereken_indicatoren(raw_data, t) is not None}
    except: return

    if not data: return
    dagen = data[next(iter(data))].index[-252:]

    for d in dagen:
        # 1. VERKOOP CHECK
        for t in list(portefeuille.keys()):
            df = data[t]
            if d not in df.index: continue
            row = df.loc[d]
            cp = float(row['p'])
            pos = portefeuille[t]
            
            sell = (pos['type']=="MR" and cp > row['ma5']) or (pos['type']=="HS" and row['ema9'] < row['ema21'])
            
            if sell or d == dagen[-1]:
                bruto = pos['aantal'] * cp
                verkoop_kost = COMMISSIE + (bruto * BEURSTAKS)
                netto_cash = bruto - verkoop_kost
                
                winst_verlies = netto_cash - pos['totale_instap']
                if winst_verlies > 0: winst_verlies *= (1 - BELASTING)
                
                cash += (pos['totale_instap'] + winst_verlies)
                stats[f"{pos['type']}_winst"] += winst_verlies
                stats[f"{pos['type']}_trades"] += 1
                del portefeuille[t]

        # 2. AANKOOP CHECK
        instap_kost = COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
        totale_kost = INZET_PER_TRADE + instap_kost

        for t, df in data.items():
            if d not in df.index or t in portefeuille: continue
            if cash < totale_kost: break 
            
            row = df.loc[d]
            cp = float(row['p'])
            
            # TREND FILTER + EMA CROSSOVER OF EXTREME DIP
            if cp > row['ema200']:
                etype = None
                # Prioriteit aan HS (Trend) omdat deze winstgevend was
                if row['ema9'] > row['ema21'] and df['ema9'].loc[:d].iloc[-2] <= df['ema21'].loc[:d].iloc[-2]:
                    etype = "HS"
                # MR alleen bij extreme paniek (RSI2 < 5)
                elif row['rsi2'] < 5:
                    etype = "MR"

                if etype:
                    portefeuille[t] = {
                        'aantal': INZET_PER_TRADE / cp,
                        'type': etype,
                        'totale_instap': totale_kost
                    }
                    cash -= totale_kost

    # Live signalen vandaag
    live = []
    for t, df in data.items():
        if len(df) < 2: continue
        last, prev = df.iloc[-1], df.iloc[-2]
        if last['p'] > last['ema200']:
            if last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21']:
                live.append(f"🚀 `{t}` (HS)")
            elif last['rsi2'] < 5:
                live.append(f"📉 `{t}` (MR)")

    rapport = (
        f"🌍 *GLOBAL POT V2 (Inzet €2000)*\n"
        f"Eindkapitaal: *€{cash:,.2f}*\n"
        f"--------------------------\n"
        f"📉 MR: €{stats['MR_winst']:,.2f} ({stats['MR_trades']}x)\n"
        f"🚀 HS: €{stats['HS_winst']:,.2f} ({stats['HS_trades']}x)\n\n"
        f"*SIGNALEN:* " + (", ".join(live[:15]) if live else "Geen")
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    run_multi_lijst_simulatie()
