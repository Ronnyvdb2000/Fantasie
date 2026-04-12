import yfinance as yf
import pandas as pd
import os
import requests
import time
from datetime import datetime

# --- CONFIGURATIE (Geoptimaliseerd voor IBKR + België) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
INZET_PER_TRADE = 2500.0  # Hogere inzet = minder impact vaste kosten
COMMISSIE = 1.50          # IBKR tarief
BEURSTAKS = 0.0035        # Belgische TOB (0.35%)
BELASTING = 0.10          # Veiligheidsmarge meerwaarde

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
        if p.empty: return None
        
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))
        
        return pd.DataFrame({
            'p': p, 'rsi2': rsi2, 
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
                for t in f.read().replace('\n', ',').split(','):
                    if t.strip(): alle.add(t.strip().upper())
    return list(alle)

def run_multi_lijst_simulatie():
    tickers = laad_alle_tickers()
    if not tickers: return

    # RELEALISTISCHE BOEKHOUDING: Geen reserve, winst wordt herbelegd
    pots = {"MR": 5000.0, "HS": 5000.0}
    winsten = {"MR": 0.0, "HS": 0.0}
    portefeuille = {} 
    live_signalen = []

    try:
        raw_data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        data = {t: bereken_indicatoren(raw_data, t) for t in tickers if bereken_indicatoren(raw_data, t) is not None}
    except: return

    if not data: return
    dagen = data[next(iter(data))].index[-252:]

    for d in dagen:
        for t, df in data.items():
            if d not in df.index: continue
            row = df.loc[d]
            cp = float(row['p'])

            if t in portefeuille:
                pos = portefeuille[t]
                sell = (pos['type']=="MR" and cp > row['ma5']) or (pos['type']=="HS" and row['ema9'] < row['ema21'])
                if sell or d == dagen[-1]:
                    # Verkoop: Trek commissie en beurstaks af van eindwaarde
                    bruto_opbrengst = pos['aantal'] * cp
                    verkoop_kost = COMMISSIE + (bruto_opbrengst * BEURSTAKS)
                    netto_cash = bruto_opbrengst - verkoop_kost
                    
                    winst = netto_cash - pos['totale_investering']
                    if winst > 0: winst *= (1 - BELASTING)
                    
                    # Kapitaal vloeit terug naar de pot inclusief winst
                    pots[pos['type']] += (pos['totale_investering'] + winst)
                    winsten[pos['type']] += winst
                    del portefeuille[t]
            else:
                etype = None
                # Strengere MR-filter (RSI2 < 10) voor betere resultaten
                if row['rsi2'] < 10 and pots["MR"] >= INZET_PER_TRADE: etype = "MR"
                elif row['ema9'] > row['ema21'] and pots["HS"] >= INZET_PER_TRADE: etype = "HS"

                if etype:
                    instap_kost = COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
                    portefeuille[t] = {
                        'aantal': INZET_PER_TRADE / cp,
                        'type': etype,
                        'totale_investering': INZET_PER_TRADE + instap_kost
                    }
                    pots[etype] -= (INZET_PER_TRADE + instap_kost)

    # Signalen vandaag
    for t, df in data.items():
        if len(df) < 2: continue
        last, prev = df.iloc[-1], df.iloc[-2]
        if last['rsi2'] < 10: live_signalen.append(f"📉 `{t}` (Dip)")
        if last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21']:
            live_signalen.append(f"🚀 `{t}` (Trend)")

    totaal = pots['MR'] + pots['HS']
    stuur_telegram(
        f"🌍 *IBKR-SIMULATIE (01-09)*\n"
        f"Start: €10,000 | Nu: *€{totaal:,.2f}*\n"
        f"--------------------------\n"
        f"📉 MR Winst: €{winsten['MR']:,.2f}\n"
        f"🚀 HS Winst: €{winsten['HS']:,.2f}\n\n"
        f"*SIGNALEN:* " + (", ".join(live_signalen[:15]) if live_signalen else "Geen")
    )

if __name__ == "__main__":
    run_multi_lijst_simulatie()
