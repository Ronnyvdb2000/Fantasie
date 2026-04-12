import yfinance as yf
import pandas as pd
import os
import requests
import time
from datetime import datetime

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
INZET_PER_TRADE = 1000.0
COMMISSIE = 1.0
BEURSTAKS = 0.0035
BELASTING = 0.10

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: 
        print(f"Telegram configuratie mist. Bericht: {bericht}")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: 
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
        if res.status_code != 200:
            print(f"Telegram fout: {res.text}")
    except Exception as e: 
        print(f"Fout bij verzenden: {e}")

def bereken_indicatoren(df, ticker):
    try:
        # Check of de data een Multi-Index heeft (standaard bij yfinance v0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df['Close'].columns:
                p = df['Close'][ticker].ffill().astype(float)
            else:
                return None
        else:
            p = df['Close'].ffill().astype(float)
            
        if p.empty or len(p) < 30:
            return None
            
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / (loss + 1e-10)
        rsi2 = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({
            'p': p, 
            'rsi2': rsi2, 
            'ma5': p.rolling(5).mean(), 
            'ema9': p.ewm(span=9, adjust=False).mean(), 
            'ema21': p.ewm(span=21, adjust=False).mean()
        }, index=df.index)
    except Exception as e:
        print(f"Indicator fout voor {ticker}: {e}")
        return None

def laad_alle_tickers():
    alle_tickers = set()
    for i in range(1, 10):
        bestandsnaam = f"tickers_0{i}.txt"
        if os.path.exists(bestandsnaam):
            with open(bestandsnaam, 'r') as f:
                inhoud = f.read().replace('\n', ',').replace(';', ',').split(',')
                for t in inhoud:
                    ticker = t.strip().upper()
                    if ticker and len(ticker) < 10:
                        alle_tickers.add(ticker)
    return list(alle_tickers)

def run_multi_lijst_simulatie():
    tickers = laad_alle_tickers()
    if not tickers:
        stuur_telegram("❌ Geen tickers gevonden in bestanden 01 t/m 09.")
        return

    pots = {"MR": 4000.0, "HS": 4000.0, "RESERVE": 2000.0}
    winsten = {"MR": 0.0, "HS": 0.0}
    portefeuille = {} 
    live_signalen = []
    data = {}

    print(f"Ophalen data voor {len(tickers)} tickers...")
    try:
        raw_data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        for t in tickers:
            df_ticker = bereken_indicatoren(raw_data, t)
            if df_ticker is not None:
                data[t] = df_ticker
    except Exception as e:
        stuur_telegram(f"⚠️ Yahoo Download Fout: {e}")
        return

    if not data:
        stuur_telegram("⚠️ Geen koersdata beschikbaar voor de tickers.")
        return

    # Backtest 252 dagen
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
                    bruto = (INZET_PER_TRADE * (cp / pos['prijs'])) - INZET_PER_TRADE
                    exit_k = COMMISSIE + (INZET_PER_TRADE * (cp / pos['prijs']) * BEURSTAKS)
                    netto = bruto - pos['t_entry'] - exit_k
                    if netto > 0: netto *= (1 - BELASTING)
                    pots[pos['type']] += (INZET_PER_TRADE + netto)
                    winsten[pos['type']] += netto
                    del portefeuille[t]
            else:
                etype = None
                if row['rsi2'] < 15 and pots["MR"] >= INZET_PER_TRADE:
                    etype = "MR"
                elif row['ema9'] > row['ema21'] and pots["HS"] >= INZET_PER_TRADE:
                    etype = "HS"

                if etype:
                    tentry = COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
                    portefeuille[t] = {'prijs': cp, 'type': etype, 't_entry': tentry}
                    pots[etype] -= INZET_PER_TRADE

    # Live signalen
    for t, df in data.items():
        if len(df) < 2: continue
        last = df.iloc[-1]
        if last['rsi2'] < 15: live_signalen.append(f"📉 `{t}` (MR)")
        if last['ema9'] > last['ema21'] and df['ema9'].iloc[-2] <= df['ema21'].iloc[-2]:
            live_signalen.append(f"🚀 `{t}` (HS)")

    totaal = pots['MR'] + pots['HS'] + pots['RESERVE']
    rapport = (
        f"🌍 *MULTI-LIST RAPPORT*\n"
        f"Gescande tickers: {len(data)}\n"
        f"Eindkapitaal: *€{totaal:,.2f}*\n"
        f"--------------------------\n"
        f"📉 MR: €{winsten['MR']:,.2f} | 🚀 HS: €{winsten['HS']:,.2f}\n\n"
        f"*SIGNALEN:* \n" + ("\n".join(live_signalen[:15]) if live_signalen else "Geen")
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    run_multi_lijst_simulatie()
