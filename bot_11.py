import yfinance as yf
import pandas as pd
import os
import requests

# --- CONFIGURATIE ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
START_KAPITAAL = 10000.0
INZET_PER_TRADE = 1000.0
COMMISSIE = 1.0
BEURSTAKS = 0.0035
BELASTING = 0.10

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_indicatoren(df):
    p = df['Close'].ffill().astype(float)
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / (loss + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    ma5 = p.rolling(5).mean()
    ema9 = p.ewm(span=9, adjust=False).mean()
    ema21 = p.ewm(span=21, adjust=False).mean()
    return pd.DataFrame({'p': p, 'rsi2': rsi2, 'ma5': ma5, 'ema9': ema9, 'ema21': ema21})

def laad_alle_tickers():
    alle_tickers = set()
    # Loop van 1 tot en met 9
    for i in range(1, 10):
        bestandsnaam = f"tickers_0{i}.txt"
        if os.path.exists(bestandsnaam):
            with open(bestandsnaam, 'r') as f:
                # Splits op komma's of nieuwe regels en maak schoon
                inhoud = f.read().replace('\n', ',').split(',')
                for t in inhoud:
                    ticker = t.strip().upper()
                    if ticker: alle_tickers.add(ticker)
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

    # Data ophalen voor alle unieke tickers
    data = {}
    print(f"Bezig met ophalen van {len(tickers)} tickers...")
    for t in tickers:
        try:
            df = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if len(df) > 260: data[t] = bereken_indicatoren(df)
        except: continue

    if not data: return

    # Backtest over 1 jaar
    dagen = data[next(iter(data))].index[-252:]

    for d in dagen:
        for t, df in data.items():
            if d not in df.index: continue
            row = df.loc[d]
            cp = float(row['p'])

            # EXIT CHECK
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

            # ENTRY CHECK (Alleen als aandeel nergens in portefeuille)
            else:
                etype = None
                if row['rsi2'] < 30 and pots["MR"] >= INZET_PER_TRADE:
                    etype = "MR"
                elif row['ema9'] > row['ema21'] and pots["HS"] >= INZET_PER_TRADE:
                    etype = "HS"

                if etype:
                    tentry = COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
                    portefeuille[t] = {'prijs': cp, 'type': etype, 't_entry': tentry}
                    pots[etype] -= INZET_PER_TRADE

    # Actuele signalen (Vandaag)
    for t, df in data.items():
        last = df.iloc[-1]
        if last['rsi2'] < 30: live_signalen.append(f"📉 `{t}`: MR Dip")
        if last['ema9'] > last['ema21'] and df['ema9'].iloc[-2] <= df['ema21'].iloc[-2]:
            live_signalen.append(f"🚀 `{t}`: HS Cross")

    totaal = pots['MR'] + pots['HS'] + pots['RESERVE']
    rapport = (
        f"🌍 *MULTI-LIST RAPPORT (01-09)*\n"
        f"Gescande tickers: {len(tickers)}\n"
        f"Eindkapitaal: *€{totaal:,.2f}*\n"
        f"--------------------------\n"
        f"📉 MR Winst: €{winsten['MR']:,.2f}\n"
        f"🚀 HS Winst: €{winsten['HS']:,.2f}\n\n"
        f"*SIGNALEN:* " + (", ".join(live_signalen[:15]) if live_signalen else "Geen")
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    run_multi_lijst_simulatie()
