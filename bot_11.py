import yfinance as yf
import pandas as pd
import os
import requests

# --- CONFIGURATIE (SAXO + 10% OPT-IN) ---
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
START_KAPITAAL = 10000.0
INZET_PER_TRADE = 2000.0
COMMISSIE = 2.0           # Saxo Bronze
BEURSTAKS = 0.0035        # TOB
BELASTING_OPT_IN = 0.10   # Jouw 10% opt-in voor meerwaardebelasting

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
        
        return pd.DataFrame({
            'p': p,
            'ema200': p.ewm(span=200, adjust=False).mean(),
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
    fiscale_reserve = 0.0  # De 10% opt-in pot
    portefeuille = {} 
    winst_stats = 0.0
    aantal_trades = 0

    try:
        raw_data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
        data = {t: bereken_indicatoren(raw_data, t) for t in tickers if bereken_indicatoren(raw_data, t) is not None}
    except: return

    if not data: return
    dagen = data[next(iter(data))].index[-252:]

    for d in dagen:
        # 1. VERKOOP LOGICA
        for t in list(portefeuille.keys()):
            row = data[t].loc[d]
            pos = portefeuille[t]
            
            if row['ema9'] < row['ema21'] or d == dagen[-1]:
                bruto = pos['aantal'] * float(row['p'])
                verkoop_kost = COMMISSIE + (bruto * BEURSTAKS)
                netto_opbrengst = bruto - verkoop_kost
                
                trade_resultaat = netto_opbrengst - pos['totale_instap']
                
                # De 10% opt-in berekening
                if trade_resultaat > 0:
                    inhouding = trade_resultaat * BELASTING_OPT_IN
                    fiscale_reserve += inhouding
                    trade_resultaat -= inhouding
                
                cash += (pos['totale_instap'] + trade_resultaat)
                winst_stats += trade_resultaat
                aantal_trades += 1
                del portefeuille[t]

        # 2. AANKOOP LOGICA (Alleen Trend / HS)
        totale_kost = INZET_PER_TRADE + COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)

        for t, df in data.items():
            if d not in df.index or t in portefeuille: continue
            if cash < totale_kost: break 
            
            row = df.loc[d]
            cp = float(row['p'])
            
            if cp > row['ema200'] and row['ema9'] > row['ema21']:
                prev_ema9 = df['ema9'].loc[:d].iloc[-2]
                prev_ema21 = df['ema21'].loc[:d].iloc[-2]
                
                if prev_ema9 <= prev_ema21:
                    portefeuille[t] = {
                        'aantal': INZET_PER_TRADE / cp,
                        'totale_instap': totale_kost
                    }
                    cash -= totale_kost

    rapport = (
        f"🌍 *SAXO BOT (10% OPT-IN)*\n"
        f"Beschikbaar Cash: *€{cash:,.2f}*\n"
        f"🏦 Fiscale Reserve (10%): €{fiscale_reserve:,.2f}\n"
        f"--------------------------\n"
        f"📈 Netto winst (na 10%): €{winst_stats:,.2f}\n"
        f"🔄 Aantal trades: {aantal_trades}\n"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    run_multi_lijst_simulatie()
