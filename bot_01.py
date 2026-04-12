import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv
import requests

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# --- CONFIGURATIE ---
START_KAPITAAL = 100000.0
INZET_PER_TRADE = 2500.0
COMMISSIE = 2.0
BEURSTAKS = 0.0035
BELASTING_OPT_IN = 0.10

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=15)
    except: pass

def bereken_indicatoren_bulk(df_close):
    """Berekent alle indicatoren voor alle tickers in één keer"""
    # Trend MA's
    sma50 = df_close.rolling(window=50).mean()
    sma200 = df_close.rolling(window=200).mean()
    sma20 = df_close.rolling(window=20).mean()
    # Hyper EMA's
    ema9 = df_close.ewm(span=9, adjust=False).mean()
    ema21 = df_close.ewm(span=21, adjust=False).mean()
    ema200_e = df_close.ewm(span=200, adjust=False).mean()
    # MR Logica (RSI2)
    delta = df_close.diff()
    gain = delta.clip(lower=0).rolling(window=2).mean()
    loss = (-delta.clip(upper=0)).rolling(window=2).mean()
    rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    ma5 = df_close.rolling(window=5).mean()
    
    return {
        'p': df_close, 'sma50': sma50, 'sma200': sma200, 'sma20': sma20,
        'ema9': ema9, 'ema21': ema21, 'ema200_e': ema200_e, 
        'rsi2': rsi2, 'ma5': ma5
    }

def run_geavanceerde_boekhouding(bestandsnaam):
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    stuur_telegram(f"🚀 Start simulatie voor {len(tickers)} tickers...")

    # STAP 1: Bulk download (Verhelpt de 35 minuten wachttijd)
    data_raw = yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
    ind = bereken_indicatoren_bulk(data_raw)
    
    cash = START_KAPITAAL
    fiscale_reserve = 0.0
    portefeuille = {} 
    dagen = data_raw.index[-252:]

    # STAP 2: Simulatie loop
    for d in dagen:
        # VERKOOP
        for t in list(portefeuille.keys()):
            pos = portefeuille[t]
            cp = ind['p'][t].loc[d]
            
            sell = False
            if pos['type'] == 'T' and ind['sma50'][t].loc[d] < ind['sma200'][t].loc[d]: sell = True
            elif pos['type'] == 'S' and ind['sma20'][t].loc[d] < ind['sma50'][t].loc[d]: sell = True
            elif pos['type'] in ['HT', 'HS'] and ind['ema9'][t].loc[d] < ind['ema21'][t].loc[d]: sell = True
            elif pos['type'] == 'MR' and cp > ind['ma5'][t].loc[d]: sell = True
            
            if sell or d == dagen[-1]:
                bruto = pos['aantal'] * cp
                netto = bruto - (COMMISSIE + (bruto * BEURSTAKS))
                winst = netto - pos['instap_kost']
                if winst > 0:
                    fiscale_reserve += (winst * BELASTING_OPT_IN)
                    winst *= (1 - BELASTING_OPT_IN)
                cash += (pos['instap_kost'] + winst)
                del portefeuille[t]

        # AANKOOP (Prioriteit: T > S > HT > HS > MR)
        kost = INZET_PER_TRADE + COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
        for strat in ['T', 'S', 'HT', 'HS', 'MR']:
            for t in tickers:
                if t in portefeuille or cash < kost or d not in ind['p'][t].index: continue
                
                cp = ind['p'][t].loc[d]
                idx = ind['p'][t].index.get_loc(d)
                if idx == 0: continue
                
                gekocht = False
                # Traag
                if strat == 'T' and ind['sma50'][t].iloc[idx] > ind['sma200'][t].iloc[idx] and ind['sma50'][t].iloc[idx-1] <= ind['sma200'][t].iloc[idx-1]:
                    if cp > ind['ema200_e'][t].iloc[idx]: gekocht = True
                # Snel
                elif strat == 'S' and ind['sma20'][t].iloc[idx] > ind['sma50'][t].iloc[idx] and ind['sma20'][t].iloc[idx-1] <= ind['sma50'][t].iloc[idx-1]:
                    if cp > ind['ema200_e'][t].iloc[idx]: gekocht = True
                # Hyper T
                elif strat == 'HT' and ind['ema9'][t].iloc[idx] > ind['ema21'][t].iloc[idx] and ind['ema9'][t].iloc[idx-1] <= ind['ema21'][t].iloc[idx-1]:
                    if cp > ind['ema200_e'][t].iloc[idx]: gekocht = True
                # Hyper S
                elif strat == 'HS' and ind['ema9'][t].iloc[idx] > ind['ema21'][t].iloc[idx] and ind['ema9'][t].iloc[idx-1] <= ind['ema21'][t].iloc[idx-1]:
                    gekocht = True
                # Mean Reversion
                elif strat == 'MR' and 0 < ind['rsi2'][t].iloc[idx] < 30:
                    gekocht = True

                if gekocht:
                    portefeuille[t] = {'aantal': INZET_PER_TRADE / cp, 'instap_kost': kost, 'type': strat}
                    cash -= kost

    # FINALE BEREKENING
    open_w = sum([p['aantal'] * data_raw[t].iloc[-1] for t, p in portefeuille.items()])
    totaal = cash + open_w
    
    rapport = (
        f"🏁 *SIMULATIE VOLTOOID*\n\n"
        f"💰 Eindwaarde: *€{totaal:,.2f}*\n"
        f"🏦 Reserve (10%): €{fiscale_reserve:,.2f}\n"
        f"📈 Rendement: *{((totaal/START_KAPITAAL)-1)*100:.2f}%*\n"
        f"💵 Cash over: €{cash:,.2f}"
    )
    stuur_telegram(rapport)

if __name__ == "__main__":
    run_geavanceerde_boekhouding("tickers_01.txt")
