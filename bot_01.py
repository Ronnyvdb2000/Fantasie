import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATIE ---
START_KAPITAAL = 100000.0
INZET_PER_TRADE = 2500.0
COMMISSIE = 2.0           # Saxo
BEURSTAKS = 0.0035        # TOB
BELASTING_OPT_IN = 0.10    # 10% reserve

def bereken_alle_indicatoren(df):
    """Behoudt exact jouw berekeningen uit de originele bot"""
    p = df.ffill().astype(float)
    # Trend MA's
    sma50 = p.rolling(window=50).mean()
    sma200 = p.rolling(window=200).mean()
    sma20 = p.rolling(window=20).mean()
    # Hyper EMA's
    ema9 = p.ewm(span=9, adjust=False).mean()
    ema21 = p.ewm(span=21, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    # MR Logica
    delta = p.diff()
    gain = delta.where(delta > 0, 0).rolling(window=2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    ma5 = p.rolling(window=5).mean()
    
    return pd.DataFrame({
        'p': p, 'sma50': sma50, 'sma200': sma200, 'sma20': sma20,
        'ema9': ema9, 'ema21': ema21, 'ema200_e': ema200, 
        'rsi2': rsi2, 'ma5': ma5
    })

def run_geavanceerde_boekhouding(bestandsnaam):
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()]

    raw_close = yf.download(tickers, period="2y", progress=False, auto_adjust=True)['Close']
    
    cash = START_KAPITAAL
    fiscale_reserve = 0.0
    portefeuille = {} 
    dagen = raw_close.index[-252:]

    for d in dagen:
        # 1. VERKOOP LOGICA (Eerst cash vrijmaken)
        for t in list(portefeuille.keys()):
            df_t = bereken_alle_indicatoren(raw_close[t])
            if d not in df_t.index: continue
            row = df_t.loc[d]
            pos = portefeuille[t]
            
            sell = False
            # Exit regels per type uit jouw bot
            if pos['type'] == 'T' and row['sma50'] < row['sma200']: sell = True
            elif pos['type'] == 'S' and row['sma20'] < row['sma50']: sell = True
            elif pos['type'] in ['HT', 'HS'] and row['ema9'] < row['ema21']: sell = True
            elif pos['type'] == 'MR' and row['p'] > row['ma5']: sell = True
            
            if sell or d == dagen[-1]:
                opbrengst = pos['aantal'] * row['p']
                kosten = COMMISSIE + (opbrengst * BEURSTAKS)
                netto = opbrengst - kosten
                winst = netto - pos['instap_kost']
                
                if winst > 0:
                    inhouding = winst * BELASTING_OPT_IN
                    fiscale_reserve += inhouding
                    winst -= inhouding
                
                cash += (pos['instap_kost'] + winst)
                del portefeuille[t]

        # 2. AANKOOP LOGICA (Volgorde: T > S > HT > HS > MR)
        kost = INZET_PER_TRADE + COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
        
        # We scannen de strategieën in jouw gewenste volgorde
        strategie_volgorde = ['T', 'S', 'HT', 'HS', 'MR']
        
        for strat in strategie_volgorde:
            for t in tickers:
                if t in portefeuille or cash < kost: continue
                
                df_t = bereken_alle_indicatoren(raw_close[t])
                if d not in df_t.index: continue
                row, prev = df_t.loc[d], df_t.shift(1).loc[d]
                
                gekocht = False
                if strat == 'T' and row['sma50'] > row['sma200'] and prev['sma50'] <= prev['sma200'] and row['p'] > row['ema200_e']:
                    gekocht = True
                elif strat == 'S' and row['sma20'] > row['sma50'] and prev['sma20'] <= prev['sma50'] and row['p'] > row['ema200_e']:
                    gekocht = True
                elif strat == 'HT' and row['ema9'] > row['ema21'] and prev['ema9'] <= prev['ema21'] and row['p'] > row['ema200_e']:
                    gekocht = True
                elif strat == 'HS' and row['ema9'] > row['ema21'] and prev['ema9'] <= prev['ema21']: # Geen filter
                    gekocht = True
                elif strat == 'MR' and 0 < row['rsi2'] < 30:
                    gekocht = True

                if gekocht:
                    portefeuille[t] = {'aantal': INZET_PER_TRADE / row['p'], 'instap_kost': kost, 'type': strat}
                    cash -= kost

    # Resultaat
    open_waarde = sum([p['aantal'] * raw_close[t].iloc[-1] for t, p in portefeuille.items()])
    totaal = cash + open_waarde
    print(f"--- 📊 EINDBALANS (Start €100k) ---")
    print(f"Netto Eindwaarde: €{totaal:,.2f}")
    print(f"Fiscale Reserve:  €{fiscale_reserve:,.2f}")
    print(f"Rendement:        {((totaal/START_KAPITAAL)-1)*100:.2f}%")

if __name__ == "__main__":
    run_geavanceerde_boekhouding("tickers_01.txt")
