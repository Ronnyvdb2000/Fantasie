import yfinance as yf
import pandas as pd
import os
import requests

# --- SAXO BOEKHOUDING CONFIGURATIE ---
START_KAPITAAL = 10000.0
INZET_PER_TRADE = 2000.0   # Maximaal 5 posities tegelijk
COMMISSIE = 2.0            # Saxo Bronze tarief
BEURSTAKS = 0.0035         # Belgische TOB
BELASTING_OPT_IN = 0.10    # Jouw 10% opt-in meerwaardereserve

def bereken_indicatoren(p):
    # De 9/21 EMA en 200 EMA logica uit jouw Bot 01
    ema9 = p.ewm(span=9, adjust=False).mean()
    ema21 = p.ewm(span=21, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    return pd.DataFrame({'p': p, 'ema9': ema9, 'ema21': ema21, 'ema200': ema200})

def run_saxo_boekhouding(bestandsnaam):
    # 1. Tickers laden
    with open(bestandsnaam, 'r') as f:
        tickers = [t.strip().upper() for t in f.read().replace('\n', ',').replace(';', ',').split(',') if t.strip()]

    # 2. Data ophalen
    raw_data = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
    
    cash = START_KAPITAAL
    fiscale_reserve = 0.0
    portefeuille = {} 
    dagen = raw_data.index[-252:] # Laatste jaar backtesten

    for d in dagen:
        # A. VERKOOP (Exit op EMA crossover)
        for t in list(portefeuille.keys()):
            df_t = bereken_indicatoren(raw_data['Close'][t].ffill())
            row = df_t.loc[d]
            pos = portefeuille[t]
            
            # Verkoopconditie: EMA9 < EMA21
            if row['ema9'] < row['ema21'] or d == dagen[-1]:
                bruto = pos['aantal'] * float(row['p'])
                verkoop_kost = COMMISSIE + (bruto * BEURSTAKS)
                netto = bruto - verkoop_kost
                
                winst = netto - pos['totale_instap']
                if winst > 0:
                    inhouding = winst * BELASTING_OPT_IN
                    fiscale_reserve += inhouding
                    winst -= inhouding
                
                cash += (pos['totale_instap'] + winst)
                del portefeuille[t]

        # B. AANKOOP (Entry op EMA crossover + EMA200 filter)
        kost_per_aankoop = INZET_PER_TRADE + COMMISSIE + (INZET_PER_TRADE * BEURSTAKS)
        
        for t in tickers:
            if t in portefeuille or cash < kost_per_aankoop: continue
            
            df_t = bereken_indicatoren(raw_data['Close'][t].ffill())
            if d not in df_t.index: continue
            
            row = df_t.loc[d]
            prev_row = df_t.shift(1).loc[d]
            
            # De Hyper T logica: EMA9 kruist EMA21 naar boven EN prijs > EMA200
            if row['p'] > row['ema200']:
                if row['ema9'] > row['ema21'] and prev_row['ema9'] <= prev_row['ema21']:
                    portefeuille[t] = {
                        'aantal': INZET_PER_TRADE / row['p'],
                        'totale_instap': kost_per_aankoop
                    }
                    cash -= kost_per_aankoop

    # Eindresultaat berekenen
    totaal_waarde = cash + sum([pos['aantal'] * raw_data['Close'][t].iloc[-1] for t, pos in portefeuille.items()])
    
    print(f"--- RESULTAAT SAXO BOEKHOUDING ---")
    print(f"Startkapitaal: €{START_KAPITAAL:,.2f}")
    print(f"Eindwaarde:    €{totaal_waarde:,.2f}")
    print(f"Netto Winst:   €{totaal_waarde - START_KAPITAAL:,.2f}")
    print(f"Fisc. Reserve: €{fiscale_reserve:,.2f}")

if __name__ == "__main__":
    run_saxo_boekhouding("tickers_01.txt")
