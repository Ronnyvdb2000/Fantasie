import yfinance as yf
import os
import time

# --- CONFIGURATIE ---
SOURCE_FILE = "tickers_04a.txt"
TARGET_FILE = "tickers_04m.txt"

def munger_keuring(ticker_symbol):
    """
    Beoordeelt een aandeel op fundamentele Munger-kwaliteit.
    """
    try:
        # We halen alleen de noodzakelijke info op
        t_obj = yf.Ticker(ticker_symbol)
        info = t_obj.info
        
        if not info:
            return False

        # Fundamentele parameters
        roe = info.get('returnOnEquity', 0)
        debt_to_equity = info.get('debtToEquity', 100) # 100 = slecht/hoog risico
        profit_margin = info.get('profitMargins', 0)
        name = info.get('shortName', ticker_symbol)

        # Munger Criteria Filters
        is_profitable = roe > 0.15          # Rendement op eigen vermogen > 15%
        is_safe = debt_to_equity < 60       # Schuldgraad onder 60%
        has_moat = profit_margin > 0.10     # Winstmarge > 10%

        if is_profitable and is_safe and has_moat:
            print(f"✅ GOEDGEKEURD: {ticker_symbol: <8} | {name: <25} | ROE: {roe:.1%} | Debt/Eq: {debt_to_equity:.1f}")
            return True
        else:
            # Optioneel: reden van afwijzing printen voor debug
            return False
            
    except Exception as e:
        print(f"⚠️ Fout bij {ticker_symbol}: {e}")
        return False

def main():
    print(f"--- START MUNGER FILTERING (bot_04m.py) ---")
    
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Bronbestand {SOURCE_FILE} niet gevonden!")
        return

    # 1. Tickers inlezen
    with open(SOURCE_FILE, 'r') as f:
        content = f.read().replace('\n', ',')
        alle_tickers = [t.strip().upper() for t in content.split(',') if t.strip()]

    print(f"Totaal aantal tickers te beoordelen: {len(alle_tickers)}")
    
    kwaliteits_tickers = []

    # 2. Beoordeling per ticker
    for symbol in alle_tickers:
        if munger_keuring(symbol):
            kwaliteits_tickers.append(symbol)
        
        # Kleine pauze om te voorkomen dat Yahoo de verbinding verbreekt
        time.sleep(0.5)

    # 3. Opslaan naar tickers_04m.txt (overschrijft bestaand bestand)
    with open(TARGET_FILE, 'w') as f:
        f.write(", ".join(kwaliteits_tickers))
        
    print(f"\n--- ANALYSE VOLTOOID ---")
    print(f"Geselecteerd: {len(kwaliteits_tickers)} van de {len(alle_tickers)} tickers.")
    print(f"Nieuwe lijst opgeslagen in: {TARGET_FILE}")

if __name__ == "__main__":
    main()
