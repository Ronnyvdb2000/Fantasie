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
        t_obj = yf.Ticker(ticker_symbol)
        info = t_obj.info
        
        if not info or 'returnOnEquity' not in info:
            return False

        # Fundamentele parameters (Munger-check)
        roe = info.get('returnOnEquity', 0)
        debt_to_equity = info.get('debtToEquity', 100) # Hoger dan 100 is meestal slecht
        profit_margin = info.get('profitMargins', 0)
        name = info.get('shortName', ticker_symbol)

        # Munger Criteria Filters: Kwaliteit boven alles
        is_profitable = roe > 0.15          # Rendement op eigen vermogen > 15%
        is_safe = debt_to_equity < 60       # Schuldgraad onder 60% (conservatief)
        has_moat = profit_margin > 0.10     # Winstmarge > 10% (gezonde business)

        if is_profitable and is_safe and has_moat:
            print(f"✅ GOEDGEKEURD: {ticker_symbol: <8} | {name: <25} | ROE: {roe:.1%} | Debt/Eq: {debt_to_equity:.1f}")
            return True
        return False
            
    except Exception:
        return False

def main():
    print(f"--- START MUNGER FILTERING (bot_04m.py) ---")
    
    # Check welke bestanden er echt zijn (voor de GitHub Action logs)
    bestanden_in_map = os.listdir('.')
    print(f"Bestanden in huidige map: {bestanden_in_map}")
    
    # Zoek naar het bronbestand (case-insensitive check)
    actueel_bronbestand = None
    for f in bestanden_in_map:
        if f.lower() == SOURCE_FILE.lower():
            actueel_bronbestand = f
            break

    if not actueel_bronbestand:
        print(f"❌ FOUT: Bronbestand '{SOURCE_FILE}' niet gevonden!")
        return

    # 1. Tickers inlezen uit het gevonden bestand
    with open(actueel_bronbestand, 'r') as f:
        content = f.read().replace('\n', ',')
        # Schoon de lijst op: hoofdletters en geen spaties
        alle_tickers = [t.strip().upper() for t in content.split(',') if t.strip()]

    print(f"Totaal aantal tickers te beoordelen: {len(alle_tickers)}")
    
    kwaliteits_tickers = []

    # 2. Beoordeling per ticker
    for symbol in alle_tickers:
        # Filter op Yahoo format (moet eindigen op .BR of .AS voor Benelux)
        if munger_keuring(symbol):
            kwaliteits_tickers.append(symbol)
        
        # Slaap 0.5 sec om Rate Limiting van Yahoo te voorkomen
        time.sleep(0.5)

    # 3. Opslaan naar tickers_04m.txt
    if kwaliteits_tickers:
        with open(TARGET_FILE, 'w') as f:
            f.write(", ".join(kwaliteits_tickers))
        print(f"\n--- ANALYSE VOLTOOID ---")
        print(f"Geselecteerd: {len(kwaliteits_tickers)} van de {len(alle_tickers)} tickers.")
        print(f"Nieuwe lijst opgeslagen in: {TARGET_FILE}")
    else:
        print("\n⚠️ Geen enkel aandeel voldeed aan de Munger criteria. Bestand niet aangemaakt.")

if __name__ == "__main__":
    main()
