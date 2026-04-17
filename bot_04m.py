import yfinance as yf
import os
import time

# --- CONFIGURATIE ---
# De bot zoekt nu slim naar dit bestand, ongeacht hoofdletters
ZOEK_BESTAND = "tickers_04a.txt" 
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
        debt_to_equity = info.get('debtToEquity', 100) 
        profit_margin = info.get('profitMargins', 0)
        name = info.get('shortName', ticker_symbol)

        # Munger Criteria Filters
        is_profitable = roe > 0.15          # ROE > 15%
        is_safe = debt_to_equity < 60       # Schuld < 60%
        has_moat = profit_margin > 0.10     # Marge > 10%

        if is_profitable and is_safe and has_moat:
            print(f"✅ GOEDGEKEURD: {ticker_symbol: <8} | {name: <25} | ROE: {roe:.1%} | Debt/Eq: {debt_to_equity:.1f}")
            return True
        return False
            
    except Exception:
        return False

def main():
    print(f"--- START MUNGER FILTERING (bot_04m.py) ---")
    
    # 1. Vind het bronbestand ongeacht hoofdletters
    bestanden = os.listdir('.')
    actueel_bestand = next((f for f in bestanden if f.lower() == ZOEK_BESTAND.lower()), None)

    if not actueel_bestand:
        print(f"❌ FOUT: Bestand '{ZOEK_BESTAND}' niet gevonden in: {bestanden}")
        return

    print(f"📖 Bronbestand gevonden: '{actueel_bestand}'")

    # 2. Tickers inlezen
    with open(actueel_bestand, 'r') as f:
        content = f.read().replace('\n', ',')
        alle_tickers = [t.strip().upper() for t in content.split(',') if t.strip()]

    print(f"Totaal aantal tickers te beoordelen: {len(alle_tickers)}")
    
    kwaliteits_tickers = []

    # 3. Analyse loop
    for symbol in alle_tickers:
        if munger_keuring(symbol):
            kwaliteits_tickers.append(symbol)
        time.sleep(0.5) # Voorkom blokkade door Yahoo

    # 4. Opslaan naar tickers_04m.txt
    if kwaliteits_tickers:
        with open(TARGET_FILE, 'w') as f:
            f.write(", ".join(kwaliteits_tickers))
        print(f"\n--- ANALYSE VOLTOOID ---")
        print(f"Geselecteerd: {len(kwaliteits_tickers)} van de {len(alle_tickers)}")
        print(f"Resultaat opgeslagen in: {TARGET_FILE}")
    else:
        print("\n⚠️ Geen enkel aandeel voldeed aan de eisen.")

if __name__ == "__main__":
    main()
