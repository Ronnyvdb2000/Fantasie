import yfinance as yf
import pandas as pd
import time

def munger_keuring(ticker_symbol):
    """
    Beoordeelt een aandeel op basis van Charlie Munger's criteria:
    1. ROE (Return on Equity) > 15%
    2. Debt-to-Equity < 0.6 (Niet te veel schulden)
    3. Net Profit Margin > 10% (Gezonde winstgevendheid)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Gegevens ophalen (met fallbacks naar 0 als data ontbreekt)
        roe = info.get('returnOnEquity', 0)
        debt_to_equity = info.get('debtToEquity', 100) # 100 is slecht
        profit_margin = info.get('profitMargins', 0)
        name = info.get('longName', ticker_symbol)
        
        # De Munger Filters
        is_profitable = roe > 0.15
        is_safe = debt_to_equity < 60  # Ratio onder 60%
        has_moat = profit_margin > 0.10
        
        if is_profitable and is_safe and has_moat:
            print(f"✅ {ticker_symbol} ({name}) GOEDGEKEURD")
            print(f"   - ROE: {roe:.2%}, Debt/Eq: {debt_to_equity:.2f}, Margin: {profit_margin:.2%}")
            return True
        else:
            print(f"❌ {ticker_symbol} AFGEWEZEN (ROE: {roe:.2%}, Schuld: {debt_to_equity})")
            return False
            
    except Exception as e:
        print(f"⚠️ Kon {ticker_symbol} niet beoordelen: {e}")
        return False

def main():
    # Lijst met ruwe tickers (bijv. Brussel en Amsterdam)
    # Voeg hier je volledige lijst toe of lees in vanuit een bestand
    input_bestand = "benelux_all.txt" 
    output_bestand = "tickers_benelux_munger.txt"
    
    with open(input_bestand, 'r') as f:
        content = f.read().replace('\n', ',')
        alle_tickers = [t.strip().upper() for t in content.split(',') if t.strip()]

    kwaliteits_tickers = []
    
    print(f"Start analyse van {len(alle_tickers)} aandelen...")
    
    for symbol in alle_tickers:
        if munger_keuring(symbol):
            kwaliteits_tickers.append(symbol)
        # Slaap even om Yahoo niet te overbelasten
        time.sleep(0.5)

    # Opslaan van de nieuwe lijst
    with open(output_bestand, 'w') as f:
        f.write(", ".join(kwaliteits_tickers))
        
    print(f"\nKlaar! Er zijn {len(kwaliteits_tickers)} kwaliteits-aandelen gevonden.")
    print(f"Je nieuwe tickerlijst staat in: {output_bestand}")

if __name__ == "__main__":
    main()
