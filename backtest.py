import yfinance as yf
import pandas as pd

def voer_backtest_uit(ticker, start_kapitaal=1000):
    print(f"--- Analyse van {ticker} ---")
    
    # Haal data op (2 jaar om SMA200 direct te kunnen berekenen voor het laatste jaar)
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    if df.empty or len(df) < 200:
        print(f"⚠️ Te weinig data voor {ticker}\n")
        return None

    # Indicatoren berekenen
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Focus op het afgelopen jaar (ca. 252 handelsdagen)
    df = df.iloc[-252:].copy() 

    positie = False
    instap_prijs = 0
    huidig_kapitaal = start_kapitaal
    aantal_trades = 0

    for i in range(1, len(df)):
        sma50_nu = df['SMA50'].iloc[i]
        sma200_nu = df['SMA200'].iloc[i]
        sma50_oud = df['SMA50'].iloc[i-1]
        sma200_oud = df['SMA200'].iloc[i-1]
        prijs = float(df['Close'].iloc[i])

        # KOOP SIGNAAL (Golden Cross)
        if not positie and sma50_nu > sma200_nu and sma50_oud <= sma200_oud:
            positie = True
            instap_prijs = prijs
            aantal_trades += 1
            print(f"  [KOOP]  Dag {i}: ${prijs:.2f}")

        # VERKOOP SIGNAAL (Death Cross)
        elif positie and sma50_nu < sma200_nu and sma50_oud >= sma200_oud:
            positie = False
            rendement = prijs / instap_prijs
            huidig_kapitaal *= rendement
            print(f"  [VERKOOP] Dag {i}: ${prijs:.2f} | Trade winst: {((rendement-1)*100):.2f}%")

    # Als we nog in een trade zitten aan het einde van het jaar
    if positie:
        laatste_prijs = float(df['Close'].iloc[-1])
        huidig_kapitaal *= (laatste_prijs / instap_prijs)

    # Vergelijking met simpelweg vasthouden (Buy & Hold)
    b_h_rendement = float(df['Close'].iloc[-1]) / float(df['Close'].iloc[0])
    b_h_eindbedrag = start_kapitaal * b_h_rendement
    
    print(f"\nEindstand {ticker}:")
    print(f"• Trades gedaan: {aantal_trades}")
    print(f"• Jouw eindbedrag: €{huidig_kapitaal:.2f}")
    print(f"• Buy & Hold eindbedrag: €{b_h_eindbedrag:.2f}")
    print("-" * 30 + "\n")
    
    return {
        'ticker': ticker,
        'bot_result': huidig_kapitaal,
        'bh_result': b_h_eindbedrag
    }

def main():
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA'] # Fallback
    
    totaal_overzicht = []
    for t in tickers:
        res = voer_backtest_uit(t)
        if res:
            totaal_overzicht.append(res)
    
    # Eindtabel printen
    if totaal_overzicht:
        print("======= TOTAAL OVERZICHT PORTFOLIO (START €1000 PER AANDEEL) =======")
        for r in totaal_overzicht:
            winst = r['bot_result'] - 1000
            print(f"{r['ticker']}: €{r['bot_result']:.2f} (Winst/Verlies: €{winst:.2f})")
        print("====================================================================")

if __name__ == "__main__":
    main()
