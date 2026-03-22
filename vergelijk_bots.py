import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# --- SETUP ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def bereken_strategie(df, inzet, snelle_ma, trage_ma):
    # Kosteninstellingen
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    df = df.copy()
    df['Fast'] = df['Close'].rolling(window=snelle_ma).mean()
    df['Slow'] = df['Close'].rolling(window=trage_ma).mean()
    
    # Test op het laatste jaar
    test_data = df.iloc[-252:].copy()
    
    positie = False
    koop_prijs = 0
    huidig_saldo = inzet
    aantal_trades = 0

    for i in range(1, len(test_data)):
        f_nu, f_oud = test_data['Fast'].iloc[i], test_data['Fast'].iloc[i-1]
        s_nu, s_oud = test_data['Slow'].iloc[i], test_data['Slow'].iloc[i-1]
        prijs = float(test_data['Close'].iloc[i])

        if not positie and f_nu > s_nu and f_oud <= s_oud:
            koop_prijs = prijs
            positie = True
            aantal_trades += 1
        elif positie and f_nu < s_nu and f_oud >= s_oud:
            bruto = inzet * (prijs / koop_prijs)
            kosten = (VASTE_KOST + (inzet * BEURSTAKS_PCT)) + (VASTE_KOST + (bruto * BEURSTAKS_PCT))
            winst = bruto - inzet - kosten
            tax = winst * MEERWAARDE_TAX_PCT if winst > 0 else 0
            huidig_saldo = bruto - (VASTE_KOST + (bruto * BEURSTAKS_PCT)) - (VASTE_KOST + (inzet * BEURSTAKS_PCT)) - tax
            positie = False
            aantal_trades += 1

    if positie:
        laatste_prijs = float(test_data['Close'].iloc[-1])
        huidig_saldo = (inzet * (laatste_prijs / koop_prijs)) - (30 + (inzet * 0.007)) # Vereenvoudigde schatting open positie

    return huidig_saldo, aantal_trades

def main():
    start_kapitaal = 50000
    inzet = 2500
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'META']

    bot1_totaal = start_kapitaal - (len(tickers) * inzet)
    bot2_totaal = start_kapitaal - (len(tickers) * inzet)
    bot1_trades = 0
    bot2_trades = 0

    for t in tickers:
        df = yf.download(t, period="2y", progress=False)
        if df.empty: continue
        
        # Bot 1: Klassiek (50/200)
        res1, t1 = bereken_strategie(df, inzet, 50, 200)
        bot1_totaal += res1
        bot1_trades += t1
        
        # Bot 2: Snel (20/50)
        res2, t2 = bereken_strategie(df, inzet, 20, 50)
        bot2_totaal += res2
        bot2_trades += t2

    rapport = f"⚔️ *BOT VERGELIJKING (1 JAAR)*\n\n"
    rapport += f"🤖 *BOT 1 (50/200 SMA)*\n"
    rapport += f"💰 Eindstand: €{bot1_totaal:,.2f}\n"
    rapport += f"🔄 Totaal Trades: {bot1_trades}\n\n"
    rapport += f"🤖 *BOT 2 (20/50 SMA)*\n"
    rapport += f"💰 Eindstand: €{bot2_totaal:,.2f}\n"
    rapport += f"🔄 Totaal Trades: {bot2_trades}\n\n"
    
    winnaar = "BOT 1" if bot1_totaal > bot2_totaal else "BOT 2"
    rapport += f"🏆 *Winnaar:* {winnaar}"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
