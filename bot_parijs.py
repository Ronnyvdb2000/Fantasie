import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import warnings

# Onderdruk waarschuwingen voor een schone GitHub-log
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATIE ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except:
        print("Telegram verzenden mislukt.")

def bereken_strategie(df, inzet, snelle_ma, trage_ma):
    # Belgische Kosten & Taksen
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
        f_nu = test_data['Fast'].iloc[i]
        f_oud = test_data['Fast'].iloc[i-1]
        s_nu = test_data['Slow'].iloc[i]
        s_oud = test_data['Slow'].iloc[i-1]
        prijs = float(test_data['Close'].iloc[i])

        if not positie and f_nu > s_nu and f_oud <= s_oud:
            koop_prijs = prijs
            positie = True
            aantal_trades += 1

        elif positie and f_nu < s_nu and f_oud >= s_oud:
            bruto_waarde = inzet * (prijs / koop_prijs)
            kosten = (VASTE_KOST + (inzet * BEURSTAKS_PCT)) + (VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT))
            winst = bruto_waarde - inzet - kosten
            tax = winst * MEERWAARDE_TAX_PCT if winst > 0 else 0
            huidig_saldo = bruto_waarde - (VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)) - (VASTE_KOST + (inzet * BEURSTAKS_PCT)) - tax
            positie = False
            aantal_trades += 1

    if positie:
        laatste_prijs = float(test_data['Close'].iloc[-1])
        huidig_saldo = (inzet * (laatste_prijs / koop_prijs)) - 35
        
    return huidig_saldo, aantal_trades

def main():
    # Duidelijke startmelding voor Parijs
    stuur_telegram("🗼 *START ANALYSE: EURONEXT PARIJS (FR)*\n_Bezig met berekenen van de 2 strategieën op de CAC 40..._")
    
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    try:
        with open('tickers_parijs.txt', 'r') as f:
            content = f.read()
            tickers = [t.strip() for t in content.split(',') if t.strip()]
    except:
        tickers = ['MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA']

    bot1_totaal = start_kapitaal - (len(tickers) * inzet_per_aandeel)
    bot2_totaal = start_kapitaal - (len(tickers) * inzet_per_aandeel)
    bot1_trades = 0
    bot2_trades = 0

    for t in tickers:
        print(f"Analyseert Parijs: {t}")
        try:
            df = yf.download(t, period="2y", interval="1d", progress=False)
            if df.empty or len(df) < 200: continue
            
            res1, trades1 = bereken_strategie(df, inzet_per_aandeel, 50, 200)
            bot1_totaal += res1
            bot1_trades += trades1
            
            res2, trades2 = bereken_strategie(df, inzet_per_aandeel, 20, 50)
            bot2_totaal += res2
            bot2_trades += trades2
        except:
            print(f"Fout bij ophalen {t}")

    # Rapportage met Parijs branding
    winnaar = "BOT 1 (50/200)" if bot1_totaal > bot2_totaal else "BOT 2 (20/50)"
    
    rapport = f"🇫🇷 *EINDRAPPORT EURONEXT PARIJS*\n"
    rapport += f"📅 Periode: Laatste 12 maanden\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"----------------------------------\n\n"
    rapport += f"🤖 *BOT 1 (Trend: 50/200 SMA)*\n"
    rapport += f"   • Eindstand: €{bot1_totaal:,.2f}\n"
    rapport += f"   • Totaal trades: {bot1_trades}\n\n"
    rapport += f"🤖 *BOT 2 (Actief: 20/50 SMA)*\n"
    rapport += f"   • Eindstand: €{bot2_totaal:,.2f}\n"
    rapport += f"   • Totaal trades: {bot2_trades}\n\n"
    rapport += f"🏆 *Winnaar Parijs:* {winnaar}"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
