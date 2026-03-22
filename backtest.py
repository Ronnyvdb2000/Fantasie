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
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except:
        return False

def voer_backtest_uit(ticker, inzet=2500):
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    # Haal 2 jaar data op
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 200: return None, []

    # Bereken SMA's over de HELE periode
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Pak nu de laatste 365 dagen voor de test
    test_periode = df.iloc[-252:].copy()

    positie = False
    koop_prijs = 0
    huidig_saldo = inzet
    trades_log = []

    for i in range(1, len(test_periode)):
        # We kijken naar de data van de test_periode, maar de SMA's zijn al berekend
        rij_nu = test_periode.iloc[i]
        rij_oud = test_periode.iloc[i-1]
        
        prijs = float(rij_nu['Close'])
        datum = test_periode.index[i].strftime('%d-%m-%Y')

        # Check voor Golden Cross (Koop)
        if not positie and rij_nu['SMA50'] > rij_nu['SMA200'] and rij_oud['SMA50'] <= rij_oud['SMA200']:
            aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
            koop_prijs = prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | ${prijs:.2f}")

        # Check voor Death Cross (Verkoop)
        elif positie and rij_nu['SMA50'] < rij_nu['SMA200'] and rij_oud['SMA50'] >= rij_oud['SMA200']:
            bruto_waarde = inzet * (prijs / koop_prijs)
            verkoop_kosten = VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)
            aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
            
            winst = bruto_waarde - inzet - aankoop_kosten - verkoop_kosten
            belasting = winst * MEERWAARDE_TAX_PCT if winst > 0 else 0
            
            netto = bruto_waarde - verkoop_kosten - aankoop_kosten - belasting
            huidig_saldo = netto
            trades_log.append(f"🔴 *{ticker} VERK*: {datum} | ${prijs:.2f} | Netto: €{netto:.2f}")
            positie = False

    # Fictieve eindwaarde berekenen als we nog een open positie hebben
    if positie:
        laatste_prijs = float(test_periode['Close'].iloc[-1])
        bruto = inzet * (laatste_prijs / koop_prijs)
        huidig_saldo = bruto - (VASTE_KOST + (bruto * BEURSTAKS_PCT)) - (VASTE_KOST + (inzet * BEURSTAKS_PCT))

    return huidig_saldo, trades_log

def main():
    stuur_telegram("🔄 *Backtest Start:* Verbinding OK. Analyse loopt...")

    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'META']

    alle_trades = []
    totaal_netto_resultaat = 0

    for t in tickers:
        print(f"Bezig met {t}...")
        eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel)
        if eind_waarde is not None:
            totaal_netto_resultaat += (eind_waarde - inzet_per_aandeel)
            alle_trades.extend(trade_history)

    eind_totaal = start_kapitaal + totaal_netto_resultaat
    
    rapport = f"📊 *RESULTAAT LAATSTE 12 MAANDEN*\n"
    rapport += f"💰 Start: €{start_kapitaal:,.2f}\n"
    rapport += f"🏦 Kosten: €15 + 0.35% | 🏛️ Tax: 10%\n\n"
    rapport += "*TRADES:*\n" + ("\n".join(alle_trades) if alle_trades else "_Geen kruisingen gevonden in deze periode._")
    rapport += f"\n\n🏁 *EINDSTAND: €{eind_totaal:,.2f}*"
    rapport += f"\n📈 Rendement: {((eind_totaal/start_kapitaal)-1)*100:.2f}%"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
