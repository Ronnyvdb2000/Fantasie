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
        # We gebruiken chunking voor het geval het rapport erg lang wordt door 5 jaar aan trades
        if len(bericht) > 4000:
            for i in range(0, len(bericht), 4000):
                requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"})
        else:
            requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
        return True
    except:
        return False

def voer_backtest_uit(ticker, inzet=2500):
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    # Haal 7 jaar data op (voldoende voor 5 jaar test + 200 dagen SMA historie)
    df = yf.download(ticker, period="7y", interval="1d", progress=False)
    if df.empty or len(df) < 200: return None, []

    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Pak de laatste 5 jaar (ca. 1260 handelsdagen)
    test_periode = df.iloc[-1260:].copy()

    positie = False
    koop_prijs = 0
    huidig_saldo = inzet
    trades_log = []

    for i in range(1, len(test_periode)):
        s50_nu = test_periode['SMA50'].iloc[i]
        s200_nu = test_periode['SMA200'].iloc[i]
        s50_oud = test_periode['SMA50'].iloc[i-1]
        s200_oud = test_periode['SMA200'].iloc[i-1]
        prijs = float(test_periode['Close'].iloc[i])
        datum = test_periode.index[i].strftime('%d-%m-%Y')

        # KOOP
        if not positie and s50_nu > s200_nu and s50_oud <= s200_oud:
            koop_prijs = prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | ${prijs:.2f}")

        # VERKOOP
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            bruto_waarde = inzet * (prijs / koop_prijs)
            verkoop_kosten = VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)
            aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
            
            winst = bruto_waarde - inzet - aankoop_kosten - verkoop_kosten
            belasting = winst * MEERWAARDE_TAX_PCT if winst > 0 else 0
            
            netto = bruto_waarde - verkoop_kosten - aankoop_kosten - belasting
            huidig_saldo = netto
            trades_log.append(f"🔴 *{ticker} VERK*: {datum} | ${prijs:.2f} | Netto: €{netto:.2f}")
            positie = False

    if positie:
        laatste_prijs = float(test_periode['Close'].iloc[-1])
        bruto = inzet * (laatste_prijs / koop_prijs)
        huidig_saldo = bruto - (VASTE_KOST + (bruto * BEURSTAKS_PCT)) - (VASTE_KOST + (inzet * BEURSTAKS_PCT))

    return huidig_saldo, trades_log

def main():
    stuur_telegram("⏳ *Lange Termijn Analyse:* 5 jaar historie wordt berekend...")

    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN']

    alle_trades = []
    totaal_netto_resultaat = 0

    for t in tickers:
        print(f"Analyseert 5 jaar data voor: {t}")
        try:
            eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel)
            if eind_waarde is not None:
                totaal_netto_resultaat += (eind_waarde - inzet_per_aandeel)
                if len(trade_history) > 0:
                    # We voegen alleen de laatste 5 trades per aandeel toe om Telegram niet te overbelasten
                    alle_trades.extend(trade_history[-5:])
        except Exception as e:
            print(f"Fout bij {t}: {e}")

    eind_totaal = start_kapitaal + totaal_netto_resultaat
    rendement = ((eind_totaal/start_kapitaal)-1)*100
    
    rapport = f"📊 *RESULTAAT LAATSTE 5 JAAR*\n"
    rapport += f"💰 Start: €{start_kapitaal:,.2f}\n"
    rapport += f"🏦 Kost: €15 + 0.35% | 🏛️ Tax: 10%\n\n"
    rapport += "*LAATSTE TRADES (Selectie):*\n" + ("\n".join(alle_trades) if alle_trades else "_Geen trades._")
    rapport += f"\n\n🏁 *EINDSTAND: €{eind_totaal:,.2f}*"
    rapport += f"\n📈 Totaal Rendement: {rendement:.2f}%"
    rapport += f"\n📅 Gem. per jaar: {(rendement/5):.2f}%"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
