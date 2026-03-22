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
    """Verstuurt het rapport naar Telegram."""
    if not TOKEN or not CHAT_ID:
        print("Telegram configuratie ontbreekt (Token/ID).")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        # Splitsen als bericht te lang is voor Telegram
        if len(bericht) > 4000:
            for i in range(0, len(bericht), 4000):
                requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"})
        else:
            requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Fout bij versturen Telegram: {e}")

def voer_backtest_uit(ticker, inzet=2500, kost_pct=0.02):
    """Berekent de prestaties per aandeel inclusief 2% kosten."""
    # Haal 2 jaar data op voor de 200-dagen berekening
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    
    if df.empty or len(df) < 200:
        return None, [f"⚠️ *{ticker}*: Onvoldoende data voor analyse."]

    # Indicatoren
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Filter op het laatste jaar (ca. 252 handelsdagen)
    df = df.iloc[-252:].copy()

    positie = False
    koop_prijs = 0
    huidig_saldo_voor_dit_aandeel = inzet
    trades_log = []

    for i in range(1, len(df)):
        s50_nu, s50_oud = df['SMA50'].iloc[i], df['SMA50'].iloc[i-1]
        s200_nu, s200_oud = df['SMA200'].iloc[i], df['SMA200'].iloc[i-1]
        prijs = float(df['Close'].iloc[i])
        datum = df.index[i].strftime('%d-%m-%Y')

        # KOOP (Golden Cross)
        if not positie and s50_nu > s200_nu and s50_oud <= s200_oud:
            aankoop_kosten = inzet * kost_pct
            koop_prijs = prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | Prijs: ${prijs:.2f} (Kosten: €{aankoop_kosten:.2f})")

        # VERKOOP (Death Cross)
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            rendement = prijs / koop_prijs
            bruto_waarde = inzet * rendement
            verkoop_kosten = bruto_waarde * kost_pct
            netto_resultaat = bruto_waarde - verkoop_kosten - (inzet * kost_pct)
            
            winst_euro = netto_resultaat - inzet
            huidig_saldo_voor_dit_aandeel = netto_resultaat
            trades_log.append(f"🔴 *{ticker} VERKOOP*: {datum} | Prijs: ${prijs:.2f} (Winst/Verlies: €{winst_euro:.2f})")
            positie = False

    # Als we aan het einde nog in een trade zitten, berekenen we de fictieve verkoopwaarde nu
    if positie:
        laatste_prijs = float(df['Close'].iloc[-1])
        rendement = laatste_prijs / koop_prijs
        huidig_saldo_voor_dit_aandeel = (inzet * rendement) - (inzet * kost_pct)

    return huidig_saldo_voor_dit_aandeel, trades_log

def main():
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    makelaar_kost = 0.02
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        tickers = ['AAPL', 'MSFT', 'NVDA'] # Standaard lijstje als bestand mist

    alle_trades = []
    # We trekken de totale inzet van de tickers af van het startkapitaal voor de cash-positie
    eindwaarde_portfolio = start_kapitaal - (len(tickers) * inzet_per_aandeel)

    for t in tickers:
        print(f"Analyseert {t}...")
        eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel, makelaar_kost)
        if eind_waarde is not None:
            eindwaarde_portfolio += eind_waarde
            alle_trades.extend(trade_history)

    # Rapport samenstellen
    rapport = "📊 *BACKTEST RAPPORT (AFGELOPEN JAAR)*\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"📈 Inzet per aandeel: €{inzet_per_aandeel:,.2f}\n"
    rapport += f"💸 Makelaarskosten: {makelaar_kost*100}%\n\n"
    
    rapport += "*TRADE LOG:*\n"
    if alle_trades:
        rapport += "\n".join(alle_trades)
    else:
        rapport += "Geen kruisingen (trades) gevonden dit jaar."
    
    rapport += f"\n\n🏁 *EINDBEDRAG PORTFOLIO: €{eindwaarde_portfolio:,.2f}*"
    rendement = ((eindwaarde_portfolio / start_kapitaal) - 1) * 100
    rapport += f"\nTotaal Rendement: {rendement:.2f}%"

    print(rapport)
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
