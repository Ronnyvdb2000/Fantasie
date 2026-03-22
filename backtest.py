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
    if len(bericht) > 4000:
        for i in range(0, len(bericht), 4000):
            requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"})
    else:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)

def voer_backtest_uit(ticker, inzet=2500):
    # Parameters voor kosten
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 200: return None, []

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df = df.iloc[-252:].copy()

    positie = False
    koop_prijs = 0
    huidig_saldo = inzet
    trades_log = []

    for i in range(1, len(df)):
        s50_nu, s50_oud = df['SMA50'].iloc[i], df['SMA50'].iloc[i-1]
        s200_nu, s200_oud = df['SMA200'].iloc[i], df['SMA200'].iloc[i-1]
        prijs = float(df['Close'].iloc[i])
        datum = df.index[i].strftime('%d-%m-%Y')

        # KOOP (Golden Cross)
        if not positie and s50_nu > s200_nu and s50_oud <= s200_oud:
            # Kosten bij aankoop: 15€ + 0.35% taks
            taks = inzet * BEURSTAKS_PCT
            totaal_kosten_koop = VASTE_KOST + taks
            koop_prijs = prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | ${prijs:.2f} (Kosten: €{totaal_kosten_koop:.2f})")

        # VERKOOP (Death Cross)
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            rendement = prijs / koop_prijs
            bruto_waarde = inzet * rendement
            
            # 1. Kosten bij verkoop
            verkoop_taks = bruto_waarde * BEURSTAKS_PCT
            totaal_kosten_verkoop = VASTE_KOST + verkoop_taks
            
            # 2. Meerwaardebelasting (alleen over de winst NA aftrek van aankoopkosten)
            aankoop_kosten_totaal = VASTE_KOST + (inzet * BEURSTAKS_PCT)
            winst = bruto_waarde - inzet - aankoop_kosten_totaal - totaal_kosten_verkoop
            
            belasting = 0
            if winst > 0:
                belasting = winst * MEERWAARDE_TAX_PCT
            
            netto_resultaat = bruto_waarde - totaal_kosten_verkoop - aankoop_kosten_totaal - belasting
            huidig_saldo = netto_resultaat
            
            trades_log.append(f"🔴 *{ticker} VERKOOP*: {datum} | ${prijs:.2f} | Netto: €{netto_resultaat:.2f} (Tax: €{belasting:.2f})")
            positie = False

    # Eindberekening als nog in positie
    if positie:
        laatste_prijs = float(df['Close'].iloc[-1])
        bruto = inzet * (laatste_prijs / koop_prijs)
        huidig_saldo = bruto - (VASTE_KOST + (bruto * BEURSTAKS_PCT)) - (VASTE_KOST + (inzet * BEURSTAKS_PCT))

    return huidig_saldo, trades_log

def main():
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA']

    alle_trades = []
    eindwaarde_portfolio = start_kapitaal - (len(tickers) * inzet_per_aandeel)

    for t in tickers:
        eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel)
        if eind_waarde is not None:
            eindwaarde_portfolio += eind_waarde
            alle_trades.extend(trade_history)

    rapport = f"📊 *BACKTEST (REALISTISCHE KOSTEN)*\n💰 Start: €{start_kapitaal:,.2f}\n"
    rapport += "🏦 Kost: €15 + 0.35% TOB | 🏛️ Tax: 10%\n\n"
    rapport += "*TRADES:*\n" + ("\n".join(alle_trades) if alle_trades else "Geen trades.")
    rapport += f"\n\n🏁 *EINDSTAND: €{eindwaarde_portfolio:,.2f}*"
    rapport += f"\nRendement: {((eindwaarde_portfolio/50000)-1)*100:.2f}%"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
