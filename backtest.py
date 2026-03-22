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
    # We splitsen lange berichten op als ze de Telegram limiet overschrijden
    if len(bericht) > 4000:
        for i in range(0, len(bericht), 4000):
            requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"})
    else:
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def voer_backtest_uit(ticker, inzet=2500, kost_pct=0.02):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 200: return None, []

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df = df.iloc[-252:].copy() # Laatste jaar

    trades = []
    positie = False
    koop_prijs = 0
    huidig_waarde = inzet

    for i in range(1, len(df)):
        s50_nu, s50_oud = df['SMA50'].iloc[i], df['SMA50'].iloc[i-1]
        s200_nu, s200_oud = df['SMA200'].iloc[i], df['SMA200'].iloc[i-1]
        prijs = float(df['Close'].iloc[i])
        datum = df.index[i].strftime('%Y-%m-%d')

        # KOOP (Golden Cross)
        if not positie and s50_nu > s200_nu and s50_oud <= s200_oud:
            kosten = inzet * kost_pct
            koop_prijs = prijs
            positie = True
            trades.append(f"🔵 *{ticker} KOOP*: {datum} op ${prijs:.2f} (Kost: €{kosten:.2f})")

        # VERKOOP (Death Cross)
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            rendement = prijs / koop_prijs
            bruto_waarde = inzet * rendement
            verkoop_kosten = bruto_waarde * kost_pct
            netto_resultaat = bruto_waarde - verkoop_kosten - (inzet * kost_pct) # Resultaat na alle kosten
            
            winst_euro = netto_resultaat - inzet
            huidig_waarde = netto_resultaat
            trades.append(f"🔴 *{ticker} VERKOOP*: {datum} op ${prijs:.2f} (Winst/Verlies: €{winst_euro:.2f})")
            positie = False

    return huidig_waarde, trades

def main():
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    makelaar_kost = 0.02
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA']

    totaal_trades_lijst = []
    eindwaarde_portfolio = start_kapitaal - (len(tickers) * inzet_per_aandeel)

    for t in tickers:
        eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel, makelaar_kost)
        if eind_waarde:
            eindwaarde_portfolio += eind_waarde
            totaal_trades_lijst.extend(trade_history)

    # Rapport opmaken
    rapport = "📊 *BACKTEST RAPPORT (1 JAAR)*\n"
    rapport += f"Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"Inzet per trade: €{inzet_per_aandeel:,.2f}\n"
    rapport += f"Makelaarscourtage: {makelaar_kost*100}%\n\n"
    
    rapport += "*TRADE HISTORIE:*\n"
    rapport += "\n".join(totaal_trades_lijst) if totaal_trades_lijst else "Geen trades gevonden."
    
    rapport += f"\n\n🏁 *EINDBEDRAG PORTFOLIO: €{eindwaarde_portfolio:,.2f}*"
    rendement = ((eindwaarde_portfolio / start_kapitaal) - 1) * 100
    rapport += f"\nTotaal Rendement: {rendement:.2f}%"

    print(rapport)
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
