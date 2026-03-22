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
    if not TOKEN or not CHAT_ID:
        print("Telegram configuratie ontbreekt.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        if len(bericht) > 4000:
            for i in range(0, len(bericht), 4000):
                requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[i:i+4000], "parse_mode": "Markdown"})
        else:
            requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"Fout bij versturen Telegram: {e}")

def voer_backtest_uit(ticker, inzet=2500, kost_pct=0.02):
    # Haal data op
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 200:
        return None, [f"⚠️ *{ticker}*: Te weinig data."]

    # Indicatoren
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df = df.iloc[-252:].copy() # Laatste jaar

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
            kosten = inzet * kost_pct
            koop_prijs = prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | ${prijs:.2f}")

        # VERKOOP (Death Cross)
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            rendement = prijs / koop_prijs
            bruto = inzet * rendement
            verkoop_kosten = bruto * kost_pct
            netto = bruto - verkoop_kosten - (inzet * kost_pct)
            huidig_saldo = netto
            trades_log.append(f"🔴 *{ticker} VERKOOP*: {datum} | ${prijs:.2f} (Netto: €{netto:.2f})")
            positie = False

    if positie:
        laatste_prijs = float(df['Close'].iloc[-1])
        huidig_saldo = (inzet * (laatste_prijs / koop_prijs)) - (inzet * kost_pct)

    return huidig_saldo, trades_log

def main():
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    makelaar_kost = 0.02
    
    try:
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except:
        tickers = ['AAPL', 'NVDA', 'TSLA']

    alle_trades = []
    eindwaarde_portfolio = start_kapitaal - (len(tickers) * inzet_per_aandeel)

    for t in tickers:
        eind_waarde, trade_history = voer_backtest_uit(t, inzet_per_aandeel, makelaar_kost)
        if eind_waarde is not None:
            eindwaarde_portfolio += eind_waarde
            alle_trades.extend(trade_history)

    rapport = f"📊 *BACKTEST RAPPORT*\n💰 Start: €{start_kapitaal:,.2f}\n💸 Kosten: 2%\n\n"
    rapport += "*TRADES:*\n" + ("\n".join(alle_trades) if alle_trades else "Geen trades.")
    rapport += f"\n\n🏁 *EINDSTAND: €{eindwaarde_portfolio:,.2f}*"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
