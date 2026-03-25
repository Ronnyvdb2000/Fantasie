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
    """Verstuurt voortgang en resultaten naar Telegram."""
    if not TOKEN or not CHAT_ID: 
        print(f"⚠️ Telegram niet geconfigureerd. Bericht:\n{bericht}")
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        # Telegram heeft een limiet van 4096 tekens per bericht
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[:4000], "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except Exception as e:
        print(f"❌ Telegram fout: {e}")
        return False

def voer_backtest_uit(ticker, inzet=2500):
    """Voert de Golden Cross / Death Cross backtest uit voor één ticker."""
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    # Haal 3 jaar data op om SMA200 buffer te hebben voor het begin van de testperiode
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    
    # Fix voor yfinance Multi-Index (indien nodig)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 250:
        print(f"⚠️ Onvoldoende data voor {ticker}")
        return None, []

    # Bereken SMA's op de volledige dataset
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Pak de laatste 252 handelsdagen (ongeveer 1 jaar) voor de eigenlijke test
    test_periode = df.iloc[-252:].copy()

    positie = False
    koop_prijs = 0
    netto_winst_totaal = 0
    trades_log = []

    for i in range(1, len(test_periode)):
        # Haal de waarden op als floats om errors te voorkomen
        s50_nu = float(test_periode['SMA50'].iloc[i])
        s200_nu = float(test_periode['SMA200'].iloc[i])
        s50_oud = float(test_periode['SMA50'].iloc[i-1])
        s200_oud = float(test_periode['SMA200'].iloc[i-1])
        
        # We handelen op de slotkoers van de dag dat het signaal bevestigd is
        sluit_prijs = float(test_periode['Close'].iloc[i])
        datum = test_periode.index[i].strftime('%d-%m-%Y')

        # 🔵 Check voor Golden Cross (Koop)
        if not positie and s50_nu > s200_nu and s50_oud <= s200_oud:
            koop_prijs = sluit_prijs
            positie = True
            trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | Slot: ${sluit_prijs:.2f}")

        # 🔴 Check voor Death Cross (Verkoop)
        elif positie and s50_nu < s200_nu and s50_oud >= s200_oud:
            aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
            bruto_waarde = inzet * (sluit_prijs / koop_prijs)
            verkoop_kosten = VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)
            
            # Winstberekening na alle kosten
            winst_voor_belasting = bruto_waarde - inzet - aankoop_kosten - verkoop_kosten
            belasting = max(0, winst_voor_belasting * MEERWAARDE_TAX_PCT)
            
            netto_resultaat = winst_voor_belasting - belasting
            netto_winst_totaal += netto_resultaat
            
            trades_log.append(f"🔴 *{ticker} VERK*: {datum} | Slot: ${sluit_prijs:.2f} | Netto: €{netto_resultaat:.2f}")
            positie = False

    # Afsluitende berekening voor openstaande posities
    if positie:
        laatste_prijs = float(test_periode['Close'].iloc[-1])
        aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
        bruto_nu = inzet * (laatste_prijs / koop_prijs)
        verkoop_kosten = VASTE_KOST + (bruto_nu * BEURSTAKS_PCT)
        winst_ongerealiseerd = bruto_nu - inzet - aankoop_kosten - verkoop_kosten
        netto_winst_totaal += (winst_ongerealiseerd - max(0, winst_ongerealiseerd * MEERWAARDE_TAX_PCT))

    return netto_winst_totaal, trades_log

def main():
    print("🚀 Backtest upgrade gestart...")
    start_kapitaal = 50000
    inzet_per_aandeel = 2500 # Hoeveel we per signaal inleggen
    
    # Lijst met tickers (probeer bestand te lezen, anders fallback)
    if os.path.exists('aandelen.txt'):
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'ASML.AS', 'AMD', 'META', 'GOOGL']

    alle_trades = []
    totaal_netto_resultaat = 0

    for t in tickers:
        print(f"Analyseert: {t}...")
        try:
            resultaat, trade_history = voer_backtest_uit(t, inzet_per_aandeel)
            if resultaat is not None:
                totaal_netto_resultaat += resultaat
                alle_trades.extend(trade_history)
        except Exception as e:
            print(f"❌ Fout bij aandeel {t}: {e}")

    # Eindrapportage
    eind_stand = start_kapitaal + totaal_netto_resultaat
    rendement_pct = ((eind_stand / start_kapitaal) - 1) * 100
    
    rapport = f"📊 *RESULTAAT BACKTEST upgrade (12 MAANDEN)*\n"
    rapport += f"───────────────────\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"🏁 Eindstand: €{eind_stand:,.2f}\n"
    rapport += f"📈 Rendement: {rendement_pct:.2f}%\n"
    rapport += f"───────────────────\n"
    rapport += f"*LAATSTE TRADES (max 15):*\n"
    
    if alle_trades:
        rapport += "\n".join(alle_trades[-15:])
    else:
        rapport += "_Geen trades gevonden in deze periode._"
    
    stuur_telegram(rapport)
    print(f"✅ Klaar! Eindstand: €{eind_stand:.2f}")

if __name__ == "__main__":
    main()
