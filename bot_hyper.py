import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

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
        res = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[:4000], "parse_mode": "Markdown"}, timeout=10)
        return res.status_code == 200
    except Exception as e:
        print(f"❌ Telegram fout: {e}")
        return False

def bereken_rsi(data, window=14):
    """Bereken de Relative Strength Index."""
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def voer_backtest_uit(ticker, inzet=2500):
    """Hyper-dynamische strategie met SMA, RSI en ATR Trailing Stop."""
    # Kosten parameters
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    # Data ophalen (3 jaar voor SMA200 buffer)
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    
    # Fix voor yfinance Multi-Index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 250:
        return None, []

    # Indicatoren berekenen
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = bereken_rsi(df['Close'])
    
    # ATR berekenen voor Trailing Stop
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()

    # Testperiode: laatste 12 maanden
    test_periode = df.iloc[-252:].copy()
    
    positie = False
    koop_prijs = 0
    hoogste_prijs_sinds_koop = 0
    netto_winst_totaal = 0
    trades_log = []

    for i in range(1, len(test_periode)):
        rij = test_periode.iloc[i]
        vorige_rij = test_periode.iloc[i-1]
        prijs_slot = float(rij['Close'])
        datum = test_periode.index[i].strftime('%d-%m-%Y')

        # --- KOOP LOGICA ---
        if not positie:
            # Condities: Golden Cross + RSI < 70 + Prijs boven SMA200
            if (rij['SMA50'] > rij['SMA200'] and vorige_rij['SMA50'] <= vorige_rij['SMA200']) and \
               (rij['RSI'] < 70) and (prijs_slot > rij['SMA200']):
                
                koop_prijs = prijs_slot
                hoogste_prijs_sinds_koop = prijs_slot
                positie = True
                trades_log.append({
                    "ticker": ticker, "datum": datum, "type": "KOOP", 
                    "prijs": prijs_slot, "reden": "Golden Cross + RSI", "netto": 0
                })

        # --- VERKOOP LOGICA ---
        elif positie:
            hoogste_prijs_sinds_koop = max(hoogste_prijs_sinds_koop, prijs_slot)
            # Dynamische Trailing Stop: 3x ATR onder de piek
            stop_loss_niveau = hoogste_prijs_sinds_koop - (3 * rij['ATR'])
            
            # Verkoop bij Death Cross OF Trailing Stop geraakt
            if (rij['SMA50'] < rij['SMA200']) or (prijs_slot < stop_loss_niveau):
                reden = "Death Cross" if rij['SMA50'] < rij['SMA200'] else "Trailing Stop"
                
                aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
                bruto_waarde = inzet * (prijs_slot / koop_prijs)
                verkoop_kosten = VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)
                
                winst_voor_tax = bruto_waarde - inzet - aankoop_kosten - verkoop_kosten
                belasting = max(0, winst_voor_tax * MEERWAARDE_TAX_PCT)
                netto_resultaat = winst_voor_tax - belasting
                
                netto_winst_totaal += netto_resultaat
                trades_log.append({
                    "ticker": ticker, "datum": datum, "type": "VERKOOP", 
                    "prijs": prijs_slot, "reden": reden, "netto": round(netto_resultaat, 2)
                })
                positie = False

    return netto_winst_totaal, trades_log

def main():
    print("🚀 Start bot_hyper.py Backtest...")
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    # Tickers laden
    if os.path.exists('aandelen.txt'):
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'ASML.AS', 'AMD', 'META', 'AMZN']

    alle_trades_raw = []
    totaal_netto_winst = 0

    for t in tickers:
        print(f"Analyseert {t}...")
        try:
            winst, trades = voer_backtest_uit(t, inzet_per_aandeel)
            if winst is not None:
                totaal_netto_winst += winst
                alle_trades_raw.extend(trades)
        except Exception as e:
            print(f"❌ Fout bij {t}: {e}")

    # Resultaten verwerken
    eind_stand = start_kapitaal + totaal_netto_winst
    rendement = ((eind_stand / start_kapitaal) - 1) * 100

    # CSV Export voor Excel
    if alle_trades_raw:
        df_export = pd.DataFrame(alle_trades_raw)
        df_export.to_csv('backtest_results.csv', index=False)
        print("📁 Resultaten opgeslagen in backtest_results.csv")

    # Telegram Rapport
    rapport = f"📊 *BOT_HYPER STRATEGIE RAPPORT*\n"
    rapport += f"───────────────────\n"
    rapport += f"💰 Startkapitaal: €{start_kapitaal:,.2f}\n"
    rapport += f"🏁 Eindstand: €{eind_stand:,.2f}\n"
    rapport += f"📈 Rendement: {rendement:.2f}%\n"
    rapport += f"───────────────────\n"
    rapport += f"*LAATSTE ACTIES (max 10):*\n"
    
    for tr in alle_trades_raw[-10:]:
        icon = "🔵" if tr['type'] == "KOOP" else "🔴"
        rapport += f"{icon} {tr['ticker']} | {tr['datum']} | ${tr['prijs']:.2f} | {tr['type']} ({tr['reden']})\n"

    stuur_telegram(rapport)
    print("✅ Backtest voltooid.")

if __name__ == "__main__":
    main()
