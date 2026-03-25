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
        requests.post(url, data={"chat_id": CHAT_ID, "text": bericht[:4000], "parse_mode": "Markdown"}, timeout=10)
    except: pass

def bereken_rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def voer_backtest_uit(ticker, inzet=2500):
    # Kosten parameters
    VASTE_KOST = 15.00
    BEURSTAKS_PCT = 0.0035
    MEERWAARDE_TAX_PCT = 0.10

    # Data ophalen
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 250: return None, []

    # Indicatoren
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = bereken_rsi(df['Close'])
    
    # ATR voor dynamische stop loss (14 dagen)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()

    test_periode = df.iloc[-252:].copy()
    positie = False
    koop_prijs = 0
    hoogste_prijs_sinds_koop = 0
    netto_winst_totaal = 0
    trades_log = []

    for i in range(1, len(test_periode)):
        rij = test_periode.iloc[i]
        vorige_rij = test_periode.iloc[i-1]
        prijs = float(rij['Close'])
        datum = test_periode.index[i].strftime('%d-%m-%Y')

        # --- KOOP LOGICA (Hyper-Dynamisch) ---
        # 1. Golden Cross 2. RSI niet overbought 3. Boven SMA200
        if not positie:
            if (rij['SMA50'] > rij['SMA200'] and vorige_rij['SMA50'] <= vorige_rij['SMA200']) and \
               (rij['RSI'] < 70) and (prijs > rij['SMA200']):
                
                koop_prijs = prijs
                hoogste_prijs_sinds_koop = prijs
                positie = True
                trades_log.append(f"🔵 *{ticker} KOOP*: {datum} | ${prijs:.2f} (RSI: {rij['RSI']:.0f})")

        # --- VERKOOP LOGICA ---
        elif positie:
            hoogste_prijs_sinds_koop = max(hoogste_prijs_sinds_koop, prijs)
            # Dynamische Trailing Stop: Verkoop als prijs > 3 * ATR onder de piek zakt
            stop_loss_niveau = hoogste_prijs_sinds_koop - (3 * rij['ATR'])
            
            # Conditie 1: Death Cross OF Conditie 2: Trailing Stop geraakt
            if (rij['SMA50'] < rij['SMA200']) or (prijs < stop_loss_niveau):
                reden = "Death Cross" if rij['SMA50'] < rij['SMA200'] else "Trailing Stop"
                
                aankoop_kosten = VASTE_KOST + (inzet * BEURSTAKS_PCT)
                bruto_waarde = inzet * (prijs / koop_prijs)
                verkoop_kosten = VASTE_KOST + (bruto_waarde * BEURSTAKS_PCT)
                
                winst_voor_tax = bruto_waarde - inzet - aankoop_kosten - verkoop_kosten
                belasting = max(0, winst_voor_tax * MEERWAARDE_TAX_PCT)
                netto = winst_voor_tax - belasting
                
                netto_winst_totaal += netto
                trades_log.append(f"🔴 *{ticker} VERK*: {datum} | ${prijs:.2f} | {reden} | Netto: €{netto:.2f}")
                positie = False

    return netto_winst_totaal, trades_log

def main():
    start_kapitaal = 50000
    inzet_per_aandeel = 2500
    
    # Laden van tickers
    tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'ASML.AS', 'AMD', 'META', 'AMZN']
    if os.path.exists('aandelen.txt'):
        with open('aandelen.txt', 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]

    totaal_netto = 0
    alle_trades = []

    for t in tickers:
        try:
            winst, history = voer_backtest_uit(t, inzet_per_aandeel)
            if winst is not None:
                totaal_netto += winst
                alle_trades.extend(history)
        except Exception as e: print(f"Fout bij {t}: {e}")

    rendement = (( (start_kapitaal + totaal_netto) / start_kapitaal) - 1) * 100
    
    rapport = f"⚡ *HYPER-DYNAMISCHE BACKTEST*\n"
    rapport += f"💰 Start: €{start_kapitaal:,.2f} | Stand: €{start_kapitaal + totaal_netto:,.2f}\n"
    rapport += f"📊 Rendement: {rendement:.2f}%\n\n"
    rapport += "*LAATSTE ACTIES:*\n" + "\n".join(alle_trades[-12:])
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
