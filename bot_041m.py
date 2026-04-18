import yfinance as yf
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
SOURCE_FILE = "tickers_041a.txt"
MASTER_FILE = "tickers_041xx.txt"
CURRENT_RUN_FILE = "tickers_041m.txt"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"⚠️ Telegram fout: {e}")

def munger_keuring(symbol):
    """Controleert fundamenten volgens Munger-criteria."""
    try:
        t = yf.Ticker(symbol)
        # Gebruik fast_info voor snelheid of info voor diepgang
        info = t.info 
        
        if not info or 'returnOnEquity' not in info:
            print(f"🟡 {symbol:8} | Geen data gevonden (check suffix .DE/.MC)")
            return False, 0, 0, 0
        
        # Data ophalen
        roe = info.get('returnOnEquity', 0) or 0
        debt = info.get('debtToEquity', 1000) or 1000 # Default hoog bij geen data
        margin = info.get('profitMargins', 0) or 0
        
        # Debugging in console
        print(f"🔍 {symbol:8} | ROE: {roe:>6.1%} | Debt: {debt:>6.2f} | Margin: {margin:>6.1%}")
        
        # Munger Criteria: 
        # ROE > 15%, Debt/Equity < 60 (sommige API's geven 0.60, andere 60), Margin > 10%
        # We accepteren schuld onder 60 (percentage) OF onder 0.6 (ratio)
        if roe > 0.15 and (debt < 60 or debt < 0.6) and margin > 0.10:
            return True, roe, debt, margin
        
        return False, 0, 0, 0
    except Exception as e:
        print(f"❌ {symbol:8} | Error: {e}")
        return False, 0, 0, 0

def main():
    print("\n--- 🕵️‍♂️ MUNGER QUALITY SCANNER START ---")
    
    # Bronbestand inladen
    if not os.path.exists(SOURCE_FILE):
        msg = f"❌ FOUT: {SOURCE_FILE} niet gevonden."
        print(msg)
        send_telegram_msg(msg)
        return

    with open(SOURCE_FILE, 'r') as f:
        data = f.read().replace('\n', ',').replace('$', '')
        scan_tickers = sorted(list(set([t.strip().upper() for t in data.split(',') if t.strip()])))

    # Master-lijst inladen
    master_tickers = []
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, 'r') as f:
            m_data = f.read().replace('\n', ',')
            master_tickers = sorted(list(set([t.strip().upper() for t in m_data.split(',') if t.strip()])))

    kwaliteits_vondsten = []
    nieuwe_vondsten_details = []

    print(f"Target: {len(scan_tickers)} tickers uit {SOURCE_FILE}\n")

    for s in scan_tickers:
        is_kwaliteit, roe, debt, margin = munger_keuring(s)
        if is_kwaliteit:
            kwaliteits_vondsten.append(s)
            if s not in master_tickers:
                nieuwe_vondsten_details.append(f"🔹 *{s}* (ROE: {roe:.1%}, Debt: {debt:.1f}, Marge: {margin:.1%})")
        
        # Kleine pauze om Yahoo API-blocking te voorkomen
        time.sleep(0.7)

    # Master-lijst bijwerken
    if nieuwe_vondsten_details:
        for detail in nieuwe_vondsten_details:
            # Ticker terug extraheren uit de string voor de lijst
            t_name = detail.split('*')[1]
            if t_name not in master_tickers:
                master_tickers.append(t_name)
        
        master_tickers = sorted(list(set(master_tickers)))
        with open(MASTER_FILE, 'w') as f:
            f.write(", ".join(master_tickers))

    # Altijd huidige run opslaan
    with open(CURRENT_RUN_FILE, 'w') as f:
        f.write(", ".join(kwaliteits_vondsten))

    # --- RAPPORTAGE ---
    nu = time.strftime("%d/%m/%Y %H:%M")
    if nieuwe_vondsten_details:
        bericht = f"🚀 *Nieuwe Munger Kwaliteit!*\n_{nu}_\n\n"
        bericht += "Nieuw in Master-lijst:\n"
        bericht += "\n".join(nieuwe_vondsten_details)
    else:
        bericht = f"✅ *Munger Scan Voltooid*\n_{nu}_\n\nGeen nieuwe toevoegingen.\n"
    
    selectie_str = ", ".join(kwaliteits_vondsten) if kwaliteits_vondsten else "Geen"
    bericht += f"\n\n💎 *Huidige Elite ({len(kwaliteits_vondsten)}):*\n`{selectie_str}`"
    bericht += f"\n\n📚 *Totaal in Master:* {len(master_tickers)}"
    
    send_telegram_msg(bericht)
    print("\n--- ✅ SCAN VOLTOOID ---")

if __name__ == "__main__":
    main()
