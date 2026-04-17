import yfinance as yf
import os
import time
import requests

# --- CONFIG ---
SOURCE_FILE = "tickers_04a.txt"
MASTER_FILE = "tickers_04xx.txt"
CURRENT_RUN_FILE = "tickers_04m.txt"

# Telegram Config (Zorg dat deze variabelen in je GitHub Secrets staan of vul ze hier in)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"⚠️ Telegram fout: {e}")

def munger_keuring(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        roe = info.get('returnOnEquity', 0)
        debt = info.get('debtToEquity', 100)
        margin = info.get('profitMargins', 0)
        
        # Munger Criteria: ROE > 15%, Schuld < 60, Marge > 10%
        if roe > 0.15 and debt < 60 and margin > 0.10:
            return True, roe, debt
        return False, 0, 0
    except:
        return False, 0, 0

def main():
    print("--- MASTER LIST UPDATER + TELEGRAM (04m -> 04xx) ---")
    
    # 1. Vind bronbestand
    bestanden = os.listdir('.')
    actueel_bron = next((f for f in bestanden if f.lower() == SOURCE_FILE.lower()), None)
    if not actueel_bron:
        print(f"❌ Bronbestand {SOURCE_FILE} niet gevonden.")
        return

    # 2. Lees tickers
    with open(actueel_bron, 'r') as f:
        data = f.read().replace('\n', ',')
        scan_tickers = [t.strip().upper() for t in data.split(',') if t.strip()]

    # 3. Master-lijst inlezen
    master_tickers = []
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, 'r') as f:
            master_data = f.read().replace('\n', ',')
            master_tickers = [t.strip().upper() for t in master_data.split(',') if t.strip()]

    # 4. Analyse en vergelijking
    kwaliteits_vondsten = []
    nieuwe_vondsten_details = []

    for s in scan_tickers:
        print(f"Check: {s}...", end=" ")
        is_kwaliteit, roe, debt = munger_keuring(s)
        if is_kwaliteit:
            print("KWALITEIT! ✅")
            kwaliteits_vondsten.append(s)
            if s not in master_tickers:
                nieuwe_vondsten_details.append(f"🔹 *{s}* (ROE: {roe:.1%}, Debt: {debt:.1f})")
        else:
            print("❌")
        time.sleep(0.5)

    # 5. Master-lijst bijwerken
    nieuwe_tickers = [t for t in kwaliteits_vondsten if t not in master_tickers]
    if nieuwe_tickers:
        master_tickers.extend(nieuwe_tickers)
        master_tickers.sort()
        
        # Sla op
        with open(MASTER_FILE, 'w') as f:
            f.write(", ".join(master_tickers))
        
        # Stuur Telegram melding
        bericht = "🚀 *Nieuwe Munger Kwaliteit ontdekt!*\n\n"
        bericht += "De volgende Benelux aandelen zijn toegevoegd aan de Master-lijst (04xx):\n"
        bericht += "\n".join(nieuwe_vondsten_details)
        bericht += f"\n\n📊 Totaal aantal kwaliteits-aandelen: {len(master_tickers)}"
        send_telegram_msg(bericht)
    else:
        print("Geen nieuwe aandelen gevonden voor de Master-lijst.")

    # Altijd de dag-lijst opslaan
    with open(CURRENT_RUN_FILE, 'w') as f:
        f.write(", ".join(kwaliteits_vondsten))

if __name__ == "__main__":
    main()
