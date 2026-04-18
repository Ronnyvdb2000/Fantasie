import yfinance as yf
import os
import time
import requests

# --- CONFIG ---
SOURCE_FILE = "tickers_05a.txt"
MASTER_FILE = "tickers_05xx.txt"
CURRENT_RUN_FILE = "tickers_05m.txt"

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
        
        # Munger Criteria
        if roe > 0.15 and debt < 60 and margin > 0.10:
            return True, roe, debt
        return False, 0, 0
    except:
        return False, 0, 0

def main():
    print("--- MASTER LIST UPDATER + ALTIJD TELEGRAM ---")
    
    bestanden = os.listdir('.')
    actueel_bron = next((f for f in bestanden if f.lower() == SOURCE_FILE.lower()), None)
    if not actueel_bron:
        send_telegram_msg(f"❌ FOUT: Bronbestand {SOURCE_FILE} niet gevonden in de repository.")
        return

    with open(actueel_bron, 'r') as f:
        data = f.read().replace('\n', ',')
        scan_tickers = [t.strip().upper() for t in data.split(',') if t.strip()]

    master_tickers = []
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, 'r') as f:
            master_data = f.read().replace('\n', ',')
            master_tickers = [t.strip().upper() for t in master_data.split(',') if t.strip()]

    kwaliteits_vondsten = []
    nieuwe_vondsten_details = []

    for s in scan_tickers:
        is_kwaliteit, roe, debt = munger_keuring(s)
        if is_kwaliteit:
            kwaliteits_vondsten.append(s)
            if s not in master_tickers:
                nieuwe_vondsten_details.append(f"🔹 *{s}* (ROE: {roe:.1%}, Debt: {debt:.1f})")
        time.sleep(0.5)

    # Master-lijst bijwerken indien nodig
    nieuwe_tickers_gevonden = len(nieuwe_vondsten_details) > 0
    if nieuwe_tickers_gevonden:
        master_tickers.extend([t.split('*')[1].split('*')[0] for t in nieuwe_vondsten_details])
        master_tickers = sorted(list(set(master_tickers)))
        with open(MASTER_FILE, 'w') as f:
            f.write(", ".join(master_tickers))

    # Altijd de dag-lijst opslaan
    with open(CURRENT_RUN_FILE, 'w') as f:
        f.write(", ".join(kwaliteits_vondsten))

    # --- TELEGRAM BERICHT OPSTELLEN ---
    if nieuwe_tickers_gevonden:
        bericht = "🚀 *Nieuwe Munger Kwaliteit ontdekt!*\n\n"
        bericht += "Toegevoegd aan Master-lijst (05xx):\n"
        bericht += "\n".join(nieuwe_vondsten_details)
    else:
        bericht = "✅ *Munger Scan Voltooid*\n\n"
        bericht += "Geen nieuwe aandelen gevonden die nog niet in de lijst stonden.\n"
    
    bericht += f"\n\n💎 *Huidige Elite Selectie ({len(kwaliteits_vondsten)}):*\n`{', '.join(kwaliteits_vondsten)}`"
    bericht += f"\n\n📚 *Totaal in Master-lijst:* {len(master_tickers)}"
    
    send_telegram_msg(bericht)

if __name__ == "__main__":
    main()
