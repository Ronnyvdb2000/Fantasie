import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv

# Laad omgevingsvariabelen
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID:
        print("Telegram configuratie ontbreekt.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram fout: {e}")

def haal_week_performance(ticker_list):
    results = []
    for t in ticker_list:
        try:
            # Haal 5 dagen aan data op
            df = yf.download(t, period="5d", progress=False)
            if df.empty or len(df) < 2:
                continue

            # Voorkom 'identically-labeled' fout door om te zetten naar pure getallen
            start_prijs = float(df['Close'].iloc[0])
            eind_prijs = float(df['Close'].iloc[-1])

            if start_prijs > 0:
                perc = ((eind_prijs - start_prijs) / start_prijs) * 100
                results.append({'ticker': t, 'perf': float(perc)})
        except Exception as e:
            print(f"Fout bij ticker {t}: {e}")
    return results

def bouw_bestandslijst():
    """
    Bouwt de volledige lijst van tickerbestanden:
      - tickers_01.txt t/m tickers_09.txt   (originele reeks)
      - tickers_041a.txt t/m tickers_059a.txt (nieuwe reeks, nummers kunnen ontbreken)
    Niet-bestaande bestanden worden verderop gewoon overgeslagen.
    """
    bestanden = []

    # Originele reeks: 01 t/m 09
    for n in range(1, 10):
        bestanden.append(f"tickers_{n:02d}.txt")

    # Nieuwe reeks: 041a t/m 059a
    for n in range(41, 60):
        bestanden.append(f"tickers_{n:03d}a.txt")

    return bestanden

def main():
    all_files = bouw_bestandslijst()

    alle_data = []

    for f_name in all_files:
        if os.path.exists(f_name):
            with open(f_name, 'r') as f:
                # Lees tickers en maak ze schoon
                tickers = [t.strip() for t in f.read().split(',') if t.strip()]
                if tickers:
                    print(f"Scannen van {f_name}...")
                    alle_data.extend(haal_week_performance(tickers))
        else:
            print(f"Bestand {f_name} niet gevonden, overslaan.")

    if not alle_data:
        stuur_telegram("📊 *Wekelijks Rapport:* Geen data gevonden om te analyseren.")
        return

    # Maak DataFrame en verwijder eventuele dubbele tickers (kunnen in meerdere lijsten voorkomen)
    df_res = pd.DataFrame(alle_data)
    df_res = df_res.drop_duplicates(subset='ticker', keep='first')

    # Sorteer op 'perf' kolom (simpele numerieke sortering)
    df_res = df_res.sort_values(by='perf', ascending=False)

    top_10 = df_res.head(10)
    bottom_10 = df_res.tail(10)

    rapport = "🏆 *WEKELIJKSE HALL OF FAME & SHAME*\n"
    rapport += "----------------------------------\n\n"

    rapport += "🚀 *TOP PERFORMERS (DEZE WEEK):*\n"
    for _, row in top_10.iterrows():
        rapport += f"• `{row['ticker']}` : +{row['perf']:.2f}%\n"

    rapport += "\n🔻 *GROOTSTE DALERS (DEZE WEEK):*\n"
    for _, row in bottom_10.iterrows():
        rapport += f"• `{row['ticker']}` : {row['perf']:.2f}%\n"

    rapport += "\n💡 *Tip:* Check of de stijgers een RSI-oververhitting vertonen voordat je actie onderneemt op Bolero!"

    stuur_telegram(rapport)
    print("Rapport verzonden naar Telegram.")

if __name__ == "__main__":
    main()
