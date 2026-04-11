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

def main():
    # Lijst met al je tickerbestanden
    all_files = [
        'tickers_01.txt', 
        'tickers_02.txt', 
        'tickers_03.txt', 
        'tickers_04.txt', 
        'tickers_05.txt',
        'tickers_06.txt',
        'tickers_07.txt',
        'tickers_08.txt',
        'tickers_09.txt'
    ]
    
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

    # Maak DataFrame en sorteer
    df_res = pd.DataFrame(alle_data)
    
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
