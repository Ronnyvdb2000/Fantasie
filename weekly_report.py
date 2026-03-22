import yfinance as yf
import pandas as pd
import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"})

def haal_week_performance(ticker_list):
    results = []
    for t in ticker_list:
        try:
            df = yf.download(t, period="5d", progress=False)
            if len(df) < 2: continue
            start_prijs = df['Close'].iloc[0]
            eind_prijs = df['Close'].iloc[-1]
            perc = ((eind_prijs - start_prijs) / start_prijs) * 100
            results.append({'ticker': t, 'perf': perc})
        except: pass
    return results

def main():
    all_files = ['tickers_benelux.txt', 'tickers_parijs.txt', 'tickers_defensie.txt', 'tickers_power.txt', 'tickers_metalen.txt']
    alle_data = []

    for f_name in all_files:
        if os.path.exists(f_name):
            with open(f_name, 'r') as f:
                tickers = [t.strip() for t in f.read().split(',') if t.strip()]
                alle_data.extend(haal_week_performance(tickers))

    if not alle_data: return

    # Sorteren op performance
    df_res = pd.DataFrame(alle_data).sort_values(by='perf', ascending=False)
    
    top_3 = df_res.head(3)
    bottom_3 = df_res.tail(3)

    rapport = "🏆 *WEKELIJKSE HALL OF FAME & SHAME*\n"
    rapport += "----------------------------------\n\n"
    
    rapport += "🚀 *TOP PERFORMERS (DEZE WEEK):*\n"
    for _, row in top_3.iterrows():
        rapport += f"• `{row['ticker']}` : +{row['perf']:.2f}%\n"

    rapport += "\n🔻 *GROOTSTE DALERS (DEZE WEEK):*\n"
    for _, row in bottom_3.iterrows():
        rapport += f"• `{row['ticker']}` : {row['perf']:.2f}%\n"

    rapport += "\n💡 *Tip:* Kijk of de stijgers een 'Golden Cross' naderen op de dagelijkse grafiek!"
    
    stuur_telegram(rapport)

if __name__ == "__main__":
    main()
