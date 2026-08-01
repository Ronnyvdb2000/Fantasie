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

def stuur_telegram_lang(volledige_tekst, limiet=4000):
    """
    Telegram staat max. 4096 tekens per bericht toe. Bij een klein aantal
    lijsten past alles in één bericht; wordt de tekst toch te lang, dan
    wordt enkel dan opgeknipt in opeenvolgende berichten (nooit halverwege
    een lijst-sectie).
    """
    if len(volledige_tekst) <= limiet:
        stuur_telegram(volledige_tekst)
        return

    secties = volledige_tekst.split("\n\n\n")
    huidig = ""
    for sectie in secties:
        kandidaat = (huidig + "\n\n\n" + sectie) if huidig else sectie
        if len(kandidaat) > limiet:
            if huidig:
                stuur_telegram(huidig)
            huidig = sectie
        else:
            huidig = kandidaat
    if huidig:
        stuur_telegram(huidig)

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

# Vriendelijke namen voor de originele lijsten 01-09 (zoals gebruikt in de andere bot)
SECTOR_NAMEN = {
    "01": "Hoogland",
    "02": "Macrotrends",
    "03": "Beursbrink",
    "04": "Benelux",
    "05": "Parijs",
    "06": "Power & AI",
    "07": "Metalen",
    "08": "Defensie",
    "09": "Varia",
}

# Vriendelijke namen voor de bekende beurs-lijsten (enkel ter info in het rapport)
BEURS_NAMEN = {
    "tickers_041a.txt": "041 Benelux",
    "tickers_042a.txt": "042 Parijs",
    "tickers_043a.txt": "043 Frankfurt",
    "tickers_044a.txt": "044 Spanje/Portugal",
    "tickers_045a.txt": "045 Londen",
    "tickers_046a.txt": "046 Milaan",
    "tickers_047a.txt": "047 Toronto",
    "tickers_048a.txt": "048 Nasdaq/NYSE",
    "tickers_049a.txt": "049 Stockholm",
    "tickers_050a.txt": "050 Zurich",
    "tickers_051a.txt": "051 Warschau",
    "tickers_052a.txt": "052 Oslo",
    "tickers_053a.txt": "053 Kopenhagen",
    "tickers_054a.txt": "054 Helsinki",
}
for _nr, _naam in SECTOR_NAMEN.items():
    BEURS_NAMEN[f"tickers_{_nr}.txt"] = f"{_nr} {_naam}"

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

def label_voor(f_name):
    return BEURS_NAMEN.get(f_name, f_name.replace('.txt', ''))

def main():
    all_files = bouw_bestandslijst()

    secties = []

    for f_name in all_files:
        if not os.path.exists(f_name):
            print(f"Bestand {f_name} niet gevonden, overslaan.")
            continue

        with open(f_name, 'r') as f:
            tickers = [t.strip() for t in f.read().split(',') if t.strip()]

        if not tickers:
            continue

        print(f"Scannen van {f_name}...")
        data = haal_week_performance(tickers)

        if not data:
            continue

        df_lijst = pd.DataFrame(data).drop_duplicates(subset='ticker', keep='first')
        df_lijst = df_lijst.sort_values(by='perf', ascending=False)

        top_5 = df_lijst.head(5)
        # Sluit tickers die al in top_5 staan uit van de dalers (kan overlappen bij kleine lijsten)
        bottom_5 = df_lijst.drop(top_5.index).tail(5)

        label = label_voor(f_name)

        sectie = f"🏆 *{label}*\n"
        sectie += "🚀 Top 5 stijgers:\n"
        for _, row in top_5.iterrows():
            sectie += f"• `{row['ticker']}` : +{row['perf']:.2f}%\n"

        sectie += "🔻 Top 5 dalers:\n"
        for _, row in bottom_5.iterrows():
            sectie += f"• `{row['ticker']}` : {row['perf']:.2f}%\n"

        secties.append(sectie.rstrip())

    if not secties:
        stuur_telegram("📊 *Wekelijks Rapport:* Geen data gevonden om te analyseren.")
        return

    kop = "🏆 *WEKELIJKSE HALL OF FAME & SHAME — PER LIJST*\n"
    kop += "==================================\n\n"
    voet = "\n\n💡 *Tip:* Check of de stijgers een RSI-oververhitting vertonen voordat je actie onderneemt op Bolero!"

    volledig_rapport = kop + "\n\n\n".join(secties) + voet

    stuur_telegram_lang(volledig_rapport)
    print("Rapport verzonden naar Telegram.")

if __name__ == "__main__":
    main()
