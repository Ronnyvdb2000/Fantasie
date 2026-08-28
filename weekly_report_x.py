"""
weekly_report_x.py
===================
Variant van weekly_report.py die NIET de volledige a-lijsten (alle tickers
per beurs) scant, maar de reeds kwaliteitsgefilterde x-lijsten
(tickers_041x.txt t/m tickers_059x.txt, geproduceerd door bot_041mV2.py),
naast de ongewijzigde originele 01-09-lijsten.

Draait volledig los van weekly_report.py: eigen outputbestand
(laatste_toppers_x.json) zodat er geen git-conflict ontstaat als beide
workflows rond hetzelfde moment lopen.
"""

import yfinance as yf
import pandas as pd
import os
import json
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

            # yfinance geeft soms MultiIndex-kolommen terug, ook voor 1 ticker.
            # Zonder deze fix is df['Close'].iloc[0] een Series i.p.v. een getal,
            # en crasht float() daarop met "must be a string or a real number".
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Voorkom 'identically-labeled' fout door om te zetten naar pure getallen
            start_prijs = float(df['Close'].iloc[0])
            eind_prijs = float(df['Close'].iloc[-1])

            if pd.isna(start_prijs) or pd.isna(eind_prijs):
                continue

            if start_prijs > 0:
                perc = ((eind_prijs - start_prijs) / start_prijs) * 100
                if pd.isna(perc):
                    continue
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
# LET OP: 055-059 waren in het origineel foutief allemaal als "054" gelabeld
# -- hier gecorrigeerd naar de juiste nummers.
BEURS_NAMEN = {
    "tickers_041x.txt": "041 Benelux Ierland",
    "tickers_042x.txt": "042 Parijs",
    "tickers_043x.txt": "043 Frankfurt",
    "tickers_044x.txt": "044 Spanje/Portugal",
    "tickers_045x.txt": "045 Londen",
    "tickers_046x.txt": "046 Milaan",
    "tickers_047x.txt": "047 Toronto",
    "tickers_048x.txt": "048 Nasdaq/NYSE",
    "tickers_049x.txt": "049 Stockholm",
    "tickers_050x.txt": "050 Zurich",
    "tickers_051x.txt": "051 Warschau",
    "tickers_052x.txt": "052 Oslo",
    "tickers_053x.txt": "053 Kopenhagen",
    "tickers_054x.txt": "054 Helsinki",
    "tickers_055x.txt": "055 CBoe",
    "tickers_056x.txt": "056 NYSE int",
    "tickers_057x.txt": "057 NYSE",
    "tickers_058x.txt": "058 TSXV",
    "tickers_059x.txt": "059 Oostenrijk/Slovenië/Slowakije",
}

for _nr, _naam in SECTOR_NAMEN.items():
    BEURS_NAMEN[f"tickers_{_nr}.txt"] = f"{_nr} {_naam}"

def bouw_bestandslijst():
    """
    Bouwt de volledige lijst van tickerbestanden:
      - tickers_01.txt t/m tickers_09.txt    (originele reeks, ongewijzigd)
      - tickers_041x.txt t/m tickers_059x.txt (kwaliteitsgefilterde x-lijsten,
        i.p.v. de volledige a-lijsten -- nummers kunnen ontbreken)
    Niet-bestaande bestanden worden verderop gewoon overgeslagen.
    """
    bestanden = []

    # Originele reeks: 01 t/m 09
    for n in range(1, 10):
        bestanden.append(f"tickers_{n:02d}.txt")

    # Kwaliteitsreeks: 041x t/m 059x
    for n in range(41, 60):
        bestanden.append(f"tickers_{n:03d}x.txt")

    return bestanden

def label_voor(f_name):
    return BEURS_NAMEN.get(f_name, f_name.replace('.txt', ''))

def main():
    all_files = bouw_bestandslijst()

    secties = []
    toppers_export = []

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
            toppers_export.append({
                "beurs": label,
                "ticker": row['ticker'],
                "week_perf": round(float(row['perf']), 2),
            })

        sectie += "🔻 Top 5 dalers:\n"
        for _, row in bottom_5.iterrows():
            sectie += f"• `{row['ticker']}` : {row['perf']:.2f}%\n"

        secties.append(sectie.rstrip())

    if not secties:
        stuur_telegram("📊 *Wekelijks Rapport (kwaliteitslijsten):* Geen data gevonden om te analyseren.")
        return

    # Schrijf de top 5 stijgers per lijst weg. Eigen bestandsnaam (met _x)
    # zodat dit niet botst met weekly_report.py's laatste_toppers.json.
    with open("laatste_toppers_x.json", "w", encoding="utf-8") as f:
        json.dump(toppers_export, f, ensure_ascii=False, indent=2)
    print(f"{len(toppers_export)} toppers weggeschreven naar laatste_toppers_x.json")

    kop = "🏆 *WEKELIJKSE HALL OF FAME & SHAME — KWALITEITSLIJSTEN (x)*\n"
    kop += "==================================\n\n"
    voet = "\n\n💡 *Tip:* Check of de stijgers een RSI-oververhitting vertonen voordat je actie onderneemt op Bolero!"

    volledig_rapport = kop + "\n\n\n".join(secties) + voet

    stuur_telegram_lang(volledig_rapport)
    print("Rapport verzonden naar Telegram.")

if __name__ == "__main__":
    main()
