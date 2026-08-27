"""
weekly_report_opvolg.py
========================
Parallelle "opvolg"-variant van weekly_report.py, bedoeld om NAAST de
bestaande wekelijkse job te draaien (eigen workflow: weekly_run_opvolg.yml,
eigen db-module: weekly_db_opvolg.py) zonder die te verstoren:

  - Stuurt GEEN volledig Hall of Fame-rapport (dat doet weekly_report.py al)
    -- enkel een korte statusmelding naar Telegram.
  - Schrijft NIET naar laatste_toppers.json / doet geen git-commit (dat
    bestand hoort bij weekly_report.py; gelijktijdig schrijven/committen
    vanuit twee jobs zou tot rommelige dubbele commits kunnen leiden).
  - Doet wel exact dezelfde scan (stijgers/dalers/neutrale steekproef) en
    schrijft die, plus de bijhorende technische/fundamentele parameters,
    weg naar de nieuwe Supabase-tabellen weekly_toppers en
    weekly_topper_parameters, voor latere correlatie-analyse.
"""

import yfinance as yf
import pandas as pd
import os
import json
import math
import random
import hashlib
import requests
from datetime import date, timedelta
from dotenv import load_dotenv

import weekly_db_opvolg as weekly_db

# Laad omgevingsvariabelen
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Aantal willekeurig getrokken "neutrale" tickers per lijst (naast de 5
# stijgers/5 dalers), zodat de latere correlatie-analyse niet enkel op de
# extremen van de weekprestatie-verdeling gebaseerd is (restriction of
# range zou de gevonden correlaties anders kunnen vertekenen).
NEUTRAAL_STEEKPROEF_GROOTTE = 10

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


def _veilig(x):
    """Zet een waarde om naar een echte Python-float, of None bij NaN/Inf/
    onbruikbare invoer (zelfde reden als _sanitize in db_logger.py/weekly_db.py)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def haal_technische_parameters(ticker, aantal_dagen=5, historiek_periode="1y"):
    """
    Berekent voor de laatste `aantal_dagen` handelsdagen een standaard set
    technische indicatoren (RSI14, MACD-histogram, ATR14%, afstand tot
    SMA50/SMA200, 20-daagse support/resistance, een ATR-gebaseerde stop).

    Dit zijn ONAFHANKELIJKE, standaardformules -- geen kopie van de exacte
    (soms bot-specifieke) berekening in bijvoorbeeld bot_00kr.py of
    bot_01xgboost.py. Voor deze correlatie-analyse maakt dat niet uit
    (consistentie tussen de 5 dagen is wat telt), maar de absolute waardes
    hoeven dus niet 1-op-1 overeen te komen met wat elders in het project
    gelogd wordt.
    """
    try:
        df = yf.download(ticker, period=historiek_periode, progress=False)
    except Exception as e:
        print(f"Technische parameters-fout bij {ticker}: {e}")
        return []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 60:
        return []

    close = df['Close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signaallijn = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signaallijn

    vorige_close = close.shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - vorige_close).abs(),
        (df['Low'] - vorige_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / close * 100

    df['sma50'] = close.rolling(50).mean()
    df['sma200'] = close.rolling(200).mean()
    df['dist_sma50_pct'] = (close - df['sma50']) / df['sma50'] * 100
    df['dist_sma200_pct'] = (close - df['sma200']) / df['sma200'] * 100

    df['support'] = df['Low'].rolling(20).min()
    df['resistance'] = df['High'].rolling(20).max()
    # ATR-stop: gangbare conventie (2x ATR onder de slotkoers), niet
    # per se identiek aan de stop-formule van een specifieke bot.
    df['stop'] = close - 2 * df['atr']

    laatste = df.tail(aantal_dagen)
    resultaat = []
    for datum, row in laatste.iterrows():
        resultaat.append({
            "datum": datum.date(),
            "close": _veilig(row['Close']),
            "rsi": _veilig(row['rsi']),
            "macd_hist": _veilig(row['macd_hist']),
            "atr_pct": _veilig(row['atr_pct']),
            "dist_sma50_pct": _veilig(row['dist_sma50_pct']),
            "dist_sma200_pct": _veilig(row['dist_sma200_pct']),
            "support": _veilig(row['support']),
            "resistance": _veilig(row['resistance']),
            "stop": _veilig(row['stop']),
        })
    return resultaat


def haal_fundamentals(ticker):
    """
    Haalt éénmalig (niet per dag) de fundamentele basisgegevens op via
    yfinance's .info. Sommige velden (roe_pct, current_ratio,
    revenue_growth_pct, eps_growth_pct, debt_to_ebitda) zijn quasi-constant
    binnen één week en worden dus letterlijk herhaald over de 5 dagen.
    Andere (shares, trailing_eps, book_value, ...) dienen enkel om er later
    per dag de prijs-afhankelijke ratio's (P/E, P/B, ...) mee te herberekenen.

    LET OP: debt_to_ebitda is hier een benadering via yfinance's
    'debtToEquity' (Debt/Equity, geen Net Debt/EBITDA zoals bot_01kasstr.py
    exact berekent) -- yfinance biedt geen kant-en-klare Net Debt/EBITDA in
    .info. eps_growth_pct is 'earningsQuarterlyGrowth', ook een benadering.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"Fundamentals-fout bij {ticker}: {e}")
        return {}

    def pct(key):
        v = info.get(key)
        return _veilig(v) * 100 if _veilig(v) is not None else None

    return {
        "shares": info.get("sharesOutstanding"),
        "trailing_eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
        "book_value": info.get("bookValue"),
        "sales_per_share": info.get("revenuePerShare"),
        "fcf": info.get("freeCashflow"),
        "dividend_rate": info.get("dividendRate"),
        "roe_pct": pct("returnOnEquity"),
        "current_ratio": _veilig(info.get("currentRatio")),
        "revenue_growth_pct": pct("revenueGrowth"),
        "eps_growth_pct": pct("earningsQuarterlyGrowth"),
        "debt_to_ebitda": _veilig(info.get("debtToEquity")),
    }


def bereken_daily_multiples(close, fundamentals):
    """Herberekent de prijs-afhankelijke ratio's voor één specifieke
    slotkoers (dus per dag opnieuw op te roepen met de close van die dag)."""
    if close is None or not fundamentals:
        return {}

    shares = fundamentals.get("shares")
    market_cap = close * shares if shares else None

    def ratio(teller_key):
        teller = fundamentals.get(teller_key)
        if not teller:
            return None
        return _veilig(close / teller)

    fcf = fundamentals.get("fcf")
    fcf_yield = _veilig(fcf / market_cap * 100) if fcf and market_cap else None
    dividend_rate = fundamentals.get("dividend_rate")
    dividend_yield = _veilig(dividend_rate / close * 100) if dividend_rate and close else None

    return {
        "market_cap": _veilig(market_cap),
        "pe_ratio": ratio("trailing_eps"),
        "forward_pe": ratio("forward_eps"),
        "pb_ratio": ratio("book_value"),
        "ps_ratio": ratio("sales_per_share"),
        "fcf_yield": fcf_yield,
        "dividend_yield": dividend_yield,
    }


def verwerk_weekly_db_export(alle_entries, week_startdatum):
    """
    alle_entries: lijst van dicts {lijst, ticker, beurs, type, rang, week_perf}
    voor stijgers, dalers ÉN de neutrale steekproef van deze run.

    Schrijft:
      1. weekly_toppers -- 1 rij per entry
      2. weekly_topper_parameters -- 5 rijen per UNIEKE ticker/beurs-combinatie
         (technische parameters per dag + eenmalig opgehaalde fundamentals +
         bevroren rank/score uit `selecties`)
    """
    if not alle_entries:
        return

    aantal_toppers = weekly_db.log_weekly_toppers([
        {
            "week_startdatum": week_startdatum,
            "lijst": e["lijst"],
            "ticker": e["ticker"],
            "beurs": e["beurs"],
            "type": e["type"],
            "rang": e.get("rang"),
            "week_perf": e["week_perf"],
        }
        for e in alle_entries
    ])
    print(f"{aantal_toppers} rijen weggeschreven naar weekly_toppers.")

    # Dedupliceren op (ticker, beurs): dezelfde ticker kan in principe niet
    # in dezelfde lijst zowel stijger als daler zijn, maar wel eens
    # voorkomen in de neutrale steekproef van een andere lijst-run; we
    # willen de dure technische/fundamentele ophaling niet dubbel doen.
    unieke_tickers = {}
    for e in alle_entries:
        unieke_tickers[(e["ticker"], e["beurs"])] = e

    rank_scores = weekly_db.haal_laatste_rank_scores([t for (t, _b) in unieke_tickers.keys()])

    parameter_rijen = []
    for (ticker, beurs), _e in unieke_tickers.items():
        technisch = haal_technische_parameters(ticker)
        if not technisch:
            print(f"Geen technische parameters voor {ticker}, overgeslagen.")
            continue

        fundamentals = haal_fundamentals(ticker)
        rank = rank_scores.get(ticker, {})

        for i, dag in enumerate(technisch, start=1):
            multiples = bereken_daily_multiples(dag["close"], fundamentals)
            rij = {
                "week_startdatum": week_startdatum,
                "ticker": ticker,
                "beurs": beurs,
                "datum": dag["datum"],
                "dag_index": i,
                "close": dag["close"],
                "rsi": dag["rsi"],
                "macd_hist": dag["macd_hist"],
                "atr_pct": dag["atr_pct"],
                "dist_sma50_pct": dag["dist_sma50_pct"],
                "dist_sma200_pct": dag["dist_sma200_pct"],
                "support": dag["support"],
                "resistance": dag["resistance"],
                "stop": dag["stop"],
                **multiples,
                "roe_pct": fundamentals.get("roe_pct"),
                "current_ratio": fundamentals.get("current_ratio"),
                "revenue_growth_pct": fundamentals.get("revenue_growth_pct"),
                "eps_growth_pct": fundamentals.get("eps_growth_pct"),
                "debt_to_ebitda": fundamentals.get("debt_to_ebitda"),
                "piotroski_score": rank.get("piotroski_score"),
                "combined_rank": rank.get("combined_rank"),
                "roc_rank": rank.get("roc_rank"),
                "ey_rank": rank.get("ey_rank"),
                "vc2_score": rank.get("vc2_score"),
                "total_score": rank.get("total_score"),
            }
            parameter_rijen.append(rij)

    aantal_params = weekly_db.log_weekly_topper_parameters(parameter_rijen)
    print(f"{aantal_params} rijen weggeschreven naar weekly_topper_parameters.")


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
    "tickers_041a.txt": "041 Benelux Ierland",
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
    "tickers_055a.txt": "054 CBoe", 
    "tickers_056a.txt": "054 NYSE int", 
    "tickers_057a.txt": "054 NYSE", 
    "tickers_058a.txt": "054 TSXV", 
    "tickers_059a.txt": "054 Osstenrijk Slovenie Slovakije",   
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

    # Maandag van de huidige week -- gedeelde sleutel tussen weekly_toppers
    # en weekly_topper_parameters.
    vandaag = date.today()
    week_startdatum = vandaag - timedelta(days=vandaag.weekday())

    db_entries = []  # stijgers + dalers + neutrale steekproef, voor weekly_db_opvolg
    aantal_lijsten_verwerkt = 0

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
        rest_na_top = df_lijst.drop(top_5.index)
        bottom_5 = rest_na_top.tail(5)

        label = label_voor(f_name)
        aantal_lijsten_verwerkt += 1

        for rang, (_, row) in enumerate(top_5.iterrows(), start=1):
            db_entries.append({
                "lijst": f_name, "ticker": row['ticker'], "beurs": label,
                "type": "stijger", "rang": rang, "week_perf": float(row['perf']),
            })

        for rang, (_, row) in enumerate(bottom_5.iterrows(), start=1):
            db_entries.append({
                "lijst": f_name, "ticker": row['ticker'], "beurs": label,
                "type": "daler", "rang": rang, "week_perf": float(row['perf']),
            })

        # Neutrale steekproef: willekeurige tickers uit deze lijst die noch
        # stijger noch daler waren, als referentiepunt voor de latere
        # correlatie-analyse (anders bestaat de dataset enkel uit de
        # extremen van de weekprestatie-verdeling -- restriction of range).
        # Seed op (week, lijst): reproduceerbaar bij een herrun binnen
        # dezelfde week, geen cherry-picking achteraf. LET OP: Python's
        # ingebouwde hash() is sinds 3.3 gerandomiseerd per proces (PYTHONHASHSEED)
        # en zou dus bij elke nieuwe GitHub Actions-run een andere seed geven --
        # hashlib.md5 is hier bewust gebruikt omdat die WEL stabiel is
        # over processen/runs heen.
        rest_na_bottom = rest_na_top.drop(bottom_5.index)
        steekproef_grootte = min(NEUTRAAL_STEEKPROEF_GROOTTE, len(rest_na_bottom))
        if steekproef_grootte > 0:
            seed_bron = f"{week_startdatum.isoformat()}|{f_name}"
            seed = int(hashlib.md5(seed_bron.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            neutrale_indices = rng.sample(list(rest_na_bottom.index), steekproef_grootte)
            for idx in neutrale_indices:
                row = rest_na_bottom.loc[idx]
                db_entries.append({
                    "lijst": f_name, "ticker": row['ticker'], "beurs": label,
                    "type": "neutraal", "rang": None, "week_perf": float(row['perf']),
                })

    if not db_entries:
        stuur_telegram("📊 *Opvolg-analyse:* geen data gevonden om te analyseren.")
        return

    try:
        verwerk_weekly_db_export(db_entries, week_startdatum)
        aantal_stijgers = sum(1 for e in db_entries if e["type"] == "stijger")
        aantal_dalers = sum(1 for e in db_entries if e["type"] == "daler")
        aantal_neutraal = sum(1 for e in db_entries if e["type"] == "neutraal")
        stuur_telegram(
            f"📈 *Opvolg-analyse voltooid* ({week_startdatum.isoformat()})\n"
            f"{aantal_lijsten_verwerkt} lijsten verwerkt — "
            f"{aantal_stijgers} stijgers, {aantal_dalers} dalers, "
            f"{aantal_neutraal} neutraal weggeschreven naar weekly_toppers/weekly_topper_parameters."
        )
    except Exception as e:
        # Een DB-fout hier mag het proces niet laten crashen -- enkel loggen
        # en ook expliciet naar Telegram melden, want er is geen ander
        # rapport dat deze run zichtbaar maakt.
        print(f"Fout bij wegschrijven naar weekly_toppers/weekly_topper_parameters: {e}")
        stuur_telegram(f"⚠️ *Opvolg-analyse mislukt:* {e}")

if __name__ == "__main__":
    main()
