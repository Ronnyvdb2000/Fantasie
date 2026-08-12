"""
analyse_selecties.py

Haalt alle records op uit de Supabase-tabel `selecties`, berekent per
strategie het rendement sinds selectiedatum (via yfinance), en toont
zowel het gemiddelde/mediaan rendement als een "getrimd" gemiddelde
(zonder de top/bottom X%) zodat losse "wow"-uitschieters de conclusie
niet vertekenen.

Vereist env var: SUPABASE_DB_URL  (zelfde secret als db_logger.py gebruikt)

Installatie:
    pip install psycopg2-binary yfinance pandas --break-system-packages

Gebruik:
    python analyse_selecties.py
    python analyse_selecties.py --strategie dm          # filter op 1 strategie
    python analyse_selecties.py --sinds 2026-05-01       # alleen selecties vanaf datum
    python analyse_selecties.py --trim 0.10               # trim 10% langs elke kant (default)
"""

import argparse
import os
import sys
from datetime import date, datetime

import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf


def haal_selecties_op(sinds: str | None, strategie: str | None) -> pd.DataFrame:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        sys.exit("Fout: env var SUPABASE_DB_URL is niet gezet.")

    query = "SELECT ticker, datum, strategie, beurs, koers, parameters FROM selecties WHERE 1=1"
    params = []
    if sinds:
        query += " AND datum >= %s"
        params.append(sinds)
    if strategie:
        query += " AND strategie = %s"
        params.append(strategie)
    query += " ORDER BY datum ASC"

    with psycopg2.connect(db_url) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        sys.exit("Geen records gevonden voor deze filter.")

    df = pd.DataFrame(rows)
    df["datum"] = pd.to_datetime(df["datum"]).dt.date
    return df


def bereken_rendementen(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt kolom 'rendement_pct' toe: huidige koers vs koers op selectiedatum."""
    resultaten = []
    tickers = df["ticker"].unique()

    print(f"Prijsdata ophalen voor {len(tickers)} tickers via yfinance...")

    vroegste_datum = df["datum"].min()
    hist_cache = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            hist = yf.download(
                ticker,
                start=vroegste_datum,
                end=date.today() + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=True,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist_cache[ticker] = hist
        except Exception as e:
            print(f"  waarschuwing: kon {ticker} niet ophalen ({e})")
            hist_cache[ticker] = None

        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} tickers verwerkt")

    laatste_koers_cache = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            continue

        # koers bij selectie: eerste beschikbare close op/na de selectiedatum
        na_selectie = hist[hist.index.date >= row["datum"]]
        if na_selectie.empty:
            continue
        koers_start = float(na_selectie["Close"].iloc[0])

        if ticker not in laatste_koers_cache:
            laatste_koers_cache[ticker] = float(hist["Close"].iloc[-1])
        koers_nu = laatste_koers_cache[ticker]

        if koers_start <= 0:
            continue

        rendement_pct = (koers_nu - koers_start) / koers_start * 100
        resultaten.append(
            {
                "ticker": ticker,
                "strategie": row["strategie"],
                "beurs": row.get("beurs"),
                "datum": row["datum"],
                "koers_start": koers_start,
                "koers_nu": koers_nu,
                "rendement_pct": rendement_pct,
            }
        )

    return pd.DataFrame(resultaten)


def getrimd_gemiddelde(reeks: pd.Series, trim: float) -> float:
    """Gemiddelde na het weglaten van de top/bottom `trim` fractie (bv. 0.10 = 10%)."""
    if len(reeks) < 5:
        return reeks.mean()
    return reeks.sort_values().iloc[int(len(reeks) * trim): len(reeks) - int(len(reeks) * trim)].mean()


def rapporteer(df: pd.DataFrame, trim: float):
    print("\n" + "=" * 70)
    print("RENDEMENT PER STRATEGIE (sinds selectiedatum tot vandaag)")
    print("=" * 70)

    for strat, groep in df.groupby("strategie"):
        n = len(groep)
        win_rate = (groep["rendement_pct"] > 0).mean() * 100
        gemiddeld = groep["rendement_pct"].mean()
        mediaan = groep["rendement_pct"].median()
        getrimd = getrimd_gemiddelde(groep["rendement_pct"], trim)
        beste = groep.nlargest(3, "rendement_pct")[["ticker", "datum", "rendement_pct"]]
        slechtste = groep.nsmallest(3, "rendement_pct")[["ticker", "datum", "rendement_pct"]]

        print(f"\n--- {strat}  (n={n}) ---")
        print(f"  win-rate:            {win_rate:5.1f}%")
        print(f"  gemiddeld rendement: {gemiddeld:6.2f}%")
        print(f"  mediaan rendement:   {mediaan:6.2f}%")
        print(f"  getrimd gemiddelde:  {getrimd:6.2f}%  (zonder top/bottom {int(trim*100)}%)")
        print(f"  top 3:")
        for _, r in beste.iterrows():
            print(f"    {r['ticker']:>8}  {r['datum']}  {r['rendement_pct']:+.2f}%")
        print(f"  bottom 3:")
        for _, r in slechtste.iterrows():
            print(f"    {r['ticker']:>8}  {r['datum']}  {r['rendement_pct']:+.2f}%")

    print("\n" + "=" * 70)
    print("TOTAAL (alle strategieen samen)")
    print("=" * 70)
    n = len(df)
    win_rate = (df["rendement_pct"] > 0).mean() * 100
    gemiddeld = df["rendement_pct"].mean()
    mediaan = df["rendement_pct"].median()
    getrimd = getrimd_gemiddelde(df["rendement_pct"], trim)
    print(f"n={n}  win-rate={win_rate:.1f}%  gemiddeld={gemiddeld:.2f}%  "
          f"mediaan={mediaan:.2f}%  getrimd={getrimd:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyseer selecties-tabel per strategie")
    parser.add_argument("--strategie", help="Filter op 1 strategie (bv. dm, cs, vcp)")
    parser.add_argument("--sinds", help="Alleen selecties vanaf deze datum (YYYY-MM-DD)")
    parser.add_argument("--trim", type=float, default=0.10,
                         help="Fractie om te trimmen langs elke kant voor getrimd gemiddelde (default 0.10)")
    parser.add_argument("--csv", help="Optioneel: schrijf ruwe resultaten weg naar dit csv-pad")
    args = parser.parse_args()

    df = haal_selecties_op(args.sinds, args.strategie)
    print(f"{len(df)} selecties opgehaald uit Supabase.")

    resultaten = bereken_rendementen(df)
    if resultaten.empty:
        sys.exit("Geen rendementen kunnen berekenen (geen prijsdata gevonden).")

    if args.csv:
        resultaten.to_csv(args.csv, index=False)
        print(f"Ruwe resultaten weggeschreven naar {args.csv}")

    rapporteer(resultaten, args.trim)


if __name__ == "__main__":
    main()
