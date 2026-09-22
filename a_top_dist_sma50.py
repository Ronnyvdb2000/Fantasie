"""
top_dist_sma50.py
===================
Ad hoc, read-only scriptje: toont de tickers met de laagste (meest
negatieve) `pct_from_ma50` uit `generieke_technicals`, gemeten binnen de
laatste --dagen dagen (default 15). Dit is exact het signaal dat in
analyseer_forward_correlaties.py naar voren kwam als sterkste (nog steeds
zwakke) 3-wekense voorspeller: begin_dist_sma50_pct, rho=-0,159,
p_aangepast=0,0001, n=797 (2026-09-22-run). Negatief rho betekent: hoe
verder ONDER het 50-daags gemiddelde bij meting, hoe iets beter het
rendement over de volgende 3 weken -- vandaar dat dit script AFLOPEND
sorteert op de meest negatieve pct_from_ma50 (verst onder het gemiddelde).

BELANGRIJK, lees dit voor gebruik: dit is GEEN koopadvies en geen
gevalideerde stock-picker. Het onderliggende signaal verklaart ~2,5% van
de variantie (R² = rho^2) in 3-wekenrendement, 1 keer gemeten op 797
ticker-weken. Op individueel-aandeelniveau is dit nog steeds overwegend
ruis -- dit script toont enkel WELKE tickers vandaag toevallig aan het
criterium voldoen, niet dat ze "goed gaan presteren". Zie
fantasie-trading-bots.md voor de volledige context en kanttekeningen.

Toont ook, ter context, welke strategieen (uit `selecties`) deze ticker
in dezelfde periode zelf ook al selecteerden -- zo zie je meteen overlap
met a_trade/a_trade_combi's eigen logica.

Vereist env var: SUPABASE_DB_URL

Gebruik:
    python top_dist_sma50.py                  # default: laatste 15 dagen, top 10
    python top_dist_sma50.py --dagen 10 --top 20
    python top_dist_sma50.py --csv resultaat.csv
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
import psycopg2


def _get_connection():
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip().strip('"').strip("'")
    if not db_url:
        sys.exit("Fout: env var SUPABASE_DB_URL is niet gezet.")
    return psycopg2.connect(db_url)


def haal_recente_technicals(conn, dagen: int) -> pd.DataFrame:
    """Per ticker de meest recente rij uit generieke_technicals binnen de
    laatste `dagen` dagen (DISTINCT ON, nieuwste datum per ticker wint).

    `datum` staat, net als in `selecties`, als ISO-tekst in de databank
    (geen native date-type) -- vandaar .isoformat() i.p.v. het date-object
    zelf door te geven. Werkt correct omdat ISO-datumstrings (YYYY-MM-DD)
    ook lexicografisch correct sorteren/vergelijken.
    """
    sinds = (date.today() - timedelta(days=dagen)).isoformat()
    query = """
        SELECT DISTINCT ON (ticker)
            ticker, datum, pct_from_ma50, pct_from_ma200, rsi14, atr14_pct,
            ibs, vol_ratio_20d, pct_from_high52w
        FROM generieke_technicals
        WHERE datum >= %s AND pct_from_ma50 IS NOT NULL
        ORDER BY ticker, datum DESC
    """
    return pd.read_sql(query, conn, params=(sinds,))


def haal_recente_strategieen(conn, tickers, dagen: int) -> dict:
    """Per ticker: welke strategieen deze ticker binnen dezelfde periode
    zelf ook al selecteerden, ter context (overlap met a_trade e.a.)."""
    if not tickers:
        return {}
    sinds = date.today() - timedelta(days=dagen)
    query = """
        SELECT DISTINCT ticker, strategie
        FROM selecties
        WHERE ticker = ANY(%s) AND datum >= %s
    """
    df = pd.read_sql(query, conn, params=(list(tickers), sinds.isoformat()))
    resultaat = {}
    for ticker, groep in df.groupby("ticker"):
        resultaat[ticker] = sorted(groep["strategie"].tolist())
    return resultaat


def main():
    parser = argparse.ArgumentParser(
        description="Toont de tickers met de laagste pct_from_ma50 uit generieke_technicals "
                    "binnen de laatste N dagen (het sterkste, nog steeds zwakke, signaal uit "
                    "analyseer_forward_correlaties.py). Geen koopadvies."
    )
    parser.add_argument("--dagen", type=int, default=15,
                        help="Hoe recent de technicals-meting moet zijn (default 15)")
    parser.add_argument("--top", type=int, default=10, help="Aantal tickers om te tonen (default 10)")
    parser.add_argument("--csv", help="Optioneel: schrijf het resultaat weg naar dit csv-pad")
    args = parser.parse_args()

    conn = _get_connection()
    try:
        df = haal_recente_technicals(conn, args.dagen)
        if df.empty:
            sys.exit(f"Geen data gevonden in generieke_technicals binnen de laatste {args.dagen} dagen.")

        df = df.sort_values("pct_from_ma50").head(args.top).reset_index(drop=True)
        strategieen = haal_recente_strategieen(conn, df["ticker"].tolist(), args.dagen)
    finally:
        conn.close()

    print(f"\nTop {len(df)} tickers, laagste pct_from_ma50, binnen de laatste {args.dagen} dagen:")
    print(
        "(negatief = onder het 50-daags gemiddelde -- signaal uit analyseer_forward_correlaties.py, "
        "rho=-0,159 op 3-wekenhorizon, R2~2,5%. GEEN koopadvies, enkel een filter.)\n"
    )
    for _, r in df.iterrows():
        strat_lijst = ", ".join(strategieen.get(r["ticker"], [])) or "-"
        print(
            f"  {r['ticker']:>10}  datum={r['datum']}  "
            f"pct_from_ma50={r['pct_from_ma50']:+.2f}%  "
            f"RSI14={r['rsi14']:.1f}  IBS={r['ibs']:.2f}  "
            f"ATR%={r['atr14_pct']:.2f}  "
            f"recent geselecteerd door: {strat_lijst}"
        )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWeggeschreven naar {args.csv}")


if __name__ == "__main__":
    main()
