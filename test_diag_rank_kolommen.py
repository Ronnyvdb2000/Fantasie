"""
test_diag_rank_kolommen.py
============================
Tijdelijk diagnostisch scriptje (workflow_dispatch, self-deletend via
test_diag_rank_kolommen.yml) om uit te zoeken waarom niveau_combined_rank/
roc_rank/ey_rank/vc2_score in de weekly-correlatie-analyse nog steeds een
kleine n tonen (4/4/4/3 op 2026-09-21) ondanks:
  1. de whitelist-fix in db_logger.py (2026-09-21)
  2. de backfill van 150 rijen per kolom in `selecties` (2026-09-21)
  3. de per-veld-query-fix in weekly_db_opvolg.py's
     haal_laatste_rank_scores() (2026-09-21)

Test 2 hypotheses, puur read-only (geen schrijfacties naar de databank):

  H1 (geen bug, gewoon weinig overlap): van de tickers met een gekende
     rank-waarde in `selecties`, komen er weinig voor in
     `weekly_topper_parameters` (ongeacht periode).

  H2 (verwachte historische leemte, geen actieve bug): rijen in
     `weekly_topper_parameters` van VOOR de fix-datum hebben terecht NULL,
     want die zijn geschreven toen de bug nog actief was. Enkel rijen NA
     de fix-datum zouden de waarde correct moeten meekrijgen.

Vereist: SUPABASE_DB_URL env var.
"""

import os
import sys

import psycopg2

FIX_DATUM = "2026-09-21"
RANK_KOLOMMEN = ["combined_rank", "roc_rank", "ey_rank", "vc2_score"]


def _connect():
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip().strip('"').strip("'")
    if not db_url:
        sys.exit("Fout: SUPABASE_DB_URL ontbreekt.")
    return psycopg2.connect(db_url)


def main():
    conn = _connect()
    try:
        with conn.cursor() as cur:
            print("== A) Unieke tickers met een gekende rank-waarde in `selecties` ==")
            for kolom in RANK_KOLOMMEN:
                cur.execute(f"SELECT count(DISTINCT ticker) FROM selecties WHERE {kolom} IS NOT NULL")
                print(f"  {kolom}: {cur.fetchone()[0]}")

            unie_voorwaarde = " OR ".join(f"{k} IS NOT NULL" for k in RANK_KOLOMMEN)
            cur.execute(f"SELECT count(DISTINCT ticker) FROM selecties WHERE {unie_voorwaarde}")
            unie_aantal = cur.fetchone()[0]
            print(f"  UNIE (minstens 1 van de 4): {unie_aantal}")

            cur.execute(f"SELECT DISTINCT ticker FROM selecties WHERE {unie_voorwaarde}")
            bekende_tickers = sorted(r[0] for r in cur.fetchall())
            print(f"  Tickers: {bekende_tickers}")

            print("\n== B) `weekly_topper_parameters` -- totalen ==")
            cur.execute(
                "SELECT count(*), min(week_startdatum), max(week_startdatum) "
                "FROM weekly_topper_parameters"
            )
            totaal, min_datum, max_datum = cur.fetchone()
            print(f"  Totaal rijen: {totaal}  (periode {min_datum} .. {max_datum})")
            for kolom in RANK_KOLOMMEN:
                cur.execute(f"SELECT count(*) FROM weekly_topper_parameters WHERE {kolom} IS NOT NULL")
                print(f"  {kolom} niet-NULL rijen (alle periodes): {cur.fetchone()[0]}")

            print(
                f"\n== C) H1-check: hoeveel van de {unie_aantal} bekende tickers "
                f"komen ooit voor in weekly_topper_parameters? =="
            )
            if bekende_tickers:
                cur.execute(
                    "SELECT count(DISTINCT ticker) FROM weekly_topper_parameters WHERE ticker = ANY(%s)",
                    (bekende_tickers,),
                )
                print(f"  Aanwezig in weekly_topper_parameters (ongeacht periode): {cur.fetchone()[0]}")
            else:
                print("  Geen bekende tickers gevonden, overgeslagen.")

            print(
                f"\n== D) H2-check: NA de fix ({FIX_DATUM}), krijgen bekende tickers "
                f"hun waarde wel correct mee? =="
            )
            for kolom in RANK_KOLOMMEN:
                cur.execute(
                    f"""
                    SELECT
                      count(*) FILTER (WHERE {kolom} IS NOT NULL) AS gevuld,
                      count(*) AS totaal
                    FROM weekly_topper_parameters
                    WHERE ticker = ANY(%s) AND week_startdatum >= %s
                    """,
                    (bekende_tickers, FIX_DATUM),
                )
                gevuld, totaal_na_fix = cur.fetchone()
                print(f"  {kolom}: {gevuld}/{totaal_na_fix} rijen NA de fix correct gevuld (van bekende tickers)")

            print(
                f"\n== E) Ter vergelijking: VOOR de fix ({FIX_DATUM}) -- "
                f"hier is NULL verwacht/normaal =="
            )
            for kolom in RANK_KOLOMMEN:
                cur.execute(
                    f"""
                    SELECT
                      count(*) FILTER (WHERE {kolom} IS NOT NULL) AS gevuld,
                      count(*) AS totaal
                    FROM weekly_topper_parameters
                    WHERE ticker = ANY(%s) AND week_startdatum < %s
                    """,
                    (bekende_tickers, FIX_DATUM),
                )
                gevuld, totaal_voor_fix = cur.fetchone()
                print(
                    f"  {kolom}: {gevuld}/{totaal_voor_fix} rijen VOOR de fix gevuld "
                    f"(verwacht: 0/{totaal_voor_fix})"
                )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
