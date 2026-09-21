"""
backfill_rank_kolommen.py
==========================
Eenmalig backfill-script: vult de wide columns combined_rank, roc_rank,
ey_rank, vc2_score in `selecties` met terugwerkende kracht vanuit de
bestaande JSON `parameters`-kolom.

Achtergrond: db_logger.py's _KOLOM_WHITELIST miste deze 4 kolomnamen tot
2026-09-21 (bot_01greenblatt.py / bot_01oshaughnessy.py). Elke insert vóór
die fix schreef de waarde daardoor enkel weg in de JSON parameters-kolom,
nooit in de eigen wide column -- ontdekt via weekly_db_opvolg.py's
haal_laatste_rank_scores(), die deze wide columns leest en voor alle 4
velden n=0 opleverde in de weekly-correlatie-analyse. Dit script haalt die
historische waarden alsnog uit de JSON en zet ze in de wide column, zodat
ze bruikbaar worden zonder op nieuwe weken te moeten wachten.

Gebruikt ctid om exact de gelezen rij te updaten (i.p.v. de combinatie
ticker+datum+strategie, want die is niet gegarandeerd uniek -- zie de
duplicaten die analyse_selecties.py al blootlegde in de selecties-tabel).

Read-only dry-run by default; --apply om effectief te schrijven.

Vereist: SUPABASE_DB_URL env var (zelfde secret als db_logger.py).

Gebruik:
    python backfill_rank_kolommen.py             # dry-run, toont wat er zou gebeuren
    python backfill_rank_kolommen.py --apply      # voert de UPDATE's effectief uit
    python backfill_rank_kolommen.py --apply --limiet-preview 50
"""

import argparse
import json
import os
import sys

import psycopg2
import psycopg2.extras

_DB_URL_ENV = "SUPABASE_DB_URL"

# Exact dezelfde 4 kolommen die op 2026-09-21 aan db_logger.py's
# _KOLOM_WHITELIST zijn toegevoegd.
KOLOMMEN = ["combined_rank", "roc_rank", "ey_rank", "vc2_score"]


def _get_connection():
    db_url = os.environ.get(_DB_URL_ENV, "")
    db_url = db_url.strip().strip('"').strip("'")
    if not db_url:
        sys.exit(f"Fout: env var {_DB_URL_ENV} ontbreekt.")
    if not (db_url.startswith("postgresql://") or db_url.startswith("postgres://")):
        sys.exit(f"Fout: {_DB_URL_ENV} lijkt geen geldige connectiestring.")
    return psycopg2.connect(db_url)


def haal_kandidaten(conn):
    """
    Haalt alle rijen op waar minstens 1 van de 4 doelkolommen NULL is
    en er een parameters-JSON aanwezig is om uit te backfillen. ctid
    wordt meegenomen om nadien exact deze rij te updaten.
    """
    voorwaarde = " OR ".join(f"{k} IS NULL" for k in KOLOMMEN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT ctid, ticker, datum, strategie, parameters
            FROM selecties
            WHERE parameters IS NOT NULL AND ({voorwaarde})
            """
        )
        return cur.fetchall()


def bepaal_updates(rij):
    """
    Geeft {kolomnaam: waarde} terug voor de kolommen die deze rij effectief
    kan backfillen: aanwezig in de JSON parameters en niet None.
    """
    parameters = rij["parameters"]
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except (TypeError, ValueError):
            return {}
    if not isinstance(parameters, dict):
        return {}

    updates = {}
    for kolom in KOLOMMEN:
        if kolom in parameters and parameters[kolom] is not None:
            updates[kolom] = parameters[kolom]
    return updates


def main():
    parser = argparse.ArgumentParser(
        description="Eenmalige backfill van combined_rank/roc_rank/ey_rank/vc2_score "
                    "vanuit de JSON parameters-kolom naar hun wide columns in `selecties`."
    )
    parser.add_argument("--apply", action="store_true",
                        help="Voer de UPDATE's effectief uit (zonder deze vlag: dry-run, enkel tellen/tonen)")
    parser.add_argument("--limiet-preview", type=int, default=20,
                        help="Aantal voorbeeldrijen om te tonen in dry-run (default 20)")
    args = parser.parse_args()

    conn = _get_connection()
    try:
        kandidaten = haal_kandidaten(conn)
        print(f"{len(kandidaten)} rijen gevonden met minstens 1 lege doelkolom en parameters-JSON.")

        per_kolom_teller = {k: 0 for k in KOLOMMEN}
        te_updaten = []
        for rij in kandidaten:
            updates = bepaal_updates(rij)
            if not updates:
                continue
            for k in updates:
                per_kolom_teller[k] += 1
            te_updaten.append((rij, updates))

        print("\nAantal rijen dat effectief backfillbaar is, per kolom:")
        for k, n in per_kolom_teller.items():
            print(f"  {k}: {n}")
        print(f"\nTotaal aantal rijen met minstens 1 backfillbare kolom: {len(te_updaten)}")

        if not args.apply:
            aantal_preview = min(args.limiet_preview, len(te_updaten))
            print(f"\n-- DRY RUN -- geen wijzigingen doorgevoerd. Voorbeeld van de eerste "
                  f"{aantal_preview} rijen:")
            for rij, updates in te_updaten[:aantal_preview]:
                print(f"  {rij['ticker']:>10}  {rij['strategie']:<20}  {rij['datum']}  -> {updates}")
            print("\nHerroep met --apply om deze wijzigingen effectief door te voeren.")
            return

        aantal_ok = 0
        with conn.cursor() as cur:
            for rij, updates in te_updaten:
                set_zin = ", ".join(f"{k} = %s" for k in updates)
                waarden = list(updates.values()) + [rij["ctid"]]
                try:
                    cur.execute(
                        f"UPDATE selecties SET {set_zin} WHERE ctid = %s",
                        waarden,
                    )
                except Exception as exc:
                    conn.rollback()
                    print(f"  waarschuwing: update voor {rij['ticker']} ({rij['datum']}) mislukt: {exc}")
                else:
                    conn.commit()
                    aantal_ok += 1

        print(f"\n{aantal_ok}/{len(te_updaten)} rijen succesvol bijgewerkt.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
