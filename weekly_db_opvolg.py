"""
weekly_db_opvolg.py
============
Schrijft de wekelijkse Hall-of-Fame-analyse weg naar twee Supabase-tabellen
(zie migratie_weekly_topper_tabellen.sql):

  - weekly_toppers            : 1 rij per ticker (stijger/daler/neutraal)
                                 met de weekprestatie.
  - weekly_topper_parameters  : 5 rijen per ticker (1 per handelsdag van
                                 de week) met de parameterwaarden van die
                                 dag.

Gebruikt dezelfde SUPABASE_DB_URL-secret en hetzelfde connectiepatroon als
db_logger.py.
"""

import os
import math
import logging
import psycopg2

logger = logging.getLogger(__name__)

_DB_URL_ENV = "SUPABASE_DB_URL"


def _get_connection():
    db_url = os.environ.get(_DB_URL_ENV, "")
    db_url = db_url.strip().strip('"').strip("'")
    if not db_url:
        raise RuntimeError(
            f"Omgevingsvariabele {_DB_URL_ENV} ontbreekt. "
            f"Voeg ze toe als GitHub Actions secret en geef ze door aan de workflow."
        )
    if not (db_url.startswith("postgresql://") or db_url.startswith("postgres://")):
        raise RuntimeError(
            f"{_DB_URL_ENV} lijkt geen geldige connectiestring: moet beginnen "
            f"met 'postgresql://' of 'postgres://'."
        )
    return psycopg2.connect(db_url)


def _sanitize(value):
    """NaN/Inf zijn geen geldige Postgres NUMERIC-waarden -> NULL."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def log_weekly_toppers(rows: list) -> int:
    """
    rows: lijst van dicts met keys:
        week_startdatum, lijst, ticker, beurs, type ('stijger'/'daler'/'neutraal'),
        rang, week_perf
    Retourneert het aantal succesvol weggeschreven rijen.
    """
    if not rows:
        return 0
    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("weekly_db.log_weekly_toppers: geen connectie: %s", exc)
        return 0

    aantal_ok = 0
    try:
        with conn.cursor() as cur:
            for r in rows:
                try:
                    cur.execute(
                        """
                        INSERT INTO weekly_toppers
                            (week_startdatum, lijst, ticker, beurs, type, rang, week_perf)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, beurs, week_startdatum, type) DO UPDATE SET
                            rang      = EXCLUDED.rang,
                            week_perf = EXCLUDED.week_perf,
                            lijst     = EXCLUDED.lijst
                        """,
                        (
                            r["week_startdatum"],
                            r["lijst"],
                            r["ticker"],
                            r.get("beurs"),
                            r["type"],
                            r.get("rang"),
                            _sanitize(r["week_perf"]),
                        ),
                    )
                except Exception as exc:
                    conn.rollback()
                    logger.error(
                        "weekly_db.log_weekly_toppers: rij overgeslagen (%s): %s",
                        r.get("ticker"), exc,
                    )
                else:
                    conn.commit()
                    aantal_ok += 1
    finally:
        conn.close()
    return aantal_ok


# Kolommen van weekly_topper_parameters, exclusief de sleutel-/metakolommen
# (week_startdatum, ticker, beurs, datum, dag_index) die apart doorgegeven worden.
_PARAM_KOLOMMEN = [
    "close", "rsi", "macd_hist", "atr_pct", "dist_sma50_pct", "dist_sma200_pct",
    "support", "resistance", "stop",
    "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "fcf_yield",
    "dividend_yield", "market_cap",
    "roe_pct", "current_ratio", "revenue_growth_pct", "eps_growth_pct",
    "debt_to_ebitda",
    "piotroski_score", "combined_rank", "roc_rank", "ey_rank",
    "vc2_score", "total_score",
]

_META_KOLOMMEN = ["week_startdatum", "ticker", "beurs", "datum", "dag_index"]


def log_weekly_topper_parameters(rows: list) -> int:
    """
    rows: lijst van dicts met de _META_KOLOMMEN + (een subset van) _PARAM_KOLOMMEN.
    Ontbrekende parameterkolommen worden als NULL weggeschreven.
    Retourneert het aantal succesvol weggeschreven rijen.
    """
    if not rows:
        return 0
    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("weekly_db.log_weekly_topper_parameters: geen connectie: %s", exc)
        return 0

    alle_kolommen = _META_KOLOMMEN + _PARAM_KOLOMMEN
    update_zin = ", ".join(f"{k} = EXCLUDED.{k}" for k in ["dag_index"] + _PARAM_KOLOMMEN)
    placeholders = ", ".join(["%s"] * len(alle_kolommen))

    aantal_ok = 0
    try:
        with conn.cursor() as cur:
            for r in rows:
                try:
                    waarden = [_sanitize(r.get(k)) for k in alle_kolommen]
                    cur.execute(
                        f"""
                        INSERT INTO weekly_topper_parameters ({", ".join(alle_kolommen)})
                        VALUES ({placeholders})
                        ON CONFLICT (ticker, beurs, datum) DO UPDATE SET {update_zin}
                        """,
                        waarden,
                    )
                except Exception as exc:
                    conn.rollback()
                    logger.error(
                        "weekly_db.log_weekly_topper_parameters: rij overgeslagen (%s, %s): %s",
                        r.get("ticker"), r.get("datum"), exc,
                    )
                else:
                    conn.commit()
                    aantal_ok += 1
    finally:
        conn.close()
    return aantal_ok


def haal_laatste_rank_scores(tickers: list) -> dict:
    """
    Haalt per ticker de meest recente gekende rank/score-waarden op uit de
    bestaande `selecties`-tabel, ongeacht hoe oud. Deze velden zijn
    cross-sectioneel (afhankelijk van het hele universum op een specifieke
    dag) en worden daarom NIET per historische dag gereconstrueerd, maar
    bevroren herhaald over de 5 dagen van de week.

    Retourneert: {ticker: {"combined_rank":..., "roc_rank":..., "ey_rank":...,
                            "vc2_score":..., "total_score":..., "piotroski_score":...}}
    Tickers zonder gekende rij in `selecties` zitten niet in het resultaat
    (-> alle 6 velden blijven NULL bij het wegschrijven).
    """
    if not tickers:
        return {}
    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("weekly_db.haal_laatste_rank_scores: geen connectie: %s", exc)
        return {}

    resultaat = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (ticker)
                    ticker, combined_rank, roc_rank, ey_rank, vc2_score,
                    total_score, piotroski_score
                FROM selecties
                WHERE ticker = ANY(%s)
                ORDER BY ticker, datum DESC
                """,
                (list(tickers),),
            )
            for row in cur.fetchall():
                resultaat[row[0]] = {
                    "combined_rank": row[1],
                    "roc_rank": row[2],
                    "ey_rank": row[3],
                    "vc2_score": row[4],
                    "total_score": row[5],
                    "piotroski_score": row[6],
                }
    except Exception as exc:
        logger.error("weekly_db.haal_laatste_rank_scores: query mislukt: %s", exc)
    finally:
        conn.close()
    return resultaat
