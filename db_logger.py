"""
db_logger.py
------------
Herbruikbare module om bot-selecties naar de Supabase-tabel `selecties`
te loggen, naast de bestaande Telegram/e-mail-notificaties.

Gebruik in een bot:

    from db_logger import log_selectie

    log_selectie(
        ticker="AAPL",
        datum="2026-08-04",
        strategie="bot_00kr",
        beurs="NASDAQ",
        koers=231.45,
        parameters={"rsi_period": 14, "threshold": 0.7},
    )

Vereist:
- pip install psycopg2-binary
- GitHub Actions secret SUPABASE_DB_URL (Project Settings -> Database -> Connection string,
  gebruik bij voorkeur de "Connection pooling" URI voor gebruik vanuit CI/CD)
"""

import os
import json
import logging
from datetime import date, datetime

import psycopg2

logger = logging.getLogger(__name__)

_DB_URL_ENV = "SUPABASE_DB_URL"


def _get_connection():
    db_url = os.environ.get(_DB_URL_ENV)
    if not db_url:
        raise RuntimeError(
            f"Omgevingsvariabele {_DB_URL_ENV} ontbreekt. "
            f"Voeg ze toe als GitHub Actions secret en geef ze door aan de workflow."
        )
    return psycopg2.connect(db_url)


def log_selectie(
    ticker: str,
    datum,
    strategie: str,
    beurs: str = None,
    koers: float = None,
    parameters: dict = None,
) -> bool:
    """
    Schrijft één selectie weg naar de `selecties`-tabel in Supabase.

    - datum: str (bv. "2026-08-04") of een date/datetime object.
    - parameters: gewone dict, wordt automatisch naar JSON geserialiseerd.

    Geeft True terug bij succes, False bij een fout (fout wordt gelogd,
    niet opgegooid -- zo blijft de bot draaien ook als de DB-insert faalt).
    """
    if isinstance(datum, (date, datetime)):
        datum = datum.isoformat()

    params_json = json.dumps(parameters) if parameters is not None else None

    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO selecties
                        (ticker, datum, strategie, beurs, koers, parameters)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (ticker, datum, strategie, beurs, koers, params_json),
                )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as exc:
        logger.error("db_logger: insert voor %s (%s) mislukt: %s", ticker, strategie, exc)
        return False


def log_selecties_bulk(rows: list) -> int:
    """
    Schrijft meerdere selecties in één connectie/transactie weg.
    rows: lijst van dicts met keys ticker, datum, strategie, beurs, koers, parameters.
    Geeft het aantal succesvol weggeschreven rijen terug.
    """
    if not rows:
        return 0

    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("db_logger: kon geen connectie maken: %s", exc)
        return 0

    aantal_ok = 0
    try:
        with conn.cursor() as cur:
            for row in rows:
                datum = row.get("datum")
                if isinstance(datum, (date, datetime)):
                    datum = datum.isoformat()
                params_json = (
                    json.dumps(row["parameters"])
                    if row.get("parameters") is not None
                    else None
                )
                try:
                    cur.execute(
                        """
                        INSERT INTO selecties
                            (ticker, datum, strategie, beurs, koers, parameters)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            row.get("ticker"),
                            datum,
                            row.get("strategie"),
                            row.get("beurs"),
                            row.get("koers"),
                            params_json,
                        ),
                    )
                    aantal_ok += 1
                except Exception as exc:
                    logger.error(
                        "db_logger: rij overgeslagen (%s): %s", row.get("ticker"), exc
                    )
        conn.commit()
    finally:
        conn.close()

    return aantal_ok
