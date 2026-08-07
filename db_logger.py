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
- GitHub Actions secret SUPABASE_DB_URL (Connect -> Direct Connection ->
  Session/Transaction pooler-string, met wachtwoord ingevuld).
  Gebruik de POOLER-variant, niet "Direct connection": GitHub Actions-
  runners hebben geen IPv6, en de Direct Connection-hostname is enkel
  over IPv6 bereikbaar.

Gedrag bij fouten:
- Als de DSN zelf ongeldig is (bv. plak-fout, ontbrekende '=', verkeerd
  wachtwoord) faalt de EERSTE aanroep, wordt dat 1x duidelijk gelogd,
  en wordt DB-logging voor de rest van deze procesrun automatisch
  uitgeschakeld -- zo blijft de bot draaien en spamt hij niet 20-30x
  dezelfde fout in de Actions-log.
- Andere fouten (bv. een tijdelijke netwerkhik) worden wel per insert
  gelogd, want die kunnen de volgende keer wel slagen.
"""

import os
import json
import logging
from datetime import date, datetime

import psycopg2

logger = logging.getLogger(__name__)

_DB_URL_ENV = "SUPABASE_DB_URL"

# Als de DSN zelf ongeldig blijkt (config-fout), zetten we dit op True
# zodat we niet bij elke ticker dezelfde fout opnieuw proberen/loggen.
_dsn_invalid = False


def _get_connection():
    db_url = os.environ.get(_DB_URL_ENV, "")
    # .strip() vangt de meest voorkomende oorzaak van "invalid dsn":
    # een per ongeluk meegekopieerde spatie/tab/regeleinde in de secret.
    db_url = db_url.strip().strip('"').strip("'")
    if not db_url:
        raise RuntimeError(
            f"Omgevingsvariabele {_DB_URL_ENV} ontbreekt. "
            f"Voeg ze toe als GitHub Actions secret en geef ze door aan de workflow."
        )
    if not (db_url.startswith("postgresql://") or db_url.startswith("postgres://")):
        raise RuntimeError(
            f"{_DB_URL_ENV} lijkt geen geldige connectiestring: moet beginnen "
            f"met 'postgresql://' of 'postgres://'. Controleer op een per "
            f"ongeluk meegekopieerde spatie, aanhalingsteken of regeleinde."
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
    global _dsn_invalid

    if _dsn_invalid:
        # We hebben al vastgesteld dat de DSN/config niet werkt deze run.
        # Niet opnieuw proberen -- dat levert enkel identieke ruis op.
        return False

    if isinstance(datum, (date, datetime)):
        datum = datum.isoformat()

    params_json = json.dumps(parameters) if parameters is not None else None

    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("db_logger: kon geen DB-connectie opzetten, DB-logging uitgeschakeld voor deze run: %s", exc)
        _dsn_invalid = True
        return False

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
    except Exception as exc:
        logger.error("db_logger: insert voor %s (%s) mislukt: %s", ticker, strategie, exc)
        return False
    finally:
        conn.close()


def log_selecties_bulk(rows: list) -> int:
    """
    Schrijft meerdere selecties in één connectie/transactie weg.
    rows: lijst van dicts met keys ticker, datum, strategie, beurs, koers, parameters.
    Geeft het aantal succesvol weggeschreven rijen terug.
    """
    global _dsn_invalid

    if not rows or _dsn_invalid:
        return 0

    try:
        conn = _get_connection()
    except Exception as exc:
        logger.error("db_logger: kon geen DB-connectie opzetten, DB-logging uitgeschakeld voor deze run: %s", exc)
        _dsn_invalid = True
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
