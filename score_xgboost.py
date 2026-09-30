#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_xgboost.py
================

Scoort NIEUWE selecties met de getrainde XGBoostV3-modellen.

Modellen:
    xgboostV3_10d_model.pkl
    xgboostV3_30d_model.pkl
    xgboostV3_60d_model.pkl

Werking
=======

- Beschikbare geldige modellen worden geladen.
- Ontbrekende modellen worden overgeslagen.
- Nieuwe selecties worden uit:
      generieke_technicals
      selecties
  gehaald.
- Alleen selecties binnen LOOKBACK_DAGEN worden verwerkt.
- Reeds opgeslagen combinaties worden niet opnieuw gescoord.
- Alle succesvolle scores worden opgeslagen.
- Alleen TOP_N wordt naar Telegram/e-mail gestuurd.
- Features en volgorde worden rechtstreeks gecontroleerd tegen
  model.feature_names_in_.
- Minstens één geldig model is vereist.

Omgevingsvariabelen
===================

Verplicht:
    SUPABASE_DB_URL

Optioneel:
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID

    EMAIL_USER
    EMAIL_PASS
    EMAIL_RECEIVER

    TOP_N
        standaard: 10

    LOOKBACK_DAGEN
        standaard: 3

Vereiste features
=================

    atr14
    atr14_pct
    rsi14
    ibs
    ma50
    ma200
    pct_from_ma50
    pct_from_ma200
    vol_ratio_20d
    high52w
    pct_from_high52w

Score
=====

De score is een modelscore/rangorde.
Het is GEEN garantie en wordt niet als een gekalibreerde
beleggingskans geïnterpreteerd.
"""

import os
import smtplib
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import xgboost


warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy"
)


# ============================================================
# CONFIGURATIE
# ============================================================

MODEL_VERSIE = "xgboostV3"

HORIZONS = [
    "10d",
    "30d",
    "60d",
]

TOP_N = int(
    os.getenv("TOP_N", "10")
)

LOOKBACK_DAGEN = int(
    os.getenv("LOOKBACK_DAGEN", "3")
)

SUPABASE_DB_URL = os.getenv(
    "SUPABASE_DB_URL",
    ""
).strip()

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()

EMAIL_USER = os.getenv(
    "EMAIL_USER",
    ""
).strip()

EMAIL_PASS = os.getenv(
    "EMAIL_PASS",
    ""
).strip()

EMAIL_RECEIVER = os.getenv(
    "EMAIL_RECEIVER",
    ""
).strip()


# Exact dezelfde features als tijdens training.
VERWACHTE_FEATURES = [
    "atr14",
    "atr14_pct",
    "rsi14",
    "ibs",
    "ma50",
    "ma200",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "high52w",
    "pct_from_high52w",
]


# ============================================================
# CONFIGURATIE CONTROLEREN
# ============================================================

def controleer_configuratie() -> None:
    """Controleert de belangrijkste instellingen."""

    if not SUPABASE_DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL ontbreekt. "
            "Deze variabele is verplicht."
        )

    if TOP_N < 1:
        raise RuntimeError(
            f"TOP_N moet minimaal 1 zijn. "
            f"Ontvangen: {TOP_N}"
        )

    if LOOKBACK_DAGEN < 0:
        raise RuntimeError(
            "LOOKBACK_DAGEN mag niet negatief zijn."
        )

    print("")
    print("=" * 70)
    print("CONFIGURATIE")
    print("=" * 70)

    print(
        f"Modelversie       : {MODEL_VERSIE}"
    )

    print(
        f"Horizons          : {', '.join(HORIZONS)}"
    )

    print(
        f"TOP_N             : {TOP_N}"
    )

    print(
        f"LOOKBACK_DAGEN    : {LOOKBACK_DAGEN}"
    )

    print(
        f"XGBoost versie     : {xgboost.__version__}"
    )

    print(
        "Telegram          : "
        + (
            "geconfigureerd"
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
            else "niet geconfigureerd"
        )
    )

    print(
        "E-mail            : "
        + (
            "geconfigureerd"
            if EMAIL_USER
            and EMAIL_PASS
            and EMAIL_RECEIVER
            else "niet geconfigureerd"
        )
    )


# ============================================================
# DATABASE
# ============================================================

def open_database():
    """Opent een PostgreSQL/Supabase verbinding."""

    try:

        conn = psycopg2.connect(
            SUPABASE_DB_URL,
            connect_timeout=20,
        )

        conn.autocommit = False

        print(
            "[DB] Verbinding met database geopend."
        )

        return conn

    except Exception as exc:

        print("")
        print("=" * 70)
        print("❌ DATABASEVERBINDING MISLUKT")
        print("=" * 70)

        print(
            f"Type fout : {type(exc).__name__}"
        )

        print(
            f"Fout      : {exc}"
        )

        if hasattr(exc, "pgcode"):
            print(
                f"pgcode    : {exc.pgcode}"
            )

        if hasattr(exc, "pgerror"):
            print(
                f"pgerror   : {exc.pgerror}"
            )

        raise RuntimeError(
            f"Databaseverbinding mislukt: {exc}"
        ) from exc


# ============================================================
# DATABASE SCHEMA DIAGNOSTIEK
# ============================================================

def controleer_database_schema(conn) -> None:
    """
    Controleert vooraf of de belangrijkste tabellen bestaan.

    Dit voorkomt dat een onduidelijke SQL-fout pas midden
    in de scoring zichtbaar wordt.
    """

    vereiste_tabellen = [
        "generieke_technicals",
        "selecties",
        "xgboost_scores",
        "xgboost_runs",
    ]

    print("")
    print("=" * 70)
    print("DATABASE SCHEMA CONTROLEREN")
    print("=" * 70)

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s)
                ORDER BY table_name
                """,
                (
                    vereiste_tabellen,
                ),
            )

            gevonden = {
                row[0]
                for row in cur.fetchall()
            }

        for tabel in vereiste_tabellen:

            if tabel in gevonden:
                print(
                    f"  ✅ {tabel}"
                )
            else:
                print(
                    f"  ❌ {tabel} ONTBREEKT"
                )

        ontbrekend = [
            tabel
            for tabel in vereiste_tabellen
            if tabel not in gevonden
        ]

        if ontbrekend:

            raise RuntimeError(
                "Ontbrekende databasetabellen: "
                + ", ".join(ontbrekend)
            )

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        print("")
        print(
            "[DB SCHEMA FOUT]"
        )

        print(
            f"Type    : {type(exc).__name__}"
        )

        print(
            f"Fout    : {exc}"
        )

        if hasattr(exc, "pgcode"):
            print(
                f"pgcode  : {exc.pgcode}"
            )

        if hasattr(exc, "pgerror"):
            print(
                f"pgerror : {exc.pgerror}"
            )

        raise


def haal_kolommen_op(
    conn,
    tabel: str,
) -> List[str]:
    """Geeft de kolommen van een public-tabel terug."""

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                ORDER BY ordinal_position
                """,
                (tabel,),
            )

            return [
                row[0]
                for row in cur.fetchall()
            ]

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        print(
            f"[DB] Kolommen van {tabel} konden "
            f"niet worden gelezen: {exc}"
        )

        return []


def controleer_belangrijke_kolommen(
    conn,
) -> None:
    """
    Laat de werkelijke kolommen zien van de tabellen die
    door score_xgboost.py worden gebruikt.

    Dit is vooral bedoeld om schema-afwijkingen onmiddellijk
    zichtbaar te maken.
    """

    vereisten = {
        "generieke_technicals": [
            "ticker",
            "datum",
            *VERWACHTE_FEATURES,
        ],
        "selecties": [
            "ticker",
            "datum",
            "strategie",
            "koers",
            "beurs",
        ],
        "xgboost_scores": [
            "ticker",
            "datum",
            "model_versie",
            "horizon",
            "score",
            "strategieen",
            "koers",
            "beurs",
        ],
    }

    print("")
    print("=" * 70)
    print("BELANGRIJKE DATABASEKOLOMMEN")
    print("=" * 70)

    fouten = []

    for tabel, vereist in vereisten.items():

        werkelijk = haal_kolommen_op(
            conn,
            tabel,
        )

        print("")
        print(
            f"[{tabel}]"
        )

        if not werkelijk:

            fouten.append(
                f"{tabel}: geen kolommen gevonden"
            )

            continue

        ontbrekend = [
            kolom
            for kolom in vereist
            if kolom not in werkelijk
        ]

        if ontbrekend:

            print(
                "  ❌ Ontbrekend:"
            )

            for kolom in ontbrekend:
                print(
                    f"     - {kolom}"
                )

            fouten.append(
                f"{tabel}: "
                + ", ".join(ontbrekend)
            )

        else:

            print(
                "  ✅ Vereiste kolommen aanwezig."
            )

    if fouten:

        print("")
        print("=" * 70)
        print("❌ DATABASESCHEMA PAST NIET BIJ SCORE SCRIPT")
        print("=" * 70)

        for fout in fouten:
            print(
                f"  - {fout}"
            )

        raise RuntimeError(
            "De database bevat niet alle vereiste "
            "kolommen. Zie bovenstaande diagnose."
        )

   
