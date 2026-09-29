#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_xgboost.py — scoort NIEUWE selecties met de getrainde xgboostV3-modellen

Laadt per horizon (10d/30d/60d) het door xgboostV3.py getrainde model:
    xgboostV3_<horizon>_model.pkl

Horizons zonder geldig modelbestand worden automatisch overgeslagen.

Werking
=======
- Kandidaten:
  (ticker, datum)-paren in generieke_technicals met datum binnen
  LOOKBACK_DAGEN dagen, die in selecties voorkomen.

- Per paar worden ook de strategieën, koers en beurs opgehaald.

- Elk paar wordt per horizon maar ÉÉN keer gescoord.
  Wat al in xgboost_scores staat, wordt niet opnieuw opgeslagen.

- ALLE succesvol gescoorde rijen worden opgeslagen in xgboost_scores.
  Niet alleen de TOP_N.

- De features worden rechtstreeks uit model.feature_names_in_ gehaald,
  zodat scoring exact dezelfde features en volgorde gebruikt als training.

- Ontbrekende horizons zijn toegestaan:
      10d aanwezig -> gebruiken
      30d ontbreekt -> overslaan
      60d ontbreekt -> overslaan

- Ongeldige/corrupte modellen worden eveneens overgeslagen.

- Minstens één geldig model is vereist.

"Score" is de rangorde van het model, geen gekalibreerde kans en geen garantie.

Env vars
========
SUPABASE_DB_URL    verplicht

TELEGRAM_TOKEN     optioneel
TELEGRAM_CHAT_ID   optioneel

EMAIL_USER         optioneel
EMAIL_PASS         optioneel
EMAIL_RECEIVER     optioneel

TOP_N              default 10
LOOKBACK_DAGEN     default 3

Vereiste tabellen
=================
generieke_technicals
selecties
xgboost_scores
xgboost_runs
"""

import os
import smtplib
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import xgboost  # noqa: F401


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

TOP_N = int(os.getenv("TOP_N", "10"))
LOOKBACK_DAGEN = int(os.getenv("LOOKBACK_DAGEN", "3"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")


# Exact dezelfde features als bij de training.
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
# SQL
# ============================================================

KANDIDATEN_QUERY = """
SELECT
    gt.*,
    sel.strategieen,
    sel.koers,
    sel.beurs
FROM generieke_technicals gt
JOIN (
    SELECT
        ticker,
        datum,
        string_agg(
            DISTINCT strategie,
            ', '
            ORDER BY strategie
        ) AS strategieen,
        MAX(koers) AS koers,
        MIN(beurs) AS beurs
    FROM selecties
    WHERE datum >= %(cutoff)s
    GROUP BY ticker, datum
) sel
    ON sel.ticker = gt.ticker
   AND sel.datum = gt.datum
WHERE NOT EXISTS (
    SELECT 1
    FROM xgboost_scores xs
    WHERE xs.ticker = gt.ticker
      AND xs.datum = gt.datum
      AND xs.model_versie = %(model_versie)s
      AND xs.horizon = %(horizon)s
);
"""


# ============================================================
# BERICHTEN
# ============================================================

def send_telegram(tekst: str) -> None:
    """
    Stuurt Telegram-bericht.
    Zonder Telegram-config wordt het bericht gewoon naar stdout geschreven.
    """

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    for i in range(0, len(tekst), 4096):
        deel = tekst[i:i + 4096]

        try:
            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": deel,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )

            if response.status_code != 200:
                print(
                    f"Telegram gaf status "
                    f"{response.status_code}: "
                    f"{response.text[:200]}"
                )

        except Exception as exc:
            print(f"Telegram fout: {exc}")


def send_email(onderwerp: str, tekst: str) -> None:
    """
    Stuurt e-mail indien de SMTP-gegevens aanwezig zijn.
    """

    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        print("[INFO] E-mail niet geconfigureerd; overgeslagen.")
        return

    server = None

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = onderwerp

        msg.attach(
            MIMEText(
                tekst,
                "plain",
                "utf-8",
            )
        )

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=20,
        )

        server.starttls()
        server.login(
            EMAIL_USER,
            EMAIL_PASS,
        )

        server.send_message(msg)

        print(
            f"Email verzonden naar "
            f"{EMAIL_RECEIVER}"
        )

    except Exception as exc:
        print(f"Email fout: {exc}")

    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass


# ============================================================
# MODELVALIDATIE
# ============================================================

def valideer_model(
    model: object,
    horizon: str,
    pad: str,
) -> Tuple[bool, str]:
    """
    Controleert of een geladen model daadwerkelijk bruikbaar is.

    Vereisten:
    - predict_proba()
    - feature_names_in_
    - exact dezelfde 11 features
    - exact dezelfde featurevolgorde
    """

    if not hasattr(model, "predict_proba"):
        return (
            False,
            "model heeft geen predict_proba()",
        )

    if not hasattr(model, "feature_names_in_"):
        return (
            False,
            "model heeft geen feature_names_in_",
        )

    try:
        echte_features = list(model.feature_names_in_)
    except Exception as exc:
        return (
            False,
            f"feature_names_in_ kon niet worden gelezen: {exc}",
        )

    if echte_features != VERWACHTE_FEATURES:
        return (
            False,
            "verkeerde features of featurevolgorde",
        )

    try:
        grootte = os.path.getsize(pad)
    except OSError as exc:
        return (
            False,
            f"bestandsgrootte kon niet worden gelezen: {exc}",
        )

    if grootte < 1000:
        return (
            False,
            f"bestand is slechts {grootte} bytes",
        )

    print(
        f"[{horizon}] model validatie OK: "
        f"{grootte:,} bytes, "
        f"{len(echte_features)} features"
    )

    return True, "OK"


def laad_modellen() -> Dict[str, object]:
    """
    Laadt alle beschikbare en geldige modellen.

    Ontbrekende modellen zijn toegestaan.

    Bijvoorbeeld:
        10d aanwezig
        30d ontbreekt
        60d ontbreekt

    resulteert in:
        {"10d": model}

    Een corrupt of ongeldig model wordt niet gebruikt.
    """

    modellen: Dict[str, object] = {}

    print("")
    print("=" * 70)
    print("XGBOOSTV3 MODELLEN LADEN")
    print("=" * 70)

    for horizon in HORIZONS:

        pad = f"{MODEL_VERSIE}_{horizon}_model.pkl"

        print("")
        print(f"[{horizon}] controle: {pad}")

        # ----------------------------------------------------
        # Bestand bestaat niet
        # ----------------------------------------------------

        if not os.path.exists(pad):
            print(
                f"[{horizon}] geen modelbestand "
                f"({pad}), overgeslagen."
            )
            continue

        # ----------------------------------------------------
        # Bestandsgrootte
        # ----------------------------------------------------

        try:
            grootte = os.path.getsize(pad)
        except OSError as exc:
            print(
                f"[{horizon}] bestandsgrootte kon niet "
                f"worden gelezen: {exc}"
            )
            continue

        print(
            f"[{horizon}] bestandsgrootte: "
            f"{grootte:,} bytes
