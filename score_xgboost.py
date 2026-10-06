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

Werking:
- Beschikbare geldige modellen worden geladen.
- Ontbrekende modellen worden overgeslagen.
- Nieuwe selecties worden uit:
      generieke_technicals
      selecties
  gehaald.
- Alleen selecties binnen LOOKBACK_DAGEN worden verwerkt.
- Reeds opgeslagen combinaties worden niet opnieuw gescoord.
- Features en volgorde worden gecontroleerd tegen
  model.feature_names_in_.
- Nieuwe scores worden opgeslagen in xgboost_scores.
- TOP_N wordt naar Telegram/e-mail gestuurd.
- Een run wordt geregistreerd in xgboost_runs wanneer
  daarvoor geschikte kolommen aanwezig zijn.
- Echte fouten eindigen met exit code 1.

Belangrijk:
De score is een modelscore/rangorde.
Het is GEEN garantie en wordt niet als een gekalibreerde
beleggingskans geïnterpreteerd.

Database-aanpassing:
- xgboost_scores.koers is optioneel.
- xgboost_scores.beurs is optioneel.
- xgboost_runs.horizon wordt gevuld wanneer deze kolom bestaat.

Datum-aanpassing:
- selecties.datum, generieke_technicals.datum en
  xgboost_scores.datum kunnen als TEXT in PostgreSQL staan.
- Daarom worden datumvelden in de SQL expliciet naar
  timestamptz gecast.

Wijzigingen v1.1 (2026-10-04):
- haal_bestaande_scores_op: xgboost_scores.datum wordt nu ook
  naar timestamptz gecast (fout: text >= timestamp).
- haal_nieuwe_selecties_op: leeftijdsgrens van 7 dagen op de
  gekoppelde technische rij, zodat nooit met verouderde features
  gescoord wordt wanneer de rij voor de selectiedatum ontbreekt.

Wijzigingen v1.2 (2026-10-06):
- sla_scores_op: 'rang' en 'n_gescoord' zijn NOT NULL in de
  database maar werden niet gevuld (NotNullViolation op
  xgboost_scores.rang). Nieuwe functie voeg_rang_toe() berekent
  per (model_versie, horizon, datum) een ranking: rang 1 =
  hoogste score, n_gescoord = aantal scores in die groep.
  Beide kolommen zijn nu opgenomen in verplichte_kolommen en
  worden vóór de INSERT ingevuld.
"""

import os
import sys
import smtplib
import traceback
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

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

MODEL_DIR = os.getenv(
    "XGBOOST_MODEL_DIR",
    "."
).strip() or "."

MODEL_FILES = {
    horizon: os.path.join(
        MODEL_DIR,
        f"{MODEL_VERSIE}_{horizon}_model.pkl",
    )
    for horizon in HORIZONS
}


# ============================================================
# EXACTE FEATURES UIT TRAINING
# ============================================================

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
# ALGEMENE HULPFUNCTIES
# ============================================================

def print_header(titel: str) -> None:
    print("")
    print("=" * 70)
    print(titel)
    print("=" * 70)


def sql_ident(naam: str) -> str:
    """
    Maakt een veilige PostgreSQL identifier.

    Alleen eenvoudige databasekolomnamen zijn toegestaan.
    """

    if not naam:
        raise ValueError(
            "Lege SQL identifier."
        )

    if not naam.replace("_", "").isalnum():
        raise ValueError(
            f"Ongeldige SQL identifier: {naam!r}"
        )

    return '"' + naam.replace('"', '""') + '"'


def normaliseer_datum(waarde):
    if waarde is None:
        return None

    try:
        if pd.isna(waarde):
            return None
    except Exception:
        pass

    try:
        ts = pd.to_datetime(
            waarde,
            errors="coerce",
        )

        if pd.isna(ts):
            return None

        return ts.to_pydatetime()

    except Exception:
        return waarde


def format_score(score: float) -> str:
    return f"{float(score):.4f}"


def voeg_rang_toe(
    resultaten: List[Dict],
) -> List[Dict]:
    """
    Voegt 'rang' en 'n_gescoord' toe aan elke rij in resultaten.

    Rang 1 = hoogste score binnen dezelfde
    (model_versie, horizon, datum).
    n_gescoord = aantal scores binnen diezelfde groep.

    Werkt in-place: muteert de dicts in de lijst.
    """
    if not resultaten:
        return resultaten

    groepen: Dict[tuple, List[Dict]] = {}

    for r in resultaten:

        d = r.get("datum")

        try:
            d_key = (
                pd.Timestamp(d).date()
                if d is not None
                else None
            )
        except Exception:
            d_key = d

        key = (
            str(r.get("model_versie", "")),
            str(r.get("horizon", "")),
            d_key,
        )

        groepen.setdefault(key, []).append(r)

    for _, groep in groepen.items():

        gesorteerd = sorted(
            groep,
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )

        n = len(gesorteerd)

        for i, r in enumerate(
            gesorteerd,
            start=1,
        ):
            r["rang"] = i
            r["n_gescoord"] = n

    return resultaten


# ============================================================
# CONFIGURATIE CONTROLEREN
# ============================================================

def controleer_configuratie() -> None:

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

    print_header(
        "XGBOOSTV3 SCORING"
    )

    print(
        f"Modelversie       : {MODEL_VERSIE}"
    )

    print(
        f"Horizons          : "
        f"{', '.join(HORIZONS)}"
    )

    print(
        f"TOP_N             : {TOP_N}"
    )

    print(
        f"LOOKBACK_DAGEN    : {LOOKBACK_DAGEN}"
    )

    print(
        f"Modelmap          : "
        f"{os.path.abspath(MODEL_DIR)}"
    )

    print(
        f"XGBoost versie    : "
        f"{xgboost.__version__}"
    )

    print(
        "Telegram          : "
        + (
            "geconfigureerd"
            if TELEGRAM_TOKEN
            and TELEGRAM_CHAT_ID
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
# DATABASE VERBINDING
# ============================================================

def open_database():

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

        print_header(
            "DATABASEVERBINDING MISLUKT"
        )

        print(
            f"Type fout : {type(exc).__name__}"
        )

        print(
            f"Fout      : {exc}"
        )

        if getattr(exc, "pgcode", None):

            print(
                f"pgcode    : {exc.pgcode}"
            )

        if getattr(exc, "pgerror", None):

            print(
                f"pgerror   : {exc.pgerror}"
            )

        raise RuntimeError(
            f"Databaseverbinding mislukt: {exc}"
        ) from exc


# ============================================================
# DATABASE SCHEMA
# ============================================================

def haal_kolommen_op(
    conn,
    tabel: str,
) -> List[str]:

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

        raise RuntimeError(
            f"Kolommen van {tabel} konden "
            f"niet worden gelezen: {exc}"
        ) from exc


def controleer_database_schema(
    conn,
) -> Dict[str, List[str]]:

    vereiste_tabellen = [
        "generieke_technicals",
        "selecties",
        "xgboost_scores",
        "xgboost_runs",
    ]

    print_header(
        "DATABASE SCHEMA CONTROLEREN"
    )

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
            (vereiste_tabellen,),
        )

        gevonden = {
            row[0]
            for row in cur.fetchall()
        }

    ontbrekend = [
        tabel
        for tabel in vereiste_tabellen
        if tabel not in gevonden
    ]

    for tabel in vereiste_tabellen:

        if tabel in gevonden:

            print(
                f"  ✅ {tabel}"
            )

        else:

            print(
                f"  ❌ {tabel} ONTBREEKT"
            )

    if ontbrekend:

        raise RuntimeError(
            "Ontbrekende databasetabellen: "
            + ", ".join(ontbrekend)
        )

    schema = {}

    for tabel in vereiste_tabellen:

        schema[tabel] = haal_kolommen_op(
            conn,
            tabel,
        )

    return schema


# ============================================================
# BELANGRIJKE KOLOMMEN CONTROLEREN
# ============================================================

def controleer_belangrijke_kolommen(
    schema: Dict[str, List[str]],
) -> None:

    print_header(
        "BELANGRIJKE DATABASEKOLOMMEN"
    )

    vereisten_verplicht = {

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
            "rang",
            "n_gescoord",
            "strategieen",
        ],
    }

    fouten = []

    for tabel, vereist in vereisten_verplicht.items():

        werkelijk = schema.get(
            tabel,
            [],
        )

        print("")
        print(
            f"[{tabel}]"
        )

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
                "  ✅ Verplichte kolommen aanwezig."
            )

        if tabel == "xgboost_scores":

            for optioneel in [
                "koers",
                "beurs",
            ]:

                if optioneel in werkelijk:

                    print(
                        f"  ℹ️ Optioneel aanwezig: "
                        f"{optioneel}"
                    )

                else:

                    print(
                        f"  ℹ️ Optioneel ontbreekt: "
                        f"{optioneel} "
                        f"(geen probleem)"
                    )

    runs_kolommen = schema.get(
        "xgboost_runs",
        [],
    )

    print("")
    print(
        "[xgboost_runs]"
    )

    if runs_kolommen:

        print(
            "  ✅ Tabel aanwezig."
        )

        print(
            "  Kolommen: "
            + ", ".join(runs_kolommen)
        )

        if "horizon" in runs_kolommen:

            print(
                "  ℹ️ horizon aanwezig — "
                "wordt verplicht gevuld."
            )

    if fouten:

        print_header(
            "DATABASESCHEMA PAST NIET BIJ SCORE SCRIPT"
        )

        for fout in fouten:

            print(
                f"  - {fout}"
            )

        raise RuntimeError(
            "De database bevat niet alle "
            "vereiste kolommen."
        )


# ============================================================
# MODELLEN LADEN
# ============================================================

def laad_modellen() -> Dict[str, object]:

    print_header(
        "XGBOOSTV3 MODELLEN LADEN"
    )

    modellen = {}

    for horizon in HORIZONS:

        bestand = MODEL_FILES[horizon]

        print("")
        print(
            f"[{horizon}]"
        )

        print(
            f"  Bestand: {bestand}"
        )

        if not os.path.exists(bestand):

            print(
                "  ⏭️ Ontbreekt — wordt overgeslagen."
            )

            continue

        grootte = os.path.getsize(
            bestand
        )

        print(
            f"  Grootte: {grootte:,} bytes"
        )

        if grootte == 0:

            print(
                "  ❌ Bestand is leeg."
            )

            continue

        try:

            model = joblib.load(
                bestand
            )

            if not hasattr(
                model,
                "predict_proba",
            ):

                print(
                    "  ❌ Model heeft geen "
                    "predict_proba()."
                )

                continue

            if not hasattr(
                model,
                "feature_names_in_",
            ):

                print(
                    "  ❌ Model heeft geen "
                    "feature_names_in_."
                )

                continue

            model_features = list(
                model.feature_names_in_
            )

            if model_features != VERWACHTE_FEATURES:

                print(
                    "  ❌ Featurevolgorde komt "
                    "niet overeen."
                )

                print(
                    f"     Model    : "
                    f"{model_features}"
                )

                print(
                    f"     Verwacht : "
                    f"{VERWACHTE_FEATURES}"
                )

                continue

            print(
                "  ✅ Model geldig."
            )

            print(
                f"  Features: "
                f"{model_features}"
            )

            modellen[horizon] = model

        except Exception as exc:

            print(
                f"  ❌ Laden mislukt: "
                f"{type(exc).__name__}: {exc}"
            )

    if not modellen:

        raise RuntimeError(
            "Geen enkel geldig "
            "XGBoostV3-model beschikbaar."
        )

    print("")
    print(
        "Beschikbare modellen: "
        + ", ".join(
            modellen.keys()
        )
    )

    ontbrekend = [
        horizon
        for horizon in HORIZONS
        if horizon not in modellen
    ]

    if ontbrekend:

        print(
            "Overgeslagen horizons: "
            + ", ".join(ontbrekend)
        )

    return modellen


# ============================================================
# NIEUWE SELECTIES OPHALEN
# ============================================================

def haal_nieuwe_selecties_op(
    conn,
) -> pd.DataFrame:

    print_header(
        "NIEUWE SELECTIES OPHALEN"
    )

    grensdatum = (
        dt.datetime.now(
            dt.timezone.utc
        )
        - dt.timedelta(
            days=LOOKBACK_DAGEN
        )
    )

    print(
        f"Vanaf datum/tijd : "
        f"{grensdatum.isoformat()}"
    )

    technische_features = ",\n".join(
        f't.{sql_ident(feature)}'
        for feature in VERWACHTE_FEATURES
    )

    query = f"""
        SELECT
            s."ticker" AS ticker,
            s."datum" AS selectie_datum,
            s."strategie" AS strategie,
            s."koers" AS selectie_koers,
            s."beurs" AS beurs,
            t."datum" AS technische_datum,
            {technische_features}
        FROM public."selecties" s

        JOIN LATERAL (

            SELECT
                t."datum",
                {
                    ", ".join(
                        f't.{sql_ident(feature)}'
                        for feature in VERWACHTE_FEATURES
                    )
                }

            FROM public."generieke_technicals" t

            WHERE
                t."ticker" = s."ticker"
                AND t."datum"::timestamptz
                    <= s."datum"::timestamptz
                AND t."datum"::timestamptz
                    >= s."datum"::timestamptz - interval '7 days'

            ORDER BY
                t."datum"::timestamptz DESC

            LIMIT 1

        ) t ON TRUE

        WHERE
            s."datum"::timestamptz >= %s

        ORDER BY
            s."datum"::timestamptz DESC,
            s."ticker" ASC
    """

    try:

        df = pd.read_sql_query(
            query,
            conn,
            params=(grensdatum,),
        )

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        raise RuntimeError(
            "Ophalen van nieuwe selecties "
            f"mislukt: {exc}"
        ) from exc

    if df.empty:

        print(
            "⚠️ Geen selecties gevonden "
            "binnen LOOKBACK_DAGEN."
        )

        return df

    print(
        f"Ruwe selectie-rijen : "
        f"{len(df)}"
    )

    df["ticker"] = (
        df["ticker"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["strategie"] = (
        df["strategie"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["ticker"] != ""
    ].copy()

    for feature in VERWACHTE_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    before = len(df)

    df = df.dropna(
        subset=VERWACHTE_FEATURES
    ).copy()

    dropped = (
        before - len(df)
    )

    if dropped:

        print(
            f"⚠️ {dropped} selectie(s) "
            "verwijderd wegens ontbrekende "
            "features."
        )

    if df.empty:

        print(
            "⚠️ Geen bruikbare selecties "
            "na featurecontrole."
        )

        return df

    df["selectie_datum"] = pd.to_datetime(
        df["selectie_datum"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["selectie_datum"]
    ).copy()

    def combine_strategies(series):

        waarden = sorted(
            {
                str(value).strip()
                for value in series
                if str(value).strip()
            }
        )

        return "; ".join(
            waarden
        )

    aggregaties = {

        "strategie":
            combine_strategies,

        "selectie_koers":
            "first",

        "beurs":
            "first",

        "technische_datum":
            "max",
    }

    for feature in VERWACHTE_FEATURES:

        aggregaties[feature] = "first"

    df = (
        df
        .groupby(
            [
                "ticker",
                "selectie_datum",
            ],
            as_index=False,
        )
        .agg(aggregaties)
    )

    print(
        f"Bruikbare unieke selecties: "
        f"{len(df)}"
    )

    return df


# ============================================================
# BESTAANDE SCORES OPHALEN
# ============================================================

def haal_bestaande_scores_op(
    conn,
    df_selecties: pd.DataFrame,
) -> set:

    if df_selecties.empty:

        return set()

    min_datum = (
        df_selecties[
            "selectie_datum"
        ].min()
    )

    query = """
        SELECT
            ticker,
            datum,
            model_versie,
            horizon
        FROM public.xgboost_scores
        WHERE datum::timestamptz >= %s
    """

    try:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (min_datum,),
            )

            rows = cur.fetchall()

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        raise RuntimeError(
            "Bestaande XGBoost-scores "
            "konden niet worden gelezen: "
            f"{exc}"
        ) from exc

    bestaande = set()

    for (
        ticker,
        datum,
        model_versie,
        horizon,
    ) in rows:

        datum_norm = (
            normaliseer_datum(
                datum
            )
        )

        if datum_norm is not None:

            datum_key = (
                pd.Timestamp(
                    datum_norm
                ).date()
            )

        else:

            datum_key = datum

        bestaande.add(
            (
                str(ticker).strip(),
                datum_key,
                str(model_versie).strip(),
                str(horizon).strip(),
            )
        )

    print(
        f"[DB] Bestaande scorecombinaties: "
        f"{len(bestaande)}"
    )

    return bestaande


# ============================================================
# SCORE BEREKENEN
# ============================================================

def positieve_score(
    model,
    X: pd.DataFrame,
) -> float:

    probabilities = model.predict_proba(
        X
    )

    if getattr(
        probabilities,
        "ndim",
        0,
    ) != 2:

        raise RuntimeError(
            "predict_proba() gaf geen "
            "2D-array terug."
        )

    if probabilities.shape[1] == 1:

        return float(
            probabilities[0, 0]
        )

    classes = list(
        getattr(
            model,
            "classes_",
            range(
                probabilities.shape[1]
            ),
        )
    )

    if 1 in classes:

        index = classes.index(1)

    else:

        index = (
            probabilities.shape[1] - 1
        )

        print(
            "[MODEL] Klasse 1 niet expliciet "
            "gevonden; laatste probability "
            "wordt gebruikt."
        )

    return float(
        probabilities[0, index]
    )


# ============================================================
# SELECTIES SCOREN
# ============================================================

def score_selecties(
    df_selecties: pd.DataFrame,
    modellen: Dict[str, object],
    bestaande: set,
) -> List[Dict]:

    print_header(
        "SELECTIES SCOREN"
    )

    resultaten = []

    if df_selecties.empty:

        print(
            "Geen selecties om te scoren."
        )

        return resultaten

    totaal = len(
        df_selecties
    )

    for positie, (_, rij) in enumerate(
        df_selecties.iterrows(),
        start=1,
    ):

        ticker = str(
            rij["ticker"]
        ).strip()

        selectie_datum_ts = pd.to_datetime(
            rij["selectie_datum"],
            errors="coerce",
        )

        if pd.isna(
            selectie_datum_ts
        ):

            print(
                f"[{positie}/{totaal}] "
                f"{ticker}: ongeldige datum."
            )

            continue

        datum_key = (
            selectie_datum_ts.date()
        )

        print(
            f"[{positie}/{totaal}] "
            f"{ticker} | "
            f"{datum_key} | "
            f"{rij.get('strategie', '')}"
        )

        X = pd.DataFrame(
            [
                {
                    feature:
                    float(rij[feature])
                    for feature
                    in VERWACHTE_FEATURES
                }
            ],
            columns=VERWACHTE_FEATURES,
        )

        for horizon, model in modellen.items():

            sleutel = (
                ticker,
                datum_key,
                MODEL_VERSIE,
                horizon,
            )

            if sleutel in bestaande:

                print(
                    f"   ⏭️ {horizon}: "
                    "al gescoord."
                )

                continue

            try:

                score = positieve_score(
                    model,
                    X,
                )

                if not pd.notna(
                    score
                ):

                    print(
                        f"   ❌ {horizon}: "
                        "score is NaN."
                    )

                    continue

                resultaat = {

                    "ticker":
                        ticker,

                    "datum":
                        selectie_datum_ts.to_pydatetime(),

                    "model_versie":
                        MODEL_VERSIE,

                    "horizon":
                        horizon,

                    "score":
                        float(score),

                    "strategieen":
                        str(
                            rij.get(
                                "strategie",
                                "",
                            )
                        ),

                    "koers":
                        (
                            float(
                                rij[
                                    "selectie_koers"
                                ]
                            )
                            if pd.notna(
                                rij.get(
                                    "selectie_koers"
                                )
                            )
                            else None
                        ),

                    "beurs":
                        str(
                            rij.get(
                                "beurs",
                                "",
                            )
                        ).strip(),
                }

                resultaten.append(
                    resultaat
                )

                print(
                    f"   ✅ {horizon}: "
                    f"{format_score(score)}"
                )

            except Exception as exc:

                print(
                    f"   ❌ {horizon}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    print("")
    print(
        f"Nieuwe scores berekend: "
        f"{len(resultaten)}"
    )

    return resultaten


# ============================================================
# SCORES OPSLAAN
# ============================================================

def sla_scores_op(
    conn,
    resultaten: List[Dict],
    schema: Dict[str, List[str]],
) -> int:

    print_header(
        "SCORES OPSLAAN"
    )

    if not resultaten:

        print(
            "Geen nieuwe scores om op te slaan."
        )

        return 0

    # v1.2: rang en n_gescoord toevoegen vóór de INSERT.
    # Zonder deze stap faalt de INSERT met NotNullViolation.
    resultaten = voeg_rang_toe(resultaten)

    beschikbare_kolommen = schema.get(
        "xgboost_scores",
        [],
    )

    # v1.2: rang en n_gescoord staan nu in de verplichte lijst,
    # want de database-kolommen zijn NOT NULL.
    verplichte_kolommen = [
        "ticker",
        "datum",
        "model_versie",
        "horizon",
        "score",
        "rang",
        "n_gescoord",
        "strategieen",
    ]

    ontbrekend = [
        kolom
        for kolom in verplichte_kolommen
        if kolom not in beschikbare_kolommen
    ]

    if ontbrekend:

        raise RuntimeError(
            "xgboost_scores mist verplichte "
            "kolommen: "
            + ", ".join(ontbrekend)
        )

    optionele_kolommen = [
        "koers",
        "beurs",
    ]

    insert_kolommen = list(
        verplichte_kolommen
    )

    for kolom in optionele_kolommen:

        if kolom in beschikbare_kolommen:

            insert_kolommen.append(
                kolom
            )

            print(
                f"[DB] xgboost_scores.{kolom}: "
                "aanwezig — wordt opgeslagen."
            )

        else:

            print(
                f"[DB] xgboost_scores.{kolom}: "
                "niet aanwezig — wordt overgeslagen."
            )

    kolommen_sql = ",\n            ".join(
        sql_ident(kolom)
        for kolom in insert_kolommen
    )

    placeholders = ",\n            ".join(
        f"%({kolom})s"
        for kolom in insert_kolommen
    )

    query = f"""
        INSERT INTO public.xgboost_scores (
            {kolommen_sql}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT DO NOTHING
    """

    insert_resultaten = []

    for resultaat in resultaten:

        record = {
            kolom:
                resultaat.get(kolom)
            for kolom in insert_kolommen
        }

        insert_resultaten.append(
            record
        )

    try:

        with conn.cursor() as cur:

            psycopg2.extras.execute_batch(
                cur,
                query,
                insert_resultaten,
                page_size=100,
            )

        conn.commit()

        print(
            f"✅ {len(resultaten)} score(s) "
            "naar xgboost_scores geschreven."
        )

        return len(resultaten)

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        print(
            f"❌ Opslaan scores mislukt: "
            f"{type(exc).__name__}: {exc}"
        )

        raise


# ============================================================
# RUN REGISTREREN
# ============================================================

def bepaal_run_horizon(
    modellen: Dict[str, object],
) -> str:

    beschikbare = [
        horizon
        for horizon in HORIZONS
        if horizon in modellen
    ]

    if not beschikbare:

        return HORIZONS[0]

    return ",".join(
        beschikbare
    )


def registreer_run(
    conn,
    schema: Dict[str, List[str]],
    status: str,
    aantal_selecties: int,
    aantal_scores: int,
    foutmelding: Optional[str] = None,
    modellen: Optional[Dict[str, object]] = None,
) -> None:

    kolommen = schema.get(
        "xgboost_runs",
        [],
    )

    if not kolommen:

        print(
            "[RUN] Geen kolommen beschikbaar."
        )

        return

    if modellen:

        run_horizon = (
            bepaal_run_horizon(
                modellen
            )
        )

    else:

        run_horizon = HORIZONS[0]

    mogelijke_waarden = {

        "model_versie":
            MODEL_VERSIE,

        "horizon":
            run_horizon,

        "status":
            status,

        "aantal_selecties":
            int(aantal_selecties),

        "aantal_scores":
            int(aantal_scores),

        "foutmelding":
            (
                str(foutmelding)[:2000]
                if foutmelding
                else None
            ),

        "gestart_op":
            dt.datetime.now(
                dt.timezone.utc
            ),

        "voltooid_op":
            dt.datetime.now(
                dt.timezone.utc
            ),

        "datum":
            dt.datetime.now(
                dt.timezone.utc
            ),
    }

    bruikbaar = [
        kolom
        for kolom in mogelijke_waarden
        if kolom in kolommen
    ]

    if "horizon" in kolommen:

        if "horizon" not in bruikbaar:

            bruikbaar.append(
                "horizon"
            )

    if not bruikbaar:

        print(
            "[RUN] Geen herkenbare kolommen "
            "om run te registreren."
        )

        return

    kolommen_sql = ", ".join(
        sql_ident(kolom)
        for kolom in bruikbaar
    )

    placeholders = ", ".join(
        f"%({kolom})s"
        for kolom in bruikbaar
    )

    query = f"""
        INSERT INTO public.xgboost_runs (
            {kolommen_sql}
        )
        VALUES (
            {placeholders}
        )
    """

    waarden = {
        kolom:
            mogelijke_waarden[kolom]
        for kolom in bruikbaar
    }

    try:

        with conn.cursor() as cur:

            cur.execute(
                query,
                waarden,
            )

        conn.commit()

        print(
            f"[RUN] Run geregistreerd: "
            f"{status} | "
            f"horizon={run_horizon}"
        )

    except Exception as exc:

        try:
            conn.rollback()
        except Exception:
            pass

        print(
            f"⚠️ Runregistratie mislukt: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# TOP RESULTATEN
# ============================================================

def bepaal_top_resultaten(
    resultaten: List[Dict],
) -> pd.DataFrame:

    if not resultaten:

        return pd.DataFrame()

    df = pd.DataFrame(
        resultaten
    )

    if df.empty:

        return df

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["score"]
    ).copy()

    if df.empty:

        return df

    df = df.sort_values(
        by=[
            "score",
            "ticker",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return (
        df
        .head(TOP_N)
        .reset_index(drop=True)
    )


# ============================================================
# TELEGRAM
# ============================================================

def stuur_telegram(
    df_top: pd.DataFrame,
) -> bool:

    if not (
        TELEGRAM_TOKEN
        and TELEGRAM_CHAT_ID
    ):

        print(
            "[Telegram] Niet geconfigureerd "
            "— overgeslagen."
        )

        return False

    if df_top.empty:

        print(
            "[Telegram] Geen TOP-resultaten."
        )

        return False

    regels = [

        "📊 XGBoostV3 — TOP nieuwe selecties",

        "",

        (
            "Modellen: "
            + ", ".join(
                sorted(
                    df_top[
                        "horizon"
                    ].unique()
                )
            )
        ),

        f"TOP_N: {TOP_N}",

        "",
    ]

    for index, rij in df_top.iterrows():

        ticker = str(
            rij["ticker"]
        )

        horizon = str(
            rij["horizon"]
        )

        score = float(
            rij["score"]
        )

        koers = rij.get(
            "koers"
        )

        beurs = str(
            rij.get(
                "beurs",
                "",
            )
            or ""
        )

        strategieen = str(
            rij.get(
                "strategieen",
                "",
            )
            or ""
        )

        if pd.notna(koers):

            koers_txt = (
                f"{float(koers):.2f}"
            )

        else:

            koers_txt = "-"

        regels.append(
            f"{index + 1}. "
            f"{ticker} | "
            f"{horizon} | "
            f"score {score:.4f} | "
            f"koers {koers_txt}"
        )

        if beurs:

            regels.append(
                f"   Beurs: {beurs}"
            )

        if strategieen:

            regels.append(
                f"   Strategie: "
                f"{strategieen}"
            )

    bericht = "\n".join(
        regels
    )

    if len(bericht) > 3900:

        bericht = (
            bericht[:3900]
            + "\n…"
        )

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}"
        "/sendMessage"
    )

    try:

        response = requests.post(
            url,
            data={
                "chat_id":
                    TELEGRAM_CHAT_ID,

                "text":
                    bericht,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("ok"):

            raise RuntimeError(
                f"Telegram antwoordde "
                f"met fout: {data}"
            )

        print(
            "✅ Telegrambericht verzonden."
        )

        return True

    except Exception as exc:

        print(
            f"❌ Telegram verzenden mislukt: "
            f"{type(exc).__name__}: {exc}"
        )

        return False


# ============================================================
# E-MAIL
# ============================================================

def stuur_email(
    df_top: pd.DataFrame,
) -> bool:

    if not (
        EMAIL_USER
        and EMAIL_PASS
        and EMAIL_RECEIVER
    ):

        print(
            "[E-mail] Niet geconfigureerd "
            "— overgeslagen."
        )

        return False

    if df_top.empty:

        print(
            "[E-mail] Geen TOP-resultaten."
        )

        return False

    onderwerp = (
        f"XGBoostV3 TOP "
        f"{len(df_top)} nieuwe selecties"
    )

    regels = [

        "XGBoostV3 — TOP nieuwe selecties",

        "",

        f"Modelversie: {MODEL_VERSIE}",

        (
            "Datum: "
            f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ),

        "",
    ]

    for index, rij in df_top.iterrows():

        koers = rij.get(
            "koers"
        )

        if pd.notna(koers):

            koers_txt = (
                f"{float(koers):.2f}"
            )

        else:

            koers_txt = "-"

        regels.extend([

            f"{index + 1}. "
            f"{rij['ticker']}",

            f"   Horizon: "
            f"{rij['horizon']}",

            f"   Score: "
            f"{float(rij['score']):.4f}",

            f"   Koers: "
            f"{koers_txt}",

            f"   Beurs: "
            f"{rij.get('beurs', '') or '-'}",

            f"   Strategie: "
            f"{rij.get('strategieen', '') or '-'}",

            "",
        ])

    body = "\n".join(
        regels
    )

    message = MIMEMultipart()

    message["From"] = EMAIL_USER
    message["To"] = EMAIL_RECEIVER
    message["Subject"] = onderwerp

    message.attach(
        MIMEText(
            body,
            "plain",
            "utf-8",
        )
    )

    try:

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=20,
        ) as server:

            server.ehlo()

            server.starttls()

            server.ehlo()

            server.login(
                EMAIL_USER,
                EMAIL_PASS,
            )

            server.send_message(
                message
            )

        print(
            "✅ E-mail verzonden."
        )

        return True

    except Exception as exc:

        print(
            f"❌ E-mail verzenden mislukt: "
            f"{type(exc).__name__}: {exc}"
        )

        return False


# ============================================================
# SAMENVATTING
# ============================================================

def toon_samenvatting(
    modellen: Dict[str, object],
    df_selecties: pd.DataFrame,
    resultaten: List[Dict],
    opgeslagen: int,
    df_top: pd.DataFrame,
) -> None:

    print_header(
        "EINDRESULTAAT"
    )

    print(
        "Beschikbare modellen : "
        + ", ".join(
            modellen.keys()
        )
    )

    print(
        f"Selecties gevonden   : "
        f"{len(df_selecties)}"
    )

    print(
        f"Nieuwe scores        : "
        f"{len(resultaten)}"
    )

    print(
        f"Scores opgeslagen    : "
        f"{opgeslagen}"
    )

    print(
        f"TOP_N                : "
        f"{len(df_top)}"
    )

    if not df_top.empty:

        print("")
        print(
            "TOP RESULTATEN:"
        )

        for index, rij in df_top.iterrows():

            print(
                f"  {index + 1:2d}. "
                f"{str(rij['ticker']):12s} "
                f"{str(rij['horizon']):4s} "
                f"score="
                f"{float(rij['score']):.4f}"
            )

    else:

        print("")
        print(
            "Geen TOP-resultaten beschikbaar."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    starttijd = (
        dt.datetime.now(
            dt.timezone.utc
        )
    )

    conn = None

    modellen = {}

    schema = {}

    df_selecties = (
        pd.DataFrame()
    )

    resultaten = []

    opgeslagen = 0

    try:

        controleer_configuratie()

        print("")
        print(
            f"[START] "
            f"{starttijd.isoformat()}"
        )

        modellen = laad_modellen()

        conn = open_database()

        schema = (
            controleer_database_schema(
                conn
            )
        )

        controleer_belangrijke_kolommen(
            schema
        )

        df_selecties = (
            haal_nieuwe_selecties_op(
                conn
            )
        )

        if df_selecties.empty:

            print_header(
                "GEEN NIEUWE SELECTIES"
            )

            print(
                "Er zijn binnen "
                "LOOKBACK_DAGEN geen "
                "bruikbare nieuwe "
                "selecties gevonden."
            )

            registreer_run(
                conn,
                schema,
                status="geen_selecties",
                aantal_selecties=0,
                aantal_scores=0,
                modellen=modellen,
            )

            return 0

        bestaande = (
            haal_bestaande_scores_op(
                conn,
                df_selecties,
            )
        )

        resultaten = (
            score_selecties(
                df_selecties,
                modellen,
                bestaande,
            )
        )

        opgeslagen = (
            sla_scores_op(
                conn,
                resultaten,
                schema,
            )
        )

        df_top = (
            bepaal_top_resultaten(
                resultaten
            )
        )

        telegram_ok = (
            stuur_telegram(
                df_top
            )
        )

        email_ok = (
            stuur_email(
                df_top
            )
        )

        registreer_run(
            conn,
            schema,
            status="succes",
            aantal_selecties=len(
                df_selecties
            ),
            aantal_scores=opgeslagen,
            modellen=modellen,
        )

        toon_samenvatting(
            modellen,
            df_selecties,
            resultaten,
            opgeslagen,
            df_top,
        )

        eindtijd = (
            dt.datetime.now(
                dt.timezone.utc
            )
        )

        duur = (
            eindtijd - starttijd
        )

        print("")
        print(
            f"[KLAAR] "
            f"{eindtijd.isoformat()}"
        )

        print(
            f"[DUUR] {duur}"
        )

        if df_top.empty:

            print(
                "[INFO] Geen nieuwe "
                "TOP-resultaten."
            )

        if (
            TELEGRAM_TOKEN
            and TELEGRAM_CHAT_ID
        ):

            print(
                "[INFO] Telegram status: "
                + (
                    "OK"
                    if telegram_ok
                    else "niet verzonden"
                )
            )

        if (
            EMAIL_USER
            and EMAIL_PASS
            and EMAIL_RECEIVER
        ):

            print(
                "[INFO] E-mail status: "
                + (
                    "OK"
                    if email_ok
                    else "niet verzonden"
                )
            )

        return 0

    except Exception as exc:

        print("")
        print("=" * 70)
        print(
            "❌ XGBOOSTV3 SCORING MISLUKT"
        )
        print("=" * 70)

        print(
            f"Type fout : "
            f"{type(exc).__name__}"
        )

        print(
            f"Fout      : {exc}"
        )

        print("")
        print(
            "VOLLEDIGE TRACEBACK:"
        )

        traceback.print_exc()

        if conn is not None:

            try:

                registreer_run(
                    conn,
                    schema,
                    status="fout",
                    aantal_selecties=len(
                        df_selecties
                    ),
                    aantal_scores=opgeslagen,
                    foutmelding=str(
                        exc
                    )[:2000],
                    modellen=modellen,
                )

            except Exception as run_exc:

                print(
                    "[RUN] "
                    "Foutregistratie mislukt: "
                    f"{run_exc}"
                )

        return 1

    finally:

        if conn is not None:

            try:

                conn.close()

                print(
                    "[DB] Verbinding gesloten."
                )

            except Exception as exc:

                print(
                    "[DB] Sluiten verbinding "
                    f"mislukt: {exc}"
                )


# ============================================================
# PROGRAMMA START HIER
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
