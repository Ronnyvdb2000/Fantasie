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

1. Laadt alle beschikbare en geldige XGBoostV3-modellen.
2. Ontbrekende modellen worden automatisch overgeslagen.
3. Haalt nieuwe (ticker, datum)-paren op uit:
       generieke_technicals
       selecties
4. Alleen selecties binnen LOOKBACK_DAGEN worden verwerkt.
5. Bestaande scores in xgboost_scores worden niet opnieuw opgeslagen.
6. Per kandidaat en per horizon wordt predict_proba() uitgevoerd.
7. ALLE succesvolle scores worden opgeslagen.
8. Alleen de TOP_N nieuwe scores worden naar Telegram/e-mail gestuurd.
9. De gebruikte features worden rechtstreeks uit
   model.feature_names_in_ gehaald.
10. Een run wordt geregistreerd in xgboost_runs indien die tabel
    beschikbaar is.

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

Vereiste model-features
=======================

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

Belangrijk
==========

"score" is uitsluitend een modelscore/rangordewaarde.
Het is GEEN garantie en wordt niet als een gekalibreerde
beleggingskans geïnterpreteerd.
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

TOP_N = int(os.getenv("TOP_N", "10"))
LOOKBACK_DAGEN = int(os.getenv("LOOKBACK_DAGEN", "3"))

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip()
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()


# Exact dezelfde features en volgorde als tijdens training.
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
# BASISCONTROLES
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
            f"TOP_N moet minimaal 1 zijn. Ontvangen: {TOP_N}"
        )

    if LOOKBACK_DAGEN < 0:
        raise RuntimeError(
            "LOOKBACK_DAGEN mag niet negatief zijn."
        )

    print("")
    print("=" * 70)
    print("CONFIGURATIE")
    print("=" * 70)
    print(f"Modelversie       : {MODEL_VERSIE}")
    print(f"Horizons          : {', '.join(HORIZONS)}")
    print(f"TOP_N             : {TOP_N}")
    print(f"LOOKBACK_DAGEN    : {LOOKBACK_DAGEN}")
    print(f"XGBoost versie     : {xgboost.__version__}")
    print(
        "Telegram          : "
        + ("geconfigureerd" if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
           else "niet geconfigureerd")
    )
    print(
        "E-mail            : "
        + ("geconfigureerd"
           if EMAIL_USER and EMAIL_PASS and EMAIL_RECEIVER
           else "niet geconfigureerd")
    )


# ============================================================
# DATABASE
# ============================================================

def open_database():
    """Opent een PostgreSQL/Supabase databaseverbinding."""

    try:
        conn = psycopg2.connect(
            SUPABASE_DB_URL,
            connect_timeout=20,
        )

        conn.autocommit = False

        print("[DB] Verbinding met database geopend.")

        return conn

    except Exception as exc:
        raise RuntimeError(
            f"Databaseverbinding mislukt: {exc}"
        ) from exc


# ============================================================
# MODELVALIDATIE
# ============================================================

def valideer_model(
    model: object,
    horizon: str,
    pad: str,
) -> Tuple[bool, str]:
    """
    Controleert of het model bruikbaar is.

    Vereisten:
    - predict_proba()
    - feature_names_in_
    - exacte features
    - exacte featurevolgorde
    - bestand >= 1000 bytes
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

    Ontbrekende of corrupte modellen worden overgeslagen.
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

        if not os.path.exists(pad):
            print(
                f"[{horizon}] modelbestand ontbreekt — "
                f"overgeslagen."
            )
            continue

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
            f"{grootte:,} bytes"
        )

        if grootte < 1000:
            print(
                f"[{horizon}] ongeldig/placeholderbestand — "
                f"overgeslagen."
            )
            continue

        try:
            model = joblib.load(pad)

        except Exception as exc:
            print(
                f"[{horizon}] joblib.load() mislukt: "
                f"{exc}"
            )
            continue

        geldig, reden = valideer_model(
            model=model,
            horizon=horizon,
            pad=pad,
        )

        if not geldig:
            print(
                f"[{horizon}] model ongeldig: {reden}"
            )
            continue

        modellen[horizon] = model

        print(
            f"[{horizon}] ✅ model geladen."
        )

    print("")
    print("=" * 70)
    print(
        f"GELDIGE MODELLEN: "
        f"{len(modellen)} / {len(HORIZONS)}"
    )
    print("=" * 70)

    for horizon in HORIZONS:
        if horizon in modellen:
            print(f"  ✅ {horizon}")
        else:
            print(f"  ⏭️ {horizon} — overgeslagen")

    if not modellen:
        raise RuntimeError(
            "Geen enkel geldig XGBoostV3-model beschikbaar."
        )

    return modellen


# ============================================================
# KANDIDATEN
# ============================================================

def haal_kandidaten(
    conn,
    horizon: str,
) -> pd.DataFrame:
    """
    Haalt nieuwe selecties op voor één horizon.

    Bestaande scores voor dezelfde:
        ticker
        datum
        model_versie
        horizon

    worden uitgesloten.
    """

    vandaag = dt.date.today()

    cutoff = vandaag - dt.timedelta(
        days=LOOKBACK_DAGEN
    )

    query = """
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
        GROUP BY
            ticker,
            datum
    ) sel
        ON sel.ticker = gt.ticker
       AND sel.datum = gt.datum

    WHERE gt.datum >= %(cutoff)s

      AND NOT EXISTS (
          SELECT 1
          FROM xgboost_scores xs
          WHERE xs.ticker = gt.ticker
            AND xs.datum = gt.datum
            AND xs.model_versie = %(model_versie)s
            AND xs.horizon = %(horizon)s
      )

    ORDER BY
        gt.datum DESC,
        gt.ticker
    """

    try:

        df = pd.read_sql_query(
            query,
            conn,
            params={
                "cutoff": cutoff,
                "model_versie": MODEL_VERSIE,
                "horizon": horizon,
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Kandidaten ophalen voor {horizon} mislukt: "
            f"{exc}"
        ) from exc

    print("")
    print(
        f"[{horizon}] kandidaten gevonden: "
        f"{len(df)}"
    )

    return df


# ============================================================
# FEATURES
# ============================================================

def bouw_feature_dataframe(
    kandidaten: pd.DataFrame,
    model: object,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bouwt exact dezelfde featurekolommen en volgorde als het model.

    Kandidaten met ontbrekende/niet-numerieke features worden
    niet gescoord.
    """

    ontbrekend = [
        feature
        for feature in VERWACHTE_FEATURES
        if feature not in kandidaten.columns
    ]

    if ontbrekend:
        raise RuntimeError(
            "Ontbrekende featurekolommen in "
            f"generieke_technicals: {ontbrekend}"
        )

    features = list(model.feature_names_in_)

    X = kandidaten[features].copy()

    for kolom in features:
        X[kolom] = pd.to_numeric(
            X[kolom],
            errors="coerce",
        )

    geldig_masker = X.notna().all(axis=1)

    X_geldig = X.loc[geldig_masker].copy()

    kandidaten_geldig = kandidaten.loc[
        geldig_masker
    ].copy()

    kandidaten_geldig.reset_index(
        drop=True,
        inplace=True,
    )

    X_geldig.reset_index(
        drop=True,
        inplace=True,
    )

    return X_geldig, kandidaten_geldig


# ============================================================
# SCORE
# ============================================================

def score_horizon(
    kandidaten: pd.DataFrame,
    model: object,
    horizon: str,
) -> List[Dict]:
    """
    Scoort alle geldige kandidaten voor één horizon.

    De score is de kans/score uit predict_proba() voor klasse 1.
    Deze wordt uitsluitend gebruikt als rangordewaarde.
    """

    if kandidaten.empty:
        return []

    X, geldig = bouw_feature_dataframe(
        kandidaten=kandidaten,
        model=model,
    )

    if X.empty:
        print(
            f"[{horizon}] geen kandidaten met "
            f"complete features."
        )
        return []

    print(
        f"[{horizon}] scoring: "
        f"{len(X)} kandidaten"
    )

    try:

        probabilities = model.predict_proba(X)

    except Exception as exc:
        raise RuntimeError(
            f"predict_proba() mislukt voor {horizon}: "
            f"{exc}"
        ) from exc

    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Onverwachte predict_proba-vorm voor "
            f"{horizon}: {probabilities.shape}"
        )

    if probabilities.shape[1] < 2:
        raise RuntimeError(
            f"Model {horizon} heeft geen tweede klasse "
            f"in predict_proba()."
        )

    scores = probabilities[:, 1]

    resultaten: List[Dict] = []

    for i, score in enumerate(scores):

        rij = geldig.iloc[i]

        ticker = str(
            rij.get("ticker", "")
        ).strip()

        if not ticker:
            continue

        datum = rij.get("datum")

        if pd.isna(datum):
            continue

        if isinstance(datum, pd.Timestamp):
            datum = datum.date()

        try:
            score_float = float(score)
        except (TypeError, ValueError):
            continue

        if not pd.notna(score_float):
            continue

        resultaat = {
            "ticker": ticker,
            "datum": datum,
            "horizon": horizon,
            "model_versie": MODEL_VERSIE,
            "score": score_float,
            "strategieen": rij.get(
                "strategieen"
            ),
            "koers": rij.get(
                "koers"
            ),
            "beurs": rij.get(
                "beurs"
            ),
        }

        resultaten.append(resultaat)

    resultaten.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    print(
        f"[{horizon}] succesvol gescoord: "
        f"{len(resultaten)}"
    )

    return resultaten


# ============================================================
# DATABASE OPSLAAN
# ============================================================

def sla_scores_op(
    conn,
    resultaten: List[Dict],
) -> int:
    """
    Slaat alle nieuwe scores op.

    Extra bescherming tegen dubbele records:
    INSERT ... WHERE NOT EXISTS

    """

    if not resultaten:
        return 0

    sql = """
    INSERT INTO xgboost_scores (
        ticker,
        datum,
        model_versie,
        horizon,
        score,
        strategieen,
        koers,
        beurs
    )
    SELECT
        %(ticker)s,
        %(datum)s,
        %(model_versie)s,
        %(horizon)s,
        %(score)s,
        %(strategieen)s,
        %(koers)s,
        %(beurs)s
    WHERE NOT EXISTS (
        SELECT 1
        FROM xgboost_scores
        WHERE ticker = %(ticker)s
          AND datum = %(datum)s
          AND model_versie = %(model_versie)s
          AND horizon = %(horizon)s
    )
    """

    opgeslagen = 0

    try:

        with conn.cursor() as cur:

            for resultaat in resultaten:

                cur.execute(
                    sql,
                    resultaat,
                )

                if cur.rowcount == 1:
                    opgeslagen += 1

        conn.commit()

    except Exception as exc:

        conn.rollback()

        raise RuntimeError(
            f"Opslaan van XGBoost-scores mislukt: "
            f"{exc}"
        ) from exc

    print(
        f"[DB] nieuwe scores opgeslagen: "
        f"{opgeslagen}"
    )

    return opgeslagen


# ============================================================
# RUN REGISTRATIE
# ============================================================

def registreer_run(
    conn,
    starttijd: dt.datetime,
    eindtijd: dt.datetime,
    modellen: List[str],
    kandidaten: int,
    scores: int,
    opgeslagen: int,
) -> None:
    """
    Registreert de run in xgboost_runs.

    De functie probeert eerst een gangbare kolomstructuur.
    Als de tabel niet bestaat of een andere structuur heeft,
    wordt de scoring NIET ongeldig verklaard.
    """

    duur = (
        eindtijd - starttijd
    ).total_seconds()

    sql = """
    INSERT INTO xgboost_runs (
        model_versie,
        run_datum,
        modellen,
        kandidaten,
        scores,
        opgeslagen,
        duur_seconden
    )
    VALUES (
        %(model_versie)s,
        %(run_datum)s,
        %(modellen)s,
        %(kandidaten)s,
        %(scores)s,
        %(opgeslagen)s,
        %(duur_seconden)s
    )
    """

    gegevens = {
        "model_versie": MODEL_VERSIE,
        "run_datum": eindtijd,
        "modellen": ", ".join(modellen),
        "kandidaten": kandidaten,
        "scores": scores,
        "opgeslagen": opgeslagen,
        "duur_seconden": duur,
    }

    try:

        with conn.cursor() as cur:
            cur.execute(
                sql,
                gegevens,
            )

        conn.commit()

        print(
            "[DB] run geregistreerd in "
            "xgboost_runs."
        )

    except Exception as exc:

        conn.rollback()

        print(
            "[WAARSCHUWING] xgboost_runs kon niet "
            f"worden bijgewerkt: {exc}"
        )


# ============================================================
# TOP SELECTIES
# ============================================================

def bepaal_top_scores(
    resultaten: List[Dict],
) -> List[Dict]:
    """
    Bepaalt de TOP_N over ALLE horizons samen.
    """

    resultaten = sorted(
        resultaten,
        key=lambda x: x["score"],
        reverse=True,
    )

    return resultaten[:TOP_N]


# ============================================================
# FORMATTERING
# ============================================================

def format_score(
    score: float,
) -> str:
    """
    Maakt een leesbare modelscore.
    """

    return f"{score:.4f}"


def format_koers(
    koers,
) -> str:

    if koers is None or pd.isna(koers):
        return "-"

    try:
        return f"{float(koers):.2f}"
    except Exception:
        return str(koers)


def bouw_telegram_bericht(
    top_scores: List[Dict],
    totaal_opgeslagen: int,
    modellen: List[str],
) -> str:
    """
    Bouwt het Telegrambericht.
    """

    nu = dt.datetime.now()

    regels = []

    regels.append(
        "🤖 XGBoostV3 — NIEUWE SELECTIES"
    )

    regels.append(
        f"Datum: {nu.strftime('%d-%m-%Y %H:%M')}"
    )

    regels.append(
        f"Modellen: {', '.join(modellen)}"
    )

    regels.append(
        f"Nieuwe scores opgeslagen: "
        f"{totaal_opgeslagen}"
    )

    regels.append("")
    regels.append(
        f"TOP {len(top_scores)}"
    )
    regels.append(
        "-" * 38
    )

    if not top_scores:
        regels.append(
            "Geen nieuwe scores."
        )

    else:

        for nummer, item in enumerate(
            top_scores,
            start=1,
        ):

            ticker = item["ticker"]
            horizon = item["horizon"]
            score = format_score(
                item["score"]
            )

            datum = item["datum"]

            if isinstance(
                datum,
                dt.datetime,
            ):
                datum_tekst = datum.strftime(
                    "%d-%m-%Y"
                )
            elif isinstance(
                datum,
                dt.date,
            ):
                datum_tekst = datum.strftime(
                    "%d-%m-%Y"
                )
            else:
                datum_tekst = str(datum)

            strategieen = (
                item.get("strategieen")
                or "-"
            )

            koers = format_koers(
                item.get("koers")
            )

            beurs = (
                item.get("beurs")
                or "-"
            )

            regels.append(
                f"{nummer}. {ticker} "
                f"| {horizon} "
                f"| score {score}"
            )

            regels.append(
                f"   Datum: {datum_tekst} "
                f"| Koers: {koers}"
            )

            regels.append(
                f"   Beurs: {beurs}"
            )

            regels.append(
                f"   Strategie: {strategieen}"
            )

            regels.append("")

    regels.append(
        "Score = modelsrangorde, "
        "geen garantie."
    )

    return "\n".join(regels)


def bouw_email_bericht(
    top_scores: List[Dict],
    totaal_opgeslagen: int,
    modellen: List[str],
) -> str:
    """
    Eenvoudige tekstversie voor e-mail.
    """

    return bouw_telegram_bericht(
        top_scores=top_scores,
        totaal_opgeslagen=totaal_opgeslagen,
        modellen=modellen,
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    tekst: str,
) -> None:
    """
    Stuurt Telegrambericht.

    Zonder Telegram-configuratie wordt het bericht naar stdout
    geschreven.
    """

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:

        print("")
        print(
            "[INFO] Telegram niet geconfigureerd."
        )
        print(
            "----- TELEGRAM BERICHT -----"
        )
        print(tekst)
        print(
            "----- EINDE TELEGRAM -----"
        )

        return

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    for positie in range(
        0,
        len(tekst),
        4096,
    ):

        deel = tekst[
            positie:positie + 4096
        ]

        try:

            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": deel,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )

            if response.status_code != 200:

                print(
                    "[WAARSCHUWING] Telegram gaf "
                    f"status {response.status_code}: "
                    f"{response.text[:300]}"
                )

            else:

                print(
                    "[Telegram] bericht verzonden."
                )

        except Exception as exc:

            print(
                f"[WAARSCHUWING] Telegram fout: {exc}"
            )


# ============================================================
# EMAIL
# ============================================================

def send_email(
    onderwerp: str,
    tekst: str,
) -> None:
    """
    Stuurt e-mail via Gmail SMTP.
    """

    if (
        not EMAIL_USER
        or not EMAIL_PASS
        or not EMAIL_RECEIVER
    ):

        print(
            "[INFO] E-mail niet geconfigureerd; "
            "overgeslagen."
        )

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

        server.ehlo()
        server.starttls()
        server.ehlo()

        server.login(
            EMAIL_USER,
            EMAIL_PASS,
        )

        server.send_message(msg)

        print(
            f"[Email] verzonden naar "
            f"{EMAIL_RECEIVER}"
        )

    except Exception as exc:

        print(
            f"[WAARSCHUWING] Email fout: {exc}"
        )

    finally:

        if server is not None:

            try:
                server.quit()

            except Exception:
                pass


# ============================================================
# HOOFDPROGRAMMA
# ============================================================

def main() -> None:

    starttijd = dt.datetime.now()

    print("")
    print("=" * 70)
    print("XGBOOSTV3 SCORING")
    print("=" * 70)
    print(
        f"Start: "
        f"{starttijd.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    controleer_configuratie()

    # --------------------------------------------------------
    # Modellen laden
    # --------------------------------------------------------

    modellen = laad_modellen()

    model_namen = list(
        modellen.keys()
    )

    # --------------------------------------------------------
    # Database openen
    # --------------------------------------------------------

    conn = open_database()

    alle_resultaten: List[Dict] = []

    totaal_kandidaten = 0

    try:

        # ----------------------------------------------------
        # Per horizon scoren
        # ----------------------------------------------------

        for horizon in HORIZONS:

            if horizon not in modellen:

                print("")
                print(
                    f"[{horizon}] geen geldig model — "
                    f"overgeslagen."
                )

                continue

            model = modellen[horizon]

            kandidaten = haal_kandidaten(
                conn=conn,
                horizon=horizon,
            )

            totaal_kandidaten += len(
                kandidaten
            )

            if kandidaten.empty:

                print(
                    f"[{horizon}] geen nieuwe "
                    f"kandidaten."
                )

                continue

            resultaten = score_horizon(
                kandidaten=kandidaten,
                model=model,
                horizon=horizon,
            )

            alle_resultaten.extend(
                resultaten
            )

        # ----------------------------------------------------
        # Alle resultaten opslaan
        # ----------------------------------------------------

        print("")
        print("=" * 70)
        print("SCORES OPSLAAN")
        print("=" * 70)

        totaal_scores = len(
            alle_resultaten
        )

        print(
            f"Succesvol berekende scores: "
            f"{totaal_scores}"
        )

        opgeslagen = sla_scores_op(
            conn=conn,
            resultaten=alle_resultaten,
        )

        # ----------------------------------------------------
        # TOP N
        # ----------------------------------------------------

        top_scores = bepaal_top_scores(
            alle_resultaten
        )

        print("")
        print("=" * 70)
        print(
            f"TOP {TOP_N} NIEUWE SCORES"
        )
        print("=" * 70)

        for nummer, item in enumerate(
            top_scores,
            start=1,
        ):

            print(
                f"{nummer:2d}. "
                f"{item['ticker']:12s} "
                f"{item['horizon']:4s} "
                f"score={item['score']:.6f}"
            )

        # ----------------------------------------------------
        # Bericht
        # ----------------------------------------------------

        telegram_tekst = bouw_telegram_bericht(
            top_scores=top_scores,
            totaal_opgeslagen=opgeslagen,
            modellen=model_namen,
        )

        email_tekst = bouw_email_bericht(
            top_scores=top_scores,
            totaal_opgeslagen=opgeslagen,
            modellen=model_namen,
        )

        send_telegram(
            telegram_tekst
        )

        send_email(
            onderwerp=(
                "XGBoostV3 — "
                "nieuwe selecties"
            ),
            tekst=email_tekst,
        )

        # ----------------------------------------------------
        # Run registreren
        # ----------------------------------------------------

        eindtijd = dt.datetime.now()

        registreer_run(
            conn=conn,
            starttijd=starttijd,
            eindtijd=eindtijd,
            modellen=model_namen,
            kandidaten=totaal_kandidaten,
            scores=totaal_scores,
            opgeslagen=opgeslagen,
        )

        # ----------------------------------------------------
        # Eindoverzicht
        # ----------------------------------------------------

        print("")
        print("=" * 70)
        print("XGBOOSTV3 RUN KLAAR")
        print("=" * 70)

        print(
            f"Modellen gebruikt : "
            f"{', '.join(model_namen)}"
        )

        print(
            f"Kandidaten        : "
            f"{totaal_kandidaten}"
        )

        print(
            f"Scores berekend   : "
            f"{totaal_scores}"
        )

        print(
            f"Nieuw opgeslagen  : "
            f"{opgeslagen}"
        )

        print(
            f"TOP_N             : "
            f"{len(top_scores)}"
        )

        print(
            f"Eindtijd          : "
            f"{eindtijd.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        print("")
        print("✅ Scoring succesvol afgerond.")

    finally:

        try:
            conn.close()

            print(
                "[DB] Databaseverbinding gesloten."
            )

        except Exception:
            pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("")
        print(
            "⚠️ Programma onderbroken."
        )

        raise SystemExit(130)

    except Exception as exc:

        print("")
        print("=" * 70)
        print("❌ XGBOOSTV3 SCORING MISLUKT")
        print("=" * 70)
        print(
            f"Fout: {exc}"
        )

        raise SystemExit(1)
