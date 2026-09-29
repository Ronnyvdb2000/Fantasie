#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgboostV3.py — wekelijkse hertraining per horizon (10d / 30d / 60d)

Versie 3.2

Train per horizon een XGBoost-classifier en sla uitsluitend geldige,
niet-lege modellen op.

Belangrijke beveiligingen:
- modelbestand wordt na joblib.dump gecontroleerd
- bestandsgrootte wordt gecontroleerd
- model wordt onmiddellijk opnieuw geladen
- predict_proba wordt gecontroleerd
- feature_names_in_ wordt gecontroleerd
- training faalt als een model niet correct opgeslagen kan worden
- controle-model wordt niet opgeslagen
- tijdssplit gebeurt op datumgrenzen
"""

import os
import math
import smtplib
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

import joblib
import pandas as pd
import psycopg2
import requests
import xgboost as xgb

from sklearn.metrics import roc_auc_score


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

FEATURE_COLUMNS = [
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

FEATURE_COLUMNS_RELATIEF = [
    "atr14_pct",
    "rsi14",
    "ibs",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "pct_from_high52w",
]

TOP_N_FRACTIE = 0.20

BASELINE_KOLOM = "pct_from_ma50"

AUC_VERSCHIL_WAARSCHUWING = 0.10

MIN_RIJEN_TRAINING = 30
MIN_RIJEN_TEST = 10


# ============================================================
# OMGEVING
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")


# ============================================================
# SQL
# ============================================================

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker,
        datum,
        fwd_ret_10d,
        fwd_ret_30d,
        fwd_ret_60d
    FROM forward_returns
    ORDER BY ticker, datum
)

SELECT
    gt.ticker,
    gt.datum,
    gt.atr14,
    gt.atr14_pct,
    gt.rsi14,
    gt.ibs,
    gt.ma50,
    gt.ma200,
    gt.pct_from_ma50,
    gt.pct_from_ma200,
    gt.vol_ratio_20d,
    gt.high52w,
    gt.pct_from_high52w,
    fr_dedup.fwd_ret_10d,
    fr_dedup.fwd_ret_30d,
    fr_dedup.fwd_ret_60d

FROM generieke_technicals gt

JOIN fr_dedup
    ON fr_dedup.ticker = gt.ticker
    AND fr_dedup.datum = gt.datum;
"""


NAN = float("nan")


# ============================================================
# HULPFUNCTIES
# ============================================================

def _is_nan(v) -> bool:
    if v is None:
        return True

    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _db_waarde(v):
    if _is_nan(v):
        return None

    return v


def _f(v, decimalen: int = 3) -> str:
    if _is_nan(v):
        return "n.v.t."

    return f"{v:.{decimalen}f}"


def _pr(v) -> str:
    if _is_nan(v):
        return "n.v.t."

    return f"{v:+.2f}%"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(tekst: str) -> None:

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    for i in range(0, len(tekst), 4096):

        try:
            r = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": tekst[i:i + 4096],
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )

            if r.status_code != 200:
                print(
                    f"Telegram gaf status {r.status_code}: "
                    f"{r.text[:200]}"
                )

        except Exception as e:
            print(f"Telegram fout: {e}")


# ============================================================
# EMAIL
# ============================================================

def send_email(onderwerp: str, tekst: str) -> None:

    if (
        not EMAIL_USER
        or not EMAIL_PASS
        or not EMAIL_RECEIVER
    ):
        return

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
        )

        server.starttls()

        server.login(
            EMAIL_USER,
            EMAIL_PASS,
        )

        server.send_message(msg)

        server.quit()

        print(
            f"Email verzonden naar {EMAIL_RECEIVER}"
        )

    except Exception as e:
        print(f"Email fout: {e}")


# ============================================================
# DATA
# ============================================================

def get_training_data(conn) -> pd.DataFrame:

    return pd.read_sql(
        JOIN_QUERY,
        conn,
    )


# ============================================================
# EVALUATIE
# ============================================================

def top_n_gemiddelde(
    df: pd.DataFrame,
    sorteer_kolom: str,
    target: str,
    n_top: int,
    aflopend: bool,
) -> float:

    gesorteerd = df.sort_values(
        sorteer_kolom,
        ascending=not aflopend,
    )

    return float(
        gesorteerd
        .head(n_top)[target]
        .mean()
    )


# ============================================================
# MODEL TRAINEN
# ============================================================

def train_en_evalueer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_column: str,
    n_top: int,
):

    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(
        train_df[features],
        train_df["is_profitable"],
    )

    proba = model.predict_proba(
        test_df[features]
    )[:, 1]

    accuratesse = float(
        model.score(
            test_df[features],
            test_df["is_profitable"],
        )
    )

    if test_df["is_profitable"].nunique() < 2:
        auc = NAN
    else:
        auc = float(
            roc_auc_score(
                test_df["is_profitable"],
                proba,
            )
        )

    eval_df = test_df.assign(
        proba=proba
    )

    top_rendement = top_n_gemiddelde(
        eval_df,
        "proba",
        target_column,
        n_top,
        aflopend=True,
    )

    return (
        model,
        accuratesse,
        auc,
        top_rendement,
    )


# ============================================================
# RESULTAAT
# ============================================================

def leeg_resultaat(
    horizon: str,
    status: str,
    opmerking: str,
) -> Dict:

    return {
        "horizon": horizon,
        "status": status,
        "opmerking": opmerking,
        "n_train": None,
        "n_test": None,
        "split_datum": None,
        "accuratesse": NAN,
        "auc": NAN,
        "auc_relatief": NAN,
        "top_n": None,
        "top_model": NAN,
        "top_baseline": NAN,
        "top_relatief": NAN,
        "test_gem": NAN,
    }


# ============================================================
# MODEL OPSLAAN + CONTROLEREN
# ============================================================

def sla_model_veilig_op(
    model,
    horizon: str,
    features: List[str],
) -> str:

    bestandsnaam = (
        f"{MODEL_VERSIE}_{horizon}_model.pkl"
    )

    print(
        f"[{horizon}] Model opslaan naar "
        f"{bestandsnaam}..."
    )

    try:

        joblib.dump(
            model,
            bestandsnaam,
        )

    except Exception as e:

        raise RuntimeError(
            f"[{horizon}] joblib.dump mislukt "
            f"voor {bestandsnaam}: {e}"
        ) from e

    # --------------------------------------------------------
    # BESTAND MOET BESTAAN
    # --------------------------------------------------------

    if not os.path.isfile(bestandsnaam):

        raise RuntimeError(
            f"[{horizon}] Modelbestand bestaat niet "
            f"na joblib.dump: {bestandsnaam}"
        )

    # --------------------------------------------------------
    # BESTAND MAG NIET LEEG ZIJN
    # --------------------------------------------------------

    bestandsgrootte = os.path.getsize(
        bestandsnaam
    )

    if bestandsgrootte <= 0:

        raise RuntimeError(
            f"[{horizon}] MODEL IS LEEG: "
            f"{bestandsnaam} "
            f"({bestandsgrootte} bytes)"
        )

    print(
        f"[{horizon}] Bestandsgrootte: "
        f"{bestandsgrootte:,} bytes"
    )

    # --------------------------------------------------------
    # DIRECT TERUGLADEN
    # --------------------------------------------------------

    try:

        controle_model = joblib.load(
            bestandsnaam
        )

    except Exception as e:

        raise RuntimeError(
            f"[{horizon}] Modelbestand bestaat maar "
            f"kan niet opnieuw geladen worden: "
            f"{bestandsnaam}: {e}"
        ) from e

    # --------------------------------------------------------
    # CONTROLE predict_proba
    # --------------------------------------------------------

    if not hasattr(
        controle_model,
        "predict_proba",
    ):

        raise RuntimeError(
            f"[{horizon}] Het opgeslagen object "
            f"heeft geen predict_proba()."
        )

    # --------------------------------------------------------
    # CONTROLE FEATURES
    # --------------------------------------------------------

    opgeslagen_features = getattr(
        controle_model,
        "feature_names_in_",
        None,
    )

    if opgeslagen_features is None:

        raise RuntimeError(
            f"[{horizon}] Model bevat geen "
            f"feature_names_in_."
        )

    opgeslagen_features = list(
        opgeslagen_features
    )

    if opgeslagen_features != list(features):

        raise RuntimeError(
            f"[{horizon}] Features in opgeslagen "
            f"model komen niet overeen.\n"
            f"Verwacht: {features}\n"
            f"Model:    {opgeslagen_features}"
        )

    print(
        f"[{horizon}] ✅ Model succesvol opgeslagen "
        f"en opnieuw geladen."
    )

    return bestandsnaam


# ============================================================
# TRAINING PER HORIZON
# ============================================================

def train_voor_horizon(
    df: pd.DataFrame,
    horizon: str,
) -> Dict:

    target_column = (
        f"fwd_ret_{horizon}"
    )

    if target_column not in df.columns:

        msg = (
            f"doelkolom '{target_column}' "
            f"ontbreekt in de dataset"
        )

        print(
            f"[{horizon}] {msg}."
        )

        return leeg_resultaat(
            horizon,
            "geen_doelkolom",
            msg,
        )

    df_horizon = df.copy()

    df_horizon["is_profitable"] = (
        df_horizon[target_column] > 0
    ).astype(int)

    df_clean = df_horizon.dropna(
        subset=FEATURE_COLUMNS
        + [target_column, "datum"]
    )

    if len(df_clean) < MIN_RIJEN_TRAINING:

        msg = (
            "nog niet genoeg data met "
            "ingevulde features "
            f"(minimaal {MIN_RIJEN_TRAINING} "
            f"vereist, nu {len(df_clean)})"
        )

        print(
            f"[{horizon}] {msg}."
        )

        return leeg_resultaat(
            horizon,
            "onvoldoende_data",
            msg,
        )

    # --------------------------------------------------------
    # DATUMS NORMALISEREN
    # --------------------------------------------------------

    df_clean = df_clean.copy()

    df_clean["datum"] = pd.to_datetime(
        df_clean["datum"]
    )

    df_clean = (
        df_clean
        .sort_values("datum")
        .reset_index(drop=True)
    )

    unieke_datums = (
        df_clean["datum"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_datum_index = int(
        len(unieke_datums) * 0.80
    )

    if split_datum_index >= len(unieke_datums):

        split_datum_index = (
            len(unieke_datums) - 1
        )

    split_datum = unieke_datums[
        split_datum_index
    ]

    train_df = df_clean[
        df_clean["datum"] < split_datum
    ].copy()

    test_df = df_clean[
        df_clean["datum"] >= split_datum
    ].copy()

    # --------------------------------------------------------
    # TESTSET CONTROLE
    # --------------------------------------------------------

    if len(test_df) < MIN_RIJEN_TEST:

        msg = (
            "te weinig recente data voor "
            "een tijdsgebaseerde test-set "
            f"(nu {len(test_df)}, minimaal "
            f"{MIN_RIJEN_TEST} vereist)"
        )

        print(
            f"[{horizon}] {msg}."
        )

        return leeg_resultaat(
            horizon,
            "te_weinig_testdata",
            msg,
        )

    # --------------------------------------------------------
    # TRAINSET CONTROLE
    # --------------------------------------------------------

    if len(train_df) < MIN_RIJEN_TRAINING:

        msg = (
            "te weinig data in de trainingsset "
            f"(nu {len(train_df)}, minimaal "
            f"{MIN_RIJEN_TRAINING} vereist)"
        )

        print(
            f"[{horizon}] {msg}."
        )

        return leeg_resultaat(
            horizon,
            "te_weinig_trainingdata",
            msg,
        )

    if (
        train_df["is_profitable"]
        .nunique()
        < 2
    ):

        msg = (
            "trainingsset bevat maar "
            "1 klasse (alles winst of alles verlies)"
        )

        print(
            f"[{horizon}] {msg}."
        )

        return leeg_resultaat(
            horizon,
            "een_klasse",
            msg,
        )

    n_top = max(
        1,
        int(
            len(test_df)
            * TOP_N_FRACTIE
        ),
    )

    print(
        f"[{horizon}] Start training op "
        f"{len(train_df)} records "
        f"(vóór {split_datum}), "
        f"test op {len(test_df)} recentere "
        f"records..."
    )

    # --------------------------------------------------------
    # HOOFDMODEL
    # --------------------------------------------------------

    (
        model,
        accuratesse,
        auc,
        top_model,
    ) = train_en_evalueer(
        train_df,
        test_df,
        FEATURE_COLUMNS,
        target_column,
        n_top,
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    top_baseline = top_n_gemiddelde(
        test_df,
        BASELINE_KOLOM,
        target_column,
        n_top,
        aflopend=False,
    )

    test_gem = float(
        test_df[target_column].mean()
    )

    # --------------------------------------------------------
    # CONTROLEMODEL
    # --------------------------------------------------------

    (
        _,
        _,
        auc_relatief,
        top_relatief,
    ) = train_en_evalueer(
        train_df,
        test_df,
        FEATURE_COLUMNS_RELATIEF,
        target_column,
        n_top,
    )

    print(
        f"[{horizon}] Model getraind. "
        f"Test-accuratesse: "
        f"{accuratesse * 100:.2f}% | "
        f"AUC: {_f(auc)}"
    )

    print(
        f"[{horizon}] Top {n_top}/"
        f"{len(test_df)} volgens model: "
        f"{target_column}={top_model:.3f}% | "
        f"baseline ({BASELINE_KOLOM}): "
        f"{top_baseline:.3f}% | "
        f"hele test-set: "
        f"{test_gem:.3f}%"
    )

    print(
        f"[{horizon}] Controle-model: "
        f"AUC {_f(auc_relatief)} | "
        f"top {n_top}: "
        f"{_pr(top_relatief)}"
    )

    # --------------------------------------------------------
    # MODEL OPSLAAN
    # --------------------------------------------------------

    sla_model_veilig_op(
        model,
        horizon,
        FEATURE_COLUMNS,
    )

    return {
        "horizon": horizon,
        "status": "getraind",
        "opmerking": None,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "split_datum": str(
            split_datum.date()
        ),
        "accuratesse": accuratesse,
        "auc": auc,
        "auc_relatief": auc_relatief,
        "top_n": int(n_top),
        "top_model": top_model,
        "top_baseline": top_baseline,
        "top_relatief": top_relatief,
        "test_gem": test_gem,
    }


# ============================================================
# VORIGE RUNS
# ============================================================

def haal_vorige_runs(
    conn,
) -> Dict[str, Dict]:

    vorige: Dict[str, Dict] = {}

    query = """
        SELECT auc, run_datum
        FROM xgboost_runs
        WHERE model_versie = %(model_versie)s
          AND horizon = %(horizon)s
          AND status = 'getraind'
          AND auc IS NOT NULL
        ORDER BY run_datum DESC
        LIMIT 1;
    """

    for horizon in HORIZONS:

        with conn.cursor() as cur:

            cur.execute(
                query,
                {
                    "model_versie": MODEL_VERSIE,
                    "horizon": horizon,
                },
            )

            rij = cur.fetchone()

        if rij:

            vorige[horizon] = {
                "auc": float(rij[0]),
                "run_datum": rij[1].strftime(
                    "%Y-%m-%d"
                ),
            }

    return vorige


# ============================================================
# LOGGEN
# ============================================================

def log_runs(
    conn,
    resultaten: List[Dict],
) -> None:

    query = """
        INSERT INTO xgboost_runs (
            model_versie,
            horizon,
            status,
            opmerking,
            n_train,
            n_test,
            split_datum,
            accuratesse,
            auc,
            auc_relatief,
            top_n,
            top_model_rendement,
            top_baseline_rendement,
            top_relatief_rendement,
            test_gemiddelde
        )
        VALUES (
            %(model_versie)s,
            %(horizon)s,
            %(status)s,
            %(opmerking)s,
            %(n_train)s,
            %(n_test)s,
            %(split_datum)s,
            %(accuratesse)s,
            %(auc)s,
            %(auc_relatief)s,
            %(top_n)s,
            %(top_model)s,
            %(top_baseline)s,
            %(top_relatief)s,
            %(test_gem)s
        );
    """

    with conn.cursor() as cur:

        for res in resultaten:

            params = {
                k: _db_waarde(v)
                for k, v in res.items()
            }

            params["model_versie"] = MODEL_VERSIE

            cur.execute(
                query,
                params,
            )

    conn.commit()


# ============================================================
# WAARSCHUWINGEN
# ============================================================

def waarschuwingen(
    res: Dict,
) -> List[str]:

    w: List[str] = []

    if (
        not _is_nan(res["top_model"])
        and not _is_nan(res["top_baseline"])
        and res["top_model"]
        <= res["top_baseline"]
    ):

        w.append(
            "model doet het niet beter "
            "dan de simpele baseline"
        )

    if (
        not _is_nan(res["auc"])
        and res["auc"] < 0.55
    ):

        w.append(
            "AUC dicht bij 0.50: "
            "weinig voorspellende waarde"
        )

    if _is_nan(
        res["auc_relatief"]
    ):

        w.append(
            "controle-model niet berekenbaar"
        )

    elif (
        not _is_nan(res["auc"])
        and (
            res["auc"]
            - res["auc_relatief"]
        )
        > AUC_VERSCHIL_WAARSCHUWING
    ):

        w.append(
            "AUC valt sterk terug zonder "
            "prijsniveau-features; mogelijk "
            "deels ticker-herkenning"
        )

    return w


# ============================================================
# BERICHT
# ============================================================

def auc_regel(
    res: Dict,
    vorig: Optional[Dict],
) -> str:

    tekst = (
        f"AUC {_f(res['auc'])}"
    )

    if (
        vorig
        and not _is_nan(res["auc"])
    ):

        delta = (
            res["auc"]
            - vorig["auc"]
        )

        tekst += (
            f" (vorige run "
            f"{vorig['run_datum']}: "
            f"{vorig['auc']:.3f}, "
            f"{delta:+.3f})"
        )

    return tekst


def bouw_bericht(
    resultaten: List[Dict],
    vorige: Dict[str, Dict],
    datum: str,
) -> str:

    regels = [
        f"🤖 XGBoostV3 — wekelijkse "
        f"hertraining ({datum})",
        "",
    ]

    for res in resultaten:

        h = res["horizon"]

        if res["status"] != "getraind":

            regels.append(
                f"[{h}] ⏳ "
                f"{res['opmerking']}"
            )

            regels.append("")
            continue

        regels.append(
            f"[{h}] ✅ train "
            f"{res['n_train']} | test "
            f"{res['n_test']} "
            f"(vanaf {res['split_datum']})"
        )

        regels.append(
            f"  {auc_regel(res, vorige.get(h))} | "
            f"accuratesse "
            f"{res['accuratesse'] * 100:.1f}%"
        )

        regels.append(
            f"  Top {res['top_n']} model: "
            f"{_pr(res['top_model'])} | "
            f"baseline ({BASELINE_KOLOM}): "
            f"{_pr(res['top_baseline'])} | "
            f"alle: {_pr(res['test_gem'])}"
        )

        regels.append(
            f"  Controle: AUC "
            f"{_f(res['auc_relatief'])} | "
            f"top {res['top_n']}: "
            f"{_pr(res['top_relatief'])}"
        )

        for w in waarschuwingen(res):

            regels.append(
                f"  ⚠️ {w}"
            )

        regels.append("")

    regels.append(
        "Trend: tabel xgboost_runs "
        "in Supabase."
    )

    return "\n".join(regels)


# ============================================================
# CONTROLE ALLE MODELFILES
# ============================================================

def controleer_alle_modelbestanden() -> None:

    print("")
    print("=" * 70)
    print("CONTROLE VAN ALLE XGBOOSTV3 MODELFILES")
    print("=" * 70)

    fouten = []

    for horizon in HORIZONS:

        bestandsnaam = (
            f"{MODEL_VERSIE}_{horizon}_model.pkl"
        )

        print(
            f"[{horizon}] Controle: "
            f"{bestandsnaam}"
        )

        if not os.path.isfile(
            bestandsnaam
        ):

            fouten.append(
                f"{bestandsnaam}: BESTAND ONTBREEKT"
            )

            continue

        grootte = os.path.getsize(
            bestandsnaam
        )

        print(
            f"[{horizon}] Grootte: "
            f"{grootte:,} bytes"
        )

        if grootte <= 0:

            fouten.append(
                f"{bestandsnaam}: BESTAND IS LEEG"
            )

            continue

        try:

            model = joblib.load(
                bestandsnaam
            )

        except Exception as e:

            fouten.append(
                f"{bestandsnaam}: kan niet laden: {e}"
            )

            continue

        if not hasattr(
            model,
            "predict_proba",
        ):

            fouten.append(
                f"{bestandsnaam}: "
                f"geen predict_proba"
            )

            continue

        features = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if features is None:

            fouten.append(
                f"{bestandsnaam}: "
                f"feature_names_in_ ontbreekt"
            )

            continue

        print(
            f"[{horizon}] ✅ geldig model "
            f"met {len(features)} features."
        )

    print("=" * 70)

    if fouten:

        print("MODEL-CONTROLE MISLUKT:")

        for fout in fouten:
            print(f"  ❌ {fout}")

        raise RuntimeError(
            "Niet alle drie XGBoostV3 "
            "modellen zijn geldig. "
            "Workflow wordt bewust gestopt."
        )

    print(
        "✅ ALLE DRIE MODELLEN ZIJN GELDIG."
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def train_xgboost3() -> None:

    db_url = os.environ.get(
        "SUPABASE_DB_URL"
    )

    if not db_url:

        raise RuntimeError(
            "SUPABASE_DB_URL ontbreekt "
            "in de omgeving."
        )

    datum = (
        dt.datetime.now(
            dt.timezone.utc
        )
        .strftime("%Y-%m-%d")
    )

    print(
        "Dataset ophalen uit Supabase "
        "(generieke_technicals + "
        "forward_returns)..."
    )

    conn = psycopg2.connect(
        db_url
    )

    try:

        df = get_training_data(
            conn
        )

        print(
            f"{len(df)} gejoinde rijen "
            f"opgehaald."
        )

        print(
            f"Aantal features per model: "
            f"{len(FEATURE_COLUMNS)}"
        )

        resultaten = []

        # ----------------------------------------------------
        # ALLE HORIZONS TRAINEN
        # ----------------------------------------------------

        for horizon in HORIZONS:

            resultaat = train_voor_horizon(
                df,
                horizon,
            )

            resultaten.append(
                resultaat
            )

        # ----------------------------------------------------
        # VORIGE RUNS
        # ----------------------------------------------------

        vorige: Dict[str, Dict] = {}

        try:

            vorige = haal_vorige_runs(
                conn
            )

        except Exception as e:

            conn.rollback()

            print(
                "[WARN] vorige runs niet "
                f"op te halen: {e}"
            )

        # ----------------------------------------------------
        # RUN LOGGEN
        # ----------------------------------------------------

        try:

            log_runs(
                conn,
                resultaten,
            )

            print(
                "Run gelogd in "
                "xgboost_runs."
            )

        except Exception as e:

            conn.rollback()

            print(
                "[WARN] loggen in "
                f"xgboost_runs mislukt: {e}"
            )

    finally:

        conn.close()

    # --------------------------------------------------------
    # KRITIEKE CONTROLE
    # --------------------------------------------------------

    # Dit gebeurt buiten de DB-verbinding.
    #
    # Als één model ontbreekt/leeg/corrupt is,
    # krijgt de GitHub Action exit code 1.
    #
    controleer_alle_modelbestanden()

    # --------------------------------------------------------
    # BERICHTEN
    # --------------------------------------------------------

    bericht = bouw_bericht(
        resultaten,
        vorige,
        datum,
    )

    send_telegram(
        bericht
    )

    send_email(
        f"XGBoostV3 hertraining {datum}",
        bericht,
    )

    print("")
    print(
        "✅ XGBoostV3 training volledig "
        "en succesvol afgerond."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train_xgboost3()
