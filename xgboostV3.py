#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgboostV3.py — wekelijkse hertraining per horizon (10d / 30d / 60d)

Versie 3.3

Belangrijk:
- Alleen succesvol getrainde modellen worden opgeslagen.
- Modellen worden eerst naar een tijdelijk bestand geschreven.
- Het tijdelijke bestand wordt gecontroleerd en daarna atomair vervangen.
- Een mislukte/te kleine/corrupte nieuwe training overschrijft nooit
  een bestaand geldig model.
- Een horizon met onvoldoende data veroorzaakt GEEN workflow failure.
- Ontbrekende modellen worden als "nog niet beschikbaar" behandeld.
- Een bestaand geldig model blijft behouden als een nieuwe training
  voor die horizon nog niet mogelijk is.
- Tijdssplit gebeurt op volledige datumgrenzen.
"""

import os
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

# Een geldig joblib-model hoort ruim groter te zijn dan dit.
# Dit is uitsluitend een sanity check; de echte controle
# gebeurt door joblib.load + predict_proba + features.
MIN_MODEL_BYTES = 1000


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
    ORDER BY
        ticker,
        datum,
        (
            (fwd_ret_10d IS NOT NULL)::int +
            (fwd_ret_30d IS NOT NULL)::int +
            (fwd_ret_60d IS NOT NULL)::int
        ) DESC
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
                    f"Telegram gaf status "
                    f"{r.status_code}: "
                    f"{r.text[:200]}"
                )

        except Exception as e:

            print(
                f"Telegram fout: {e}"
            )


# ============================================================
# EMAIL
# ============================================================

def send_email(
    onderwerp: str,
    tekst: str,
) -> None:

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
            f"Email verzonden naar "
            f"{EMAIL_RECEIVER}"
        )

    except Exception as e:

        print(
            f"Email fout: {e}"
        )


# ============================================================
# DATA
# ============================================================

def get_training_data(
    conn,
) -> pd.DataFrame:

    return pd.read_sql(
        JOIN_QUERY,
        conn,
    )


def toon_data_diagnose(
    df: pd.DataFrame,
) -> None:

    print("")
    print("=" * 70)
    print("DATA-DIAGNOSE PER HORIZON")
    print("=" * 70)

    for horizon in HORIZONS:

        target = f"fwd_ret_{horizon}"

        if target not in df.columns:

            print(
                f"[{horizon}] {target}: "
                f"ONTBREEKT"
            )

            continue

        aantal_target = int(
            df[target].notna().sum()
        )

        aantal_datums = int(
            df.loc[
                df[target].notna(),
                "datum",
            ].nunique()
        )

        df_temp = df.dropna(
            subset=FEATURE_COLUMNS
            + [target, "datum"]
        )

        print(
            f"[{horizon}] "
            f"{target}: "
            f"{aantal_target} niet-lege labels | "
            f"{aantal_datums} datums | "
            f"{len(df_temp)} complete records"
        )

    print("=" * 70)
    print("")


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
# MODEL VALIDATIE
# ============================================================

def valideer_modelbestand(
    bestandsnaam: str,
    features: List[str],
    horizon: str,
    stil: bool = False,
) -> bool:

    if not os.path.isfile(bestandsnaam):

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"bestand ontbreekt: "
                f"{bestandsnaam}"
            )

        return False

    try:

        grootte = os.path.getsize(
            bestandsnaam
        )

    except Exception as e:

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"bestandsgrootte niet "
                f"leesbaar: {e}"
            )

        return False

    if grootte < MIN_MODEL_BYTES:

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"bestand te klein: "
                f"{grootte} bytes"
            )

        return False

    try:

        model = joblib.load(
            bestandsnaam
        )

    except Exception as e:

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"joblib.load mislukt: "
                f"{e}"
            )

        return False

    if not hasattr(
        model,
        "predict_proba",
    ):

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"model heeft geen "
                f"predict_proba()"
            )

        return False

    opgeslagen_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if opgeslagen_features is None:

        if not stil:
            print(
                f"[{horizon}] ❌ "
                f"feature_names_in_ ontbreekt"
            )

        return False

    opgeslagen_features = list(
        opgeslagen_features
    )

    if opgeslagen_features != list(
        features
    ):

        if not stil:

            print(
                f"[{horizon}] ❌ "
                f"features komen niet overeen."
            )

            print(
                f"    Verwacht: "
                f"{features}"
            )

            print(
                f"    Model:    "
                f"{opgeslagen_features}"
            )

        return False

    if not stil:

        print(
            f"[{horizon}] ✅ geldig model "
            f"({grootte:,} bytes, "
            f"{len(opgeslagen_features)} features)"
        )

    return True


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

    tijdelijk_bestand = (
        f"{bestandsnaam}.tmp"
    )

    print(
        f"[{horizon}] Model veilig opslaan "
        f"naar {bestandsnaam}..."
    )

    # --------------------------------------------------------
    # Eventueel oud tijdelijk bestand verwijderen
    # --------------------------------------------------------

    if os.path.exists(
        tijdelijk_bestand
    ):

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Eerst naar .tmp schrijven
    # --------------------------------------------------------

    try:

        joblib.dump(
            model,
            tijdelijk_bestand,
        )

    except Exception as e:

        if os.path.exists(
            tijdelijk_bestand
        ):

            try:
                os.remove(
                    tijdelijk_bestand
                )
            except Exception:
                pass

        raise RuntimeError(
            f"[{horizon}] joblib.dump "
            f"mislukt: {e}"
        ) from e

    # --------------------------------------------------------
    # Tijdelijk bestand controleren
    # --------------------------------------------------------

    if not os.path.isfile(
        tijdelijk_bestand
    ):

        raise RuntimeError(
            f"[{horizon}] tijdelijk "
            f"modelbestand bestaat niet."
        )

    grootte = os.path.getsize(
        tijdelijk_bestand
    )

    print(
        f"[{horizon}] Tijdelijk model: "
        f"{grootte:,} bytes"
    )

    if grootte < MIN_MODEL_BYTES:

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

        raise RuntimeError(
            f"[{horizon}] tijdelijk "
            f"modelbestand is ongeldig "
            f"({grootte} bytes)."
        )

    # --------------------------------------------------------
    # Tijdelijk bestand laden
    # --------------------------------------------------------

    try:

        controle_model = joblib.load(
            tijdelijk_bestand
        )

    except Exception as e:

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

        raise RuntimeError(
            f"[{horizon}] tijdelijk "
            f"model kan niet geladen "
            f"worden: {e}"
        ) from e

    # --------------------------------------------------------
    # predict_proba
    # --------------------------------------------------------

    if not hasattr(
        controle_model,
        "predict_proba",
    ):

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

        raise RuntimeError(
            f"[{horizon}] model heeft "
            f"geen predict_proba()."
        )

    # --------------------------------------------------------
    # Featurecontrole
    # --------------------------------------------------------

    opgeslagen_features = getattr(
        controle_model,
        "feature_names_in_",
        None,
    )

    if opgeslagen_features is None:

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

        raise RuntimeError(
            f"[{horizon}] model bevat "
            f"geen feature_names_in_."
        )

    opgeslagen_features = list(
        opgeslagen_features
    )

    if opgeslagen_features != list(
        features
    ):

        try:
            os.remove(
                tijdelijk_bestand
            )
        except Exception:
            pass

        raise RuntimeError(
            f"[{horizon}] features "
            f"komen niet overeen.\n"
            f"Verwacht: {features}\n"
            f"Model:    {opgeslagen_features}"
        )

    # --------------------------------------------------------
    # Atomair vervangen
    #
    # BELANGRIJK:
    # Een bestaand geldig model wordt pas hier vervangen,
    # nadat het nieuwe model volledig gecontroleerd is.
    # --------------------------------------------------------

    try:

        os.replace(
            tijdelijk_bestand,
            bestandsnaam,
        )

    except Exception as e:

        if os.path.exists(
            tijdelijk_bestand
        ):

            try:
                os.remove(
                    tijdelijk_bestand
                )
            except Exception:
                pass

        raise RuntimeError(
            f"[{horizon}] atomair "
            f"vervangen van model "
            f"mislukt: {e}"
        ) from e

    # --------------------------------------------------------
    # Eindcontrole na replace
    # --------------------------------------------------------

    if not valideer_modelbestand(
        bestandsnaam,
        features,
        horizon,
        stil=False,
    ):

        raise RuntimeError(
            f"[{horizon}] modelbestand "
            f"faalt eindcontrole na opslaan."
        )

    print(
        f"[{horizon}] ✅ Model succesvol "
        f"opgeslagen en gevalideerd."
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
            f"[{horizon}] ⏭️ {msg}."
        )

        return leeg_resultaat(
            horizon,
            "geen_doelkolom",
            msg,
        )

    df_horizon = df.copy()

    # Eerst controleren hoeveel labels werkelijk bestaan.
    aantal_labels = int(
        df_horizon[target_column]
        .notna()
        .sum()
    )

    if aantal_labels == 0:

        msg = (
            f"geen ingevulde {target_column} "
            f"labels beschikbaar"
        )

        print(
            f"[{horizon}] ⏳ {msg}."
        )

        return leeg_resultaat(
            horizon,
            "onvoldoende_data",
            msg,
        )

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
            f"[{horizon}] ⏳ {msg}."
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

    if len(unieke_datums) < 2:

        msg = (
            "onvoldoende verschillende "
            "datums voor een tijdssplit"
        )

        print(
            f"[{horizon}] ⏳ {msg}."
        )

        return leeg_resultaat(
            horizon,
            "onvoldoende_datums",
            msg,
        )

    split_datum_index = int(
        len(unieke_datums) * 0.80
    )

    # Zorg dat er altijd minstens één datum
    # voor de testperiode overblijft.
    split_datum_index = min(
        max(split_datum_index, 1),
        len(unieke_datums) - 1,
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
            f"[{horizon}] ⏳ {msg}."
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
            f"[{horizon}] ⏳ {msg}."
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
            f"[{horizon}] ⏳ {msg}."
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
        f"test op {len(test_df)} "
        f"recentere records..."
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
        f"{target_column}="
        f"{top_model:.3f}% | "
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
    # MODEL VEILIG OPSLAAN
    # --------------------------------------------------------

    try:

        sla_model_veilig_op(
            model,
            horizon,
            FEATURE_COLUMNS,
        )

    except Exception as e:

        # Een nieuwe training mag nooit eindigen met
        # een corrupt modelbestand.
        print(
            f"[{horizon}] ❌ "
            f"Model niet gepubliceerd: {e}"
        )

        raise

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

    if _is_nan(res["auc"]):

        tekst = "AUC n.v.t."

    else:

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


def model_status(
    horizon: str,
) -> str:

    bestandsnaam = (
        f"{MODEL_VERSIE}_{horizon}_model.pkl"
    )

    if valideer_modelbestand(
        bestandsnaam,
        FEATURE_COLUMNS,
        horizon,
        stil=True,
    ):

        return "bestaand geldig model"

    return "nog geen geldig model"


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

            status_model = model_status(
                h
            )

            regels.append(
                f"[{h}] ⏳ "
                f"{res['opmerking']}"
            )

            regels.append(
                f"  Modelstatus: "
                f"{status_model}"
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

def controleer_alle_modelbestanden() -> bool:

    print("")
    print("=" * 70)
    print("CONTROLE VAN XGBOOSTV3 MODELFILES")
    print("=" * 70)

    aantal_geldig = 0

    for horizon in HORIZONS:

        bestandsnaam = (
            f"{MODEL_VERSIE}_{horizon}_model.pkl"
        )

        print(
            f"[{horizon}] Controle: "
            f"{bestandsnaam}"
        )

        geldig = valideer_modelbestand(
            bestandsnaam,
            FEATURE_COLUMNS,
            horizon,
            stil=False,
        )

        if geldig:

            aantal_geldig += 1

        else:

            print(
                f"[{horizon}] ⚠️ "
                f"Geen geldig model beschikbaar."
            )

    print("=" * 70)

    if aantal_geldig == 0:

        print(
            "❌ GEEN ENKEL GELDIG "
            "XGBOOSTV3 MODEL BESCHIKBAAR."
        )

        raise RuntimeError(
            "Er is geen enkel geldig "
            "XGBoostV3 model beschikbaar."
        )

    print(
        f"✅ {aantal_geldig}/"
        f"{len(HORIZONS)} "
        f"XGBoostV3 modellen geldig."
    )

    print(
        "Niet-beschikbare horizons worden "
        "overgeslagen totdat voldoende "
        "historische data beschikbaar is."
    )

    print("=" * 70)

    return True


# ============================================================
# ONGELDIGE PLACEHOLDERS OPRUIMEN
# ============================================================

def verwijder_ongeldige_placeholders() -> None:

    print("")
    print(
        "Controle op oude lege/corrupte "
        "modelbestanden..."
    )

    for horizon in HORIZONS:

        bestandsnaam = (
            f"{MODEL_VERSIE}_{horizon}_model.pkl"
        )

        if not os.path.isfile(
            bestandsnaam
        ):
            continue

        geldig = valideer_modelbestand(
            bestandsnaam,
            FEATURE_COLUMNS,
            horizon,
            stil=True,
        )

        if geldig:
            continue

        grootte = os.path.getsize(
            bestandsnaam
        )

        # Alleen duidelijk ongeldige kleine bestanden
        # automatisch verwijderen.
        if grootte < MIN_MODEL_BYTES:

            try:

                os.remove(
                    bestandsnaam
                )

                print(
                    f"[{horizon}] 🧹 "
                    f"Oude ongeldige placeholder "
                    f"verwijderd: "
                    f"{bestandsnaam} "
                    f"({grootte} bytes)"
                )

            except Exception as e:

                raise RuntimeError(
                    f"[{horizon}] Kan ongeldig "
                    f"placeholderbestand niet "
                    f"verwijderen: {e}"
                ) from e

        else:

            # Een groter maar corrupt bestand mag niet
            # automatisch verwijderd worden.
            print(
                f"[{horizon}] ⚠️ Ongeldig "
                f"modelbestand van {grootte:,} bytes "
                f"blijft staan voor handmatige controle."
            )


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

    resultaten = []
    vorige: Dict[str, Dict] = {}

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

        toon_data_diagnose(
            df
        )

        # ----------------------------------------------------
        # OUDE 1-BYTE PLACEHOLDERS VERWIJDEREN
        # ----------------------------------------------------

        verwijder_ongeldige_placeholders()

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
    # MODEL-CONTROLE
    # --------------------------------------------------------

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
