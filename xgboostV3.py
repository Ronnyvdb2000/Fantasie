#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xgboostV3.py  —  wekelijkse hertraining per horizon (10d / 30d / 60d)  v3.1

Traint per horizon een XGBoost-classifier die voorspelt of een aandeel na N
handelsdagen winstgevend is, op basis van strategie-onafhankelijke technische
indicatoren (generieke_technicals) met als label forward_returns.

Evaluatie (tijdsgebaseerde split: train op de oudste 80% van de datums, test
op de strikt recentere 20%): accuratesse, AUC en het gemiddelde rendement van
de top-N% volgens het model, vergeleken met een simpele baseline
(pct_from_ma50) en het gemiddelde van de hele test-set.

NIEUW in v3.1
=============
- Elke run (ook een overgeslagen horizon) wordt gelogd in de Supabase-tabel
  `xgboost_runs`, zodat de trend over de weken te volgen is.
- Na elke run een samenvatting via Telegram en e-mail (naast de tekst in de
  GitHub Actions-log). Bevat de vergelijking met de vorige run.
- CONTROLE-MODEL: dezelfde training, maar enkel met schaal-vrije features
  (geen absolute prijsniveaus zoals ma50/ma200/high52w/atr14). Absolute
  prijsniveaus kunnen als "vingerafdruk" van een ticker werken; een model dat
  daardoor een ticker herkent scoort hoger dan het in werkelijkheid kan
  voorspellen. Valt de AUC zonder die kolommen sterk terug, dan geeft de
  samenvatting een waarschuwing.

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN / TELEGRAM_CHAT_ID en
EMAIL_USER / EMAIL_PASS / EMAIL_RECEIVER (allemaal optioneel: zonder wordt
enkel naar stdout geprint).

Let op: de tabel xgboost_runs moet bestaan. Ontbreekt ze, dan verschijnt een
waarschuwing in de log, maar training en berichten lopen gewoon door.
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

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

MODEL_VERSIE = "xgboostV3"
HORIZONS = ["10d", "30d", "60d"]

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

# Controle-model: enkel schaal-vrije features (geen absolute prijsniveaus).
FEATURE_COLUMNS_RELATIEF = [
    "atr14_pct",
    "rsi14",
    "ibs",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "pct_from_high52w",
]

# Aandeel van de test-set dat als "top-selectie" wordt behandeld (0.20 = top 20%).
TOP_N_FRACTIE = 0.20

# Enige tot nu toe bevestigd significante losse voorspeller in dit project
# (analyseer_forward_correlaties.py): hoe verder onder MA50, hoe hoger het
# rendement nadien. Dient als baseline om te checken of XGBoost iets toevoegt.
BASELINE_KOLOM = "pct_from_ma50"

# Waarschuwing als de AUC met alle features méér dan dit hoger ligt dan die
# van het controle-model (vuistregel, geen wetenschappelijke grens).
AUC_VERSCHIL_WAARSCHUWING = 0.10

MIN_RIJEN_TRAINING = 30
MIN_RIJEN_TEST = 10

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker, datum, fwd_ret_10d, fwd_ret_30d, fwd_ret_60d
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
JOIN fr_dedup ON fr_dedup.ticker = gt.ticker AND fr_dedup.datum = gt.datum;
"""

NAN = float("nan")


# ============================================================
# HULPFUNCTIES
# ============================================================

def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _db_waarde(v):
    """NaN/None -> None voor de database."""
    if _is_nan(v):
        return None
    return v


def _f(v, decimalen: int = 3) -> str:
    return "n.v.t." if _is_nan(v) else f"{v:.{decimalen}f}"


def _pr(v) -> str:
    return "n.v.t." if _is_nan(v) else f"{v:+.2f}%"


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(tekst), 4096):
        try:
            r = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst[i:i + 4096],
                      "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"Telegram gaf status {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"Telegram fout: {e}")


def send_email(onderwerp: str, tekst: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = onderwerp
        msg.attach(MIMEText(tekst, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email fout: {e}")


def get_training_data(conn) -> pd.DataFrame:
    return pd.read_sql(JOIN_QUERY, conn)


# ============================================================
# TRAINING + EVALUATIE
# ============================================================

def top_n_gemiddelde(df: pd.DataFrame, sorteer_kolom: str, target: str,
                     n_top: int, aflopend: bool) -> float:
    gesorteerd = df.sort_values(sorteer_kolom, ascending=not aflopend)
    return float(gesorteerd.head(n_top)[target].mean())


def train_en_evalueer(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      features: List[str], target_column: str, n_top: int):
    """Traint een model en geeft (model, accuratesse, auc, top_n_rendement)."""
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_df[features], train_df["is_profitable"])

    proba = model.predict_proba(test_df[features])[:, 1]
    accuratesse = float(model.score(test_df[features], test_df["is_profitable"]))

    if test_df["is_profitable"].nunique() < 2:
        auc = NAN
    else:
        auc = float(roc_auc_score(test_df["is_profitable"], proba))

    eval_df = test_df.assign(proba=proba)
    top_rendement = top_n_gemiddelde(eval_df, "proba", target_column, n_top, aflopend=True)
    return model, accuratesse, auc, top_rendement


def leeg_resultaat(horizon: str, status: str, opmerking: str) -> Dict:
    return {
        "horizon": horizon, "status": status, "opmerking": opmerking,
        "n_train": None, "n_test": None, "split_datum": None,
        "accuratesse": NAN, "auc": NAN, "auc_relatief": NAN,
        "top_n": None, "top_model": NAN, "top_baseline": NAN,
        "top_relatief": NAN, "test_gem": NAN,
    }


def train_voor_horizon(df: pd.DataFrame, horizon: str) -> Dict:
    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        msg = f"doelkolom '{target_column}' ontbreekt in de dataset"
        print(f"[{horizon}] {msg}.")
        return leeg_resultaat(horizon, "geen_doelkolom", msg)

    df_horizon = df.copy()
    df_horizon["is_profitable"] = (df_horizon[target_column] > 0).astype(int)
    df_clean = df_horizon.dropna(subset=FEATURE_COLUMNS + [target_column, "datum"])

    if len(df_clean) < MIN_RIJEN_TRAINING:
        msg = (f"nog niet genoeg data met ingevulde features "
               f"(minimaal {MIN_RIJEN_TRAINING} vereist, nu {len(df_clean)})")
        print(f"[{horizon}] {msg}.")
        return leeg_resultaat(horizon, "onvoldoende_data", msg)

    # Tijdsgebaseerde split: train op de oudste 80% van de datums, test strikt
    # op de recentere 20% (geen toekomstinformatie in de trainingsset).
    df_clean = df_clean.sort_values("datum").reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.8)
    split_datum = df_clean.iloc[split_idx]["datum"]

    train_df = df_clean[df_clean["datum"] <= split_datum]
    test_df = df_clean[df_clean["datum"] > split_datum]

    if len(test_df) < MIN_RIJEN_TEST:
        msg = (f"te weinig recente data voor een tijdsgebaseerde test-set "
               f"(nu {len(test_df)}, minimaal {MIN_RIJEN_TEST} vereist)")
        print(f"[{horizon}] {msg}.")
        return leeg_resultaat(horizon, "te_weinig_testdata", msg)

    if train_df["is_profitable"].nunique() < 2:
        msg = "trainingsset bevat maar 1 klasse (alles winst of alles verlies)"
        print(f"[{horizon}] {msg}.")
        return leeg_resultaat(horizon, "een_klasse", msg)

    n_top = max(1, int(len(test_df) * TOP_N_FRACTIE))

    print(f"[{horizon}] Start training op {len(train_df)} records (tot en met "
          f"{split_datum}), test op {len(test_df)} recentere records...")

    model, accuratesse, auc, top_model = train_en_evalueer(
        train_df, test_df, FEATURE_COLUMNS, target_column, n_top)

    # Baseline en test-gemiddelde hangen niet van het model af.
    top_baseline = top_n_gemiddelde(test_df, BASELINE_KOLOM, target_column, n_top, aflopend=False)
    test_gem = float(test_df[target_column].mean())

    # Controle-model met enkel schaal-vrije features (wordt niet opgeslagen).
    _, _, auc_relatief, top_relatief = train_en_evalueer(
        train_df, test_df, FEATURE_COLUMNS_RELATIEF, target_column, n_top)

    print(f"[{horizon}] Model getraind. Test-accuratesse: {accuratesse * 100:.2f}%  |  "
          f"AUC: {_f(auc)}" + ("  (0.50 = geen edge boven toeval)" if not _is_nan(auc) else ""))
    print(f"[{horizon}] Top {n_top}/{len(test_df)} volgens model: gemiddeld {target_column}="
          f"{top_model:.3f}%  |  baseline ({BASELINE_KOLOM}): {top_baseline:.3f}%  |  "
          f"hele test-set: {test_gem:.3f}%")
    print(f"[{horizon}] Controle (enkel schaal-vrije features): AUC {_f(auc_relatief)}  |  "
          f"top {n_top}: {_pr(top_relatief)}")

    bestandsnaam = f"{MODEL_VERSIE}_{horizon}_model.pkl"
    joblib.dump(model, bestandsnaam)
    print(f"[{horizon}] Getraind model opgeslagen als {bestandsnaam}")

    return {
        "horizon": horizon, "status": "getraind", "opmerking": None,
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "split_datum": str(split_datum),
        "accuratesse": accuratesse, "auc": auc, "auc_relatief": auc_relatief,
        "top_n": int(n_top), "top_model": top_model, "top_baseline": top_baseline,
        "top_relatief": top_relatief, "test_gem": test_gem,
    }


# ============================================================
# TREND: LOGGEN IN SUPABASE + VORIGE RUN OPHALEN
# ============================================================

def haal_vorige_runs(conn) -> Dict[str, Dict]:
    """Laatste succesvol getrainde run per horizon (vóór de huidige)."""
    vorige: Dict[str, Dict] = {}
    query = """
        SELECT auc, run_datum
        FROM xgboost_runs
        WHERE model_versie = %(model_versie)s AND horizon = %(horizon)s
          AND status = 'getraind' AND auc IS NOT NULL
        ORDER BY run_datum DESC
        LIMIT 1;
    """
    for horizon in HORIZONS:
        with conn.cursor() as cur:
            cur.execute(query, {"model_versie": MODEL_VERSIE, "horizon": horizon})
            rij = cur.fetchone()
        if rij:
            vorige[horizon] = {"auc": float(rij[0]), "run_datum": rij[1].strftime("%Y-%m-%d")}
    return vorige


def log_runs(conn, resultaten: List[Dict]) -> None:
    query = """
        INSERT INTO xgboost_runs (
            model_versie, horizon, status, opmerking, n_train, n_test, split_datum,
            accuratesse, auc, auc_relatief, top_n, top_model_rendement,
            top_baseline_rendement, top_relatief_rendement, test_gemiddelde
        ) VALUES (
            %(model_versie)s, %(horizon)s, %(status)s, %(opmerking)s, %(n_train)s,
            %(n_test)s, %(split_datum)s, %(accuratesse)s, %(auc)s, %(auc_relatief)s,
            %(top_n)s, %(top_model)s, %(top_baseline)s, %(top_relatief)s, %(test_gem)s
        );
    """
    with conn.cursor() as cur:
        for res in resultaten:
            params = {k: _db_waarde(v) for k, v in res.items()}
            params["model_versie"] = MODEL_VERSIE
            cur.execute(query, params)
    conn.commit()


# ============================================================
# BERICHT (Telegram + e-mail)
# ============================================================

def waarschuwingen(res: Dict) -> List[str]:
    w: List[str] = []
    if not _is_nan(res["top_model"]) and not _is_nan(res["top_baseline"]) \
            and res["top_model"] <= res["top_baseline"]:
        w.append("model doet het niet beter dan de simpele baseline")
    if not _is_nan(res["auc"]) and res["auc"] < 0.55:
        w.append("AUC dicht bij 0.50: weinig voorspellende waarde")
    if _is_nan(res["auc_relatief"]):
        w.append("controle-model niet berekenbaar")
    elif not _is_nan(res["auc"]) and res["auc"] - res["auc_relatief"] > AUC_VERSCHIL_WAARSCHUWING:
        w.append("AUC valt sterk terug zonder prijsniveau-features (atr14/ma50/ma200/high52w): "
                 "mogelijk deels ticker-herkenning, voorzichtig interpreteren")
    return w


def auc_regel(res: Dict, vorig: Optional[Dict]) -> str:
    tekst = f"AUC {_f(res['auc'])}"
    if vorig and not _is_nan(res["auc"]):
        delta = res["auc"] - vorig["auc"]
        tekst += f" (vorige run {vorig['run_datum']}: {vorig['auc']:.3f}, {delta:+.3f})"
    return tekst


def bouw_bericht(resultaten: List[Dict], vorige: Dict[str, Dict], datum: str) -> str:
    regels = [f"🤖 XGBoostV3 — wekelijkse hertraining ({datum})", ""]
    for res in resultaten:
        h = res["horizon"]
        if res["status"] != "getraind":
            regels.append(f"[{h}] ⏳ {res['opmerking']}")
            regels.append("")
            continue
        regels.append(f"[{h}] ✅ train {res['n_train']} | test {res['n_test']} (na {res['split_datum']})")
        regels.append(f"  {auc_regel(res, vorige.get(h))}  |  accuratesse {res['accuratesse'] * 100:.1f}%")
        regels.append(f"  Top {res['top_n']} model: {_pr(res['top_model'])}  |  baseline "
                      f"({BASELINE_KOLOM}): {_pr(res['top_baseline'])}  |  alle: {_pr(res['test_gem'])}")
        regels.append(f"  Controle (schaal-vrije features): AUC {_f(res['auc_relatief'])}  |  "
                      f"top {res['top_n']}: {_pr(res['top_relatief'])}")
        for w in waarschuwingen(res):
            regels.append(f"  ⚠️ {w}")
        regels.append("")
    regels.append("Trend: tabel xgboost_runs in Supabase.")
    return "\n".join(regels)


# ============================================================
# MAIN
# ============================================================

def train_xgboost3() -> None:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt in de omgeving.")

    datum = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    print("Dataset ophalen uit Supabase (generieke_technicals + forward_returns, "
          "join op ticker+datum, horizons: " + ", ".join(HORIZONS) + ")...")

    conn = psycopg2.connect(db_url)
    try:
        df = get_training_data(conn)
        print(f"{len(df)} gejoinde rijen opgehaald.")
        print(f"Aantal features per model: {len(FEATURE_COLUMNS)} "
              f"(controle-model: {len(FEATURE_COLUMNS_RELATIEF)})")

        resultaten = [train_voor_horizon(df, h) for h in HORIZONS]

        # Vorige runs ophalen VOOR de huidige gelogd wordt (voor de trend-regel).
        vorige: Dict[str, Dict] = {}
        try:
            vorige = haal_vorige_runs(conn)
        except Exception as e:
            conn.rollback()
            print(f"[WARN] vorige runs niet op te halen (bestaat tabel xgboost_runs?): {e}")

        try:
            log_runs(conn, resultaten)
            print("Run gelogd in xgboost_runs.")
        except Exception as e:
            conn.rollback()
            print(f"[WARN] loggen in xgboost_runs mislukt: {e}")
    finally:
        conn.close()

    bericht = bouw_bericht(resultaten, vorige, datum)
    send_telegram(bericht)
    send_email(f"XGBoostV3 hertraining {datum}", bericht)


if __name__ == "__main__":
    train_xgboost3()
