#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xgboostV4.py — wekelijkse hertraining met uitgebreide features
=================================================================

VOORTBOUWEND OP xgboostV3.py:
- Behoudt de veilige atomaire model-opslag
- Behoudt het notificatiesysteem
- Behoudt de statuslogica (geen workflow failure bij te weinig data)

NIEUW t.o.v. V3:
1. Uitgebreide feature-set: extra technische + fundamentele features
2. Drie modellen per horizon: technisch / technisch+fundamenteel / fundamenteel
3. Ranking-target als alternatief voor classificatie (abs / top25)
4. Baseline-AUC berekening (naast baseline top-20%)
5. Dynamische feature-selectie: features met <50% dekking worden
   automatisch uitgesloten (voorkomt dat dropna de hele dataset wist
   door één all-NaN kolom zoals hv60 of dagen_sinds_low52w)
6. Feature-diagnose per horizon: toont per feature de dekking

Belangrijk:
- Een horizon met onvoldoende data geeft status "overgeslagen".
- Een bestaand geldig model blijft behouden als een nieuwe training
  voor die horizon nog niet mogelijk is.
- De run faalt alleen als GEEN enkele horizon een model oplevert.
"""

import os
import smtplib
import warnings
import datetime as dt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psycopg2
import requests
import xgboost as xgb

from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


# ============================================================
# CONFIGURATIE
# ============================================================

MODEL_VERSIE = "xgboostV4"
HORIZONS = ["10d", "30d", "60d"]

# Uitgebreide technische features (bron: generieke_technicals)
FEATURE_TECHNISCH = [
    # Baseline V3
    "atr14_pct", "rsi14", "ibs",
    "pct_from_ma50", "pct_from_ma200",
    "vol_ratio_20d", "pct_from_high52w",
    # Nieuw in V4
    "macd_hist", "bb_percent_b", "stoch_k", "stoch_d",
    "adx14", "rel_sterkte_20d",
    "hv20", "hv60", "dagen_sinds_low52w", "vol_ratio_50d",
]

# Fundamentele features (bron: generieke_fundamentals)
FEATURE_FUNDAMENTEEL = [
    "trailing_pe", "price_to_book", "fcf_yield", "dividend_yield",
    "piotroski_score", "revenue_growth_pct", "eps_growth_pct",
    "net_debt_ebitda", "current_ratio", "analisten_count",
]

FEATURE_ALLES = FEATURE_TECHNISCH + FEATURE_FUNDAMENTEEL

# Schaalvrij controlemodel (geen absolute prijsniveaus)
FEATURE_RELATIEF = [
    "atr14_pct", "rsi14", "ibs",
    "pct_from_ma50", "pct_from_ma200",
    "vol_ratio_20d", "pct_from_high52w",
    "macd_hist", "bb_percent_b", "stoch_k", "adx14", "rel_sterkte_20d",
]

# Minimale dekking per feature (0.50 = 50% van de rijen met geldig target
# moet een niet-NaN waarde hebben, anders wordt de feature uitgesloten)
MIN_FEATURE_DEKKING = 0.50

# Minimale aantal bruikbare features om een model te trainen
MIN_FEATURES_TECHNISCH = 3
MIN_FEATURES_FUNDAMENTEEL = 3

TOP_N_FRACTIE = 0.20
BASELINE_KOLOM = "pct_from_ma50"
AUC_VERSCHIL_WAARSCHUWING = 0.10

MIN_RIJEN_TRAINING = 30
MIN_RIJEN_TEST = 10
MIN_MODEL_BYTES = 1000

# Ranking-target optie: "abs" = fwd_ret > 0 (V3), "top25" = top 25% per dag
TARGET_MODE = os.environ.get("XGB_TARGET_MODE", "abs")


# ============================================================
# OMGEVING
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

NAN = float("nan")


# ============================================================
# SQL — met LEFT JOIN naar fundamentals
# ============================================================

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker, datum,
        fwd_ret_10d, fwd_ret_30d, fwd_ret_60d
    FROM forward_returns
    ORDER BY
        ticker, datum,
        (
            (fwd_ret_10d IS NOT NULL)::int +
            (fwd_ret_30d IS NOT NULL)::int +
            (fwd_ret_60d IS NOT NULL)::int
        ) DESC
),
fund_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker, datum,
        market_cap, trailing_pe, price_to_book, dividend_yield,
        current_ratio, revenue_growth_pct, fcf_yield,
        net_debt_ebitda, payout_pct, analisten_count,
        piotroski_score, eps_growth_pct, eps_cagr_pct, peg_ratio
    FROM generieke_fundamentals
    ORDER BY ticker, datum DESC
)
SELECT
    gt.ticker,
    gt.datum,
    -- Technisch
    gt.atr14, gt.atr14_pct, gt.rsi14, gt.ibs,
    gt.ma50, gt.ma200, gt.pct_from_ma50, gt.pct_from_ma200,
    gt.vol_ratio_20d, gt.high52w, gt.pct_from_high52w,
    gt.macd, gt.macd_signaal, gt.macd_hist,
    gt.bb_breedte, gt.bb_percent_b, gt.stoch_k, gt.stoch_d,
    gt.adx14, gt.rel_sterkte_20d, gt.hv20, gt.hv60,
    gt.dagen_sinds_low52w, gt.vol_ratio_50d,
    -- Fundamenteel
    fd.market_cap, fd.trailing_pe, fd.price_to_book, fd.dividend_yield,
    fd.current_ratio, fd.revenue_growth_pct, fd.fcf_yield,
    fd.net_debt_ebitda, fd.payout_pct, fd.analisten_count,
    fd.piotroski_score, fd.eps_growth_pct, fd.eps_cagr_pct, fd.peg_ratio,
    -- Targets
    fr_dedup.fwd_ret_10d, fr_dedup.fwd_ret_30d, fr_dedup.fwd_ret_60d
FROM generieke_technicals gt
JOIN fr_dedup
    ON fr_dedup.ticker = gt.ticker
    AND fr_dedup.datum = gt.datum
LEFT JOIN fund_dedup fd
    ON fd.ticker = gt.ticker
    AND fd.datum = gt.datum;
"""


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
    return None if _is_nan(v) else v


def _f(v, decimalen: int = 3) -> str:
    return "n.v.t." if _is_nan(v) else f"{v:.{decimalen}f}"


def _pr(v) -> str:
    return "n.v.t." if _is_nan(v) else f"{v:+.2f}%"


# ============================================================
# NOTIFICATIES
# ============================================================

def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(tekst), 4096):
        try:
            r = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID,
                      "text": tekst[i:i + 4096],
                      "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                print(f"Telegram status {r.status_code}: {r.text[:200]}")
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


# ============================================================
# DATA
# ============================================================

def get_training_data(conn) -> pd.DataFrame:
    return pd.read_sql(JOIN_QUERY, conn)


def toon_data_diagnose(df: pd.DataFrame) -> None:
    print("")
    print("=" * 70)
    print("DATA-DIAGNOSE PER HORIZON")
    print("=" * 70)

    for horizon in HORIZONS:
        target = f"fwd_ret_{horizon}"
        if target not in df.columns:
            print(f"[{horizon}] {target}: ONTBREEKT")
            continue

        n_target = int(df[target].notna().sum())
        n_datums = int(df.loc[df[target].notna(), "datum"].nunique())

        print(
            f"[{horizon}] {target}: "
            f"{n_target} labels | {n_datums} datums"
        )

    print("=" * 70)


def diagnose_features(df: pd.DataFrame, target_column: str) -> None:
    """Toont per feature de dekking binnen rijen met een geldig target."""
    print()
    print("=" * 78)
    print(f"FEATURE-DIAGNOSE voor {target_column}")
    print("=" * 78)

    mask = df[target_column].notna()
    subset = df[mask]
    n_totaal = len(subset)

    if n_totaal == 0:
        print(f"  Geen rijen met geldig {target_column}.")
        return

    print(f"  Basis: {n_totaal} rijen met geldig target")
    print(f"  {'Feature':<24} {'Gevuld':>8} {'Dekking':>9}  Status")
    print("  " + "-" * 62)

    alle_features = FEATURE_TECHNISCH + FEATURE_FUNDAMENTEEL
    for feat in alle_features:
        if feat not in subset.columns:
            print(f"  {feat:<24} {'-':>8} {'ONTBREEKT':>9}")
            continue
        n_ok = int(subset[feat].notna().sum())
        pct = n_ok / n_totaal * 100
        if pct >= 90:
            status = "OK"
        elif pct >= 50:
            status = "zwak"
        elif pct > 0:
            status = "⚠️  te weinig"
        else:
            status = "❌ ALL-NaN"
        print(f"  {feat:<24} {n_ok:>8} {pct:>8.1f}%  {status}")

    print("=" * 78)


# ============================================================
# DYNAMISCHE FEATURE-SELECTIE
# ============================================================

def selecteer_features(
    df: pd.DataFrame,
    kandidaten: List[str],
    target_column: str,
    min_dekking: float = MIN_FEATURE_DEKKING,
) -> Tuple[List[str], List[str]]:
    """
    Retourneert (bruikbare_features, uitgesloten_features).
    Een feature is bruikbaar als hij bestaat en voor minstens `min_dekking`
    niet-NaN is binnen de rijen met een geldig target.
    Volgorde van `kandidaten` blijft behouden.
    """
    mask = df[target_column].notna()
    subset = df[mask]
    n = len(subset)

    if n == 0:
        return [], list(kandidaten)

    bruikbaar = []
    uitgesloten = []
    for feat in kandidaten:
        if feat not in subset.columns:
            uitgesloten.append(feat)
            continue
        pct = subset[feat].notna().sum() / n
        if pct >= min_dekking:
            bruikbaar.append(feat)
        else:
            uitgesloten.append(feat)

    return bruikbaar, uitgesloten


# ============================================================
# EVALUATIE
# ============================================================

def top_n_gemiddelde(df, sorteer_kolom, target, n_top, aflopend) -> float:
    gesorteerd = df.sort_values(sorteer_kolom, ascending=not aflopend)
    return float(gesorteerd.head(n_top)[target].mean())


def baseline_auc(df, target_col, baseline_kolom) -> float:
    """AUC van de baseline als ranking (laagste pct_from_ma50 = hoogste prob)."""
    if df["is_profitable"].nunique() < 2:
        return NAN
    prob_baseline = -df[baseline_kolom].rank(pct=True)
    return float(roc_auc_score(df["is_profitable"], prob_baseline))


# ============================================================
# MODEL TRAINEN
# ============================================================

def train_en_evalueer(train_df, test_df, features, target_column, n_top):
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        missing=np.nan,
    )

    model.fit(train_df[features], train_df["is_profitable"])

    proba = model.predict_proba(test_df[features])[:, 1]

    accuratesse = float(model.score(test_df[features], test_df["is_profitable"]))

    auc = NAN if test_df["is_profitable"].nunique() < 2 else float(
        roc_auc_score(test_df["is_profitable"], proba)
    )

    eval_df = test_df.assign(proba=proba)
    top_rendement = top_n_gemiddelde(eval_df, "proba", target_column, n_top, True)

    return model, accuratesse, auc, top_rendement


# ============================================================
# RESULTAAT (leeg)
# ============================================================

def leeg_resultaat(horizon, status, opmerking) -> Dict:
    return {
        "horizon": horizon, "status": status, "opmerking": opmerking,
        "n_train": None, "n_test": None, "split_datum": None,
        "accuratesse": NAN, "auc": NAN, "auc_relatief": NAN,
        "auc_technisch": NAN, "auc_fundamenteel": NAN, "auc_baseline": NAN,
        "top_n": None,
        "top_model": NAN, "top_baseline": NAN, "top_relatief": NAN,
        "top_technisch": NAN, "top_fundamenteel": NAN,
        "test_gem": NAN, "target_mode": TARGET_MODE,
    }


# ============================================================
# MODEL VALIDATIE + OPSLAG
# ============================================================

def valideer_modelbestand(bestandsnaam, features, horizon, stil=False) -> bool:
    if not os.path.isfile(bestandsnaam):
        if not stil:
            print(f"[{horizon}] ❌ bestand ontbreekt: {bestandsnaam}")
        return False
    try:
        grootte = os.path.getsize(bestandsnaam)
    except Exception as e:
        if not stil:
            print(f"[{horizon}] ❌ grootte niet leesbaar: {e}")
        return False
    if grootte < MIN_MODEL_BYTES:
        if not stil:
            print(f"[{horizon}] ❌ te klein: {grootte} bytes")
        return False
    try:
        model = joblib.load(bestandsnaam)
    except Exception as e:
        if not stil:
            print(f"[{horizon}] ❌ joblib.load mislukt: {e}")
        return False
    if not hasattr(model, "predict_proba"):
        if not stil:
            print(f"[{horizon}] ❌ geen predict_proba()")
        return False

    opgeslagen = getattr(model, "feature_names_in_", None)
    if opgeslagen is None:
        if not stil:
            print(f"[{horizon}] ❌ feature_names_in_ ontbreekt")
        return False

    if list(opgeslagen) != list(features):
        if not stil:
            print(f"[{horizon}] ❌ features komen niet overeen")
        return False

    if not stil:
        print(f"[{horizon}] ✅ geldig model ({grootte:,} bytes, "
              f"{len(opgeslagen)} features)")
    return True


def sla_model_veilig_op(model, horizon, features, variant="tech_fund") -> str:
    bestandsnaam = f"{MODEL_VERSIE}_{horizon}_{variant}.pkl"
    tijdelijk = f"{bestandsnaam}.tmp"

    if os.path.exists(tijdelijk):
        try: os.remove(tijdelijk)
        except Exception: pass

    try:
        joblib.dump(model, tijdelijk)
    except Exception as e:
        if os.path.exists(tijdelijk):
            try: os.remove(tijdelijk)
            except Exception: pass
        raise RuntimeError(f"[{horizon}] joblib.dump mislukt: {e}") from e

    if not os.path.isfile(tijdelijk):
        raise RuntimeError(f"[{horizon}] tijdelijk bestand ontbreekt")

    grootte = os.path.getsize(tijdelijk)
    if grootte < MIN_MODEL_BYTES:
        try: os.remove(tijdelijk)
        except Exception: pass
        raise RuntimeError(f"[{horizon}] tijdelijk bestand te klein: {grootte}")

    try:
        controle = joblib.load(tijdelijk)
    except Exception as e:
        try: os.remove(tijdelijk)
        except Exception: pass
        raise RuntimeError(f"[{horizon}] tijdelijk model niet laadbaar: {e}") from e

    if not hasattr(controle, "predict_proba"):
        try: os.remove(tijdelijk)
        except Exception: pass
        raise RuntimeError(f"[{horizon}] geen predict_proba()")

    opgeslagen = getattr(controle, "feature_names_in_", None)
    if opgeslagen is None or list(opgeslagen) != list(features):
        try: os.remove(tijdelijk)
        except Exception: pass
        raise RuntimeError(f"[{horizon}] feature-mismatch")

    try:
        os.replace(tijdelijk, bestandsnaam)
    except Exception as e:
        if os.path.exists(tijdelijk):
            try: os.remove(tijdelijk)
            except Exception: pass
        raise RuntimeError(f"[{horizon}] atomair vervangen mislukt: {e}") from e

    if not valideer_modelbestand(bestandsnaam, features, horizon, stil=False):
        raise RuntimeError(f"[{horizon}] eindcontrole mislukt")

    print(f"[{horizon}] ✅ model opgeslagen: {bestandsnaam}")
    return bestandsnaam


# ============================================================
# TARGET-VOORBEREIDING
# ============================================================

def _target_voor_mode(df: pd.DataFrame, target_column: str) -> pd.Series:
    if TARGET_MODE == "top25":
        return df.groupby("datum")[target_column].transform(
            lambda x: (x.rank(pct=True) >= 0.75).astype(int)
        )
    return (df[target_column] > 0).astype(int)


# ============================================================
# TRAINING PER HORIZON
# ============================================================

def train_voor_horizon(df: pd.DataFrame, horizon: str) -> Dict:
    target_column = f"fwd_ret_{horizon}"

    if target_column not in df.columns:
        return leeg_resultaat(horizon, "geen_doelkolom",
                              f"doelkolom '{target_column}' ontbreekt")

    df_h = df.copy()
    if int(df_h[target_column].notna().sum()) == 0:
        return leeg_resultaat(horizon, "onvoldoende_data",
                              f"geen ingevulde {target_column} labels")

    df_h["is_profitable"] = _target_voor_mode(df_h, target_column)

    # ---- Dynamische feature-selectie ----
    print()
    print(f"[{horizon}] Feature-selectie op basis van dekking "
          f"(min {MIN_FEATURE_DEKKING * 100:.0f}%)...")

    tech_bruikbaar, tech_uit = selecteer_features(
        df_h, FEATURE_TECHNISCH, target_column,
    )
    fund_bruikbaar, fund_uit = selecteer_features(
        df_h, FEATURE_FUNDAMENTEEL, target_column,
    )
    rel_bruikbaar, _ = selecteer_features(
        df_h, FEATURE_RELATIEF, target_column,
    )

    # Behoud de originele volgorde (handig voor feature_names_in_)
    tech_bruikbaar = [f for f in FEATURE_TECHNISCH if f in tech_bruikbaar]
    fund_bruikbaar = [f for f in FEATURE_FUNDAMENTEEL if f in fund_bruikbaar]
    rel_bruikbaar = [f for f in FEATURE_RELATIEF if f in rel_bruikbaar]
    alles_bruikbaar = tech_bruikbaar + fund_bruikbaar

    if tech_uit:
        print(f"[{horizon}] ⚠️  technische features uitgesloten: {tech_uit}")
    if fund_uit:
        print(f"[{horizon}] ⚠️  fundamentele features uitgesloten: {fund_uit}")

    print(f"[{horizon}] Bruikbare features: "
          f"tech={len(tech_bruikbaar)} fund={len(fund_bruikbaar)} "
          f"totaal={len(alles_bruikbaar)}")

    if len(tech_bruikbaar) < MIN_FEATURES_TECHNISCH:
        return leeg_resultaat(
            horizon, "geen_features",
            f"te weinig bruikbare technische features "
            f"({len(tech_bruikbaar)} < {MIN_FEATURES_TECHNISCH})"
        )

    # ---- Basissplit: dropna op alleen de BRUIKBARE tech features ----
    df_clean = df_h.dropna(subset=tech_bruikbaar + [target_column, "datum"])
    if len(df_clean) < MIN_RIJEN_TRAINING:
        return leeg_resultaat(horizon, "onvoldoende_data",
                              f"te weinig rijen na dropna ({len(df_clean)})")

    df_clean = df_clean.copy()
    df_clean["datum"] = pd.to_datetime(df_clean["datum"])
    df_clean = df_clean.sort_values("datum").reset_index(drop=True)

    unieke_datums = (
        df_clean["datum"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    if len(unieke_datums) < 2:
        return leeg_resultaat(horizon, "onvoldoende_datums",
                              "onvoldoende datums voor split")

    split_idx = min(max(int(len(unieke_datums) * 0.80), 1),
                    len(unieke_datums) - 1)
    split_datum = unieke_datums[split_idx]

    train_df = df_clean[df_clean["datum"] < split_datum].copy()
    test_df = df_clean[df_clean["datum"] >= split_datum].copy()

    if len(test_df) < MIN_RIJEN_TEST:
        return leeg_resultaat(horizon, "te_weinig_testdata",
                              f"te weinig testdata ({len(test_df)})")
    if len(train_df) < MIN_RIJEN_TRAINING:
        return leeg_resultaat(horizon, "te_weinig_trainingdata",
                              f"te weinig traindata ({len(train_df)})")
    if train_df["is_profitable"].nunique() < 2:
        return leeg_resultaat(horizon, "een_klasse",
                              "trainingsset bevat maar 1 klasse")

    n_top = max(1, int(len(test_df) * TOP_N_FRACTIE))

    print(f"[{horizon}] Start: train={len(train_df)}, test={len(test_df)}, "
          f"split={split_datum.date()}, target={TARGET_MODE}")

    # ---- Baseline ----
    top_baseline = top_n_gemiddelde(
        test_df, BASELINE_KOLOM, target_column, n_top, False,
    )
    auc_baseline = baseline_auc(test_df, target_column, BASELINE_KOLOM)
    test_gem = float(test_df[target_column].mean())

    # ---- Hoofdmodel (tech + fund, alleen bruikbare) ----
    try:
        model_main, acc, auc, top_main = train_en_evalueer(
            train_df, test_df, alles_bruikbaar, target_column, n_top,
        )
    except Exception as e:
        print(f"[{horizon}] hoofdmodel faalde: {e}")
        return leeg_resultaat(horizon, "train_fout", str(e))

    # ---- Controlemodel (relatief) ----
    auc_rel, top_rel = NAN, NAN
    if len(rel_bruikbaar) >= MIN_FEATURES_TECHNISCH:
        try:
            _, _, auc_rel, top_rel = train_en_evalueer(
                train_df, test_df, rel_bruikbaar, target_column, n_top,
            )
        except Exception:
            pass

    # ---- Technisch-only model ----
    auc_tech, top_tech = NAN, NAN
    try:
        _, _, auc_tech, top_tech = train_en_evalueer(
            train_df, test_df, tech_bruikbaar, target_column, n_top,
        )
    except Exception:
        pass

    # ---- Fundamenteel-only (alleen als genoeg bruikbare features) ----
    auc_fund, top_fund = NAN, NAN
    if len(fund_bruikbaar) >= MIN_FEATURES_FUNDAMENTEEL:
        try:
            train_f = train_df.dropna(subset=fund_bruikbaar)
            test_f = test_df.dropna(subset=fund_bruikbaar)
            if (len(train_f) >= MIN_RIJEN_TRAINING
                    and len(test_f) >= MIN_RIJEN_TEST):
                _, _, auc_fund, top_fund = train_en_evalueer(
                    train_f, test_f, fund_bruikbaar, target_column,
                    max(1, int(len(test_f) * TOP_N_FRACTIE)),
                )
        except Exception:
            pass

    # ---- Opslaan van het hoofdmodel ----
    try:
        sla_model_veilig_op(
            model_main, horizon, alles_bruikbaar, variant="tech_fund",
        )
    except Exception as e:
        print(f"[{horizon}] ❌ model niet gepubliceerd: {e}")
        return leeg_resultaat(horizon, "opslag_fout", str(e))

    print(
        f"[{horizon}] AUC tech+fund={_f(auc)} | "
        f"tech={_f(auc_tech)} | fund={_f(auc_fund)} | "
        f"relatief={_f(auc_rel)} | baseline={_f(auc_baseline)}"
    )
    print(
        f"[{horizon}] Top {n_top}: model={_pr(top_main)} | "
        f"baseline={_pr(top_baseline)} | test={_pr(test_gem)}"
    )

    return {
        "horizon": horizon,
        "status": "getraind",
        "opmerking": None,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "split_datum": str(split_datum.date()),
        "accuratesse": acc,
        "auc": auc,
        "auc_relatief": auc_rel,
        "auc_technisch": auc_tech,
        "auc_fundamenteel": auc_fund,
        "auc_baseline": auc_baseline,
        "top_n": int(n_top),
        "top_model": top_main,
        "top_baseline": top_baseline,
        "top_relatief": top_rel,
        "top_technisch": top_tech,
        "top_fundamenteel": top_fund,
        "test_gem": test_gem,
        "target_mode": TARGET_MODE,
    }


# ============================================================
# VORIGE RUNS + LOGGING
# ============================================================

def haal_vorige_runs(conn) -> Dict[str, Dict]:
    vorige = {}
    query = """
        SELECT auc, run_datum FROM xgboost_runs
        WHERE model_versie = %(mv)s AND horizon = %(h)s
          AND status = 'getraind' AND auc IS NOT NULL
        ORDER BY run_datum DESC LIMIT 1;
    """
    for h in HORIZONS:
        with conn.cursor() as cur:
            cur.execute(query, {"mv": "xgboostV3", "h": h})
            rij = cur.fetchone()
        if rij:
            vorige[h] = {
                "auc": float(rij[0]),
                "run_datum": rij[1].strftime("%Y-%m-%d"),
            }
    return vorige


def log_runs(conn, resultaten: List[Dict]) -> None:
    query = """
        INSERT INTO xgboost_runs (
            model_versie, horizon, status, opmerking,
            n_train, n_test, split_datum,
            accuratesse, auc, auc_relatief,
            top_n, top_model_rendement, top_baseline_rendement,
            top_relatief_rendement, test_gemiddelde
        ) VALUES (
            %(model_versie)s, %(horizon)s, %(status)s, %(opmerking)s,
            %(n_train)s, %(n_test)s, %(split_datum)s,
            %(accuratesse)s, %(auc)s, %(auc_relatief)s,
            %(top_n)s, %(top_model)s, %(top_baseline)s,
            %(top_relatief)s, %(test_gem)s
        );
    """
    with conn.cursor() as cur:
        for res in resultaten:
            params = {k: _db_waarde(v) for k, v in res.items()}
            params["model_versie"] = MODEL_VERSIE
            if res["status"] == "getraind":
                extra = (
                    f"AUC_tech={_f(res.get('auc_technisch'))} "
                    f"AUC_fund={_f(res.get('auc_fundamenteel'))} "
                    f"AUC_baseline={_f(res.get('auc_baseline'))} "
                    f"target={res.get('target_mode')}"
                )
                params["opmerking"] = extra
            cur.execute(query, params)
    conn.commit()


# ============================================================
# WAARSCHUWINGEN
# ============================================================

def waarschuwingen(res: Dict) -> List[str]:
    w = []
    if not _is_nan(res["top_model"]) and not _is_nan(res["top_baseline"]):
        if res["top_model"] <= res["top_baseline"]:
            w.append("model doet het niet beter dan baseline")
    if not _is_nan(res["auc"]) and res["auc"] < 0.55:
        w.append("AUC dicht bij 0.50: weinig voorspellende waarde")
    if not _is_nan(res.get("auc_technisch")) and not _is_nan(res.get("auc")):
        if res["auc"] < res["auc_technisch"] - 0.02:
            w.append("fundamentele features verslechteren het model")
    if not _is_nan(res.get("auc_fundamenteel")):
        if res["auc_fundamenteel"] > res["auc"] + 0.02:
            w.append("fundamenteel-only presteert beter — overweeg dat als hoofdmodel")
    return w


# ============================================================
# BERICHT
# ============================================================

def bouw_bericht(resultaten: List[Dict], vorige: Dict, datum: str) -> str:
    regels = [f"🤖 XGBoostV4 — hertraining ({datum})", ""]
    for res in resultaten:
        h = res["horizon"]
        if res["status"] != "getraind":
            regels.append(f"[{h}] ⏳ {res['opmerking']}")
            regels.append("")
            continue

        regels.append(
            f"[{h}] ✅ train {res['n_train']} | test {res['n_test']} "
            f"(vanaf {res['split_datum']})"
        )
        regels.append(
            f"  AUC tech+fund={_f(res['auc'])} | "
            f"tech={_f(res['auc_technisch'])} | "
            f"fund={_f(res['auc_fundamenteel'])} | "
            f"baseline={_f(res['auc_baseline'])}"
        )
        regels.append(
            f"  Top {res['top_n']}: model={_pr(res['top_model'])} | "
            f"baseline={_pr(res['top_baseline'])} | test={_pr(res['test_gem'])}"
        )
        for w in waarschuwingen(res):
            regels.append(f"  ⚠️ {w}")
        regels.append("")

    return "\n".join(regels)


# ============================================================
# MAIN
# ============================================================

def train_xgboost4() -> None:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt")

    datum = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"XGBoostV4 training — {datum} (target_mode={TARGET_MODE})")

    conn = psycopg2.connect(db_url)
    resultaten = []
    vorige = {}

    try:
        df = get_training_data(conn)
        print(f"{len(df)} gejoinde rijen opgehaald")

        toon_data_diagnose(df)

        # Feature-diagnose op de horizon met de meeste labels
        for h in HORIZONS:
            target = f"fwd_ret_{h}"
            if target in df.columns and df[target].notna().sum() > 0:
                diagnose_features(df, target)
                break

        for horizon in HORIZONS:
            try:
                resultaten.append(train_voor_horizon(df, horizon))
            except Exception as e:
                print(f"[{horizon}] onverwachte fout: {e}")
                resultaten.append(leeg_resultaat(horizon, "fout", str(e)))

        try:
            vorige = haal_vorige_runs(conn)
        except Exception as e:
            conn.rollback()
            print(f"[WARN] vorige runs niet op te halen: {e}")

        try:
            log_runs(conn, resultaten)
            print("Run gelogd in xgboost_runs")
        except Exception as e:
            conn.rollback()
            print(f"[WARN] loggen mislukt: {e}")

    finally:
        conn.close()

    n_getraind = sum(1 for r in resultaten if r["status"] == "getraind")
    bericht = bouw_bericht(resultaten, vorige, datum)
    send_telegram(bericht)
    send_email(f"XGBoostV4 hertraining {datum}", bericht)

    if n_getraind == 0:
        raise SystemExit("Geen enkele horizon kon getraind worden.")

    print(f"\n✅ XGBoostV4 training afgerond ({n_getraind} modellen)")


if __name__ == "__main__":
    train_xgboost4()
