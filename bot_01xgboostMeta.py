#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01xgboostMeta.py  —  XGBOOST RICHTINGSVOORSPELLING OP FOREX (Twelve Data) v1.0

Zelfde onderliggende methode als bot_01xgboost.py (XGBoost-classificatie op
technische features), maar op een ANDER universum en via een ANDERE databron:
forex-paren (majors + kruisen) via de Twelve Data REST-API, in plaats van
aandelen via yfinance. Oorspronkelijk bedoeld als "XGBoost + MetaTrader 5",
maar MT5 vereist een lokaal draaiende terminal (niet mogelijk op GitHub
Actions cloud-runners) en er is geen broker/MT5-account beschikbaar — Twelve
Data geeft dezelfde soort dagkoersen (OHLC) zonder die afhankelijkheid, dus
dit draait volledig in de cloud zoals de andere bots.

BELANGRIJK — lees dit voor je op de live-signalen vertrouwt:
  Dit is opnieuw een black-box-classificatiemodel, dus dezelfde
  overfitting-/featurelekkage-risico's als bot_01xgboost.py, dat na
  uitgebreide walk-forward-validatie GEEN bewezen edge bleek te hebben
  (AUC ~0.51-0.53, niet beter dan buy-and-hold). Dit is een NIEUW
  experiment op een ander universum (forex i.p.v. aandelen) en een andere
  tijdshorizon-context (forex beweegt doorgaans veel minder per dag dan
  aandelen) — dat kan anders uitpakken, maar begin dus zelf ook hier
  wantrouwig: vertrouw enkel op de `backtest`-resultaten hieronder
  (walk-forward, out-of-sample), niet op de trainingsaccuracy.

  Kosten/belasting-kanttekening: er is (nog) geen broker/MT5-account, dus
  SPREAD_PCT hieronder is een RUWE SCHATTING (typische retail-spreads voor
  majors/kruisen), geen echte broker-data. TAX_RATE staat op 0 -- de
  fiscale behandeling van speculatieve forex-CFD-winst in België is geen
  eenvoudige meerwaardebelasting zoals bij aandelen en hangt af van de
  concrete situatie; dit script doet daar bewust geen aanname over. Zie dit
  dus als een eerste ruwe backtest, niet als een netto-rendementsbelofte.

Drie modi, zelfde patroon als bot_01xgboost.py:
  python bot_01xgboostMeta.py train      # (her)traint het model, slaat het op
  python bot_01xgboostMeta.py backtest   # walk-forward backtest (out-of-sample)
  python bot_01xgboostMeta.py live       # live rapport met het opgeslagen model

Vereist env var TWELVE_DATA_API_KEY (gratis tier: twelvedata.com/pricing).
"""

import os
import sys
import json
import math
import time
import warnings
import datetime as dt
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

try:
    from db_logger import log_selectie
except Exception as _e:
    print(f"[WARN] db_logger niet beschikbaar ({_e}) — DB-logging wordt overgeslagen")
    def log_selectie(*args, **kwargs):
        return False

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

FX_START_CAPITAL = 50_000.0

# Ruwe schatting, GEEN echte broker-spread (zie docstring hierboven).
# Majors zijn doorgaans krapper (~0.5-1.5 pips) dan kruisen (~1.5-3 pips).
SPREAD_PCT_MAJOR  = 0.0006   # ~0.06% round-trip, ruwe aanname voor majors
SPREAD_PCT_KRUIS  = 0.0015   # ~0.15% round-trip, ruwe aanname voor kruispaaren
TAX_RATE          = 0.0      # bewust op 0 gelaten, zie docstring

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
TWELVE_DATA_URL     = "https://api.twelvedata.com/time_series"

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

# Forex-universum: majors (bevatten USD) + belangrijkste kruispaaren.
# Twelve Data-notatie: "EUR/USD" (met slash).
FOREX_GROEPEN: Dict[str, List[str]] = {
    "Majors": [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
        "USD/CAD", "AUD/USD", "NZD/USD",
    ],
    "Kruispaaren": [
        "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD",
        "GBP/JPY", "GBP/CHF", "GBP/AUD",
        "AUD/JPY", "AUD/NZD", "CAD/JPY", "CHF/JPY", "NZD/JPY",
    ],
}

def alle_paren() -> List[str]:
    out = []
    for lijst in FOREX_GROEPEN.values():
        out.extend(lijst)
    return out

def groep_van(paar: str) -> str:
    for naam, lijst in FOREX_GROEPEN.items():
        if paar in lijst:
            return naam
    return "Overig"

def spread_pct_van(paar: str) -> float:
    return SPREAD_PCT_MAJOR if groep_van(paar) == "Majors" else SPREAD_PCT_KRUIS

XGB_CFG = {
    "rsi_period":          14,
    "macd_fast":           12,
    "macd_slow":           26,
    "macd_signal":         9,
    "atr_period":          14,
    "sma_short_period":    50,
    "sma_trend_period":    200,

    # Forex beweegt per dag doorgaans veel minder dan aandelen -- drempel
    # en horizon staan daarom lager dan in bot_01xgboost.py (3.0% / 10d).
    # Middenzone (tussen -threshold en +threshold) blijft uit de training,
    # net als bij bot_01xgboost.py.
    "horizon_days":        10,
    "label_threshold_pct": 1.0,

    "n_estimators":        400,
    "max_depth":           4,
    "learning_rate":       0.05,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    5,
    "early_stopping_rounds": 30,

    # Walk-forward backtest
    "is_window":           504,
    "rescan_days":         63,
    "min_train_rows":      500,

    # Live selectie
    "min_proba":           0.62,
    "top_n_per_groep":     5,
}

# Geen vol_ratio_20d hier: forex-volume via retail-feeds (ook Twelve Data)
# is tick-volume van één bron, geen echt verhandeld volume zoals bij
# aandelen -- te onbetrouwbaar om als feature te gebruiken.
FEATURE_COLUMNS = [
    "rsi", "macd_hist", "macd_hist_slope",
    "dist_sma50_pct", "dist_sma200_pct",
    "atr_pct",
    "ret_5d", "ret_20d", "ret_60d",
]

MODEL_PATH      = "model_xgboostmeta.json"
MODEL_META_PATH = "model_xgboostmeta_meta.json"

BACKTEST_START = "2019-01-01"
BACKTEST_END   = dt.date.today().isoformat()

# Twelve Data gratis tier: ~8 requests/minuut. 1 seconde extra marge per call.
TWELVE_DATA_SLEEP_SEC = 8.0


# ============================================================
# HULPFUNCTIES
# ============================================================

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.status_code == 429:
            wacht = resp.json().get("parameters", {}).get("retry_after", 5)
            time.sleep(wacht + 1)
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"Telegram fout: {e}")

def send_email(subject: str, body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        clean = body.replace("*", "").replace("`", "").replace("•", "-").replace("_", "")
        msg.attach(MIMEText(clean, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email fout: {e}")


# ============================================================
# TECHNISCHE INDICATOREN (identiek aan bot_01xgboost.py)
# ============================================================

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period
            + valid.iloc[i] / period
        )
    return result

def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = _wilder_smooth(gain, period) / (_wilder_smooth(loss, period) + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macd_series(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


# ============================================================
# DATA DOWNLOAD — Twelve Data time_series (D1)
# ============================================================

# GLOBALE THROTTLE — module-niveau, geldt over ALLE aanroepen in deze
# procesrun heen (dus ook over de grens tussen bv. Majors en Kruispaaren),
# in tegenstelling tot de vorige versie die enkel pauzeerde TUSSEN paren
# binnen één download_history_alle_paren()-aanroep en zo de rate limit
# net op de groepsgrens liet doorbreken.
_laatste_call_tijd: Optional[float] = None

def _throttle_twelve_data() -> None:
    global _laatste_call_tijd
    nu = time.monotonic()
    if _laatste_call_tijd is not None:
        verstreken = nu - _laatste_call_tijd
        if verstreken < TWELVE_DATA_SLEEP_SEC:
            time.sleep(TWELVE_DATA_SLEEP_SEC - verstreken)
    _laatste_call_tijd = time.monotonic()

def download_forex_history(paar: str, outputsize: int = 5000, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Haalt dagelijkse OHLC-candles op voor één forex-paar via Twelve Data.
    Geeft None terug bij een blijvende fout (bv. onbekend paar) -- de
    aanroeper slaat dat paar dan gewoon over. Bij een rate-limit-fout wordt
    (max_retries keer) een volle minuut gewacht en opnieuw geprobeerd, in
    plaats van het paar meteen stilzwijgend op te geven."""
    if not TWELVE_DATA_API_KEY:
        print("[ERROR] TWELVE_DATA_API_KEY ontbreekt.")
        return None

    params = {
        "symbol": paar,
        "interval": "1day",
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
    }

    for poging in range(max_retries + 1):
        _throttle_twelve_data()
        try:
            resp = requests.get(TWELVE_DATA_URL, params=params, timeout=20)
            data = resp.json()
        except Exception as e:
            print(f"[WARN] {paar}: request mislukt ({e})")
            return None

        if isinstance(data, dict) and data.get("status") == "error":
            bericht = str(data.get("message", ""))
            is_rate_limit = "api credits" in bericht.lower() or "run out" in bericht.lower()
            if is_rate_limit and poging < max_retries:
                print(f"[WARN] {paar}: rate limit bereikt, 65s wachten en opnieuw proberen "
                      f"(poging {poging + 1}/{max_retries})...")
                time.sleep(65)
                continue
            print(f"[WARN] {paar}: Twelve Data fout — {bericht}")
            return None

        values = data.get("values") if isinstance(data, dict) else None
        if not values:
            print(f"[WARN] {paar}: geen data ontvangen")
            return None

        df = pd.DataFrame(values)
        kolommen_nodig = {"datetime", "open", "high", "low", "close"}
        if not kolommen_nodig.issubset(df.columns):
            print(f"[WARN] {paar}: onverwacht antwoordformaat")
            return None

        df["Date"]  = pd.to_datetime(df["datetime"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df["Paar"] = paar
        df = df[["Date", "Open", "High", "Low", "Close", "Paar"]].dropna()
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    return None

def download_history_alle_paren(paren: List[str], outputsize: int = 5000) -> pd.DataFrame:
    """Haalt alle paren na elkaar op. De throttle zit nu IN
    download_forex_history zelf (globaal, module-niveau), dus dit werkt ook
    correct als deze functie meerdere keren na elkaar wordt aangeroepen
    (bv. één keer per beurs-groep, zoals in run_live_engine)."""
    frames = []
    for paar in paren:
        df = download_forex_history(paar, outputsize=outputsize)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# FEATURE ENGINEERING (zelfde opzet als bot_01xgboost.py, zonder volume)
# ============================================================

def build_features(df_paar: pd.DataFrame) -> pd.DataFrame:
    df = df_paar.set_index("Date")
    close, high, low = df["Close"], df["High"], df["Low"]

    rsi = compute_rsi_series(close, XGB_CFG["rsi_period"])
    _, _, macd_hist = compute_macd_series(
        close, XGB_CFG["macd_fast"], XGB_CFG["macd_slow"], XGB_CFG["macd_signal"]
    )
    atr = compute_atr_series(high, low, close, XGB_CFG["atr_period"])
    sma_short = close.rolling(XGB_CFG["sma_short_period"]).mean()
    sma_trend = close.rolling(XGB_CFG["sma_trend_period"]).mean()

    feat = pd.DataFrame(index=df.index)
    feat["rsi"]              = rsi
    feat["macd_hist"]        = macd_hist
    feat["macd_hist_slope"]  = macd_hist.diff(3)
    feat["dist_sma50_pct"]   = (close - sma_short) / sma_short * 100.0
    feat["dist_sma200_pct"]  = (close - sma_trend) / sma_trend * 100.0
    feat["atr_pct"]          = atr / close * 100.0
    feat["ret_5d"]           = close.pct_change(5) * 100.0
    feat["ret_20d"]          = close.pct_change(20) * 100.0
    feat["ret_60d"]          = close.pct_change(60) * 100.0

    horizon = XGB_CFG["horizon_days"]
    thr     = XGB_CFG["label_threshold_pct"]
    fwd_return_pct = (close.shift(-horizon) / close - 1.0) * 100.0

    label = pd.Series(np.nan, index=df.index)
    label[fwd_return_pct > thr]  = 1.0
    label[fwd_return_pct < -thr] = 0.0

    feat["fwd_return_pct"] = fwd_return_pct
    feat["label"] = label
    feat["Close"] = close
    return feat

def build_dataset(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for paar, g in hist.groupby("Paar"):
        g = g.sort_values("Date")
        if len(g) < XGB_CFG["sma_trend_period"] + XGB_CFG["horizon_days"] + 20:
            continue
        feat = build_features(g)
        feat = feat.reset_index()
        feat.insert(0, "Paar", paar)
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# MODEL TRAIN / LADEN / OPSLAAN
# ============================================================

def _make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=XGB_CFG["n_estimators"],
        max_depth=XGB_CFG["max_depth"],
        learning_rate=XGB_CFG["learning_rate"],
        subsample=XGB_CFG["subsample"],
        colsample_bytree=XGB_CFG["colsample_bytree"],
        min_child_weight=XGB_CFG["min_child_weight"],
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=XGB_CFG["early_stopping_rounds"],
        n_jobs=-1,
    )

def fit_model(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> xgb.XGBClassifier:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    model = _make_model()
    if val_df is not None and len(val_df) > 0:
        model.fit(
            X_train, y_train,
            eval_set=[(val_df[FEATURE_COLUMNS], val_df["label"])],
            verbose=False,
        )
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(X_train, y_train, verbose=False)
    return model

def save_model(model: xgb.XGBClassifier, trained_until: str, n_rows: int, val_metrics: Dict) -> None:
    model.save_model(MODEL_PATH)
    meta = {
        "trained_at": today_str(),
        "trained_until": trained_until,
        "n_rows": n_rows,
        "feature_columns": FEATURE_COLUMNS,
        "config": XGB_CFG,
        "val_metrics": val_metrics,
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Model opgeslagen: {MODEL_PATH} + {MODEL_META_PATH}")

def load_model() -> Tuple[Optional[xgb.XGBClassifier], Optional[Dict]]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_META_PATH):
        return None, None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ============================================================
# MODE: TRAIN
# ============================================================

def run_train():
    print(f"{'='*60}")
    print(f"XGBOOSTMETA (FOREX) TRAINING  {today_str()}")
    print(f"{'='*60}")

    paren = alle_paren()
    print(f"Universum: {len(paren)} forex-paren ({', '.join(paren)})")

    hist = download_history_alle_paren(paren, outputsize=5000)
    if hist.empty:
        print("[ERROR] Geen historische data opgehaald van Twelve Data.")
        return
    print(f"Historische data: {len(hist)} rijen over {hist['Paar'].nunique()} paren")

    dataset = build_dataset(hist)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"])
    if len(dataset) < XGB_CFG["min_train_rows"]:
        print(f"[ERROR] Te weinig trainingsrijen ({len(dataset)}), stop.")
        return

    dataset = dataset.sort_values("Date")
    print(f"Trainingsdataset: {len(dataset)} rijen")
    print(f"Klassebalans: {dataset['label'].mean()*100:.1f}% positief (label=1)")

    split_idx = int(len(dataset) * 0.85)
    split_date = dataset.iloc[split_idx]["Date"]
    train_df = dataset[dataset["Date"] < split_date]
    val_df   = dataset[dataset["Date"] >= split_date]
    print(f"Train: {len(train_df)} rijen tot {split_date.date()} | Validatie: {len(val_df)} rijen erna")

    model = fit_model(train_df, val_df)

    val_proba = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    val_pred  = (val_proba >= XGB_CFG["min_proba"]).astype(int)
    metrics = {
        "auc": float(roc_auc_score(val_df["label"], val_proba)),
        "accuracy_at_threshold": float(accuracy_score(val_df["label"], val_pred)) if val_pred.sum() > 0 else None,
        "precision_at_threshold": float(precision_score(val_df["label"], val_pred, zero_division=0)),
        "n_signals_at_threshold": int(val_pred.sum()),
        "n_val_rows": len(val_df),
    }
    print(f"\nValidatie (out-of-sample, {split_date.date()} -> nu):")
    print(f"  AUC:                    {metrics['auc']:.3f}  (0.5 = toeval, 1.0 = perfect)")
    print(f"  Precisie bij proba>={XGB_CFG['min_proba']}: {metrics['precision_at_threshold']:.3f}"
          f"  over {metrics['n_signals_at_threshold']} signalen")

    save_model(model, trained_until=str(dataset["Date"].max().date()), n_rows=len(dataset), val_metrics=metrics)
    print(f"\n{'='*60}\nKlaar. Gebruik 'backtest' voor een volledige walk-forward-validatie.")


# ============================================================
# MODE: BACKTEST — walk-forward, out-of-sample
# ============================================================

@dataclass
class Trade:
    paar: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    proba: float
    pnl: float

def _simuleer_kosten_pnl(paar: str, entry_price: float, exit_price: float, bedrag: float) -> float:
    """Bruto -> netto PnL, kosten hier enkel als spread (geen vaste/percentage
    broker-commissie zoals bij aandelen -- forex-CFD's werken doorgaans via
    spread). Zie docstring bovenaan: SPREAD_PCT is een ruwe schatting."""
    aantal = bedrag / entry_price
    bruto_pnl = aantal * (exit_price - entry_price)
    kosten = bedrag * spread_pct_van(paar)
    netto_voor_belasting = bruto_pnl - kosten
    if netto_voor_belasting > 0:
        netto_voor_belasting *= (1 - TAX_RATE)
    return netto_voor_belasting

def run_backtest():
    print(f"{'='*60}")
    print(f"XGBOOSTMETA (FOREX) WALK-FORWARD BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")
    print(
        "LET OP: kosten hier zijn een ruwe spread-schatting, geen echte"
        "\nbroker-data (er is nog geen account) -- behandel het resultaat als"
        "\neen eerste indicatie, niet als een gegarandeerd netto-rendement.\n"
    )

    paren = alle_paren()
    hist = download_history_alle_paren(paren, outputsize=5000)
    if hist.empty:
        print("[ERROR] Geen historische data opgehaald.")
        return

    dataset = build_dataset(hist)
    dataset = dataset[dataset["Date"] >= pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=XGB_CFG["is_window"] * 2)]
    if dataset.empty:
        print("[ERROR] Onvoldoende data voor backtestvenster.")
        return

    alle_datums = sorted(dataset["Date"].unique())
    is_window   = XGB_CFG["is_window"]
    rescan_days = XGB_CFG["rescan_days"]
    horizon     = XGB_CFG["horizon_days"]

    start_idx = next((i for i, d in enumerate(alle_datums) if d >= pd.Timestamp(BACKTEST_START)), None)
    if start_idx is None or start_idx < is_window:
        start_idx = is_window
    if start_idx >= len(alle_datums):
        print("[ERROR] Backtestperiode te kort t.o.v. is_window.")
        return

    kapitaal        = FX_START_CAPITAL
    equity_curve     : List[Tuple[pd.Timestamp, float]] = [(alle_datums[start_idx], kapitaal)]
    alle_trades      : List[Trade] = []
    baseline_returns : List[float] = []
    fold_metrics     : List[Dict]  = []

    idx = start_idx
    model = None
    while idx < len(alle_datums) - horizon:
        train_start = alle_datums[max(0, idx - is_window)]
        train_end   = alle_datums[idx]

        train_fold = dataset[(dataset["Date"] >= train_start) & (dataset["Date"] < train_end)]
        train_fold = train_fold.dropna(subset=FEATURE_COLUMNS + ["label"])

        fold_auc = None
        if len(train_fold) >= XGB_CFG["min_train_rows"]:
            split_i = int(len(train_fold) * 0.85)
            tf = train_fold.sort_values("Date")
            val_fold = tf.iloc[split_i:]
            fit_fold = tf.iloc[:split_i]
            model = fit_model(fit_fold, val_fold if len(val_fold) > 20 else None)
            if len(val_fold) > 20 and val_fold["label"].nunique() == 2:
                proba_v = model.predict_proba(val_fold[FEATURE_COLUMNS])[:, 1]
                fold_auc = float(roc_auc_score(val_fold["label"], proba_v))

        test_end_idx = min(idx + rescan_days, len(alle_datums) - horizon - 1)

        if model is not None and kapitaal > 0:
            periode_datum = alle_datums[idx]
            dagrijen = dataset[(dataset["Date"] == periode_datum)].dropna(subset=FEATURE_COLUMNS)
            if not dagrijen.empty:
                proba = model.predict_proba(dagrijen[FEATURE_COLUMNS])[:, 1]
                dagrijen = dagrijen.assign(proba=proba)
                kandidaten = dagrijen[dagrijen["proba"] >= XGB_CFG["min_proba"]].sort_values("proba", ascending=False)
                top = kandidaten.head(XGB_CFG["top_n_per_groep"] * len(FOREX_GROEPEN))
                if not top.empty:
                    bedrag_per_positie = kapitaal / len(top)
                    for _, row in top.iterrows():
                        exit_rij = dataset[(dataset["Paar"] == row["Paar"]) &
                                            (dataset["Date"] > periode_datum)].sort_values("Date")
                        if len(exit_rij) < horizon:
                            continue
                        exit_row = exit_rij.iloc[horizon - 1]
                        pnl = _simuleer_kosten_pnl(row["Paar"], row["Close"], exit_row["Close"], bedrag_per_positie)
                        kapitaal += pnl
                        alle_trades.append(Trade(
                            paar=row["Paar"], entry_date=periode_datum, exit_date=exit_row["Date"],
                            entry_price=row["Close"], exit_price=exit_row["Close"],
                            proba=float(row["proba"]), pnl=pnl,
                        ))

        periode_data = dataset[(dataset["Date"] >= alle_datums[idx]) & (dataset["Date"] < alle_datums[test_end_idx])]
        if not periode_data.empty:
            gem_fwd = periode_data["fwd_return_pct"].mean()
            if not math.isnan(gem_fwd):
                baseline_returns.append(gem_fwd)

        if fold_auc is not None:
            fold_metrics.append({"periode_start": str(alle_datums[idx].date()), "auc": fold_auc,
                                  "n_train": len(train_fold)})

        equity_curve.append((alle_datums[test_end_idx - 1] if test_end_idx > 0 else alle_datums[idx], kapitaal))

        if kapitaal <= 0:
            print(f"[STOP] Kapitaal uitgeput op {alle_datums[idx].date()} (EUR {kapitaal:,.2f}) — backtest gestopt.")
            break

        idx = test_end_idx if test_end_idx > idx else idx + rescan_days

    print(f"\n{'='*60}")
    print("📊 BACKTEST RESULTATEN — bot_01xgboostMeta (walk-forward, forex)")
    print(f"{'='*60}")

    if not alle_trades:
        print("Geen enkele trade uitgevoerd — min_proba mogelijk te streng, of te weinig data.")
        return

    totaal_rendement_pct = (kapitaal - FX_START_CAPITAL) / FX_START_CAPITAL * 100
    winnaars = [t for t in alle_trades if t.pnl > 0]
    win_rate = len(winnaars) / len(alle_trades) * 100

    eq_series = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
    eq_returns = eq_series.pct_change().dropna()
    sharpe = 0.0
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        periodes_per_jaar = 252 / rescan_days
        sharpe = (eq_returns.mean() / eq_returns.std()) * math.sqrt(periodes_per_jaar)
    cum_max = eq_series.cummax()
    drawdown = (eq_series - cum_max) / cum_max
    max_dd_pct = drawdown.min() * 100

    print(f"Periode:              {alle_datums[start_idx].date()} -> {alle_datums[-1].date()}")
    print(f"Aantal trades:        {len(alle_trades)}")
    print(f"Win rate:             {win_rate:.1f}%")
    print(f"Eindkapitaal:         EUR {kapitaal:,.2f}  (start EUR {FX_START_CAPITAL:,.2f})")
    print(f"Totaal rendement:     {totaal_rendement_pct:+.1f}%  (na geschatte spread-kosten, vóór belasting)")
    print(f"Sharpe (per periode): {sharpe:.2f}")
    print(f"Max drawdown:         {max_dd_pct:.1f}%")

    if baseline_returns:
        print(f"\nBaseline 'alle paren' gem. fwd-return/periode: {np.mean(baseline_returns):+.2f}%")
    if fold_metrics:
        gem_auc = np.mean([f["auc"] for f in fold_metrics])
        print(f"\nGemiddelde out-of-sample AUC over {len(fold_metrics)} walk-forward folds: {gem_auc:.3f}")
        print("(0.50 = geen voorspellende waarde, model is dan ruis — dit is de belangrijkste")
        print(" cijfer om te checken vóór je hier vertrouwen aan geeft)")

    print(f"\n{'='*60}")


# ============================================================
# TELEGRAM + EMAIL OUTPUT
# ============================================================

def _proba_bar(proba: float) -> str:
    n = round(proba * 6)
    return "█" * n + "░" * (6 - n) + f" {proba*100:.0f}%"

@dataclass
class LiveSignaal:
    paar: str
    price: float
    proba: float
    rsi: float
    macd_hist: float
    dist_sma50_pct: float
    dist_sma200_pct: float
    atr_pct: float
    ret_5d: float
    ret_20d: float
    ret_60d: float

def format_bericht(groep_naam: str, signalen: List[LiveSignaal], n_geanalyseerd: int) -> Optional[str]:
    if not signalen:
        return None
    nu = today_str()

    def sig_regel(s: LiveSignaal) -> str:
        return (
            f"• `{s.paar}` {_proba_bar(s.proba)} | {s.price:.5f} | "
            f"RSI:{s.rsi:.0f} | vs SMA200:{s.dist_sma200_pct:+.1f}%"
        )

    delen = [
        f"🤖 *XGBOOST FOREX RICHTINGSVOORSPELLING — {groep_naam}*",
        f"_{nu} | {n_geanalyseerd} geanalyseerd | {len(signalen)} signalen (proba>={XGB_CFG['min_proba']:.0%})_",
        "─────────────────────────────",
        "\n\n".join(sig_regel(s) for s in signalen),
        "─────────────────────────────",
        f"⚙️ _model-kans op >{XGB_CFG['label_threshold_pct']:.1f}% beweging binnen {XGB_CFG['horizon_days']} handelsdagen — "
        f"experimenteel, geen bewezen edge, kosten/belasting zijn ruwe schattingen (geen broker-account)_",
    ]
    return "\n\n".join(delen)


# ============================================================
# MODE: LIVE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"XGBOOSTMETA (FOREX) — LIVE  {today_str()}")
    print(f"{'='*60}")

    model, meta = load_model()
    if model is None:
        print(
            f"[ERROR] Geen opgeslagen model gevonden ({MODEL_PATH}). "
            f"Voer eerst 'python bot_01xgboostMeta.py train' uit (en commit het model-bestand)."
        )
        return
    print(f"Model geladen — getraind op {meta.get('trained_at')}, data tot {meta.get('trained_until')}, "
          f"validatie-AUC {meta.get('val_metrics', {}).get('auc', '?')}")

    email_delen: List[str] = []

    for groep_naam, paren in FOREX_GROEPEN.items():
        print(f"\nAnalyseren: {groep_naam} ({len(paren)} paren)...")
        hist = download_history_alle_paren(paren, outputsize=800)  # ruim genoeg voor SMA200 + buffer
        if hist.empty:
            print("  → geen data, overslaan")
            continue

        signalen: List[LiveSignaal] = []
        n_geanalyseerd = 0
        ds = build_dataset(hist)
        if not ds.empty:
            laatste_per_paar = ds.sort_values("Date").groupby("Paar").tail(1)
            laatste_per_paar = laatste_per_paar.dropna(subset=FEATURE_COLUMNS)
            n_geanalyseerd = len(laatste_per_paar)
            if not laatste_per_paar.empty:
                proba_arr = model.predict_proba(laatste_per_paar[FEATURE_COLUMNS])[:, 1]
                laatste_per_paar = laatste_per_paar.assign(proba=proba_arr)
                for _, row in laatste_per_paar.iterrows():
                    proba = float(row["proba"])
                    if proba >= XGB_CFG["min_proba"]:
                        signalen.append(LiveSignaal(
                            paar=row["Paar"], price=safe_float(row["Close"]), proba=proba,
                            rsi=safe_float(row["rsi"]), macd_hist=safe_float(row["macd_hist"]),
                            dist_sma50_pct=safe_float(row["dist_sma50_pct"]),
                            dist_sma200_pct=safe_float(row["dist_sma200_pct"]),
                            atr_pct=safe_float(row["atr_pct"]),
                            ret_5d=safe_float(row["ret_5d"]), ret_20d=safe_float(row["ret_20d"]),
                            ret_60d=safe_float(row["ret_60d"]),
                        ))

        signalen.sort(key=lambda s: s.proba, reverse=True)
        top = signalen[:XGB_CFG["top_n_per_groep"]]
        print(f"  → {len(top)} van {len(signalen)} signalen (proba>={XGB_CFG['min_proba']:.0%}) "
              f"uit {n_geanalyseerd} geanalyseerd")

        for rank, s in enumerate(top, start=1):
            log_selectie(
                ticker=s.paar,
                datum=today_str(),
                strategie="bot_01xgboostMeta",
                beurs=groep_naam,
                koers=s.price,
                parameters={
                    "rank": rank,
                    "proba": round(s.proba, 4),
                    "rsi": round(s.rsi, 1) if not math.isnan(s.rsi) else None,
                    "macd_hist": round(s.macd_hist, 5) if not math.isnan(s.macd_hist) else None,
                    "dist_sma50_pct": round(s.dist_sma50_pct, 2) if not math.isnan(s.dist_sma50_pct) else None,
                    "dist_sma200_pct": round(s.dist_sma200_pct, 2) if not math.isnan(s.dist_sma200_pct) else None,
                    "atr_pct": round(s.atr_pct, 2) if not math.isnan(s.atr_pct) else None,
                    "ret_5d": round(s.ret_5d, 2) if not math.isnan(s.ret_5d) else None,
                    "ret_20d": round(s.ret_20d, 2) if not math.isnan(s.ret_20d) else None,
                    "ret_60d": round(s.ret_60d, 2) if not math.isnan(s.ret_60d) else None,
                    "horizon_days": XGB_CFG["horizon_days"],
                },
            )

        bericht = format_bericht(groep_naam, top, n_geanalyseerd)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")
        else:
            print(f"  → Overgeslagen: {groep_naam} (geen signalen)")

    if email_delen:
        send_email(
            f"XGBoost Forex Richtingsvoorspelling rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "train":
        run_train()
    elif mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
