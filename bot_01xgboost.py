#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01xgboost.py  —  XGBOOST RICHTINGSVOORSPELLING ENGINE v1.0

Voorspelt met XGBoost (gradient boosting op tabulaire features) de kans dat
een aandeel over HORIZON_DAYS handelsdagen meer dan LABEL_THRESHOLD_PCT stijgt,
op basis van dezelfde technische indicatoren die de regel-gebaseerde bots al
berekenen (RSI, MACD, ATR, afstand tot SMA50/SMA200, volume- en rendements-
momentum). Het verschil met bot_00kr/bot_01marktsent is niet de databron maar
de manier van combineren: in plaats van vaste drempels/AND-voorwaarden leert
het model zelf welke combinaties van features voorspellende waarde hebben.

BELANGRIJK — lees dit voor je op de live-signalen vertrouwt:
  Dit is een black-box-classificatiemodel, geen doorzichtige regel. Het is
  gevoelig voor overfitting op de trainingsperiode (andere marktregime =
  ander gedrag) en voor featurelekkage. Vertrouw NIET op de eigen accuracy
  van het model — vertrouw enkel op de walk-forward `backtest`-resultaten
  hieronder (out-of-sample, met dezelfde kosten/fiscaliteit als de andere
  bots), en vergelijk expliciet met de baseline (buy-and-hold universum) en
  met de tickers die de bestaande 7 strategieën al selecteren via de
  `selecties`-tabel, voor je dit als een 8e volwaardige strategie beschouwt.

Twee losse modellen-in-1-run-cycli:
  1. `train`    — bouwt het feature/label-dataset over het volledige
                  x-lijst-universum (041x-059x), traint een finaal XGBoost-
                  model op alle beschikbare geschiedenis (met de laatste
                  ~15% chronologisch als validatieset voor early stopping),
                  en slaat het model + metadata op (model_xgboost.json /
                  model_xgboost_meta.json). Dit is het model dat `live`
                  gebruikt. Bedoeld om periodiek (bv. wekelijks) te herhalen
                  zodat het model niet stil bevriest — zie run_01xgboost.yml.
  2. `backtest` — walk-forward validatie: her-traint het model elke
                  RESCAN_DAYS handelsdagen op een rollend IS_WINDOW-venster
                  en test enkel op de daaropvolgende (nog ongeziene) periode.
                  Simuleert een portefeuille met dezelfde kosten/fiscaliteit
                  als bot_01marktsent/bot_01cointegr (TRADE_COST_FIXED,
                  TRADE_COST_PCT, TAX_RATE) en rapporteert rendement, Sharpe,
                  max drawdown en win rate tegenover twee baselines (buy-and-
                  hold universum, willekeurige selectie) — zodat duidelijk
                  wordt of het model iets toevoegt boven ruis.
  3. `live`     — laadt het opgeslagen model, scant het x-lijst-universum,
                  rapporteert per beurs de top LIVE_CFG['top_n_per_beurs']
                  tickers met de hoogste voorspelde kans (Telegram + email),
                  en logt elke selectie naar de gedeelde Supabase
                  `selecties`-tabel onder strategie "bot_01xgboost" (vereist
                  de db_logger.py-whitelist-uitbreiding, zie
                  migratie_xgboost_kolommen.sql).

Gebruik:
  python bot_01xgboost.py train      # (her)traint het model, slaat het op
  python bot_01xgboost.py backtest   # walk-forward backtest (out-of-sample)
  python bot_01xgboost.py live       # live rapport met het opgeslagen model
"""

import os
import sys
import json
import math
import time
import warnings
import datetime as dt
import smtplib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf
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

START_CAPITAL        = 50_000.0
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
SLIPPAGE_PCT         = 0.001

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

BEURS_NAMEN = {
    "tickers_041x.txt": "041 Benelux Ierland",
    "tickers_042x.txt": "042 Parijs",
    "tickers_043x.txt": "043 Frankfurt",
    "tickers_044x.txt": "044 Spanje/Portugal",
    "tickers_045x.txt": "045 Londen",
    "tickers_046x.txt": "046 Milaan",
    "tickers_047x.txt": "047 Toronto",
    "tickers_048x.txt": "048 Nasdaq/NYSE",
    "tickers_049x.txt": "049 Stockholm",
    "tickers_050x.txt": "050 Zurich",
    "tickers_051x.txt": "051 Warschau",
    "tickers_052x.txt": "052 Oslo",
    "tickers_053x.txt": "053 Kopenhagen",
    "tickers_054x.txt": "054 Helsinki",
    "tickers_055x.txt": "055 CBoe",
    "tickers_056x.txt": "056 NYSE int",
    "tickers_057x.txt": "057 NYSE",
    "tickers_058x.txt": "058 TSXV",
    "tickers_059x.txt": "059 Oostenrijk Slovenie Slovakije",
}

def bouw_bestandslijst() -> List[str]:
    """Zelfde dynamische patroon als bot_00kr/bot_01marktsent: 041x t/m 059x,
    ontbrekende bestanden worden verderop overgeslagen. Enkel de x-lijsten
    (reeds gescreende kwaliteitsaandelen) i.p.v. de volledige a-lijsten —
    zowel voor snelheid als omdat het trainingsdataset anders enorm en
    ruizig wordt met dunverhandelde namen."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

XGB_CFG = {
    "rsi_period":          14,
    "macd_fast":           12,
    "macd_slow":           26,
    "macd_signal":         9,
    "atr_period":          14,
    "sma_short_period":    50,
    "sma_trend_period":    200,
    "vol_avg_period":      20,

    # Label: forward return over HORIZON_DAYS handelsdagen. Middenzone
    # (tussen -threshold en +threshold) wordt UIT de training gelaten —
    # standaardtechniek om het model niet te laten leren op ruis rond nul,
    # en threshold ligt ruim boven de ronde-trip-kosten (~0.7% + vaste kost)
    # zodat een "1"-label ook na kosten een winstkans betekent.
    "horizon_days":        10,
    "label_threshold_pct": 3.0,

    "n_estimators":        400,
    "max_depth":           4,
    "learning_rate":       0.05,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    5,
    "early_stopping_rounds": 30,

    # Walk-forward backtest
    "is_window":           504,   # ~2 handelsjaren in-sample trainingsvenster
    "rescan_days":         63,    # her-train elke ~1 kwartaal, test enkel daarna
    "min_train_rows":      500,   # te weinig trainingsrijen -> fold overslaan

    # Live selectie
    "min_proba":           0.62,
    "top_n_per_beurs":     5,
}

FEATURE_COLUMNS = [
    "rsi", "macd_hist", "macd_hist_slope",
    "dist_sma50_pct", "dist_sma200_pct",
    "atr_pct", "vol_ratio_20d",
    "ret_5d", "ret_20d", "ret_60d",
    "markt_ret_5d",  # cross-sectionele context, zie add_market_context()
]

MODEL_PATH      = "model_xgboost.json"
MODEL_META_PATH = "model_xgboost_meta.json"

BACKTEST_START = "2019-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES (identiek patroon aan bot_00kr.py / bot_01marktsent.py)
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
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

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# TECHNISCHE INDICATOREN (Wilder-smoothing, identiek aan bot_00kr/bot_01marktsent)
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
# DATA DOWNLOAD (identiek aan bot_01marktsent.py)
# ============================================================

def _normalise(df_raw, ticker: str) -> Optional[pd.DataFrame]:
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return None
    df = df_raw.copy().dropna(how="all")
    if df.empty:
        return None
    if df.index.name in ("Date", "Datetime") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Date" in df.columns:
        df = df.loc[:, ~df.columns.duplicated()]
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    df["Ticker"] = ticker
    return df

def download_history(tickers: List[str], period: str = "10y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    kwargs = dict(tickers=tickers, auto_adjust=True, group_by="ticker",
                  progress=False, threads=True, period=period)
    frames = []
    try:
        data = yf.download(**kwargs)
    except Exception as e:
        print(f"[WARN] Batch mislukt ({e}), probeer 1-voor-1...")
        data = pd.DataFrame()

    if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_level = 1
            for lvl in range(data.columns.nlevels):
                if any(t in set(data.columns.get_level_values(lvl)) for t in tickers):
                    ticker_level = lvl
                    break
            available = set(data.columns.get_level_values(ticker_level))
            for t in tickers:
                if t not in available:
                    continue
                try:
                    norm = _normalise(data.xs(t, axis=1, level=ticker_level).copy(), t)
                    if norm is not None:
                        frames.append(norm)
                except Exception as e:
                    print(f"[WARN] {t}: fout ({e})")
        else:
            norm = _normalise(data, tickers[0])
            if norm is not None:
                frames.append(norm)

    if not frames:
        for t in tickers:
            try:
                raw = yf.download(t, period=period, auto_adjust=True, progress=False)
                if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                    continue
                if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                norm = _normalise(raw, t)
                if norm is not None:
                    frames.append(norm)
                time.sleep(0.2)
            except Exception as e:
                print(f"[WARN] {t}: mislukt ({e})")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Verwacht een DataFrame voor ÉÉN ticker, met kolommen Date/Open/High/Low/
    Close/Volume, chronologisch gesorteerd. Geeft een DataFrame terug,
    geïndexeerd op Date, met de FEATURE_COLUMNS + 'label' + 'fwd_return_pct'
    + 'Close'. Alle features zijn causaal (enkel data tot en met de rij zelf)
    — enkel 'label'/'fwd_return_pct' kijken vooruit (horizon_days), en zijn
    dus NIET bruikbaar als feature, enkel als trainingsdoel.
    """
    df = df_ticker.set_index("Date")
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi = compute_rsi_series(close, XGB_CFG["rsi_period"])
    _, _, macd_hist = compute_macd_series(
        close, XGB_CFG["macd_fast"], XGB_CFG["macd_slow"], XGB_CFG["macd_signal"]
    )
    atr = compute_atr_series(high, low, close, XGB_CFG["atr_period"])
    sma_short = close.rolling(XGB_CFG["sma_short_period"]).mean()
    sma_trend = close.rolling(XGB_CFG["sma_trend_period"]).mean()
    vol_avg   = vol.rolling(XGB_CFG["vol_avg_period"]).mean()

    feat = pd.DataFrame(index=df.index)
    feat["rsi"]              = rsi
    feat["macd_hist"]        = macd_hist
    feat["macd_hist_slope"]  = macd_hist.diff(3)
    feat["dist_sma50_pct"]   = (close - sma_short) / sma_short * 100.0
    feat["dist_sma200_pct"]  = (close - sma_trend) / sma_trend * 100.0
    feat["atr_pct"]          = atr / close * 100.0
    feat["vol_ratio_20d"]    = vol / (vol_avg + 1e-9)
    feat["ret_5d"]           = close.pct_change(5) * 100.0
    feat["ret_20d"]          = close.pct_change(20) * 100.0
    feat["ret_60d"]          = close.pct_change(60) * 100.0

    horizon = XGB_CFG["horizon_days"]
    thr     = XGB_CFG["label_threshold_pct"]
    fwd_return_pct = (close.shift(-horizon) / close - 1.0) * 100.0

    label = pd.Series(np.nan, index=df.index)
    label[fwd_return_pct > thr]  = 1.0
    label[fwd_return_pct < -thr] = 0.0
    # middenzone (-thr <= return <= thr) blijft NaN -> uitgesloten van training

    feat["fwd_return_pct"] = fwd_return_pct
    feat["label"] = label
    feat["Close"] = close
    return feat

def add_market_context(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt 'markt_ret_5d' toe: het cross-sectionele gemiddelde 5-daagse
    rendement over alle tickers in de meegegeven dataset, per datum.

    Waarom: de per-ticker features (RSI/MACD/SMA-afstand/...) zeggen niets
    over het bredere marktregime op dat moment. Een RSI van 65 betekent iets
    anders tijdens een brede rally dan tijdens een correctie. Zonder deze
    context moet het model marktbewegingen impliciet raden uit enkel
    single-ticker-data -- wat het waarschijnlijk (mede) te zwak maakt
    (validatie-AUC ~0,52 zonder deze feature).

    Belangrijke beperking, bewust als eerste eenvoudige stap: de populatie
    waarover het gemiddelde berekend wordt is exact de dataset die wordt
    meegegeven aan deze functie. Bij train/backtest is dat het volledige
    universum (041-059 samen); bij live scan (run_live_engine) is dat enkel
    de tickers van de beurs die op dat moment gescand wordt -- dus daar is
    het al impliciet een beurs-niveau in plaats van universum-niveau
    context. Een striktere, bewust gekozen sector- of beurs-segmentatie
    (i.p.v. deze toevallige asymmetrie) is de logische vervolgstap.
    """
    if dataset.empty or "ret_5d" not in dataset.columns:
        dataset["markt_ret_5d"] = np.nan
        return dataset
    daggemiddelden = dataset.groupby("Date")["ret_5d"].transform("mean")
    dataset = dataset.copy()
    dataset["markt_ret_5d"] = daggemiddelden
    return dataset

def build_dataset(hist: pd.DataFrame) -> pd.DataFrame:
    """hist = output van download_history() over meerdere tickers.
    Geeft één lange DataFrame terug met kolommen Ticker/Date/FEATURE_COLUMNS/
    label/fwd_return_pct/Close, één rij per (ticker, datum)."""
    frames = []
    for ticker, g in hist.groupby("Ticker"):
        g = g.sort_values("Date")
        if len(g) < XGB_CFG["sma_trend_period"] + XGB_CFG["horizon_days"] + 20:
            continue  # te weinig geschiedenis voor stabiele features
        feat = build_features(g)
        feat = feat.reset_index()
        feat.insert(0, "Ticker", ticker)
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = add_market_context(out)
    return out


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
        # geen validatieset (bv. korte walk-forward fold) -> geen early stopping
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
# MODE: TRAIN — traint het finale model op alle geschiedenis
# ============================================================

def run_train():
    print(f"{'='*60}")
    print(f"XGBOOST TRAINING  {today_str()}")
    print(f"{'='*60}")

    universum = set()
    for f_name in bouw_bestandslijst():
        universum.update(load_tickers_from_file(f_name))
    universum = sorted(universum)
    if not universum:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return
    print(f"Universum: {len(universum)} tickers over de x-lijsten (041-059)")

    hist = download_history(universum, period="10y")
    if hist.empty:
        print("[ERROR] Geen historische data opgehaald.")
        return
    print(f"Historische data: {len(hist)} rijen over {hist['Ticker'].nunique()} tickers")

    dataset = build_dataset(hist)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"])
    if len(dataset) < XGB_CFG["min_train_rows"]:
        print(f"[ERROR] Te weinig trainingsrijen ({len(dataset)}), stop.")
        return

    dataset = dataset.sort_values("Date")
    print(f"Trainingsdataset: {len(dataset)} rijen (na uitsluiten NaN-features en neutrale labelzone)")
    print(f"Klassebalans: {dataset['label'].mean()*100:.1f}% positief (label=1)")

    # Laatste ~15% chronologisch als validatieset (geen shuffle: dat zou
    # look-ahead-lekkage geven doordat dicht-bij-elkaar-liggende dagen van
    # dezelfde ticker sterk gecorreleerd zijn).
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
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    proba: float
    pnl: float

def _simuleer_kosten_pnl(entry_price: float, exit_price: float, bedrag: float) -> float:
    """Bruto -> netto PnL voor één positie, incl. in-/uitstapkosten en belasting op winst."""
    aantal = bedrag / entry_price
    bruto_pnl = aantal * (exit_price - entry_price)
    kosten = trade_cost(bedrag) + trade_cost(aantal * exit_price)
    netto_voor_belasting = bruto_pnl - kosten
    if netto_voor_belasting > 0:
        netto_voor_belasting *= (1 - TAX_RATE)
    return netto_voor_belasting

def run_backtest():
    print(f"{'='*60}")
    print(f"XGBOOST WALK-FORWARD BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")
    print(
        "LET OP: dit test enkel of het model zelf (uit-eigen-verleden-geleerd,"
        "\nnooit getraind op de testperiode) betrouwbaar is — het test NIET of"
        "\nde gevonden tickers overlappen met wat cs/dm/db/kr/mr/ms/vcp al"
        "\nvinden. Vergelijk daarvoor achteraf tegen de Supabase `selecties`-tabel.\n"
    )

    universum = set()
    for f_name in bouw_bestandslijst():
        universum.update(load_tickers_from_file(f_name))
    universum = sorted(universum)
    if not universum:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    hist = download_history(universum, period="10y")
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

    kapitaal        = START_CAPITAL
    equity_curve     : List[Tuple[pd.Timestamp, float]] = [(alle_datums[start_idx], kapitaal)]
    alle_trades      : List[Trade] = []
    baseline_returns : List[float] = []   # gelijkgewogen "koop alles"-baseline, per rescan-periode
    random_returns   : List[float] = []   # willekeurige N-selectie-baseline, per rescan-periode
    fold_metrics     : List[Dict]  = []

    idx = start_idx
    model = None
    while idx < len(alle_datums) - horizon:
        train_start = alle_datums[max(0, idx - is_window)]
        train_end   = alle_datums[idx]

        train_fold = dataset[(dataset["Date"] >= train_start) & (dataset["Date"] < train_end)]
        train_fold = train_fold.dropna(subset=FEATURE_COLUMNS + ["label"])

        if len(train_fold) >= XGB_CFG["min_train_rows"]:
            split_i = int(len(train_fold) * 0.85)
            tf = train_fold.sort_values("Date")
            val_fold = tf.iloc[split_i:]
            fit_fold = tf.iloc[:split_i]
            model = fit_model(fit_fold, val_fold if len(val_fold) > 20 else None)
            fold_auc = None
            if len(val_fold) > 20 and val_fold["label"].nunique() == 2:
                proba_v = model.predict_proba(val_fold[FEATURE_COLUMNS])[:, 1]
                fold_auc = float(roc_auc_score(val_fold["label"], proba_v))
        # geen 'else': als er geen genoeg data is, hergebruiken we gewoon het vorige model
        # (of slaan de periode over als er nog geen enkel model getraind is)

        test_end_idx = min(idx + rescan_days, len(alle_datums) - horizon - 1)

        # ---- Instappen: ÉÉN keer per rescan-periode, niet elke dag erin ----
        # rescan_days (63) >> horizon_days (10), dus posities die hier geopend
        # worden zijn altijd al gesloten voor de volgende periode begint -- er
        # is dus geen overlap tussen periodes en kapitaal hoeft niet per open
        # positie apart gereserveerd te worden. (Eerdere versie opende elke
        # dag binnen de periode een nieuwe ronde posities op het dan-actuele
        # kapitaal, wat tot 60+ overlappende, deels ongereserveerde posities
        # tegelijk leidde en het kapitaal ongecontroleerd negatief liet gaan.)
        if model is not None and kapitaal > 0:
            periode_datum = alle_datums[idx]
            dagrijen = dataset[(dataset["Date"] == periode_datum)].dropna(subset=FEATURE_COLUMNS)
            if not dagrijen.empty:
                proba = model.predict_proba(dagrijen[FEATURE_COLUMNS])[:, 1]
                dagrijen = dagrijen.assign(proba=proba)
                kandidaten = dagrijen[dagrijen["proba"] >= XGB_CFG["min_proba"]].sort_values("proba", ascending=False)
                top = kandidaten.head(XGB_CFG["top_n_per_beurs"])
                if not top.empty:
                    # equal-weight over het volledige beschikbare kapitaal -- veilig
                    # omdat er geen overlappende posities van vorige periodes meer
                    # open staan op dit punt.
                    bedrag_per_positie = kapitaal / len(top)
                    for _, row in top.iterrows():
                        exit_rij = dataset[(dataset["Ticker"] == row["Ticker"]) &
                                            (dataset["Date"] > periode_datum)].sort_values("Date")
                        if len(exit_rij) < horizon:
                            continue
                        exit_row = exit_rij.iloc[horizon - 1]
                        pnl = _simuleer_kosten_pnl(row["Close"], exit_row["Close"], bedrag_per_positie)
                        kapitaal += pnl
                        alle_trades.append(Trade(
                            ticker=row["Ticker"], entry_date=periode_datum, exit_date=exit_row["Date"],
                            entry_price=row["Close"], exit_price=exit_row["Close"],
                            proba=float(row["proba"]), pnl=pnl,
                        ))

        # baselines over dezelfde periode, voor context
        periode_data = dataset[(dataset["Date"] >= alle_datums[idx]) & (dataset["Date"] < alle_datums[test_end_idx])]
        if not periode_data.empty:
            gem_fwd = periode_data["fwd_return_pct"].mean()
            if not math.isnan(gem_fwd):
                baseline_returns.append(gem_fwd)
            steekproef = periode_data.sample(min(XGB_CFG["top_n_per_beurs"], len(periode_data)), random_state=idx)
            rnd_fwd = steekproef["fwd_return_pct"].mean()
            if not math.isnan(rnd_fwd):
                random_returns.append(rnd_fwd)

        if model is not None and 'fold_auc' in dict(locals()) and fold_auc is not None:
            fold_metrics.append({"periode_start": str(alle_datums[idx].date()), "auc": fold_auc,
                                  "n_train": len(train_fold)})

        equity_curve.append((alle_datums[test_end_idx - 1] if test_end_idx > 0 else alle_datums[idx], kapitaal))

        if kapitaal <= 0:
            print(f"[STOP] Kapitaal uitgeput op {alle_datums[idx].date()} (EUR {kapitaal:,.2f}) — backtest gestopt.")
            break

        idx = test_end_idx if test_end_idx > idx else idx + rescan_days

    # ---------------------------------------------------------------
    # RAPPORTAGE
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 BACKTEST RESULTATEN — bot_01xgboost (walk-forward)")
    print(f"{'='*60}")

    if not alle_trades:
        print("Geen enkele trade uitgevoerd — min_proba/top_n mogelijk te streng, of te weinig data.")
        return

    totaal_rendement_pct = (kapitaal - START_CAPITAL) / START_CAPITAL * 100
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
    print(f"Eindkapitaal:         EUR {kapitaal:,.2f}  (start EUR {START_CAPITAL:,.2f})")
    print(f"Totaal rendement:     {totaal_rendement_pct:+.1f}%  (na kosten + belasting)")
    print(f"Sharpe (per periode): {sharpe:.2f}")
    print(f"Max drawdown:         {max_dd_pct:.1f}%")

    if baseline_returns:
        print(f"\nBaseline 'koop alles' gem. fwd-return/periode: {np.mean(baseline_returns):+.2f}%")
    if random_returns:
        print(f"Baseline willekeurige selectie gem. fwd-return/periode: {np.mean(random_returns):+.2f}%")
    if fold_metrics:
        gem_auc = np.mean([f["auc"] for f in fold_metrics])
        print(f"\nGemiddelde out-of-sample AUC over {len(fold_metrics)} walk-forward folds: {gem_auc:.3f}")
        print("(0.50 = geen voorspellende waarde, model is dan ruis — dit is de belangrijkste")
        print(" cijfer om te checken vóór je hier vertrouwen aan geeft)")

    print(
        "\nVolgende stap voor een eerlijk oordeel: vergelijk de tickers in de trades hierboven"
        "\nmet wat cs/dm/db/kr/mr/ms/vcp in dezelfde periode selecteerden (Supabase `selecties`),"
        "\nen kijk naar overlap + rendementscorrelatie — niet enkel naar dit standalone cijfer."
    )
    print(f"\n{'='*60}")


# ============================================================
# TELEGRAM + EMAIL OUTPUT — één bericht per beurs (identiek patroon)
# ============================================================

def _proba_bar(proba: float) -> str:
    n = round(proba * 6)
    return "█" * n + "░" * (6 - n) + f" {proba*100:.0f}%"

@dataclass
class LiveSignaal:
    ticker: str
    price: float
    proba: float
    rsi: float
    macd_hist: float
    dist_sma50_pct: float
    dist_sma200_pct: float
    atr_pct: float
    vol_ratio_20d: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    markt_ret_5d: float

def format_bericht(exchange_name: str, signalen: List[LiveSignaal], n_geanalyseerd: int) -> Optional[str]:
    if not signalen:
        return None
    nu = today_str()

    def sig_regel(s: LiveSignaal) -> str:
        return (
            f"• `{s.ticker}` {_proba_bar(s.proba)} | EUR{s.price:.2f} | "
            f"RSI:{s.rsi:.0f} | vs SMA200:{s.dist_sma200_pct:+.1f}% | {_yahoo_link(s.ticker)}"
        )

    delen = [
        f"🤖 *XGBOOST RICHTINGSVOORSPELLING — {exchange_name}*",
        f"_{nu} | {n_geanalyseerd} geanalyseerd | {len(signalen)} signalen (proba>={XGB_CFG['min_proba']:.0%})_",
        "─────────────────────────────",
        "\n\n".join(sig_regel(s) for s in signalen),
        "─────────────────────────────",
        f"⚙️ _model-kans op >{XGB_CFG['label_threshold_pct']:.0f}% stijging binnen {XGB_CFG['horizon_days']} handelsdagen — "
        f"zie backtest-resultaten voor de betrouwbaarheid hiervan, dit is geen doorzichtige regel_",
    ]
    return "\n\n".join(delen)


# ============================================================
# MODE: LIVE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"XGBOOST RICHTINGSVOORSPELLING — LIVE  {today_str()}")
    print(f"{'='*60}")

    model, meta = load_model()
    if model is None:
        print(
            f"[ERROR] Geen opgeslagen model gevonden ({MODEL_PATH}). "
            f"Voer eerst 'python bot_01xgboost.py train' uit (en commit het model-bestand)."
        )
        return
    print(f"Model geladen — getraind op {meta.get('trained_at')}, data tot {meta.get('trained_until')}, "
          f"validatie-AUC {meta.get('val_metrics', {}).get('auc', '?')}")

    exchange_tickers: Dict[str, List[str]] = {}
    for f_name in bouw_bestandslijst():
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name = label_voor(f_name)
        exchange_tickers[ex_name] = tlist
        print(f"  {ex_name}: {len(tlist)} tickers")

    if not exchange_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        hist = download_history(tlist, period="2y")  # ruim genoeg voor SMA200 + buffer
        if hist.empty:
            print("  → geen data, overslaan")
            continue

        signalen: List[LiveSignaal] = []
        n_geanalyseerd = 0
        ds = build_dataset(hist)  # incl. markt_ret_5d, cross-sectioneel over déze beurs
        if not ds.empty:
            # laatste beschikbare dag per ticker afzonderlijk (niet één gedeelde
            # datum voor de hele beurs) -- market-contextkolom is per rij al
            # correct berekend over de tickers die op precies díé datum in de
            # dataset zaten, dus dit blijft correct ook als niet elke ticker
            # exact dezelfde laatste handelsdag heeft.
            laatste_per_ticker = ds.sort_values("Date").groupby("Ticker").tail(1)
            laatste_per_ticker = laatste_per_ticker.dropna(subset=FEATURE_COLUMNS)
            n_geanalyseerd = len(laatste_per_ticker)
            if not laatste_per_ticker.empty:
                proba_arr = model.predict_proba(laatste_per_ticker[FEATURE_COLUMNS])[:, 1]
                laatste_per_ticker = laatste_per_ticker.assign(proba=proba_arr)
                for _, row in laatste_per_ticker.iterrows():
                    proba = float(row["proba"])
                    if proba >= XGB_CFG["min_proba"]:
                        signalen.append(LiveSignaal(
                            ticker=row["Ticker"], price=safe_float(row["Close"]), proba=proba,
                            rsi=safe_float(row["rsi"]), macd_hist=safe_float(row["macd_hist"]),
                            dist_sma50_pct=safe_float(row["dist_sma50_pct"]),
                            dist_sma200_pct=safe_float(row["dist_sma200_pct"]),
                            atr_pct=safe_float(row["atr_pct"]), vol_ratio_20d=safe_float(row["vol_ratio_20d"]),
                            ret_5d=safe_float(row["ret_5d"]), ret_20d=safe_float(row["ret_20d"]),
                            ret_60d=safe_float(row["ret_60d"]), markt_ret_5d=safe_float(row["markt_ret_5d"]),
                        ))

        signalen.sort(key=lambda s: s.proba, reverse=True)
        top = signalen[:XGB_CFG["top_n_per_beurs"]]
        print(f"  → {len(top)} van {len(signalen)} signalen (proba>={XGB_CFG['min_proba']:.0%}) "
              f"uit {n_geanalyseerd} geanalyseerd")

        for rank, s in enumerate(top, start=1):
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01xgboost",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "rank": rank,
                    "proba": round(s.proba, 4),
                    "rsi": round(s.rsi, 1) if not math.isnan(s.rsi) else None,
                    "macd_hist": round(s.macd_hist, 4) if not math.isnan(s.macd_hist) else None,
                    "dist_sma50_pct": round(s.dist_sma50_pct, 2) if not math.isnan(s.dist_sma50_pct) else None,
                    "dist_sma200_pct": round(s.dist_sma200_pct, 2) if not math.isnan(s.dist_sma200_pct) else None,
                    "atr_pct": round(s.atr_pct, 2) if not math.isnan(s.atr_pct) else None,
                    "vol_ratio_20d": round(s.vol_ratio_20d, 2) if not math.isnan(s.vol_ratio_20d) else None,
                    "ret_5d": round(s.ret_5d, 2) if not math.isnan(s.ret_5d) else None,
                    "ret_20d": round(s.ret_20d, 2) if not math.isnan(s.ret_20d) else None,
                    "ret_60d": round(s.ret_60d, 2) if not math.isnan(s.ret_60d) else None,
                    "markt_ret_5d": round(s.markt_ret_5d, 2) if not math.isnan(s.markt_ret_5d) else None,
                    "horizon_days": XGB_CFG["horizon_days"],
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, top, n_geanalyseerd)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")
        else:
            print(f"  → Overgeslagen: {ex_name} (geen signalen)")

    if email_delen:
        send_email(
            f"XGBoost Richtingsvoorspelling rapport {today_str()}",
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
