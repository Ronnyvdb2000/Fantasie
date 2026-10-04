#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta_model.py  —  META-MODEL OP PARAMETER-FEATURES  v0.1

Bouwt een meta-model dat per (datum, aandeel) een score geeft op basis
van de features uit features.MODEL_FEATURES. Doel: het model leert welke
combinatie van parameterwaarden samenhangt met een goede fwd_ret_20d.

METHODE
=======
- Data: selecties + forward_returns + generieke_technicals.
- Feature engineering: per datum cross-sectionele ranks (0..1) zodat het
  model cross-sectioneel leert en niet het marktregime oppikt.
- Model: Ridge regression. Bewust simpel — LightGBM komt pas als Ridge
  al signaal laat zien.
- Validatie: expanding-window walk-forward over unieke datums. Train op
  alle datums tot T-1, test op datum T. Geen random split, geen leakage.
- Metriek: gemiddelde cross-sectionele IC per fold, IR, hit-rate van
  top-decile, plus gemiddelde feature-coëfficiënten.

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel).
"""

import os
import sys
import argparse
from typing import List

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import requests

from features import MODEL_FEATURES, MODEL_TARGET, MODEL_HORIZON

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

GENERIEKE_TECHNICALS_KOLOMMEN = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")


def haal_data(conn, features: List[str]) -> pd.DataFrame:
    s_feats = [f for f in features if f not in GENERIEKE_TECHNICALS_KOLOMMEN]
    g_feats = [f for f in features if f in GENERIEKE_TECHNICALS_KOLOMMEN]

    s_lijst = ", ".join(f"s.{k}" for k in s_feats)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g_feats)) if g_feats else ""

    query = f"""
        SELECT s.ticker, s.datum, s.strategie, {s_lijst}{g_lijst},
               f.{MODEL_TARGET} AS target
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE f.{MODEL_TARGET} IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def cross_sectionele_rank(df: pd.DataFrame, kolommen: List[str]) -> pd.DataFrame:
    """Vervang elke feature door zijn rank binnen de datum (0..1)."""
    out = df.copy()
    for k in kolommen:
        out[k] = out.groupby("datum")[k].rank(pct=True)
    return out


def walk_forward(df: pd.DataFrame, features: List[str],
                 min_train_datums: int = 10,
                 min_train_rijen: int = 200,
                 min_test_rijen: int = 20) -> pd.DataFrame:
    """Expanding-window walk-forward over datums. Geen leakage."""
    df = df.sort_values("datum").reset_index(drop=True)
    datums = sorted(df["datum"].unique())

    resultaten = []
    for i in range(min_train_datums, len(datums)):
        test_datum = datums[i]
        train_datums = datums[:i]

        train = df[df["datum"].isin(train_datums)].dropna(subset=features + ["target"])
        test = df[df["datum"] == test_datum].dropna(subset=features + ["target"])

        if len(train) < min_train_rijen or len(test) < min_test_rijen:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features].values)
        X_test = scaler.transform(test[features].values)

        model = Ridge(alpha=1.0)
        model.fit(X_train, train["target"].values)
        preds = model.predict(X_test)

        if len(np.unique(preds)) < 2:
            continue
        ic, _ = stats.spearmanr(preds, test["target"].values)
        if np.isnan(ic):
            continue

        n_top = max(1, int(len(test) * 0.10))
        top_idx = np.argsort(preds)[-n_top:]
        top_ret = test["target"].values[top_idx].mean()
        markt_ret = test["target"].values.mean()

        resultaten.append({
            "datum": test_datum,
            "n_train": len(train),
            "n_test": len(test),
            "ic": ic,
            "top_decile_ret": top_ret,
            "markt_ret": markt_ret,
            "excess_top": top_ret - markt_ret,
            "coefs": dict(zip(features, model.coef_.tolist())),
        })

    return pd.DataFrame(resultaten)


def print_rapport(res: pd.DataFrame, features: List[str]) -> str:
    if res.empty:
        return "Geen enkele fold produceerde een voorspelling."

    ics = res["ic"].values
    gem_ic = ics.mean()
    std_ic = ics.std(ddof=1) if len(ics) > 1 else np.nan
    ir = gem_ic / std_ic if std_ic and std_ic > 0 else np.nan
    t_stat = gem_ic / (std_ic / np.sqrt(len(ics))) if std_ic and std_ic > 0 else np.nan
    hit = (res["excess_top"] > 0).mean()

    regels = [
        f"\n{'=' * 90}",
        f"META-MODEL — walk-forward evaluatie op {MODEL_TARGET}",
        "=" * 90,
        f"Features ({len(features)}): {', '.join(features)}",
        f"Aantal folds: {len(res)}",
    ]
    if not np.isnan(std_ic):
        regels.append(f"Gemiddelde IC: {gem_ic:+.4f}  (std {std_ic:.4f})")
        regels.append(f"IR (IC/std): {ir:.3f}")
        regels.append(f"t-stat: {t_stat:.2f}")
    else:
        regels.append(f"Gemiddelde IC: {gem_ic:+.4f}")
    regels.append(f"Hit rate top-decile > markt: {hit:.1%}")
    regels.append("")
    regels.append("Per fold:")
    regels.append(f"{'datum':<12}{'n_train':>8}{'n_test':>8}{'IC':>10}"
                  f"{'top-decile':>12}{'markt':>10}{'excess':>10}")
    for _, r in res.iterrows():
        regels.append(
            f"{r['datum']:<12}{int(r['n_train']):>8}{int(r['n_test']):>8}"
            f"{r['ic']:>+10.4f}{r['top_decile_ret']:>+12.4f}"
            f"{r['markt_ret']:>+10.4f}{r['excess_top']:>+10.4f}"
        )

    coef_df = pd.DataFrame([r["coefs"] for _, r in res.iterrows()])
    gem_coefs = coef_df.mean().sort_values(key=lambda s: s.abs(), ascending=False)
    regels.append("\nGemiddelde coëfficiënten (belang features):")
    for naam, c in gem_coefs.items():
        regels.append(f"  {naam:<24} {c:+.4f}")

    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-datums", type=int, default=10,
                        help="Aantal initiële datums dat alleen als training dient")
    parser.add_argument("--min-train-rijen", type=int, default=200)
    parser.add_argument("--min-test-rijen", type=int, default=20)
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    features = list(MODEL_FEATURES)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = haal_data(conn, features)
        print(f"{len(df)} rijen opgehaald voor {MODEL_TARGET}.")
        if df.empty:
            print("Geen data.")
            return

        df = cross_sectionele_rank(df, features)

        res = walk_forward(df, features,
                           min_train_datums=args.min_train_datums,
                           min_train_rijen=args.min_train_rijen,
                           min_test_rijen=args.min_test_rijen)

        tekst = print_rapport(res, features)

        if not res.empty:
            gem_ic = res["ic"].mean()
            hit = (res["excess_top"] > 0).mean()
            samenvatting = (
                f"Meta-model {MODEL_TARGET}\n"
                f"Features: {len(features)}\n"
                f"Folds: {len(res)}\n"
                f"Gem. IC: {gem_ic:+.4f}\n"
                f"Hit rate top-decile: {hit:.1%}"
            )
            send_telegram(samenvatting)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
