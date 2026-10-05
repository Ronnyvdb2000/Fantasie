#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_meta_model_score.py — berekent meta-model scores voor de laatste
selectie-datum (of een opgegeven datum) en schrijft ze naar
meta_model_scores.

Gebruikt het nieuwste model uit meta_model_models. Past dezelfde
cross-sectionele ranking en scaler toe als tijdens training.

Stuurt optioneel een Telegram-bericht met top 10 + score-verdeling.

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel).

WIJZIGINGEN NA REVIEW (2026-10-05):
A. bouw_telegram_bericht stond twee keer in het bestand; Python gebruikte de
   laatste (kale) versie, dus de versie met "Toelichting" draaide nooit.
   De kale versie is hernoemd naar bouw_telegram_bericht_kaal() en wordt
   niet meer gebruikt (bewaard als documentatie).
B. send_telegram controleert nu de HTTP-statuscode en print bij een fout.
C. Beveiliging tegen scaler_std == 0 (deling door nul) bij het scoren.
D. Diagnose in de log: per strategie hoeveel selecties wegvallen omdat een
   feature ontbreekt (bv. geen "score"), zodat zichtbaar is welke bots niet
   gescoord worden.
"""

import os
import sys
import argparse

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import requests

from features import MODEL_HORIZON

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

GENERIEKE_TECHNICALS = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # OUDE CODE (zonder statuscontrole), bewust bewaard als documentatie:
    # try:
    #     requests.post(
    #         f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
    #         json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst},
    #         timeout=10,
    #     )
    # except Exception as e:
    #     print(f"Telegram fout: {e}")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst},
            timeout=10,
        )
        if r.status_code != 200:
            print(f"Telegram status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Telegram fout: {e}")

def bouw_telegram_bericht(df: pd.DataFrame, datum: str, versie: str) -> str:
    """Telegram-bericht met top 10 en score-verdeling."""
    scores = df["score"].values
    mediaan = float(np.median(scores))
    laagste = float(scores.min())
    hoogste = float(scores.max())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

    top10 = df.nlargest(10, "score")[["ticker", "strategie", "score"]]

    regels = [
        f"Meta-model scores {datum}",
        f"Model: {versie}",
        f"Totaal: {len(df)} scores",
        "",
        "Top 10:",
    ]
    for i, (_, r) in enumerate(top10.iterrows(), start=1):
        regels.append(f"{i:>2}. {r['ticker']:<12} {r['strategie']:<22} {r['score']:+.3f}")

    regels += [
        "",
        "Verdeling:",
        f"  hoogste   {hoogste:+.3f}",
        f"  mediaan   {mediaan:+.3f}",
        f"  laagste   {laagste:+.3f}",
        f"  std       {std:.3f}",
        f"  spreiding {hoogste - laagste:.3f}",
        "",
        "Toelichting:",
        "  mediaan   = middelste score van alle selecties;",
        "              negatief betekent dat het gros van de",
        "              selecties onder nul scoort.",
        "  std       = standaardafwijking; hoe hoger, hoe",
        "              meer de scores uit elkaar liggen.",
        "  spreiding = hoogste - laagste; grootte van de",
        "              bandbreedte. Kleine spreiding (<1)",
        "              betekent weinig onderscheidend vermogen.",
    ]
    return "\n".join(regels)

def haal_laatste_model(conn):
    query = """
        SELECT versie, horizon_dagen, features, coefs, scaler_mean, scaler_std, ridge_alpha
        FROM meta_model_models
        ORDER BY getraind_op DESC
        LIMIT 1;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        row = cur.fetchone()
    return row


def haal_selecties(conn, features, datum=None):
    """Haal selecties op voor een specifieke datum, of de laatste datum
    waarvoor selecties EN generieke_technicals bestaan."""
    s = [f for f in features if f not in GENERIEKE_TECHNICALS]
    g = [f for f in features if f in GENERIEKE_TECHNICALS]
    s_lijst = ", ".join(f"s.{k}" for k in s)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g)) if g else ""

    if datum is None:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(s.datum)
                FROM selecties s
                JOIN generieke_technicals g
                  ON s.ticker = g.ticker AND s.datum = g.datum;
            """)
            datum = cur.fetchone()[0]
    print(f"Scoren voor datum: {datum}")

    query = f"""
        SELECT s.ticker, s.datum, s.strategie, {s_lijst}{g_lijst}
        FROM selecties s
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE s.datum = %s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (datum,))
        return pd.DataFrame(cur.fetchall()), datum


# Oude, kale variant (zonder 'Toelichting'). Overschreef voorheen de
# volledige versie hierboven doordat de naam gelijk was. Bewust bewaard
# onder een andere naam; wordt niet aangeroepen.
def bouw_telegram_bericht_kaal(df: pd.DataFrame, datum: str, versie: str) -> str:
    """Telegram-bericht met top 10 en score-verdeling."""
    scores = df["score"].values
    mediaan = float(np.median(scores))
    laagste = float(scores.min())
    hoogste = float(scores.max())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

    top10 = df.nlargest(10, "score")[["ticker", "strategie", "score"]]

    regels = [
        f"Meta-model scores {datum}",
        f"Model: {versie}",
        f"Totaal: {len(df)} scores",
        "",
        "Top 10:",
    ]
    for i, (_, r) in enumerate(top10.iterrows(), start=1):
        regels.append(f"{i:>2}. {r['ticker']:<12} {r['strategie']:<22} {r['score']:+.3f}")

    regels += [
        "",
        "Verdeling:",
        f"  hoogste  {hoogste:+.3f}",
        f"  mediaan  {mediaan:+.3f}",
        f"  laagste  {laagste:+.3f}",
        f"  std      {std:.3f}",
        f"  spreiding {hoogste - laagste:.3f}",
    ]
    return "\n".join(regels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datum", type=str, default=None,
                        help="Specifieke datum (YYYY-MM-DD). Default: laatste selectiedatum met technicals.")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        model_row = haal_laatste_model(conn)
        if model_row is None:
            print("Geen model gevonden. Draai eerst a_meta_model_train.py.")
            sys.exit(1)

        versie = model_row["versie"]
        features = list(model_row["features"])
        coefs = np.array(model_row["coefs"], dtype=float)
        mean = np.array(model_row["scaler_mean"], dtype=float)
        std = np.array(model_row["scaler_std"], dtype=float)
        # Beveiliging: een schaal van 0 (constante feature) zou deling door
        # nul geven. StandardScaler zet die normaal al op 1.0; dit is extra.
        std = np.where(std == 0, 1.0, std)

        print(f"Model: {versie}")
        print(f"Horizon: {model_row['horizon_dagen']}d, features: {len(features)}")

        df, datum = haal_selecties(conn, features, args.datum)
        print(f"{len(df)} selecties opgehaald.")
        if df.empty:
            print("Geen selecties voor die datum.")
            return

        for k in features:
            df[k] = df[k].rank(pct=True)

        voor_dropna = df
        df = df.dropna(subset=features)
        gedropt = voor_dropna.loc[~voor_dropna.index.isin(df.index), "strategie"]
        if len(gedropt) > 0:
            print("Niet gescoord (feature ontbreekt), per strategie:")
            for strat, n in gedropt.value_counts().items():
                print(f"  {strat:<24} {n}")
        print(f"{len(df)} rijen met complete features.")

        X = (df[features].values - mean) / std
        scores = X @ coefs

        df["score"] = scores

        insert = """
            INSERT INTO meta_model_scores
                (datum, ticker, strategie, score, model_versie, horizon_dagen)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (datum, ticker, strategie, model_versie, horizon_dagen)
            DO UPDATE SET score = EXCLUDED.score, created_at = now();
        """
        rijen = [
            (datum, r["ticker"], r["strategie"], float(r["score"]), versie, MODEL_HORIZON)
            for _, r in df.iterrows()
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert, rijen, page_size=500)
        conn.commit()
        print(f"{len(rijen)} scores opgeslagen in meta_model_scores.")

        top = df.nlargest(10, "score")[["ticker", "strategie", "score"]]
        print("\nTop 10 scores:")
        print(top.to_string(index=False))

        if len(df) > 0:
            send_telegram(bouw_telegram_bericht(df, datum, versie))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
