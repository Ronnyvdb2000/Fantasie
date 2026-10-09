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

WIJZIGINGEN v1.3 (2026-10-09):
G. Fix: 'Kleine spreiding (<1)' werd door Telegram's HTML-parser gelezen
   als het begin van een HTML-tag → Bad Request 400. Vervangen door
   '&lt;1' (HTML-entiteit voor '<').
"""

import os
import sys
import argparse
import html

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


# ============================================================
# TELEGRAM
# ============================================================

def _esc(s) -> str:
    """Escaped speciale HTML-tekens (&, <, >)."""
    return html.escape(str(s))


def _format_koers(koers) -> str:
    """Formatteert de koers met 2 decimalen of '-' als onbekend."""
    try:
        if koers is None or pd.isna(koers):
            return "-"
        return f"{float(koers):.2f}"
    except Exception:
        return "-"


def send_telegram(tekst: str) -> None:
    """Verstuurt een bericht in HTML-modus (klikbare links, vet)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": tekst,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"Telegram status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Telegram fout: {e}")


def bouw_telegram_bericht(df: pd.DataFrame, datum: str, versie: str) -> str:
    """
    Telegram-bericht (HTML) met top 10, score-verdeling, koers en een
    klikbare Yahoo Finance-link per ticker.
    """
    scores = df["score"].values
    mediaan = float(np.median(scores))
    laagste = float(scores.min())
    hoogste = float(scores.max())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

    # Zorg dat koers er is (kan None zijn)
    if "koers" not in df.columns:
        df["koers"] = np.nan

    top10 = df.nlargest(10, "score")[["ticker", "strategie", "score", "koers"]]

    regels = [
        "📊 <b>Meta-model scores</b>",
        "",
        f"Datum: <b>{_esc(datum)}</b>",
        f"Model: <code>{_esc(versie)}</code>",
        f"Totaal: <b>{len(df)}</b> scores",
        "",
        "<b>Top 10:</b>",
    ]

    for i, (_, r) in enumerate(top10.iterrows(), start=1):
        ticker = str(r["ticker"])
        ticker_url = ticker.replace(" ", "")
        yahoo_url = f"https://finance.yahoo.com/quote/{ticker_url}"
        koers_str = _format_koers(r.get("koers"))
        score_val = float(r["score"])

        regels.append(
            f"{i:>2}. <b>{_esc(ticker)}</b> — "
            f"{_esc(r['strategie'])} — "
            f"score <b>{score_val:+.3f}</b> — "
            f"koers {koers_str} — "
            f'<a href="{yahoo_url}">📈 Grafiek</a>'
        )

    regels += [
        "",
        "<b>Verdeling:</b>",
        f"  hoogste   {hoogste:+.3f}",
        f"  mediaan   {mediaan:+.3f}",
        f"  laagste   {laagste:+.3f}",
        f"  std       {std:.3f}",
        f"  spreiding {hoogste - laagste:.3f}",
        "",
        "<b>Toelichting:</b>",
        "  mediaan   = middelste score van alle selecties;",
        "              negatief betekent dat het gros van de",
        "              selecties onder nul scoort.",
        "  std       = standaardafwijking; hoe hoger, hoe",
        "              meer de scores uit elkaar liggen.",
        "  spreiding = hoogste - laagste; grootte van de",
        "              bandbreedte. Kleine spreiding (&lt;1)",
        "              betekent weinig onderscheidend vermogen.",
    ]
    return "\n".join(regels)


# ============================================================
# DATABASE
# ============================================================

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
    waarvoor selecties EN generieke_technicals bestaan.

    v1.2: s.koers wordt ook opgehaald, zodat het Telegram-bericht
    de koers kan tonen.
    """
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

    # v1.2: s.koers toegevoegd
    query = f"""
        SELECT s.ticker, s.datum, s.strategie, s.koers, {s_lijst}{g_lijst}
        FROM selecties s
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE s.datum = %s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (datum,))
        return pd.DataFrame(cur.fetchall()), datum


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datum",
        type=str,
        default=None,
        help=(
            "Specifieke datum (YYYY-MM-DD). "
            "Default: laatste selectiedatum met technicals."
        ),
    )
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

        top = df.nlargest(10, "score")[["ticker", "strategie", "score", "koers"]]
        print("\nTop 10 scores:")
        print(top.to_string(index=False))

        if len(df) > 0:
            send_telegram(bouw_telegram_bericht(df, datum, versie))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
