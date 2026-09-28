#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_xgboost.py  —  scoort NIEUWE selecties met de getrainde xgboostV3-modellen  v1.0

Laadt per horizon (10d/30d/60d) het door xgboostV3.py getrainde model
(xgboostV3_<horizon>_model.pkl, uit de repo), rangschikt de recente selecties
uit `selecties` op modelscore en stuurt de top-N via Telegram en e-mail.
Horizons zonder modelbestand (nog niet getraind) worden overgeslagen.

Werking
=======
- Kandidaten: (ticker, datum)-paren in generieke_technicals met datum binnen
  LOOKBACK_DAGEN dagen, die in `selecties` voorkomen. Per paar worden ook de
  strategieën getoond die het aandeel selecteerden, plus koers en beurs.
- Elk paar wordt per horizon maar ÉÉN keer gescoord: wat al in xgboost_scores
  staat, komt niet opnieuw in het bericht. Zo krijg je dagelijks enkel nieuwe
  selecties te zien.
- ALLE gescoorde rijen (niet enkel de top-N) worden bewaard in xgboost_scores.
  Dat is bewust: pas als je later de gerealiseerde rendementen uit
  forward_returns naast de scores legt, zie je of de top-N in de praktijk
  beter presteert dan de rest. Dat is de echte out-of-sample controle.
- De kolommen waarop gescoord wordt komen uit het model zelf
  (model.feature_names_in_), zodat de features altijd met de training
  overeenkomen.

"Score" is de rangorde van het model, geen gekalibreerde kans en geen garantie.
De kwaliteit van het model (AUC, controle-AUC, trainingsdatum) staat in elk
bericht, uit de laatste rij in xgboost_runs.

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN / TELEGRAM_CHAT_ID en
EMAIL_USER / EMAIL_PASS / EMAIL_RECEIVER (optioneel), TOP_N (default 10),
LOOKBACK_DAGEN (default 3).

Vereist de tabellen xgboost_scores en (voor de kwaliteitsregel) xgboost_runs.
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
import psycopg2.extras
import requests
import xgboost  # noqa: F401  (nodig om het opgeslagen XGBClassifier-model te kunnen laden)

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

MODEL_VERSIE = "xgboostV3"
HORIZONS = ["10d", "30d", "60d"]

TOP_N = int(os.getenv("TOP_N", "10"))
LOOKBACK_DAGEN = int(os.getenv("LOOKBACK_DAGEN", "3"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

# Nog niet gescoorde (ticker, datum)-paren met technicals, plus de strategieën
# die ze selecteerden. datum is tekst in ISO-formaat, dus direct vergelijkbaar.
KANDIDATEN_QUERY = """
SELECT gt.*, sel.strategieen, sel.koers, sel.beurs
FROM generieke_technicals gt
JOIN (
    SELECT ticker, datum,
           string_agg(DISTINCT strategie, ', ' ORDER BY strategie) AS strategieen,
           MAX(koers) AS koers,
           MIN(beurs) AS beurs
    FROM selecties
    WHERE datum >= %(cutoff)s
    GROUP BY ticker, datum
) sel ON sel.ticker = gt.ticker AND sel.datum = gt.datum
WHERE NOT EXISTS (
    SELECT 1 FROM xgboost_scores xs
    WHERE xs.ticker = gt.ticker AND xs.datum = gt.datum
      AND xs.model_versie = %(model_versie)s AND xs.horizon = %(horizon)s
);
"""


# ============================================================
# BERICHTEN
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


# ============================================================
# SCOREN
# ============================================================

def score_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Voegt score en rang toe (rang 1 = hoogste score). Rijen met ontbrekende
    features vallen af, net als bij de training."""
    features = list(model.feature_names_in_)
    df = df.dropna(subset=features).copy()
    if df.empty:
        return df
    df["score"] = model.predict_proba(df[features])[:, 1]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rang"] = df.index + 1
    return df


def haal_kandidaten(conn, horizon: str, cutoff: str) -> pd.DataFrame:
    return pd.read_sql(
        KANDIDATEN_QUERY, conn,
        params={"cutoff": cutoff, "model_versie": MODEL_VERSIE, "horizon": horizon},
    )


def sla_scores_op(conn, horizon: str, df: pd.DataFrame) -> None:
    n = len(df)
    rows = []
    for r in df.itertuples():
        strategieen = None if pd.isna(r.strategieen) else str(r.strategieen)
        rows.append((MODEL_VERSIE, horizon, r.ticker, r.datum,
                     float(r.score), int(r.rang), n, strategieen))
    query = """
        INSERT INTO xgboost_scores
            (model_versie, horizon, ticker, datum, score, rang, n_gescoord, strategieen)
        VALUES %s
        ON CONFLICT (model_versie, horizon, ticker, datum) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, query, rows)
    conn.commit()


def haal_model_kwaliteit(conn, horizon: str) -> Optional[Dict]:
    """AUC, controle-AUC en datum van de laatste getrainde run (uit xgboost_runs)."""
    query = """
        SELECT auc, auc_relatief, run_datum
        FROM xgboost_runs
        WHERE model_versie = %(model_versie)s AND horizon = %(horizon)s
          AND status = 'getraind'
        ORDER BY run_datum DESC
        LIMIT 1;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, {"model_versie": MODEL_VERSIE, "horizon": horizon})
            rij = cur.fetchone()
    except Exception as e:
        conn.rollback()
        print(f"[WARN] modelkwaliteit niet op te halen: {e}")
        return None
    if not rij:
        return None
    return {"auc": rij[0], "auc_relatief": rij[1], "run_datum": rij[2].strftime("%Y-%m-%d")}


def _fmt(v, decimalen: int = 3) -> str:
    return "n.v.t." if v is None or pd.isna(v) else f"{float(v):.{decimalen}f}"


def bouw_sectie(horizon: str, df: pd.DataFrame, kwaliteit: Optional[Dict]) -> str:
    kop = f"[{horizon}] {len(df)} nieuwe selecties gescoord"
    if kwaliteit:
        kop += (f" | model-AUC {_fmt(kwaliteit['auc'])} (controle "
                f"{_fmt(kwaliteit['auc_relatief'])}, getraind {kwaliteit['run_datum']})")
    regels = [kop]
    for r in df.head(TOP_N).itertuples():
        strategieen = "onbekend" if pd.isna(r.strategieen) else r.strategieen
        koers = "" if pd.isna(r.koers) else f" | koers {float(r.koers):.2f}"
        regels.append(f"{r.rang:>2}. {r.ticker} ({r.datum}) score {r.score:.2f} "
                      f"← {strategieen}{koers}")
    return "\n".join(regels)


def bouw_bericht(secties: List[str], datum: str) -> str:
    return "\n\n".join(
        [f"🎯 XGBoostV3 — top {TOP_N} nieuwe selecties ({datum})"]
        + secties
        + ["Score = rangorde van het model, geen gekalibreerde kans en geen garantie. "
           "Gerealiseerde resultaten volg je via xgboost_scores + forward_returns."]
    )


# ============================================================
# MAIN
# ============================================================

def laad_modellen() -> Dict[str, object]:
    modellen = {}
    for horizon in HORIZONS:
        pad = f"{MODEL_VERSIE}_{horizon}_model.pkl"
        if os.path.exists(pad):
            modellen[horizon] = joblib.load(pad)
            print(f"[{horizon}] model geladen ({pad}).")
        else:
            print(f"[{horizon}] geen modelbestand ({pad}), overgeslagen.")
    return modellen


def run_scoring() -> None:
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL ontbreekt in de omgeving.")

    modellen = laad_modellen()
    if not modellen:
        print("Geen enkel model beschikbaar, niets te scoren.")
        return

    nu = dt.datetime.now(dt.timezone.utc)
    datum = nu.strftime("%Y-%m-%d")
    cutoff = (nu - dt.timedelta(days=LOOKBACK_DAGEN)).strftime("%Y-%m-%d")

    secties: List[str] = []
    conn = psycopg2.connect(db_url)
    try:
        for horizon, model in modellen.items():
            kandidaten = haal_kandidaten(conn, horizon, cutoff)
            gescoord = score_dataframe(model, kandidaten)
            print(f"[{horizon}] {len(kandidaten)} kandidaten sinds {cutoff}, "
                  f"{len(gescoord)} met volledige features gescoord.")
            if gescoord.empty:
                continue
            sla_scores_op(conn, horizon, gescoord)
            kwaliteit = haal_model_kwaliteit(conn, horizon)
            secties.append(bouw_sectie(horizon, gescoord, kwaliteit))
    finally:
        conn.close()

    if not secties:
        print("Geen nieuwe selecties om te melden.")
        return

    bericht = bouw_bericht(secties, datum)
    send_telegram(bericht)
    send_email(f"XGBoostV3 top selecties {datum}", bericht)


if __name__ == "__main__":
    run_scoring()
