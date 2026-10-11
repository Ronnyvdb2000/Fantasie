#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_par_model.py — lineair multi-factor model op basis van rolling IC-gewichten
================================================================================

Werkwijze:
  1. Voor elke horizon (5d, 10d, 20d, 30d, 60d): bereken per parameter de IC
     over de laatste ROLLING_DAGEN kalenderdagen (≈20 handelsdagen) tot
     score_datum - EMBARGO_DAGEN. Fallback naar FALLBACK_DAGEN als er te
     weinig datums zijn.
  2. Winsorize de IC's op ±IC_MAX_ABS.
  3. Filter op |IC| >= MIN_IC_ABS. Parameters met minder dan MIN_DATUMS_IC
     datums dekking worden overgeslagen.
  4. Voor elke (ticker, score_datum): rangschik elke parameter cross-
     sectioneel (0..1) en bereken score = Σ (IC_i × rang_i) / Σ |IC_i|.
  5. Top-N per horizon = picks.
  6. Output: CSV per horizon + Telegram-bericht + optioneel schrijven naar
     selecties (strategie a_par_model_v1).

Env:
  SUPABASE_DB_URL          verplicht
  TELEGRAM_TOKEN           optioneel
  TELEGRAM_CHAT_ID         optioneel
"""

import os
import sys
import argparse
import html

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATIE
# ============================================================

HORIZONS_DEFAULT = "5d,10d,20d,30d,60d"
TOP_N_DEFAULT = 10
MIN_IC_DEFAULT = 0.02
ROLLING_DAGEN_DEFAULT = 28
FALLBACK_DAGEN_DEFAULT = 90
EMBARGO_DAGEN_DEFAULT = 25
IC_MAX_ABS_DEFAULT = 0.15
MIN_DATUMS_IC_DEFAULT = 10
MIN_TICKERS_PER_DATUM = 20
MIN_PARAMS_VOOR_MODEL = 5

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Kolommen die geen factor zijn: identifiers, absolute prijsniveaus, targets
NIET_FACTOR = {
    "ticker", "datum", "beurs", "strategie", "bijgewerkt_op", "id",
    "entry_koers", "high52w", "ma50", "ma200", "market_cap",
    "ema8", "ema20",
    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d", "fwd_ret_60d",
    "fwd_close_5d", "fwd_close_10d", "fwd_close_20d",
    "fwd_close_30d", "fwd_close_60d",
    "koers", "score",
}


# ============================================================
# HULPFUNCTIES
# ============================================================

def _esc(s) -> str:
    return html.escape(str(s))


def fmt(v, decimals=4) -> str:
    try:
        if v is None or pd.isna(v):
            return "n.v.t."
        return f"{float(v):.{decimals}f}"
    except Exception:
        return "n.v.t."


def send_telegram(tekst: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Secrets ontbreken — bericht NIET verstuurd.")
        return False
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
            print(f"[Telegram] FOUT {r.status_code}: {r.text[:200]}")
            return False
        print("[Telegram] Bericht verzonden.")
        return True
    except Exception as e:
        print(f"[Telegram] Exception: {e}")
        return False


# ============================================================
# DATA OPHALEN
# ============================================================

def get_data(conn) -> pd.DataFrame:
    """Haal technische + fundamentele + forward returns op."""
    query = """
    WITH tech AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, atr14, atr14_pct, rsi14, ibs,
            ma50, ma200, pct_from_ma50, pct_from_ma200,
            vol_ratio_20d, high52w, pct_from_high52w,
            macd, macd_signaal, macd_hist, bb_breedte, bb_percent_b,
            stoch_k, stoch_d, adx14, rel_sterkte_20d, hv20, hv60,
            dagen_sinds_low52w, vol_ratio_50d,
            ema8, ema20, pct_from_ema8, pct_from_ema20, ema8_minus_ema20
        FROM generieke_technicals
        ORDER BY ticker, datum
    ),
    fund AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, market_cap, trailing_pe, price_to_book,
            dividend_yield, current_ratio, revenue_growth_pct, fcf_yield,
            net_debt_ebitda, payout_pct, analisten_count,
            piotroski_score, eps_growth_pct, eps_cagr_pct, peg_ratio
        FROM generieke_fundamentals
        ORDER BY ticker, datum DESC
    ),
    fr AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum,
            fwd_ret_5d, fwd_ret_10d, fwd_ret_20d, fwd_ret_30d, fwd_ret_60d
        FROM forward_returns
        ORDER BY ticker, datum
    )
    SELECT
        t.ticker, t.datum,
        t.atr14, t.atr14_pct, t.rsi14, t.ibs,
        t.ma50, t.ma200, t.pct_from_ma50, t.pct_from_ma200,
        t.vol_ratio_20d, t.high52w, t.pct_from_high52w,
        t.macd, t.macd_signaal, t.macd_hist, t.bb_breedte, t.bb_percent_b,
        t.stoch_k, t.stoch_d, t.adx14, t.rel_sterkte_20d, t.hv20, t.hv60,
        t.dagen_sinds_low52w, t.vol_ratio_50d,
        t.ema8, t.ema20, t.pct_from_ema8, t.pct_from_ema20, t.ema8_minus_ema20,
        f.market_cap, f.trailing_pe, f.price_to_book, f.dividend_yield,
        f.current_ratio, f.revenue_growth_pct, f.fcf_yield,
        f.net_debt_ebitda, f.payout_pct, f.analisten_count,
        f.piotroski_score, f.eps_growth_pct, f.eps_cagr_pct, f.peg_ratio,
        fr.fwd_ret_5d, fr.fwd_ret_10d, fr.fwd_ret_20d,
        fr.fwd_ret_30d, fr.fwd_ret_60d
    FROM tech t
    LEFT JOIN fund f ON f.ticker = t.ticker AND f.datum = t.datum
    LEFT JOIN fr ON fr.ticker = t.ticker AND fr.datum = t.datum;
    """
    df = pd.read_sql(query, conn)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    return df


# ============================================================
# IC-BEREKENING (rolling window + fallback)
# ============================================================

def bereken_ic_per_parameter(
    df: pd.DataFrame,
    target_kolom: str,
    train_start: pd.Timestamp,
    train_einde: pd.Timestamp,
    min_datums: int,
    ic_max_abs: float,
) -> pd.DataFrame:
    """Bereken per parameter IC op [train_start, train_einde], winsorized."""
    train = df[(df["datum"] >= train_start) & (df["datum"] <= train_einde)]
    if len(train) < MIN_TICKERS_PER_DATUM:
        return pd.DataFrame()

    params = [
        c for c in df.columns
        if c not in NIET_FACTOR
        and c != target_kolom
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rijen = []
    for p in params:
        sub = train[["datum", p, target_kolom]].dropna()
        if len(sub) < MIN_TICKERS_PER_DATUM:
            continue

        ics = []
        for _, groep in sub.groupby("datum"):
            if len(groep) < MIN_TICKERS_PER_DATUM:
                continue
            rho, _ = spearmanr(groep[p], groep[target_kolom])
            if not np.isnan(rho):
                ics.append(rho)

        if len(ics) < min_datums:
            continue

        ics = np.array(ics)
        ic_gem = float(np.mean(ics))
        ic_gem_wins = max(min(ic_gem, ic_max_abs), -ic_max_abs)
        ic_std = float(np.std(ics))
        ic_ir = ic_gem / ic_std if ic_std > 1e-9 else 0.0

        rijen.append({
            "parameter": p,
            "ic": round(ic_gem_wins, 4),
            "ic_raw": round(ic_gem, 4),
            "ic_abs": round(abs(ic_gem_wins), 4),
            "ic_ir": round(ic_ir, 3),
            "n_datums": len(ics),
            "n_obs": len(sub),
        })

    if not rijen:
        return pd.DataFrame()
    return pd.DataFrame(rijen).sort_values("ic_abs", ascending=False)


def ic_met_fallback(
    df, target_kolom, score_datum,
    embargo_dagen, rolling_dagen, fallback_dagen,
    min_datums, ic_max_abs,
):
    """Probeert eerst rolling_dagen, valt terug op fallback_dagen."""
    train_einde = score_datum - pd.Timedelta(days=embargo_dagen)

    for dagen, label in [
        (rolling_dagen, f"rolling {rolling_dagen}d"),
        (fallback_dagen, f"fallback {fallback_dagen}d"),
    ]:
        train_start = train_einde - pd.Timedelta(days=dagen)
        gewichten = bereken_ic_per_parameter(
            df, target_kolom, train_start, train_einde,
            min_datums, ic_max_abs,
        )
        if not gewichten.empty and len(gewichten) >= MIN_PARAMS_VOOR_MODEL:
            return gewichten, label

    return pd.DataFrame(), "geen"


# ============================================================
# SCORING
# ============================================================

def score_voor_datum(df, score_datum, gewichten, top_n):
    """Scoort alle aandelen op score_datum; top-N met score."""
    dag = df[df["datum"] == score_datum].copy()
    if dag.empty:
        return pd.DataFrame()

    params = [p for p in gewichten["parameter"] if p in dag.columns]
    if not params:
        return pd.DataFrame()

    gewicht_per_param = dict(zip(gewichten["parameter"], gewichten["ic"]))
    totaal_gewicht = sum(abs(gewicht_per_param[p]) for p in params)
    if totaal_gewicht < 1e-9:
        return pd.DataFrame()

    rang_kolommen = {}
    for p in params:
        rang_kolommen[p] = f"{p}__rang"
        dag[rang_kolommen[p]] = dag[p].rank(pct=True, na_option="keep")

    def _score(rij):
        score = 0.0
        for p in params:
            rang = rij.get(rang_kolommen[p])
            if pd.isna(rang):
                continue
            score += gewicht_per_param[p] * rang
        return score / totaal_gewicht

    dag["score"] = dag.apply(_score, axis=1)

    top = (
        dag.dropna(subset=["score"])
        .nlargest(top_n, "score")
        [["ticker", "datum", "score"]]
    )
    return top
  

# ============================================================
# TELEGRAM
# ============================================================

def bouw_telegram_bericht(
    picks_per_horizon: dict,
    gewichten_per_horizon: dict,
    window_labels: dict,
    score_datum: pd.Timestamp,
) -> str:
    regels = [
        "🎯 <b>A Parameter-model — top picks</b>",
        "",
        f"Datum: <b>{_esc(score_datum.date())}</b>",
        "",
    ]

    for horizon in ["5d", "10d", "20d", "30d", "60d"]:
        picks = picks_per_horizon.get(horizon)
        gewichten = gewichten_per_horizon.get(horizon)
        window = window_labels.get(horizon, "?")

        if picks is None or picks.empty:
            regels.append(f"<b>{horizon}</b> — geen picks")
            regels.append("")
            continue

        regels.append(
            f"<b>Horizon {horizon}</b> ({window}, {len(gewichten)} params)"
        )
        for i, (_, r) in enumerate(picks.head(10).iterrows(), start=1):
            ticker = str(r["ticker"])
            yahoo_url = f"https://finance.yahoo.com/quote/{ticker}"
            score = float(r["score"])
            regels.append(
                f"  {i:>2}. <b>{_esc(ticker)}</b> "
                f"<code>{score:+.3f}</code> "
                f'<a href="{yahoo_url}">📈</a>'
            )
        regels.append("")

    return "\n".join(regels)


# ============================================================
# DB-WRITE
# ============================================================

def schrijf_naar_selecties(conn, picks_per_horizon, score_datum):
    """Schrijft per horizon de top-N naar selecties onder a_par_model_v1."""
    insert = """
        INSERT INTO selecties (ticker, datum, strategie, beurs, koers, parameters)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
    """

    rijen = []
    for horizon, picks in picks_per_horizon.items():
        if picks is None or picks.empty:
            continue
        for _, r in picks.iterrows():
            params = {
                "score": float(r["score"]),
                "model": "a_par_model_v1",
                "horizon": horizon,
                "grafiek": f"https://finance.yahoo.com/quote/{r['ticker']}",
            }
            rijen.append((
                r["ticker"],
                score_datum.strftime("%Y-%m-%d"),
                "a_par_model_v1",
                None, None,
                psycopg2.extras.Json(params),
            ))

    if not rijen:
        return 0

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, insert, rijen, page_size=200)
        conn.commit()
        print(f"[DB] {len(rijen)} picks geschreven naar selecties.")
        return len(rijen)
    except Exception as e:
        conn.rollback()
        print(f"[DB] Schrijven mislukt: {e}")
        return 0


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default=HORIZONS_DEFAULT)
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--min-ic", type=float, default=MIN_IC_DEFAULT)
    ap.add_argument("--rolling-dagen", type=int, default=ROLLING_DAGEN_DEFAULT)
    ap.add_argument("--fallback-dagen", type=int, default=FALLBACK_DAGEN_DEFAULT)
    ap.add_argument("--embargo-dagen", type=int, default=EMBARGO_DAGEN_DEFAULT)
    ap.add_argument("--ic-max-abs", type=float, default=IC_MAX_ABS_DEFAULT)
    ap.add_argument("--min-datums-ic", type=int, default=MIN_DATUMS_IC_DEFAULT)
    ap.add_argument("--datum", default=None)
    ap.add_argument("--geen-db-write", action="store_true")
    ap.add_argument("--geen-notify", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        return 1

    print()
    print("=" * 70)
    print("A PARAMETER-MODEL")
    print("=" * 70)
    print(f"Horizons        : {', '.join(horizons)}")
    print(f"Top-N           : {args.top_n}")
    print(f"Min |IC|        : {args.min_ic}")
    print(f"Rolling window  : {args.rolling_dagen}d (fallback {args.fallback_dagen}d)")
    print(f"Embargo         : {args.embargo_dagen}d")
    print(f"Winsorize       : ±{args.ic_max_abs}")
    print(f"Min datums IC   : {args.min_datums_ic}")

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        print("\nData ophalen...")
        df = get_data(conn)
        print(f"{len(df):,} rijen | {df['datum'].nunique()} datums | "
              f"{df['ticker'].nunique()} tickers")

        if args.datum:
            score_datum = pd.Timestamp(args.datum)
        else:
            score_datum = df["datum"].max()
        print(f"\nScore-datum: {score_datum.date()}")
        print(f"Train tot:   {(score_datum - pd.Timedelta(days=args.embargo_dagen)).date()}")

        picks_per_horizon = {}
        gewichten_per_horizon = {}
        window_labels = {}

        for horizon in horizons:
            target_kolom = f"fwd_ret_{horizon}"
            if target_kolom not in df.columns:
                print(f"\n[{horizon}] kolom {target_kolom} ontbreekt — overgeslagen")
                continue

            print(f"\n[{horizon}] IC berekenen...")
            gewichten, window = ic_met_fallback(
                df, target_kolom, score_datum,
                args.embargo_dagen, args.rolling_dagen, args.fallback_dagen,
                args.min_datums_ic, args.ic_max_abs,
            )

            if gewichten.empty:
                print(f"[{horizon}] geen parameters met genoeg data — overgeslagen")
                continue

            gewichten = gewichten[gewichten["ic_abs"] >= args.min_ic].copy()
            if gewichten.empty:
                print(f"[{horizon}] geen parameters met |IC| >= {args.min_ic}")
                continue

            print(f"[{horizon}] {len(gewichten)} parameters (window: {window})")
            for _, r in gewichten.head(5).iterrows():
                print(f"    {r['parameter']:<24} "
                      f"IC={r['ic']:+.4f} (raw {r['ic_raw']:+.4f}) "
                      f"IR={r['ic_ir']:+.2f} n={r['n_datums']}")

            picks = score_voor_datum(df, score_datum, gewichten, args.top_n)
            if picks.empty:
                print(f"[{horizon}] geen picks (geen data voor score_datum)")
                continue

            picks_per_horizon[horizon] = picks
            gewichten_per_horizon[horizon] = gewichten
            window_labels[horizon] = window

            print(f"[{horizon}] Top {len(picks)}:")
            for i, (_, r) in enumerate(picks.iterrows(), start=1):
                print(f"    {i:>2}. {r['ticker']:<12} score={r['score']:+.4f}")

            csv_path = os.path.join(RESULTS_DIR, f"a_par_model_{horizon}_picks.csv")
            picks.to_csv(csv_path, index=False)
            w_path = os.path.join(RESULTS_DIR, f"a_par_model_{horizon}_weights.csv")
            gewichten.to_csv(w_path, index=False)

        if not picks_per_horizon:
            print("\nGeen enkele horizon leverde picks op.")
            return 1

        if not args.geen_db_write:
            schrijf_naar_selecties(conn, picks_per_horizon, score_datum)

        if not args.geen_notify:
            bericht = bouw_telegram_bericht(
                picks_per_horizon, gewichten_per_horizon,
                window_labels, score_datum,
            )
            send_telegram(bericht)

        print("\n✅ Klaar.")
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
