#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_forward_returns.py  —  EERSTE VERKENNING VAN forward_returns  v1.0

Beantwoordt de kernvraag van de volgende Selecties-analyse-stap: per
strategie, wat was het rendement na EXACT dezelfde horizons (5/10/20
handelsdagen) -- appels met appels, in plaats van de impliciete, wisselende
horizon van de vorige, ad-hoc Selecties-analyses.

Voor elke strategie x horizon wordt berekend:
  - n              aantal gelabelde (ticker, datum)-combinaties
  - winrate        % met een positief rendement
  - gemiddelde     ruw gemiddelde rendement (%)
  - mediaan        mediaan rendement (%) -- robuuster tegen outliers
  - getrimd_gem    10%-getrimd gemiddelde (zelfde conventie als eerdere
                    Selecties-analyses in dit project)
  - p_waarde       one-sample t-test tegen gemiddelde=0 (H0: geen edge)
  - significant    of deze strategie de Benjamini-Hochberg FDR-correctie
                    doorstaat, GEPOOLD over alle strategieën op DEZE
                    horizon (zelfde bh_correctie()-implementatie als
                    bot_01repititief.py, hier toegepast op strategieën
                    i.p.v. seizoensbuckets) -- dit beantwoordt direct
                    "hoeveel van deze schijnbare winnaars zijn gewoon
                    toeval, gegeven dat er ~20 strategieën tegelijk
                    getest worden?"

Dit script schrijft niets weg -- het is een leesscript, geen bot. Output:
een tabel naar stdout (en optioneel Telegram als secrets aanwezig zijn),
gesorteerd op getrimd gemiddelde van de langste horizon (20d).

GEBRUIK
=======
  python analyse_forward_returns.py [--min-n 30] [--horizon 20]

Env vars:
  SUPABASE_DB_URL     - Postgres connectiestring
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  - optioneel
  FDR_ALPHA           - Benjamini-Hochberg alpha (default 0.05, zelfde
                        als bot_01repititief na de verstrenging)
"""

import os
import sys
import argparse
from typing import Dict, List

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from scipy import stats
import requests

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FDR_ALPHA = float(os.environ.get("FDR_ALPHA", "0.05"))

HORIZONS = [5, 10, 20]
TRIM_PCT = 0.10  # 10%-getrimd gemiddelde, zelfde conventie als eerdere analyses


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")


# --------------------------------------------------------------------------
# Zelfde Benjamini-Hochberg-implementatie als bot_01repititief.py, hier
# toegepast op strategieën i.p.v. seizoensbuckets.
# --------------------------------------------------------------------------
def bh_correctie(p_waarden: List[float], alpha: float) -> List[bool]:
    m = len(p_waarden)
    if m == 0:
        return []
    volgorde = sorted(range(m), key=lambda i: p_waarden[i])
    laatste_significante_rank = -1
    for rang, idx in enumerate(volgorde, start=1):
        if p_waarden[idx] <= (rang / m) * alpha:
            laatste_significante_rank = rang
    mask = [False] * m
    if laatste_significante_rank >= 0:
        for rang, idx in enumerate(volgorde, start=1):
            if rang <= laatste_significante_rank:
                mask[idx] = True
    return mask


def getrimd_gemiddelde(waarden: pd.Series, trim_pct: float = TRIM_PCT) -> float:
    if len(waarden) < 5:
        return float(waarden.mean())
    return float(stats.trim_mean(waarden.dropna(), trim_pct))


# --------------------------------------------------------------------------
# Data ophalen: JOIN selecties + forward_returns
# --------------------------------------------------------------------------
def haal_gelabelde_data(conn) -> pd.DataFrame:
    query = """
        SELECT s.strategie, s.beurs, s.ticker, s.datum,
               f.fwd_ret_5d, f.fwd_ret_10d, f.fwd_ret_20d
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        WHERE f.fwd_ret_5d IS NOT NULL OR f.fwd_ret_10d IS NOT NULL OR f.fwd_ret_20d IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Per strategie x horizon statistieken berekenen
# --------------------------------------------------------------------------
def bereken_statistieken(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    resultaten = []
    for horizon in HORIZONS:
        kolom = f"fwd_ret_{horizon}d"
        sub = df[["strategie", kolom]].dropna()

        per_strategie = []
        for strategie, groep in sub.groupby("strategie"):
            waarden = groep[kolom]
            n = len(waarden)
            if n < min_n:
                continue
            if n >= 3 and waarden.std() > 0:
                t_stat, p_waarde = stats.ttest_1samp(waarden, 0.0)
            else:
                p_waarde = 1.0
            per_strategie.append({
                "horizon": horizon, "strategie": strategie, "n": n,
                "winrate_pct": round((waarden > 0).mean() * 100, 1),
                "gemiddelde_pct": round(waarden.mean(), 2),
                "mediaan_pct": round(waarden.median(), 2),
                "getrimd_gem_pct": round(getrimd_gemiddelde(waarden), 2),
                "p_waarde": p_waarde,
            })

        if not per_strategie:
            continue

        p_waarden = [r["p_waarde"] for r in per_strategie]
        significant_mask = bh_correctie(p_waarden, FDR_ALPHA)
        for r, sig in zip(per_strategie, significant_mask):
            r["significant_fdr"] = sig
        resultaten.extend(per_strategie)

    return pd.DataFrame(resultaten)


# --------------------------------------------------------------------------
# Rapportage
# --------------------------------------------------------------------------
def print_rapport(stats_df: pd.DataFrame, horizon: int) -> str:
    sub = stats_df[stats_df["horizon"] == horizon].sort_values("getrimd_gem_pct", ascending=False)
    regels = [f"\n{'=' * 90}", f"HORIZON: {horizon} handelsdagen  (FDR alpha={FDR_ALPHA}, min_n toegepast)", "=" * 90]
    regels.append(f"{'strategie':<24}{'n':>7}{'winrate%':>10}{'gem%':>8}{'mediaan%':>10}{'getrimd%':>10}{'p-waarde':>10}  significant?")
    for _, r in sub.iterrows():
        vlag = "✓ JA" if r["significant_fdr"] else "  nee"
        regels.append(
            f"{r['strategie']:<24}{r['n']:>7}{r['winrate_pct']:>9.1f}%{r['gemiddelde_pct']:>7.2f}%"
            f"{r['mediaan_pct']:>9.2f}%{r['getrimd_gem_pct']:>9.2f}%{r['p_waarde']:>10.4f}  {vlag}"
        )
    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=30, help="minimum aantal observaties om een strategie mee te nemen")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        df = haal_gelabelde_data(conn)
        print(f"{len(df)} gelabelde (ticker, datum, strategie)-rijen opgehaald uit de JOIN.")
        if df.empty:
            print("Nog geen gelabelde data beschikbaar.")
            return

        stats_df = bereken_statistieken(df, args.min_n)
        if stats_df.empty:
            print(f"Geen enkele strategie haalt min_n={args.min_n} observaties op nog een horizon.")
            return

        telegram_delen = []
        for horizon in HORIZONS:
            tekst = print_rapport(stats_df, horizon)
            telegram_delen.append(tekst)

        aantal_sig_20d = stats_df[(stats_df["horizon"] == 20) & (stats_df["significant_fdr"])].shape[0]
        totaal_20d = stats_df[stats_df["horizon"] == 20].shape[0]
        samenvatting = (
            f"📊 *Forward-Returns analyse*\n\n"
            f"Op 20d-horizon: {aantal_sig_20d}/{totaal_20d} strategieën significant "
            f"na BH-correctie (alpha={FDR_ALPHA})\n\n"
            "Volledige tabel: zie workflow-log."
        )
        send_telegram(samenvatting)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
