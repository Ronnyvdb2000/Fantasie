#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_parameter_correlatie.py  —  WELKE PARAMETER ZEGT IETS?  v2.0

Per INDIVIDUELE SELECTIE (rij = 1 aandeel op 1 moment) wordt gekeken of de
ruwe parameterwaarde samenhangt met het nadien behaalde rendement -- maar
niet meer gepoold over alle datums. In plaats daarvan:

  * Numerieke parameters: per datum cross-sectionele Spearman (rank IC),
    daarna t-toets op de reeks IC's. Dit isoleert het SELECTIE-effect
    ("aandeel met hoge RSI vandaag doet het morgen beter dan een aandeel
    met lage RSI vandaag") van het MARKTREGIME-effect ("op dagen dat alle
    RSI's hoog zijn, gaat de markt omhoog").

  * Boolean parameters: per datum mediaan(True) - mediaan(False), daarna
    t-toets op de reeks verschillen. Zelfde robuustheid tegen scheve
    rendementsverdelingen, zonder regime-besmetting.

Horizonten: 5, 10 (primair), 20, 30, 60 handelsdagen.

FDR-correctie (BH) wordt per horizon toegepast -- elke horizon is een
aparte onderzoeksvraag. In de samenvatting wordt de primaire horizon (10d)
apart uitgelicht; andere horizonten zijn exploratief.

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel), FDR_ALPHA (default 0.05)
"""

import os
import sys
import argparse
from typing import List, Optional, Tuple

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

HORIZONS = [5, 10, 20, 30, 60]
PRIMAIRE_HORIZON = 10

# Minimum aantal datums waarop een parameter cross-sectioneel evalueerbaar
# is voordat we een t-toets durven rapporteren. Onder deze drempel is de
# reeks IC's te kort om iets zinnigs over significantie te zeggen.
MIN_DAGEN = 10

# Minimum aantal aandelen per datum om een cross-sectionele IC te berekenen.
# Te klein -> ruis. Default via CLI overschrijfbaar.
DEFAULT_MIN_PER_DATUM = 15

UITGESLOTEN_KOLOMMEN = {
    "id", "ticker", "datum", "strategie", "beurs", "koers",
    "parameters", "grafiek", "rsi_label", "macd_label",
}
NUMERIEKE_TYPES = {"double precision", "integer", "numeric", "real", "bigint", "smallint"}
BOOLEAN_TYPES = {"boolean"}

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
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst},  # plain text, geen Markdown-parse-ellende
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")


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


def haal_feature_kolommen(conn) -> Tuple[List[str], List[str]]:
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'selecties';
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    numeriek, boolean = [], []
    for naam, dtype in rows:
        if naam in UITGESLOTEN_KOLOMMEN:
            continue
        if dtype in NUMERIEKE_TYPES:
            numeriek.append(naam)
        elif dtype in BOOLEAN_TYPES:
            boolean.append(naam)
    return sorted(numeriek), sorted(boolean)


def haal_gelabelde_data(conn, kolommen: List[str]) -> pd.DataFrame:
    s_kolommen = [k for k in kolommen if k not in GENERIEKE_TECHNICALS_KOLOMMEN]
    g_kolommen = [k for k in kolommen if k in GENERIEKE_TECHNICALS_KOLOMMEN]

    s_lijst = ", ".join(f"s.{k}" for k in s_kolommen)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g_kolommen)) if g_kolommen else ""

    # fwd_ret_30d en fwd_ret_60d moeten bestaan in forward_returns; zo niet,
    # faalt de query hieronder -- dat is bewust, dan weet je dat je view
    # nog niet uitgebreid is.
    query = f"""
        SELECT s.datum, s.strategie, {s_lijst}{g_lijst},
               f.fwd_ret_5d, f.fwd_ret_10d, f.fwd_ret_20d, f.fwd_ret_30d, f.fwd_ret_60d
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def _t_toets_op_reeks(waarden: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Gemiddelde, t-stat en tweezijdige p van een reeks (bv. per-datum IC's)."""
    n = len(waarden)
    if n < MIN_DAGEN:
        return None
    sd = waarden.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return None
    t = waarden.mean() / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(waarden.mean()), float(t), float(p)


def cross_sectionele_ic(df: pd.DataFrame, kolom: str, ret_kolom: str,
                        min_per_datum: int) -> Optional[dict]:
    """Rekenkundig gemiddelde van per-datum cross-sectionele Spearman IC's,
    met t-toets op de reeks IC's. Dit is de standaard rank-IC methodiek."""
    ics = []
    n_obs = 0
    for _, g in df.groupby("datum"):
        sub = g[[kolom, ret_kolom]].dropna()
        if len(sub) < min_per_datum or sub[kolom].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(sub[kolom], sub[ret_kolom])
        if not np.isnan(rho):
            ics.append(rho)
            n_obs += len(sub)
    if len(ics) < MIN_DAGEN:
        return None
    toets = _t_toets_op_reeks(np.array(ics))
    if toets is None:
        return None
    gem, t, p = toets
    return {"coefficient": gem, "t_stat": t, "p_waarde": p,
            "n_dagen": len(ics), "n_obs": n_obs}


def cross_sectioneel_boolean_verschil(df: pd.DataFrame, kolom: str, ret_kolom: str,
                                      min_per_datum: int) -> Optional[dict]:
    """Per datum mediaan(True) - mediaan(False), dan t-toets op de reeks
    verschillen. Robuust tegen scheve rendementen en tegen regime-effect."""
    diffs = []
    n_obs = 0
    for _, g in df.groupby("datum"):
        sub = g[[kolom, ret_kolom]].dropna()
        sub = sub[sub[kolom].notna()]
        waar = sub[sub[kolom].astype(bool)][ret_kolom]
        onwaar = sub[~sub[kolom].astype(bool)][ret_kolom]
        if len(waar) < 3 or len(onwaar) < 3:
            continue
        diffs.append(waar.median() - onwaar.median())
        n_obs += len(sub)
    if len(diffs) < MIN_DAGEN:
        return None
    toets = _t_toets_op_reeks(np.array(diffs))
    if toets is None:
        return None
    gem, t, p = toets
    return {"coefficient": gem, "t_stat": t, "p_waarde": p,
            "n_dagen": len(diffs), "n_obs": n_obs}


def bereken_correlaties(df: pd.DataFrame, numeriek: List[str], boolean: List[str],
                        min_n: int, min_per_datum: int) -> pd.DataFrame:
    resultaten = []
    for horizon in HORIZONS:
        ret_kolom = f"fwd_ret_{horizon}d"
        if ret_kolom not in df.columns:
            print(f"  (horizon {horizon}d overgeslagen: {ret_kolom} niet aanwezig)")
            continue
        df_h = df[df[ret_kolom].notna()].copy()
        if df_h.empty:
            continue

        per_horizon = []

        for kolom in numeriek:
            if kolom not in df_h.columns:
                continue
            if df_h[kolom].dropna().shape[0] < min_n:
                continue
            res = cross_sectionele_ic(df_h, kolom, ret_kolom, min_per_datum)
            if res is None:
                continue
            res.update({
                "horizon": horizon, "parameter": kolom, "type": "numeriek",
                "n_strategieen": int(df_h[df_h[kolom].notna()]["strategie"].nunique()),
            })
            per_horizon.append(res)

        for kolom in boolean:
            if kolom not in df_h.columns:
                continue
            if df_h[kolom].dropna().shape[0] < min_n:
                continue
            res = cross_sectioneel_boolean_verschil(df_h, kolom, ret_kolom, min_per_datum)
            if res is None:
                continue
            res.update({
                "horizon": horizon, "parameter": kolom, "type": "boolean",
                "n_strategieen": int(df_h[df_h[kolom].notna()]["strategie"].nunique()),
            })
            per_horizon.append(res)

        if not per_horizon:
            continue
        p_waarden = [r["p_waarde"] for r in per_horizon]
        significant_mask = bh_correctie(p_waarden, FDR_ALPHA)
        for r, sig in zip(per_horizon, significant_mask):
            r["significant_fdr"] = sig
        resultaten.extend(per_horizon)

    return pd.DataFrame(resultaten)


def print_rapport(stats_df: pd.DataFrame, horizon: int) -> str:
    sub = stats_df[stats_df["horizon"] == horizon].copy()
    if sub.empty:
        return f"\nGeen resultaten voor horizon {horizon}d."
    sub["abs_coef"] = sub["coefficient"].abs()
    sub = sub.sort_values("abs_coef", ascending=False)

    markering = "  ★ PRIMAIR" if horizon == PRIMAIRE_HORIZON else ""
    regels = [
        f"\n{'=' * 118}",
        f"HORIZON: {horizon} handelsdagen{markering}  —  cross-sectionele IC per datum "
        f"(FDR alpha={FDR_ALPHA}, {len(sub)} parameters getest)",
        "=" * 118,
        f"{'parameter':<26}{'type':<10}{'#dagen':>8}{'#obs':>8}{'#strat':>8}"
        f"{'IC / Δmediaan':>16}{'t-stat':>9}{'p':>10}  sig?  opm",
    ]
    for _, r in sub.iterrows():
        vlag = "✓" if r["significant_fdr"] else " "
        label = "ρ" if r["type"] == "numeriek" else "Δ%"
        opm = "⚠ 1 strat" if r["n_strategieen"] <= 1 else ""
        regels.append(
            f"{r['parameter']:<26}{r['type']:<10}{int(r['n_dagen']):>8}{int(r['n_obs']):>8}"
            f"{int(r['n_strategieen']):>8}{r['coefficient']:>13.4f} {label:>2}"
            f"{r['t_stat']:>9.2f}{r['p_waarde']:>10.4f}  {vlag}    {opm}"
        )
    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def print_decay_curve(stats_df: pd.DataFrame) -> str:
    """Voor elke parameter die op minstens één horizon significant is,
    toon de IC over alle horizonten. Laat zien of het signaal soepel
    afneemt (bruikbaar) of omslaat van teken (verdacht / horizon-specifiek)."""
    if stats_df.empty:
        return ""
    sig_params = stats_df.loc[stats_df["significant_fdr"], "parameter"].unique()
    if len(sig_params) == 0:
        return "\nGeen enkele parameter significant op eender welke horizon; geen decay-curve."

    regels = [
        f"\n{'=' * 118}",
        "DECAY-CURVE — IC/Δmediaan per horizon voor elke significante parameter",
        "=" * 118,
        f"{'parameter':<26}{'type':<10}" + "".join(f"{h:>10}d" for h in HORIZONS),
    ]
    for param in sorted(sig_params):
        ptype = stats_df.loc[stats_df["parameter"] == param, "type"].iloc[0]
        row = [f"{param:<26}{ptype:<10}"]
        for h in HORIZONS:
            match = stats_df[(stats_df["parameter"] == param) & (stats_df["horizon"] == h)]
            if match.empty:
                row.append(f"{'—':>11}")
            else:
                c = match["coefficient"].iloc[0]
                ster = "*" if match["significant_fdr"].iloc[0] else " "
                row.append(f"{c:>+10.4f}{ster}")
        regels.append("".join(row))
    regels.append("(* = significant na BH-correctie op die horizon)")
    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=50,
                        help="minimum aantal niet-lege observaties per parameter/horizon")
    parser.add_argument("--min-per-datum", type=int, default=DEFAULT_MIN_PER_DATUM,
                        help="minimum aantal aandelen per datum voor een cross-sectionele IC")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        numeriek, boolean = haal_feature_kolommen(conn)
        numeriek = sorted(set(numeriek) | set(GENERIEKE_TECHNICALS_KOLOMMEN))
        print(f"{len(numeriek)} numerieke kolommen (incl. generieke_technicals) + {len(boolean)} boolean-kolommen.")
        print(f"Numeriek: {', '.join(numeriek)}")
        print(f"Boolean: {', '.join(boolean)}\n")

        alle_kolommen = numeriek + boolean
        df = haal_gelabelde_data(conn, alle_kolommen)
        print(f"{len(df)} rijen opgehaald (gepoold over alle strategieën).\n")

        if df.empty:
            print("Nog geen gelabelde data beschikbaar.")
            return

        stats_df = bereken_correlaties(df, numeriek, boolean,
                                       args.min_n, args.min_per_datum)
        if stats_df.empty:
            print(f"Geen enkele parameter haalt de drempels (min_n={args.min_n}, "
                  f"min_per_datum={args.min_per_datum}, min_dagen={MIN_DAGEN}).")
            return

        telegram_delen = []
        for horizon in HORIZONS:
            if horizon in stats_df["horizon"].values:
                telegram_delen.append(print_rapport(stats_df, horizon))
        telegram_delen.append(print_decay_curve(stats_df))

        # Samenvatting op basis van primaire horizon
        prim = stats_df[stats_df["horizon"] == PRIMAIRE_HORIZON]
        aantal_sig = int(prim["significant_fdr"].sum()) if not prim.empty else 0
        totaal = len(prim)
        top_sig = prim[prim["significant_fdr"]] \
            .assign(abs_coef=lambda d: d["coefficient"].abs()) \
            .sort_values("abs_coef", ascending=False)
        top_namen = ", ".join(top_sig["parameter"].head(5)) if not top_sig.empty else "geen"

        # Tel ook significantie op andere horizonten, zodat je weet of er
        # signaal is dat enkel op lange of korte termijn opduikt.
        extra = []
        for h in HORIZONS:
            if h == PRIMAIRE_HORIZON:
                continue
            sh = stats_df[stats_df["horizon"] == h]
            if sh.empty:
                continue
            extra.append(f"{h}d: {int(sh['significant_fdr'].sum())}/{len(sh)}")
        extra_txt = " | ".join(extra)

        samenvatting = (
            f"Parameter-correlatie analyse (cross-sectionele IC)\n\n"
            f"Primair ({PRIMAIRE_HORIZON}d): {aantal_sig}/{totaal} significant na BH "
            f"(alpha={FDR_ALPHA})\n"
            f"Sterkste: {top_namen}\n\n"
            f"Andere horizonten: {extra_txt}\n\n"
            "Volledige tabellen + decay-curve: zie workflow-log."
        )
        send_telegram(samenvatting)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
