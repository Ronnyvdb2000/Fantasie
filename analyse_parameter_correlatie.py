#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_parameter_correlatie.py  —  WELKE PARAMETER ZEGT IETS?  v2.3

Cross-sectionele IC per datum, t-toets op de reeks IC's. Horizonten
5/10/20/30/60d, primair = 10d.

v2.3: --use-whitelist toegevoegd. Beperkt de analyse tot de features uit
features.ANALYSE_WHITELIST.

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

from features import ANALYSE_WHITELIST

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FDR_ALPHA = float(os.environ.get("FDR_ALPHA", "0.05"))

HORIZONS = [5, 10, 20, 30, 60]
PRIMAIRE_HORIZON = 10
DEFAULT_MIN_DAGEN = 10
DEFAULT_MIN_PER_DATUM = 15
DEFAULT_MIN_STRATEGIEEN = 1

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
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst},
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


def _t_toets_op_reeks(waarden: np.ndarray, min_dagen: int) -> Optional[Tuple[float, float, float]]:
    n = len(waarden)
    if n < min_dagen:
        return None
    sd = waarden.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return None
    t = waarden.mean() / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(waarden.mean()), float(t), float(p)


def cross_sectionele_ic(df: pd.DataFrame, kolom: str, ret_kolom: str,
                        min_per_datum: int, min_dagen: int) -> Optional[dict]:
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
    if len(ics) < min_dagen:
        return None
    toets = _t_toets_op_reeks(np.array(ics), min_dagen)
    if toets is None:
        return None
    gem, t, p = toets
    return {"coefficient": gem, "t_stat": t, "p_waarde": p,
            "n_dagen": len(ics), "n_obs": n_obs}


def cross_sectioneel_boolean_verschil(df: pd.DataFrame, kolom: str, ret_kolom: str,
                                      min_per_datum: int, min_dagen: int) -> Optional[dict]:
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
    if len(diffs) < min_dagen:
        return None
    toets = _t_toets_op_reeks(np.array(diffs), min_dagen)
    if toets is None:
        return None
    gem, t, p = toets
    return {"coefficient": gem, "t_stat": t, "p_waarde": p,
            "n_dagen": len(diffs), "n_obs": n_obs}


def bereken_correlaties(df: pd.DataFrame, numeriek: List[str], boolean: List[str],
                        min_n: int, min_per_datum: int, min_dagen: int,
                        min_strategieen: int) -> pd.DataFrame:
    resultaten = []
    overgeslagen = []
    genegeerd_weinig_strat = []

    for horizon in HORIZONS:
        ret_kolom = f"fwd_ret_{horizon}d"
        if ret_kolom not in df.columns:
            print(f"  horizon {horizon}d: kolom {ret_kolom} ontbreekt -- overgeslagen")
            overgeslagen.append(horizon)
            continue
        df_h = df[df[ret_kolom].notna()].copy()
        if df_h.empty:
            print(f"  horizon {horizon}d: 0 rijen met niet-lege {ret_kolom} -- overgeslagen")
            overgeslagen.append(horizon)
            continue

        n_datums = df_h["datum"].nunique()
        print(f"  horizon {horizon}d: {len(df_h)} rijen, {n_datums} unieke datums")

        per_horizon = []

        for kolom in numeriek + boolean:
            if kolom not in df_h.columns:
                continue
            niet_leeg = df_h[df_h[kolom].notna()]
            if niet_leeg.shape[0] < min_n:
                continue

            n_strat = int(niet_leeg["strategie"].nunique())
            if n_strat < min_strategieen:
                genegeerd_weinig_strat.append((horizon, kolom, n_strat))
                continue

            if kolom in numeriek:
                res = cross_sectionele_ic(df_h, kolom, ret_kolom, min_per_datum, min_dagen)
                soort = "numeriek"
            else:
                res = cross_sectioneel_boolean_verschil(df_h, kolom, ret_kolom, min_per_datum, min_dagen)
                soort = "boolean"
            if res is None:
                continue

            res.update({
                "horizon": horizon, "parameter": kolom, "type": soort,
                "n_strategieen": n_strat,
            })
            per_horizon.append(res)

        if not per_horizon:
            print(f"  horizon {horizon}d: geen parameter haalt de drempels -- overgeslagen")
            overgeslagen.append(horizon)
            continue

        p_waarden = [r["p_waarde"] for r in per_horizon]
        significant_mask = bh_correctie(p_waarden, FDR_ALPHA)
        for r, sig in zip(per_horizon, significant_mask):
            r["significant_fdr"] = sig
        resultaten.extend(per_horizon)

    if genegeerd_weinig_strat:
        unieke = sorted({(p, n) for _, p, n in genegeerd_weinig_strat})
        print(f"\nParameters overgeslagen wegens < {min_strategieen} strategieën "
              f"({len(unieke)} unieke parameter(s)):")
        for naam, n in unieke:
            print(f"  {naam:<30} max {n} strategie(ën)")
    if overgeslagen:
        print(f"\nHorizonten zonder resultaten: {overgeslagen}")
    return pd.DataFrame(resultaten)


def print_rapport(stats_df: pd.DataFrame, horizon: int,
                  min_dagen: int, min_strategieen: int) -> str:
    sub = stats_df[stats_df["horizon"] == horizon].copy()
    if sub.empty:
        return f"\nGeen resultaten voor horizon {horizon}d."
    sub["abs_coef"] = sub["coefficient"].abs()
    sub = sub.sort_values("abs_coef", ascending=False)

    markering = "  * PRIMAIR" if horizon == PRIMAIRE_HORIZON else ""
    waarschuw = ""
    if min_dagen < DEFAULT_MIN_DAGEN:
        waarschuw += f" [min_dagen={min_dagen} < {DEFAULT_MIN_DAGEN}]"

    regels = [
        f"\n{'=' * 118}",
        f"HORIZON: {horizon} handelsdagen{markering}  --  cross-sectionele IC per datum"
        f" (FDR alpha={FDR_ALPHA}, min_dagen={min_dagen}, min_strat={min_strategieen},"
        f" {len(sub)} parameters getest){waarschuw}",
        "=" * 118,
        f"{'parameter':<26}{'type':<10}{'#dagen':>8}{'#obs':>8}{'#strat':>8}"
        f"{'IC / dmediaan':>16}{'t-stat':>9}{'p':>10}  sig?",
    ]
    for _, r in sub.iterrows():
        vlag = "JA" if r["significant_fdr"] else "  "
        label = "rho" if r["type"] == "numeriek" else "d%"
        regels.append(
            f"{r['parameter']:<26}{r['type']:<10}{int(r['n_dagen']):>8}{int(r['n_obs']):>8}"
            f"{int(r['n_strategieen']):>8}{r['coefficient']:>13.4f} {label:>2}"
            f"{r['t_stat']:>9.2f}{r['p_waarde']:>10.4f}  {vlag}"
        )
    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def print_decay_curve(stats_df: pd.DataFrame) -> str:
    if stats_df.empty:
        return ""
    sig_params = stats_df.loc[stats_df["significant_fdr"], "parameter"].unique()
    if len(sig_params) == 0:
        return "\nGeen enkele parameter significant op eender welke horizon; geen decay-curve."

    regels = [
        f"\n{'=' * 118}",
        "DECAY-CURVE -- IC/dmediaan per horizon voor elke significante parameter",
        "=" * 118,
        f"{'parameter':<26}{'type':<10}{'#strat':>8}" + "".join(f"{h:>10}d" for h in HORIZONS),
    ]
    for param in sorted(sig_params):
        ptype = stats_df.loc[stats_df["parameter"] == param, "type"].iloc[0]
        max_strat = int(stats_df.loc[stats_df["parameter"] == param, "n_strategieen"].max())
        row = [f"{param:<26}{ptype:<10}{max_strat:>8}"]
        for h in HORIZONS:
            match = stats_df[(stats_df["parameter"] == param) & (stats_df["horizon"] == h)]
            if match.empty:
                row.append(f"{'--':>11}")
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
    parser.add_argument("--min-dagen", type=int, default=DEFAULT_MIN_DAGEN,
                        help=f"minimum aantal datums voor een geldige t-toets (default {DEFAULT_MIN_DAGEN})")
    parser.add_argument("--min-strategieen", type=int, default=DEFAULT_MIN_STRATEGIEEN,
                        help=f"minimum aantal strategieën dat een parameter vult (default "
                             f"{DEFAULT_MIN_STRATEGIEEN}; gebruik 3+ voor een generiek filter-model)")
    parser.add_argument("--use-whitelist", action="store_true",
                        help="Beperk de analyse tot de features in features.ANALYSE_WHITELIST")
    args = parser.parse_args()

    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    if args.min_dagen < DEFAULT_MIN_DAGEN:
        print(f"LET OP: --min-dagen={args.min_dagen} < {DEFAULT_MIN_DAGEN}. "
              f"Resultaten zijn indicatief, niet betrouwbaar voor filtering.\n")
    if args.min_strategieen > 1:
        print(f"Filter actief: parameters met < {args.min_strategieen} strategieën "
              f"worden volledig overgeslagen.\n")

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        numeriek, boolean = haal_feature_kolommen(conn)
        numeriek = sorted(set(numeriek) | set(GENERIEKE_TECHNICALS_KOLOMMEN))

        if args.use_whitelist:
            numeriek = [k for k in numeriek if k in ANALYSE_WHITELIST]
            boolean = [k for k in boolean if k in ANALYSE_WHITELIST]
            print(f"Whitelist actief: {len(numeriek)} numerieke + {len(boolean)} boolean features.")
            print(f"Features: {', '.join(numeriek + boolean)}\n")

        print(f"{len(numeriek)} numerieke kolommen + {len(boolean)} boolean-kolommen.\n")

        alle_kolommen = numeriek + boolean
        df = haal_gelabelde_data(conn, alle_kolommen)
        print(f"{len(df)} rijen opgehaald.\n")

        if df.empty:
            print("Nog geen gelabelde data beschikbaar.")
            return

        print("Data per horizon:")
        stats_df = bereken_correlaties(df, numeriek, boolean,
                                       args.min_n, args.min_per_datum, args.min_dagen,
                                       args.min_strategieen)
        if stats_df.empty:
            print(f"\nGeen enkele parameter haalt de drempels "
                  f"(min_n={args.min_n}, min_per_datum={args.min_per_datum}, "
                  f"min_dagen={args.min_dagen}, min_strategieen={args.min_strategieen}).")
            return

        for horizon in HORIZONS:
            if horizon in stats_df["horizon"].values:
                print_rapport(stats_df, horizon, args.min_dagen, args.min_strategieen)
        print_decay_curve(stats_df)

        prim = stats_df[stats_df["horizon"] == PRIMAIRE_HORIZON]
        aantal_sig = int(prim["significant_fdr"].sum()) if not prim.empty else 0
        totaal = len(prim)
        top_sig = prim[prim["significant_fdr"]] \
            .assign(abs_coef=lambda d: d["coefficient"].abs()) \
            .sort_values("abs_coef", ascending=False)
        top_namen = ", ".join(top_sig["parameter"].head(5)) if not top_sig.empty else "geen"

        extra = []
        for h in HORIZONS:
            if h == PRIMAIRE_HORIZON:
                continue
            sh = stats_df[stats_df["horizon"] == h]
            if sh.empty:
                continue
            extra.append(f"{h}d: {int(sh['significant_fdr'].sum())}/{len(sh)}")
        extra_txt = " | ".join(extra) if extra else "geen"

        waarschuwing = ""
        if args.min_dagen < DEFAULT_MIN_DAGEN:
            waarschuwing += f"\n!! min_dagen={args.min_dagen} < {DEFAULT_MIN_DAGEN} -- indicatief\n"

        samenvatting = (
            f"Parameter-correlatie analyse (cross-sectionele IC)\n"
            f"min_strategieen={args.min_strategieen}, min_dagen={args.min_dagen}\n"
            f"whitelist={'ja' if args.use_whitelist else 'nee'}\n"
            f"{waarschuwing}\n"
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
