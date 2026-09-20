#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_parameter_correlatie.py  —  WELKE PARAMETER ZEGT IETS?  v1.0

Andere invalshoek dan analyse_forward_returns.py: in plaats van per
STRATEGIE te middelen (waarbij het effect van een goede parameterwaarde
vermengd raakt met alles wat toevallig nog in die strategie zit), wordt
hier per INDIVIDUELE SELECTIE (elke rij = 1 aandeel op 1 moment, gepoold
over ALLE ~20 strategieën heen) gekeken of de ruwe waarde van elke
parameter (ATR%, RSI, afstand tot MA50/MA200, kwaliteitsscores, ...)
samenhangt met het nadien behaalde rendement.

Dit isoleert het effect van de parameter zelf, los van welke bot 'm
gebruikte -- en is de logische voorbereidende stap voor het geplande
meta-model: hier zie je welke features OP ZICHZELF al iets zeggen.

METHODE
=======
- Kolommen worden dynamisch opgehaald uit information_schema.columns van
  de `selecties`-tabel (geen hardcoded lijst, blijft dus correct als er
  later nieuwe kolommen bijkomen via een migratie).
- Numerieke kolommen (double precision/integer/numeric/real/bigint):
  Spearman-correlatie met fwd_ret_Nd (robuuster dan Pearson tegen
  scheve verdelingen/outliers, zoals we al zagen bij bot_00vcp's
  ATR-gerelateerde extreme waarden).
- Boolean-kolommen (bv. breakout, stage2, fcf_growing, fcf_consistent):
  Mann-Whitney U-test (True-groep vs False-groep), robuuster dan een
  gewone t-test voor scheve rendementsverdelingen.
- Identificatie-/metadatakolommen (id, ticker, datum, strategie, beurs,
  koers, parameters, grafiek, tekst-labels als rsi_label/macd_label)
  worden overgeslagen -- geen zinvolle correlatie met een numerieke
  waarde, tekst-categorieën zijn een aparte analyse.
- Benjamini-Hochberg FDR-correctie over ALLE (parameter, horizon)-testen
  SAMEN per horizon (dus als er 50 parameters getest worden op 20d, is
  dat 50 gelijktijdige testen die gecorrigeerd worden) -- zelfde
  discipline als analyse_forward_returns.py en bot_01repititief.py.
- Enkel parameters met minstens MIN_N niet-lege waarnemingen worden
  meegenomen (een kolom die maar door 1 strategie gevuld wordt heeft per
  definitie een kleinere n dan bv. 'score', wat oneerlijke vergelijkingen
  zou geven bij te weinig data).

GEBRUIK
=======
  python analyse_parameter_correlatie.py [--min-n 50] [--horizon 20]

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel), FDR_ALPHA (default 0.05)
"""

import os
import sys
import argparse
from typing import List, Tuple

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

# Kolommen die geen kandidaat-feature zijn, ook al staan ze in selecties
UITGESLOTEN_KOLOMMEN = {
    "id", "ticker", "datum", "strategie", "beurs", "koers",
    "parameters", "grafiek", "rsi_label", "macd_label",
}
NUMERIEKE_TYPES = {"double precision", "integer", "numeric", "real", "bigint", "smallint"}
BOOLEAN_TYPES = {"boolean"}


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


def bh_correctie(p_waarden: List[float], alpha: float) -> List[bool]:
    """Zelfde implementatie als analyse_forward_returns.py / bot_01repititief.py."""
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
    """Geeft (numerieke_kolommen, boolean_kolommen) terug, dynamisch opgehaald."""
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


GENERIEKE_TECHNICALS_KOLOMMEN = [
    "atr14", "atr14_pct", "rsi14", "ibs", "ma50", "ma200",
    "pct_from_ma50", "pct_from_ma200", "vol_ratio_20d", "high52w", "pct_from_high52w",
]


def haal_gelabelde_data(conn, kolommen: List[str]) -> pd.DataFrame:
    # generieke_technicals-kolommen apart houden: die komen uit een eigen
    # tabel (g.), niet uit selecties (s.) -- en zijn per definitie voor
    # ELKE strategie ingevuld, dus n_strategieen zal daar altijd het
    # volledige aantal actieve strategieën tonen i.p.v. "toevallig 1 bot".
    s_kolommen = [k for k in kolommen if k not in GENERIEKE_TECHNICALS_KOLOMMEN]
    g_kolommen = [k for k in kolommen if k in GENERIEKE_TECHNICALS_KOLOMMEN]

    s_lijst = ", ".join(f"s.{k}" for k in s_kolommen)
    g_lijst = ("," + ", ".join(f"g.{k}" for k in g_kolommen)) if g_kolommen else ""

    query = f"""
        SELECT s.strategie, {s_lijst}{g_lijst}, f.fwd_ret_5d, f.fwd_ret_10d, f.fwd_ret_20d
        FROM selecties s
        JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        LEFT JOIN generieke_technicals g
          ON s.ticker = g.ticker AND s.datum = g.datum
        WHERE f.fwd_ret_5d IS NOT NULL OR f.fwd_ret_10d IS NOT NULL OR f.fwd_ret_20d IS NOT NULL;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def bereken_correlaties(df: pd.DataFrame, numeriek: List[str], boolean: List[str], min_n: int) -> pd.DataFrame:
    resultaten = []
    for horizon in HORIZONS:
        ret_kolom = f"fwd_ret_{horizon}d"
        if ret_kolom not in df.columns:
            continue
        per_horizon = []

        for kolom in numeriek:
            if kolom not in df.columns:
                continue
            sub = df[["strategie", kolom, ret_kolom]].dropna()
            n = len(sub)
            if n < min_n or sub[kolom].nunique() < 2:
                continue
            rho, p = stats.spearmanr(sub[kolom], sub[ret_kolom])
            if np.isnan(rho):
                continue
            n_strat = sub["strategie"].nunique()
            per_horizon.append({
                "horizon": horizon, "parameter": kolom, "type": "numeriek",
                "n": n, "n_strategieen": n_strat, "coefficient": round(rho, 3), "p_waarde": p,
            })

        for kolom in boolean:
            if kolom not in df.columns:
                continue
            sub = df[["strategie", kolom, ret_kolom]].dropna()
            n = len(sub)
            groep_waar = sub[sub[kolom] == True][ret_kolom]
            groep_onwaar = sub[sub[kolom] == False][ret_kolom]
            if n < min_n or len(groep_waar) < 10 or len(groep_onwaar) < 10:
                continue
            u_stat, p = stats.mannwhitneyu(groep_waar, groep_onwaar, alternative="two-sided")
            verschil = round(groep_waar.median() - groep_onwaar.median(), 2)
            n_strat = sub["strategie"].nunique()
            per_horizon.append({
                "horizon": horizon, "parameter": kolom, "type": "boolean",
                "n": n, "n_strategieen": n_strat, "coefficient": verschil, "p_waarde": p,
            })

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
    sub["abs_coef"] = sub["coefficient"].abs()
    sub = sub.sort_values("abs_coef", ascending=False)

    regels = [f"\n{'=' * 105}", f"HORIZON: {horizon} handelsdagen — gepoold over ALLE strategieën "
              f"(FDR alpha={FDR_ALPHA}, {len(sub)} parameters getest)", "=" * 105]
    regels.append(f"{'parameter':<28}{'type':<11}{'n':>7}{'#strat':>8}{'coëfficiënt/verschil':>22}{'p-waarde':>12}  significant?  let op")
    for _, r in sub.iterrows():
        vlag = "✓ JA" if r["significant_fdr"] else "  nee"
        label = "spearman ρ" if r["type"] == "numeriek" else "Δmediaan%"
        waarschuwing = "⚠️ feitelijk 1 strategie" if r["n_strategieen"] <= 1 else ""
        regels.append(
            f"{r['parameter']:<28}{r['type']:<11}{r['n']:>7}{r['n_strategieen']:>8}"
            f"{r['coefficient']:>15.3f} ({label}){r['p_waarde']:>12.4f}  {vlag}  {waarschuwing}"
        )
    tekst = "\n".join(regels)
    print(tekst)
    return tekst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=50, help="minimum aantal observaties per parameter/horizon")
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
        print(f"{len(df)} gelabelde rijen opgehaald (gepoold over alle strategieën).\n")

        if df.empty:
            print("Nog geen gelabelde data beschikbaar.")
            return

        stats_df = bereken_correlaties(df, numeriek, boolean, args.min_n)
        if stats_df.empty:
            print(f"Geen enkele parameter haalt min_n={args.min_n} op een horizon.")
            return

        telegram_delen = []
        for horizon in HORIZONS:
            tekst = print_rapport(stats_df, horizon)
            telegram_delen.append(tekst)

        aantal_sig_20d = stats_df[(stats_df["horizon"] == 20) & (stats_df["significant_fdr"])].shape[0]
        totaal_20d = stats_df[stats_df["horizon"] == 20].shape[0]
        top_sig = stats_df[(stats_df["horizon"] == 20) & (stats_df["significant_fdr"])] \
            .assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)
        top_namen = ", ".join(top_sig["parameter"].head(5)) if not top_sig.empty else "geen"

        samenvatting = (
            f"📊 *Parameter-correlatie analyse*\n\n"
            f"Op 20d-horizon: {aantal_sig_20d}/{totaal_20d} parameters significant "
            f"na BH-correctie (alpha={FDR_ALPHA})\n"
            f"Sterkste: {top_namen}\n\n"
            "Volledige tabel: zie workflow-log."
        )
        send_telegram(samenvatting)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
