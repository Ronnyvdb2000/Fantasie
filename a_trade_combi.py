#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_combi_volatiel.py — GEWOGEN KWALITEITSSCORE x VOLATILITEIT  (v6)

Wijzigingen t.o.v. v5:
  - DYNAMISCHE GEWICHTEN: de hardcoded GEWICHTEN-tabel wordt bij elke run
    vervangen door een berekening uit `forward_returns`. Per strategie
    wordt de Sharpe-ratio (gemiddeld rendement / standaardafwijking)
    berekend op fwd_ret_10d en genormaliseerd naar [0,1]. Als een
    strategie minder dan MIN_OBSERVATIES picks heeft, valt die terug op
    het hardcoded gewicht. Als de DB-query volledig faalt, gebruikt de
    hele run de hardcoded fallback.
  - Alle overige logica (opwarm-retry, trechter, retry-backoff, ATR-filter,
    rapportage) is ONGEWIJZIGD t.o.v. v5.

Fix (2026-10-08): hoogl.analyse_ticker() retourneert sinds de rate-limit-
backoff een tuple (signaal, was_rate_limited). selecties_hoogl() verwachtte
nog een los signaal en crashte met "AttributeError: 'tuple' object has no
attribute 'score'". Opgelost via _hoogl_analyse_met_retry(). kasstr en fisher
zijn ongewijzigd (hun analyse_ticker is niet nagekeken).

Ratio: wegen op Sharpe i.p.v. ruwe return compenseert voor strategieën
met veel variantie (zoals bot_00oshaughnessy met n=24 en gem +1.54% maar
grote uitschieters).

Fallback-gewichten (identiek aan v5, afgeleid van ruwe gemiddelde returns):
    kasstr 0.85 | fisher 0.55 | vcp 0.55 | hoogl 0.50 | repititief 0.15

Als je eenmalig wilt forceren op de fallback, zet FORCE_STATIC_GEWICHTEN=1.

De rest van de architectuur (trechter, retry-backoff, opwarm-stap) blijft
zoals in v5 -- zie die docstring in Git voor de volledige onderbouwing.
"""

import os
import time
from typing import Dict, List, Set, Tuple

import psycopg2
import psycopg2.extras

import bot_00kr as kr
import bot_01kasstr as kasstr
import bot_00Fisher as fisher
import bot_01repititief as repititief
import bot_00vcp as vcp
import bot_01hoogl as hoogl
import db_logger

# ============================================================
# CONFIG
# ============================================================

MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "4.0"))
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "25.0"))
TOP_N       = int(os.getenv("TOP_N", "10"))

RETRY_POGINGEN     = int(os.getenv("RETRY_POGINGEN", "3"))
RETRY_BASIS_WACHT  = float(os.getenv("RETRY_BASIS_WACHT", "8"))

WARMUP_POGINGEN    = int(os.getenv("WARMUP_POGINGEN", "6"))
WARMUP_WACHT       = float(os.getenv("WARMUP_WACHT", "60"))

# Statische fallback-gewichten (identiek aan v5)
GEWICHTEN_FALLBACK = {
    "kasstr":     0.85,
    "fisher":     0.55,
    "vcp":        0.55,
    "hoogl":      0.50,
    "repititief": 0.15,
}

# Dynamische gewicht-berekening
FORCE_STATIC_GEWICHTEN = os.getenv("FORCE_STATIC_GEWICHTEN", "0") == "1"
MIN_OBSERVATIES        = int(os.getenv("MIN_OBSERVATIES", "30"))
GEWICHT_HORIZON        = os.getenv("GEWICHT_HORIZON", "10d")  # 10d / 30d / 60d
GEWICHTEN_MIN          = 0.05   # ondergrens na normalisatie
GEWICHTEN_MAX          = 1.00   # bovengrens na normalisatie

STRATEGIE_LABELS = {
    "kasstr": "bot_01kasstr", "fisher": "bot_00Fisher",
    "vcp": "bot_00vcp", "hoogl": "bot_01hoogl", "repititief": "bot_01repititief",
}

# Mapping van interne key -> DB-strategienaam in forward_returns
STRATEGIE_DB_NAAM = {
    "kasstr":     "bot_01kasstr",
    "fisher":     "bot_00Fisher",
    "vcp":        "bot_00vcp",
    "hoogl":      "bot_01hoogl",
    "repititief": "bot_01repititief",
}

LIVE_FETCH_STRATEGIEEN = {"kasstr", "fisher", "hoogl"}


# ============================================================
# HULPFUNCTIES
# ============================================================

def bouw_exchange_tickers() -> Tuple[Dict[str, List[str]], List[str]]:
    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []
    for f_name in kr.bouw_bestandslijst():
        tlist = kr.load_tickers_from_file(f_name)
        if not tlist:
            continue
        ex_name = kr.label_voor(f_name)
        exchange_tickers[ex_name] = tlist
        all_tickers.extend(tlist)
    all_tickers = sorted(set(all_tickers))
    return exchange_tickers, all_tickers


def _yahoo_link(ticker: str) -> str:
    return kr._yahoo_link(ticker)


def met_retry(fn, ticker: str, label: str):
    for poging in range(RETRY_POGINGEN):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "Too Many Requests" in str(e) or "Rate limited" in str(e)
            laatste_poging = poging == RETRY_POGINGEN - 1
            if is_rate_limit and not laatste_poging:
                wacht = RETRY_BASIS_WACHT * (2 ** poging)
                print(f"  [retry] {label} {ticker}: rate limited, wacht {wacht:.0f}s (poging {poging+1}/{RETRY_POGINGEN})...")
                time.sleep(wacht)
                continue
            if not is_rate_limit:
                print(f"  [WARN] {label} {ticker}: fout — {e}")
            return None
    return None


def warm_up_yfinance() -> bool:
    for poging in range(WARMUP_POGINGEN):
        try:
            test = kr.download_history(["AAPL"], period="5d")
            if test is not None and not test.empty:
                print(f"[opwarm] Yahoo-sessie actief na poging {poging+1}/{WARMUP_POGINGEN}")
                return True
        except Exception as e:
            print(f"[opwarm] poging {poging+1}/{WARMUP_POGINGEN} gaf fout: {e}")
        if poging < WARMUP_POGINGEN - 1:
            print(f"[opwarm] geen data, wacht {WARMUP_WACHT:.0f}s voor volgende poging...")
            time.sleep(WARMUP_WACHT)
    print("[opwarm] Yahoo blijft ontoegankelijk na alle opwarm-pogingen — run gaat toch door.")
    return False


# ============================================================
# DYNAMISCHE GEWICHTEN
# ============================================================

def herbereken_gewichten() -> Dict[str, float]:
    """
    Berekent gewichten uit forward_returns via Sharpe-ratio per strategie.
    Fallback naar GEWICHTEN_FALLBACK bij onvoldoende data of DB-fout.
    """
    if FORCE_STATIC_GEWICHTEN:
        print("[gewichten] FORCE_STATIC_GEWICHTEN=1 — statische fallback gebruikt.")
        return dict(GEWICHTEN_FALLBACK)

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("[gewichten] SUPABASE_DB_URL ontbreekt — statische fallback gebruikt.")
        return dict(GEWICHTEN_FALLBACK)

    target_kolom = f"fwd_ret_{GEWICHT_HORIZON}"
    db_namen = list(STRATEGIE_DB_NAAM.values())

    query = f"""
        SELECT
            strategie,
            COUNT(*)              AS n,
            AVG({target_kolom})   AS gem,
            STDDEV({target_kolom}) AS std
        FROM forward_returns
        WHERE {target_kolom} IS NOT NULL
          AND strategie = ANY(%(strats)s)
        GROUP BY strategie;
    """

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, {"strats": db_namen})
                rows = cur.fetchall()
    except Exception as e:
        print(f"[gewichten] DB-query faalde: {e} — statische fallback gebruikt.")
        return dict(GEWICHTEN_FALLBACK)

    # Mapping db_naam -> interne key
    db_naar_key = {v: k for k, v in STRATEGIE_DB_NAAM.items()}

    # Sharpe per strategie
    sharpe_per_key: Dict[str, float] = {}
    n_per_key: Dict[str, int] = {}

    for r in rows:
        db_naam = r["strategie"]
        key = db_naar_key.get(db_naam)
        if key is None:
            continue
        n = int(r["n"] or 0)
        n_per_key[key] = n
        if n < MIN_OBSERVATIES:
            continue
        gem = float(r["gem"]) if r["gem"] is not None else 0.0
        std = float(r["std"]) if r["std"] is not None else 0.0
        # Sharpe met kleine-deler-epsilon om deling-door-nul te vermijden
        sharpe = gem / std if std > 1e-9 else 0.0
        sharpe_per_key[key] = sharpe

    print()
    print("=" * 70)
    print(f"DYNAMISCHE GEWICHTEN (horizon={GEWICHT_HORIZON}, "
          f"min_n={MIN_OBSERVATIES})")
    print("=" * 70)
    print(f"  {'Strategie':<14} {'n':>7} {'Sharpe':>9} {'Gewicht':>9}  Status")
    print("  " + "-" * 60)

    if not sharpe_per_key:
        print("  Geen enkele strategie met voldoende observaties — fallback.")
        for key, gewicht in GEWICHTEN_FALLBACK.items():
            n = n_per_key.get(key, 0)
            print(f"  {key:<14} {n:>7} {'-':>9} {gewicht:>9.3f}  fallback")
        print("=" * 70)
        return dict(GEWICHTEN_FALLBACK)

    # Normaliseer Sharpe naar [GEWICHTEN_MIN, GEWICHTEN_MAX]
    sharpe_waarden = list(sharpe_per_key.values())
    min_s = min(sharpe_waarden)
    max_s = max(sharpe_waarden)
    bereik = max_s - min_s

    gewichten: Dict[str, float] = {}
    for key, sharpe in sharpe_per_key.items():
        if bereik < 1e-9:
            # Alle Sharpe-waarden identiek: geef ze allemaal hetzelfde gewicht
            gewicht = (GEWICHTEN_MIN + GEWICHTEN_MAX) / 2
        else:
            genorm = (sharpe - min_s) / bereik
            gewicht = GEWICHTEN_MIN + genorm * (GEWICHTEN_MAX - GEWICHTEN_MIN)
        gewichten[key] = round(gewicht, 3)

    # Vul aan met fallback voor strategieën zonder data
    for key, fallback in GEWICHTEN_FALLBACK.items():
        if key not in gewichten:
            gewichten[key] = fallback

    for key in GEWICHTEN_FALLBACK:
        n = n_per_key.get(key, 0)
        sharpe_str = f"{sharpe_per_key[key]:.4f}" if key in sharpe_per_key else "-"
        status = "dynamisch" if key in sharpe_per_key else "fallback"
        print(f"  {key:<14} {n:>7} {sharpe_str:>9} "
              f"{gewichten[key]:>9.3f}  {status}")

    print("=" * 70)
    return gewichten


# ============================================================
# STAP 1 — bulk-strategieën
# ============================================================

def compute_atr_pct(exchange_tickers, all_tickers) -> Dict[str, float]:
    print("[atr] Koersdata (3y) via bot_00kr's ATR-berekening...")
    df = kr.download_history(all_tickers, period="3y")
    if df.empty:
        return {}
    atr_pct: Dict[str, float] = {}
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)]
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = kr.analyse_ticker(ticker, group)
            if sig is not None and sig.price and sig.price > 0:
                atr_pct[ticker] = round(sig.atr / sig.price * 100, 2)
    return atr_pct


def selecties_repititief(exchange_tickers, all_tickers) -> Dict[str, Set[str]]:
    lookback = repititief.SZ_CFG["lookback_years"]
    print(f"[repititief] Koersdata ({lookback}y)...")
    df = repititief.download_history(all_tickers, period=f"{lookback}y")
    if df.empty:
        return {}
    result: Dict[str, Set[str]] = {}
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)]
        kandidaten = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            k = repititief.analyseer_ticker(ticker, group)
            if k is not None:
                kandidaten.append(k)
        if not kandidaten:
            result[ex_name] = set()
            continue
        fdr_significant = repititief.pas_bh_toe_op_beurs(kandidaten, repititief.SZ_CFG["fdr_alpha"])
        result[ex_name] = {s.ticker for s in fdr_significant}
    return result


def selecties_vcp(exchange_tickers, all_tickers) -> Dict[str, Set[str]]:
    print("[vcp] Koersdata (2y)...")
    df = vcp.download_history(all_tickers, period="2y")
    if df.empty:
        return {}
    result: Dict[str, Set[str]] = {}
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)]
        geselecteerd = {t for t, g in df_ex.groupby("Ticker", sort=False) if vcp.analyse_ticker(t, g)}
        result[ex_name] = geselecteerd
    return result


# ============================================================
# STAP 2 — live-fetch-strategieën
# ============================================================

def selecties_kasstr(shortlist_per_beurs: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    totaal = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"[kasstr] Fundamentals per ticker (live yfinance-calls) — {totaal} tickers op shortlist...")
    result: Dict[str, Set[str]] = {}
    for ex_name, shortlist in shortlist_per_beurs.items():
        geselecteerd = set()
        for ticker in shortlist:
            sig = met_retry(lambda t=ticker: kasstr.analyse_ticker(t), ticker, "kasstr")
            if sig is not None and sig.score >= kasstr.FCF_CFG["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(0.15)
        result[ex_name] = geselecteerd
    return result


def selecties_fisher(shortlist_per_beurs: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    totaal = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"[fisher] Fundamentals per ticker (live yfinance-calls) — {totaal} tickers op shortlist...")
    cfg = fisher.FISHER_CFG
    result: Dict[str, Set[str]] = {}
    for ex_name, shortlist in shortlist_per_beurs.items():
        geselecteerd = set()
        for ticker in shortlist:
            sig = met_retry(lambda t=ticker: fisher.analyse_ticker(t, cfg), ticker, "fisher")
            if sig is not None and sig.score >= cfg["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(cfg["throttle_sec"])
        result[ex_name] = geselecteerd
    return result


def _hoogl_analyse_met_retry(ticker: str, cfg: dict):
    """hoogl.analyse_ticker retourneert (signaal, was_rate_limited).
    Probeert opnieuw met backoff bij rate limiting en geeft enkel het
    signaal (of None) terug, zodat de aanroeper niet hoeft te unpacken."""
    for poging in range(RETRY_POGINGEN):
        sig, was_rate_limited = hoogl.analyse_ticker(ticker, cfg)
        if not was_rate_limited:
            return sig
        if poging < RETRY_POGINGEN - 1:
            wacht = RETRY_BASIS_WACHT * (2 ** poging)
            print(f"  [retry] hoogl {ticker}: rate limited, wacht {wacht:.0f}s (poging {poging+1}/{RETRY_POGINGEN})...")
            time.sleep(wacht)
    return None


def selecties_hoogl(shortlist_per_beurs: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    totaal = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"[hoogl] Fundamentals per ticker (live yfinance-calls) — {totaal} tickers op shortlist...")
    cfg = hoogl.MODUS_CFG["live"]
    result: Dict[str, Set[str]] = {}
    for ex_name, shortlist in shortlist_per_beurs.items():
        geselecteerd = set()
        for ticker in shortlist:
            # OUD (crashte: analyse_ticker retourneert een tuple):
            # sig = met_retry(lambda t=ticker: hoogl.analyse_ticker(t, cfg), ticker, "hoogl")
            sig = _hoogl_analyse_met_retry(ticker, cfg)
            if sig is not None and sig.score >= cfg["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(cfg["throttle_sec"])
        result[ex_name] = geselecteerd
    return result


# ============================================================
# STAP 3 — combineren + rapporteren
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"COMBI-SELECTIE VOLATIEL v6 (dynamische gewichten)  {kr.today_str()}")
    print(f"  ATR% tussen {MIN_ATR_PCT} en {MAX_ATR_PCT}")
    print(f"{'='*60}")

    # --- Dynamische gewichten berekenen ---
    gewichten = herbereken_gewichten()
    print(f"Gewichten: {gewichten}\n")

    exchange_tickers, all_tickers = bouw_exchange_tickers()
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return
    print(f"Totaal universum: {len(all_tickers)} unieke tickers over {len(exchange_tickers)} beurzen\n")

    warm_up_yfinance()

    atr_pct = compute_atr_pct(exchange_tickers, all_tickers)

    per_strategie: Dict[str, Dict[str, Set[str]]] = {}
    per_strategie["repititief"] = selecties_repititief(exchange_tickers, all_tickers)
    per_strategie["vcp"] = selecties_vcp(exchange_tickers, all_tickers)

    shortlist_per_beurs: Dict[str, Set[str]] = {}
    for ex_name in exchange_tickers:
        shortlist = set()
        for strat_key in ("repititief", "vcp"):
            shortlist |= per_strategie[strat_key].get(ex_name, set())
        shortlist_per_beurs[ex_name] = shortlist
    totaal_shortlist = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"\n[trechter] {totaal_shortlist} tickers op de shortlist "
          f"(van {len(all_tickers)} in het volledige universum)\n")

    per_strategie["kasstr"] = selecties_kasstr(shortlist_per_beurs)
    per_strategie["fisher"] = selecties_fisher(shortlist_per_beurs)
    per_strategie["hoogl"] = selecties_hoogl(shortlist_per_beurs)

    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        gewogen_score: Dict[str, float] = {}
        bijdragen: Dict[str, List[str]] = {}
        for strat_key, per_ex in per_strategie.items():
            geselecteerd = per_ex.get(ex_name, set())
            gewicht = gewichten.get(strat_key, 0.0)
            for ticker in geselecteerd:
                gewogen_score[ticker] = gewogen_score.get(ticker, 0.0) + gewicht
                bijdragen.setdefault(ticker, []).append(STRATEGIE_LABELS[strat_key])

        kandidaten = []
        for ticker, score in gewogen_score.items():
            atr = atr_pct.get(ticker)
            if atr is None or not (MIN_ATR_PCT <= atr <= MAX_ATR_PCT):
                continue
            kandidaten.append((ticker, score, atr, bijdragen[ticker]))

        kandidaten.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top = kandidaten[:TOP_N]

        print(f"\n{ex_name}: {len(gewogen_score)} tickers met >=1 selectie, "
              f"{len(kandidaten)} na volatiliteitsfilter, top {len(top)} gerapporteerd")

        for ticker, score, atr, strategieen in top:
            db_logger.log_selectie(
                ticker=ticker,
                datum=kr.today_str(),
                strategie="bot_combi_volatiel",
                beurs=ex_name,
                koers=None,
                parameters={
                    "gewogen_score": round(score, 2),
                    "strategieen": ", ".join(sorted(strategieen)),
                    "atr_pct": atr,
                    "grafiek": f"https://finance.yahoo.com/quote/{ticker}",
                },
            )

        if not top:
            continue

        gewicht_str = ", ".join(
            f"{k}={v:.2f}" for k, v in gewichten.items()
        )
        delen = [
            f"🎯 *Combi-Selectie Volatiel — {ex_name}*",
            f"_{kr.today_str()} | gewichten: {gewicht_str} "
            f"| ATR% {MIN_ATR_PCT}-{MAX_ATR_PCT}%_",
            "─────────────────────────────",
        ]
        for ticker, score, atr, strategieen in top:
            delen.append(
                f"• `{ticker}` — score {score:.2f} | ATR {atr:.1f}% "
                f"{_yahoo_link(ticker)}\n"
                f"  {', '.join(sorted(strategieen))}"
            )
        bericht = "\n\n".join(delen)
        kr.send_telegram_message(bericht)
        email_delen.append(bericht)
        print(f"  → Telegram verstuurd: {ex_name}")

    if email_delen:
        kr.send_email(
            subject=f"Combi-Selectie Volatiel rapport {kr.today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


if __name__ == "__main__":
    run_live_engine()
