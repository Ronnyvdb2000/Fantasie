#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_combi_volatiel.py — GEWOGEN KWALITEITSSCORE x VOLATILITEIT  (v4)

Herzien op basis van de workflow-log van de run van 2026-09-06: die run
kreeg 5.693 "Too Many Requests"-fouten van Yahoo Finance. Oorzaak
(bevestigd via de log, niet gegokt): bot_01kasstr, bot_00Fisher en
bot_01hoogl doen elk honderden tot duizenden LOSSE live
`yf.Ticker(ticker).info`-calls — de meest rate-limit-gevoelige
Yahoo-endpoint. Los draaien deze bots prima (elk in hun eigen
GitHub Actions-job, op een ander moment). In deze combi-bot draaien ze
na elkaar in dezelfde sessie, dus hun verzoeken stapelen zich op tegen
dezelfde IP-limiet — die werd binnen enkele minuten bereikt en bleef de
rest van de run (~25 min) actief.

TWEE MAATREGELEN in v4 (samen, op vraag van de gebruiker):

  1) TRECHTER — kasstr/fisher/hoogl draaien niet langer op het volledige
     x-universum per beurs, maar enkel nog op de tickers die al
     minstens 1 stem hebben van de goedkope, bulk-gebaseerde strategieën
     (bot_00kr's ATR-check op "heeft ATR% berekend", bot_00vcp,
     bot_01repititief — alle drie via bulk yf.download(), niet per-ticker
     live calls, en dus veel minder rate-limit-gevoelig). Dit verlaagt
     het live-fetch-volume typisch met >90% (in de log van 06/09 vond
     vcp bv. maar 31 van de ~180 Nasdaq/NYSE x-tickers interessant).
     Consequentie: kasstr/fisher/hoogl worden effectief gebruikt als
     BEVESTIGING bovenop een technisch/seizoensgebonden signaal, niet
     meer als volledig onafhankelijke full-universe screener. Gegeven de
     externe rate-limit-beperking is dit een bewuste, uitgelegde
     trade-off — geen stille wijziging.

  2) RETRY-MET-BACKOFF — elke resterende live-fetch-call (op de kleinere
     shortlist) krijgt tot 3 pogingen met oplopende wachttijd (8s, 16s,
     32s) bij een "Too Many Requests"-fout, i.p.v. meteen opgeven.

Gewichten, uitsluitingen en volatiliteitsfilter ongewijzigd t.o.v. v3
(zie die docstring-geschiedenis in Git voor de volledige onderbouwing):

    bot_01kasstr       0.85
    bot_00Fisher       0.55
    bot_00vcp          0.55
    bot_01hoogl        0.50
    bot_01repititief   0.15
    bot_00kr           niet in de stemming, enkel voor ATR%-berekening

Score per ticker = som van de gewichten van elke strategie die de ticker
vandaag zou selecteren via haar eigen, ongewijzigde analyse_ticker-functie
en eigen score-drempel — geen scoringslogica is herschreven.

Rapportage ongewijzigd: enkel tickers met gewogen score > 0 EN binnen de
ATR%-range (4%-25%), top N per beurs. Eén Telegram-bericht per beurs, één
samenvattende e-mail, db_logger onder strategie "bot_combi_volatiel".
"""

import os
import time
from typing import Dict, List, Set, Tuple

import bot_00kr as kr                 # enkel voor ATR%-berekening + gedeelde hulpfuncties
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
RETRY_BASIS_WACHT  = float(os.getenv("RETRY_BASIS_WACHT", "8"))  # seconden, verdubbelt per poging

# Gewicht = gemiddelde van de getrimde gemiddelde-return (%) over de
# metingen van 2026-08-31 en 2026-09-06. bot_00kr bewust NIET opgenomen.
GEWICHTEN = {
    "kasstr":     0.85,
    "fisher":     0.55,
    "vcp":        0.55,
    "hoogl":      0.50,
    "repititief": 0.15,
}

STRATEGIE_LABELS = {
    "kasstr": "bot_01kasstr", "fisher": "bot_00Fisher",
    "vcp": "bot_00vcp", "hoogl": "bot_01hoogl", "repititief": "bot_01repititief",
}

# Enkel deze strategieën doen live per-ticker .info-calls en worden dus
# getrechterd tot de shortlist. vcp/repititief blijven op het volledige
# universum draaien (bulk yf.download(), veel minder rate-limit-gevoelig).
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
    """Voert fn() uit; bij 'Too Many Requests' tot RETRY_POGINGEN keer
    opnieuw proberen met oplopende wachttijd. Geeft None terug bij
    definitieve mislukking (ticker wordt dan overgeslagen, zoals voorheen)."""
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


# ============================================================
# STAP 1 — bulk-strategieën (goedkoop, volledig universum)
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
# STAP 2 — live-fetch-strategieën (duur, enkel op de shortlist)
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


def selecties_hoogl(shortlist_per_beurs: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    totaal = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"[hoogl] Fundamentals per ticker (live yfinance-calls) — {totaal} tickers op shortlist...")
    cfg = hoogl.MODUS_CFG["live"]
    result: Dict[str, Set[str]] = {}
    for ex_name, shortlist in shortlist_per_beurs.items():
        geselecteerd = set()
        for ticker in shortlist:
            sig = met_retry(lambda t=ticker: hoogl.analyse_ticker(t, cfg), ticker, "hoogl")
            if sig is not None and sig.score >= cfg["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(cfg["throttle_sec"])
        result[ex_name] = geselecteerd
    return result


# ============================================================
# STAP 3 — combineren (gewogen), filteren op volatiliteit, rapporteren
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"COMBI-SELECTIE VOLATIEL v4 (trechter + retry-backoff)  {kr.today_str()}")
    print(f"  ATR% tussen {MIN_ATR_PCT} en {MAX_ATR_PCT} | gewichten: {GEWICHTEN}")
    print(f"{'='*60}")

    exchange_tickers, all_tickers = bouw_exchange_tickers()
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return
    print(f"Totaal universum: {len(all_tickers)} unieke tickers over {len(exchange_tickers)} beurzen\n")

    # --- bulk-strategieën, volledig universum ---
    atr_pct = compute_atr_pct(exchange_tickers, all_tickers)

    per_strategie: Dict[str, Dict[str, Set[str]]] = {}
    per_strategie["repititief"] = selecties_repititief(exchange_tickers, all_tickers)
    per_strategie["vcp"] = selecties_vcp(exchange_tickers, all_tickers)

    # --- shortlist opbouwen: unie van tickers met >=1 stem uit de bulk-strategieën ---
    shortlist_per_beurs: Dict[str, Set[str]] = {}
    for ex_name in exchange_tickers:
        shortlist = set()
        for strat_key in ("repititief", "vcp"):
            shortlist |= per_strategie[strat_key].get(ex_name, set())
        shortlist_per_beurs[ex_name] = shortlist
    totaal_shortlist = sum(len(s) for s in shortlist_per_beurs.values())
    print(f"\n[trechter] {totaal_shortlist} tickers op de shortlist (van {len(all_tickers)} in het volledige universum) "
          f"voor de live-fetch-strategieën\n")

    # --- live-fetch-strategieën, enkel op de shortlist ---
    per_strategie["kasstr"] = selecties_kasstr(shortlist_per_beurs)
    per_strategie["fisher"] = selecties_fisher(shortlist_per_beurs)
    per_strategie["hoogl"] = selecties_hoogl(shortlist_per_beurs)

    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        gewogen_score: Dict[str, float] = {}
        bijdragen: Dict[str, List[str]] = {}
        for strat_key, per_ex in per_strategie.items():
            geselecteerd = per_ex.get(ex_name, set())
            gewicht = GEWICHTEN[strat_key]
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

        delen = [
            f"🎯 *Combi-Selectie Volatiel — {ex_name}*",
            f"_{kr.today_str()} | gewogen score (kasstr=0.85, fisher/vcp=0.55, hoogl=0.5, repititief=0.15) "
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
