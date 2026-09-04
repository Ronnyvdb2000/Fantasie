#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_combi_volatiel.py — GEWOGEN KWALITEITSSCORE x VOLATILITEIT  (v2)

Herzien op basis van de Selecties-analyse van 2026-08-31: i.p.v. alle 11
strategieën gelijk te tellen (v1), gebruikt deze versie enkel de strategieën
met een BEWEZEN positieve getrimde gemiddelde-return op hun eigen huidige
code (zelfde bestandsnaam/strategie-label als in de performance-analyse —
geverifieerd via de GitHub-commit-historie, geen aannames):

    strategie      getrimd gem.   n      win-rate   gewicht
    bot_00kr           +1.2%     1332      55%        1.2
    bot_01kasstr        +1.2%      744      58%        1.2
    bot_00Fisher        +0.7%       88      61%        0.7
    bot_01repititief    +0.6%      306      51%        0.6
    bot_00vcp           +0.6%     1767      50%        0.6
    bot_01hoogl         +0.4%      431      46%        0.4

BEWUST NIET meegenomen, met reden:
  - bot_00cs, bot_00ms, bot_01marktsent  → aantoonbaar NEGATIEVE getrimde
    edge (−0.6%, −0.8%, −0.4%) op hun huidige code. Meestemmen zou het
    signaal verzwakken, niet versterken.
  - bot_00db  → n=12.853 (59% van alle selecties in de hele dataset) tegen
    amper +0.2% getrimd edge. Zo'n hoge vuurfrequentie met verwaarloosbare
    edge zou "overlap" bijna altijd laten kloppen zonder er iets aan toe
    te voegen — verdunt het signaal i.p.v. het te versterken.
  - bot_00graham, bot_00oshaughnessy, bot_00greenblatt → de GOEDE cijfers
    in de analyse ("bot_01graham" +0.4%/wr57%, "bot_01oshaughnessy"
    +0.5%/wr60%, "bot_01greenblatt" +0.1%/wr47%) horen bij oudere versies
    van deze scripts die intussen herschreven/hernoemd zijn (bevestigd via
    GitHub-commits van 24-25/08/2026). bot_00graham (de HUIDIGE code) heeft
    zelf een apart, veel zwakker trackrecord: wr 24%, getrimd +0.0%. Voor
    bot_00oshaughnessy/bot_00greenblatt bestaat er nog geen trackrecord
    onder hun huidige naam. De code die goed presteerde bestaat dus niet
    meer in de huidige vorm — parameters overnemen zou op een aanname
    berusten, niet op bewijs.
  - bot_00dm, bot_01cointegr, bot_01xgboostMeta, bot_00mr → niet in de
    performance-analyse opgenomen (geen trackrecord om op te wegen) resp.
    structureel niet passend (mr = apart portfoliosysteem, xgboostMeta =
    forex i.p.v. aandelen).

Score per ticker = som van de gewichten van elke strategie (uit bovenstaande
tabel) die de ticker VANDAAG zou selecteren via haar eigen, ongewijzigde
analyse_ticker-functie en eigen score-drempel — geen enkele scoringslogica
is herschreven, enkel geïmporteerd en aangeroepen.

Volatiliteitsfilter: ATR% (14-daagse ATR/koers×100) via bot_00kr's eigen
ATR-berekening, los van bot_00kr's eigen score. Standaard 4%-25% ("vrij tot
sterk volatiel" — de bovengrens is enkel om de bekende yfinance
databug-uitschieters (foutieve valutaomrekening) uit te sluiten, niet om
sterke volatiliteit te beperken).

Rapportage: enkel tickers met gewogen score > 0 (dus geselecteerd door
minstens 1 van de 6 bewezen strategieën) EN binnen de ATR%-range, top N
per beurs, gesorteerd op gewogen score. Zelfde architectuur als de andere
bots: één Telegram-bericht per beurs, één samenvattende e-mail, db_logger
onder strategie "bot_combi_volatiel", geen CSV.
"""

import os
import time
from typing import Dict, List, Set, Tuple

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

MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "4.0"))    # "vrij" volatiel ondergrens
MAX_ATR_PCT = float(os.getenv("MAX_ATR_PCT", "25.0"))   # "sterk" toegelaten, outliers eruit
TOP_N       = int(os.getenv("TOP_N", "10"))

# Gewicht = getrimd gemiddelde (%) uit de Selecties-analyse van 2026-08-31,
# enkel strategieën met bewezen positieve edge op hun HUIDIGE code.
GEWICHTEN = {
    "kr":         1.2,
    "kasstr":     1.2,
    "fisher":     0.7,
    "repititief": 0.6,
    "vcp":        0.6,
    "hoogl":      0.4,
}

STRATEGIE_LABELS = {
    "kr": "bot_00kr", "kasstr": "bot_01kasstr", "fisher": "bot_00Fisher",
    "repititief": "bot_01repititief", "vcp": "bot_00vcp", "hoogl": "bot_01hoogl",
}


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


# ============================================================
# STAP 1 — per bewezen strategie: welke tickers selecteert ze vandaag?
# ============================================================

def kr_data_en_selecties(exchange_tickers, all_tickers):
    """Geeft selecties én ATR%-waarden terug (voor het volatiliteitsfilter)."""
    print("[kr] Koersdata (3y)...")
    df = kr.download_history(all_tickers, period="3y")
    if df.empty:
        return {}, {}
    result: Dict[str, Set[str]] = {}
    atr_pct: Dict[str, float] = {}
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)]
        geselecteerd = set()
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = kr.analyse_ticker(ticker, group)
            if sig is None:
                continue
            if sig.price and sig.price > 0:
                atr_pct[ticker] = round(sig.atr / sig.price * 100, 2)
            if sig.score >= kr.KS_CFG["min_score"]:
                geselecteerd.add(ticker)
        result[ex_name] = geselecteerd
    return result, atr_pct


def selecties_kasstr(exchange_tickers) -> Dict[str, Set[str]]:
    print("[kasstr] Fundamentals per ticker (live yfinance-calls)...")
    result: Dict[str, Set[str]] = {}
    for ex_name, tlist in exchange_tickers.items():
        geselecteerd = set()
        for ticker in tlist:
            sig = kasstr.analyse_ticker(ticker)
            if sig is not None and sig.score >= kasstr.FCF_CFG["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(0.15)
        result[ex_name] = geselecteerd
    return result


def selecties_fisher(exchange_tickers) -> Dict[str, Set[str]]:
    print("[fisher] Fundamentals per ticker (live yfinance-calls)...")
    cfg = fisher.FISHER_CFG
    result: Dict[str, Set[str]] = {}
    for ex_name, tlist in exchange_tickers.items():
        geselecteerd = set()
        for ticker in tlist:
            sig = fisher.analyse_ticker(ticker, cfg)
            if sig is not None and sig.score >= cfg["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(cfg["throttle_sec"])
        result[ex_name] = geselecteerd
    return result


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


def selecties_hoogl(exchange_tickers) -> Dict[str, Set[str]]:
    print("[hoogl] Fundamentals per ticker (live yfinance-calls)...")
    cfg = hoogl.MODUS_CFG["live"]
    result: Dict[str, Set[str]] = {}
    for ex_name, tlist in exchange_tickers.items():
        geselecteerd = set()
        for ticker in tlist:
            sig = hoogl.analyse_ticker(ticker, cfg)
            if sig is not None and sig.score >= cfg["min_score"]:
                geselecteerd.add(ticker)
            time.sleep(cfg["throttle_sec"])
        result[ex_name] = geselecteerd
    return result


# ============================================================
# STAP 2 — combineren (gewogen), filteren op volatiliteit, rapporteren
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"COMBI-SELECTIE VOLATIEL v2 (gewogen)  {kr.today_str()}")
    print(f"  ATR% tussen {MIN_ATR_PCT} en {MAX_ATR_PCT} | gewichten: {GEWICHTEN}")
    print(f"{'='*60}")

    exchange_tickers, all_tickers = bouw_exchange_tickers()
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return
    print(f"Totaal universum: {len(all_tickers)} unieke tickers over {len(exchange_tickers)} beurzen\n")

    per_strategie: Dict[str, Dict[str, Set[str]]] = {}
    per_strategie["kr"], atr_pct = kr_data_en_selecties(exchange_tickers, all_tickers)
    per_strategie["kasstr"] = selecties_kasstr(exchange_tickers)
    per_strategie["fisher"] = selecties_fisher(exchange_tickers)
    per_strategie["repititief"] = selecties_repititief(exchange_tickers, all_tickers)
    per_strategie["vcp"] = selecties_vcp(exchange_tickers, all_tickers)
    per_strategie["hoogl"] = selecties_hoogl(exchange_tickers)

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
            f"_{kr.today_str()} | gewogen score (kr/kasstr=1.2, fisher=0.7, repititief/vcp=0.6, hoogl=0.4) "
            f"| ATR% {MIN_ATR_PCT}-{MAX_ATR_PCT}%_",
            "─────────────────────────────",
        ]
        for ticker, score, atr, strategieen in top:
            delen.append(
                f"• `{ticker}` — score {score:.1f} | ATR {atr:.1f}% "
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
