#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_combi_volatiel.py — GEWOGEN KWALITEITSSCORE x VOLATILITEIT  (v3)

Herzien op basis van de Selecties-analyse van 2026-09-06 (tweede meting,
eerste was 2026-08-31):

  bot_00kr is VOLLEDIG UIT DE STEMMING gehaald. Bij de eerste meting had
  het de hoogste getrimde edge (+1,2%, n=1332, wr55%) en dus het hoogste
  gewicht in v2. Bij de tweede meting (6 dagen later, n=1456) is dat
  volledig omgeslagen naar −0,5% met wr41% — en cruciaal: zelfs de
  MEDIAAN sloeg om (+0,5% → −0,7%). Een mediaan is ongevoelig voor
  uitschieters, dus een omslag daarin wijst op een structurele
  verslechtering, niet op een paar tegenvallers. Met n>1300 op beide
  metingen is dit bovendien geen ruis door een kleine steekproef — het is
  een betekenisvolle omslag. bot_00kr wordt daarom niet langer als
  stemmende strategie gebruikt, maar blijft wél geïmporteerd omdat zijn
  ATR-berekening (voor het volatiliteitsfilter) hergebruikt wordt, los
  van zijn eigen (nu niet-vertrouwde) score.

  De overige 5 strategieën met op BEIDE metingen een positieve getrimde
  edge blijven erin, met als gewicht het GEMIDDELDE van de twee metingen
  (dempt week-op-week ruis t.o.v. één momentopname):

    strategie          31/08   06/09   gewicht (gemiddelde)
    bot_01kasstr       +1.2%   +0.5%   0.85
    bot_00Fisher       +0.7%   +0.4%   0.55
    bot_00vcp          +0.6%   +0.5%   0.55
    bot_01hoogl        +0.4%   +0.6%   0.50
    bot_01repititief   +0.6%   −0.3%   0.15

  bot_01repititief is de enige twijfelgeval: omgeslagen naar negatief bij
  de tweede meting, maar minder uitgesproken dan kr (mediaan bleef
  nagenoeg vlak: +0,1% → +0,0%, wat eerder op een handvol tegenvallers
  wijst dan op een structurele kentering). Blijft daarom voorlopig mee,
  maar met een sterk gereduceerd gewicht (0.15) dat de onzekerheid
  weerspiegelt — niet weggegooid, niet vertrouwd.

BEWUST NIET meegenomen (ongewijzigd t.o.v. v2, bevestigd door de tweede
meting):
  - bot_00cs, bot_00ms → eerste meting negatief; bot_00ms is intussen wel
    verbeterd naar +0,2% maar met bescheiden n=262 en zwakke edge, nog
    niet overtuigend genoeg om op te nemen.
  - bot_01marktsent → nog steeds negatief op beide metingen (−0,4% / −0,5%).
  - bot_00db → n=18.950 (61% van alle selecties), maar nu ronduit
    negatief (−0,2%, was al maar +0,2%). Bevestigt: hoge vuurfrequentie
    zonder edge, hoe meer data resolveert hoe duidelijker.
  - bot_00graham → substantieel verbeterd (n=552, wr49%, +0,4%) maar dit
    is pas de EERSTE meting onder de huidige (F-Score) code — nog geen
    tweede bevestiging, dus nog niet opgenomen. Kandidaat voor een
    volgende herziening als dit stand houdt.
  - bot_00oshaughnessy → veelbelovend (n=25, wr84%, +2,0%) maar n=25 is
    te klein om al te vertrouwen (zie hoe bot_01greenblatt met evenveel
    n=30 in 6 dagen omsloeg van +0,1% naar −0,4% zonder dat de strategie
    veranderde — pure rijpings-ruis op dat niveau).
  - bot_00dm, bot_01cointegr, bot_00mr, bot_01xgboostMeta → ongewijzigd
    (geen trackrecord resp. structureel niet passend, zie v1/v2).

Score per ticker = som van de gewichten van elke strategie die de ticker
vandaag zou selecteren via haar eigen, ongewijzigde analyse_ticker-functie
en eigen score-drempel — geen scoringslogica is herschreven.

Volatiliteitsfilter ongewijzigd t.o.v. v2: ATR% (14-daagse ATR/koers×100)
via bot_00kr's eigen ATR-berekening (los van bot_00kr's eigen — nu
genegeerde — score), standaard 4%-25% ("vrij tot sterk volatiel").

Rapportage: enkel tickers met gewogen score > 0 EN binnen de ATR%-range,
top N per beurs, gesorteerd op gewogen score. Zelfde architectuur: één
Telegram-bericht per beurs, één samenvattende e-mail, db_logger onder
strategie "bot_combi_volatiel", geen CSV.
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

# Gewicht = gemiddelde van de getrimde gemiddelde-return (%) over de
# metingen van 2026-08-31 en 2026-09-06. bot_00kr bewust NIET opgenomen
# (zie module-docstring — significante omslag naar negatief, incl. mediaan).
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
# ATR% — uitsluitend voor het volatiliteitsfilter, los van bot_00kr's
# eigen (niet langer vertrouwde) score/selectie
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


# ============================================================
# STAP 1 — per bevestigde strategie: welke tickers selecteert ze vandaag?
# ============================================================

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
    print(f"COMBI-SELECTIE VOLATIEL v3 (gewogen, kr uit stemming)  {kr.today_str()}")
    print(f"  ATR% tussen {MIN_ATR_PCT} en {MAX_ATR_PCT} | gewichten: {GEWICHTEN}")
    print(f"{'='*60}")

    exchange_tickers, all_tickers = bouw_exchange_tickers()
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return
    print(f"Totaal universum: {len(all_tickers)} unieke tickers over {len(exchange_tickers)} beurzen\n")

    atr_pct = compute_atr_pct(exchange_tickers, all_tickers)

    per_strategie: Dict[str, Dict[str, Set[str]]] = {}
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
