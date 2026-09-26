#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_generieke_fundamentals.py  —  GEDEELDE FUNDAMENTELE PARAMETERS  v1.0

DOEL
====
Analoog aan bouw_generieke_technicals.py, maar voor fundamentele data
(FCF yield, P/E, P/B, Piotroski F-Score, EPS-groei, PEG, ...): berekent deze
centraal en strategie-onafhankelijk -- één rij per (ticker, datum), niet
enkel voor de bots die deze parameters toevallig zelf al gebruiken
(bot_01kasstr, bot_00graham, bot_00greenblatt, bot_00oshaughnessy, ...).
Zo krijgt OOK een selectie van bv. bot_00db of bot_00vcp deze parameters
ingevuld, i.p.v. dat ze strategie-exclusief blijven zoals in `selecties`.

KRITIEK VERSCHIL MET bouw_generieke_technicals.py — GEEN BACKFILL:
yfinance geeft voor fundamentals (marktkap, P/E, balans, cashflow-reeks)
ENKEL de HUIDIGE momentopname, nooit een historisch puntmoment. Zou dit
script, net als de technicals-variant, de volledige historische
`selecties`-achterstand proberen invullen, dan zou een selectie van drie
maanden geleden de fundamentals van VANDAAG toegewezen krijgen -- klassieke
look-ahead bias (toekomstige bedrijfsinformatie in een verleden-rij).

Daarom verwerkt dit script UITSLUITEND (ticker, datum)-paren met een datum
binnen RECENTE_DAGEN_LIMIET dagen vóór vandaag. Een selectie die ouder is
en nog geen rij heeft, krijgt er NOOIT een -- dat is bewust, geen bug.

HERGEBRUIKTE LOGICA (bewust niet opnieuw uitgevonden):
- Piotroski F-Score (0-9): identieke implementatie als bot_01kasstr.py /
  bot_00graham.py (welke op hun beurt identiek zijn aan elkaar).
- FCF-reeks/yield/trend/consistentie + shareholder-return/payout:
  overgenomen uit bot_01kasstr.py.
- EPS-groei (totaal) + EPS-CAGR (jaarlijks) + PEG-ratio: overgenomen uit
  bot_00graham.py.

Per ticker kost dit tot 4 yfinance-calls (info, cashflow, balance_sheet,
financials) -- net als bot_00graham.py, dus merkelijk zwaarder per ticker
dan bouw_generieke_technicals.py (dat enkel prijs/volumehistoriek nodig
heeft). De RECENTE_DAGEN_LIMIET + het feit dat éénzelfde ticker binnen die
periode maar ÉÉN keer bevraagd wordt (ongeacht hoeveel keer of door hoeveel
strategieën hij recent geselecteerd werd) houdt het volume beheersbaar.

GEBRUIK
=======
  python bouw_generieke_fundamentals.py build

Env vars: SUPABASE_DB_URL (verplicht), TELEGRAM_TOKEN/TELEGRAM_CHAT_ID
(optioneel), MAX_TICKERS_PER_RUN (default 200 -- lager dan de technicals-
variant omdat elke ticker hier tot 4x zoveel yfinance-calls kost),
RECENTE_DAGEN_LIMIET (default 5)
"""

import os
import sys
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
import yfinance as yf
import requests

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_TICKERS_PER_RUN = int(os.environ.get("MAX_TICKERS_PER_RUN", "200"))
RECENTE_DAGEN_LIMIET = int(os.environ.get("RECENTE_DAGEN_LIMIET", "5"))

FCF_MIN_YEARS = 2


def vandaag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def send_telegram(tekst: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(tekst)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")


def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default


# --------------------------------------------------------------------------
# Hergebruikt uit bot_01kasstr.py / bot_00graham.py (identiek overgenomen)
# --------------------------------------------------------------------------
def _row(df, names: List[str]):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def _fcf_series(cashflow) -> List[float]:
    if cashflow is None or cashflow.empty:
        return []
    fcf_row = _row(cashflow, ["Free Cash Flow"])
    if fcf_row is not None:
        vals = [safe_float(v) for v in fcf_row.tolist()]
    else:
        ocf = _row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
        capex = _row(cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"])
        if ocf is None or capex is None:
            return []
        vals = [safe_float(o) - abs(safe_float(c)) for o, c in zip(ocf.tolist(), capex.tolist())]
    vals = [v for v in vals if not math.isnan(v)]
    return list(reversed(vals))


def _shareholder_return(cashflow) -> float:
    if cashflow is None or cashflow.empty:
        return 0.0
    div = _row(cashflow, ["Cash Dividends Paid", "Common Stock Dividend Paid", "Payment Of Dividends"])
    bb = _row(cashflow, ["Repurchase Of Capital Stock", "Common Stock Repurchase", "Repurchase Of Common Stock"])
    total = 0.0
    if div is not None:
        total += abs(safe_float(div.iloc[0], 0.0))
    if bb is not None:
        total += abs(safe_float(bb.iloc[0], 0.0))
    return total


def _net_income_series(financials) -> List[float]:
    row = _row(financials, ["Net Income", "Net Income Common Stockholders"])
    if row is None:
        return []
    vals = [safe_float(v) for v in row.tolist()]
    vals = [v for v in vals if not math.isnan(v)]
    return list(reversed(vals))


def _piotroski_f_score(tk, cashflow) -> int:
    """Identieke implementatie als bot_01kasstr.py/bot_00graham.py.
    Retourneert -1 als er geen 2 jaar balans + resultatenrekening is."""
    try:
        bs = tk.balance_sheet
        inc = tk.financials
    except Exception:
        return -1

    if bs is None or bs.empty or inc is None or inc.empty:
        return -1
    if bs.shape[1] < 2 or inc.shape[1] < 2:
        return -1

    def val(df, names, col):
        row = _row(df, names)
        if row is None or col >= len(row):
            return float("nan")
        return safe_float(row.iloc[col])

    ta0, ta1 = val(bs, ["Total Assets"], 0), val(bs, ["Total Assets"], 1)
    ca0, ca1 = val(bs, ["Current Assets", "Total Current Assets"], 0), val(bs, ["Current Assets", "Total Current Assets"], 1)
    cl0, cl1 = val(bs, ["Current Liabilities", "Total Current Liabilities"], 0), val(bs, ["Current Liabilities", "Total Current Liabilities"], 1)
    ltd0, ltd1 = val(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0), val(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
    sh0, sh1 = val(bs, ["Share Issued", "Ordinary Shares Number"], 0), val(bs, ["Share Issued", "Ordinary Shares Number"], 1)

    ni0, ni1 = val(inc, ["Net Income"], 0), val(inc, ["Net Income"], 1)
    rev0, rev1 = val(inc, ["Total Revenue"], 0), val(inc, ["Total Revenue"], 1)
    gp0, gp1 = val(inc, ["Gross Profit"], 0), val(inc, ["Gross Profit"], 1)

    ocf_row = _row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
    ocf0 = safe_float(ocf_row.iloc[0]) if ocf_row is not None and len(ocf_row) > 0 else float("nan")

    vlaggen = {}
    roa0 = ni0 / ta0 if not math.isnan(ni0) and not math.isnan(ta0) and ta0 != 0 else float("nan")
    roa1 = ni1 / ta1 if not math.isnan(ni1) and not math.isnan(ta1) and ta1 != 0 else float("nan")
    vlaggen["roa_positief"] = (not math.isnan(roa0)) and roa0 > 0
    vlaggen["ocf_positief"] = (not math.isnan(ocf0)) and ocf0 > 0
    vlaggen["roa_gestegen"] = (not math.isnan(roa0)) and (not math.isnan(roa1)) and roa0 > roa1
    vlaggen["winstkwaliteit"] = (not math.isnan(ocf0)) and (not math.isnan(ni0)) and ocf0 > ni0

    lev0 = ltd0 / ta0 if not math.isnan(ltd0) and not math.isnan(ta0) and ta0 != 0 else float("nan")
    lev1 = ltd1 / ta1 if not math.isnan(ltd1) and not math.isnan(ta1) and ta1 != 0 else float("nan")
    vlaggen["hefboom_gedaald"] = (not math.isnan(lev0)) and (not math.isnan(lev1)) and lev0 < lev1

    cr0 = ca0 / cl0 if not math.isnan(ca0) and not math.isnan(cl0) and cl0 != 0 else float("nan")
    cr1 = ca1 / cl1 if not math.isnan(ca1) and not math.isnan(cl1) and cl1 != 0 else float("nan")
    vlaggen["liquiditeit_gestegen"] = (not math.isnan(cr0)) and (not math.isnan(cr1)) and cr0 > cr1

    vlaggen["geen_verwatering"] = (not math.isnan(sh0)) and (not math.isnan(sh1)) and sh0 <= sh1

    gm0 = gp0 / rev0 if not math.isnan(gp0) and not math.isnan(rev0) and rev0 != 0 else float("nan")
    gm1 = gp1 / rev1 if not math.isnan(gp1) and not math.isnan(rev1) and rev1 != 0 else float("nan")
    vlaggen["marge_gestegen"] = (not math.isnan(gm0)) and (not math.isnan(gm1)) and gm0 > gm1

    at0 = rev0 / ta0 if not math.isnan(rev0) and not math.isnan(ta0) and ta0 != 0 else float("nan")
    at1 = rev1 / ta1 if not math.isnan(rev1) and not math.isnan(ta1) and ta1 != 0 else float("nan")
    vlaggen["efficientie_gestegen"] = (not math.isnan(at0)) and (not math.isnan(at1)) and at0 > at1

    return sum(1 for v in vlaggen.values() if v)


# --------------------------------------------------------------------------
# Kern: alle fundamentals voor één ticker in één keer ophalen
# --------------------------------------------------------------------------
def haal_fundamentals_op(ticker: str) -> Optional[dict]:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception as e:
        print(f"  [WARN] {ticker}: info-ophalen mislukt ({e})")
        return None

    market_cap = safe_float(info.get("marketCap"))
    if math.isnan(market_cap) or market_cap <= 0:
        return None

    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = None
    try:
        financials = tk.financials
    except Exception:
        financials = None

    # --- FCF (bot_01kasstr-logica) ---
    fcf_series = _fcf_series(cashflow)
    if len(fcf_series) >= FCF_MIN_YEARS:
        fcf_now = fcf_series[-1]
        fcf_yield = (fcf_now / market_cap) * 100 if market_cap > 0 else float("nan")
        # Zelfde data-kwaliteitscheck als bot_01kasstr.py: implausibele FCF
        # yield (valuta-/eenheid-mismatch) -> op None laten i.p.v. vervuild
        # signaal doorgeven.
        if not math.isnan(fcf_yield) and abs(fcf_yield) > 100:
            fcf_yield = float("nan")
        fcf_growing = fcf_series[-1] > fcf_series[0]
        fcf_consistent = all(v > 0 for v in fcf_series)
        sh_return = _shareholder_return(cashflow)
        payout_pct = (sh_return / fcf_now * 100) if fcf_now > 0 else float("nan")
    else:
        fcf_yield = float("nan")
        fcf_growing = None
        fcf_consistent = None
        payout_pct = float("nan")

    # --- Balans-kwaliteit (net debt/EBITDA, bot_01kasstr-logica) ---
    total_debt = safe_float(info.get("totalDebt"), 0.0)
    total_cash = safe_float(info.get("totalCash"), 0.0)
    ebitda = safe_float(info.get("ebitda"))
    net_debt = total_debt - total_cash
    net_debt_ebitda = net_debt / ebitda if not math.isnan(ebitda) and ebitda > 0 else float("nan")

    # --- EPS-groei/CAGR/PEG (bot_00graham-logica) ---
    ni_series = _net_income_series(financials)
    eps_growth_pct = float("nan")
    eps_cagr_pct = float("nan")
    pe_ratio = safe_float(info.get("trailingPE"))
    if len(ni_series) >= 2:
        oudste, nieuwste = ni_series[0], ni_series[-1]
        if oudste > 0:
            eps_growth_pct = (nieuwste - oudste) / oudste * 100
        if oudste > 0 and nieuwste > 0:
            eps_cagr_pct = ((nieuwste / oudste) ** (1 / (len(ni_series) - 1)) - 1) * 100
    peg_ratio = float("nan")
    if not math.isnan(pe_ratio) and pe_ratio > 0 and not math.isnan(eps_cagr_pct) and eps_cagr_pct > 0:
        peg_ratio = pe_ratio / eps_cagr_pct

    # --- Piotroski F-Score ---
    piotroski = _piotroski_f_score(tk, cashflow)

    revenue_growth_raw = safe_float(info.get("revenueGrowth"))

    def rond(v, decimalen=4):
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, decimalen)
        except Exception:
            return None

    return {
        "market_cap": rond(market_cap, 0),
        "trailing_pe": rond(pe_ratio, 2),
        "price_to_book": rond(info.get("priceToBook"), 2),
        "dividend_yield": rond(info.get("dividendYield"), 2),
        "current_ratio": rond(info.get("currentRatio"), 2),
        "revenue_growth_pct": rond(revenue_growth_raw * 100, 2) if not math.isnan(revenue_growth_raw) else None,
        "fcf_yield": rond(fcf_yield, 2),
        "fcf_years": len(fcf_series) if fcf_series else None,
        "fcf_growing": fcf_growing,
        "fcf_consistent": fcf_consistent,
        "net_debt_ebitda": rond(net_debt_ebitda, 2),
        "payout_pct": rond(payout_pct, 1),
        "analisten_count": rond(info.get("numberOfAnalystOpinions"), 0),
        "piotroski_score": piotroski if piotroski >= 0 else None,
        "eps_growth_pct": rond(eps_growth_pct, 1),
        "eps_cagr_pct": rond(eps_cagr_pct, 1),
        "peg_ratio": rond(peg_ratio, 2),
    }


# --------------------------------------------------------------------------
# Stap 1: welke recente (ticker, datum)-paren ontbreken nog?
# --------------------------------------------------------------------------
def haal_openstaande_paren(conn) -> List[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENTE_DAGEN_LIMIET)).strftime("%Y-%m-%d")
    query = """
        SELECT DISTINCT s.ticker, s.datum
        FROM selecties s
        LEFT JOIN generieke_fundamentals gf
          ON s.ticker = gf.ticker AND s.datum = gf.datum
        WHERE gf.ticker IS NULL
          AND s.ticker NOT LIKE '%%/%%'
          AND s.datum::date >= %(cutoff)s
        ORDER BY s.ticker, s.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, {"cutoff": cutoff})
        return cur.fetchall()


# --------------------------------------------------------------------------
# Stap 2: upsert
# --------------------------------------------------------------------------
KOLOMMEN = [
    "ticker", "datum", "market_cap", "trailing_pe", "price_to_book",
    "dividend_yield", "current_ratio", "revenue_growth_pct", "fcf_yield",
    "fcf_years", "fcf_growing", "fcf_consistent", "net_debt_ebitda",
    "payout_pct", "analisten_count", "piotroski_score", "eps_growth_pct",
    "eps_cagr_pct", "peg_ratio",
]


def upsert_rijen(conn, rijen: List[dict]) -> int:
    if not rijen:
        return 0
    kolom_lijst = ", ".join(KOLOMMEN)
    placeholders = ", ".join(f"%({k})s" for k in KOLOMMEN)
    update_lijst = ", ".join(f"{k} = EXCLUDED.{k}" for k in KOLOMMEN if k not in ("ticker", "datum"))
    update_lijst += ", bijgewerkt_op = now()"

    query = f"""
        INSERT INTO generieke_fundamentals ({kolom_lijst})
        VALUES ({placeholders})
        ON CONFLICT (ticker, datum)
        DO UPDATE SET {update_lijst};
    """
    with conn.cursor() as cur:
        for rij in rijen:
            cur.execute(query, {k: rij.get(k) for k in KOLOMMEN})
    conn.commit()
    return len(rijen)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_build():
    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        open_paren = haal_openstaande_paren(conn)
        print(f"{len(open_paren)} recente (ticker, datum)-paren nog niet aanwezig in generieke_fundamentals "
              f"(venster: laatste {RECENTE_DAGEN_LIMIET} dagen -- oudere rijen worden bewust nooit backfilled).")

        if not open_paren:
            print("Niets te doen.")
            return

        per_ticker: Dict[str, List[str]] = {}
        for r in open_paren:
            per_ticker.setdefault(r["ticker"], []).append(r["datum"])

        tickers = sorted(per_ticker.keys())[:MAX_TICKERS_PER_RUN]
        overgeslagen = len(per_ticker) - len(tickers)
        print(f"{len(tickers)} unieke tickers te verwerken dit run"
              + (f" ({overgeslagen} tickers volgen in een volgend run)" if overgeslagen > 0 else ""))

        totaal_bijgewerkt = 0
        for i, ticker in enumerate(tickers, start=1):
            # ÉÉN fundamentals-ophaling per ticker, hergebruikt voor elke
            # recente datum waarop die ticker geselecteerd werd -- fundamentals
            # veranderen niet binnen een paar dagen, dus dit bespaart calls
            # zonder de "geen backfill van oude data"-garantie te schenden.
            fundamentals = haal_fundamentals_op(ticker)
            if fundamentals is not None:
                rijen = [{"ticker": ticker, "datum": d, **fundamentals} for d in per_ticker[ticker]]
                totaal_bijgewerkt += upsert_rijen(conn, rijen)
            else:
                print(f"  [WARN] {ticker}: geen bruikbare fundamentals, overgeslagen")
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} tickers verwerkt...")
            time.sleep(0.15)

        send_telegram(
            f"📊 *Generieke Fundamentals — {vandaag()}*\n\n"
            f"{totaal_bijgewerkt} (ticker, datum)-rijen bijgewerkt\n"
            f"{len(tickers)} unieke tickers verwerkt dit run"
            + (f" ({overgeslagen} tickers nog te gaan)" if overgeslagen > 0 else "")
        )
        print(f"\nKlaar. {totaal_bijgewerkt} rijen bijgewerkt.")
    finally:
        conn.close()


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "build"
    run_build()
