#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00graham.py  —  BENJAMIN GRAHAM "DEFENSIVE INVESTOR" SELECTIE ENGINE v1.2

Screent op de klassieke criteria voor de "defensive investor" uit Benjamin
Grahams The Intelligent Investor, vertaald naar wat via yfinance meetbaar is,
aangevuld met Joseph Piotroski's F-Score als kwaliteitscheck en Peter Lynch's
PEG-ratio als groei-versus-waardering-check.

Criteria (score 0-9):
  1. Adequate omvang       — marketCap >= marktkap_min (groot/gevestigd genoeg,
                              Graham mikte op stabiliteit i.p.v. speculatieve small caps)
  2. Sterke financiële conditie — currentRatio >= current_ratio_min (werkkapitaal-buffer)
  3. Winststabiliteit      — Net Income was in ALLE beschikbare jaren positief
                              (Graham eiste 10 jaar zonder verlies; yfinance geeft
                              doorgaans ~4 jaar resultatenrekening — zie beperking hieronder)
  4. Dividendtrack record  — dividendYield > 0 (bedrijf keert dividend uit; Graham eiste
                              een ononderbroken reeks van 20 jaar, wat via yfinance niet
                              verifieerbaar is — dit is dus een sterk versimpelde proxy)
  5. Winstgroei            — Net Income laatste beschikbare jaar t.o.v. oudste beschikbare
                              jaar >= winstgroei_min_pct (Graham: minstens 1/3 groei over
                              10 jaar; hier geschaald naar de ~4 jaar die yfinance aanbiedt,
                              zie beperking hieronder)
  6. Gematigde koers-winstverhouding — trailingPE <= pe_max
  7. Gematigde koers-boekwaarde / Graham Number — priceToBook <= pb_max
                              OF trailingPE * priceToBook <= graham_multiple_max (22.5,
                              Grahams eigen vuistregel: PE x PB mag niet boven de 22,5 liggen)
  8. Piotroski F-Score     — F-Score (0-9, zie hieronder) >= piotroski_min telt als 1 punt.
                              Ontbreekt de data (< 2 jaar balans/resultatenrekening), dan
                              telt dit criterium niet mee (geen punt, sluit niet uit) —
                              zelfde behandeling als een onbekende waarde bij de andere
                              criteria hierboven.
  9. Peter Lynch PEG-ratio — PEG = trailingPE / eps_cagr_pct (jaarlijkse samengestelde
                              groeivoet, NIET dezelfde grootheid als de totale groei uit
                              criterium 5 — die is over de hele periode, niet per jaar,
                              en is dus expliciet herberekend als CAGR voor deze PEG).
                              PEG < peg_max (Lynch's vuistregel: PEG < 1 is aantrekkelijk)
                              telt als 1 punt. Ontbreekt de data of is de CAGR <= 0
                              (krimpende winst), dan telt dit criterium niet mee.
                              LET OP — overlap met criteria 5 en 6: PEG combineert exact
                              dezelfde twee grootheden (winstgroei en P/E) die Graham al
                              apart scoort. Een aandeel dat daar al goed op scoort haalt
                              dit punt dus vaak automatisch mee — dit criterium voegt
                              minder onafhankelijke informatie toe dan Piotroski (dat wél
                              een nieuwe dimensie — balans/kwaliteitstrend — binnenbrengt).

Piotroski F-Score (0-9, hetzelfde patroon/dezelfde negen boolean-criteria als in
bot_01kasstr.py — winstgevendheid, hefboom/liquiditeit, efficiëntie, telkens dit
jaar vs. vorig jaar). Hier NIET louter informatief zoals in kasstr: het volledige
F-Score wordt gerapporteerd, én een drempel (>= piotroski_min) telt mee als 8ste
scorepunt hierboven. Vereist tk.balance_sheet + tk.financials + tk.cashflow
(2 jaar), dus 2 extra yfinance-calls per ticker t.o.v. v1.0 — een volledige run
duurt daardoor merkelijk langer.

BEPERKINGEN (belangrijk, in tegenstelling tot Grahams oorspronkelijke 10/20-jaar eisen):
  - yfinance levert doorgaans slechts ~4 jaar jaarlijkse resultatenrekening
    (tk.financials) i.p.v. Grahams 10 jaar. Winststabiliteit en winstgroei
    worden dus beoordeeld over de beschikbare periode, niet over 10 jaar.
  - Dividendtrack record is een momentopname (betaalt nu dividend), geen
    verificatie van 20 jaar ononderbroken uitkering.
  - marketCap/koers worden niet omgerekend naar één munt — net als bij de
    andere bots is dit een ruwe grens in de eigen valuta van de ticker.
  - Deduplicatie (v1.3): de 041-059 bestanden overlappen soms (dezelfde
    ticker in meerdere lijsten) — enkel de eerst gescande beurs telt mee,
    latere occurrences van diezelfde ticker worden overgeslagen. Nodig
    gebleken bij bot_00greenblatt/bot_00oshaughnessy (bv. 'ALL' dubbel in
    zowel 048 Nasdaq/NYSE als 057 NYSE); hier preventief toegepast.
  - Piotroski F-Score vereist minstens 2 jaar balans + resultatenrekening;
    ontbreekt die data, dan is de score -1 ("n.v.t.") en telt hij niet mee.

TWEE MODI: enkel 'live' (zoals bot_01kasstr/bot_01hoogl-live) — geen aparte
full-scan, Grahams criteria zijn strikt genoeg om ook op de x-kwaliteitslijsten
te draaien.

Rapportage: Telegram + email, één bericht per beurs, top 5 per beurs,
lege beurzen worden overgeslagen.

Supabase: logt naar de bestaande gedeelde `selecties`-tabel (db_logger.py)
onder strategie "bot_00graham". Parameters current_ratio, eps_years,
eps_all_positive, eps_growth_pct, pe_ratio, pb_ratio en graham_number_ok
moeten aan db_logger.py's _KOLOM_WHITELIST toegevoegd worden — vereist de
bijhorende ALTER TABLE-migratie, zie migratie_graham_kolommen.sql.
piotroski_score is al gewhitelist (toegevoegd bij bot_01kasstr) — geen
nieuwe migratie nodig voor die kolom. piotroski_detail (de per-criterium
breakdown) blijft, net als bij kasstr, enkel in de JSON parameters-kolom
staan. peg_ratio, eps_cagr_pct en peg_ok zijn nieuw in v1.2 — zie
migratie_graham_kolommen_peg.sql.

Gebruik:
  python bot_00graham.py live      # dagelijks rapport (x-lijsten)
  python bot_00graham.py backtest  # niet ondersteund (idem bot_01kasstr/bot_01hoogl:
                                     # geen bruikbare historische fundamentals-reeks
                                     # per scandatum via yfinance) — print uitleg en stopt
"""

import os
import sys
import math
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf
import requests

try:
    from db_logger import log_selectie
except Exception as _e:
    print(f"[WARN] db_logger niet beschikbaar ({_e}) — DB-logging wordt overgeslagen")
    def log_selectie(*args, **kwargs):
        return False

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

BEURS_NAMEN = {
    "tickers_041x.txt": "041 Benelux Ierland",
    "tickers_042x.txt": "042 Parijs",
    "tickers_043x.txt": "043 Frankfurt",
    "tickers_044x.txt": "044 Spanje/Portugal",
    "tickers_045x.txt": "045 Londen",
    "tickers_046x.txt": "046 Milaan",
    "tickers_047x.txt": "047 Toronto",
    "tickers_048x.txt": "048 Nasdaq/NYSE",
    "tickers_049x.txt": "049 Stockholm",
    "tickers_050x.txt": "050 Zurich",
    "tickers_051x.txt": "051 Warschau",
    "tickers_052x.txt": "052 Oslo",
    "tickers_053x.txt": "053 Kopenhagen",
    "tickers_054x.txt": "054 Helsinki",
    "tickers_055x.txt": "055 CBoe",
    "tickers_056x.txt": "056 NYSE int",
    "tickers_057x.txt": "057 NYSE",
    "tickers_058x.txt": "058 TSXV",
    "tickers_059x.txt": "059 Oostenrijk Slovenie Slovakije",
}

def bouw_bestandslijst() -> List[str]:
    """Zelfde dynamische 041-059 opbouw als bot_00kr / bot_01kasstr / bot_01hoogl."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

GRAHAM_CFG = {
    "marktkap_min":        2_000_000_000.0,  # "adequate omvang" — ondergrens (valuta van de ticker zelf)
    "current_ratio_min":   2.0,              # werkkapitaal >= 2x kortlopende schulden
    "winstgroei_min_pct":  10.0,             # groei Net Income over beschikbare periode (~4j i.p.v. Grahams 10j)
    "pe_max":              15.0,
    "pb_max":              1.5,
    "graham_multiple_max": 22.5,             # PE x PB vuistregel
    "piotroski_min":       7.0,              # F-Score >= 7/9 telt als 1 extra punt
    "peg_max":             1.0,              # Lynch: PEG < 1 = aantrekkelijk
    "min_score":           7,                # was 6/8 (75%) — op 7/9 (78%) vergelijkbaar strikt
    "top_n":               5,
    "strategie":           "bot_00graham",
    "throttle_sec":        0.15,
}

# ============================================================
# HULPFUNCTIES  (identiek patroon aan bot_01kasstr.py / bot_01hoogl.py)
# ============================================================

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram fout: {e}")

def send_email(subject: str, body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        clean = body.replace("*", "").replace("`", "").replace("•", "-").replace("_", "")
        msg.attach(MIMEText(clean, "plain", "utf-8"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        print(f"Email verzonden naar {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email fout: {e}")

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# FUNDAMENTELE DATA — WINSTSTABILITEIT & WINSTGROEI (Net Income-reeks)
# ============================================================

def _row(df, names: List[str]):
    """Zoekt de eerste bestaande rij (op naam) in een yfinance-DataFrame.
    yfinance-versies verschillen in exacte rijnamen, dus meerdere varianten proberen."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None

def _net_income_series(financials) -> List[float]:
    """Net Income per beschikbaar jaar, oudste eerst."""
    row = _row(financials, ["Net Income", "Net Income Common Stockholders"])
    if row is None:
        return []
    vals = [safe_float(v) for v in row.tolist()]
    vals = [v for v in vals if not math.isnan(v)]
    return list(reversed(vals))  # yfinance-kolommen staan nieuwste-eerst -> omdraaien


def _piotroski_f_score(tk, cashflow):
    """Berekent Piotroski F-Score (0-9) op basis van dit jaar vs. vorig jaar.
    Identieke implementatie als bot_01kasstr.py. Retourneert (score, label,
    detail_dict). Score = -1 als er geen 2 jaar balans + resultatenrekening
    beschikbaar is via yfinance."""
    try:
        bs  = tk.balance_sheet
        inc = tk.financials
    except Exception:
        return -1, "n.v.t. (data-fout)", {}

    if bs is None or bs.empty or inc is None or inc.empty:
        return -1, "n.v.t. (geen balans/resultaten)", {}
    if bs.shape[1] < 2 or inc.shape[1] < 2:
        return -1, "n.v.t. (< 2 jaar data)", {}

    def val(df, names, col):
        row = _row(df, names)
        if row is None or col >= len(row):
            return float("nan")
        return safe_float(row.iloc[col])

    ta0, ta1   = val(bs, ["Total Assets"], 0), val(bs, ["Total Assets"], 1)
    ca0, ca1   = val(bs, ["Current Assets", "Total Current Assets"], 0), val(bs, ["Current Assets", "Total Current Assets"], 1)
    cl0, cl1   = val(bs, ["Current Liabilities", "Total Current Liabilities"], 0), val(bs, ["Current Liabilities", "Total Current Liabilities"], 1)
    ltd0, ltd1 = val(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0), val(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
    sh0, sh1   = val(bs, ["Share Issued", "Ordinary Shares Number"], 0), val(bs, ["Share Issued", "Ordinary Shares Number"], 1)

    ni0, ni1   = val(inc, ["Net Income"], 0), val(inc, ["Net Income"], 1)
    rev0, rev1 = val(inc, ["Total Revenue"], 0), val(inc, ["Total Revenue"], 1)
    gp0, gp1   = val(inc, ["Gross Profit"], 0), val(inc, ["Gross Profit"], 1)

    ocf_row = _row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
    ocf0 = safe_float(ocf_row.iloc[0]) if ocf_row is not None and len(ocf_row) > 0 else float("nan")

    score, vlaggen = 0, {}

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

    score = sum(1 for v in vlaggen.values() if v)
    kort = {
        "roa_positief": "ROA", "ocf_positief": "OCF", "roa_gestegen": "ΔROA",
        "winstkwaliteit": "Accruals", "hefboom_gedaald": "Hefboom", "liquiditeit_gestegen": "Liquid.",
        "geen_verwatering": "GeenDilutie", "marge_gestegen": "Marge", "efficientie_gestegen": "Efficiëntie",
    }
    label = f"{score}/9 (" + " ".join((("✓" if vlaggen[k] else "✗") + kort[k]) for k in kort) + ")"
    return score, label, vlaggen


@dataclass
class GrahamSignaal:
    ticker:              str
    price:               float
    score:               int
    market_cap:          float
    current_ratio:       float
    eps_years:           int
    eps_all_positive:    bool
    div_yield:           float
    eps_growth_pct:      float
    pe_ratio:            float
    pb_ratio:            float
    graham_number_ok:    bool
    piotroski_score:     float
    piotroski_ok:        bool
    eps_cagr_pct:        float
    peg_ratio:           float
    peg_ok:              bool
    marktkap_label:      str
    current_ratio_label: str
    stabiliteit_label:   str
    dividend_label:      str
    groei_label:         str
    pe_label:            str
    pb_label:            str
    piotroski_label:     str
    peg_label:           str

def analyse_ticker(ticker: str, cfg: dict) -> Optional[GrahamSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        market_cap    = safe_float(info.get("marketCap"))
        current_ratio = safe_float(info.get("currentRatio"))
        div_yield     = safe_float(info.get("dividendYield"), 0.0)
        pe_ratio      = safe_float(info.get("trailingPE"))
        pb_ratio      = safe_float(info.get("priceToBook"))

        score = 0

        # 1. Adequate omvang
        if not math.isnan(market_cap) and market_cap >= cfg["marktkap_min"]:
            score += 1
            marktkap_label = f"✓ {market_cap/1e9:.2f}B (>= {cfg['marktkap_min']/1e9:.0f}B)"
        else:
            marktkap_label = f"✗ {market_cap/1e9:.2f}B" if not math.isnan(market_cap) else "✗ onbekend"

        # 2. Sterke financiële conditie
        if not math.isnan(current_ratio) and current_ratio >= cfg["current_ratio_min"]:
            score += 1
            current_ratio_label = f"✓ {current_ratio:.2f} (>= {cfg['current_ratio_min']:.1f})"
        else:
            current_ratio_label = f"✗ {current_ratio:.2f}" if not math.isnan(current_ratio) else "✗ onbekend"

        # 3 & 5. Winststabiliteit + winstgroei — op basis van de Net Income-reeks
        try:
            financials = tk.financials
        except Exception:
            financials = None
        ni_series = _net_income_series(financials)
        eps_years = len(ni_series)

        if eps_years >= 2:
            eps_all_positive = all(v > 0 for v in ni_series)
            if eps_all_positive:
                score += 1
                stabiliteit_label = f"✓ {eps_years}/{eps_years} jaar winst positief"
            else:
                stabiliteit_label = f"✗ {sum(1 for v in ni_series if v > 0)}/{eps_years} jaar winst positief"

            oudste, nieuwste = ni_series[0], ni_series[-1]
            if oudste > 0:
                eps_growth_pct = (nieuwste - oudste) / oudste * 100
            else:
                eps_growth_pct = float("nan")

            if not math.isnan(eps_growth_pct) and eps_growth_pct >= cfg["winstgroei_min_pct"]:
                score += 1
                groei_label = f"✓ {eps_growth_pct:+.1f}% over {eps_years} jaar"
            else:
                groei_label = f"✗ {eps_growth_pct:+.1f}% over {eps_years} jaar" if not math.isnan(eps_growth_pct) else "✗ onbekend"

            # CAGR (jaarlijkse samengestelde groeivoet) t.b.v. PEG — bewust NIET
            # hetzelfde als eps_growth_pct hierboven (dat is totale groei over de hele periode)
            if oudste > 0 and nieuwste > 0 and eps_years >= 2:
                eps_cagr_pct = ((nieuwste / oudste) ** (1 / (eps_years - 1)) - 1) * 100
            else:
                eps_cagr_pct = float("nan")
        else:
            eps_all_positive = False
            eps_growth_pct = float("nan")
            eps_cagr_pct = float("nan")
            stabiliteit_label = "✗ onbekend (< 2 jaar resultatenrekening)"
            groei_label = "✗ onbekend (< 2 jaar resultatenrekening)"

        # 4. Dividendtrack record (versimpelde proxy — zie beperking in module-docstring)
        if not math.isnan(div_yield) and div_yield > 0:
            score += 1
            dividend_label = f"✓ yield {div_yield:.2f}%"
        else:
            dividend_label = "✗ geen dividend"

        # 6. Gematigde P/E
        if not math.isnan(pe_ratio) and 0 < pe_ratio <= cfg["pe_max"]:
            score += 1
            pe_label = f"✓ {pe_ratio:.1f}x (<= {cfg['pe_max']:.0f}x)"
        else:
            pe_label = f"✗ {pe_ratio:.1f}x" if not math.isnan(pe_ratio) else "✗ onbekend"

        # 7. Gematigde P/B of Graham Number (PE x PB <= 22,5)
        graham_multiple = pe_ratio * pb_ratio if not math.isnan(pe_ratio) and not math.isnan(pb_ratio) else float("nan")
        pb_ok = not math.isnan(pb_ratio) and 0 < pb_ratio <= cfg["pb_max"]
        multiple_ok = not math.isnan(graham_multiple) and graham_multiple <= cfg["graham_multiple_max"]
        graham_number_ok = pb_ok or multiple_ok
        if graham_number_ok:
            score += 1
            pb_label = f"✓ P/B {pb_ratio:.2f} | PExP/B {graham_multiple:.1f} (<= {cfg['graham_multiple_max']:.1f})"
        else:
            pb_label = (
                f"✗ P/B {pb_ratio:.2f} | PExP/B {graham_multiple:.1f}"
                if not math.isnan(graham_multiple) else "✗ onbekend"
            )

        # 8. Piotroski F-Score >= piotroski_min telt als 1 punt (ontbrekende data: geen punt, geen uitsluiting)
        try:
            cashflow = tk.cashflow
        except Exception:
            cashflow = None
        piotroski_score, piotroski_label, _piotroski_vlaggen = _piotroski_f_score(tk, cashflow)
        piotroski_ok = piotroski_score >= cfg["piotroski_min"]
        if piotroski_ok:
            score += 1

        # 9. Peter Lynch PEG-ratio (PE / jaarlijkse CAGR) — zie module-docstring
        # voor de kanttekening over overlap met criteria 5 en 6.
        if not math.isnan(pe_ratio) and pe_ratio > 0 and not math.isnan(eps_cagr_pct) and eps_cagr_pct > 0:
            peg_ratio = pe_ratio / eps_cagr_pct
            peg_ok = peg_ratio < cfg["peg_max"]
            if peg_ok:
                score += 1
                peg_label = f"✓ {peg_ratio:.2f} (<{cfg['peg_max']:.1f}, CAGR {eps_cagr_pct:.1f}%/j)"
            else:
                peg_label = f"✗ {peg_ratio:.2f} (CAGR {eps_cagr_pct:.1f}%/j)"
        else:
            peg_ratio = float("nan")
            peg_ok = False
            peg_label = "✗ onbekend (geen positieve CAGR of P/E)"

        return GrahamSignaal(
            ticker=ticker, price=round(price, 2), score=score,
            market_cap=round(market_cap, 0) if not math.isnan(market_cap) else 0.0,
            current_ratio=round(current_ratio, 2) if not math.isnan(current_ratio) else 0.0,
            eps_years=eps_years, eps_all_positive=eps_all_positive,
            div_yield=round(div_yield, 2) if not math.isnan(div_yield) else 0.0,
            eps_growth_pct=round(eps_growth_pct, 1) if not math.isnan(eps_growth_pct) else 0.0,
            pe_ratio=round(pe_ratio, 1) if not math.isnan(pe_ratio) else 0.0,
            pb_ratio=round(pb_ratio, 2) if not math.isnan(pb_ratio) else 0.0,
            graham_number_ok=graham_number_ok,
            piotroski_score=piotroski_score, piotroski_ok=piotroski_ok,
            eps_cagr_pct=round(eps_cagr_pct, 1) if not math.isnan(eps_cagr_pct) else 0.0,
            peg_ratio=round(peg_ratio, 2) if not math.isnan(peg_ratio) else 0.0,
            peg_ok=peg_ok,
            marktkap_label=marktkap_label, current_ratio_label=current_ratio_label,
            stabiliteit_label=stabiliteit_label, dividend_label=dividend_label,
            groei_label=groei_label, pe_label=pe_label, pb_label=pb_label,
            piotroski_label=piotroski_label, peg_label=peg_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één bericht per exchange
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (9 - score) + f" {score}/9"

def format_bericht(exchange_name: str, signalen: List[GrahamSignaal], alle: List[GrahamSignaal], cfg: dict) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None

    nu     = today_str()
    top_n  = cfg["top_n"]
    top_tonen = sorted(alle, key=lambda s: (s.score, -s.pe_ratio if s.pe_ratio > 0 else -999), reverse=True)[:top_n]
    max_sc = max((s.score for s in signalen), default=0) if signalen else 0
    lbl    = {9: "⭐ PERFECTE SCORE (9/9)", 8: "🟡 STERK (8/9)", 7: "🟠 GOED (7/9)"}.get(max_sc, "📊")

    def sig_regel(s: GrahamSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"P/E:{s.pe_ratio:.1f} | P/B:{s.pb_ratio:.2f} | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += (
                f"\n  {s.marktkap_label} | {s.current_ratio_label}"
                f"\n  Winst: {s.stabiliteit_label} | Groei: {s.groei_label}"
                f"\n  Dividend: {s.dividend_label} | Waardering: {s.pb_label}"
                f"\n  Piotroski F-Score: {s.piotroski_label}"
                f"\n  PEG (Lynch): {s.peg_label}"
            )
        return r

    delen = [
        f"📜 *GRAHAM DEFENSIVE INVESTOR — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(signalen)} kandidaten (score>={cfg['min_score']})_",
        "─────────────────────────────",
        f"🏆 *TOP {top_n} HOOGSTE SCORE:*",
        "\n\n".join(sig_regel(s, detail=True) for s in top_tonen),
    ]

    overige = [s for s in signalen if s not in top_tonen]
    if overige:
        delen += ["─────────────────────────────", f"*{lbl} — overige kandidaten:*"]
        for s in overige:
            delen.append(sig_regel(s))

    delen.append(
        f"⚙️ _marktkap>={cfg['marktkap_min']/1e9:.0f}B | current ratio>={cfg['current_ratio_min']:.1f} | "
        f"winst positief alle jaren | dividend uitkerend | winstgroei>={cfg['winstgroei_min_pct']:.0f}% | "
        f"P/E<={cfg['pe_max']:.0f}x | P/B<={cfg['pb_max']:.1f} of PExP/B<={cfg['graham_multiple_max']:.1f} | "
        f"Piotroski F-Score>={cfg['piotroski_min']:.0f}/9 | PEG<{cfg['peg_max']:.1f}_"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    cfg = GRAHAM_CFG
    print(f"{'='*60}")
    print(f"GRAHAM DEFENSIVE INVESTOR — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    for f_name in bouw_bestandslijst():
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name = label_voor(f_name)
        exchange_tickers[ex_name] = tlist
        print(f"  {ex_name}: {len(tlist)} tickers")

    if not exchange_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    email_delen: List[str] = []
    reeds_verwerkt: set = set()  # dedupliceert tickers die in meerdere 041-059 bestanden
                                  # voorkomen (overlap tussen beurslijsten) — zonder dit zou
                                  # zo'n ticker in twee beurzen se top-5 tegelijk kunnen
                                  # verschijnen. Eerste beurs waarin de ticker voorkomt wint.
    totaal_gedupliceerd = 0

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")

        alle: List[GrahamSignaal] = []
        for ticker in tlist:
            if ticker in reeds_verwerkt:
                totaal_gedupliceerd += 1
                continue
            reeds_verwerkt.add(ticker)
            sig = analyse_ticker(ticker, cfg)
            if sig is not None:
                alle.append(sig)
                if sig.score >= cfg["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/9 | P/E={sig.pe_ratio:.1f} | P/B={sig.pb_ratio:.2f}")
            time.sleep(cfg["throttle_sec"])  # lichte throttle tegen Yahoo rate-limits

        kandidaten = [s for s in alle if s.score >= cfg["min_score"]]
        kandidaten.sort(key=lambda s: (s.score, -s.pe_ratio if s.pe_ratio > 0 else -999), reverse=True)
        signalen = kandidaten[:cfg["top_n"]]

        print(f"  → top {len(signalen)} van {len(kandidaten)} kandidaten (score >= {cfg['min_score']}) uit {len(alle)} geanalyseerd")

        for rank, s in enumerate(signalen, start=1):
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie=cfg["strategie"],
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "rank": rank,
                    "market_cap": s.market_cap,
                    "current_ratio": s.current_ratio,
                    "eps_years": s.eps_years,
                    "eps_all_positive": s.eps_all_positive,
                    "div_yield": s.div_yield,
                    "eps_growth_pct": s.eps_growth_pct,
                    "pe_ratio": s.pe_ratio,
                    "pb_ratio": s.pb_ratio,
                    "graham_number_ok": s.graham_number_ok,
                    "piotroski_score": s.piotroski_score,
                    "piotroski_detail": s.piotroski_label,
                    "eps_cagr_pct": s.eps_cagr_pct,
                    "peg_ratio": s.peg_ratio,
                    "peg_ok": s.peg_ok,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, signalen, alle, cfg)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")
        else:
            print(f"  → Overgeslagen: {ex_name}")

    if email_delen:
        send_email(
            f"Graham Defensive Investor rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    if totaal_gedupliceerd:
        print(f"\nDeduplicatie: {totaal_gedupliceerd} dubbele ticker-occurrence(s) overgeslagen "
              f"(overlap tussen beursbestanden) — elke ticker enkel meegeteld in de eerst "
              f"gescande beurs waarin hij voorkomt.")

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_00graham: yfinance biedt geen "
        "betrouwbare historische reeks van fundamentele data (current ratio, "
        "winstreeks, P/E, P/B) per scandatum in het verleden, enkel de laatste "
        "~4 jaarrapporten en de actuele ratio's. Gebruik 'live' om het huidige "
        "universum te screenen."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
