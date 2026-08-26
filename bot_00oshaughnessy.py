#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00oshaughnessy.py  —  O'SHAUGHNESSY "TRENDING VALUE" RANKING ENGINE v1.0

Implementeert de "Trending Value"-strategie uit James O'Shaughnessys What
Works on Wall Street: net als bot_01greenblatt GEEN drempel-score, maar een
tweetrapse GLOBALE RANKING over het volledige gescande universum:

  STAP 1 — VALUE COMPOSITE TWO (VC2): elk aandeel krijgt op elk van de
  onderstaande 7 ratio's een PERCENTIEL binnen het volledige universum
  (100 = goedkoopst/beste, 0 = duurst/slechtste). VC2 = gemiddelde van de
  beschikbare percentielen (minstens 5 van de 7 vereist, anders uitgesloten):
    - P/E            (trailingPE)                          — lager = beter
    - P/B            (priceToBook)                          — lager = beter
    - P/S            (priceToSalesTrailing12Months)          — lager = beter
    - P/CF           (marketCap / Operating Cash Flow)       — lager = beter
      (toegevoegd na een vraag over David Dremans contrarian-methodiek —
      Dreman rangschikt op de laagste P/E-, P/B- of P/CF-deciles van de
      markt; P/E en P/B zaten al in VC2, P/CF ontbrak nog en is hier
      toegevoegd als 7de factor i.p.v. een aparte Dreman-bot, aangezien
      een losstaande bot grotendeels dezelfde aandelen zou opleveren)
    - EV/EBITDA      (enterpriseValue / EBITDA)              — lager = beter
    - EV/FCF         (enterpriseValue / Free Cash Flow)      — lager = beter
    - Shareholder Yield = dividendYield + buyback-yield      — hoger = beter
      (buyback-yield = % afname uitstaande aandelen t.o.v. vorig jaar)

  STAP 2 — TRENDING: enkel binnen het goedkoopste decile — d.w.z. de top
  value_percentile_min% (default 90) VAN DE VC2-SCORES ONDERLING GERANGSCHIKT
  in dit universum, NIET een absolute VC2-score van 90 op de 0-100-schaal
  (een gemiddelde van 6 percentielen clustert vanzelf rond 50, dus een
  absolute drempel van 90 zou de decile leegtrekken) — wordt vervolgens
  gerangschikt op 6-maands prijsmomentum (O'Shaughnessys eigen keuze in het
  boek). De top N (default 25 — de portefeuillegrootte die hij voor
  Trending Value hanteert) met het beste momentum wordt gerapporteerd.

  Koershistoriek (voor stap 2) wordt bewust ENKEL opgehaald voor de
  overlevers van stap 1 — niet voor het volledige universum — om het
  aantal yfinance-calls beheersbaar te houden.

AFWIJKING VAN HET GEBRUIKELIJKE PER-BEURS-PATROON (bewust, zelfde reden als
bot_01greenblatt): dit is een globale cross-market ranking, geen per-beurs
top-5. Percentielen zijn per definitie universum-breed (een percentiel
binnen slechts 40 tickers van 1 beurs betekent iets anders dan binnen 1900
tickers), dus rapportage gebeurt in de plaats via één (opgesplitst)
Telegram-bericht + één samenvattende e-mail.

UITSLUITINGEN:
  - marketCap < marktkap_min (default 200M, O'Shaughnessys eigen
    benadering van zijn "All Stocks"-universum in het boek).
  - Minder dan 4 van de 6 VC2-ratio's beschikbaar/zinvol (bv. negatieve
    EBITDA of FCF maakt die ratio onbruikbaar als "goedkoop"-signaal).
  GEEN sector-uitsluiting (in tegenstelling tot bot_01greenblatt) —
  O'Shaughnessys eigen tests in het boek sluiten financials niet
  systematisch uit voor de Value Composite; dit is een bewuste
  methodologische keuze, geen omissie. Let op: EV/EBITDA en P/B kunnen
  voor financiële instellingen wel minder betekenisvol zijn door hun
  hefboomstructuur — dit is een gekende beperking van de originele
  methodiek zelf, niet iets wat dit script probeert te compenseren.

BEPERKINGEN:
  - Deduplicatie (v1.1): bevestigd nodig na de eerste live run (2026-08-25) —
    'ALL' verscheen dubbel in de top 25 (048 Nasdaq/NYSE + 057 NYSE, exact
    dezelfde VC2/momentum). dedupliceer_op_ticker() houdt enkel de eerst
    gescande occurrence over, vóór VC2-percentielen berekend worden.
  - EBITDA/FCF/aandelenaantal komen uit het laatste jaarrapport
    (tk.financials / tk.cashflow / tk.balance_sheet), niet TTM.
  - Percentielen worden herberekend bij elke run op het dan gescande
    universum — dit is geen vaste backtest-ranking uit het boek zelf.

Rapportage: Telegram (globale top N, in blokken van 15) + samenvattende
e-mail. Geen CSV.

Supabase: logt de top N naar de bestaande gedeelde `selecties`-tabel onder
strategie "bot_01oshaughnessy". pe_ratio en pb_ratio zijn al gewhitelist
(bot_01graham) — ps_ratio, ev_ebitda, ev_fcf, shareholder_yield, vc2_score
en momentum_6m_pct via migratie_oshaughnessy_kolommen.sql; pcf_ratio (v1.1,
de Dreman-factor) via migratie_oshaughnessy_kolommen_pcf.sql.

Gebruik:
  python bot_01oshaughnessy.py live      # wekelijkse globale ranking
  python bot_01oshaughnessy.py backtest  # niet ondersteund (zelfde reden
                                            # als bot_01greenblatt/bot_01graham)
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
import pandas as pd
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
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

OSHAUGHNESSY_CFG = {
    "marktkap_min":         200_000_000.0,  # O'Shaughnessys "All Stocks"-benadering
    "min_vc2_ratios":       5,              # minstens 5 van de 7 VC2-ratio's nodig
    "value_percentile_min": 90.0,           # goedkoopste decile gaat door naar stap 2
    "momentum_maanden":     6,              # O'Shaughnessys eigen keuze voor Trending Value
    "top_n_global":         25,             # O'Shaughnessys portefeuillegrootte voor Trending Value
    "telegram_chunk":       15,
    "strategie":            "bot_01oshaughnessy",
    "throttle_sec":         0.15,
}

# ============================================================
# HULPFUNCTIES (identiek patroon aan de andere bots)
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

def _row(df, names: List[str]):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


# ============================================================
# STAP 1a — RUWE RATIO'S PER TICKER (nog geen percentielen)
# ============================================================

@dataclass
class RuweSignaal:
    ticker:            str
    exchange:          str
    price:             float
    pe:                float  # nan toegestaan (ontbrekend)
    pb:                float
    ps:                float
    pcf:               float
    ev_ebitda:         float
    ev_fcf:            float
    shareholder_yield: float

def analyse_ticker_ruw(ticker: str, exchange: str, cfg: dict) -> Optional[RuweSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        market_cap = safe_float(info.get("marketCap"))
        if math.isnan(market_cap) or market_cap < cfg["marktkap_min"]:
            return None

        pe = safe_float(info.get("trailingPE"))
        pb = safe_float(info.get("priceToBook"))
        ps = safe_float(info.get("priceToSalesTrailing12Months"))

        ev = safe_float(info.get("enterpriseValue"))
        if math.isnan(ev) or ev <= 0:
            total_debt = safe_float(info.get("totalDebt"), 0.0)
            total_cash = safe_float(info.get("totalCash"), 0.0)
            ev = market_cap + total_debt - total_cash

        try:
            inc = tk.financials
            bs  = tk.balance_sheet
            cf  = tk.cashflow
        except Exception:
            inc, bs, cf = None, None, None

        # EV/EBITDA: eerst yfinance's eigen 'enterpriseToEbitda', dan EBITDA-rij, dan EBIT+D&A
        ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
        if math.isnan(ev_ebitda) or ev_ebitda <= 0:
            ebitda_row = _row(inc, ["EBITDA"]) if inc is not None else None
            ebitda = safe_float(ebitda_row.iloc[0]) if ebitda_row is not None and len(ebitda_row) > 0 else float("nan")
            if math.isnan(ebitda):
                ebit_row = _row(inc, ["EBIT", "Operating Income"]) if inc is not None else None
                da_row   = _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]) if cf is not None else None
                ebit = safe_float(ebit_row.iloc[0]) if ebit_row is not None and len(ebit_row) > 0 else float("nan")
                da   = safe_float(da_row.iloc[0]) if da_row is not None and len(da_row) > 0 else float("nan")
                ebitda = ebit + da if not math.isnan(ebit) and not math.isnan(da) else float("nan")
            ev_ebitda = ev / ebitda if not math.isnan(ebitda) and ebitda > 0 and not math.isnan(ev) and ev > 0 else float("nan")

        # EV/FCF: FCF = Operating Cash Flow - CapEx
        ocf_row   = _row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]) if cf is not None else None
        capex_row = _row(cf, ["Capital Expenditure", "Capital Expenditures"]) if cf is not None else None
        ocf   = safe_float(ocf_row.iloc[0]) if ocf_row is not None and len(ocf_row) > 0 else float("nan")
        capex = safe_float(capex_row.iloc[0]) if capex_row is not None and len(capex_row) > 0 else float("nan")
        if not math.isnan(ocf) and not math.isnan(capex):
            fcf = ocf - abs(capex)
            ev_fcf = ev / fcf if fcf > 0 and not math.isnan(ev) and ev > 0 else float("nan")
        else:
            ev_fcf = float("nan")

        # P/CF: marketCap / Operating Cash Flow (Dreman-factor, hergebruikt dezelfde OCF als EV/FCF)
        pcf = market_cap / ocf if not math.isnan(ocf) and ocf > 0 and not math.isnan(market_cap) else float("nan")

        # P/S fallback via financials als yfinance's eigen veld ontbreekt
        if math.isnan(ps) or ps <= 0:
            rev_row = _row(inc, ["Total Revenue"]) if inc is not None else None
            rev = safe_float(rev_row.iloc[0]) if rev_row is not None and len(rev_row) > 0 else float("nan")
            ps = market_cap / rev if not math.isnan(rev) and rev > 0 else float("nan")

        # Shareholder yield = dividendYield (al in %) + buyback-yield (afname aandelenaantal, in %)
        div_yield = safe_float(info.get("dividendYield"), 0.0)
        sh_row = _row(bs, ["Share Issued", "Ordinary Shares Number"]) if bs is not None else None
        if sh_row is not None and len(sh_row) >= 2:
            sh_nu, sh_vorig = safe_float(sh_row.iloc[0]), safe_float(sh_row.iloc[1])
            buyback_yield = (sh_vorig - sh_nu) / sh_vorig * 100 if sh_vorig > 0 else 0.0
        else:
            buyback_yield = 0.0
        shareholder_yield = div_yield + buyback_yield

        return RuweSignaal(
            ticker=ticker, exchange=exchange, price=round(price, 2),
            pe=pe if not math.isnan(pe) and pe > 0 else float("nan"),
            pb=pb if not math.isnan(pb) and pb > 0 else float("nan"),
            ps=ps if not math.isnan(ps) and ps > 0 else float("nan"),
            pcf=pcf if not math.isnan(pcf) and pcf > 0 else float("nan"),
            ev_ebitda=ev_ebitda if not math.isnan(ev_ebitda) and ev_ebitda > 0 else float("nan"),
            ev_fcf=ev_fcf if not math.isnan(ev_fcf) and ev_fcf > 0 else float("nan"),
            shareholder_yield=round(shareholder_yield, 2),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# STAP 1b — VC2 PERCENTIELEN (universum-breed, via pandas)
# ============================================================

def dedupliceer_op_ticker(alle: List[RuweSignaal]) -> List[RuweSignaal]:
    """Zelfde reden als bot_01greenblatt.py: tickerbestanden 041-059 overlappen
    deels, zonder dit zou een ticker dubbel in de VC2-ranking (en dus mogelijk
    dubbel in de top N) terechtkomen. Behoudt de eerste occurrence."""
    gezien = set()
    resultaat = []
    for s in alle:
        if s.ticker in gezien:
            continue
        gezien.add(s.ticker)
        resultaat.append(s)
    return resultaat


def bereken_vc2(alle: List[RuweSignaal], cfg: dict) -> pd.DataFrame:
    """Bouwt een DataFrame met percentielen per VC2-ratio en de VC2-compositescore.
    100 = beste/goedkoopste percentiel op elke ratio. Rijen met minder dan
    min_vc2_ratios geldige ratio's worden verwijderd."""
    df = pd.DataFrame([{
        "ticker": s.ticker, "exchange": s.exchange, "price": s.price,
        "pe": s.pe, "pb": s.pb, "ps": s.ps, "pcf": s.pcf, "ev_ebitda": s.ev_ebitda,
        "ev_fcf": s.ev_fcf, "shareholder_yield": s.shareholder_yield,
    } for s in alle])

    goedkoop_lager_is_beter = ["pe", "pb", "ps", "pcf", "ev_ebitda", "ev_fcf"]
    for kol in goedkoop_lager_is_beter:
        df[f"pct_{kol}"] = df[kol].rank(pct=True, ascending=False) * 100
    df["pct_shareholder_yield"] = df["shareholder_yield"].rank(pct=True, ascending=True) * 100

    pct_kolommen = [f"pct_{k}" for k in goedkoop_lager_is_beter] + ["pct_shareholder_yield"]
    df["vc2_geldige_ratios"] = df[pct_kolommen].notna().sum(axis=1)
    df["vc2_score"] = df[pct_kolommen].mean(axis=1, skipna=True)

    df = df[df["vc2_geldige_ratios"] >= cfg["min_vc2_ratios"]].copy()
    return df


# ============================================================
# STAP 2 — MOMENTUM (enkel voor de goedkoopste decile)
# ============================================================

def bereken_momentum_pct(ticker: str, maanden: int) -> float:
    try:
        hist = yf.Ticker(ticker).history(period=f"{maanden}mo")
        if hist is None or hist.empty or len(hist) < 2:
            return float("nan")
        start, eind = safe_float(hist["Close"].iloc[0]), safe_float(hist["Close"].iloc[-1])
        if math.isnan(start) or start <= 0:
            return float("nan")
        return (eind - start) / start * 100
    except Exception:
        return float("nan")


@dataclass
class TrendingValueSignaal:
    ticker:            str
    exchange:          str
    price:             float
    pe:                float
    pb:                float
    ps:                float
    pcf:               float
    ev_ebitda:         float
    ev_fcf:            float
    shareholder_yield: float
    vc2_score:         float
    momentum_pct:      float


# ============================================================
# TELEGRAM + EMAIL OUTPUT
# ============================================================

def _sig_regel(s: TrendingValueSignaal, rank: int) -> str:
    def fmt(v):
        return f"{v:.1f}" if not math.isnan(v) else "n/b"
    return (
        f"{rank}. `{s.ticker}` ({s.exchange}) | VC2:{s.vc2_score:.0f} | Mom6m:{s.momentum_pct:+.1f}% | "
        f"PE:{fmt(s.pe)} PB:{fmt(s.pb)} PS:{fmt(s.ps)} PCF:{fmt(s.pcf)} EV/EBITDA:{fmt(s.ev_ebitda)} EV/FCF:{fmt(s.ev_fcf)} "
        f"SY:{s.shareholder_yield:.1f}% | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}"
    )

def bouw_telegram_berichten(top: List[TrendingValueSignaal], universum_grootte: int, waarde_decile_grootte: int, cfg: dict) -> List[str]:
    nu = today_str()
    chunk = cfg["telegram_chunk"]
    blokken = [top[i:i + chunk] for i in range(0, len(top), chunk)]
    berichten = []
    for idx, blok in enumerate(blokken, start=1):
        regels = [
            f"📈 *TRENDING VALUE (O'Shaughnessy) — GLOBALE TOP {len(top)}*  ({idx}/{len(blokken)})",
            f"_{nu} | {universum_grootte} tickers gescreend | {waarde_decile_grootte} in goedkoopste VC2-decile_",
            "─────────────────────────────",
        ]
        start_rank = (idx - 1) * chunk + 1
        for offset, s in enumerate(blok):
            regels.append(_sig_regel(s, start_rank + offset))
        if idx == len(blokken):
            regels.append("─────────────────────────────")
            regels.append(
                "⚙️ _VC2 = gemiddeld percentiel over P/E, P/B, P/S, P/CF, EV/EBITDA, EV/FCF (lager=beter) "
                "en Shareholder Yield (hoger=beter), 100=beste. Enkel de goedkoopste "
                f"{100 - cfg['value_percentile_min']:.0f}% VAN DEZE RUN (relatieve rang, geen absolute "
                f"VC2-drempel) gaat door naar de {cfg['momentum_maanden']}-maands momentumranking. "
                f"marketCap>={cfg['marktkap_min']/1e6:.0f}M._"
            )
        berichten.append("\n".join(regels))
    return berichten


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    cfg = OSHAUGHNESSY_CFG
    print(f"{'='*60}")
    print(f"O'SHAUGHNESSY TRENDING VALUE — LIVE  {today_str()}")
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

    # --- STAP 1a: ruwe ratio's over het volledige universum ---
    alle_ruw: List[RuweSignaal] = []
    universum_grootte = 0
    for ex_name, tlist in exchange_tickers.items():
        print(f"\nStap 1 — ratio's ophalen: {ex_name} ({len(tlist)} tickers)...")
        gevonden = 0
        for ticker in tlist:
            universum_grootte += 1
            sig = analyse_ticker_ruw(ticker, ex_name, cfg)
            if sig is not None:
                alle_ruw.append(sig)
                gevonden += 1
            time.sleep(cfg["throttle_sec"])
        print(f"  → {gevonden}/{len(tlist)} tickers met marketCap/prijsdata OK")

    if not alle_ruw:
        print("[ERROR] Geen enkele ticker doorstond de basisfilters.")
        return

    aantal_voor_dedup = len(alle_ruw)
    alle_ruw = dedupliceer_op_ticker(alle_ruw)
    if len(alle_ruw) < aantal_voor_dedup:
        print(f"Deduplicatie: {aantal_voor_dedup - len(alle_ruw)} dubbele ticker(s) verwijderd "
              f"(overlap tussen beursbestanden) — {len(alle_ruw)} unieke tickers over.")

    # --- STAP 1b: VC2-percentielen ---
    df = bereken_vc2(alle_ruw, cfg)
    print(f"\nVC2 berekend voor {len(df)}/{len(alle_ruw)} tickers (>= {cfg['min_vc2_ratios']}/7 ratio's geldig)")

    drempel = df["vc2_score"].quantile(cfg["value_percentile_min"] / 100.0)
    waarde_decile = df[df["vc2_score"] >= drempel].copy()
    # (drempel is de 90e-percentielwaarde van de VC2-compositescore ZELF binnen dit
    #  universum — niet de rauwe constante 90 op de 0-100-schaal. Een gemiddelde van
    #  6 percentielen clustert immers vanzelf rond 50, dus vrijwel nooit >=90 in
    #  absolute zin; de relatieve top-10%-by-rank is wat O'Shaughnessy bedoelt met
    #  "goedkoopste decile".)
    if waarde_decile.empty:
        waarde_decile = df.sort_values("vc2_score", ascending=False).head(max(int(len(df) * 0.1), cfg["top_n_global"]))

    print(f"Goedkoopste VC2-decile: {len(waarde_decile)} tickers → momentum ophalen...")

    # --- STAP 2: momentum enkel voor de decile ---
    trending: List[TrendingValueSignaal] = []
    for _, rij in waarde_decile.iterrows():
        mom = bereken_momentum_pct(rij["ticker"], cfg["momentum_maanden"])
        if math.isnan(mom):
            continue
        trending.append(TrendingValueSignaal(
            ticker=rij["ticker"], exchange=rij["exchange"], price=rij["price"],
            pe=rij["pe"], pb=rij["pb"], ps=rij["ps"], pcf=rij["pcf"],
            ev_ebitda=rij["ev_ebitda"], ev_fcf=rij["ev_fcf"],
            shareholder_yield=rij["shareholder_yield"], vc2_score=round(rij["vc2_score"], 1),
            momentum_pct=round(mom, 1),
        ))
        time.sleep(cfg["throttle_sec"])

    if not trending:
        print("[ERROR] Geen enkele ticker in de VC2-decile had bruikbare koershistoriek voor momentum.")
        return

    trending.sort(key=lambda s: s.momentum_pct, reverse=True)
    top = trending[: cfg["top_n_global"]]

    print(f"\nTop {len(top)} (hoogste {cfg['momentum_maanden']}-maands momentum binnen goedkoopste VC2-decile):")
    for rank, s in enumerate(top, start=1):
        print(f"  {rank}. {s.ticker} ({s.exchange}) VC2={s.vc2_score:.0f} Mom={s.momentum_pct:+.1f}%")

    for rank, s in enumerate(top, start=1):
        log_selectie(
            ticker=s.ticker,
            datum=today_str(),
            strategie=cfg["strategie"],
            beurs=s.exchange,
            koers=s.price,
            parameters={
                "rank": rank,
                "vc2_score": s.vc2_score,
                "momentum_6m_pct": s.momentum_pct,
                "pe_ratio": None if math.isnan(s.pe) else s.pe,
                "pb_ratio": None if math.isnan(s.pb) else s.pb,
                "ps_ratio": None if math.isnan(s.ps) else s.ps,
                "pcf_ratio": None if math.isnan(s.pcf) else s.pcf,
                "ev_ebitda": None if math.isnan(s.ev_ebitda) else s.ev_ebitda,
                "ev_fcf": None if math.isnan(s.ev_fcf) else s.ev_fcf,
                "shareholder_yield": s.shareholder_yield,
                "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
            },
        )

    berichten = bouw_telegram_berichten(top, universum_grootte, len(waarde_decile), cfg)
    for b in berichten:
        send_telegram_message(b)

    send_email(
        f"Trending Value (O'Shaughnessy) rapport {today_str()}",
        "\n\n" + ("=" * 40 + "\n\n").join(berichten),
    )

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_01oshaughnessy: yfinance biedt "
        "geen betrouwbare historische reeks van fundamentele ratio's per scandatum "
        "in het verleden. Gebruik 'live' om het huidige universum te rangschikken."
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
