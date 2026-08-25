#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00greenblatt.py  —  JOEL GREENBLATT "MAGIC FORMULA" RANKING ENGINE v1.0

Implementeert Greenblatts Magic Formula uit The Little Book That Beats the
Market: GEEN drempel-score zoals bot_00graham/bot_01kasstr, maar een
RELATIEVE RANKING van het volledige gescande universum op twee metrics:

  - Return on Capital (ROC)   = EBIT / (Net Working Capital + Net Fixed Assets)
                                 hoe efficiënt zet het bedrijf kapitaal om in winst
  - Earnings Yield (EY)       = EBIT / Enterprise Value
                                 hoeveel bedrijfswinst je koopt per euro ondernemingswaarde

Elke ticker krijgt een ROC-rang en een EY-rang (1 = beste) binnen het VOLLEDIGE
gescande universum (alle beurzen samen — ratio's zijn dimensieloos, dus
vergelijkbaar over beurzen/valuta's heen, in tegenstelling tot bv. marketCap).
combined_rank = roc_rank + ey_rank; laagste som = beste Magic Formula-kandidaat.

UITSLUITINGEN (zoals Greenblatt zelf voorschrijft):
  - Financiële sector (banken, verzekeraars) en nutsbedrijven: ROC/EBIT hebben
    voor deze sectoren geen vergelijkbare betekenis (regelgeving, leverage-
    structuur), dus expliciet uitgesloten via yfinance's `sector`-veld.
  - marketCap < marktkap_min: te kleine/illiquide namen (Greenblatt zelf
    hanteerde oorspronkelijk $50-100M als praktische ondergrens).
  - EBIT <= 0 of Invested Capital (NWC + Net Fixed Assets) <= 0: ROC/EY zijn
    dan niet zinvol interpreteerbaar.
  - |ROC| of |EY| > 300% (ratio_plafond): een Return on Capital of Earnings
    Yield van honderden procenten is economisch zo goed als altijd een
    databug (bv. een foutieve/te kleine noemer door een eenheden- of
    valuta-mismatch in yfinance bij bepaalde cross-listed tickers), niet
    een legitiem signaal — zelfde aanpak als bot_01kasstr's |FCF yield|>100%-
    uitsluiting. Bevestigd nodig na de eerste live run (2026-08-24): PZC.L
    kwam op #1 met een ROC van 912%, KSPI met een EY van 6492%.

AFWIJKING VAN HET GEBRUIKELIJKE BOT-PATROON (bewust): de andere bots sturen
één bericht PER BEURS met een top-5 die enkel binnen die beurs concurreert.
Dat past niet bij Greenblatts methodiek — die vraagt uitdrukkelijk om de
beste ~20-30 kansen uit het VOLLEDIGE investeerbare universum, ongeacht
beurs. Dit script rangschikt dus globaal en stuurt in de plaats daarvan:
  - één (eventueel in stukken opgesplitst) Telegram-bericht met de globale
    top N (default 30, Greenblatts eigen suggestie voor portefeuillegrootte)
  - één samenvattende e-mail met diezelfde lijst + methodologie-notities

BEPERKINGEN:
  - Deduplicatie (v1.1): de 041-059 tickerbestanden overlappen soms (bv.
    'ALL' stond zowel in 048 Nasdaq/NYSE als 057 NYSE) — zonder correctie
    zou zo'n ticker dubbel meetellen in de globale ranking. dedupliceer_op_
    ticker() houdt enkel de eerst gescande occurrence over, vóór de ranking.
  - EBIT/balansposten komen uit het laatste beschikbare jaarrapport via
    yfinance (tk.financials / tk.balance_sheet), niet TTM-cijfers.
  - Enterprise Value: gebruikt yfinance's `enterpriseValue` uit `info` waar
    beschikbaar; valt anders terug op marketCap + totalDebt - totalCash
    (ruwe benadering, geen correctie voor minderheidsbelangen/preferente
    aandelen zoals Greenblatts striktere definitie).
  - Net Working Capital wordt vereenvoudigd als Current Assets - Current
    Liabilities (Greenblatts eigen definitie trekt er ook rentedragende
    kortlopende schulden/overtollige cash uit, wat via yfinance niet
    betrouwbaar te scheiden is).
  - Greenblatt herbalanceert normaal 1x per jaar (of gespreid per kwartaal
    om fiscale redenen) — dit script herberekent de ranking bij elke run;
    de workflow draait daarom wekelijks, niet dagelijks zoals de meeste
    andere bots.

Rapportage: Telegram (globale top N, in blokken van 15 om Telegrams
berichtlimiet te respecteren) + één samenvattende e-mail. Geen CSV.

Supabase: logt de top N naar de bestaande gedeelde `selecties`-tabel onder
strategie "bot_00greenblatt". Nieuwe parameters (roc, earnings_yield,
roc_rank, ey_rank, combined_rank, ebit, invested_capital, enterprise_value,
sector) moeten aan db_logger.py's _KOLOM_WHITELIST toegevoegd worden —
vereist de bijhorende migratie, zie migratie_greenblatt_kolommen.sql.

Gebruik:
  python bot_00greenblatt.py live      # wekelijkse globale ranking
  python bot_00greenblatt.py backtest  # niet ondersteund (zelfde reden als
                                          # bot_00graham/bot_01kasstr/bot_01hoogl)
"""

import os
import sys
import math
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass, field
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
    """Zelfde dynamische 041-059 opbouw als de andere bots."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

UITGESLOTEN_SECTOREN = {
    "Financial Services", "Financials", "Insurance",
    "Utilities", "Utilities—Regulated Electric", "Utilities—Regulated Gas",
}

GREENBLATT_CFG = {
    "marktkap_min":   50_000_000.0,   # Greenblatts eigen praktische ondergrens ($50-100M)
    "top_n_global":   30,             # Greenblatts suggestie voor portefeuillegrootte
    "ratio_plafond":  3.0,            # |ROC| of |EY| > 300% wordt uitgesloten (databug-signaal)
    "telegram_chunk": 15,             # kandidaten per Telegram-bericht (berichtlimiet)
    "strategie":      "bot_00greenblatt",
    "throttle_sec":   0.15,
}

# ============================================================
# HULPFUNCTIES  (identiek patroon aan bot_00graham.py / bot_01kasstr.py)
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
    """Zoekt de eerste bestaande rij (op naam) in een yfinance-DataFrame."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


# ============================================================
# MAGIC FORMULA — EBIT, ROC, EARNINGS YIELD PER TICKER
# ============================================================

@dataclass
class GreenblattSignaal:
    ticker:            str
    exchange:          str
    price:             float
    sector:            str
    ebit:              float
    invested_capital:  float
    roc:               float
    enterprise_value:  float
    earnings_yield:    float
    roc_rank:          int = 0
    ey_rank:           int = 0
    combined_rank:     int = 0

def analyse_ticker(ticker: str, exchange: str, cfg: dict) -> Optional[GreenblattSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        sector = info.get("sector") or "Onbekend"
        if sector in UITGESLOTEN_SECTOREN:
            return None

        market_cap = safe_float(info.get("marketCap"))
        if math.isnan(market_cap) or market_cap < cfg["marktkap_min"]:
            return None

        try:
            inc = tk.financials
            bs  = tk.balance_sheet
        except Exception:
            return None
        if inc is None or inc.empty or bs is None or bs.empty:
            return None

        ebit_row = _row(inc, ["EBIT", "Operating Income"])
        if ebit_row is None or len(ebit_row) == 0:
            return None
        ebit = safe_float(ebit_row.iloc[0])
        if math.isnan(ebit) or ebit <= 0:
            return None  # Greenblatt vereist positieve winst

        ca_row  = _row(bs, ["Current Assets", "Total Current Assets"])
        cl_row  = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
        ppe_row = _row(bs, ["Net PPE", "Property Plant And Equipment Net", "Net Tangible Assets"])
        if ca_row is None or cl_row is None or ppe_row is None:
            return None
        ca  = safe_float(ca_row.iloc[0])
        cl  = safe_float(cl_row.iloc[0])
        ppe = safe_float(ppe_row.iloc[0])
        if any(math.isnan(v) for v in (ca, cl, ppe)):
            return None

        nwc = ca - cl
        invested_capital = nwc + ppe
        if invested_capital <= 0:
            return None  # ROC niet zinvol interpreteerbaar

        roc = ebit / invested_capital
        if abs(roc) > cfg["ratio_plafond"]:
            return None  # implausibele ROC (>300%) wijst op een databug (bv. verkeerde
                         # eenheid/valuta tussen EBIT en balansposten), zelfde aanpak als
                         # bot_01kasstr's |FCF yield|>100%-uitsluiting

        ev = safe_float(info.get("enterpriseValue"))
        if math.isnan(ev) or ev <= 0:
            total_debt = safe_float(info.get("totalDebt"), 0.0)
            total_cash = safe_float(info.get("totalCash"), 0.0)
            ev = market_cap + total_debt - total_cash
        if math.isnan(ev) or ev <= 0:
            return None

        earnings_yield = ebit / ev
        if abs(earnings_yield) > cfg["ratio_plafond"]:
            return None  # zelfde reden — implausibele EY (>300%) wijst op een databug,
                         # niet op een legitiem signaal

        return GreenblattSignaal(
            ticker=ticker, exchange=exchange, price=round(price, 2), sector=sector,
            ebit=round(ebit, 0), invested_capital=round(invested_capital, 0),
            roc=round(roc, 4), enterprise_value=round(ev, 0),
            earnings_yield=round(earnings_yield, 4),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


def dedupliceer_op_ticker(alle: List[GreenblattSignaal]) -> List[GreenblattSignaal]:
    """Sommige tickerbestanden (041-059) overlappen deels (bv. dezelfde aandelen
    genoteerd in zowel 048 Nasdaq/NYSE als 057 NYSE) — zonder deduplicatie zou
    zo'n ticker twee keer meetellen in de globale ranking en dus dubbel in de
    top N kunnen verschijnen, wat een legitieme andere kandidaat verdringt.
    Behoudt de EERSTE occurrence (op volgorde van de beursscan) per ticker."""
    gezien = set()
    resultaat = []
    for s in alle:
        if s.ticker in gezien:
            continue
        gezien.add(s.ticker)
        resultaat.append(s)
    return resultaat


def rangschik_globaal(alle: List[GreenblattSignaal]) -> List[GreenblattSignaal]:
    """Wijst roc_rank, ey_rank en combined_rank toe over het VOLLEDIGE universum
    (alle beurzen samen — zie module-docstring). 1 = beste."""
    op_roc = sorted(alle, key=lambda s: s.roc, reverse=True)
    for i, s in enumerate(op_roc, start=1):
        s.roc_rank = i

    op_ey = sorted(alle, key=lambda s: s.earnings_yield, reverse=True)
    for i, s in enumerate(op_ey, start=1):
        s.ey_rank = i

    for s in alle:
        s.combined_rank = s.roc_rank + s.ey_rank

    return sorted(alle, key=lambda s: s.combined_rank)


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één globale ranking, in blokken opgesplitst
# ============================================================

def _sig_regel(s: GreenblattSignaal, rank: int) -> str:
    return (
        f"{rank}. `{s.ticker}` ({s.exchange}) | ROC:{s.roc*100:.1f}% (#{s.roc_rank}) | "
        f"EY:{s.earnings_yield*100:.1f}% (#{s.ey_rank}) | som:{s.combined_rank} | "
        f"EUR{s.price:.2f} | {_yahoo_link(s.ticker)}"
    )

def bouw_telegram_berichten(top: List[GreenblattSignaal], universum_grootte: int, cfg: dict) -> List[str]:
    nu = today_str()
    chunk = cfg["telegram_chunk"]
    blokken = [top[i:i + chunk] for i in range(0, len(top), chunk)]
    berichten = []
    for idx, blok in enumerate(blokken, start=1):
        regels = [
            f"🧙 *MAGIC FORMULA (Greenblatt) — GLOBALE TOP {len(top)}*  ({idx}/{len(blokken)})",
            f"_{nu} | {universum_grootte} tickers gescreend over alle beurzen_",
            "─────────────────────────────",
        ]
        start_rank = (idx - 1) * chunk + 1
        for offset, s in enumerate(blok):
            regels.append(_sig_regel(s, start_rank + offset))
        if idx == len(blokken):
            regels.append("─────────────────────────────")
            regels.append(
                "⚙️ _Rang = optelsom van ROC-rang + Earnings Yield-rang binnen het volledige "
                "universum (laagste som = beste). Uitgesloten: financiële sector, nutsbedrijven, "
                f"marketCap < {cfg['marktkap_min']/1e6:.0f}M, EBIT<=0 of Invested Capital<=0, "
                f"|ROC| of |EY| > {cfg['ratio_plafond']*100:.0f}% (databug-filter)._"
            )
        berichten.append("\n".join(regels))
    return berichten


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    cfg = GREENBLATT_CFG
    print(f"{'='*60}")
    print(f"GREENBLATT MAGIC FORMULA — LIVE  {today_str()}")
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

    alle: List[GreenblattSignaal] = []
    universum_grootte = 0

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        gevonden_in_beurs = 0
        for ticker in tlist:
            universum_grootte += 1
            sig = analyse_ticker(ticker, ex_name, cfg)
            if sig is not None:
                alle.append(sig)
                gevonden_in_beurs += 1
            time.sleep(cfg["throttle_sec"])
        print(f"  → {gevonden_in_beurs}/{len(tlist)} tickers voldoen aan de basisfilters (sector/marktkap/EBIT/IC)")

    if not alle:
        print("[ERROR] Geen enkele ticker voldoet aan de Magic Formula-basisfilters.")
        return

    aantal_voor_dedup = len(alle)
    alle = dedupliceer_op_ticker(alle)
    if len(alle) < aantal_voor_dedup:
        print(f"Deduplicatie: {aantal_voor_dedup - len(alle)} dubbele ticker(s) verwijderd "
              f"(overlap tussen beursbestanden) — {len(alle)} unieke tickers over.")

    gerangschikt = rangschik_globaal(alle)
    top = gerangschikt[: cfg["top_n_global"]]

    print(f"\nGlobale ranking klaar: {len(alle)} gekwalificeerde tickers over {len(exchange_tickers)} beurzen")
    print(f"Top {len(top)} (laagste combined_rank):")
    for rank, s in enumerate(top, start=1):
        print(f"  {rank}. {s.ticker} ({s.exchange}) ROC={s.roc*100:.1f}% EY={s.earnings_yield*100:.1f}% som={s.combined_rank}")

    for rank, s in enumerate(top, start=1):
        log_selectie(
            ticker=s.ticker,
            datum=today_str(),
            strategie=cfg["strategie"],
            beurs=s.exchange,
            koers=s.price,
            parameters={
                "rank": rank,
                "roc": s.roc,
                "earnings_yield": s.earnings_yield,
                "roc_rank": s.roc_rank,
                "ey_rank": s.ey_rank,
                "combined_rank": s.combined_rank,
                "ebit": s.ebit,
                "invested_capital": s.invested_capital,
                "enterprise_value": s.enterprise_value,
                "sector": s.sector,
                "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
            },
        )

    berichten = bouw_telegram_berichten(top, universum_grootte, cfg)
    for b in berichten:
        send_telegram_message(b)

    send_email(
        f"Magic Formula (Greenblatt) rapport {today_str()}",
        "\n\n" + ("=" * 40 + "\n\n").join(berichten),
    )

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_00greenblatt: yfinance biedt "
        "geen betrouwbare historische reeks van EBIT/balansposten per scandatum "
        "in het verleden, enkel het laatste jaarrapport en de actuele enterprise "
        "value. Gebruik 'live' om het huidige universum te rangschikken."
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
