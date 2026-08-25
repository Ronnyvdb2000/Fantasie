#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00Fisher.py  —  KENNETH FISHER PRICE-TO-SALES SELECTIE ENGINE v1.0

Screent op Kenneth Fishers Price-to-Sales-methodiek uit Super Stocks (1984):
in tegenstelling tot bot_00graham/bot_00greenblatt/bot_00oshaughnessy vereist
dit script GEEN positieve winst — P/S werkt ook voor verlieslatende of
cyclische bedrijven waar P/E, EBIT-gebaseerde ROC en de meeste VC2-factoren
simpelweg niet te berekenen zijn. Dat is de expliciete niche van deze bot:
kandidaten vinden die de andere drie bots structureel missen.

Criteria (score 0-5):
  1. Price-to-Sales, SECTORAFHANKELIJK (Fishers eigen onderscheid — technologie
     tolereert een hogere P/S door hogere marges/groeipotentieel):
       - sector "Technology"                → P/S <= psr_max_tech (3.0)
       - alle andere sectoren                → P/S <= psr_max_default (1.5)
  2. Adequate omvang — marketCap >= marktkap_min (liquiditeit/investeerbaarheid)
  3. Omzetgroei positief — Total Revenue laatste jaar > vorig jaar (i.p.v.
     winstgroei zoals bij Graham, want winst kan hier negatief zijn)
  4. Debt/Equity <= debt_equity_max — Fishers "Super Company"-vereiste: een
     gezonde balans is juist BELANGRIJKER hier, want er is geen winst die de
     schuld kan opvangen bij tegenslag
  5. Current ratio >= current_ratio_min (1.0, lager dan Grahams 2.0 — deze
     bedrijven zijn vaker jong/cyclisch/in herstel, een striktere lat zou de
     hele doelgroep uitsluiten)

BEWUST GEEN winstcriterium, geen P/E, geen ROC — dat zou de kern van Fishers
methodiek (bruikbaar zonder winst) tenietdoen.

RISICO (expliciet vermeld, geen aanbeveling): een lage P/S bij een
verlieslatend bedrijf is geen garantie op koopjes — het kan een "value trap"
zijn (omzet die zelf ook binnenkort instort). Zonder de winstvalidatie die
Graham/Greenblatt/Piotroski wél hebben, is de spreiding in uitkomsten groter:
meer opwaarts potentieel als het bedrijf herstelt, maar ook meer kans op een
totale nul dan bij de winstgevende, kwaliteitsgefilterde aandelen elders.

BEPERKINGEN:
  - Sectorclassificatie komt van yfinance's `sector`-veld — enkel "Technology"
    krijgt de ruimere drempel, alle andere sectoren (incl. bv. "Communication
    Services", waar sommige tech-achtige bedrijven in vallen) krijgen de
    strengere default-drempel. Dit is een vereenvoudiging t.o.v. Fishers
    fijnmazigere sectorindeling.
  - Omzetgroei is 1 jaar (laatste vs vorige), niet Fishers voorkeur voor een
    meerjarige trend — yfinance's income_stmt geeft doorgaans ~4 jaar, maar
    voor de kernscore wordt hier enkel het laatste jaarpaar gebruikt.
  - Debt/Equity komt uit yfinance's `debtToEquity`-veld (indien beschikbaar);
    ontbreekt het, dan telt dit criterium niet mee (geen punt, sluit niet uit).
  - Deduplicatie (v1.0, vanaf het begin toegepast — geleerd bij bot_00greenblatt/
    bot_00oshaughnessy): de 041-059 bestanden overlappen soms, een lopende
    "reeds verwerkt"-set voorkomt dat eenzelfde ticker in twee beurzen tegelijk
    verschijnt.

TWEE MODI: enkel 'live' — geen backtest (zelfde reden als de andere bots:
yfinance biedt geen betrouwbare historische fundamentals per scandatum).

Rapportage: Telegram + email, één bericht per beurs, top 5 per beurs, zelfde
patroon als bot_00graham.py. Draait WEKELIJKS (niet dagelijks) — Fisher-
kandidaten zijn geen dagelijks signaal, en dit houdt de GitHub Actions-
looptijd binnen de perken naast de andere wekelijkse bots.

Supabase: logt naar de bestaande gedeelde `selecties`-tabel onder strategie
"bot_00Fisher". Nieuwe kolommen: ps_ratio (al gewhitelist via bot_00oshaughnessy),
sector, revenue_growth_pct, debt_to_equity — zie migratie_fisher_kolommen.sql.

Gebruik:
  python bot_00Fisher.py live
  python bot_00Fisher.py backtest  # niet ondersteund
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
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

FISHER_CFG = {
    "psr_max_tech":       3.0,
    "psr_max_default":    1.5,
    "marktkap_min":       200_000_000.0,
    "debt_equity_max":    150.0,   # yfinance's debtToEquity is in procentpunten (150 = 1.5x)
    "current_ratio_min":  1.0,
    "min_score":          4,
    "top_n":              5,
    "strategie":          "bot_00Fisher",
    "throttle_sec":       0.15,
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
# FISHER P/S-ANALYSE PER TICKER
# ============================================================

@dataclass
class FisherSignaal:
    ticker:              str
    price:               float
    score:               int
    sector:              str
    ps_ratio:            float
    market_cap:          float
    revenue_growth_pct:  float
    debt_to_equity:      float
    current_ratio:       float
    ps_label:            str
    marktkap_label:      str
    groei_label:         str
    schuld_label:        str
    current_ratio_label: str

def analyse_ticker(ticker: str, cfg: dict) -> Optional[FisherSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        sector     = info.get("sector") or "Onbekend"
        market_cap = safe_float(info.get("marketCap"))
        ps_ratio   = safe_float(info.get("priceToSalesTrailing12Months"))

        score = 0

        # 1. P/S, sectorafhankelijk
        psr_drempel = cfg["psr_max_tech"] if sector == "Technology" else cfg["psr_max_default"]
        if not math.isnan(ps_ratio) and 0 < ps_ratio <= psr_drempel:
            score += 1
            ps_label = f"✓ {ps_ratio:.2f} (<= {psr_drempel:.1f}, sector: {sector})"
        else:
            ps_label = f"✗ {ps_ratio:.2f} (grens {psr_drempel:.1f}, sector: {sector})" if not math.isnan(ps_ratio) else "✗ onbekend"

        # 2. Adequate omvang
        if not math.isnan(market_cap) and market_cap >= cfg["marktkap_min"]:
            score += 1
            marktkap_label = f"✓ {market_cap/1e6:.0f}M (>= {cfg['marktkap_min']/1e6:.0f}M)"
        else:
            marktkap_label = f"✗ {market_cap/1e6:.0f}M" if not math.isnan(market_cap) else "✗ onbekend"

        # 3. Omzetgroei (i.p.v. winstgroei — winst kan hier negatief zijn)
        try:
            inc = tk.financials
        except Exception:
            inc = None
        rev_row = _row(inc, ["Total Revenue"]) if inc is not None else None
        if rev_row is not None and len(rev_row) >= 2:
            rev_nu, rev_vorig = safe_float(rev_row.iloc[0]), safe_float(rev_row.iloc[1])
            if not math.isnan(rev_nu) and not math.isnan(rev_vorig) and rev_vorig > 0:
                revenue_growth_pct = (rev_nu - rev_vorig) / rev_vorig * 100
                if revenue_growth_pct > 0:
                    score += 1
                    groei_label = f"✓ omzet {revenue_growth_pct:+.1f}%"
                else:
                    groei_label = f"✗ omzet {revenue_growth_pct:+.1f}%"
            else:
                revenue_growth_pct = float("nan")
                groei_label = "✗ onbekend"
        else:
            revenue_growth_pct = float("nan")
            groei_label = "✗ onbekend (< 2 jaar resultatenrekening)"

        # 4. Debt/Equity — belangrijker HIER dan bij winstgevende bedrijven
        debt_to_equity = safe_float(info.get("debtToEquity"))
        if not math.isnan(debt_to_equity) and debt_to_equity <= cfg["debt_equity_max"]:
            score += 1
            schuld_label = f"✓ D/E {debt_to_equity:.0f} (<= {cfg['debt_equity_max']:.0f})"
        else:
            schuld_label = f"✗ D/E {debt_to_equity:.0f}" if not math.isnan(debt_to_equity) else "✗ onbekend"

        # 5. Current ratio (lagere lat dan Graham — jongere/cyclische doelgroep)
        current_ratio = safe_float(info.get("currentRatio"))
        if not math.isnan(current_ratio) and current_ratio >= cfg["current_ratio_min"]:
            score += 1
            current_ratio_label = f"✓ {current_ratio:.2f} (>= {cfg['current_ratio_min']:.1f})"
        else:
            current_ratio_label = f"✗ {current_ratio:.2f}" if not math.isnan(current_ratio) else "✗ onbekend"

        return FisherSignaal(
            ticker=ticker, price=round(price, 2), score=score, sector=sector,
            ps_ratio=round(ps_ratio, 2) if not math.isnan(ps_ratio) else 0.0,
            market_cap=round(market_cap, 0) if not math.isnan(market_cap) else 0.0,
            revenue_growth_pct=round(revenue_growth_pct, 1) if not math.isnan(revenue_growth_pct) else 0.0,
            debt_to_equity=round(debt_to_equity, 1) if not math.isnan(debt_to_equity) else 0.0,
            current_ratio=round(current_ratio, 2) if not math.isnan(current_ratio) else 0.0,
            ps_label=ps_label, marktkap_label=marktkap_label,
            groei_label=groei_label, schuld_label=schuld_label,
            current_ratio_label=current_ratio_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (5 - score) + f" {score}/5"

def format_bericht(exchange_name: str, signalen: List[FisherSignaal], alle: List[FisherSignaal], cfg: dict) -> Optional[str]:
    if not alle:
        return None

    nu = today_str()
    top_n = cfg["top_n"]
    top_tonen = sorted(alle, key=lambda s: (s.score, -s.ps_ratio if s.ps_ratio > 0 else -999), reverse=True)[:top_n]

    def sig_regel(s: FisherSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"P/S:{s.ps_ratio:.2f} | {s.sector} | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += (
                f"\n  {s.ps_label} | {s.marktkap_label}"
                f"\n  Omzetgroei: {s.groei_label} | Schuld: {s.schuld_label}"
                f"\n  Liquiditeit: {s.current_ratio_label}"
            )
        return r

    delen = [
        f"📊 *FISHER PRICE-TO-SALES — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(signalen)} kandidaten (score>={cfg['min_score']})_",
        "─────────────────────────────",
        f"🏆 *TOP {top_n} HOOGSTE SCORE:*",
        "\n\n".join(sig_regel(s, detail=True) for s in top_tonen),
    ]

    overige = [s for s in signalen if s not in top_tonen]
    if overige:
        delen += ["─────────────────────────────", "*Overige kandidaten:*"]
        for s in overige:
            delen.append(sig_regel(s))

    delen.append(
        f"⚙️ _P/S<={cfg['psr_max_tech']:.1f} (tech) of <={cfg['psr_max_default']:.1f} (overig) | "
        f"marktkap>={cfg['marktkap_min']/1e6:.0f}M | omzetgroei>0% | D/E<={cfg['debt_equity_max']:.0f} | "
        f"current ratio>={cfg['current_ratio_min']:.1f}. GEEN winstvereiste — werkt ook op "
        f"verlieslatende bedrijven, zie risico-noot in de module-docstring._"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    cfg = FISHER_CFG
    print(f"{'='*60}")
    print(f"FISHER PRICE-TO-SALES — LIVE  {today_str()}")
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
    reeds_verwerkt: set = set()  # dedup vanaf het begin ingebouwd (geleerd bij greenblatt/oshaughnessy)
    totaal_gedupliceerd = 0

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")

        alle: List[FisherSignaal] = []
        for ticker in tlist:
            if ticker in reeds_verwerkt:
                totaal_gedupliceerd += 1
                continue
            reeds_verwerkt.add(ticker)
            sig = analyse_ticker(ticker, cfg)
            if sig is not None:
                alle.append(sig)
                if sig.score >= cfg["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/5 | P/S={sig.ps_ratio:.2f} | {sig.sector}")
            time.sleep(cfg["throttle_sec"])

        kandidaten = [s for s in alle if s.score >= cfg["min_score"]]
        kandidaten.sort(key=lambda s: (s.score, -s.ps_ratio if s.ps_ratio > 0 else -999), reverse=True)
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
                    "sector": s.sector,
                    "ps_ratio": s.ps_ratio,
                    "market_cap": s.market_cap,
                    "revenue_growth_pct": s.revenue_growth_pct,
                    "debt_to_equity": s.debt_to_equity,
                    "current_ratio": s.current_ratio,
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
            f"Fisher Price-to-Sales rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    if totaal_gedupliceerd:
        print(f"\nDeduplicatie: {totaal_gedupliceerd} dubbele ticker-occurrence(s) overgeslagen "
              f"(overlap tussen beursbestanden).")

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_00Fisher: yfinance biedt "
        "geen betrouwbare historische reeks van fundamentele data per scandatum "
        "in het verleden. Gebruik 'live' om het huidige universum te screenen."
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
