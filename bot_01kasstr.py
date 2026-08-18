#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01kasstr.py  —  KASSTROOM ONDERWAARDERING SELECTIE ENGINE v1.1

Screent op onderwaardering via vrije kasstroom (Free Cash Flow), gecombineerd
met kwaliteitsfilters om de bekende valkuilen van een pure FCF-multiple-screen
te ondervangen (value traps, kasstroom-volatiliteit, cyclische vertekening,
verborgen schuldrisico).

Criteria (score 0-6):
  1. FCF yield        — FCF / marktkap >= FCF_YIELD_MIN (goedkoop t.o.v. kasstroom)
  2. FCF-trend         — FCF laatste jaar > FCF oudste jaar in de reeks (groeiend, geen value trap)
  3. FCF-consistentie   — FCF was in ALLE beschikbare jaren positief (geen volatiele/verlieslatende jaren)
  4. Omzetgroei        — revenueGrowth (YoY) > 0 (onderbouwt dat de FCF niet uit krimp komt)
  5. Balans-kwaliteit   — Net Debt / EBITDA <= NET_DEBT_EBITDA_MAX (schuld draagbaar t.o.v. kasstroom)
  6. Aandeelhoudersvriendelijkheid — dividend+buyback payout uit FCF > 0% en <= 100% (houdbaar, niet nul)

Rapportage: enkel de top 5 hoogst scorende aandelen per beurs (Telegram + email).

Supabase: logt naar de bestaande gedeelde `selecties`-tabel (db_logger.py),
onder strategie "bot_01kasstr". De nieuwe parameters van deze bot
(fcf_yield, fcf_years, fcf_growing, fcf_consistent, rev_growth_pct,
net_debt_ebitda, payout_pct) zijn toegevoegd aan db_logger.py's
_KOLOM_WHITELIST zodat ze als eigen kolommen worden weggeschreven i.p.v.
enkel in de JSON parameters-kolom — vereist de bijhorende ALTER TABLE-
migratie op Supabase, zie migratie_kasstr_kolommen.sql.

Gebruik:
  python bot_01kasstr.py live     # live rapport
  python bot_01kasstr.py backtest # niet ondersteund (fundamentele data heeft geen bruikbare
                                   # historische reeks via yfinance) — print uitleg en stopt
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
    """Zelfde dynamische 041-059 opbouw als bot_00kr / weekly_report.py."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

FCF_CFG = {
    "fcf_yield_min":       5.0,   # % — ondergrens FCF/marktkap om als 'goedkoop' te tellen
    "net_debt_ebitda_max": 3.0,   # jaren kasstroom om nettoschuld af te betalen
    "payout_max_pct":      100.0, # dividend+buyback mag niet meer zijn dan 100% van FCF
    "min_years_required":  2,     # minimum aantal jaarlijkse cashflow-kolommen om trend/consistentie te beoordelen
    "min_score":           4,
}

# ============================================================
# HULPFUNCTIES  (identiek patroon aan bot_00kr.py)
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
# FUNDAMENTELE DATA — FCF, TREND, BALANS
# ============================================================

def _row(df, names: List[str]):
    """Zoekt de eerste bestaande rij (op naam) in een yfinance cashflow/balance
    DataFrame. yfinance-versies verschillen in exacte rijnamen, dus we proberen
    meerdere varianten."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None

def _fcf_series(cashflow) -> List[float]:
    """Geeft FCF per beschikbaar jaar, oudste eerst. Gebruikt de kant-en-klare
    'Free Cash Flow'-rij indien aanwezig, anders Operating Cash Flow - Capex."""
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
    return list(reversed(vals))  # yfinance kolommen staan nieuwste-eerst -> omdraaien

def _shareholder_return(cashflow) -> float:
    """Som van (absolute) dividenden + buybacks van het meest recente jaar."""
    if cashflow is None or cashflow.empty:
        return 0.0
    div = _row(cashflow, ["Cash Dividends Paid", "Common Stock Dividend Paid", "Payment Of Dividends"])
    bb  = _row(cashflow, ["Repurchase Of Capital Stock", "Common Stock Repurchase", "Repurchase Of Common Stock"])
    total = 0.0
    if div is not None:
        total += abs(safe_float(div.iloc[0], 0.0))
    if bb is not None:
        total += abs(safe_float(bb.iloc[0], 0.0))
    return total


@dataclass
class FCFSignaal:
    ticker:          str
    price:           float
    score:           int
    market_cap:      float
    fcf_now:         float
    fcf_yield:       float
    fcf_years:       int
    fcf_growing:     bool
    fcf_consistent:  bool
    rev_growth_pct:  float
    net_debt_ebitda: float
    payout_pct:      float
    div_yield:       float
    yield_label:     str
    trend_label:     str
    consist_label:   str
    growth_label:    str
    debt_label:      str
    payout_label:    str

def analyse_ticker(ticker: str) -> Optional[FCFSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        market_cap = safe_float(info.get("marketCap"))
        if math.isnan(price) or price <= 0 or math.isnan(market_cap) or market_cap <= 0:
            return None

        cashflow = tk.cashflow
        fcf_series = _fcf_series(cashflow)
        if len(fcf_series) < FCF_CFG["min_years_required"]:
            return None
        fcf_now = fcf_series[-1]

        # Data-kwaliteitscheck: een FCF yield buiten [-100%, +100%] wijst vrijwel
        # altijd op een valuta- of eenheid-mismatch tussen marktkap en cashflow
        # in de yfinance-data (vaak bij cross-listed aandelen/ADR's), niet op
        # een echte onderwaardering. Ticker volledig overslaan i.p.v. met een
        # vervuild signaal laten meetellen.
        _fcf_yield_raw = (fcf_now / market_cap) * 100 if market_cap > 0 else float("nan")
        if not math.isnan(_fcf_yield_raw) and abs(_fcf_yield_raw) > 100:
            print(f"[WARN] {ticker}: FCF yield implausibel ({_fcf_yield_raw:.0f}%) — data-mismatch, overgeslagen")
            return None

        score = 0

        # 1. FCF yield
        fcf_yield = (fcf_now / market_cap) * 100 if market_cap > 0 else float("nan")
        if not math.isnan(fcf_yield) and fcf_yield >= FCF_CFG["fcf_yield_min"]:
            score += 1
            yield_label = f"✓ {fcf_yield:.1f}% (>= {FCF_CFG['fcf_yield_min']:.0f}%)"
        else:
            yield_label = f"✗ {fcf_yield:.1f}%" if not math.isnan(fcf_yield) else "✗ onbekend"

        # 2. FCF-trend (tegen value traps)
        fcf_growing = fcf_series[-1] > fcf_series[0]
        if fcf_growing:
            score += 1
            trend_label = f"✓ groeiend ({fcf_series[0]:,.0f} -> {fcf_series[-1]:,.0f})"
        else:
            trend_label = f"✗ dalend ({fcf_series[0]:,.0f} -> {fcf_series[-1]:,.0f})"

        # 3. FCF-consistentie (tegen kasstroom-volatiliteit)
        fcf_consistent = all(v > 0 for v in fcf_series)
        if fcf_consistent:
            score += 1
            consist_label = f"✓ positief in {len(fcf_series)}/{len(fcf_series)} jaar"
        else:
            n_pos = sum(1 for v in fcf_series if v > 0)
            consist_label = f"✗ positief in {n_pos}/{len(fcf_series)} jaar"

        # 4. Omzetgroei (onderbouwt dat FCF niet uit krimp komt)
        rev_growth = safe_float(info.get("revenueGrowth"))
        rev_growth_pct = rev_growth * 100 if not math.isnan(rev_growth) else float("nan")
        if not math.isnan(rev_growth_pct) and rev_growth_pct > 0:
            score += 1
            growth_label = f"✓ {rev_growth_pct:+.1f}%"
        else:
            growth_label = f"✗ {rev_growth_pct:+.1f}%" if not math.isnan(rev_growth_pct) else "✗ onbekend"

        # 5. Balans-kwaliteit (tegen verborgen schuldrisico achter lage multiple)
        total_debt = safe_float(info.get("totalDebt"), 0.0)
        total_cash = safe_float(info.get("totalCash"), 0.0)
        ebitda     = safe_float(info.get("ebitda"))
        net_debt   = total_debt - total_cash
        if not math.isnan(ebitda) and ebitda > 0:
            net_debt_ebitda = net_debt / ebitda
            if net_debt_ebitda <= FCF_CFG["net_debt_ebitda_max"]:
                score += 1
                debt_label = f"✓ {net_debt_ebitda:.1f}x EBITDA"
            else:
                debt_label = f"✗ {net_debt_ebitda:.1f}x EBITDA"
        else:
            net_debt_ebitda = float("nan")
            debt_label = "✗ EBITDA onbekend"

        # 6. Aandeelhoudersvriendelijkheid (bevestigt dat FCF ook terugvloeit, houdbaar)
        sh_return  = _shareholder_return(cashflow)
        payout_pct = (sh_return / fcf_now * 100) if fcf_now > 0 else float("nan")
        div_yield  = safe_float(info.get("dividendYield"), 0.0)
        if not math.isnan(payout_pct) and 0 < payout_pct <= FCF_CFG["payout_max_pct"]:
            score += 1
            payout_label = f"✓ {payout_pct:.0f}% van FCF"
        elif not math.isnan(payout_pct) and payout_pct > FCF_CFG["payout_max_pct"]:
            payout_label = f"✗ {payout_pct:.0f}% van FCF (onhoudbaar)"
        else:
            payout_label = "✗ geen dividend/buyback"

        return FCFSignaal(
            ticker=ticker, price=round(price, 2), score=score,
            market_cap=market_cap, fcf_now=fcf_now,
            fcf_yield=round(fcf_yield, 2) if not math.isnan(fcf_yield) else 0.0,
            fcf_years=len(fcf_series), fcf_growing=fcf_growing, fcf_consistent=fcf_consistent,
            rev_growth_pct=round(rev_growth_pct, 2) if not math.isnan(rev_growth_pct) else 0.0,
            net_debt_ebitda=round(net_debt_ebitda, 2) if not math.isnan(net_debt_ebitda) else 0.0,
            payout_pct=round(payout_pct, 1) if not math.isnan(payout_pct) else 0.0,
            div_yield=round(div_yield, 2) if not math.isnan(div_yield) else 0.0,
            yield_label=yield_label, trend_label=trend_label, consist_label=consist_label,
            growth_label=growth_label, debt_label=debt_label, payout_label=payout_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één bericht per exchange
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (6 - score) + f" {score}/6"

def format_bericht(exchange_name: str, signalen: List[FCFSignaal], alle: List[FCFSignaal]) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None

    nu     = today_str()
    top3   = sorted(alle, key=lambda s: (s.score, s.fcf_yield), reverse=True)[:3]
    max_sc = max((s.score for s in signalen), default=0) if signalen else 0
    lbl    = {6: "⭐ PERFECTE SCORE (6/6)", 5: "🟡 STERK (5/6)", 4: "🟠 WATCHLIST (4/6)"}.get(max_sc, "📊")

    def sig_regel(s: FCFSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"FCF yield:{s.fcf_yield:.1f}% | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += (
                f"\n  {s.trend_label} | {s.consist_label}"
                f"\n  Omzet: {s.growth_label} | Schuld: {s.debt_label} | Payout: {s.payout_label}"
            )
        return r

    delen = [
        f"💰 *KASSTROOM ONDERWAARDERING — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(signalen)} kandidaten (score>={FCF_CFG['min_score']})_",
        "─────────────────────────────",
        f"🏆 *TOP 3 HOOGSTE SCORE:*",
        "\n\n".join(sig_regel(s, detail=True) for s in top3),
    ]

    overige = [s for s in signalen if s not in top3]
    if overige:
        delen += ["─────────────────────────────", f"*{lbl} — overige kandidaten:*"]
        for s in overige:
            delen.append(sig_regel(s))

    delen.append(
        f"⚙️ _FCF yield>={FCF_CFG['fcf_yield_min']:.0f}% | groeiende & consistente FCF | "
        f"omzetgroei>0 | NetDebt/EBITDA<={FCF_CFG['net_debt_ebitda_max']:.0f}x | payout<=100% FCF_"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"KASSTROOM ONDERWAARDERING — LIVE  {today_str()}")
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

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")

        alle: List[FCFSignaal] = []
        for ticker in tlist:
            sig = analyse_ticker(ticker)
            if sig is not None:
                alle.append(sig)
                if sig.score >= FCF_CFG["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/6 | FCF yield={sig.fcf_yield:.1f}%")
            time.sleep(0.15)  # lichte throttle tegen Yahoo rate-limits (fundamentals-calls per ticker)

        kandidaten = [s for s in alle if s.score >= FCF_CFG["min_score"]]
        kandidaten.sort(key=lambda s: (s.score, s.fcf_yield), reverse=True)
        signalen = kandidaten[:5]  # enkel top 5 per beurs

        print(f"  → top {len(signalen)} van {len(kandidaten)} kandidaten (score >= {FCF_CFG['min_score']}) uit {len(alle)} geanalyseerd")

        for rank, s in enumerate(signalen, start=1):
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01kasstr",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "rank": rank,
                    "fcf_yield": s.fcf_yield,
                    "fcf_years": s.fcf_years,
                    "fcf_growing": s.fcf_growing,
                    "fcf_consistent": s.fcf_consistent,
                    "rev_growth_pct": s.rev_growth_pct,
                    "net_debt_ebitda": s.net_debt_ebitda,
                    "payout_pct": s.payout_pct,
                    "div_yield": s.div_yield,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, signalen, alle)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")
        else:
            print(f"  → Overgeslagen: {ex_name}")

    if email_delen:
        send_email(
            f"Kasstroom Onderwaardering rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_00fcf: yfinance biedt geen "
        "betrouwbare historische reeks van fundamentele data (marktkap/FCF/schuld) "
        "per scandatum in het verleden, enkel de laatste ~4 jaarrapporten. "
        "Gebruik 'live' om het huidige universum te screenen."
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
