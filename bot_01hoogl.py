#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01hoogl.py  —  GARP ONDERWAARDERING SELECTIE ENGINE v1.0

Screent op "Growth At a Reasonable Price", geïnspireerd op de
selectiecriteria uit de TopAandelen.com-rapporten (Jack Hoogland):
terugverdienperiode t.o.v. boekwaarde, koers-winstverhouding op
verwachte winst, winstgroei en rendement op eigen vermogen (RoE).

Criteria (score 0-4):
  1. RoE                — returnOnEquity >= ROE_MIN (winst wordt goed herbelegd)
  2. Terugverdienperiode  — (koers - boekwaarde/aandeel) / winst huidig jaar <= TERUGVERDIEN_MAX_JAAR
                            (boekwaarde >= koers geeft een negatieve/lage periode en telt ook mee)
  3. Forward P/E          — koers / verwachte winst komend jaar, tussen 0 en FWD_PE_MAX
  4. Verwachte winstgroei — (forwardEps - trailingEps) / trailingEps > 0%

Rapportage: enkel de top 5 hoogst scorende aandelen per beurs (Telegram + email).

Supabase: logt naar de bestaande gedeelde `selecties`-tabel (db_logger.py),
onder strategie "bot_01hoogl". De nieuwe parameters van deze bot
(roe_pct, terugverdienperiode, forward_pe, verwachte_winstgroei_pct) zijn
toegevoegd aan db_logger.py's _KOLOM_WHITELIST zodat ze als eigen kolommen
worden weggeschreven i.p.v. enkel in de JSON parameters-kolom — vereist de
bijhorende ALTER TABLE-migratie op Supabase, zie migratie_hoogl_kolommen.sql.

Gebruik:
  python bot_01hoogl.py live     # live rapport
  python bot_01hoogl.py backtest # niet ondersteund (analyst-EPS-schattingen hebben geen
                                   # bruikbare historische reeks via yfinance) — print uitleg en stopt
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
    """Zelfde dynamische 041-059 opbouw als bot_00kr / bot_01kasstr / weekly_report.py."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

HOOGL_CFG = {
    "roe_min":               15.0,  # % — RoE-ondergrens (Hoogland-rapporten + kasstr-filosofie)
    "terugverdien_max_jaar": 15.0,  # jaren om koers-boekwaarde-gat terug te verdienen met huidige winst
    "fwd_pe_max":            15.0,  # forward P/E-bovengrens
    "min_score":             3,
}

# ============================================================
# HULPFUNCTIES  (identiek patroon aan bot_01kasstr.py)
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
# FUNDAMENTELE DATA — RoE, TERUGVERDIENPERIODE, FORWARD P/E, GROEI
# ============================================================

@dataclass
class HooglSignaal:
    ticker:                   str
    price:                    float
    score:                    int
    roe_pct:                  float
    terugverdienperiode:      float
    forward_pe:               float
    verwachte_winstgroei_pct: float
    div_yield:                float
    roe_label:                str
    terugverdien_label:       str
    fwd_pe_label:             str
    groei_label:              str

def analyse_ticker(ticker: str) -> Optional[HooglSignaal]:
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}

        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if math.isnan(price) or price <= 0:
            return None

        boekwaarde   = safe_float(info.get("bookValue"))
        winst_nu     = safe_float(info.get("trailingEps"))
        winst_verw   = safe_float(info.get("forwardEps"))
        roe          = safe_float(info.get("returnOnEquity"))
        fwd_pe       = safe_float(info.get("forwardPE"))
        div_yield    = safe_float(info.get("dividendYield"), 0.0)

        score = 0

        # 1. RoE (winst wordt goed herbelegd)
        roe_pct = roe * 100 if not math.isnan(roe) else float("nan")
        if not math.isnan(roe_pct) and roe_pct >= HOOGL_CFG["roe_min"]:
            score += 1
            roe_label = f"✓ {roe_pct:.1f}% (>= {HOOGL_CFG['roe_min']:.0f}%)"
        else:
            roe_label = f"✗ {roe_pct:.1f}%" if not math.isnan(roe_pct) else "✗ onbekend"

        # 2. Terugverdienperiode: (koers - boekwaarde/aandeel) / winst huidig jaar
        if not math.isnan(boekwaarde) and not math.isnan(winst_nu) and winst_nu > 0:
            terug_te_verdienen = price - boekwaarde
            terugverdienperiode = terug_te_verdienen / winst_nu
            if terugverdienperiode <= HOOGL_CFG["terugverdien_max_jaar"]:
                score += 1
                terugverdien_label = f"✓ {terugverdienperiode:.1f} jaar"
            else:
                terugverdien_label = f"✗ {terugverdienperiode:.1f} jaar"
        else:
            terugverdienperiode = float("nan")
            terugverdien_label = "✗ onbekend (geen winst/boekwaarde)"

        # 3. Forward P/E
        if not math.isnan(fwd_pe) and 0 < fwd_pe <= HOOGL_CFG["fwd_pe_max"]:
            score += 1
            fwd_pe_label = f"✓ {fwd_pe:.1f}x (<= {HOOGL_CFG['fwd_pe_max']:.0f}x)"
        else:
            fwd_pe_label = f"✗ {fwd_pe:.1f}x" if not math.isnan(fwd_pe) else "✗ onbekend"

        # 4. Verwachte winstgroei (forward EPS vs huidige EPS)
        if not math.isnan(winst_nu) and not math.isnan(winst_verw) and winst_nu > 0:
            verwachte_groei_pct = (winst_verw - winst_nu) / winst_nu * 100
            if verwachte_groei_pct > 0:
                score += 1
                groei_label = f"✓ {verwachte_groei_pct:+.1f}%"
            else:
                groei_label = f"✗ {verwachte_groei_pct:+.1f}%"
        else:
            verwachte_groei_pct = float("nan")
            groei_label = "✗ onbekend"

        return HooglSignaal(
            ticker=ticker, price=round(price, 2), score=score,
            roe_pct=round(roe_pct, 1) if not math.isnan(roe_pct) else 0.0,
            terugverdienperiode=round(terugverdienperiode, 1) if not math.isnan(terugverdienperiode) else 0.0,
            forward_pe=round(fwd_pe, 1) if not math.isnan(fwd_pe) else 0.0,
            verwachte_winstgroei_pct=round(verwachte_groei_pct, 1) if not math.isnan(verwachte_groei_pct) else 0.0,
            div_yield=round(div_yield, 2) if not math.isnan(div_yield) else 0.0,
            roe_label=roe_label, terugverdien_label=terugverdien_label,
            fwd_pe_label=fwd_pe_label, groei_label=groei_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één bericht per exchange
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (4 - score) + f" {score}/4"

def format_bericht(exchange_name: str, signalen: List[HooglSignaal], alle: List[HooglSignaal]) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None

    nu     = today_str()
    top3   = sorted(alle, key=lambda s: (s.score, -s.terugverdienperiode), reverse=True)[:3]
    max_sc = max((s.score for s in signalen), default=0) if signalen else 0
    lbl    = {4: "⭐ PERFECTE SCORE (4/4)", 3: "🟡 STERK (3/4)"}.get(max_sc, "📊")

    def sig_regel(s: HooglSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"RoE:{s.roe_pct:.1f}% | Terugverdien:{s.terugverdienperiode:.1f}j | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += (
                f"\n  {s.roe_label} | {s.terugverdien_label}"
                f"\n  Fwd P/E: {s.fwd_pe_label} | Verw. winstgroei: {s.groei_label}"
            )
        return r

    delen = [
        f"📈 *GARP ONDERWAARDERING — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(signalen)} kandidaten (score>={HOOGL_CFG['min_score']})_",
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
        f"⚙️ _RoE>={HOOGL_CFG['roe_min']:.0f}% | terugverdienperiode<={HOOGL_CFG['terugverdien_max_jaar']:.0f}j | "
        f"forward P/E<={HOOGL_CFG['fwd_pe_max']:.0f}x | verwachte winstgroei>0%_"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"GARP ONDERWAARDERING — LIVE  {today_str()}")
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

        alle: List[HooglSignaal] = []
        for ticker in tlist:
            sig = analyse_ticker(ticker)
            if sig is not None:
                alle.append(sig)
                if sig.score >= HOOGL_CFG["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/4 | RoE={sig.roe_pct:.1f}%")
            time.sleep(0.15)  # lichte throttle tegen Yahoo rate-limits (fundamentals-calls per ticker)

        kandidaten = [s for s in alle if s.score >= HOOGL_CFG["min_score"]]
        kandidaten.sort(key=lambda s: (s.score, -s.terugverdienperiode), reverse=True)
        signalen = kandidaten[:5]  # enkel top 5 per beurs

        print(f"  → top {len(signalen)} van {len(kandidaten)} kandidaten (score >= {HOOGL_CFG['min_score']}) uit {len(alle)} geanalyseerd")

        for rank, s in enumerate(signalen, start=1):
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01hoogl",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "rank": rank,
                    "roe_pct": s.roe_pct,
                    "terugverdienperiode": s.terugverdienperiode,
                    "forward_pe": s.forward_pe,
                    "verwachte_winstgroei_pct": s.verwachte_winstgroei_pct,
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
            f"GARP Onderwaardering rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_01hoogl: yfinance biedt geen "
        "betrouwbare historische reeks van analyst-EPS-schattingen (forwardEps/forwardPE) "
        "per scandatum in het verleden, enkel de actuele consensus. "
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
