#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01hoogl.py  —  GARP ONDERWAARDERING SELECTIE ENGINE v2.0

Screent op "Growth At a Reasonable Price", geïnspireerd op de
selectiecriteria uit de TopAandelen.com-rapporten (Jack Hoogland):
terugverdienperiode t.o.v. boekwaarde, koers-winstverhouding op
verwachte winst, winstgroei en rendement op eigen vermogen (RoE).

Criteria (score 0-6):
  1. RoE                — returnOnEquity >= roe_min (winst wordt goed herbelegd)
  2. Terugverdienperiode  — (koers - boekwaarde/aandeel) / winst huidig jaar <= terugverdien_max_jaar
                            (boekwaarde >= koers geeft een negatieve/lage periode en telt ook mee)
  3. Forward P/E          — koers / verwachte winst komend jaar, tussen 0 en fwd_pe_max
  4. Verwachte winstgroei — (forwardEps - trailingEps) / trailingEps > 0%
  5. Marktkap (small/midcap) — marketCap <= marktkap_max (BeursBrink-stijl: focus op kleinere,
                                onderbelichte bedrijven i.p.v. large caps)
  6. Lage analist-coverage    — numberOfAnalystOpinions <= analisten_max (hoe minder analisten volgen
                                het aandeel, hoe groter de kans op een "vergeten pareltje")

TWEE MODI:
  live      — dagelijks (ma-vr), scant de voorgefilterde tickers_0NNx.txt
              (Nitro-kwaliteitslijsten), top 5 per beurs, score>=3.
              Universum: ~1.200 tickers, run duurt ~15-20 minuten.
  full      — wekelijks (zaterdag), scant de RUWE tickers_0NNa.txt
              (alle tickers per beurs, ~15.300 in totaal), top 3 per
              beurs, strenger (score>=4 = perfecte score). Draait op
              een moment dat geen enkele beurs wereldwijd open is, om
              GitHub Actions-wachtrijen op populaire cron-tijden te
              vermijden. Logt onder een aparte strategienaam
              ("bot_01hoogl_full") zodat dit niet door de dagelijkse
              resultaten heen loopt in Supabase.

Rapportage: Telegram + email, één bericht per beurs, lege beurzen worden
overgeslagen.

Supabase: logt naar de bestaande gedeelde `selecties`-tabel (db_logger.py).
De parameters van deze bot (roe_pct, terugverdienperiode, forward_pe,
verwachte_winstgroei_pct) zijn toegevoegd aan db_logger.py's
_KOLOM_WHITELIST — vereist de bijhorende ALTER TABLE-migratie op
Supabase, zie migratie_hoogl_kolommen.sql.

Gebruik:
  python bot_01hoogl.py live      # dagelijks rapport (x-lijsten)
  python bot_01hoogl.py full      # wekelijks full-scan rapport (a-lijsten, strenger)
  python bot_01hoogl.py backtest  # niet ondersteund (analyst-EPS-schattingen hebben geen
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

# Beursnamen per nummer (suffix-onafhankelijk: x of a wordt los toegevoegd)
BEURS_NAMEN = {
    "041": "041 Benelux Ierland",
    "042": "042 Parijs",
    "043": "043 Frankfurt",
    "044": "044 Spanje/Portugal",
    "045": "045 Londen",
    "046": "046 Milaan",
    "047": "047 Toronto",
    "048": "048 Nasdaq/NYSE",
    "049": "049 Stockholm",
    "050": "050 Zurich",
    "051": "051 Warschau",
    "052": "052 Oslo",
    "053": "053 Kopenhagen",
    "054": "054 Helsinki",
    "055": "055 CSE",
    "056": "056 NYSE int",
    "057": "057 NYSE",
    "058": "058 TSXV",
    "059": "059 Oostenrijk Slovenie Slovakije",
}

def bouw_bestandslijst(suffix: str) -> List[str]:
    """Dynamische 041-059 opbouw. suffix='x' -> kwaliteitslijst, suffix='a' -> alle tickers."""
    return [f"tickers_{n:03d}{suffix}.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    getal = f_name.replace("tickers_", "")[:3]
    return BEURS_NAMEN.get(getal, f_name.replace(".txt", ""))

# Twee configuraties: dagelijks (x-lijsten, ruimer) vs wekelijks (a-lijsten, strenger)
MODUS_CFG = {
    "live": {
        "bestand_suffix":       "x",
        "roe_min":              15.0,
        "terugverdien_max_jaar": 15.0,
        "fwd_pe_max":           15.0,
        "marktkap_max":         5_000_000_000.0,  # small/midcap-plafond (valuta van de ticker zelf)
        "analisten_max":        5,
        "min_score":            4,   # was 3/4 (75%) — op 4/6 (~67%) vergelijkbaar strikt
        "top_n":                5,
        "strategie":            "bot_01hoogl",
        "label":                "GARP ONDERWAARDERING",
        "throttle_sec":         0.15,
    },
    "full": {
        "bestand_suffix":       "a",
        "roe_min":              18.0,   # strenger: hogere kwaliteitsdrempel op ongefilterd universum
        "terugverdien_max_jaar": 10.0,  # strenger: kortere terugverdienperiode
        "fwd_pe_max":           12.0,   # strenger: lagere forward P/E-cap
        "marktkap_max":         5_000_000_000.0,
        "analisten_max":        5,
        "min_score":            6,      # strenger: enkel de perfecte score
        "top_n":                3,
        "strategie":            "bot_01hoogl_full",
        "label":                "GARP ONDERWAARDERING — WEEKLY FULL SCAN",
        "throttle_sec":         0.12,
    },
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
    market_cap:               float
    analisten_count:          float
    roe_label:                str
    terugverdien_label:       str
    fwd_pe_label:             str
    groei_label:              str
    marktkap_label:           str
    analisten_label:          str

def analyse_ticker(ticker: str, cfg: dict) -> Optional[HooglSignaal]:
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
        market_cap   = safe_float(info.get("marketCap"))

        score = 0

        # 1. RoE (winst wordt goed herbelegd)
        roe_pct = roe * 100 if not math.isnan(roe) else float("nan")
        if not math.isnan(roe_pct) and roe_pct >= cfg["roe_min"]:
            score += 1
            roe_label = f"✓ {roe_pct:.1f}% (>= {cfg['roe_min']:.0f}%)"
        else:
            roe_label = f"✗ {roe_pct:.1f}%" if not math.isnan(roe_pct) else "✗ onbekend"

        # 2. Terugverdienperiode: (koers - boekwaarde/aandeel) / winst huidig jaar
        if not math.isnan(boekwaarde) and not math.isnan(winst_nu) and winst_nu > 0:
            terug_te_verdienen = price - boekwaarde
            terugverdienperiode = terug_te_verdienen / winst_nu
            if terugverdienperiode <= cfg["terugverdien_max_jaar"]:
                score += 1
                terugverdien_label = f"✓ {terugverdienperiode:.1f} jaar"
            else:
                terugverdien_label = f"✗ {terugverdienperiode:.1f} jaar"
        else:
            terugverdienperiode = float("nan")
            terugverdien_label = "✗ onbekend (geen winst/boekwaarde)"

        # 3. Forward P/E
        if not math.isnan(fwd_pe) and 0 < fwd_pe <= cfg["fwd_pe_max"]:
            score += 1
            fwd_pe_label = f"✓ {fwd_pe:.1f}x (<= {cfg['fwd_pe_max']:.0f}x)"
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

        # 5. Marktkap (small/midcap) — BeursBrink-stijl: kleinere, onderbelichte bedrijven
        if not math.isnan(market_cap) and market_cap > 0 and market_cap <= cfg["marktkap_max"]:
            score += 1
            marktkap_label = f"✓ {market_cap/1e9:.2f}B (<= {cfg['marktkap_max']/1e9:.0f}B)"
        else:
            marktkap_label = f"✗ {market_cap/1e9:.2f}B" if not math.isnan(market_cap) else "✗ onbekend"

        # 6. Lage analist-coverage — hoe minder gevolgd, hoe groter de kans op een "vergeten pareltje"
        analisten_raw = safe_float(info.get("numberOfAnalystOpinions"))
        if not math.isnan(analisten_raw):
            analisten_count = analisten_raw
            if analisten_count <= cfg["analisten_max"]:
                score += 1
                analisten_label = f"✓ {analisten_count:.0f} analisten (<= {cfg['analisten_max']})"
            else:
                analisten_label = f"✗ {analisten_count:.0f} analisten"
        else:
            # onbekend aantal analisten telt niet mee als punt, maar sluit de ticker niet uit
            analisten_count = float("nan")
            analisten_label = "✗ coverage onbekend"

        return HooglSignaal(
            ticker=ticker, price=round(price, 2), score=score,
            roe_pct=round(roe_pct, 1) if not math.isnan(roe_pct) else 0.0,
            terugverdienperiode=round(terugverdienperiode, 1) if not math.isnan(terugverdienperiode) else 0.0,
            forward_pe=round(fwd_pe, 1) if not math.isnan(fwd_pe) else 0.0,
            verwachte_winstgroei_pct=round(verwachte_groei_pct, 1) if not math.isnan(verwachte_groei_pct) else 0.0,
            div_yield=round(div_yield, 2) if not math.isnan(div_yield) else 0.0,
            market_cap=round(market_cap, 0) if not math.isnan(market_cap) else 0.0,
            analisten_count=round(analisten_count, 0) if not math.isnan(analisten_count) else -1.0,
            roe_label=roe_label, terugverdien_label=terugverdien_label,
            fwd_pe_label=fwd_pe_label, groei_label=groei_label,
            marktkap_label=marktkap_label, analisten_label=analisten_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één bericht per exchange
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (6 - score) + f" {score}/6"

def format_bericht(exchange_name: str, signalen: List[HooglSignaal], alle: List[HooglSignaal], cfg: dict) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None

    nu        = today_str()
    top_n     = cfg["top_n"]
    top_tonen = sorted(alle, key=lambda s: (s.score, -s.terugverdienperiode), reverse=True)[:top_n]
    max_sc    = max((s.score for s in signalen), default=0) if signalen else 0
    lbl       = {6: "⭐ PERFECTE SCORE (6/6)", 5: "🟡 STERK (5/6)", 4: "🟠 GOED (4/6)"}.get(max_sc, "📊")

    def sig_regel(s: HooglSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"RoE:{s.roe_pct:.1f}% | Terugverdien:{s.terugverdienperiode:.1f}j | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += (
                f"\n  {s.roe_label} | {s.terugverdien_label}"
                f"\n  Fwd P/E: {s.fwd_pe_label} | Verw. winstgroei: {s.groei_label}"
                f"\n  Marktkap: {s.marktkap_label} | Coverage: {s.analisten_label}"
            )
        return r

    delen = [
        f"📈 *{cfg['label']} — {exchange_name}*",
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
        f"⚙️ _RoE>={cfg['roe_min']:.0f}% | terugverdienperiode<={cfg['terugverdien_max_jaar']:.0f}j | "
        f"forward P/E<={cfg['fwd_pe_max']:.0f}x | verwachte winstgroei>0% | "
        f"marktkap<={cfg['marktkap_max']/1e9:.0f}B | analisten<={cfg['analisten_max']}_"
    )
    return "\n\n".join(delen)


# ============================================================
# ENGINE  — gedeeld door live (x-lijsten) en full (a-lijsten)
# ============================================================

def run_engine(modus: str):
    cfg = MODUS_CFG[modus]
    print(f"{'='*60}")
    print(f"{cfg['label']}  {today_str()}  [bestand-suffix: {cfg['bestand_suffix']}]")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    for f_name in bouw_bestandslijst(cfg["bestand_suffix"]):
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
            sig = analyse_ticker(ticker, cfg)
            if sig is not None:
                alle.append(sig)
                if sig.score >= cfg["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/4 | RoE={sig.roe_pct:.1f}%")
            time.sleep(cfg["throttle_sec"])  # lichte throttle tegen Yahoo rate-limits

        kandidaten = [s for s in alle if s.score >= cfg["min_score"]]
        kandidaten.sort(key=lambda s: (s.score, -s.terugverdienperiode), reverse=True)
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
                    "roe_pct": s.roe_pct,
                    "terugverdienperiode": s.terugverdienperiode,
                    "forward_pe": s.forward_pe,
                    "verwachte_winstgroei_pct": s.verwachte_winstgroei_pct,
                    "div_yield": s.div_yield,
                    "market_cap": s.market_cap,
                    "analisten_count": s.analisten_count,
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
            f"{cfg['label']} {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


def run_backtest():
    print(
        "Backtest wordt niet ondersteund voor bot_01hoogl: yfinance biedt geen "
        "betrouwbare historische reeks van analyst-EPS-schattingen (forwardEps/forwardPE) "
        "per scandatum in het verleden, enkel de actuele consensus. "
        "Gebruik 'live' of 'full' om het huidige universum te screenen."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    elif mode == "full":
        run_engine("full")
    else:
        run_engine("live")
