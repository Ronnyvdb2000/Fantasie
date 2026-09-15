#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01beurssig.py  —  BEURSSIGNALEN-METHODIEK NAGEBOUWD  v1.0

Reverse-engineerde replica van de methode van beurssignalen.com, op basis
van hun eigen omschrijving (VWAP-analyse, Overhead Supply, volumebalken,
insiderscreening) en van de statistische analyse van hun resultaten-Excel
(65,5% winrate, geen vaste stop-loss, winnaars sneller verkocht dan
verliezers, universum = brede lijst bekende large/mid-caps i.p.v. een
fundamenteel gefilterde shortlist).

Criteria (score 0-3, of 0-4 op Amerikaanse beurzen):
  1. VWAP-trend       — koers > VWAP20 EN koers > VWAP50 (opwaartse trend
                         t.o.v. volume-gewogen gemiddelde koers), maar niet
                         meer dan 8% boven VWAP20 uitgelopen (geen achterna-
                         hollen van een reeds uitgebreide move).
  2. Overhead Supply   — geen betekenisvolle volume-node (top-kwartiel van
                         het volume-prijsprofiel over de laatste 60 dagen)
                         nog boven de huidige koers: de weerstandszone van
                         opgestapeld volume is "opgeruimd".
  3. Volumebalken      — recent volume >= 1,5x het 20-daags gemiddelde
                         (accumulatie-signaal).
  4. Insiderscreening  — ENKEL gemeten op Amerikaanse beurzen (048/055/056/057,
                         SEC Form 4-data via yfinance); netto insider-aankopen
                         > 0 over de laatste 90 dagen. Op alle andere beurzen
                         telt dit criterium niet mee (noch in teller, noch in
                         noemer) — score blijft daar dus op 0-3 staan, geen
                         kunstmatig lager percentage voor EU-aandelen.

TWEE MODI:
  live   — dagelijks (ma-vr), scant tickers_0NNx.txt (jouw eigen kwaliteits-
           gefilterde lijsten, ~1.900 tickers). Snel (~15-20 min).
  full   — wekelijks (zaterdag), scant tickers_0NNa.txt (ALLE tickers per
           beurs, ~13.400 in totaal) — dit is het universum dat qua
           samenstelling het dichtst bij Beurssignalen's eigen (niet
           kwaliteitsgefilterde) selectie ligt. Draait in het weekend
           wanneer geen enkele beurs open is, en omdat de looptijd te lang
           is voor een dagelijkse cron.

Belangrijk verschil met je andere bots: dit repliceert enkel hun INSTAP-
methodiek. De analyse van hun Excel toonde een asymmetrisch EXIT-patroon
(winnaars mediaan 27 dagen aangehouden, verliezers mediaan 62 dagen, geen
vaste stop-loss) — dat gedrag is bewust NIET nagebouwd, want "verliezers
laten lopen" is geen strategie die je zou willen kopiëren, enkel een
observatie over hun risicobeheer.

Rapportage: Telegram + email, één bericht per beurs, lege beurzen
overgeslagen — zelfde patroon als bot_01hoogl.py.

Supabase: logt naar de bestaande `selecties`-tabel (db_logger.py).
Nieuwe parameters (vwap20_pct, vwap50_pct, insider_net_shares) toegevoegd
aan db_logger.py's _KOLOM_WHITELIST — zie migratie_beurssig_kolommen.sql.
vol_ratio_20d en breakout worden HERGEBRUIKT (bestaan al als kolom).

Gebruik:
  python bot_01beurssig.py live      # dagelijks rapport (x-lijsten)
  python bot_01beurssig.py full      # wekelijks full-scan rapport (a-lijsten)
  python bot_01beurssig.py backtest  # niet ondersteund (insider-transactiedata
                                        heeft geen bruikbare historische reeks
                                        per scandatum via yfinance) — print
                                        uitleg en stopt
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
import numpy as np
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
    "055": "055 CBoe",
    "056": "056 NYSE int",
    "057": "057 NYSE",
    "058": "058 TSXV",
    "059": "059 Oostenrijk Slovenie Slovakije",
}

# Beursnummers waar SEC Form 4 insiderdata via yfinance betrouwbaar is
US_BEURS_NUMMERS = {"048", "055", "056", "057"}

def bouw_bestandslijst(suffix: str) -> List[str]:
    return [f"tickers_{n:03d}{suffix}.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    getal = f_name.replace("tickers_", "")[:3]
    return BEURS_NAMEN.get(getal, f_name.replace(".txt", ""))

def is_us_beurs(f_name: str) -> bool:
    getal = f_name.replace("tickers_", "")[:3]
    return getal in US_BEURS_NUMMERS

MODUS_CFG = {
    "live": {
        "bestand_suffix":  "x",
        "vwap_max_boven":  0.08,     # koers max 8% boven VWAP20
        "overhead_pctl":   75,       # top-kwartiel volume-nodes = "significant"
        "vol_ratio_min":   1.5,
        "min_score":       3,        # van 3 (niet-VS) of van 4 (VS, insider als bonus)
        "top_n":           5,
        "strategie":       "bot_01beurssig",
        "label":           "BEURSSIGNALEN-METHODIEK",
        "throttle_sec":    0.15,
    },
    "full": {
        "bestand_suffix":  "a",
        "vwap_max_boven":  0.06,     # strenger: minder ver boven VWAP toegelaten
        "overhead_pctl":   70,       # strenger: iets ruimere definitie van "overhead"
        "vol_ratio_min":   1.8,      # strenger: duidelijkere volume-piek vereist
        "min_score":       3,
        "top_n":           3,
        "strategie":       "bot_01beurssig_full",
        "label":           "BEURSSIGNALEN-METHODIEK — WEEKLY FULL SCAN",
        "throttle_sec":    0.12,
    },
}

# ============================================================
# HULPFUNCTIES (identiek patroon aan bot_01hoogl.py)
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
# TECHNISCHE ANALYSE — VWAP, OVERHEAD SUPPLY, VOLUME, INSIDERS
# ============================================================

@dataclass
class BeursSigSignaal:
    ticker:              str
    price:               float
    score:               int
    max_score:           int
    is_us:               bool
    vwap20_pct:          float   # % afstand koers t.o.v. VWAP20 (+ = erboven)
    vwap50_pct:          float
    vol_ratio_20d:       float
    overhead_cleared:    bool
    insider_net_shares:  float   # NaN = onbekend/niet van toepassing
    vwap_label:          str
    overhead_label:      str
    volume_label:        str
    insider_label:       str

def _vwap(closes: np.ndarray, volumes: np.ndarray, n: int) -> float:
    c, v = closes[-n:], volumes[-n:]
    if v.sum() <= 0:
        return float("nan")
    return float((c * v).sum() / v.sum())

def _overhead_cleared(closes: np.ndarray, volumes: np.ndarray, price: float, pctl: int) -> bool:
    """
    Bouwt een eenvoudig volume-prijsprofiel over de meegegeven reeks (laatste
    ~60 dagen) in 20 bins. Als geen enkele bin met een volume boven het
    `pctl`-percentiel een prijsrange heeft die (>1% boven) de huidige koers
    ligt, is de 'overhead supply' opgeruimd (koers staat vrij t.o.v. oud
    opgestapeld volume).
    """
    if len(closes) < 20 or volumes.sum() <= 0:
        return False
    hist_vol, edges = np.histogram(closes, bins=20, weights=volumes)
    nonzero = hist_vol[hist_vol > 0]
    if len(nonzero) == 0:
        return False
    drempel = np.percentile(nonzero, pctl)
    significante_bins = np.where(hist_vol >= drempel)[0]
    for i in significante_bins:
        bin_boven = edges[i + 1]  # bovengrens van de bin
        if bin_boven > price * 1.01:
            return False  # er hangt nog een significante volume-node boven de koers
    return True

def _insider_net_shares(ticker: str, dagen: int = 90) -> float:
    """Netto insider-aankopen (shares) over de laatste `dagen`. NaN = onbekend."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.insider_transactions
        if df is None or df.empty or "Start Date" not in df.columns:
            return float("nan")
        cutoff = dt.datetime.now() - dt.timedelta(days=dagen)
        df = df.copy()
        df["Start Date"] = pd_to_datetime_safe(df["Start Date"])
        recent = df[df["Start Date"] >= cutoff]
        if recent.empty:
            return float("nan")
        tekst = recent["Text"].astype(str).str.lower() if "Text" in recent.columns else recent.get("Transaction", "").astype(str).str.lower()
        shares = recent["Shares"].apply(safe_float) if "Shares" in recent.columns else recent.get("Value", 0).apply(safe_float)
        koop = shares[tekst.str.contains("purchase|buy", na=False)].sum()
        verkoop = shares[tekst.str.contains("sale|sell", na=False)].sum()
        return float(koop - verkoop)
    except Exception:
        return float("nan")

def pd_to_datetime_safe(series):
    import pandas as pd
    return pd.to_datetime(series, errors="coerce")

def analyse_ticker(ticker: str, cfg: dict, us_beurs: bool) -> Optional[BeursSigSignaal]:
    try:
        hist = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 55 or "Close" not in hist.columns:
            return None

        closes  = hist["Close"].to_numpy()
        volumes = hist["Volume"].to_numpy()
        price   = safe_float(closes[-1])
        if math.isnan(price) or price <= 0:
            return None

        score, max_score = 0, 3

        # 1. VWAP-trend
        vwap20 = _vwap(closes, volumes, 20)
        vwap50 = _vwap(closes, volumes, 50)
        vwap20_pct = (price / vwap20 - 1) * 100 if not math.isnan(vwap20) and vwap20 > 0 else float("nan")
        vwap50_pct = (price / vwap50 - 1) * 100 if not math.isnan(vwap50) and vwap50 > 0 else float("nan")
        vwap_ok = (
            not math.isnan(vwap20_pct) and not math.isnan(vwap50_pct)
            and vwap20_pct > 0 and vwap50_pct > 0
            and vwap20_pct <= cfg["vwap_max_boven"] * 100
        )
        if vwap_ok:
            score += 1
            vwap_label = f"✓ +{vwap20_pct:.1f}% boven VWAP20, +{vwap50_pct:.1f}% boven VWAP50"
        else:
            vwap_label = (
                f"✗ VWAP20:{vwap20_pct:+.1f}% | VWAP50:{vwap50_pct:+.1f}%"
                if not math.isnan(vwap20_pct) else "✗ onbekend"
            )

        # 2. Overhead Supply
        overhead_cleared = _overhead_cleared(closes[-60:], volumes[-60:], price, cfg["overhead_pctl"])
        if overhead_cleared:
            score += 1
            overhead_label = "✓ opgeruimd (geen significante volume-node meer boven koers)"
        else:
            overhead_label = "✗ nog weerstandszone (hoog-volume node) boven koers"

        # 3. Volumebalken
        basis_vol = volumes[-21:-1]
        vol_ratio_20d = float(volumes[-1] / basis_vol.mean()) if len(basis_vol) == 20 and basis_vol.mean() > 0 else float("nan")
        vol_ok = not math.isnan(vol_ratio_20d) and vol_ratio_20d >= cfg["vol_ratio_min"]
        if vol_ok:
            score += 1
            volume_label = f"✓ {vol_ratio_20d:.2f}x 20d-gemiddelde"
        else:
            volume_label = f"✗ {vol_ratio_20d:.2f}x" if not math.isnan(vol_ratio_20d) else "✗ onbekend"

        # 4. Insiderscreening — enkel op Amerikaanse beurzen
        insider_net_shares = float("nan")
        insider_label = "— n.v.t. (geen betrouwbare insiderdata buiten VS)"
        if us_beurs:
            max_score = 4
            insider_net_shares = _insider_net_shares(ticker)
            if not math.isnan(insider_net_shares) and insider_net_shares > 0:
                score += 1
                insider_label = f"✓ netto {insider_net_shares:,.0f} aandelen gekocht (90d)"
            elif not math.isnan(insider_net_shares):
                insider_label = f"✗ netto {insider_net_shares:,.0f} aandelen (90d)"
            else:
                insider_label = "✗ geen insidertransacties gevonden (90d)"

        return BeursSigSignaal(
            ticker=ticker, price=round(price, 2), score=score, max_score=max_score,
            is_us=us_beurs,
            vwap20_pct=round(vwap20_pct, 1) if not math.isnan(vwap20_pct) else 0.0,
            vwap50_pct=round(vwap50_pct, 1) if not math.isnan(vwap50_pct) else 0.0,
            vol_ratio_20d=round(vol_ratio_20d, 2) if not math.isnan(vol_ratio_20d) else 0.0,
            overhead_cleared=overhead_cleared,
            insider_net_shares=insider_net_shares,
            vwap_label=vwap_label, overhead_label=overhead_label,
            volume_label=volume_label, insider_label=insider_label,
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT — één bericht per beurs
# ============================================================

def _score_bar(score: int, max_score: int) -> str:
    return "█" * score + "░" * (max_score - score) + f" {score}/{max_score}"

def format_bericht(exchange_name: str, signalen: List[BeursSigSignaal], alle: List[BeursSigSignaal], cfg: dict) -> Optional[str]:
    if not alle:
        return None

    nu    = today_str()
    top_n = cfg["top_n"]
    top_tonen = sorted(alle, key=lambda s: (s.score / s.max_score, s.vol_ratio_20d), reverse=True)[:top_n]

    def sig_regel(s: BeursSigSignaal, detail: bool = False) -> str:
        r = (
            f"• `{s.ticker}` {_score_bar(s.score, s.max_score)} | {s.price:.2f} | "
            f"Vol:{s.vol_ratio_20d:.2f}x | {_yahoo_link(s.ticker)}"
        )
        if detail:
            r += f"\n  {s.vwap_label}\n  {s.overhead_label}\n  {s.volume_label}\n  {s.insider_label}"
        return r

    delen = [
        f"📊 *{cfg['label']} — {exchange_name}*",
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
        f"⚙️ _VWAP: koers boven VWAP20+VWAP50, max +{cfg['vwap_max_boven']*100:.0f}% erboven | "
        f"Overhead Supply opgeruimd | Volume>={cfg['vol_ratio_min']:.1f}x 20d-gem. | "
        f"Insiderscreening enkel op VS-beurzen_"
    )
    return "\n\n".join(delen)


# ============================================================
# ENGINE
# ============================================================

def run_engine(modus: str):
    cfg = MODUS_CFG[modus]
    print(f"{'='*60}")
    print(f"{cfg['label']}  {today_str()}  [bestand-suffix: {cfg['bestand_suffix']}]")
    print(f"{'='*60}")

    bestanden = bouw_bestandslijst(cfg["bestand_suffix"])
    email_delen: List[str] = []

    for f_name in bestanden:
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name  = label_voor(f_name)
        us_beurs = is_us_beurs(f_name)
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers, VS-beurs: {us_beurs})...")

        alle: List[BeursSigSignaal] = []
        for ticker in tlist:
            sig = analyse_ticker(ticker, cfg, us_beurs)
            if sig is not None:
                alle.append(sig)
                if sig.score >= cfg["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/{sig.max_score}")
            time.sleep(cfg["throttle_sec"])

        kandidaten = [s for s in alle if s.score >= cfg["min_score"]]
        kandidaten.sort(key=lambda s: (s.score / s.max_score, s.vol_ratio_20d), reverse=True)
        signalen = kandidaten[:cfg["top_n"]]

        print(f"  → top {len(signalen)} van {len(kandidaten)} kandidaten uit {len(alle)} geanalyseerd")

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
                    "vwap20_pct": s.vwap20_pct,
                    "vwap50_pct": s.vwap50_pct,
                    "vol_ratio_20d": s.vol_ratio_20d,
                    "breakout": s.overhead_cleared,
                    "insider_net_shares": None if math.isnan(s.insider_net_shares) else s.insider_net_shares,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, signalen, alle, cfg)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print("  → Telegram verstuurd")
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
        "Backtest wordt niet ondersteund voor bot_01beurssig: yfinance biedt geen "
        "betrouwbare historische reeks van insidertransacties per scandatum in het "
        "verleden (enkel de actuele lijst), en het volume-prijsprofiel voor "
        "'overhead supply' zou voor elke historische dag apart herberekend moeten "
        "worden op enkel de tot dan bekende data — te zwaar voor een eerste versie. "
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
