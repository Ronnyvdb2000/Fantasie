#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01repititief.py  —  SEIZOENSEFFECTEN ENGINE v1.1
Combineert weekdag-, maand- en kwartaaleffecten per ticker tot één
seizoensscore, met live signalen (Telegram/mail) én de onderliggende
historische statistiek (gemiddelde, win rate, p-waarde) mee in het bericht.

v1.1 t.o.v. v1.0:
  - Benjamini-Hochberg FDR-correctie (zelfde aanpak als bot_01cointegr.py)
    i.p.v. een vaste p<0.10 per individuele test. Elk buckettype (weekdag/
    maand/kwartaal) is een eigen "test-familie": de p-waarden van bv. alle
    weekdag-tests binnen één beurs worden samen gecorrigeerd, zo ook voor
    maand en kwartaal apart. Zonder deze correctie komt bij honderden
    tickers x 3 tests puur toeval al als "significant" naar boven.
  - Telegram-verzending met status-check + retry/backoff bij 429 (rate
    limit) en een korte throttle tussen berichten — voorheen faalden
    verzendingen naar dezelfde chat soms stil zonder foutmelding.
  - Backtest (maand-effect) past dezelfde BH-correctie toe: per testjaar
    worden de getrainde p-waarden per maand over alle tickers gepoold en
    gecorrigeerd, in plaats van een vaste p<0.10 per ticker/maand.

Opgebouwd op het patroon van bot_00kr.py: zelfde bestandslijst-opbouw,
zelfde download_history-hulpfunctie, zelfde db_logger-integratie, zelfde
live/backtest CLI.

Gebruik:
  python bot_01repititief.py live      # live rapport
  python bot_01repititief.py backtest  # walk-forward backtest (maand-effect)
"""

import os
import sys
import math
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy import stats

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

START_CAPITAL    = 50_000.0
SLIPPAGE_PCT     = 0.001
TRADE_COST_FIXED = 15.0
TRADE_COST_PCT   = 0.0035
TAX_RATE         = 0.10

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

TELEGRAM_THROTTLE_SEC = 1.2   # min. tijd tussen 2 berichten naar dezelfde chat
TELEGRAM_MAX_POGINGEN = 4

# Zelfde namenlijst/bestandsopbouw als bot_00kr.py
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

SZ_CFG = {
    "lookback_years": 15,   # hoeveel historiek max wordt opgehaald
    "min_jaren":      5,    # minimum aantal jaar data om een ticker mee te nemen
    "fdr_alpha":      0.10, # gewenste FDR (Benjamini-Hochberg) per buckettype/beurs
    "top_n":          8,    # aantal tickers per beurs in het Telegram-bericht
}

BUCKET_TYPES = ("weekdag", "maand", "kwartaal")
WEEKDAGEN = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag"]
MAANDEN   = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]

BACKTEST_START      = "2021-01-01"
BACKTEST_END        = dt.date.today().isoformat()
BACKTEST_MIN_TRAIN  = 5   # jaar trainingsdata vereist vóór een testjaar meetelt


# ============================================================
# HULPFUNCTIES — 1-op-1 uit bot_00kr.py
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

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

def send_telegram_message(text: str) -> bool:
    """
    Stuurt één Telegram-bericht. Controleert de statuscode (voorheen werd
    een 429/ander foutantwoord stil genegeerd) en retryt met backoff bij
    rate-limiting. Geeft True/False terug zodat de caller weet of het
    bericht écht is aangekomen.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for poging in range(1, TELEGRAM_MAX_POGINGEN + 1):
        try:
            resp = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=15,
            )
        except Exception as e:
            print(f"Telegram fout (poging {poging}): {e}")
            time.sleep(2 * poging)
            continue

        if resp.status_code == 200:
            return True

        if resp.status_code == 429:
            retry_after = 3
            try:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 3)
            except Exception:
                pass
            print(f"Telegram 429 (rate limit), wacht {retry_after}s en probeer opnieuw...")
            time.sleep(retry_after + 0.5)
            continue

        print(f"Telegram fout {resp.status_code}: {resp.text[:300]}")
        return False

    print("Telegram: alle pogingen mislukt, bericht NIET verstuurd.")
    return False

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
# DATA DOWNLOAD — 1-op-1 uit bot_00kr.py
# ============================================================

def _normalise(df_raw, ticker: str) -> Optional[pd.DataFrame]:
    if df_raw is None or not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        return None
    df = df_raw.copy().dropna(how="all")
    if df.empty:
        return None
    if df.index.name in ("Date", "Datetime") or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Date" in df.columns:
        df = df.loc[:, ~df.columns.duplicated()]
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    df["Ticker"] = ticker
    return df

def download_history(tickers: List[str], period: str = "15y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    kwargs = dict(tickers=tickers, auto_adjust=True, group_by="ticker",
                  progress=False, threads=True, period=period)
    frames = []
    try:
        data = yf.download(**kwargs)
    except Exception as e:
        print(f"[WARN] Batch mislukt ({e}), probeer 1-voor-1...")
        data = pd.DataFrame()

    if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            ticker_level = 1
            for lvl in range(data.columns.nlevels):
                if any(t in set(data.columns.get_level_values(lvl)) for t in tickers):
                    ticker_level = lvl
                    break
            available = set(data.columns.get_level_values(ticker_level))
            for t in tickers:
                if t not in available:
                    continue
                try:
                    norm = _normalise(data.xs(t, axis=1, level=ticker_level).copy(), t)
                    if norm is not None:
                        frames.append(norm)
                except Exception as e:
                    print(f"[WARN] {t}: fout ({e})")
        else:
            norm = _normalise(data, tickers[0])
            if norm is not None:
                frames.append(norm)

    if not frames:
        for t in tickers:
            try:
                raw = yf.download(t, period=period, auto_adjust=True, progress=False)
                if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                    continue
                if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                norm = _normalise(raw, t)
                if norm is not None:
                    frames.append(norm)
                time.sleep(0.2)
            except Exception as e:
                print(f"[WARN] {t}: mislukt ({e})")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# BENJAMINI-HOCHBERG FDR-CORRECTIE
# ============================================================

def bh_correctie(p_waarden: List[float], alpha: float) -> List[bool]:
    """
    Standaard Benjamini-Hochberg procedure. Geeft per index terug of die
    test significant is NA correctie voor multiple testing.
    p_(rang) <= (rang/m) * alpha  ->  grootste rang die hieraan voldoet
    bepaalt de afkap; alle testen met een kleinere of gelijke rang worden
    significant verklaard.
    """
    m = len(p_waarden)
    if m == 0:
        return []
    volgorde = sorted(range(m), key=lambda i: p_waarden[i])
    laatste_significante_rank = -1
    for rang, idx in enumerate(volgorde, start=1):
        if p_waarden[idx] <= (rang / m) * alpha:
            laatste_significante_rank = rang
    mask = [False] * m
    if laatste_significante_rank >= 0:
        for rang, idx in enumerate(volgorde, start=1):
            if rang <= laatste_significante_rank:
                mask[idx] = True
    return mask


# ============================================================
# SEIZOENSANALYSE
# ============================================================

@dataclass
class SeizoenKandidaat:
    """Ruwe, ongecorrigeerde bucket-statistieken — vóór BH-correctie."""
    ticker:     str
    price:      float
    weekdag:    Optional[dict]
    maand:      Optional[dict]
    kwartaal:   Optional[dict]
    jaren_data: float

@dataclass
class SeizoenSignaal:
    """Eindresultaat na BH-correctie: enkel de significante buckets tellen mee."""
    ticker:               str
    price:                float
    score:                float
    weekdag:              Optional[dict]     # ruwe stats, altijd bewaard voor logging
    maand:                Optional[dict]
    kwartaal:             Optional[dict]
    significante_buckets: Set[str]            # subset van BUCKET_TYPES die BH doorstond
    jaren_data:           float

def bereken_bucket_stats(rendementen: np.ndarray) -> Optional[dict]:
    """(gemiddelde, win_rate, p_waarde, n) voor een reeks dagrendementen."""
    n = len(rendementen)
    if n < 5:
        return None
    gemiddelde = float(np.mean(rendementen))
    win_rate   = float(np.mean(rendementen > 0))
    _, p_waarde = stats.ttest_1samp(rendementen, 0.0)
    return {"gemiddelde": gemiddelde, "win_rate": win_rate, "p_waarde": float(p_waarde), "n": n}

def analyseer_ticker(ticker: str, df_ticker: pd.DataFrame) -> Optional[SeizoenKandidaat]:
    """Berekent de ruwe bucket-stats. Geen significantie-filtering hier —
    dat gebeurt achteraf, gepoold over alle tickers van de beurs (BH)."""
    try:
        g = df_ticker.sort_values("Date").copy()
        if len(g) < 250 * SZ_CFG["min_jaren"]:
            return None

        g = g.set_index("Date")
        rendement = g["Close"].pct_change().dropna()
        aantal_jaren = (rendement.index[-1] - rendement.index[0]).days / 365.25
        if aantal_jaren < SZ_CFG["min_jaren"]:
            return None

        current_price = safe_float(g["Close"].iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        vandaag = dt.datetime.now(dt.timezone.utc)
        huidige_weekdag = vandaag.weekday()               # 0=maandag ... 4=vrijdag
        huidige_maand   = vandaag.month
        huidig_kwartaal = (vandaag.month - 1) // 3 + 1

        weekdag_stats  = bereken_bucket_stats(rendement[rendement.index.weekday == huidige_weekdag].values)
        maand_stats    = bereken_bucket_stats(rendement[rendement.index.month == huidige_maand].values)
        kwartaal_stats = bereken_bucket_stats(rendement[rendement.index.quarter == huidig_kwartaal].values)

        if not (weekdag_stats or maand_stats or kwartaal_stats):
            return None

        return SeizoenKandidaat(
            ticker=ticker, price=round(current_price, 2),
            weekdag=weekdag_stats, maand=maand_stats, kwartaal=kwartaal_stats,
            jaren_data=round(aantal_jaren, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None

def pas_bh_toe_op_beurs(kandidaten: List[SeizoenKandidaat], alpha: float) -> List[SeizoenSignaal]:
    """
    Voert per buckettype (weekdag/maand/kwartaal) een eigen BH-correctie uit,
    gepoold over alle kandidaten van deze beurs — dat zijn de drie
    test-families. Enkel kandidaten met minstens 1 buckettype dat de
    correctie doorstaat, komen terug als SeizoenSignaal.
    """
    significant_per_kandidaat: Dict[int, Set[str]] = {}

    for bucket_key in BUCKET_TYPES:
        indices = [i for i, k in enumerate(kandidaten) if getattr(k, bucket_key) is not None]
        if not indices:
            continue
        p_waarden = [getattr(kandidaten[i], bucket_key)["p_waarde"] for i in indices]
        mask = bh_correctie(p_waarden, alpha)
        for pos, kandidaat_idx in enumerate(indices):
            if mask[pos]:
                significant_per_kandidaat.setdefault(kandidaat_idx, set()).add(bucket_key)

    resultaten: List[SeizoenSignaal] = []
    for i, k in enumerate(kandidaten):
        sig_keys = significant_per_kandidaat.get(i)
        if not sig_keys:
            continue
        score = sum(
            getattr(k, key)["gemiddelde"] * (1 - getattr(k, key)["p_waarde"])
            for key in sig_keys
        )
        resultaten.append(SeizoenSignaal(
            ticker=k.ticker, price=k.price, score=score,
            weekdag=k.weekdag, maand=k.maand, kwartaal=k.kwartaal,
            significante_buckets=sig_keys, jaren_data=k.jaren_data,
        ))
    return resultaten


# ============================================================
# TELEGRAM + EMAIL OUTPUT — één bericht per exchange
# ============================================================

def _bucket_regel(key: str, stat: Optional[dict], significante_buckets: Set[str]) -> Optional[str]:
    if key not in significante_buckets or not stat:
        return None
    vandaag = dt.datetime.now(dt.timezone.utc)
    if key == "weekdag":
        kort, label = "wk", WEEKDAGEN[vandaag.weekday()]
    elif key == "maand":
        kort, label = "mnd", MAANDEN[vandaag.month - 1]
    else:
        kort, label = "kw", f"Q{(vandaag.month - 1)//3 + 1}"
    return (f"{kort} {label}: {stat['gemiddelde']*100:+.2f}% "
            f"(winrate {stat['win_rate']*100:.0f}%, n={stat['n']}, p={stat['p_waarde']:.3f})")

def sig_regel(s: SeizoenSignaal) -> str:
    onderbouwing = [r for r in (
        _bucket_regel("weekdag", s.weekdag, s.significante_buckets),
        _bucket_regel("maand", s.maand, s.significante_buckets),
        _bucket_regel("kwartaal", s.kwartaal, s.significante_buckets),
    ) if r]
    return (
        f"• `{s.ticker}` score {s.score*100:+.2f}% | EUR{s.price:.2f} | "
        f"{s.jaren_data:.0f}j data | {_yahoo_link(s.ticker)}\n"
        f"  {' | '.join(onderbouwing)}"
    )

def format_bericht(exchange_name: str, top: List[SeizoenSignaal], alle: List[SeizoenKandidaat]) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None
    nu = today_str()
    delen = [
        f"📅 *SEIZOENSEFFECTEN — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(top)} met FDR-significant seizoenspatroon_",
        "─────────────────────────────",
        "\n\n".join(sig_regel(s) for s in top),
        "─────────────────────────────",
        f"⚙️ _Weekdag/maand/kwartaal t.o.v. eigen historiek (max {SZ_CFG['lookback_years']}j) | "
        f"Benjamini-Hochberg FDR={SZ_CFG['fdr_alpha']:.2f} per buckettype | "
        f"score = gewogen som significante gemiddeldes_",
    ]
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"SEIZOENSEFFECTEN — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for f_name in bouw_bestandslijst():
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name = label_voor(f_name)
        exchange_tickers[ex_name] = tlist
        all_tickers.extend(tlist)
        print(f"  {ex_name}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print(f"Data downloaden ({SZ_CFG['lookback_years']} jaar)...")
    df = download_history(all_tickers, period=f"{SZ_CFG['lookback_years']}y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers, {len(df)} rijen")
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        kandidaten: List[SeizoenKandidaat] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            k = analyseer_ticker(ticker, group)
            if k is not None:
                kandidaten.append(k)

        if not kandidaten:
            print(f"  → Overgeslagen: {ex_name} (te weinig data)")
            continue

        top_alles = pas_bh_toe_op_beurs(kandidaten, SZ_CFG["fdr_alpha"])
        print(f"  {len(kandidaten)} geanalyseerd, {len(top_alles)} FDR-significant")
        for s in top_alles:
            print(f"  ✓ {s.ticker}: score {s.score*100:+.2f}% | buckets: {', '.join(sorted(s.significante_buckets))}")

        if not top_alles:
            print(f"  → Overgeslagen: {ex_name} (niets overleeft de FDR-correctie)")
            continue

        top_alles.sort(key=lambda s: s.score, reverse=True)
        top = top_alles[:SZ_CFG["top_n"]]

        for s in top:
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01repititief",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                    "jaren_data": s.jaren_data,
                    "significante_buckets": sorted(s.significante_buckets),
                    "weekdag_gemiddelde": s.weekdag["gemiddelde"] if s.weekdag else None,
                    "weekdag_winrate":    s.weekdag["win_rate"] if s.weekdag else None,
                    "weekdag_p":          s.weekdag["p_waarde"] if s.weekdag else None,
                    "maand_gemiddelde":   s.maand["gemiddelde"] if s.maand else None,
                    "maand_winrate":      s.maand["win_rate"] if s.maand else None,
                    "maand_p":            s.maand["p_waarde"] if s.maand else None,
                    "kwartaal_gemiddelde": s.kwartaal["gemiddelde"] if s.kwartaal else None,
                    "kwartaal_winrate":    s.kwartaal["win_rate"] if s.kwartaal else None,
                    "kwartaal_p":          s.kwartaal["p_waarde"] if s.kwartaal else None,
                },
            )

        bericht = format_bericht(ex_name, top, kandidaten)
        if bericht:
            ok = send_telegram_message(bericht)
            print(f"  → Telegram {'verstuurd' if ok else 'MISLUKT'}")
            email_delen.append(bericht)
            time.sleep(TELEGRAM_THROTTLE_SEC)  # voorkom rate-limit bij volgende beurs

    if email_delen:
        send_email(
            f"Seizoenseffecten rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE — walk-forward op het maand-effect (met BH)
# ============================================================
# Enkel het maand-effect wordt hier effectief "verhandeld": voor elk testjaar
# wordt de maand-bucket-statistiek herberekend op basis van UITSLUITEND de
# jaren die vóór dat testjaar liggen (expanding window, geen look-ahead).
# Per testjaar en per kalendermaand worden de getrainde p-waarden over ALLE
# tickers gepoold en BH-gecorrigeerd (zelfde principe als in de live-engine);
# enkel wie na correctie significant én positief is, wordt verhandeld.
# Weekdag- en kwartaaleffect staan in live-mode als extra onderbouwing, maar
# worden hier niet apart backtest — dat is een vergelijkbare tweede loop,
# laat het weten als je die ook wil.

def _train_maand_stats_per_ticker(rendement: pd.Series, test_jaar: int) -> Dict[int, dict]:
    """Ruwe (ongecorrigeerde) maand-bucket-stats op basis van de trainingsjaren."""
    train = rendement[rendement.index.year < test_jaar]
    if train.empty:
        return {}
    jaren_train = (train.index[-1] - train.index[0]).days / 365.25
    if jaren_train < BACKTEST_MIN_TRAIN:
        return {}
    resultaat = {}
    for maand in range(1, 13):
        stat = bereken_bucket_stats(train[train.index.month == maand].values)
        if stat:
            resultaat[maand] = stat
    return resultaat

def run_backtest():
    print(f"{'='*60}")
    print(f"SEIZOENSEFFECTEN BACKTEST (maand-effect, BH-gecorrigeerd)  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for f_name in bouw_bestandslijst():
        all_tickers.extend(load_tickers_from_file(f_name))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden ({SZ_CFG['lookback_years']}y)...")
    df = download_history(all_tickers, period=f"{SZ_CFG['lookback_years']}y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    # Per ticker de rendementenreeks + geïndexeerde koersen klaarzetten
    reeksen: Dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date").set_index("Date")
        rendement = g["Close"].pct_change().dropna()
        if not rendement.empty:
            reeksen[ticker] = (rendement, g)

    cash = START_CAPITAL
    trades: List[Dict] = []
    testjaren = range(pd.Timestamp(BACKTEST_START).year, pd.Timestamp(BACKTEST_END).year + 1)

    for test_jaar in testjaren:
        # 1) per ticker de ruwe getrainde maand-stats ophalen (geen look-ahead)
        getraind: Dict[str, Dict[int, dict]] = {}
        for ticker, (rendement, _g) in reeksen.items():
            stats_per_maand = _train_maand_stats_per_ticker(rendement, test_jaar)
            if stats_per_maand:
                getraind[ticker] = stats_per_maand

        if not getraind:
            continue

        # 2) BH-correctie per kalendermaand, gepoold over alle tickers dit testjaar
        significant_dit_jaar: Dict[str, Set[int]] = {}
        for maand in range(1, 13):
            tickers_met_maand = [t for t, sm in getraind.items() if maand in sm]
            if not tickers_met_maand:
                continue
            p_waarden = [getraind[t][maand]["p_waarde"] for t in tickers_met_maand]
            mask = bh_correctie(p_waarden, SZ_CFG["fdr_alpha"])
            for t, sig in zip(tickers_met_maand, mask):
                if sig and getraind[t][maand]["gemiddelde"] > 0:
                    significant_dit_jaar.setdefault(t, set()).add(maand)

        # 3) enkel de significante ticker/maand-combinaties effectief verhandelen
        for ticker, maanden in significant_dit_jaar.items():
            _rendement, g = reeksen[ticker]
            for maand in maanden:
                maand_data = g[(g.index.year == test_jaar) & (g.index.month == maand)]
                if len(maand_data) < 5:
                    continue

                entry_price = safe_float(maand_data["Close"].iloc[0]) * (1 + SLIPPAGE_PCT)
                exit_price  = safe_float(maand_data["Close"].iloc[-1]) * (1 - SLIPPAGE_PCT)
                if math.isnan(entry_price) or math.isnan(exit_price) or entry_price <= 0:
                    continue

                risico_bedrag = cash * 0.05
                aandelen = max(1, int(risico_bedrag / entry_price))

                investering = entry_price * aandelen + trade_cost(entry_price * aandelen)
                gross = exit_price * aandelen
                cost  = trade_cost(gross)
                pnl   = gross - cost - investering
                tax   = pnl * TAX_RATE if pnl > 0 else 0.0
                net   = pnl - tax

                trades.append({
                    "ticker": ticker, "jaar": test_jaar, "maand": maand,
                    "train_gemiddelde": getraind[ticker][maand]["gemiddelde"],
                    "train_p": getraind[ticker][maand]["p_waarde"],
                    "entry_price": round(entry_price, 4), "exit_price": round(exit_price, 4),
                    "size": aandelen, "pnl": round(pnl, 2), "tax": round(tax, 2),
                    "net": round(net, 2),
                })

    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("seizoen_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        totaal_net = tdf["net"].sum()
        print(f"\n{'='*60}")
        print(f"Trades (maand-effect, BH) : {n}")
        print(f"Winnaars                  : {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor              : {pf:.2f}")
        print(f"Netto resultaat            : EUR{totaal_net:,.2f} (som over alle posities, geen samengesteld kapitaal)")
        print(f"Belasting                  : EUR{tdf['tax'].sum():,.2f}")
        print(f"{'='*60}")
        print("Opgeslagen: seizoen_backtest_trades.csv")
    else:
        print("Geen trades gegenereerd (na BH-correctie bleef geen enkele ticker/maand-combinatie over).")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
