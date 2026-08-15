#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01cointegr.py  — STATISTISCHE ARBITRAGE ENGINE v1.1
Gebaseerd op co-integratie en mean reversion (Pairs Trading).

Strategie:
  - Test paren binnen elke beurs op co-integratie (Engle-Granger),
    met een correlatie-prefilter en Benjamini-Hochberg FDR-correctie
    (voorkomt valse cointegratie door multiple testing op duizenden paren)
  - Handel de spread via z-score: entry bij ±2σ, exit bij ±0.5σ
  - Backtest = walk-forward: paren-selectie gebeurt op een in-sample
    venster (IS_WINDOW), en wordt vervolgens RESCAN_DAYS lang out-of-
    sample verhandeld voor herselectie — geen look-ahead, geen
    in-sample bias

v1.1 fixes t.o.v. v1.0:
  - Backtest segmenteert per beurs (i.p.v. alle 19 beurzen samen te
    testen — dat maakte runs potentieel dagen lang)
  - Correlatie-prefilter vóór de dure coint()-test
  - Cointegratie-rescan elke ~20 handelsdagen i.p.v. elke 5 dagen;
    z-score-checks tussendoor blijven dagelijks en zijn goedkoop
  - Benjamini-Hochberg FDR-correctie op de p-waarden (multiple testing)
  - Walk-forward: selectievenster en handelsvenster overlappen niet
  - Half-life via OLS mét intercept (was: regressie door oorsprong)
  - Echte marge-reservering bij het openen van een positie (was: cash
    werd alleen bij sluiten aangepast, waardoor MAX_TOTAL_PAIRS geen
    harde kapitaalgrens was)
  - statsmodels-import naar de top (was: per functie-aanroep)

Gebruik:
  python bot_01cointegr.py live      # live rapport (Telegram + Email)
  python bot_01cointegr.py backtest  # walk-forward backtest
"""

import os
import sys
import math
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

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

START_CAPITAL        = 50_000.0
MAX_PAIRS            = 5           # max aantal paren per beurs
MAX_TOTAL_PAIRS      = 20          # max totaal open posities over alle beurzen
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10

# Pairs trading parameters
LOOKBACK             = 60          # rolling window voor z-score
ENTRY_Z              = 2.0         # entry drempel
EXIT_Z               = 0.5         # exit drempel
MAX_HALF_LIFE         = 30          # maximale half-life in dagen

# Multiple-testing & performance parameters
CORR_PREFILTER       = 0.7         # min. |correlatie| vóór coint()-test wordt gedraaid
FDR_ALPHA            = 0.05        # Benjamini-Hochberg FDR-drempel op p-waarden
IS_WINDOW            = 252         # in-sample venster voor paren-selectie (~1 handelsjaar)
RESCAN_DAYS          = 20          # herselecteer paren elke ~20 handelsdagen (out-of-sample erna)
ALLOC_PCT            = 0.10        # % van vrij kapitaal gereserveerd per nieuw paar

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

# ============================================================
# HERGEBRUIK: Laad tickers uit jouw bestaande bestanden
# ============================================================

def bouw_bestandslijst() -> List[str]:
    """Bouwt lijst van tickers_041a.txt t/m tickers_059a.txt (jouw formaat)."""
    return [f"tickers_{n:03d}a.txt" for n in range(41, 60)]

def laad_tickers_uit_bestand(path: str) -> List[str]:
    """Leest tickers uit bestand — hergebruik logica uit jouw andere bots."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(";", ",").replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))

BEURS_NAMEN = {
    "tickers_041a.txt": "041 Benelux Ierland",
    "tickers_042a.txt": "042 Parijs",
    "tickers_043a.txt": "043 Frankfurt",
    "tickers_044a.txt": "044 Spanje/Portugal",
    "tickers_045a.txt": "045 Londen",
    "tickers_046a.txt": "046 Milaan",
    "tickers_047a.txt": "047 Toronto",
    "tickers_048a.txt": "048 Nasdaq/NYSE",
    "tickers_049a.txt": "049 Stockholm",
    "tickers_050a.txt": "050 Zurich",
    "tickers_051a.txt": "051 Warschau",
    "tickers_052a.txt": "052 Oslo",
    "tickers_053a.txt": "053 Kopenhagen",
    "tickers_054a.txt": "054 Helsinki",
    "tickers_055a.txt": "055 CBoe",
    "tickers_056a.txt": "056 NYSE int",
    "tickers_057a.txt": "057 NYSE",
    "tickers_058a.txt": "058 TSXV",
    "tickers_059a.txt": "059 Oostenrijk Slovenie Slovakije",
}

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

def laad_exchange_tickers() -> Dict[str, List[str]]:
    """Bouwt {beursnaam: [tickers]} — identiek gebruikt door live en backtest,
    zodat beide op dezelfde per-beurs segmentatie draaien."""
    exchange_tickers: Dict[str, List[str]] = {}
    for f_name in bouw_bestandslijst():
        tlist = laad_tickers_uit_bestand(f_name)
        if not tlist:
            continue
        exchange_tickers[label_voor(f_name)] = tlist
    return exchange_tickers

# ============================================================
# HERGEBRUIK: Email & Telegram functies
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "Markdown", "disable_web_page_preview": True},
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
    return f"[{ticker}](https://finance.yahoo.com/quote/{ticker})"

# ============================================================
# CO-INTEGRATIE & HALF-LIFE
# ============================================================

def schat_half_life(spread: pd.Series) -> float:
    """Schat half-life van mean reversion (in dagen) via OLS mét intercept:
    spread_diff = alpha + beta * spread_lag."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    min_len = min(len(spread_lag), len(spread_diff))
    if min_len < 10:
        return float('inf')
    spread_lag = spread_lag.iloc[-min_len:]
    spread_diff = spread_diff.iloc[-min_len:]
    try:
        X = sm.add_constant(spread_lag.values)
        model = sm.OLS(spread_diff.values, X).fit()
        beta = model.params[1]
    except Exception:
        return float('inf')
    if beta < 0:
        return -np.log(2) / beta
    return float('inf')

def fdr_correctie(p_waarden: List[float], alpha: float = FDR_ALPHA) -> List[bool]:
    """Benjamini-Hochberg FDR-correctie voor multiple testing.
    Retourneert een boolean mask (zelfde volgorde als p_waarden) die aangeeft
    welke p-waarden significant blijven na correctie."""
    n = len(p_waarden)
    if n == 0:
        return []
    geindexeerd = sorted(enumerate(p_waarden), key=lambda x: x[1])
    grens_idx = -1
    for k, (_, p) in enumerate(geindexeerd):
        drempel = (k + 1) / n * alpha
        if p <= drempel:
            grens_idx = k
    mask = [False] * n
    for k in range(grens_idx + 1):
        orig_idx, _ = geindexeerd[k]
        mask[orig_idx] = True
    return mask

# ============================================================
# DATA DOWNLOAD
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

def download_history(tickers: List[str], period: str = "2y") -> pd.DataFrame:
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
                except Exception:
                    continue
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
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ============================================================
# PAIR ANALYSE
# ============================================================

@dataclass
class PairSignaal:
    pair:          str
    ticker_a:      str
    ticker_b:      str
    p_value:       float
    half_life:     float
    current_z:     float
    spread:        float
    spread_mean:   float
    spread_std:    float
    price_a:       float
    price_b:       float
    signal:        str          # "LONG", "SHORT", of "NEUTRAL"
    score:         float

def analyseer_pair_ruw(
    ticker_a: str,
    ticker_b: str,
    data: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> Optional[PairSignaal]:
    """Analyseer 1 paar (z-score + co-integratie + half-life), zónder de
    p-waarde-drempel toe te passen — die gebeurt achteraf via FDR-correctie
    over de hele geteste batch (zie selecteer_gecointegreerde_paren)."""
    try:
        if ticker_a not in data.columns or ticker_b not in data.columns:
            return None
        if len(data[ticker_a].dropna()) < lookback or len(data[ticker_b].dropna()) < lookback:
            return None

        spread = data[ticker_a] / data[ticker_b]
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()

        current_spread = spread.iloc[-1]
        current_mean = spread_mean.iloc[-1]
        current_std = spread_std.iloc[-1]

        if math.isnan(current_std) or current_std <= 0:
            return None

        z_score = (current_spread - current_mean) / current_std

        _, p_value, _ = coint(data[ticker_a], data[ticker_b])

        half_life = schat_half_life(spread)
        if half_life > MAX_HALF_LIFE or half_life <= 0:
            return None

        if z_score > ENTRY_Z:
            signal = "SHORT"
        elif z_score < -ENTRY_Z:
            signal = "LONG"
        else:
            signal = "NEUTRAL"

        score_val = (1 / (p_value + 0.001)) * (1 / (half_life + 1))

        # LET OP: coint()/pandas geven numpy float64 terug. round() op een
        # float64 blijft een float64 — als dat ongewijzigd in een SQL-query
        # terechtkomt (via db_logger) wordt het letterlijk als "np.float64(...)"
        # ingevoegd, wat de insert doet crashen ("schema np does not exist").
        # Daarom hier expliciet casten naar native Python float.
        return PairSignaal(
            pair=f"{ticker_a}/{ticker_b}",
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            p_value=float(round(p_value, 4)),
            half_life=float(round(half_life, 1)),
            current_z=float(round(z_score, 2)),
            spread=float(round(current_spread, 4)),
            spread_mean=float(round(current_mean, 4)),
            spread_std=float(round(current_std, 4)),
            price_a=float(round(data[ticker_a].iloc[-1], 2)),
            price_b=float(round(data[ticker_b].iloc[-1], 2)),
            signal=signal,
            score=float(score_val),
        )
    except Exception:
        return None

def selecteer_gecointegreerde_paren(
    tickers: List[str],
    data: pd.DataFrame,
    corr_prefilter: float = CORR_PREFILTER,
    fdr_alpha: float = FDR_ALPHA,
    lookback: int = LOOKBACK,
) -> List[PairSignaal]:
    """Selecteert co-geïntegreerde paren uit een ticker-universum:
    1) correlatie-prefilter (goedkoop) om de kandidatenset te beperken
    2) coint()-test enkel op de overgebleven kandidaten
    3) Benjamini-Hochberg FDR-correctie op alle p-waarden in de batch,
       zodat multiple testing over duizenden paren geen valse
       cointegratie oplevert
    """
    beschikbaar = [t for t in tickers if t in data.columns]
    if len(beschikbaar) < 2:
        return []

    sub = data[beschikbaar].dropna()
    if sub.shape[1] < 2 or len(sub) < lookback:
        return []

    corr_matrix = sub.corr().abs()
    kandidaten = [
        (a, b) for a, b in combinations(beschikbaar, 2)
        if a in corr_matrix.index and b in corr_matrix.columns
        and corr_matrix.loc[a, b] >= corr_prefilter
    ]
    if not kandidaten:
        return []

    ruw: List[PairSignaal] = []
    for a, b in kandidaten:
        sig = analyseer_pair_ruw(a, b, sub, lookback)
        if sig is not None:
            ruw.append(sig)
    if not ruw:
        return []

    mask = fdr_correctie([s.p_value for s in ruw], fdr_alpha)
    significant = [s for s, keep in zip(ruw, mask) if keep]
    significant.sort(key=lambda s: s.score, reverse=True)
    return significant

def snel_zscore(
    ticker_a: str,
    ticker_b: str,
    data: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> Optional[Tuple[float, float, float, float]]:
    """Goedkope z-score-herberekening voor een reeds geselecteerd paar
    (géén coint-test) — gebruikt voor de dagelijkse entry/exit-checks
    tussen twee walk-forward herselecties in."""
    if ticker_a not in data.columns or ticker_b not in data.columns:
        return None
    spread = data[ticker_a] / data[ticker_b]
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    current_spread = spread.iloc[-1]
    current_mean = spread_mean.iloc[-1]
    current_std = spread_std.iloc[-1]
    if math.isnan(current_std) or current_std <= 0:
        return None
    z = (current_spread - current_mean) / current_std
    return z, current_spread, current_mean, current_std

# ============================================================
# PAIRS ENGINE — LIVE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"PAIRS TRADING — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers = laad_exchange_tickers()
    all_tickers = sorted(set(t for tlist in exchange_tickers.values() for t in tlist))
    for ex_name, tlist in exchange_tickers.items():
        print(f"  {ex_name}: {len(tlist)} tickers")

    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print("Koersdata downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    pivot = pivot.dropna(axis=1, how='all')
    print(f"Data geladen: {len(pivot.columns)} tickers met voldoende data")

    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")

        signalen = selecteer_gecointegreerde_paren(tlist, pivot)
        top_signalen = signalen[:MAX_PAIRS]
        actieve_signalen = [s for s in top_signalen if s.signal != "NEUTRAL"]

        print(f"  → {len(signalen)} co-geintegreerde paren (na FDR-correctie)")
        print(f"  → {len(actieve_signalen)} actieve signalen")

        if actieve_signalen:
            for s in actieve_signalen[:3]:
                log_selectie(
                    ticker=s.pair,
                    datum=today_str(),
                    strategie="bot_01cointegr",
                    beurs=ex_name,
                    koers=s.spread,
                    parameters={
                        "signal": s.signal,
                        "z_score": s.current_z,
                        "p_value": s.p_value,
                        "half_life": s.half_life,
                        "spread_mean": s.spread_mean,
                        "spread_std": s.spread_std,
                        "price_a": s.price_a,
                        "price_b": s.price_b,
                        "score": s.score,
                        "grafiek": f"https://finance.yahoo.com/quote/{s.ticker_a}/chart?p={s.ticker_a}",
                    },
                )

            bericht = format_pairs_bericht(ex_name, actieve_signalen)
            if bericht:
                send_telegram_message(bericht)
                email_delen.append(bericht)
                print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Geen actieve signalen voor {ex_name}")

    if email_delen:
        send_email(
            subject=f"PAIRS TRADING rapport {today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")

# ============================================================
# FORMATTER — Telegram bericht
# ============================================================

def format_pairs_bericht(exchange_name: str, signalen: List[PairSignaal]) -> Optional[str]:
    if not signalen:
        return None

    nu = today_str()

    def signal_emoji(s: str) -> str:
        if s == "LONG":
            return "🟢 LONG"
        elif s == "SHORT":
            return "🔴 SHORT"
        return "⚪ NEUTRAL"

    top = signalen[:3]

    delen = [
        f"📊 *PAIRS TRADING — {exchange_name}*",
        f"_{nu} | {len(signalen)} actieve signals_",
        "─────────────────────────────",
        "🏆 *TOP SIGNALEN:*",
    ]

    for s in top:
        delen.append(
            f"• {signal_emoji(s.signal)} `{s.pair}`\n"
            f"  Z-score: {s.current_z:.2f} | p={s.p_value:.4f} | Half-life: {s.half_life:.1f}d\n"
            f"  Spread: {s.spread:.4f} (mean={s.spread_mean:.4f}, σ={s.spread_std:.4f})\n"
            f"  {_yahoo_link(s.ticker_a)} / {_yahoo_link(s.ticker_b)} | "
            f"Koers: {s.price_a:.2f} / {s.price_b:.2f}"
        )

    if len(signalen) > 3:
        extra = signalen[3:]
        delen.append("─────────────────────────────")
        delen.append("📋 *Overige signalen:*")
        for s in extra:
            emoji = "🟢" if s.signal == "LONG" else "🔴" if s.signal == "SHORT" else "⚪"
            delen.append(f"  {emoji} `{s.pair}` | z={s.current_z:.2f} | p={s.p_value:.4f}")

    delen.append(
        f"\n⚙️ _Entry: ±{ENTRY_Z}σ | Exit: ±{EXIT_Z}σ | "
        f"Lookback: {LOOKBACK}d | FDR α: {FDR_ALPHA}_"
    )

    return "\n\n".join(delen)

# ============================================================
# BACKTEST ENGINE — WALK-FORWARD, PER BEURS GESEGMENTEERD
# ============================================================

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()

def run_backtest():
    print(f"{'='*60}")
    print(f"PAIRS TRADING BACKTEST (walk-forward)  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")

    exchange_tickers = laad_exchange_tickers()
    all_tickers = sorted(set(t for tlist in exchange_tickers.values() for t in tlist))
    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} over {len(exchange_tickers)} beurzen | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    pivot = pivot.dropna(axis=1, how='all')
    print(f"Data: {len(pivot.columns)} tickers | {len(pivot)} dagen")

    cash = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    equity_curve: List[Dict] = []
    pair_cache: Dict[str, List[PairSignaal]] = {}
    laatste_selectie_idx = -(RESCAN_DAYS + 1)

    dates = sorted(pivot.index)
    start_idx = max(LOOKBACK, IS_WINDOW)
    print(f"Backtest van {dates[start_idx].date()} tot {dates[-1].date()}")

    for i, date in enumerate(dates):
        if i < start_idx:
            continue

        data_till = pivot.iloc[:i + 1]
        current_prices = pivot.iloc[i]

        # --- Exits (goedkope z-score-check, geen coint-hertest) ---
        for pair_key, pos in list(positions.items()):
            a, b = pair_key.split('/')
            if a not in current_prices or b not in current_prices:
                continue
            z_info = snel_zscore(a, b, data_till)
            if z_info is None:
                continue
            z = z_info[0]

            exit_signal = abs(z) < EXIT_Z or pos['days'] >= 60
            if exit_signal:
                if pos['direction'] == 'LONG':
                    pnl_a = (current_prices[a] - pos['entry_a']) * pos['size_a']
                    pnl_b = (pos['entry_b'] - current_prices[b]) * pos['size_b']
                else:
                    pnl_a = (pos['entry_a'] - current_prices[a]) * pos['size_a']
                    pnl_b = (current_prices[b] - pos['entry_b']) * pos['size_b']

                gross_pnl = pnl_a + pnl_b
                cost = trade_cost(abs(gross_pnl)) if gross_pnl > 0 else 0
                tax = gross_pnl * TAX_RATE if gross_pnl > 0 else 0
                net_pnl = gross_pnl - cost - tax

                cash += pos['reserved'] + net_pnl  # marge vrijgeven + resultaat verrekenen

                trades.append({
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'pair': pair_key,
                    'direction': pos['direction'],
                    'entry_z': pos['entry_z'],
                    'exit_z': z,
                    'gross_pnl': round(gross_pnl, 2),
                    'net_pnl': round(net_pnl, 2),
                    'days': pos['days'],
                })
                del positions[pair_key]
            else:
                positions[pair_key]['days'] += 1

        # --- Walk-forward herselectie: enkel met data tot nu (in-sample venster) ---
        if i - laatste_selectie_idx >= RESCAN_DAYS:
            is_data = data_till.tail(IS_WINDOW)
            totaal_paren = 0
            for ex_name, tlist in exchange_tickers.items():
                geselecteerd = selecteer_gecointegreerde_paren(tlist, is_data)
                pair_cache[ex_name] = geselecteerd[:MAX_PAIRS]
                totaal_paren += len(pair_cache[ex_name])
            laatste_selectie_idx = i
            print(f"  [{date.date()}] walk-forward herselectie: {totaal_paren} paren over alle beurzen")

        # --- Nieuwe posities openen (dagelijks, enkel op de cache — geen nieuwe coint-tests) ---
        if len(positions) < MAX_TOTAL_PAIRS:
            for ex_name, kandidaten in pair_cache.items():
                for kand in kandidaten:
                    if len(positions) >= MAX_TOTAL_PAIRS:
                        break
                    a, b = kand.ticker_a, kand.ticker_b
                    pair_key = f"{a}/{b}"
                    if pair_key in positions:
                        continue
                    if a not in current_prices or b not in current_prices:
                        continue
                    if math.isnan(current_prices[a]) or math.isnan(current_prices[b]):
                        continue

                    z_info = snel_zscore(a, b, data_till)
                    if z_info is None:
                        continue
                    z = z_info[0]
                    if abs(z) <= ENTRY_Z:
                        continue
                    richting = "SHORT" if z > ENTRY_Z else "LONG"

                    investering_per_pair = cash * ALLOC_PCT
                    if investering_per_pair <= 0 or investering_per_pair > cash:
                        continue
                    size_a = int(investering_per_pair / current_prices[a] * 0.5)
                    size_b = int(investering_per_pair / current_prices[b] * 0.5)
                    if size_a <= 0 or size_b <= 0:
                        continue

                    cash -= investering_per_pair  # marge reserveren — geen over-allocatie meer mogelijk
                    positions[pair_key] = {
                        'entry_date': date,
                        'entry_a': current_prices[a],
                        'entry_b': current_prices[b],
                        'size_a': size_a,
                        'size_b': size_b,
                        'direction': richting,
                        'entry_z': z,
                        'days': 0,
                        'reserved': investering_per_pair,
                    }

        # --- Equity curve: vrij cash + gereserveerde marge + unrealized pnl ---
        portfolio_value = cash
        for pair_key, pos in positions.items():
            a, b = pair_key.split('/')
            if a in current_prices and b in current_prices:
                if pos['direction'] == 'LONG':
                    value_a = (current_prices[a] - pos['entry_a']) * pos['size_a']
                    value_b = (pos['entry_b'] - current_prices[b]) * pos['size_b']
                else:
                    value_a = (pos['entry_a'] - current_prices[a]) * pos['size_a']
                    value_b = (current_prices[b] - pos['entry_b']) * pos['size_b']
                portfolio_value += pos['reserved'] + value_a + value_b

        equity_curve.append({'date': date, 'value': portfolio_value})

    # --- Resultaten ---
    if trades:
        tdf = pd.DataFrame(trades)
        final_value = equity_curve[-1]['value'] if equity_curve else cash
        total_return = (final_value - START_CAPITAL) / START_CAPITAL * 100
        n_win = (tdf['net_pnl'] > 0).sum()
        win_rate = n_win / len(tdf) * 100

        gross_profit = tdf[tdf['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(tdf[tdf['net_pnl'] <= 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['value']) / equity_df['peak']
        max_dd = equity_df['drawdown'].max() * 100

        os.makedirs('backtest_results', exist_ok=True)
        tdf.to_csv('backtest_results/backtest_cointegr.csv', index=False)

        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTATEN — bot_01cointegr (walk-forward)")
        print(f"{'='*60}")
        print(f"Startkapitaal     : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal      : EUR{final_value:>12,.2f}")
        print(f"Totaal rendement  : {total_return:>+11.1f}%")
        print(f"Trades            : {len(tdf):>12}")
        print(f"Winnaars          : {n_win:>12} ({win_rate:.1f}%)")
        print(f"Profit Factor     : {profit_factor:>12.2f}")
        print(f"Max Drawdown      : {max_dd:>11.1f}%")
        print(f"Gem. houdduur     : {tdf['days'].mean():>11.1f} dagen")
        print(f"{'='*60}")

        top_pairs = tdf.groupby('pair')['net_pnl'].sum().sort_values(ascending=False).head(5)
        print("\n🏆 Beste paren:")
        for pair, pnl in top_pairs.items():
            print(f"  {pair}: EUR{pnl:,.2f}")

        print("\n💾 Resultaten opgeslagen in backtest_results/backtest_cointegr.csv")
    else:
        print("Geen trades gegenereerd.")

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
