#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00ms.py  —  MINERVINI SEPA ENGINE v1.0
Gebaseerd op Mark Minervini's SEPA methodologie (Specific Entry Point Analysis).
Zelfde structuur, tickerbestanden en Telegram output als bot_00xxxV2.py.

Criteria — 8-punt Trend Template + VCP:
  TREND TEMPLATE (Stage 2 uptrend — alle 8 verplicht):
    1. Close > MA150 en Close > MA200
    2. MA150 > MA200
    3. MA200 stijgt (trendend omhoog)
    4. MA50 > MA150 en MA50 > MA200
    5. Close > MA50
    6. Close binnen 25% van 52-weekse high
    7. Close minstens 30% boven 52-weekse low
    8. RS Rating ≥ 70 (relatieve sterkte vs. universe)

  VCP SCORE (0-4 punten — entry timing):
    1. Volatiliteit contractie — ATR daalt over 3 periodes
    2. Volume droogvalt — volume daalt naar lows van consolidatie
    3. Pullback verkleint — elke correctie kleiner dan vorige
    4. Pivot breakout — prijs doorbreekt recente weerstand op volume

Gebruik:
  python bot_00ms.py          # live rapport
  python bot_00ms.py backtest # backtest modus

GitHub Actions: dagelijks om 21:50 UTC (na bot_00xxxV2 en bot_00kr)
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL        = 50_000.0
MAX_POSITIONS        = 10
MIN_CASH_RATIO       = 0.10
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001

TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
MAX_HOLD_DAYS        = 60

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGES = {
    "041 Benelux":     "tickers_041x.txt",
    "042 Parijs":      "tickers_042x.txt",
    "043 Frankfurt":   "tickers_043x.txt",
    "044 Spanje/Port": "tickers_044x.txt",
    "045 Londen":      "tickers_045x.txt",
    "046 Milaan":      "tickers_046x.txt",
    "047 Toronto":     "tickers_047x.txt",
    "048 Nasdaq/NYSE": "tickers_048x.txt",
}

# Minervini SEPA parameters
MS_CFG = {
    # Trend Template MAs
    "ma_fast":          50,
    "ma_mid":           150,
    "ma_slow":          200,
    # Stage 2 criteria
    "max_from_high_pct":   25.0,   # max 25% onder 52-weekse high
    "min_from_low_pct":    30.0,   # min 30% boven 52-weekse low
    "rs_min":              70,     # min RS rating (0-99)
    # VCP parameters
    "atr_period":          14,
    "vcp_lookback":        60,     # dagen voor VCP detectie
    "vol_contraction_pct": 20.0,   # ATR moet X% gedaald zijn
    "volume_dry_pct":      30.0,   # volume moet X% onder gem. zijn
    "pivot_lookback":      20,     # dagen voor pivot weerstand
    "pivot_breakout_vol":  1.5,    # volume moet X× gem. zijn bij breakout
    # Stop en TP
    "stop_pct":            8.0,    # Minervini: max 7-8% stop
    "min_score":           2,      # min VCP score om te rapporteren (0-4)
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES  (identiek aan bot_00xxxV2.py)
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

def format_price(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    return f"{val:.2f}"

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

def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# ATR SIZING  (identiek aan bot_00xxxV2.py — 5% risico)
# ============================================================

def bereken_positie(
    portfolio_waarde: float,
    entry_prijs:      float,
    stop_prijs:       float,
    risico_pct:       float = RISICO_PCT_PER_TRADE,
) -> Tuple[int, float]:
    risico_eur   = portfolio_waarde * risico_pct
    stop_afstand = entry_prijs - stop_prijs
    if stop_afstand <= 0:
        return 0, 0.0
    aandelen    = max(1, int(risico_eur / stop_afstand))
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, max_verlies

def sizing_tekst(ticker, prijs, stop, resistance, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    stop_slip   = stop
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop_slip)
    tp          = resistance
    investering = round(entry * aandelen, 2)
    slip_est    = round(entry * SLIPPAGE_PCT * aandelen * 2, 2)
    kosten      = round(trade_cost(investering), 2)
    rr          = ((tp - entry) / (entry - stop_slip)) if (entry - stop_slip) > 0 else 0
    return (
        f"  📐 *Sizing:*\n"
        f"  Entry geschat : EUR{entry:.2f}\n"
        f"  Stop-Loss     : EUR{stop_slip:.2f}  ({MS_CFG['stop_pct']:.0f}% max)\n"
        f"  Take-Profit   : EUR{tp:.2f}  (pivot weerstand)\n"
        f"  R/R ratio     : {rr:.1f}:1\n"
        f"  Aandelen      : {aandelen} stuks\n"
        f"  Investering   : EUR{investering:,.2f}\n"
        f"  Max verlies   : EUR{max_loss:,.2f}  (5% portfolio)\n"
        f"  Slippage est. : EUR{slip_est:.2f}\n"
        f"  Kosten        : EUR{kosten:.2f}"
    )


# ============================================================
# DATA DOWNLOAD  (identiek aan bot_00xxxV2.py)
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
# INDICATOREN
# ============================================================

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period
            + valid.iloc[i] / period
        )
    return result


def add_indicators(df: pd.DataFrame, universe_rs: Dict[str, float]) -> pd.DataFrame:
    """For-loop per ticker — voegt MA50/150/200, ATR, Volume MA, RS toe."""
    parts = []
    for ticker, group in df.groupby("Ticker", sort=False):
        g     = group.copy()
        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]
        vol   = g["Volume"]

        g["MA50"]  = close.rolling(MS_CFG["ma_fast"]).mean()
        g["MA150"] = close.rolling(MS_CFG["ma_mid"]).mean()
        g["MA200"] = close.rolling(MS_CFG["ma_slow"]).mean()

        # MA200 richting: stijgt over laatste 20 handelsdagen
        g["MA200_slope"] = g["MA200"].diff(20)

        # ATR14
        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"] = _wilder_smooth(tr, 14)

        # Volume MA20
        g["VolMA20"] = vol.rolling(20).mean()

        # 52-weekse high/low
        g["High52w"] = close.rolling(252).max()
        g["Low52w"]  = close.rolling(252).min()

        # RS Rating (relatieve sterkte vs. universe — 0-99)
        g["RS"] = universe_rs.get(ticker, 50.0)

        g["Ticker"] = ticker
        parts.append(g)

    if not parts:
        return df
    return pd.concat(parts).sort_values(["Ticker", "Date"]).reset_index(drop=True)


def compute_rs_ratings(df: pd.DataFrame) -> Dict[str, float]:
    """
    Berekent RS Rating per ticker: 12-maands prijsprestatie
    gerangschikt als percentiel binnen het universe (0-99).
    Gewogen: laatste kwartaal telt dubbel (Minervini stijl).
    """
    rs_map: Dict[str, float] = {}
    perf: Dict[str, float] = {}

    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date")
        if len(g) < 252:
            continue
        c_now  = safe_float(g["Close"].iloc[-1])
        c_3m   = safe_float(g["Close"].iloc[-63])   # ~3 maanden
        c_12m  = safe_float(g["Close"].iloc[-252])  # ~12 maanden
        if math.isnan(c_now) or math.isnan(c_12m) or c_12m <= 0 or c_3m <= 0:
            continue
        # Gewogen: 40% laatste kwartaal, 60% rest van het jaar
        perf_3m  = (c_now - c_3m)  / c_3m
        perf_12m = (c_now - c_12m) / c_12m
        perf[ticker] = 0.4 * perf_3m + 0.6 * perf_12m

    if not perf:
        return rs_map

    values = sorted(perf.values())
    n = len(values)
    for ticker, p in perf.items():
        rank = sum(1 for v in values if v < p)
        rs_map[ticker] = round((rank / n) * 99, 1)

    return rs_map


# ============================================================
# TREND TEMPLATE  (8 criteria — allemaal verplicht)
# ============================================================

def check_trend_template(row: pd.Series) -> Tuple[bool, List[str]]:
    """
    Controleert alle 8 Minervini Trend Template criteria.
    Geeft (passed, detail_list) terug.
    """
    close  = safe_float(row.get("Close"))
    ma50   = safe_float(row.get("MA50"))
    ma150  = safe_float(row.get("MA150"))
    ma200  = safe_float(row.get("MA200"))
    ma200s = safe_float(row.get("MA200_slope"))
    h52    = safe_float(row.get("High52w"))
    l52    = safe_float(row.get("Low52w"))
    rs     = safe_float(row.get("RS"), 0.0)

    checks = []
    passed = True

    def chk(condition: bool, ok_msg: str, fail_msg: str):
        nonlocal passed
        if condition:
            checks.append(f"✓ {ok_msg}")
        else:
            checks.append(f"✗ {fail_msg}")
            passed = False

    # 1. Close > MA150 en MA200
    chk(not math.isnan(close) and not math.isnan(ma150) and not math.isnan(ma200)
        and close > ma150 and close > ma200,
        f"Close>{MS_CFG['ma_mid']}/{MS_CFG['ma_slow']}MA",
        f"Close niet boven MA150/MA200")

    # 2. MA150 > MA200
    chk(not math.isnan(ma150) and not math.isnan(ma200) and ma150 > ma200,
        "MA150>MA200",
        "MA150 niet boven MA200")

    # 3. MA200 stijgt
    chk(not math.isnan(ma200s) and ma200s > 0,
        "MA200 stijgt",
        "MA200 daalt/vlak")

    # 4. MA50 > MA150 en MA200
    chk(not math.isnan(ma50) and not math.isnan(ma150) and not math.isnan(ma200)
        and ma50 > ma150 and ma50 > ma200,
        "MA50>MA150/MA200",
        "MA50 niet boven MA150/MA200")

    # 5. Close > MA50
    chk(not math.isnan(close) and not math.isnan(ma50) and close > ma50,
        "Close>MA50",
        "Close niet boven MA50")

    # 6. Binnen 25% van 52-weekse high
    if not math.isnan(h52) and h52 > 0:
        pct_from_high = (h52 - close) / h52 * 100
        chk(pct_from_high <= MS_CFG["max_from_high_pct"],
            f"{pct_from_high:.1f}% onder 52w high (≤25%)",
            f"{pct_from_high:.1f}% onder 52w high (>25%)")
    else:
        checks.append("✗ geen 52w high data")
        passed = False

    # 7. Minstens 30% boven 52-weekse low
    if not math.isnan(l52) and l52 > 0:
        pct_from_low = (close - l52) / l52 * 100
        chk(pct_from_low >= MS_CFG["min_from_low_pct"],
            f"{pct_from_low:.1f}% boven 52w low (≥30%)",
            f"{pct_from_low:.1f}% boven 52w low (<30%)")
    else:
        checks.append("✗ geen 52w low data")
        passed = False

    # 8. RS Rating ≥ 70
    chk(rs >= MS_CFG["rs_min"],
        f"RS={rs:.0f} (≥70)",
        f"RS={rs:.0f} (<70)")

    return passed, checks


# ============================================================
# VCP DETECTIE  (score 0-4)
# ============================================================

def detect_vcp(g: pd.DataFrame) -> Tuple[int, List[str], float, float]:
    """
    Detecteert Volatility Contraction Pattern.
    Geeft (score, detail_list, pivot_prijs, stop_prijs) terug.
    """
    score   = 0
    details = []
    close   = g["Close"]
    volume  = g["Volume"]
    atr     = g["ATR14"]
    vol_ma  = g["VolMA20"]

    if len(g) < MS_CFG["vcp_lookback"] + 10:
        return 0, ["onvoldoende data"], float("nan"), float("nan")

    recent     = g.iloc[-MS_CFG["vcp_lookback"]:]
    very_recent = g.iloc[-MS_CFG["pivot_lookback"]:]

    # ── 1. Volatiliteit contractie ──────────────────────────────
    # ATR nu vs. ATR begin van VCP window
    atr_now   = safe_float(atr.iloc[-1])
    atr_start = safe_float(atr.iloc[-MS_CFG["vcp_lookback"]])
    if not math.isnan(atr_now) and not math.isnan(atr_start) and atr_start > 0:
        contraction = (atr_start - atr_now) / atr_start * 100
        if contraction >= MS_CFG["vol_contraction_pct"]:
            score += 1
            details.append(f"✓ ATR contractie {contraction:.1f}% (min {MS_CFG['vol_contraction_pct']}%)")
        else:
            details.append(f"✗ ATR contractie {contraction:.1f}% (min {MS_CFG['vol_contraction_pct']}%)")
    else:
        details.append("✗ ATR data onvoldoende")

    # ── 2. Volume droogvalt bij lows ────────────────────────────
    vol_now  = safe_float(volume.iloc[-1])
    vol_mean = safe_float(vol_ma.iloc[-1])
    if not math.isnan(vol_now) and not math.isnan(vol_mean) and vol_mean > 0:
        vol_ratio = vol_now / vol_mean * 100
        if vol_ratio <= (100 - MS_CFG["volume_dry_pct"]):
            score += 1
            details.append(f"✓ Volume droogvalt {vol_ratio:.0f}% van gem.")
        else:
            details.append(f"✗ Volume {vol_ratio:.0f}% van gem. (max {100-MS_CFG['volume_dry_pct']:.0f}%)")
    else:
        details.append("✗ Volume data onvoldoende")

    # ── 3. Pullbacks worden kleiner ─────────────────────────────
    # Splits VCP window in 3 gelijke delen, vergelijk range
    n = len(recent)
    third = n // 3
    if third >= 5:
        r1 = recent.iloc[:third]["Close"]
        r2 = recent.iloc[third:2*third]["Close"]
        r3 = recent.iloc[2*third:]["Close"]
        range1 = float(r1.max() - r1.min())
        range2 = float(r2.max() - r2.min())
        range3 = float(r3.max() - r3.min())
        if range1 > 0 and range2 < range1 and range3 < range2:
            score += 1
            details.append(
                f"✓ Pullbacks krimpen: {range1:.2f}→{range2:.2f}→{range3:.2f}"
            )
        else:
            details.append(
                f"✗ Pullbacks krimpen niet: {range1:.2f}→{range2:.2f}→{range3:.2f}"
            )
    else:
        details.append("✗ Onvoldoende data voor pullback analyse")

    # ── 4. Pivot breakout ───────────────────────────────────────
    # Prijs boven recent hoogste punt (pivot) op verhoogd volume?
    pivot_high   = float(very_recent["Close"].iloc[:-1].max())  # hoogste excl. vandaag
    current      = safe_float(close.iloc[-1])
    vol_recent_mean = safe_float(vol_ma.iloc[-MS_CFG["pivot_lookback"]])
    vol_today    = safe_float(volume.iloc[-1])

    breakout = (
        not math.isnan(current) and current > pivot_high
        and not math.isnan(vol_today) and not math.isnan(vol_recent_mean)
        and vol_recent_mean > 0
        and vol_today >= vol_recent_mean * MS_CFG["pivot_breakout_vol"]
    )
    if breakout:
        score += 1
        details.append(f"✓ Pivot breakout boven {pivot_high:.2f} op {vol_today/vol_recent_mean:.1f}× volume")
    else:
        details.append(f"✗ Geen pivot breakout (pivot={pivot_high:.2f})")

    # ── Stop en pivot prijs ─────────────────────────────────────
    # Stop: laagste close in recente consolidatie
    stop_prijs  = float(very_recent["Close"].min())
    # Extra marge: max 8% stop (Minervini regel)
    stop_max    = current * (1 - MS_CFG["stop_pct"] / 100)
    stop_prijs  = max(stop_prijs, stop_max)

    # Pivot = weerstand = hoogste punt van VCP window
    pivot_prijs = float(recent["Close"].max())

    return score, details, pivot_prijs, stop_prijs


# ============================================================
# SEPA SIGNAAL
# ============================================================

@dataclass
class SEPASignaal:
    ticker:          str
    price:           float
    trend_passed:    bool
    trend_details:   List[str]
    vcp_score:       int          # 0-4
    vcp_details:     List[str]
    pivot:           float        # weerstand / TP
    stop:            float
    rs:              float
    pct_from_high:   float
    pct_from_low:    float
    atr:             float
    total_score:     float        # gewogen totaal voor ranking


def analyse_ticker(ticker: str, g: pd.DataFrame) -> Optional[SEPASignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < MS_CFG["ma_slow"] + 10:
            return None

        last = g.iloc[-1]
        current_price = safe_float(last.get("Close"))
        if current_price <= 0 or math.isnan(current_price):
            return None

        # ── Trend Template ──────────────────────────────────────
        trend_passed, trend_details = check_trend_template(last)

        # Alleen rapporteren als Trend Template slaagt
        if not trend_passed:
            return None

        # ── VCP ─────────────────────────────────────────────────
        vcp_score, vcp_details, pivot, stop = detect_vcp(g)

        if vcp_score < MS_CFG["min_score"]:
            return None

        # ── Extra metrics ───────────────────────────────────────
        h52 = safe_float(last.get("High52w"))
        l52 = safe_float(last.get("Low52w"))
        rs  = safe_float(last.get("RS"), 0.0)
        atr = safe_float(last.get("ATR14"), current_price * 0.02)

        pct_from_high = ((h52 - current_price) / h52 * 100) if h52 > 0 else 0.0
        pct_from_low  = ((current_price - l52) / l52 * 100) if l52 > 0 else 0.0

        # Totaal score voor ranking: RS + VCP score gewogen
        total_score = rs * 0.4 + vcp_score * 15 + (25 - pct_from_high) * 0.5

        return SEPASignaal(
            ticker=ticker,
            price=round(current_price, 2),
            trend_passed=trend_passed,
            trend_details=trend_details,
            vcp_score=vcp_score,
            vcp_details=vcp_details,
            pivot=round(pivot, 2),
            stop=round(stop, 2),
            rs=round(rs, 1),
            pct_from_high=round(pct_from_high, 1),
            pct_from_low=round(pct_from_low, 1),
            atr=round(atr, 4),
            total_score=round(total_score, 1),
        )

    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM OUTPUT  (zelfde stijl als bot_00xxxV2.py)
# ============================================================

def _vcp_bar(score: int) -> str:
    filled = "█" * score
    empty  = "░" * (4 - score)
    return f"{filled}{empty} {score}/4"


def format_ms_per_exchange(
    exchange_name:   str,
    signalen:        List[SEPASignaal],
    portfolio_waarde: float,
) -> Tuple[str, str]:
    nu = today_str()

    def detail_blok(sigs: List[SEPASignaal]) -> str:
        if not sigs:
            return "_Geen kandidaten_"
        lines = []
        for s in sigs:
            rr = ((s.pivot - s.price) / (s.price - s.stop)) if (s.price - s.stop) > 0 else 0
            lines.append(
                f"• `{s.ticker}` | VCP: {_vcp_bar(s.vcp_score)} | RS:{s.rs:.0f} | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                f"  {s.pct_from_high:.1f}% onder 52w high | {s.pct_from_low:.1f}% boven 52w low\n"
                + "\n".join(f"  {d}" for d in s.vcp_details) + "\n"
                + sizing_tekst(s.ticker, s.price, s.stop, s.pivot, portfolio_waarde)
            )
        return "\n\n".join(lines)

    def trend_blok(sigs: List[SEPASignaal]) -> str:
        """Compact trend template detail per ticker."""
        if not sigs:
            return "_Geen_"
        lines = []
        for s in sigs[:5]:  # max 5 in deel2
            lines.append(
                f"• `{s.ticker}` RS:{s.rs:.0f}\n"
                + "\n".join(f"  {d}" for d in s.trend_details)
            )
        return "\n\n".join(lines)

    # Top 2 over alle signalen (voor ranking)
    top2     = signalen[:2]
    vcp4     = [s for s in signalen if s.vcp_score == 4]
    vcp3     = [s for s in signalen if s.vcp_score == 3]
    vcp2     = [s for s in signalen if s.vcp_score == 2]

    deel1 = "\n\n".join([
        f"📈 *MINERVINI SEPA — {exchange_name}*",
        f"_{nu} | Stage 2 + VCP filter | Min VCP score: {MS_CFG['min_score']}/4_",
        f"_Trend Template (8/8 verplicht) + Volatility Contraction Pattern_",
        "─────────────────────────────",
        f"🏆 *TOP 2 HOOGSTE POTENTIEEL:*",
        detail_blok(top2) if top2 else "_Geen kandidaten vandaag_",
        "─────────────────────────────",
        f"🔥 *PERFECTE VCP (4/4):*",
        detail_blok(vcp4) if vcp4 else "_Geen_",
        f"⚡ *STERKE VCP (3/4):*",
        detail_blok(vcp3) if vcp3 else "_Geen_",
    ])

    deel2_parts = [
        f"📈 *MINERVINI SEPA — {exchange_name} (2/2)*",
        "",
        f"📊 *WATCHLIST VCP (2/4):*",
        detail_blok(vcp2) if vcp2 else "_Geen_",
        "",
        "─────────────────────────────",
        f"🔍 *TREND TEMPLATE DETAIL (top 5):*",
        trend_blok(signalen),
        "",
        "─────────────────────────────",
        f"📊 *SAMENVATTING:*",
        f"  Kandidaten (Stage 2 + VCP≥{MS_CFG['min_score']}) : {len(signalen)}",
        f"  VCP 4/4 : {len(vcp4)}",
        f"  VCP 3/4 : {len(vcp3)}",
        f"  VCP 2/4 : {len(vcp2)}",
        "",
        "⚙️ *PARAMETERS:*",
        f"_Trend Template: MA50>MA150>MA200 stijgend_",
        f"_Max 25% onder 52w high | Min 30% boven 52w low_",
        f"_RS Rating ≥ 70 | Stop max {MS_CFG['stop_pct']:.0f}%_",
        f"_VCP: ATR contractie ≥{MS_CFG['vol_contraction_pct']:.0f}% | Volume droogvalt_",
        f"_Risico: 5% portfolio per trade | Slippage: 0.1%_",
    ]

    return deel1, "\n".join(deel2_parts)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"MINERVINI SEPA — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        if tlist:
            exchange_tickers[ex_name] = tlist
            all_tickers.extend(tlist)
            print(f"  {ex_name}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print("Data downloaden (2 jaar)...")

    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers")
    print("RS ratings berekenen...")
    rs_ratings = compute_rs_ratings(df)
    print(f"RS ratings: {len(rs_ratings)} tickers")

    print("Indicatoren berekenen...")
    df = add_indicators(df, rs_ratings)

    portfolio_waarde = START_CAPITAL

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        signalen: List[SEPASignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group)
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: VCP={sig.vcp_score}/4 | RS={sig.rs:.0f} | "
                    f"{sig.pct_from_high:.1f}% onder high"
                )

        # Sorteren: totaal_score hoog→laag
        signalen.sort(key=lambda s: s.total_score, reverse=True)
        print(f"  → {len(signalen)} SEPA kandidaten")

        deel1, deel2 = format_ms_per_exchange(ex_name, signalen, portfolio_waarde)
        send_telegram_message(deel1)
        time.sleep(1)
        send_telegram_message(deel2)

        if signalen:
            _log_csv(signalen, ex_name)

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# CSV LOGGING
# ============================================================

def _log_csv(signalen: List[SEPASignaal], exchange: str):
    fname  = f"ms_signalen_{exchange.split()[0]}_{today_str()}.csv"
    header = ["datum","exchange","ticker","vcp_score","rs","price",
              "pivot","stop","pct_from_high","pct_from_low","total_score"]
    ensure_csv_header(fname, header)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in signalen:
            w.writerow([
                today_str(), exchange, s.ticker, s.vcp_score, s.rs,
                s.price, s.pivot, s.stop,
                s.pct_from_high, s.pct_from_low, s.total_score,
            ])
    print(f"  CSV: {fname}")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"MINERVINI SEPA BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for path in EXCHANGES.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    all_dates = sorted(df["Date"].dt.date.unique())
    print(f"Handelsdagen: {len(all_dates)}")

    # RS berekenen op volledige dataset
    rs_ratings = compute_rs_ratings(df)
    df = add_indicators(df, rs_ratings)

    cash      = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict] = []

    # Wekelijks scannen (elke maandag)
    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig:
                continue

            entry      = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, max_loss = bereken_positie(cash, entry, sig.stop)
            if aandelen <= 0:
                continue
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue

            cash -= investering
            positions[ticker] = {
                "entry_date":  scan_date,
                "entry_price": round(entry, 4),
                "size":        aandelen,
                "stop":        sig.stop,
                "tp":          sig.pivot,
                "vcp_score":   sig.vcp_score,
                "rs":          sig.rs,
                "days":        0,
                "cost":        trade_cost(investering),
            }

        # Dagelijkse exit
        day_df = df[df["Date"] == pd.Timestamp(scan_date)].copy()
        price_map: Dict[str, float] = {}
        for _, row in day_df.iterrows():
            t = row.get("Ticker")
            c = safe_float(row.get("Close"))
            if t and not math.isnan(c):
                price_map[t] = c

        for ticker, pos in list(positions.items()):
            pos["days"] += 1
            if ticker not in price_map:
                continue
            close  = price_map[ticker]
            reason = None

            if close <= pos["stop"]:
                reason = f"SL ({pos['stop']:.2f})"
            elif close >= pos["tp"]:
                reason = f"TP ({pos['tp']:.2f})"
            elif pos["days"] >= MAX_HOLD_DAYS:
                reason = f"Time ({pos['days']}d)"

            if reason:
                exit_slip = close * (1 - SLIPPAGE_PCT)
                gross     = exit_slip * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax       = pnl * TAX_RATE if pnl > 0 else 0.0
                cash     += gross - cost - tax
                trades.append({
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   scan_date.isoformat(),
                    "ticker":      ticker,
                    "vcp_score":   pos["vcp_score"],
                    "rs":          pos["rs"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  round(exit_slip, 4),
                    "size":        pos["size"],
                    "pnl":         round(pnl, 2),
                    "tax":         round(tax, 2),
                    "net":         round(pnl - tax, 2),
                    "reason":      reason,
                    "days":        pos["days"],
                })
                del positions[ticker]

    # Resultaten
    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("ms_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final_val = cash + sum(
            price_map.get(t, p["entry_price"]) * p["size"]
            for t, p in positions.items()
        )
        print(f"\n{'='*60}")
        print(f"Startkapitaal    : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal     : EUR{final_val:>12,.2f}")
        print(f"Totaal rendement : {(final_val-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
        print(f"\n{'VCP score':<12} {'#':>4} {'Win%':>6} {'Net':>10}")
        for sc, g in tdf.groupby("vcp_score"):
            wr = (g["net"] > 0).sum() / len(g) * 100
            print(f"VCP {sc}/4      {len(g):>4} {wr:>5.1f}% {g['net'].sum():>+10.2f}")
        print(f"{'='*60}")
        print(f"Opgeslagen: ms_backtest_trades.csv")
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
