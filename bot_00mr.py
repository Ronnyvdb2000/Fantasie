#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00mr.py  —  MULTI-SYSTEEM ENGINE v1.0
Combineert twee strategieën in één bot:

  SYSTEEM 1 — IBS + RSI Mean Reversion (EOD)
    Koop paniek:
      - IBS < 0.2
      - RSI(3) < 10
      - Close > MA200 (geen falling knife)
    Exit:
      - Slot boven MA20, OF
      - IBS > 0.7

  SYSTEEM 2 — Opening Range Breakout (ORB, intraday)
    - Eerste 15 minuten = opening range
    - Long als prijs boven high van range breekt op volume
    - Stop onder low van range
    - TP = 2× risk

Risk & portfolio:
  - Max 10 posities totaal
  - 40% kapitaal voor mean reversion, 30% voor ORB, 30% reserve
  - 3-4% equity per trade
  - Max dagverlies 3% → bot stopt

Gebruik:
  python bot_00mr.py eod      # EOD run: IBS+RSI signalen
  python bot_00mr.py orb      # Intraday run: ORB detectie
  python bot_00mr.py status   # Portfolio status + performance
  python bot_00mr.py backtest # Backtest beide systemen

GitHub Actions:
  EOD:      cron "30 21 * * 1-5"
  Intraday: cron "30 09 * * 1-5"  (na opening Europese markten)
"""

import os
import sys
import math
import csv
import json
import warnings
import datetime as dt
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

PORTFOLIO_FILE   = "mr_portfolio.json"
TRADES_FILE      = "mr_trades.csv"
SNAPSHOT_FILE    = "mr_snapshots.csv"
PERF_FILE        = "mr_performance.json"

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

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

CFG = {
    # Portfolio
    "start_capital":        50_000.0,
    "max_positions":        10,
    "min_cash_ratio":       0.10,

    # Risk budgetten (som = 100%)
    "budget_mr":            0.40,   # 40% voor mean reversion
    "budget_orb":           0.30,   # 30% voor ORB
    "budget_reserve":       0.30,   # 30% cash reserve

    # Per trade risico
    "risico_pct":           0.04,   # 4% equity per trade

    # Max dagverlies
    "max_daily_loss_pct":   0.03,   # 3% → bot stopt

    # Kosten
    "slippage":             0.001,
    "cost_fixed":           15.0,
    "cost_pct":             0.0035,
    "tax_rate":             0.10,

    # ── SYSTEEM 1: IBS + RSI Mean Reversion ──────────────────
    "ibs_entry":            0.20,   # IBS < 0.2 bij entry
    "ibs_exit":             0.70,   # IBS > 0.7 bij exit
    "rsi_period":           3,      # RSI periode
    "rsi_entry":            10.0,   # RSI < 10 bij entry
    "ma_trend":             200,    # MA voor trend filter
    "ma_exit":              20,     # MA20 voor exit
    "atr_period":           14,
    "mr_max_hold":          10,     # max houdduur mean reversion

    # ── SYSTEEM 2: ORB ────────────────────────────────────────
    "orb_minutes":          15,     # opening range = eerste 15 min
    "orb_vol_mult":         1.5,    # volume filter breakout
    "orb_tp_mult":          2.0,    # TP = 2× risk
    "orb_max_hold":         1,      # intraday: sluit zelfde dag
}


# ============================================================
# HULPFUNCTIES
# ============================================================

def trade_cost(amount: float) -> float:
    return CFG["cost_fixed"] + amount * CFG["cost_pct"]

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

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

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Splits lange berichten
    for i in range(0, len(text), 4000):
        try:
            requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i+4000],
                      "parse_mode": "Markdown"},
                timeout=10,
            )
            time.sleep(0.5)
        except Exception as e:
            print(f"Telegram fout: {e}")

def ensure_csv(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# PORTFOLIO STATE
# ============================================================

def load_portfolio() -> Dict:
    default = {
        "cash":           CFG["start_capital"],
        "positions":      {},
        "daily_pnl":      0.0,
        "daily_start":    today_str(),
        "total_trades":   0,
    }
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            # Reset dagverlies als nieuwe dag
            if data.get("daily_start") != today_str():
                data["daily_pnl"]   = 0.0
                data["daily_start"] = today_str()
            return data
        except Exception:
            pass
    return default

def save_portfolio(p: Dict) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

def portfolio_waarde(p: Dict, prices: Dict[str, float]) -> float:
    pos_val = sum(
        prices.get(t, pos["entry_price"]) * pos["size"]
        for t, pos in p["positions"].items()
    )
    return p["cash"] + pos_val

def risk_budget(p: Dict, systeem: str, prices: Dict[str, float]) -> float:
    """Beschikbaar kapitaal voor een systeem op basis van risk budget."""
    totaal = portfolio_waarde(p, prices)
    pct    = CFG[f"budget_{systeem}"]
    return totaal * pct

def max_daily_loss_bereikt(p: Dict, prices: Dict[str, float]) -> bool:
    totaal = portfolio_waarde(p, prices)
    return p["daily_pnl"] <= -(totaal * CFG["max_daily_loss_pct"])

def log_trade(datum, ticker, systeem, side, price, size, cost, pnl, tax, net, reason=""):
    ensure_csv(TRADES_FILE, ["datum","ticker","systeem","side","price","size",
                              "cost","pnl","tax","net","reason"])
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datum, ticker, systeem, side, price, size,
                                  cost, pnl, tax, net, reason])

def log_snapshot(p: Dict, prices: Dict[str, float]) -> None:
    ensure_csv(SNAPSHOT_FILE, ["datum","cash","pos_value","total","n_pos",
                                "daily_pnl"])
    totaal  = portfolio_waarde(p, prices)
    pos_val = totaal - p["cash"]
    with open(SNAPSHOT_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([today_str(), round(p["cash"], 2),
                                  round(pos_val, 2), round(totaal, 2),
                                  len(p["positions"]), round(p["daily_pnl"], 2)])


# ============================================================
# SIZING
# ============================================================

def bereken_size(
    portfolio_totaal: float,
    entry:            float,
    stop:             float,
    risico_pct:       float = None,
) -> Tuple[int, float]:
    if risico_pct is None:
        risico_pct = CFG["risico_pct"]
    risico_eur   = portfolio_totaal * risico_pct
    stop_afstand = entry - stop
    if stop_afstand <= 0 or entry <= 0:
        return 0, 0.0
    aandelen    = max(1, int(risico_eur / stop_afstand))
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, max_verlies

def sizing_tekst(entry, stop, tp, aandelen, max_loss, portfolio_totaal) -> str:
    investering = round(entry * aandelen, 2)
    rr          = ((tp - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  📐 Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | TP: EUR{tp:.2f}\n"
        f"  R/R: {rr:.1f}:1 | {aandelen} stuks | EUR{investering:,.2f}\n"
        f"  Max verlies: EUR{max_loss:,.2f} ({CFG['risico_pct']*100:.0f}% equity)"
    )


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

def download_eod(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    frames = []
    try:
        data = yf.download(tickers, auto_adjust=True, group_by="ticker",
                           progress=False, threads=True, period=period)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_level = 1
                for lvl in range(data.columns.nlevels):
                    if any(t in set(data.columns.get_level_values(lvl)) for t in tickers):
                        ticker_level = lvl
                        break
                for t in tickers:
                    try:
                        norm = _normalise(data.xs(t, axis=1, level=ticker_level).copy(), t)
                        if norm is not None:
                            frames.append(norm)
                    except Exception:
                        pass
            else:
                norm = _normalise(data, tickers[0])
                if norm is not None:
                    frames.append(norm)
    except Exception as e:
        print(f"[WARN] Batch download mislukt: {e}")

    if not frames:
        for t in tickers:
            try:
                raw = yf.download(t, period=period, auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    norm = _normalise(raw, t)
                    if norm is not None:
                        frames.append(norm)
                time.sleep(0.2)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def download_intraday(tickers: List[str], interval: str = "15m") -> pd.DataFrame:
    """Download intraday data via yfinance (max 60 dagen terug)."""
    frames = []
    for t in tickers:
        try:
            raw = yf.download(t, period="5d", interval=interval,
                              auto_adjust=True, progress=False)
            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            if "Datetime" in raw.columns:
                raw = raw.rename(columns={"Datetime": "Date"})
            raw["Ticker"] = t
            frames.append(raw)
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] {t} intraday: {e}")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# TECHNISCHE INDICATOREN
# ============================================================

def _wilder(series: pd.Series, period: int) -> pd.Series:
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

def add_eod_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt IBS, RSI(3), MA20, MA200 toe via for-loop per ticker."""
    parts = []
    for ticker, g in df.groupby("Ticker", sort=False):
        g     = g.copy()
        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]

        # IBS = Internal Bar Strength
        g["IBS"]   = (close - low) / (high - low + 1e-9)
        # RSI(3)
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        rs    = _wilder(gain, CFG["rsi_period"]) / (_wilder(loss, CFG["rsi_period"]) + 1e-9)
        g["RSI3"]  = 100 - (100 / (1 + rs))
        # MAs
        g["MA20"]  = close.rolling(CFG["ma_exit"]).mean()
        g["MA200"] = close.rolling(CFG["ma_trend"]).mean()
        # ATR
        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"] = _wilder(tr, CFG["atr_period"])

        g["Ticker"] = ticker
        parts.append(g)

    if not parts:
        return df
    return pd.concat(parts).sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ============================================================
# SYSTEEM 1: IBS + RSI MEAN REVERSION
# ============================================================

@dataclass
class MRSignaal:
    ticker:    str
    price:     float
    ibs:       float
    rsi3:      float
    ma20:      float
    ma200:     float
    atr:       float
    stop:      float
    tp:        float
    rr:        float
    exchange:  str = ""

@dataclass
class MRExitSignaal:
    ticker:    str
    price:     float
    ma20:      float
    ibs:       float
    reden:     str
    pnl_est:   float


def generate_mr_signals(df: pd.DataFrame, exchange: str) -> Tuple[List[MRSignaal], List[MRExitSignaal]]:
    """
    Genereert IBS+RSI entry signalen en exit checks voor bestaande posities.
    """
    entries: List[MRSignaal]     = []
    exits:   List[MRExitSignaal] = []

    # Laatste dag per ticker
    last_df = df.sort_values("Date").groupby("Ticker").last().reset_index()

    for _, row in last_df.iterrows():
        ticker = row.get("Ticker")
        close  = safe_float(row.get("Close"))
        high   = safe_float(row.get("High"))
        low    = safe_float(row.get("Low"))
        ibs    = safe_float(row.get("IBS"))
        rsi3   = safe_float(row.get("RSI3"))
        ma20   = safe_float(row.get("MA20"))
        ma200  = safe_float(row.get("MA200"))
        atr    = safe_float(row.get("ATR14"))

        if any(math.isnan(x) for x in [close, ibs, rsi3, ma20, ma200]):
            continue
        if close <= 0 or atr <= 0 or math.isnan(atr):
            continue

        # ── Entry: paniek koop ───────────────────────────────────
        if (ibs < CFG["ibs_entry"] and
            rsi3 < CFG["rsi_entry"] and
            close > ma200):

            stop = low - 0.5 * atr         # stop onder daglow - 0.5 ATR
            tp   = ma20                    # TP = terug naar MA20
            rr   = ((tp - close) / (close - stop)) if (close - stop) > 0 else 0

            if rr > 0.5:  # minimale R/R
                entries.append(MRSignaal(
                    ticker=ticker, price=round(close, 2),
                    ibs=round(ibs, 3), rsi3=round(rsi3, 1),
                    ma20=round(ma20, 2), ma200=round(ma200, 2),
                    atr=round(atr, 4),
                    stop=round(stop, 2),
                    tp=round(tp, 2),
                    rr=round(rr, 2),
                    exchange=exchange,
                ))

        # ── Exit check ───────────────────────────────────────────
        # (Voor posities die open staan — gebaseerd op dagdata)
        if ibs > CFG["ibs_exit"]:
            exits.append(MRExitSignaal(
                ticker=ticker, price=round(close, 2),
                ma20=round(ma20, 2), ibs=round(ibs, 3),
                reden=f"IBS={ibs:.2f} > {CFG['ibs_exit']}",
                pnl_est=0.0,
            ))
        elif close > ma20:
            exits.append(MRExitSignaal(
                ticker=ticker, price=round(close, 2),
                ma20=round(ma20, 2), ibs=round(ibs, 3),
                reden=f"Slot {close:.2f} > MA20 {ma20:.2f}",
                pnl_est=0.0,
            ))

    entries.sort(key=lambda s: s.ibs)   # laagste IBS = meest oververkocht
    return entries, exits


# ============================================================
# SYSTEEM 2: OPENING RANGE BREAKOUT (ORB)
# ============================================================

@dataclass
class ORBSignaal:
    ticker:       str
    orb_high:     float   # high van opening range
    orb_low:      float   # low van opening range
    orb_range:    float   # range breedte
    breakout:     bool    # prijs boven orb_high
    current:      float   # huidige prijs
    stop:         float   # onder orb_low
    tp:           float   # orb_high + 2× risk
    vol_ratio:    float   # volume ratio
    exchange:     str = ""


def generate_orb_signals(tickers: List[str], exchange: str) -> List[ORBSignaal]:
    """
    Detecteert ORB op 15m candles.
    Opening range = eerste 15 minuten van de handelsdag.
    """
    print(f"  ORB: intraday data ophalen voor {len(tickers)} tickers...")
    df = download_intraday(tickers, interval="15m")
    if df.empty:
        return []

    signalen: List[ORBSignaal] = []
    today    = pd.Timestamp.now().date()

    for ticker, g in df.groupby("Ticker", sort=False):
        g_today = g[g["Date"].dt.date == today].copy()
        if len(g_today) < 2:
            continue

        g_today = g_today.sort_values("Date")

        # Opening range = eerste candle (15m)
        orb_candle = g_today.iloc[0]
        orb_high   = safe_float(orb_candle.get("High"))
        orb_low    = safe_float(orb_candle.get("Low"))
        orb_vol    = safe_float(orb_candle.get("Volume"))

        if math.isnan(orb_high) or math.isnan(orb_low) or orb_high <= orb_low:
            continue

        orb_range = orb_high - orb_low
        risk      = orb_range  # risk = breedte opening range

        # Huidige prijs = laatste candle
        last       = g_today.iloc[-1]
        current    = safe_float(last.get("Close"))
        current_vol = safe_float(last.get("Volume"))
        if math.isnan(current):
            continue

        # Volume gemiddelde (alle candles vandaag)
        vol_mean  = g_today["Volume"].mean()
        vol_ratio = (current_vol / vol_mean) if vol_mean > 0 else 0.0

        # Breakout boven ORB high op volume
        breakout = (current > orb_high and
                    vol_ratio >= CFG["orb_vol_mult"])

        stop = orb_low
        tp   = orb_high + CFG["orb_tp_mult"] * risk

        if breakout or current > orb_high * 0.995:  # ook bijna-breakouts tonen
            signalen.append(ORBSignaal(
                ticker=ticker,
                orb_high=round(orb_high, 2),
                orb_low=round(orb_low, 2),
                orb_range=round(orb_range, 2),
                breakout=breakout,
                current=round(current, 2),
                stop=round(stop, 2),
                tp=round(tp, 2),
                vol_ratio=round(vol_ratio, 2),
                exchange=exchange,
            ))

    signalen.sort(key=lambda s: (s.breakout, s.vol_ratio), reverse=True)
    return signalen


# ============================================================
# PERFORMANCE TRACKER
# ============================================================

def compute_performance() -> Dict:
    """Berekent performance per systeem uit trades CSV."""
    perf = {
        "MR":  {"trades": 0, "win": 0, "net": 0.0, "tax": 0.0},
        "ORB": {"trades": 0, "win": 0, "net": 0.0, "tax": 0.0},
    }
    if not os.path.exists(TRADES_FILE):
        return perf

    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            return perf
        for systeem in ["MR", "ORB"]:
            sub = df[df["systeem"] == systeem]
            if sub.empty:
                continue
            sells = sub[sub["side"] == "SELL"]
            if sells.empty:
                continue
            perf[systeem]["trades"] = len(sells)
            perf[systeem]["win"]    = int((sells["net"] > 0).sum())
            perf[systeem]["net"]    = round(float(sells["net"].sum()), 2)
            perf[systeem]["tax"]    = round(float(sells["tax"].sum()), 2)
    except Exception as e:
        print(f"[WARN] Performance berekening: {e}")

    return perf

def compute_cagr_drawdown() -> Tuple[float, float]:
    """Berekent CAGR en max drawdown uit portfolio snapshots."""
    if not os.path.exists(SNAPSHOT_FILE):
        return 0.0, 0.0
    try:
        df = pd.read_csv(SNAPSHOT_FILE)
        if len(df) < 2:
            return 0.0, 0.0
        df["datum"] = pd.to_datetime(df["datum"])
        start_val = float(df["total"].iloc[0])
        end_val   = float(df["total"].iloc[-1])
        days      = (df["datum"].iloc[-1] - df["datum"].iloc[0]).days
        years     = max(days / 365.25, 1 / 365.25)
        cagr      = ((end_val / start_val) ** (1 / years) - 1) * 100 if start_val > 0 else 0.0
        peak      = df["total"].cummax()
        dd        = ((df["total"] - peak) / peak * 100).min()
        return round(cagr, 1), round(dd, 1)
    except Exception:
        return 0.0, 0.0


# ============================================================
# TELEGRAM RAPPORTEN
# ============================================================

def rapport_eod(
    mr_entries:   Dict[str, List[MRSignaal]],
    mr_exits:     Dict[str, List[MRExitSignaal]],
    portfolio:    Dict,
    prices:       Dict[str, float],
) -> str:
    nu    = now_str()
    totaal = portfolio_waarde(portfolio, prices)
    perf  = compute_performance()
    cagr, dd = compute_cagr_drawdown()

    # Max dagverlies check
    dagverlies_pct = abs(portfolio["daily_pnl"]) / totaal * 100 if totaal > 0 else 0

    lines = [
        f"📊 *MULTI-SYSTEEM BOT — EOD RAPPORT*",
        f"_{nu}_",
        f"",
        f"💼 *PORTFOLIO STATUS:*",
        f"  Totaal waarde : EUR{totaal:,.2f}",
        f"  Cash          : EUR{portfolio['cash']:,.2f}",
        f"  Posities      : {len(portfolio['positions'])}/{CFG['max_positions']}",
        f"  Dagverlies    : EUR{portfolio['daily_pnl']:+,.2f} ({dagverlies_pct:.1f}%)",
        f"  CAGR          : {cagr:+.1f}%",
        f"  Max Drawdown  : {dd:.1f}%",
        f"",
    ]

    # Max dagverlies waarschuwing
    if max_daily_loss_bereikt(portfolio, prices):
        lines.append(f"🚨 *MAX DAGVERLIES BEREIKT ({CFG['max_daily_loss_pct']*100:.0f}%) — GEEN NIEUWE TRADES*")
        lines.append("")

    # Risk budgetten
    lines += [
        f"💰 *RISK BUDGETTEN:*",
        f"  MR  (40%): EUR{risk_budget(portfolio, 'mr', prices):,.0f}",
        f"  ORB (30%): EUR{risk_budget(portfolio, 'orb', prices):,.0f}",
        f"  Reserve  : EUR{risk_budget(portfolio, 'reserve', prices):,.0f}",
        f"",
        f"─────────────────────────────",
    ]

    # Performance per systeem
    lines += [f"📈 *PERFORMANCE PER SYSTEEM:*"]
    for sys_name, p in perf.items():
        if p["trades"] > 0:
            wr = p["win"] / p["trades"] * 100
            lines.append(
                f"  {sys_name}: {p['trades']} trades | WR:{wr:.0f}% | "
                f"Net:EUR{p['net']:+,.2f} | Tax:EUR{p['tax']:,.2f}"
            )
        else:
            lines.append(f"  {sys_name}: nog geen trades")
    lines.append("")
    lines.append("─────────────────────────────")

    # IBS+RSI Entry signalen
    lines.append(f"🛡️ *SYSTEEM 1: IBS+RSI MEAN REVERSION — ENTRY SIGNALEN*")
    total_mr = sum(len(v) for v in mr_entries.values())
    if total_mr == 0:
        lines.append("_Geen paniek-koop signalen vandaag_")
    else:
        for ex, sigs in mr_entries.items():
            if not sigs:
                continue
            lines.append(f"\n*{ex}:*")
            for s in sigs[:5]:  # max 5 per exchange
                n, ml = bereken_size(totaal, s.price, s.stop)
                lines.append(
                    f"• `{s.ticker}` | IBS:{s.ibs:.2f} | RSI3:{s.rsi3:.1f} | "
                    f"EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                    + sizing_tekst(
                        s.price * (1 + CFG["slippage"]),
                        s.stop, s.tp, n, ml, totaal
                    )
                )
    lines.append("")

    # Exit signalen
    total_exits = sum(len(v) for v in mr_exits.values())
    if total_exits > 0:
        lines.append(f"🔴 *SYSTEEM 1: EXIT SIGNALEN*")
        for ex, exits in mr_exits.items():
            # Alleen exits voor open posities
            open_tickers = set(portfolio["positions"].keys())
            rel_exits    = [e for e in exits if e.ticker in open_tickers]
            for e in rel_exits:
                pos = portfolio["positions"].get(e.ticker, {})
                pnl = (e.price - pos.get("entry_price", e.price)) * pos.get("size", 0)
                lines.append(
                    f"• `{e.ticker}` 🔴 VERKOOP | {e.reden}\n"
                    f"  Prijs: EUR{e.price:.2f} | PnL est: EUR{pnl:+.2f}\n"
                    f"  Commando: `/sell {e.ticker} {e.price:.2f}`"
                )
        lines.append("")

    # Open posities
    if portfolio["positions"]:
        lines.append("📂 *OPEN POSITIES:*")
        for t, pos in portfolio["positions"].items():
            cur = prices.get(t, pos["entry_price"])
            pnl = (cur - pos["entry_price"]) * pos["size"]
            lines.append(
                f"  • `{t}` [{pos['systeem']}] {pos['size']}× @ "
                f"EUR{pos['entry_price']:.2f} | nu EUR{cur:.2f} | "
                f"PnL: EUR{pnl:+.2f} | dag {pos.get('days_open', 0)}"
            )
        lines.append("")

    lines += [
        "⚙️ *PARAMETERS:*",
        f"_IBS entry <{CFG['ibs_entry']} | RSI(3) <{CFG['rsi_entry']} | Close>MA{CFG['ma_trend']}_",
        f"_Exit: IBS>{CFG['ibs_exit']} OF Close>MA{CFG['ma_exit']}_",
        f"_Risico: {CFG['risico_pct']*100:.0f}% equity/trade | Max dagverlies: {CFG['max_daily_loss_pct']*100:.0f}%_",
    ]

    return "\n".join(lines)


def rapport_orb(
    orb_signalen: Dict[str, List[ORBSignaal]],
    portfolio:    Dict,
    prices:       Dict[str, float],
) -> str:
    nu    = now_str()
    totaal = portfolio_waarde(portfolio, prices)

    lines = [
        f"⚡ *SYSTEEM 2: ORB RAPPORT — {nu}*",
        f"_Opening Range Breakout | 15m candles_",
        f"_Budget: EUR{risk_budget(portfolio, 'orb', prices):,.0f} (30% portfolio)_",
        f"",
    ]

    if max_daily_loss_bereikt(portfolio, prices):
        lines.append(f"🚨 *MAX DAGVERLIES BEREIKT — GEEN ORB TRADES*")
        return "\n".join(lines)

    total = sum(len(v) for v in orb_signalen.values())
    if total == 0:
        lines.append("_Geen ORB breakouts vandaag_")
    else:
        for ex, sigs in orb_signalen.items():
            if not sigs:
                continue
            lines.append(f"*{ex}:*")
            for s in sigs[:3]:  # max 3 per exchange
                n, ml = bereken_size(
                    totaal,
                    s.current * (1 + CFG["slippage"]),
                    s.stop,
                )
                status = "🚀 BREAKOUT" if s.breakout else "⏳ Bijna"
                lines.append(
                    f"• `{s.ticker}` {status} | EUR{s.current:.2f} | "
                    f"Vol:{s.vol_ratio:.1f}× | {_yahoo_link(s.ticker)}\n"
                    f"  ORB: {s.orb_low:.2f}–{s.orb_high:.2f} "
                    f"(range EUR{s.orb_range:.2f})\n"
                    + sizing_tekst(
                        s.current * (1 + CFG["slippage"]),
                        s.stop, s.tp, n, ml, totaal
                    )
                )
            lines.append("")

    lines += [
        "⚙️ *ORB PARAMETERS:*",
        f"_Opening range: eerste {CFG['orb_minutes']} minuten_",
        f"_Breakout: boven ORB high op ≥{CFG['orb_vol_mult']}× volume_",
        f"_Stop: onder ORB low | TP: {CFG['orb_tp_mult']}× risk_",
        f"_Exit: zelfde dag (intraday only)_",
    ]

    return "\n".join(lines)


# ============================================================
# STATUS RAPPORT
# ============================================================

def rapport_status(portfolio: Dict, prices: Dict[str, float]) -> str:
    totaal = portfolio_waarde(portfolio, prices)
    perf   = compute_performance()
    cagr, dd = compute_cagr_drawdown()

    lines = [
        f"📊 *PORTFOLIO STATUS — {now_str()}*",
        f"",
        f"💼 EUR{totaal:,.2f} totaal | EUR{portfolio['cash']:,.2f} cash",
        f"📈 CAGR: {cagr:+.1f}% | Max DD: {dd:.1f}%",
        f"",
        f"*Open posities ({len(portfolio['positions'])}/{CFG['max_positions']}):*",
    ]

    for t, pos in portfolio["positions"].items():
        cur  = prices.get(t, pos["entry_price"])
        pnl  = (cur - pos["entry_price"]) * pos["size"]
        pnl_pct = pnl / (pos["entry_price"] * pos["size"]) * 100
        lines.append(
            f"  • `{t}` [{pos['systeem']}] {pos['size']}× @ "
            f"EUR{pos['entry_price']:.2f} → EUR{cur:.2f} "
            f"({pnl_pct:+.1f}%) | {pos.get('days_open', 0)}d"
        )

    lines += ["", "*Performance per systeem:*"]
    for sys_name, p in perf.items():
        if p["trades"] > 0:
            wr = p["win"] / p["trades"] * 100
            lines.append(
                f"  {sys_name}: {p['trades']} trades | "
                f"{wr:.0f}% win | EUR{p['net']:+,.2f} netto"
            )

    return "\n".join(lines)


# ============================================================
# EOD RUN
# ============================================================

def run_eod():
    print(f"{'='*60}")
    print(f"MULTI-SYSTEEM BOT — EOD  {today_str()}")
    print(f"{'='*60}")

    portfolio = load_portfolio()

    # Alle tickers laden
    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []
    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        if tlist:
            exchange_tickers[ex_name] = tlist
            all_tickers.extend(tlist)
    all_tickers = sorted(set(all_tickers))
    print(f"Totaal tickers: {len(all_tickers)}")

    # EOD data downloaden
    print("EOD data downloaden...")
    df = download_eod(all_tickers, period="1y")
    if df.empty:
        print("[ERROR] Geen data.")
        return
    df = add_eod_indicators(df)

    # Huidige prijzen
    prices: Dict[str, float] = {}
    last_df = df.groupby("Ticker").last().reset_index()
    for _, row in last_df.iterrows():
        t = row.get("Ticker")
        c = safe_float(row.get("Close"))
        if t and not math.isnan(c):
            prices[t] = c

    # Dagverlies check
    if max_daily_loss_bereikt(portfolio, prices):
        print(f"🚨 MAX DAGVERLIES BEREIKT — geen nieuwe trades")
        send_telegram(f"🚨 *MAX DAGVERLIES BEREIKT* ({CFG['max_daily_loss_pct']*100:.0f}%) — bot gestopt voor vandaag")
        return

    # Days open bijwerken
    for t in portfolio["positions"]:
        portfolio["positions"][t]["days_open"] = \
            portfolio["positions"][t].get("days_open", 0) + 1

    # IBS+RSI signalen per exchange
    mr_entries: Dict[str, List[MRSignaal]]     = {}
    mr_exits:   Dict[str, List[MRExitSignaal]] = {}

    email_delen: List[str] = []
  
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)].copy()
        entries, exits = generate_mr_signals(df_ex, ex_name)

        # Verwerk exits voor open posities
        for e in exits:
            if e.ticker in portfolio["positions"]:
                pos       = portfolio["positions"][e.ticker]
                exit_p    = e.price * (1 - CFG["slippage"])
                gross     = exit_p * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos.get("cost", 0))
                tax       = pnl * CFG["tax_rate"] if pnl > 0 else 0.0
                portfolio["cash"] += gross - cost - tax
                portfolio["daily_pnl"] = portfolio.get("daily_pnl", 0) + pnl - tax
                log_trade(today_str(), e.ticker, "MR", "SELL",
                          exit_p, pos["size"], cost, pnl, tax, pnl - tax, e.reden)
                del portfolio["positions"][e.ticker]
                print(f"  EXIT {e.ticker}: {e.reden} | PnL: EUR{pnl:+.2f}")

        mr_entries[ex_name] = entries
        mr_exits[ex_name]   = exits

        # Verwerk entries
        totaal = portfolio_waarde(portfolio, prices)
        budget = risk_budget(portfolio, "mr", prices)

        for sig in entries:
            if sig.ticker in portfolio["positions"]:
                continue
            if len(portfolio["positions"]) >= CFG["max_positions"]:
                break
            if max_daily_loss_bereikt(portfolio, prices):
                break

            entry_p  = sig.price * (1 + CFG["slippage"])
            n, ml    = bereken_size(totaal, entry_p, sig.stop)
            if n <= 0:
                continue
            investering = entry_p * n + trade_cost(entry_p * n)
            if investering > portfolio["cash"] or investering > budget:
                continue

            portfolio["cash"] -= investering
            portfolio["positions"][sig.ticker] = {
                "entry_price": round(entry_p, 4),
                "size":        n,
                "stop":        sig.stop,
                "tp":          sig.tp,
                "systeem":     "MR",
                "exchange":    ex_name,
                "days_open":   0,
                "cost":        trade_cost(entry_p * n),
            }
            portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
            log_trade(today_str(), sig.ticker, "MR", "BUY",
                      entry_p, n, trade_cost(entry_p * n), 0, 0, 0, "IBS+RSI entry")
            print(f"  KOOP {sig.ticker}: IBS={sig.ibs:.2f} RSI3={sig.rsi3:.1f} | {n}× EUR{entry_p:.2f}")

    # Max houdduur check
    for t, pos in list(portfolio["positions"].items()):
        if (pos.get("systeem") == "MR" and
                pos.get("days_open", 0) >= CFG["mr_max_hold"]):
            cur   = prices.get(t, pos["entry_price"])
            exit_p = cur * (1 - CFG["slippage"])
            gross  = exit_p * pos["size"]
            cost   = trade_cost(gross)
            pnl    = gross - cost - (pos["entry_price"] * pos["size"] + pos.get("cost", 0))
            tax    = pnl * CFG["tax_rate"] if pnl > 0 else 0.0
            portfolio["cash"] += gross - cost - tax
            portfolio["daily_pnl"] = portfolio.get("daily_pnl", 0) + pnl - tax
            log_trade(today_str(), t, "MR", "SELL",
                      exit_p, pos["size"], cost, pnl, tax, pnl - tax,
                      f"Max houdduur {CFG['mr_max_hold']}d")
            del portfolio["positions"][t]
            print(f"  TIME EXIT {t}: max houdduur | PnL: EUR{pnl:+.2f}")

    # Snapshot en opslaan
    log_snapshot(portfolio, prices)
    save_portfolio(portfolio)

    # Telegram rapport
    rapport = rapport_eod(mr_entries, mr_exits, portfolio, prices)
    send_telegram(rapport)
    send_email(f"Multi-Systeem EOD rapport {today_str()}", rapport.replace("*","").replace("`",""))
    print("EOD rapport verzonden.")


# ============================================================
# ORB RUN (INTRADAY)
# ============================================================

def run_orb():
    print(f"{'='*60}")
    print(f"MULTI-SYSTEEM BOT — ORB  {now_str()}")
    print(f"{'='*60}")

    portfolio = load_portfolio()

    # Alleen top exchanges voor ORB (volume belangrijk)
    orb_exchanges = {
        "048 Nasdaq/NYSE": "tickers_048x.txt",
        "047 Toronto":     "tickers_047x.txt",
        "045 Londen":      "tickers_045x.txt",
        "041 Benelux":     "tickers_041x.txt",
    }

    # Huidige prijzen ophalen voor portfolio waarde
    all_tickers = []
    for path in orb_exchanges.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))

    prices: Dict[str, float] = {}
    try:
        data = yf.download(all_tickers[:50], period="1d", auto_adjust=True,
                           progress=False, group_by="ticker")
        if data is not None and not data.empty and isinstance(data.columns, pd.MultiIndex):
            for t in all_tickers[:50]:
                try:
                    c = safe_float(data.xs(t, axis=1, level=1)["Close"].iloc[-1])
                    if not math.isnan(c):
                        prices[t] = c
                except Exception:
                    pass
    except Exception:
        pass

    if max_daily_loss_bereikt(portfolio, prices):
        print("Max dagverlies bereikt — geen ORB trades")
        return

    orb_signalen: Dict[str, List[ORBSignaal]] = {}
    for ex_name, path in orb_exchanges.items():
        tlist = load_tickers_from_file(path)
        if not tlist:
            continue
        # Beperk tot 30 tickers voor snelheid
        tlist = tlist[:30]
        print(f"  ORB scan: {ex_name} ({len(tlist)} tickers)...")
        sigs = generate_orb_signals(tlist, ex_name)
        orb_signalen[ex_name] = sigs

        # Verwerk breakout entries
        totaal = portfolio_waarde(portfolio, prices)
        budget = risk_budget(portfolio, "orb", prices)

        for sig in [s for s in sigs if s.breakout]:
            if sig.ticker in portfolio["positions"]:
                continue
            if len(portfolio["positions"]) >= CFG["max_positions"]:
                break

            entry_p  = sig.current * (1 + CFG["slippage"])
            n, ml    = bereken_size(totaal, entry_p, sig.stop)
            if n <= 0:
                continue
            investering = entry_p * n + trade_cost(entry_p * n)
            if investering > portfolio["cash"] or investering > budget:
                continue

            portfolio["cash"] -= investering
            portfolio["positions"][sig.ticker] = {
                "entry_price": round(entry_p, 4),
                "size":        n,
                "stop":        sig.stop,
                "tp":          sig.tp,
                "systeem":     "ORB",
                "exchange":    ex_name,
                "days_open":   0,
                "cost":        trade_cost(investering),
                "intraday":    True,
            }
            log_trade(today_str(), sig.ticker, "ORB", "BUY",
                      entry_p, n, trade_cost(investering), 0, 0, 0, "ORB breakout")
            prices[sig.ticker] = sig.current
            print(f"  ORB KOOP {sig.ticker}: EUR{sig.current:.2f} | vol {sig.vol_ratio:.1f}×")

    save_portfolio(portfolio)

    rapport = rapport_orb(orb_signalen, portfolio, prices)
    send_telegram(rapport)
    send_email(f"ORB rapport {today_str()}", rapport.replace("*","").replace("`",""))
    print("ORB rapport verzonden.")


# ============================================================
# STATUS RUN
# ============================================================

def run_status():
    portfolio = load_portfolio()
    tickers   = list(portfolio["positions"].keys())
    prices    = {}
    if tickers:
        try:
            data = yf.download(tickers, period="1d", auto_adjust=True,
                               progress=False, group_by="ticker")
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    for t in tickers:
                        try:
                            c = safe_float(data.xs(t, axis=1, level=1)["Close"].iloc[-1])
                            if not math.isnan(c):
                                prices[t] = c
                        except Exception:
                            pass
                else:
                    c = safe_float(data["Close"].iloc[-1])
                    if not math.isnan(c) and tickers:
                        prices[tickers[0]] = c
        except Exception:
            pass

    rapport = rapport_status(portfolio, prices)
    send_telegram(rapport)
    print(rapport)


# ============================================================
# BACKTEST
# ============================================================

def run_backtest():
    """Backtest van IBS+RSI systeem op historische EOD data."""
    print(f"{'='*60}")
    print(f"MULTI-SYSTEEM BACKTEST — IBS+RSI  {BACKTEST_START} → {BACKTEST_END}")
    print(f"NB: ORB backtest vereist historische intraday data (niet beschikbaar via yfinance)")
    print(f"{'='*60}")

    all_tickers = []
    for path in EXCHANGES.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (3y)...")
    df = download_eod(all_tickers, period="3y")
    if df.empty:
        print("[ERROR] Geen data.")
        return
    df = add_eod_indicators(df)

    all_dates = sorted(df["Date"].dt.date.unique())
    print(f"Handelsdagen: {len(all_dates)}")

    cash      = CFG["start_capital"]
    positions: Dict[str, Dict] = {}
    trades:    List[Dict] = []
    snapshots: List[Dict] = []

    for date in all_dates:
        day_df = df[df["Date"] == pd.Timestamp(date)].copy()

        prices: Dict[str, float] = {}
        for _, row in day_df.iterrows():
            t = row.get("Ticker")
            c = safe_float(row.get("Close"))
            if t and not math.isnan(c):
                prices[t] = c

        # Days open
        for pos in positions.values():
            pos["days"] = pos.get("days", 0) + 1

        # Exits
        for t, pos in list(positions.items()):
            if t not in prices:
                continue
            row_s = day_df[day_df["Ticker"] == t]
            if row_s.empty:
                continue
            row   = row_s.iloc[0]
            close = prices[t]
            ibs   = safe_float(row.get("IBS"))
            ma20  = safe_float(row.get("MA20"))

            reason = None
            if not math.isnan(ibs) and ibs > CFG["ibs_exit"]:
                reason = f"IBS exit ({ibs:.2f})"
            elif not math.isnan(ma20) and close > ma20:
                reason = f"MA20 exit ({ma20:.2f})"
            elif pos["days"] >= CFG["mr_max_hold"]:
                reason = f"Time exit ({pos['days']}d)"
            elif close <= pos["stop"]:
                reason = f"Stop ({pos['stop']:.2f})"

            if reason:
                exit_p = close * (1 - CFG["slippage"])
                gross  = exit_p * pos["size"]
                cost   = trade_cost(gross)
                pnl    = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax    = pnl * CFG["tax_rate"] if pnl > 0 else 0.0
                cash  += gross - cost - tax
                trades.append({
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   date.isoformat(),
                    "ticker":      t,
                    "entry_price": pos["entry_price"],
                    "exit_price":  round(exit_p, 4),
                    "size":        pos["size"],
                    "ibs_entry":   pos.get("ibs_entry", 0),
                    "rsi_entry":   pos.get("rsi_entry", 0),
                    "pnl":         round(pnl, 2),
                    "tax":         round(tax, 2),
                    "net":         round(pnl - tax, 2),
                    "reason":      reason,
                    "days":        pos["days"],
                })
                del positions[t]

        # Entries
        totaal = cash + sum(prices.get(t, p["entry_price"]) * p["size"]
                            for t, p in positions.items())

        for _, row in day_df.iterrows():
            t   = row.get("Ticker")
            if not t or t in positions:
                continue
            if len(positions) >= CFG["max_positions"]:
                break

            close = safe_float(row.get("Close"))
            ibs   = safe_float(row.get("IBS"))
            rsi3  = safe_float(row.get("RSI3"))
            ma200 = safe_float(row.get("MA200"))
            ma20  = safe_float(row.get("MA20"))
            atr   = safe_float(row.get("ATR14"))
            low   = safe_float(row.get("Low"))

            if any(math.isnan(x) for x in [close, ibs, rsi3, ma200, atr]):
                continue

            if not (ibs < CFG["ibs_entry"] and
                    rsi3 < CFG["rsi_entry"] and
                    close > ma200):
                continue

            entry_p = close * (1 + CFG["slippage"])
            stop    = low - 0.5 * atr
            tp      = ma20
            n, _    = bereken_size(totaal, entry_p, stop)
            if n <= 0:
                continue
            investering = entry_p * n + trade_cost(entry_p * n)
            if investering > cash:
                continue

            cash -= investering
            positions[t] = {
                "entry_date":  date,
                "entry_price": round(entry_p, 4),
                "size":        n,
                "stop":        round(stop, 4),
                "tp":          round(tp, 4) if not math.isnan(tp) else round(entry_p * 1.05, 4),
                "cost":        trade_cost(investering),
                "days":        0,
                "ibs_entry":   round(ibs, 3),
                "rsi_entry":   round(rsi3, 1),
            }

        # Snapshot
        totaal = cash + sum(prices.get(t, p["entry_price"]) * p["size"]
                            for t, p in positions.items())
        snapshots.append({"date": date.isoformat(), "total": round(totaal, 2)})

    # Resultaten
    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("mr_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        snap = pd.DataFrame(snapshots)
        snap["peak"] = snap["total"].cummax()
        snap["dd"]   = (snap["total"] - snap["peak"]) / snap["peak"] * 100
        start = CFG["start_capital"]
        end   = snap["total"].iloc[-1]
        years = max(len(snap) / 252, 0.1)
        cagr  = ((end / start) ** (1 / years) - 1) * 100

        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTATEN — IBS+RSI Mean Reversion")
        print(f"{'='*60}")
        print(f"Startkapitaal    : EUR{start:>12,.2f}")
        print(f"Eindkapitaal     : EUR{end:>12,.2f}")
        print(f"Totaal rendement : {(end-start)/start*100:>+.1f}%")
        print(f"CAGR             : {cagr:>+.1f}%")
        print(f"Max Drawdown     : {snap['dd'].min():>.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"\nExit redenen:")
        for r, g in tdf.groupby("reason"):
            wr = (g["net"] > 0).sum() / len(g) * 100
            print(f"  {r[:30]:<30} {len(g):>4} trades | {wr:>5.1f}% win | EUR{g['net'].sum():>+10.2f}")
        print(f"{'='*60}")
        print(f"Opgeslagen: mr_backtest_trades.csv")
    else:
        print("Geen trades gegenereerd.")


BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "eod"
    if mode == "eod":
        run_eod()
    elif mode == "orb":
        run_orb()
    elif mode == "status":
        run_status()
    elif mode == "backtest":
        run_backtest()
    else:
        print(f"Onbekende modus: {mode}")
        print("Gebruik: python bot_00mr.py [eod|orb|status|backtest]")
