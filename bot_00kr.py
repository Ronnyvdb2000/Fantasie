#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00kr.py  —  KRITISCHE SELECTIE ENGINE v1.1
Gebaseerd op de publiek bekende criteria van Satilmis Ersintepe.

Criteria (score 0-5):
  1. RSI maandelijks  — oververkocht (<40) of momentum hervatting (>55)
  2. MACD maandelijks — bullish crossover of histogram groeit
  3. Risk/Reward >=25% — opwaarts naar jaarweerstand vs. 2xATR stop
  4. Support nabijheid — prijs binnen 3% van 52-weekse bodem
  5. Dividend yield   — yield > 0% (bonus)

Gebruik:
  python bot_00kr.py live     # live rapport
  python bot_00kr.py backtest # backtest modus
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
ATR_STOP_MULT        = 2.0
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10

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

KS_CFG = {
    "rsi_period":          14,
    "rsi_oversold":        40.0,
    "rsi_momentum":        55.0,
    "macd_fast":           12,
    "macd_slow":           26,
    "macd_signal":         9,
    "min_rr_pct":          25.0,
    "atr_stop_mult":       2.0,
    "atr_period":          14,
    "support_lookback":    252,
    "support_proximity":   3.0,
    "resistance_lookback": 252,
    "min_score":           3,
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES
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

def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# ATR SIZING
# ============================================================

def bereken_atr_positie(
    portfolio_waarde: float,
    entry_prijs:      float,
    atr:              float,
    sl_mult:          float = ATR_STOP_MULT,
    risico_pct:       float = RISICO_PCT_PER_TRADE,
) -> Tuple[int, float, float]:
    risico_eur   = portfolio_waarde * risico_pct
    stop_afstand = sl_mult * atr
    if stop_afstand <= 0 or entry_prijs <= 0:
        return 0, entry_prijs, 0.0
    aandelen    = max(1, int(risico_eur / stop_afstand))
    stop_loss   = entry_prijs - stop_afstand
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, stop_loss, max_verlies

def sizing_tekst(ticker, prijs, atr, portfolio_waarde, sl_mult=2.0, tp_pct=0.25) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, stop, max_loss = bereken_atr_positie(portfolio_waarde, entry, atr, sl_mult)
    tp          = entry * (1 + tp_pct)
    investering = round(entry * aandelen, 2)
    kosten      = round(trade_cost(investering), 2)
    return (
        f"  Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | TP: EUR{tp:.2f}\n"
        f"  {aandelen} stuks | EUR{investering:,.2f} | Max verlies: EUR{max_loss:,.2f}"
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

def download_history(tickers: List[str], period: str = "3y") -> pd.DataFrame:
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
# TECHNISCHE INDICATOREN
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

def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = _wilder_smooth(gain, period) / (_wilder_smooth(loss, period) + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macd_series(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


# ============================================================
# KRITISCHE SELECTIE SIGNAAL
# ============================================================

@dataclass
class KSSignaal:
    ticker:       str
    price:        float
    score:        int
    atr:          float
    resistance:   float
    stop:         float
    rr_pct:       float
    support:      float
    rsi_monthly:  float
    rsi_label:    str
    macd_label:   str
    rr_label:     str
    support_label: str
    div_label:    str
    div_yield:    float
    tp_pct:       float

def analyse_ticker(ticker: str, df_ticker: pd.DataFrame) -> Optional[KSSignaal]:
    try:
        g = df_ticker.sort_values("Date").copy()
        if len(g) < 60:
            return None

        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0:
            return None

        atr_day = compute_atr_series(high, low, close, KS_CFG["atr_period"])
        atr_val = safe_float(atr_day.iloc[-1])
        if math.isnan(atr_val) or atr_val <= 0:
            atr_val = current_price * 0.02

        score = 0

        g_indexed = g.set_index("Date")
        monthly   = g_indexed["Close"].resample("ME").last().dropna()

        # 1. RSI maandelijks
        rsi_label = "neutraal"
        rsi_val   = 50.0
        if len(monthly) >= KS_CFG["rsi_period"] + 2:
            rsi_m   = compute_rsi_series(monthly, KS_CFG["rsi_period"])
            rsi_val = safe_float(rsi_m.iloc[-1], 50.0)
            rsi_prv = safe_float(rsi_m.iloc[-2], 50.0)
            if rsi_val < KS_CFG["rsi_oversold"]:
                score += 1
                rsi_label = f"✓ oververkocht ({rsi_val:.1f})"
            elif rsi_val > KS_CFG["rsi_momentum"] and rsi_prv <= KS_CFG["rsi_momentum"]:
                score += 1
                rsi_label = f"✓ momentum ({rsi_val:.1f} up)"
            elif KS_CFG["rsi_oversold"] <= rsi_val <= 52 and rsi_val > rsi_prv:
                score += 1
                rsi_label = f"✓ herstel ({rsi_val:.1f} up)"
            else:
                rsi_label = f"✗ neutraal ({rsi_val:.1f})"
        else:
            rsi_label = "✗ onvoldoende data"

        # 2. MACD maandelijks
        macd_label = "✗ neutraal"
        if len(monthly) >= KS_CFG["macd_slow"] + KS_CFG["macd_signal"] + 2:
            macd_line, sig_line, hist = compute_macd_series(
                monthly, KS_CFG["macd_fast"], KS_CFG["macd_slow"], KS_CFG["macd_signal"]
            )
            m_now = safe_float(macd_line.iloc[-1])
            m_prv = safe_float(macd_line.iloc[-2])
            s_now = safe_float(sig_line.iloc[-1])
            s_prv = safe_float(sig_line.iloc[-2])
            h_now = safe_float(hist.iloc[-1])
            h_prv = safe_float(hist.iloc[-2])
            if (m_prv < s_prv) and (m_now >= s_now):
                score += 1
                macd_label = "✓ bullish crossover"
            elif h_now > h_prv and h_now > 0:
                score += 1
                macd_label = f"✓ histogram up ({h_now:.3f})"
            else:
                macd_label = f"✗ neutraal ({m_now:.3f})"

        # 3. Risk/Reward >= 25%
        lb         = min(KS_CFG["resistance_lookback"], len(close))
        resistance = float(close.iloc[-lb:].max())
        stop_price = current_price - KS_CFG["atr_stop_mult"] * atr_val
        rr_pct     = 0.0
        rr_label   = "✗ onvoldoende"
        if resistance > current_price and stop_price > 0:
            upside   = (resistance - current_price) / current_price * 100
            downside = (current_price - stop_price) / current_price * 100
            if downside > 0:
                ratio  = upside / downside
                rr_pct = upside
                if upside >= KS_CFG["min_rr_pct"] and ratio >= 1.5:
                    score   += 1
                    rr_label = f"✓ {upside:.1f}% opwaarts (ratio {ratio:.1f}:1)"
                else:
                    rr_label = f"✗ {upside:.1f}% / ratio {ratio:.1f}:1"

        tp_pct = rr_pct / 100.0

        # 4. Support nabijheid
        lb_sup        = min(KS_CFG["support_lookback"], len(close))
        support_level = float(close.iloc[-lb_sup:].min())
        dist_pct      = ((current_price - support_level) / support_level) * 100
        if dist_pct <= KS_CFG["support_proximity"]:
            score        += 1
            support_label = f"✓ {dist_pct:.1f}% boven bodem ({support_level:.2f})"
        else:
            support_label = f"✗ {dist_pct:.1f}% boven bodem ({support_level:.2f})"

        # 5. Dividend
        div_yield = 0.0
        div_label = "✗ geen dividend"
        try:
            info      = yf.Ticker(ticker).info
            div_yield = (info.get("dividendYield") or 0.0) * 100
            if div_yield > 0:
                score    += 1
                div_label = f"✓ {div_yield:.2f}%"
        except Exception:
            pass

        return KSSignaal(
            ticker=ticker, price=round(current_price, 2), score=score,
            atr=round(atr_val, 4), resistance=round(resistance, 2),
            stop=round(stop_price, 2), rr_pct=round(rr_pct, 1),
            support=round(support_level, 2), rsi_monthly=round(rsi_val, 1),
            rsi_label=rsi_label, macd_label=macd_label,
            rr_label=rr_label, support_label=support_label,
            div_label=div_label, div_yield=round(div_yield, 2),
            tp_pct=max(tp_pct, 0.25),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT  — één bericht per exchange
# ============================================================

def _score_bar(score: int) -> str:
    return "█" * score + "░" * (5 - score) + f" {score}/5"

def format_bericht(
    exchange_name:   str,
    signalen:        List[KSSignaal],   # score >= min_score
    alle:            List[KSSignaal],   # alle tickers voor top-2
    portfolio_waarde: float,
) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None

    nu     = today_str()
    top2   = sorted(alle, key=lambda s: (s.score * 10 + s.rr_pct), reverse=True)[:2]
    max_sc = max((s.score for s in signalen), default=0) if signalen else 0
    toon   = [s for s in signalen if s.score == max_sc] if signalen else []
    lbl    = {5: "⭐ PERFECTE SCORE (5/5)", 4: "🟡 STERK (4/5)",
              3: "🟠 WATCHLIST (3/5)"}.get(max_sc, "📊")

    def sig_regel(s: KSSignaal) -> str:
        return (
            f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
            f"RR:{s.rr_pct:.1f}% | {_yahoo_link(s.ticker)}\n"
            + sizing_tekst(s.ticker, s.price, s.atr, portfolio_waarde,
                           KS_CFG["atr_stop_mult"], s.tp_pct)
        )

    delen = [
        f"🔍 *KRITISCHE SELECTIE — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(signalen)} kandidaten (score>={KS_CFG['min_score']})_",
        "─────────────────────────────",
        f"🏆 *TOP 2 HOOGSTE POTENTIEEL:*",
        "\n\n".join(sig_regel(s) for s in top2),
    ]

    if toon:
        delen += ["─────────────────────────────", f"*{lbl}:*"]
        for s in [x for x in toon if x not in top2]:
            delen.append(
                f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | "
                f"RR:{s.rr_pct:.1f}% | RSI:{s.rsi_monthly:.1f}\n"
                f"  {s.rsi_label} | {s.macd_label}"
            )

    delen.append(
        f"⚙️ _Stop 2×ATR | RR>={KS_CFG['min_rr_pct']:.0f}% | "
        f"RSI maandelijks | Risico 5% portfolio_"
    )
    return "\n\n".join(delen)


# ============================================================
# CSV LOGGING
# ============================================================

def _log_csv(signalen: List[KSSignaal], exchange: str):
    fname  = f"ks_signalen_{exchange.split()[0]}_{today_str()}.csv"
    header = ["datum","exchange","ticker","score","price","rsi_monthly",
              "rsi_label","macd_label","rr_pct","rr_label","support",
              "support_label","resistance","stop","div_yield","div_label"]
    ensure_csv_header(fname, header)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in signalen:
            w.writerow([today_str(), exchange, s.ticker, s.score, s.price,
                        s.rsi_monthly, s.rsi_label, s.macd_label,
                        s.rr_pct, s.rr_label, s.support, s.support_label,
                        s.resistance, s.stop, s.div_yield, s.div_label])
    print(f"  CSV: {fname}")


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"KRITISCHE SELECTIE — LIVE  {today_str()}")
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
    print("Data downloaden (3 jaar)...")
    df = download_history(all_tickers, period="3y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers, {len(df)} rijen")
    portfolio_waarde = START_CAPITAL
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        alle: List[KSSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group)
            if sig is not None:
                alle.append(sig)
                if sig.score >= KS_CFG["min_score"]:
                    print(f"  ✓ {ticker}: score {sig.score}/5 | RSI={sig.rsi_monthly:.1f} | RR={sig.rr_pct:.1f}%")

        signalen = [s for s in alle if s.score >= KS_CFG["min_score"]]
        signalen.sort(key=lambda s: (s.score, s.rr_pct), reverse=True)
        alle.sort(key=lambda s: (s.score * 10 + s.rr_pct), reverse=True)

        print(f"  → {len(signalen)} kandidaten (score >= {KS_CFG['min_score']}) | top-2 uit {len(alle)} geanalyseerd")

        bericht = format_bericht(ex_name, signalen, alle, portfolio_waarde)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")
        else:
            print(f"  → Overgeslagen: {ex_name}")

        if signalen:
            _log_csv(signalen, ex_name)

    if email_delen:
        send_email(
            f"Kritische Selectie rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"KRITISCHE SELECTIE BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
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

    all_dates  = sorted(df["Date"].dt.date.unique())
    cash       = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []

    scan_dates = []
    prev_month = None
    for d in all_dates:
        if d.month != prev_month:
            scan_dates.append(d)
            prev_month = d.month

    print(f"Scanmomenten: {len(scan_dates)} (maandelijks)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig or sig.score < KS_CFG["min_score"]:
                continue
            entry       = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, stop, _ = bereken_atr_positie(cash, entry, sig.atr)
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue
            cash -= investering
            positions[ticker] = {
                "entry_date": scan_date, "entry_price": round(entry, 4),
                "size": aandelen, "stop": sig.stop, "tp": sig.resistance,
                "score": sig.score, "days": 0, "cost": trade_cost(investering),
            }

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
            elif pos["days"] >= 60:
                reason = "Time (60d)"

            if reason:
                exit_slip = close * (1 - SLIPPAGE_PCT)
                gross     = exit_slip * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax       = pnl * TAX_RATE if pnl > 0 else 0.0
                cash     += gross - cost - tax
                trades.append({
                    "entry_date": pos["entry_date"].isoformat(),
                    "exit_date":  scan_date.isoformat(),
                    "ticker": ticker, "score": pos["score"],
                    "entry_price": pos["entry_price"],
                    "exit_price": round(exit_slip, 4),
                    "size": pos["size"], "pnl": round(pnl, 2),
                    "tax": round(tax, 2), "net": round(pnl - tax, 2),
                    "reason": reason, "days": pos["days"],
                })
                del positions[ticker]

    if trades:
        tdf  = pd.DataFrame(trades)
        tdf.to_csv("ks_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final = cash + sum(price_map.get(t, p["entry_price"]) * p["size"]
                           for t, p in positions.items())
        print(f"\n{'='*60}")
        print(f"Startkapitaal : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal  : EUR{final:>12,.2f}")
        print(f"Rendement     : {(final-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades        : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor : {pf:.2f}")
        print(f"Belasting     : EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur : {tdf['days'].mean():.1f} dagen")
        print(f"{'='*60}")
        print("Opgeslagen: ks_backtest_trades.csv")
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
