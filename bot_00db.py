#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00db.py  —  DARVAS BOX ENGINE v1.0
Gebaseerd op de originele methode van Nicolas Darvas (1950s).

Score systeem (0-5):
  1. Nieuw 52-weekse high in de laatste 5 dagen
  2. Box gevormd (top + bottom vastgesteld)
  3. Breakout boven box top vandaag of gisteren
  4. Volume breakout >= 1.5x gemiddelde
  5. Stijgende trend (Close > MA200)

Gebruik:
  python bot_00db.py live     # live rapport
  python bot_00db.py backtest # backtest modus

GitHub Actions: dagelijks om 21:55 UTC
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
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10
MAX_HOLD_DAYS        = 40

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

DB_CFG = {
    "high52w_lookback":     252,
    "new_high_days":        5,
    "box_confirm_days":     3,
    "box_bottom_pct":       10.0,
    "breakout_vol_mult":    1.5,
    "vol_ma_period":        20,
    "ma_trend":             200,
    "min_score":            3,
    "atr_period":           14,
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
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# ATR SIZING
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

def sizing_tekst(ticker, prijs, stop, box_top, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    investering = round(entry * aandelen, 2)
    kosten      = round(trade_cost(investering), 2)
    rr          = ((box_top * 1.15 - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | R/R: {rr:.1f}:1\n"
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
# DARVAS BOX DETECTIE
# ============================================================

@dataclass
class DarvasBox:
    top:         float
    bottom:      float
    formed:      bool
    days_in_box: int

def detect_darvas_box(close: pd.Series, high: pd.Series) -> Optional[DarvasBox]:
    if len(close) < DB_CFG["box_confirm_days"] + 5:
        return None
    highs  = high.values
    closes = close.values
    n      = len(highs)

    box_top_idx = None
    for i in range(n - DB_CFG["box_confirm_days"] - 1,
                   max(n - 60, DB_CFG["box_confirm_days"]) - 1, -1):
        candidate = highs[i]
        confirm   = all(highs[i + j] < candidate
                        for j in range(1, DB_CFG["box_confirm_days"] + 1))
        if confirm:
            box_top_idx = i
            break

    if box_top_idx is None:
        return None

    box_top      = float(highs[box_top_idx])
    max_bottom   = box_top * (1 - DB_CFG["box_bottom_pct"] / 100)
    slice_closes = closes[box_top_idx:]
    valid_bottoms = [c for c in slice_closes if c >= max_bottom]
    if not valid_bottoms:
        return None

    box_bottom  = float(min(valid_bottoms))
    days_in_box = n - box_top_idx

    return DarvasBox(
        top=round(box_top, 4),
        bottom=round(box_bottom, 4),
        formed=True,
        days_in_box=days_in_box,
    )


# ============================================================
# DARVAS SIGNAAL
# ============================================================

@dataclass
class DarvasSignaal:
    ticker:           str
    price:            float
    score:            int
    box:              DarvasBox
    breakout:         bool
    vol_ratio:        float
    new_high:         bool
    trend_ok:         bool
    ma200:            float
    high52w:          float
    pct_above_bottom: float
    score_labels:     List[str]
    total_score:      float

def analyse_ticker(ticker: str, g: pd.DataFrame) -> Optional[DarvasSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < DB_CFG["ma_trend"] + 10:
            return None

        close  = g["Close"]
        high   = g["High"]
        volume = g["Volume"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        ma200     = safe_float(close.rolling(DB_CFG["ma_trend"]).mean().iloc[-1])
        vol_ma    = volume.rolling(DB_CFG["vol_ma_period"]).mean()
        vol_now   = safe_float(volume.iloc[-1])
        vol_avg   = safe_float(vol_ma.iloc[-1])
        vol_ratio = (vol_now / vol_avg) if vol_avg > 0 else 0.0

        lb         = min(DB_CFG["high52w_lookback"], len(close))
        h52w       = float(close.iloc[-lb:].max())
        recent_max = float(close.iloc[-DB_CFG["new_high_days"]:].max())
        new_high   = (recent_max >= h52w * 0.995)

        box = detect_darvas_box(close, high)
        if box is None:
            return None

        pct_above_bottom = ((current_price - box.bottom) / box.bottom * 100)
        breakout = current_price > box.top
        if not breakout and len(close) >= 2:
            breakout = safe_float(close.iloc[-2]) > box.top

        score  = 0
        labels = []

        if new_high:
            score += 1
            labels.append(f"✓ Nieuw 52w high ({recent_max:.2f})")
        else:
            labels.append(f"✗ Geen nieuw high (high={h52w:.2f})")

        if box.formed:
            score += 1
            labels.append(f"✓ Box gevormd: {box.bottom:.2f}-{box.top:.2f} ({box.days_in_box}d)")
        else:
            labels.append("✗ Geen box")

        if breakout:
            score += 1
            labels.append(f"✓ Breakout boven {box.top:.2f}")
        else:
            labels.append(f"✗ Geen breakout (top={box.top:.2f})")

        if vol_ratio >= DB_CFG["breakout_vol_mult"]:
            score += 1
            labels.append(f"✓ Volume {vol_ratio:.1f}x gem.")
        else:
            labels.append(f"✗ Volume {vol_ratio:.1f}x gem. (min {DB_CFG['breakout_vol_mult']}x)")

        trend_ok = not math.isnan(ma200) and current_price > ma200
        if trend_ok:
            score += 1
            labels.append(f"✓ Close > MA200 ({ma200:.2f})")
        else:
            labels.append(f"✗ Close < MA200 ({ma200:.2f})")

        if score < DB_CFG["min_score"]:
            return None

        total_score = (
            score * 10
            + (vol_ratio * 5 if vol_ratio >= DB_CFG["breakout_vol_mult"] else 0)
            + (10 if breakout else 0)
            + (5 if new_high else 0)
        )

        return DarvasSignaal(
            ticker=ticker, price=round(current_price, 2),
            score=score, box=box, breakout=breakout,
            vol_ratio=round(vol_ratio, 2), new_high=new_high,
            trend_ok=trend_ok,
            ma200=round(ma200, 2) if not math.isnan(ma200) else 0.0,
            high52w=round(h52w, 2),
            pct_above_bottom=round(pct_above_bottom, 1),
            score_labels=labels,
            total_score=round(total_score, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 5) -> str:
    return "█" * score + "░" * (max_score - score) + f" {score}/{max_score}"

def format_bericht(
    exchange_name:    str,
    signalen:         List[DarvasSignaal],
    portfolio_waarde: float,
) -> Optional[str]:
    if not signalen:
        return None

    nu        = today_str()
    max_score = max(s.score for s in signalen)
    toon      = [s for s in signalen if s.score == max_score]
    top2      = signalen[:2]
    lbl       = {5: "⭐ PERFECTE SCORE (5/5)", 4: "🔥 STERK (4/5)",
                 3: "📊 WATCHLIST (3/5)"}.get(max_score, "📊")

    delen = [
        f"📦 *DARVAS BOX — {exchange_name}*",
        f"_{nu} | {len(signalen)} kandidaten | {sum(s.breakout for s in signalen)} breakouts_",
        "─────────────────────────────",
        "🏆 *TOP 2:*",
    ]
    for s in top2:
        bw = ((s.box.top - s.box.bottom) / s.box.bottom * 100)
        delen.append(
            f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} {_yahoo_link(s.ticker)}\n"
            f"  Box: EUR{s.box.bottom:.2f}-EUR{s.box.top:.2f} ({bw:.1f}%) | Vol:{s.vol_ratio:.1f}x\n"
            + sizing_tekst(s.ticker, s.price, s.box.bottom, s.box.top, portfolio_waarde)
        )

    delen += ["─────────────────────────────", f"*{lbl}:*"]
    extra = [s for s in toon if s not in top2]
    if extra:
        for s in extra:
            delen.append(
                f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} | "
                f"{'BREAKOUT' if s.breakout else 'in box'} | vol {s.vol_ratio:.1f}x"
            )
    else:
        delen.append("_Zie top 2 hierboven_")

    delen.append(
        f"⚙️ _Box+{DB_CFG['box_confirm_days']}d | Breakout {DB_CFG['breakout_vol_mult']}x vol | Risico 5%_"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"DARVAS BOX — LIVE  {today_str()}")
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
    portfolio_waarde = START_CAPITAL
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        signalen: List[DarvasSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group)
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: score {sig.score}/5 | "
                    f"box {sig.box.bottom:.2f}-{sig.box.top:.2f} | "
                    f"vol {sig.vol_ratio:.1f}x | "
                    f"{'BREAKOUT' if sig.breakout else 'in box'}"
                )

        signalen.sort(key=lambda s: s.total_score, reverse=True)
        print(f"  → {len(signalen)} Darvas kandidaten | {sum(s.breakout for s in signalen)} breakouts")

        bericht = format_bericht(ex_name, signalen, portfolio_waarde)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Overgeslagen (geen signalen): {ex_name}")

    if email_delen:
        send_email(
            subject=f"Darvas Box rapport {today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"DARVAS BOX BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
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
    price_map: Dict[str, float] = {}

    print(f"Handelsdagen: {len(all_dates)}")

    for datum in all_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(datum)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(datum)].copy()

        price_map = {}
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
            elif pos["days"] >= MAX_HOLD_DAYS:
                reason = f"Time ({pos['days']}d)"
            elif close > pos["box_top"] * 1.05:
                reason = "Nieuwe box trailing"
            if reason:
                exit_slip = close * (1 - SLIPPAGE_PCT)
                gross     = exit_slip * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax       = pnl * TAX_RATE if pnl > 0 else 0.0
                cash     += gross - cost - tax
                trades.append({
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   datum.isoformat(),
                    "ticker":      ticker,
                    "score":       pos["score"],
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

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig or not sig.breakout:
                continue
            entry       = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, _ = bereken_positie(cash, entry, sig.box.bottom)
            if aandelen <= 0:
                continue
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue
            cash -= investering
            positions[ticker] = {
                "entry_date":  datum,
                "entry_price": round(entry, 4),
                "size":        aandelen,
                "stop":        sig.box.bottom,
                "box_top":     sig.box.top,
                "score":       sig.score,
                "days":        0,
                "cost":        trade_cost(investering),
            }

    if trades:
        tdf  = pd.DataFrame(trades)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final_val = cash + sum(
            price_map.get(t, p["entry_price"]) * p["size"]
            for t, p in positions.items())
        print(f"\n{'='*60}")
        print(f"Startkapitaal    : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal     : EUR{final_val:>12,.2f}")
        print(f"Totaal rendement : {(final_val-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
        print(f"{'='*60}")
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
