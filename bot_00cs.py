#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00cs.py  —  CAN SLIM ENGINE v1.0
Gebaseerd op William O'Neil's CAN SLIM methode.

CAN SLIM criteria (score 0-7):
  C — Current Earnings    : kwartaal EPS groei >= 25% YoY
  A — Annual Earnings     : jaarlijkse EPS groei >= 25% over 3 jaar
  N — New High            : prijs binnen 15% van 52-weekse high
  S — Supply & Demand     : volume stijgt bij koersstijging
  L — Leader              : RS Rating >= 80 vs. universe
  I — Institutional       : institutionele ownership > 0
  M — Market Direction    : aandeel boven MA200

Gebruik:
  python bot_00cs.py live     # live rapport
  python bot_00cs.py backtest # backtest modus
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
MAX_HOLD_DAYS        = 60

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

CS_CFG = {
    "eps_quarterly_growth":  25.0,
    "eps_annual_growth":     25.0,
    "eps_annual_years":      3,
    "max_from_high_pct":     15.0,
    "vol_ma_period":         50,
    "vol_up_days":           10,
    "rs_min":                80,
    "ma_trend":              200,
    "stop_pct":              8.0,
    "atr_period":            14,
    "min_score":             4,
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
# ATR + SIZING
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

def sizing_tekst(ticker, prijs, stop, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    investering = round(entry * aandelen, 2)
    kosten      = round(trade_cost(investering), 2)
    rr          = ((prijs * 1.20 - entry) / (entry - stop)) if (entry - stop) > 0 else 0
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
# RS RATING
# ============================================================

def compute_rs_ratings(df: pd.DataFrame) -> Dict[str, float]:
    perf: Dict[str, float] = {}
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date")
        if len(g) < 252:
            continue
        c_now = safe_float(g["Close"].iloc[-1])
        c_3m  = safe_float(g["Close"].iloc[-63])
        c_12m = safe_float(g["Close"].iloc[-252])
        if math.isnan(c_now) or math.isnan(c_12m) or c_12m <= 0 or c_3m <= 0:
            continue
        perf[ticker] = 0.4 * (c_now - c_3m) / c_3m + 0.6 * (c_now - c_12m) / c_12m

    rs_map: Dict[str, float] = {}
    if not perf:
        return rs_map
    values = sorted(perf.values())
    n = len(values)
    for ticker, p in perf.items():
        rank = sum(1 for v in values if v < p)
        rs_map[ticker] = round((rank / n) * 99, 1)
    return rs_map


# ============================================================
# FUNDAMENTELE DATA
# ============================================================

def get_fundamentals(ticker: str) -> Dict:
    result = {
        "eps_q_growth":    float("nan"),
        "eps_annual_ok":   False,
        "eps_annual_cagr": float("nan"),
        "institutional":   False,
        "inst_pct":        0.0,
    }
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        q_growth = info.get("earningsQuarterlyGrowth")
        if q_growth is not None:
            result["eps_q_growth"] = round(float(q_growth) * 100, 1)
        try:
            income = t.financials
            if income is not None and not income.empty:
                if "Net Income" in income.index:
                    net = income.loc["Net Income"].dropna().sort_index()
                    if len(net) >= CS_CFG["eps_annual_years"]:
                        vals = net.values[-CS_CFG["eps_annual_years"]:]
                        if vals[0] > 0 and vals[-1] > 0:
                            cagr = ((vals[-1] / vals[0]) **
                                    (1 / (CS_CFG["eps_annual_years"] - 1)) - 1) * 100
                            result["eps_annual_cagr"] = round(cagr, 1)
                            grew_each_year = all(
                                vals[i] > vals[i - 1] for i in range(1, len(vals))
                            )
                            result["eps_annual_ok"] = (
                                cagr >= CS_CFG["eps_annual_growth"] and grew_each_year
                            )
        except Exception:
            pass
        inst_pct = info.get("heldPercentInstitutions", 0.0) or 0.0
        result["institutional"] = inst_pct > 0.05
        result["inst_pct"] = round(inst_pct * 100, 1)
    except Exception:
        pass
    return result


# ============================================================
# CAN SLIM SIGNAAL
# ============================================================

@dataclass
class CSSignaal:
    ticker:        str
    price:         float
    score:         int
    score_labels:  List[str]
    c_ok:          bool
    a_ok:          bool
    n_ok:          bool
    s_ok:          bool
    l_ok:          bool
    i_ok:          bool
    m_ok:          bool
    eps_q_growth:  float
    eps_cagr:      float
    rs:            float
    pct_from_high: float
    vol_ratio:     float
    inst_pct:      float
    ma200:         float
    stop:          float
    high52w:       float
    total_score:   float

def analyse_ticker(
    ticker:       str,
    g:            pd.DataFrame,
    rs_ratings:   Dict[str, float],
    fundamentals: Dict,
) -> Optional[CSSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < CS_CFG["ma_trend"] + 10:
            return None

        close  = g["Close"]
        high   = g["High"]
        volume = g["Volume"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        ma200   = safe_float(close.rolling(CS_CFG["ma_trend"]).mean().iloc[-1])
        lb      = min(252, len(close))
        h52w    = float(close.iloc[-lb:].max())
        rs      = rs_ratings.get(ticker, 0.0)

        hl  = high - close.shift()
        lcp = (g["Low"] - close.shift()).abs()
        hcp = (high - close.shift()).abs()
        tr  = pd.concat([high - g["Low"], hcp, lcp], axis=1).max(axis=1)
        stop = current_price * (1 - CS_CFG["stop_pct"] / 100)

        pct_from_high = ((h52w - current_price) / h52w * 100) if h52w > 0 else 100.0

        recent = g.iloc[-CS_CFG["vol_up_days"]:].copy()
        recent["up"] = recent["Close"] > recent["Close"].shift()
        vol_up   = recent.loc[recent["up"] == True,  "Volume"].mean()
        vol_down = recent.loc[recent["up"] == False, "Volume"].mean()
        vol_ratio = (vol_up / vol_down) if (vol_down > 0 and not math.isnan(vol_down)) else 1.0
        if math.isnan(vol_ratio):
            vol_ratio = 1.0

        score  = 0
        labels = []

        def chk(ok: bool, letter: str, ok_msg: str, fail_msg: str) -> bool:
            nonlocal score
            if ok:
                score += 1
                labels.append(f"✓ {letter}: {ok_msg}")
            else:
                labels.append(f"✗ {letter}: {fail_msg}")
            return ok

        eps_q = fundamentals.get("eps_q_growth", float("nan"))
        c_ok  = not math.isnan(eps_q) and eps_q >= CS_CFG["eps_quarterly_growth"]
        chk(c_ok, "C",
            f"kwartaal EPS +{eps_q:.1f}% YoY",
            f"kwartaal EPS {eps_q:.1f}% YoY" if not math.isnan(eps_q) else "geen EPS data")

        a_ok     = fundamentals.get("eps_annual_ok", False)
        eps_cagr = fundamentals.get("eps_annual_cagr", float("nan"))
        chk(a_ok, "A",
            f"EPS CAGR {eps_cagr:.1f}% over {CS_CFG['eps_annual_years']}j",
            f"EPS CAGR {eps_cagr:.1f}%" if not math.isnan(eps_cagr) else "geen jaar EPS data")

        n_ok = pct_from_high <= CS_CFG["max_from_high_pct"]
        chk(n_ok, "N",
            f"{pct_from_high:.1f}% onder 52w high",
            f"{pct_from_high:.1f}% onder 52w high")

        s_ok = vol_ratio >= 1.2
        chk(s_ok, "S",
            f"volume up-days {vol_ratio:.1f}x down-days",
            f"volume up-days {vol_ratio:.1f}x down-days")

        l_ok = rs >= CS_CFG["rs_min"]
        chk(l_ok, "L",
            f"RS={rs:.0f} (>={CS_CFG['rs_min']})",
            f"RS={rs:.0f} (<{CS_CFG['rs_min']})")

        i_ok     = fundamentals.get("institutional", False)
        inst_pct = fundamentals.get("inst_pct", 0.0)
        chk(i_ok, "I",
            f"institutioneel {inst_pct:.1f}%",
            f"institutioneel {inst_pct:.1f}%")

        m_ok = not math.isnan(ma200) and current_price > ma200
        chk(m_ok, "M",
            f"Close > MA200 ({ma200:.2f})",
            f"Close < MA200 ({ma200:.2f})")

        if score < CS_CFG["min_score"]:
            return None

        total_score = (
            score * 10
            + (rs * 0.2 if l_ok else 0)
            + (eps_q * 0.1 if c_ok and not math.isnan(eps_q) else 0)
            + (10 if n_ok and pct_from_high <= 5 else 0)
        )

        return CSSignaal(
            ticker=ticker, price=round(current_price, 2),
            score=score, score_labels=labels,
            c_ok=c_ok, a_ok=a_ok, n_ok=n_ok,
            s_ok=s_ok, l_ok=l_ok, i_ok=i_ok, m_ok=m_ok,
            eps_q_growth=round(eps_q, 1) if not math.isnan(eps_q) else 0.0,
            eps_cagr=round(eps_cagr, 1) if not math.isnan(eps_cagr) else 0.0,
            rs=round(rs, 1), pct_from_high=round(pct_from_high, 1),
            vol_ratio=round(vol_ratio, 2), inst_pct=round(inst_pct, 1),
            ma200=round(ma200, 2) if not math.isnan(ma200) else 0.0,
            stop=round(stop, 2), high52w=round(h52w, 2),
            total_score=round(total_score, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 7) -> str:
    return "█" * score + "░" * (max_score - score) + f" {score}/{max_score}"

def format_bericht(
    exchange_name:    str,
    signalen:         List[CSSignaal],
    portfolio_waarde: float,
) -> Optional[str]:
    if not signalen:
        return None

    nu        = today_str()
    max_score = max(s.score for s in signalen)
    toon      = [s for s in signalen if s.score == max_score]
    top2      = signalen[:2]

    lbl = {7: "⭐ PERFECTE SCORE (7/7)", 6: "🔥 UITSTEKEND (6/7)",
           5: "⚡ STERK (5/7)", 4: "📊 WATCHLIST (4/7)"}.get(max_score, "📊")

    def sig_blok(sigs: List[CSSignaal]) -> str:
        lines = []
        for s in sigs:
            lines.append(
                f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                f"  RS:{s.rs:.0f} | EPS +{s.eps_q_growth:.1f}% | Inst:{s.inst_pct:.1f}%\n"
                f"  {s.pct_from_high:.1f}% onder 52w high | Vol:{s.vol_ratio:.1f}x\n"
                + sizing_tekst(s.ticker, s.price, s.stop, portfolio_waarde)
            )
        return "\n\n".join(lines)

    delen = [
        f"📈 *CAN SLIM — {exchange_name}*",
        f"_{nu} | {len(signalen)} kandidaten_",
        "─────────────────────────────",
        f"🏆 *TOP 2:*",
        sig_blok(top2),
        "─────────────────────────────",
        f"*{lbl}:*",
    ]
    extra = [s for s in toon if s not in top2]
    if extra:
        for s in extra:
            delen.append(f"• `{s.ticker}` {_score_bar(s.score)} | EUR{s.price:.2f} | RS:{s.rs:.0f}")
    else:
        delen.append("_Zie top 2 hierboven_")

    delen.append(f"⚙️ _Stop {CS_CFG['stop_pct']:.0f}% | RS>={CS_CFG['rs_min']} | Risico 5%_")
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"CAN SLIM — LIVE  {today_str()}")
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
    print("Koersdata downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers")
    print("RS ratings berekenen...")
    rs_ratings = compute_rs_ratings(df)

    rs_candidates = [t for t, r in rs_ratings.items() if r >= 70]
    print(f"Fundamentele data ophalen voor {len(rs_candidates)} RS-kandidaten...")
    fundamentals: Dict[str, Dict] = {}
    for i, ticker in enumerate(rs_candidates, 1):
        fundamentals[ticker] = get_fundamentals(ticker)
        if i % 20 == 0:
            print(f"  {i}/{len(rs_candidates)} fundamentals opgehaald...")
        time.sleep(0.3)

    portfolio_waarde = START_CAPITAL
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        signalen: List[CSSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            fund = fundamentals.get(ticker, {})
            sig  = analyse_ticker(ticker, group, rs_ratings, fund)
            if sig:
                signalen.append(sig)
                print(f"  ✓ {ticker}: {sig.score}/7 | RS={sig.rs:.0f} | EPS+{sig.eps_q_growth:.0f}%")

        signalen.sort(key=lambda s: s.total_score, reverse=True)
        print(f"  → {len(signalen)} CAN SLIM kandidaten")

        bericht = format_bericht(ex_name, signalen, portfolio_waarde)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Overgeslagen (geen signalen): {ex_name}")

    # Één samenvattingsmail
    if email_delen:
        send_email(
            subject=f"CAN SLIM rapport {today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"CAN SLIM BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
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
    rs_ratings = compute_rs_ratings(df)
    cash       = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []
    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(scan_date)].copy()

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

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group, rs_ratings, {})
            if not sig or not (sig.n_ok and sig.l_ok and sig.m_ok):
                continue
            entry       = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, _ = bereken_positie(cash, entry, sig.stop)
            if aandelen <= 0:
                continue
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue
            cash -= investering
            positions[ticker] = {
                "entry_date": scan_date, "entry_price": round(entry, 4),
                "size": aandelen, "stop": sig.stop, "score": sig.score,
                "days": 0, "cost": trade_cost(investering),
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
