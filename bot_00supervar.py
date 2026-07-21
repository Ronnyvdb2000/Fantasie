#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00super.py  —  SUPERBOT v1.0
Combineert de beste onderdelen van de bestaande bots tot één dagelijkse
selectie van het sterkste, bovengemiddeld volatiele aandeel per beurs.

Samengesteld uit:
  - Trend Template (Stage 2 uptrend, 7-punts, uit bot_00ms.py)
  - VCP score 0-4 (volatility contraction + pivot breakout, uit bot_00ms.py)
  - RS Rating (relatieve sterkte 3m/12m, uit bot_00ms.py)
  - Volatiliteits-filter (nieuw): alleen aandelen bovengemiddeld volatiel
    t.o.v. hun eigen beursuniversum komen in aanmerking

Selectie:
  Per beurs wordt precies 1 kandidaat gekozen — de hoogste total_score
  onder de aandelen die (a) door de Trend Template + VCP-drempel komen
  én (b) een volatiliteit hebben boven de mediaan van hun beurs.
  Beurzen zonder geldige kandidaat worden overgeslagen (geen bericht).

Output — zelfde bekende wijze als de overige bots:
  - 1 Telegram-bericht per beurs (alleen als er een winnaar is)
  - 1 samenvattende e-mail voor de hele run (Gmail SMTP/App Password)
  - Geen CSV-logging

Beurzen: alle 041-059 waarvoor een tickers_XXXx.txt bestand bestaat.
Beurzen die nog niet zijn aangemaakt (bv. 055-059) worden automatisch
meegenomen zodra het bestand er is — draai zonder aanpassing.

Draaimoment — 2 runs vlak vóór elke market open:
  De analyse gebruikt altijd de laatst afgesloten dagcandle (van gisteren
  op een ochtend-run), dat verandert niet met het tijdstip. Wat wél
  verandert is wanneer je het signaal ontvangt — vlak vóór open, zodat
  je nog kan handelen op de opening:
    - Europa-run  ~06:30 UTC (vóór Londen/Frankfurt/Parijs/etc. open
      rond 07:00-08:00 UTC, seizoensafhankelijk door zomer-/wintertijd)
    - Noord-Amerika-run ~13:15 UTC (vóór Nasdaq/NYSE/Toronto open om
      13:30 UTC EDT / 14:30 UTC EST)
  Let op: cron-tijden zijn vaste UTC-tijden en schuiven dus 1 uur mee
  met de eigen zomer-/wintertijd-overgangen van elke beurs.

Gebruik:
  python bot_00super.py europa       # alleen Europese beurzen
  python bot_00super.py namerika     # alleen Toronto + Nasdaq/NYSE
  python bot_00super.py alle         # alles in 1 run (default)
"""

import os
import sys
import math
import warnings
import datetime as dt
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
RISICO_PCT_PER_TRADE = 0.05
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

BEURS_NAMEN = {
    "041": "Benelux",         "042": "Parijs",       "043": "Frankfurt",
    "044": "Spanje/Portugal", "045": "Londen",       "046": "Milaan",
    "047": "Toronto",         "048": "Nasdaq/NYSE",  "049": "Stockholm",
    "050": "Zurich",          "051": "Warschau",     "052": "Oslo",
    "053": "Kopenhagen",      "054": "Helsinki",     "055": "Beurs 055",
    "056": "Beurs 056",       "057": "Beurs 057",    "058": "Beurs 058",
    "059": "Beurs 059",
}
REEKS_START = 41
REEKS_EINDE = 59

# Regio-indeling voor de 2 open-tijd runs. 047/048 zijn Noord-Amerika,
# de rest (incl. nog niet aangemaakte 055-059) wordt als Europa behandeld
# totdat je een nieuwe beurs toevoegt met een andere regio.
NOORD_AMERIKA_BEURZEN = {"047", "048"}

SUPER_CFG = {
    "ma_fast":             50,
    "ma_mid":              150,
    "ma_slow":             200,
    "max_from_high_pct":   25.0,
    "min_from_low_pct":    30.0,
    "rs_min":              70,
    "vcp_lookback":        60,
    "vol_contraction_pct": 20.0,
    "volume_dry_pct":      30.0,
    "pivot_lookback":      20,
    "pivot_breakout_vol":  1.5,
    "stop_pct":            8.0,
    "min_vcp_score":       2,
    "volpct_lookback":     20,   # dagen voor realized volatility
}


def pad_export(g: str) -> str:
    return f"tickers_{g}x.txt"


# ============================================================
# HULPFUNCTIES (zelfde als overige bots)
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

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


# ============================================================
# POSITIE SIZING
# ============================================================

def bereken_positie(portfolio_waarde, entry_prijs, stop_prijs, risico_pct=RISICO_PCT_PER_TRADE):
    risico_eur   = portfolio_waarde * risico_pct
    stop_afstand = entry_prijs - stop_prijs
    if stop_afstand <= 0:
        return 0, 0.0
    aandelen    = max(1, int(risico_eur / stop_afstand))
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, max_verlies

def sizing_tekst(ticker, prijs, stop, resistance, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    tp          = resistance
    investering = round(entry * aandelen, 2)
    kosten      = round(trade_cost(investering), 2)
    rr          = ((tp - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | TP: EUR{tp:.2f}\n"
        f"  R/R: {rr:.1f}:1 | {aandelen} stuks | EUR{investering:,.2f}\n"
        f"  Max verlies: EUR{max_loss:,.2f} | Kosten: EUR{kosten:.2f}"
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
    parts = []
    for ticker, group in df.groupby("Ticker", sort=False):
        g     = group.copy()
        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]
        vol   = g["Volume"]

        g["MA50"]        = close.rolling(SUPER_CFG["ma_fast"]).mean()
        g["MA150"]       = close.rolling(SUPER_CFG["ma_mid"]).mean()
        g["MA200"]       = close.rolling(SUPER_CFG["ma_slow"]).mean()
        g["MA200_slope"] = g["MA200"].diff(20)

        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"]   = _wilder_smooth(tr, 14)
        g["VolMA20"] = vol.rolling(20).mean()
        g["High52w"] = close.rolling(252).max()
        g["Low52w"]  = close.rolling(252).min()
        g["RS"]      = universe_rs.get(ticker, 50.0)

        # Volatiliteit: geannualiseerde realized volatility (%) over laatste N dagen
        dagrendement = close.pct_change()
        g["VolPct"] = (
            dagrendement.rolling(SUPER_CFG["volpct_lookback"], min_periods=10).std()
            * math.sqrt(252) * 100
        )
        g["Ticker"] = ticker
        parts.append(g)

    if not parts:
        return df
    return pd.concat(parts).sort_values(["Ticker", "Date"]).reset_index(drop=True)

def compute_rs_ratings(df: pd.DataFrame) -> Dict[str, float]:
    rs_map: Dict[str, float] = {}
    perf:   Dict[str, float] = {}
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

    if not perf:
        return rs_map
    values = sorted(perf.values())
    n = len(values)
    for ticker, p in perf.items():
        rank = sum(1 for v in values if v < p)
        rs_map[ticker] = round((rank / n) * 99, 1)
    return rs_map


# ============================================================
# TREND TEMPLATE
# ============================================================

def check_trend_template(row: pd.Series) -> Tuple[bool, List[str]]:
    close  = safe_float(row.get("Close"))
    ma50   = safe_float(row.get("MA50"))
    ma150  = safe_float(row.get("MA150"))
    ma200  = safe_float(row.get("MA200"))
    ma200s = safe_float(row.get("MA200_slope"))
    h52    = safe_float(row.get("High52w"))
    l52    = safe_float(row.get("Low52w"))
    rs     = safe_float(row.get("RS"), 0.0)

    checks, passed = [], True

    def chk(condition, ok_msg, fail_msg):
        nonlocal passed
        if condition:
            checks.append(f"✓ {ok_msg}")
        else:
            checks.append(f"✗ {fail_msg}")
            passed = False

    chk(not math.isnan(close) and not math.isnan(ma150) and not math.isnan(ma200)
        and close > ma150 and close > ma200,
        "Close>MA150/MA200", "Close niet boven MA150/MA200")
    chk(not math.isnan(ma150) and not math.isnan(ma200) and ma150 > ma200,
        "MA150>MA200", "MA150 niet boven MA200")
    chk(not math.isnan(ma200s) and ma200s > 0,
        "MA200 stijgt", "MA200 daalt/vlak")
    chk(not math.isnan(ma50) and not math.isnan(ma150) and not math.isnan(ma200)
        and ma50 > ma150 and ma50 > ma200,
        "MA50>MA150/MA200", "MA50 niet boven MA150/MA200")
    chk(not math.isnan(close) and not math.isnan(ma50) and close > ma50,
        "Close>MA50", "Close niet boven MA50")

    if not math.isnan(h52) and h52 > 0:
        pct_from_high = (h52 - close) / h52 * 100
        chk(pct_from_high <= SUPER_CFG["max_from_high_pct"],
            f"{pct_from_high:.1f}% onder 52w high", f"{pct_from_high:.1f}% onder 52w high (>25%)")
    else:
        checks.append("✗ geen 52w high data"); passed = False

    if not math.isnan(l52) and l52 > 0:
        pct_from_low = (close - l52) / l52 * 100
        chk(pct_from_low >= SUPER_CFG["min_from_low_pct"],
            f"{pct_from_low:.1f}% boven 52w low", f"{pct_from_low:.1f}% boven 52w low (<30%)")
    else:
        checks.append("✗ geen 52w low data"); passed = False

    chk(rs >= SUPER_CFG["rs_min"], f"RS={rs:.0f} (>=70)", f"RS={rs:.0f} (<70)")
    return passed, checks


# ============================================================
# VCP DETECTIE
# ============================================================

def detect_vcp(g: pd.DataFrame) -> Tuple[int, List[str], float, float]:
    score, details = 0, []
    close, volume, atr, vol_ma = g["Close"], g["Volume"], g["ATR14"], g["VolMA20"]

    if len(g) < SUPER_CFG["vcp_lookback"] + 10:
        return 0, ["onvoldoende data"], float("nan"), float("nan")

    recent      = g.iloc[-SUPER_CFG["vcp_lookback"]:]
    very_recent = g.iloc[-SUPER_CFG["pivot_lookback"]:]

    atr_now   = safe_float(atr.iloc[-1])
    atr_start = safe_float(atr.iloc[-SUPER_CFG["vcp_lookback"]])
    if not math.isnan(atr_now) and not math.isnan(atr_start) and atr_start > 0:
        contraction = (atr_start - atr_now) / atr_start * 100
        if contraction >= SUPER_CFG["vol_contraction_pct"]:
            score += 1; details.append(f"✓ ATR contractie {contraction:.1f}%")
        else:
            details.append(f"✗ ATR contractie {contraction:.1f}%")
    else:
        details.append("✗ ATR data onvoldoende")

    vol_now, vol_mean = safe_float(volume.iloc[-1]), safe_float(vol_ma.iloc[-1])
    if not math.isnan(vol_now) and not math.isnan(vol_mean) and vol_mean > 0:
        vol_ratio = vol_now / vol_mean * 100
        if vol_ratio <= (100 - SUPER_CFG["volume_dry_pct"]):
            score += 1; details.append(f"✓ Volume droogvalt {vol_ratio:.0f}% van gem.")
        else:
            details.append(f"✗ Volume {vol_ratio:.0f}% van gem.")
    else:
        details.append("✗ Volume data onvoldoende")

    n = len(recent); third = n // 3
    if third >= 5:
        r1, r2, r3 = recent.iloc[:third]["Close"], recent.iloc[third:2*third]["Close"], recent.iloc[2*third:]["Close"]
        range1, range2, range3 = float(r1.max()-r1.min()), float(r2.max()-r2.min()), float(r3.max()-r3.min())
        if range1 > 0 and range2 < range1 and range3 < range2:
            score += 1; details.append(f"✓ Pullbacks krimpen: {range1:.2f}->{range2:.2f}->{range3:.2f}")
        else:
            details.append(f"✗ Pullbacks krimpen niet")
    else:
        details.append("✗ Onvoldoende data voor pullback analyse")

    pivot_high      = float(very_recent["Close"].iloc[:-1].max())
    current         = safe_float(close.iloc[-1])
    vol_recent_mean = safe_float(vol_ma.iloc[-SUPER_CFG["pivot_lookback"]])
    vol_today       = safe_float(volume.iloc[-1])

    breakout = (
        not math.isnan(current) and current > pivot_high
        and not math.isnan(vol_today) and not math.isnan(vol_recent_mean)
        and vol_recent_mean > 0
        and vol_today >= vol_recent_mean * SUPER_CFG["pivot_breakout_vol"]
    )
    if breakout:
        score += 1; details.append(f"✓ Pivot breakout boven {pivot_high:.2f}")
    else:
        details.append(f"✗ Geen pivot breakout (pivot={pivot_high:.2f})")

    stop_prijs  = max(float(very_recent["Close"].min()), current * (1 - SUPER_CFG["stop_pct"] / 100))
    pivot_prijs = float(recent["Close"].max())
    return score, details, pivot_prijs, stop_prijs


# ============================================================
# SIGNAAL
# ============================================================

@dataclass
class SuperSignaal:
    ticker:        str
    price:         float
    vcp_score:     int
    pivot:         float
    stop:          float
    rs:            float
    pct_from_high: float
    vol_pct:       float
    total_score:   float

def analyse_ticker(ticker: str, g: pd.DataFrame, vol_drempel: float) -> Optional[SuperSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < SUPER_CFG["ma_slow"] + 10:
            return None

        last = g.iloc[-1]
        current_price = safe_float(last.get("Close"))
        if current_price <= 0 or math.isnan(current_price):
            return None

        trend_passed, _ = check_trend_template(last)
        if not trend_passed:
            return None

        vcp_score, _, pivot, stop = detect_vcp(g)
        if vcp_score < SUPER_CFG["min_vcp_score"]:
            return None

        vol_pct = safe_float(last.get("VolPct"))
        if math.isnan(vol_pct) or vol_pct < vol_drempel:
            return None  # niet bovengemiddeld volatiel t.o.v. eigen beurs

        h52 = safe_float(last.get("High52w"))
        rs  = safe_float(last.get("RS"), 0.0)
        pct_from_high = ((h52 - current_price) / h52 * 100) if h52 > 0 else 0.0

        total_score = (
            rs * 0.4
            + vcp_score * 15
            + (25 - pct_from_high) * 0.5
            + min(vol_pct, 100) * 0.3   # volatiliteits-bonus, afgetopt
        )

        return SuperSignaal(
            ticker=ticker, price=round(current_price, 2), vcp_score=vcp_score,
            pivot=round(pivot, 2), stop=round(stop, 2), rs=round(rs, 1),
            pct_from_high=round(pct_from_high, 1), vol_pct=round(vol_pct, 1),
            total_score=round(total_score, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# BERICHT
# ============================================================

def _vcp_bar(score: int) -> str:
    return "█" * score + "░" * (4 - score) + f" {score}/4"

def format_bericht(beurs_naam: str, s: SuperSignaal, portfolio_waarde: float) -> str:
    nu = today_str()
    rr = ((s.pivot - s.price) / (s.price - s.stop)) if (s.price - s.stop) > 0 else 0
    return "\n\n".join([
        f"🏆 *SUPERBOT — {beurs_naam}*",
        f"_{nu} | Trend Template + VCP + Volatiliteitsfilter_",
        "─────────────────────────────",
        f"• `{s.ticker}` VCP:{_vcp_bar(s.vcp_score)} RS:{s.rs:.0f} Vol:{s.vol_pct:.0f}%\n"
        f"  EUR{s.price:.2f} {_yahoo_link(s.ticker)}\n"
        f"  {s.pct_from_high:.1f}% onder 52w high | R/R:{rr:.1f}:1 | Score:{s.total_score:.1f}\n"
        + sizing_tekst(s.ticker, s.price, s.stop, s.pivot, portfolio_waarde),
        f"⚙️ _Stop max {SUPER_CFG['stop_pct']:.0f}% | RS>={SUPER_CFG['rs_min']} | "
        f"VCP>={SUPER_CFG['min_vcp_score']}/4 | Vol boven beursmediaan_",
    ])


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine(regio: str = "alle"):
    print(f"{'='*60}\nSUPERBOT — LIVE  {today_str()}  [regio: {regio}]\n{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for nr in range(REEKS_START, REEKS_EINDE + 1):
        g = f"0{nr}"
        is_na = g in NOORD_AMERIKA_BEURZEN
        if regio == "europa" and is_na:
            continue
        if regio == "namerika" and not is_na:
            continue
        pad = pad_export(g)
        if not os.path.exists(pad):
            continue
        tlist = load_tickers_from_file(pad)
        if tlist:
            naam = BEURS_NAMEN.get(g, f"Beurs {g}")
            exchange_tickers[f"{g} {naam}"] = tlist
            all_tickers.extend(tlist)
            print(f"  {g} {naam}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden voor 041-059.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print("Data downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers")
    rs_ratings = compute_rs_ratings(df)
    df = add_indicators(df, rs_ratings)

    portfolio_waarde = START_CAPITAL
    email_delen: List[str] = []
    geen_kandidaat: List[str] = []

    for ex_naam, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_naam} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()
        if df_ex.empty:
            geen_kandidaat.append(ex_naam)
            continue

        # Volatiliteits-mediaan van de hele beurs (laatste geldige waarde per ticker)
        laatste = df_ex.sort_values("Date").groupby("Ticker", sort=False).tail(1)
        vol_waarden = laatste["VolPct"].dropna()
        print(f"  Volatiliteitsdata: {len(vol_waarden)}/{len(laatste)} tickers geldig")
        if vol_waarden.empty:
            print(f"  → Geen volatiliteitsdata beschikbaar voor {ex_naam} (te weinig koersdata) — overgeslagen")
            geen_kandidaat.append(ex_naam)
            continue
        vol_drempel = float(vol_waarden.median())

        kandidaten: List[SuperSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group, vol_drempel)
            if sig:
                kandidaten.append(sig)

        if not kandidaten:
            print(f"  → Geen kandidaat boven beursmediaan volatiliteit ({vol_drempel:.0f}%)")
            geen_kandidaat.append(ex_naam)
            continue

        kandidaten.sort(key=lambda s: s.total_score, reverse=True)
        winnaar = kandidaten[0]
        print(f"  → Winnaar: {winnaar.ticker} (score {winnaar.total_score}, "
              f"vol {winnaar.vol_pct}% vs mediaan {vol_drempel:.0f}%)")

        bericht = format_bericht(ex_naam, winnaar, portfolio_waarde)
        send_telegram_message(bericht)
        email_delen.append(bericht)

    if geen_kandidaat:
        print(f"\nGeen kandidaat voor: {', '.join(geen_kandidaat)}")

    if email_delen:
        send_email(
            f"Superbot rapport {today_str()} — {len(email_delen)} picks",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )
    else:
        geen_resultaat_tekst = (
            f"🔍 *SUPERBOT — {today_str()}*\n"
            f"_Regio: {regio}_\n\n"
            f"Geen enkel aandeel voldeed vandaag aan alle eisen "
            f"(Trend Template + VCP + bovengemiddelde volatiliteit) "
            f"op {len(exchange_tickers)} gescande beurzen:\n"
            f"{', '.join(geen_kandidaat) if geen_kandidaat else '—'}"
        )
        send_telegram_message(geen_resultaat_tekst)
        send_email(f"Superbot rapport {today_str()} — geen kandidaten", geen_resultaat_tekst)
        print("\nGeen enkele beurs leverde een winnaar op vandaag — melding verstuurd.")

    print(f"\n{'='*60}\nKlaar.")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "alle"
    if arg not in ("europa", "namerika", "alle"):
        print(f"[WARN] Onbekend argument '{arg}', gebruik 'alle'.")
        arg = "alle"
    run_live_engine(arg)
