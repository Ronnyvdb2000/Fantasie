#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL ENGINE v5.0 - ATR Sizing + Slippage + Manuele Executie

Verbeteringen t.o.v. v4.0:
- ATR-gebaseerde positiesizing: risico = 10% van portfolio per trade
- Slippage gesimuleerd: entry op Open van T+1 (realistisch)
- Sizing-advies in elk Telegram signaal (aandelen, stop, max verlies)
- Backtest gebruikt T+1 open als entry (geen lookahead bias)
- Slippage van 0.1% per kant in backtest
- days_open fix: correct bijgehouden in live engine
- Sharpe per strategie in backtest statistieken
- Alle v4.0 functionaliteit behouden

Gebruik:
    python bot_global_v5.py              -> live signaal-engine
    python bot_global_v5.py backtest     -> volledige backtest
    python bot_global_v5.py apply        -> verwerk commands.txt
"""

import os
import sys
import math
import csv
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL        = 50_000.0
MAX_POSITIONS        = 10
MIN_CASH_RATIO       = 0.10      # minimum 10% cash buffer
RISICO_PCT_PER_TRADE = 0.10      # 10% portfolio risico per trade (ATR sizing)
ATR_STOP_MULT        = 2.0       # stop = entry - 2×ATR
ATR_TP_MULT_BASIS    = 3.0       # basis TP multiplier (overschreven per strategie)
SLIPPAGE_PCT         = 0.001     # 0.1% slippage per kant

TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10

MAX_HOLD_DAYS        = 20

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LIVE_TRADES_FILE    = "trades_live.csv"
LIVE_POSITIONS_FILE = "positions_live.csv"
LIVE_PORTFOLIO_FILE = "portfolio_live.csv"

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

FALLBACK_TICKERS = {
    "048 Nasdaq/NYSE": [
        "AAPL","MSFT","NVDA","META","GOOGL",
        "AMZN","TSLA","AMD","INTC","NFLX",
        "ORCL","CRM","ADBE","QCOM","TXN",
    ],
    "041 Benelux": [
        "ASML","AD.AS","INGA.AS","PHIA.AS","UNA.AS",
        "ABN.AS","NN.AS","RAND.AS","WKL.AS","BESI.AS",
        "AKZA.AS","HEIA.AS","IMCD.AS","DSM.AS","AGN.AS",
    ],
}

BACKTEST_START = "2019-01-01"
BACKTEST_END   = dt.date.today().isoformat()

# Stop/TP multipliers per strategie
STRAT_CONFIG = {
    "Traag":       {"sl_mult": 2.0, "tp_mult": 4.0},
    "Snel":        {"sl_mult": 2.0, "tp_mult": 3.0},
    "Hyper Trend": {"sl_mult": 2.5, "tp_mult": 5.0},
    "Hyper Scalp": {"sl_mult": 1.5, "tp_mult": 2.5},
    "MRA Snel":    {"sl_mult": 2.0, "tp_mult": 3.0},
    "MRA Traag":   {"sl_mult": 2.5, "tp_mult": 4.0},
}


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


def format_price(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "n/a"
    return f"{val:.2f}"


def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(",", "\n").replace("$", "")
    result = []
    for line in raw.splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            result.append(t)
    return sorted(list(set(result)))


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram niet geconfigureerd.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram fout: {e}")
        print(text)


def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


# ============================================================
# ATR SIZING  ← NIEUW in v5.0
# ============================================================

def bereken_atr_positie(
    portfolio_waarde: float,
    entry_prijs:      float,
    atr:              float,
    sl_mult:          float = ATR_STOP_MULT,
    risico_pct:       float = RISICO_PCT_PER_TRADE,
) -> Tuple[int, float, float]:
    """
    Berekent positiegrootte op basis van ATR-risico.

    Formule:
        risico_eur  = portfolio × risico_pct
        stop_afstand = sl_mult × ATR
        aandelen    = risico_eur / stop_afstand

    Returns:
        (aandelen, stop_loss_prijs, max_verlies_eur)
    """
    risico_eur    = portfolio_waarde * risico_pct
    stop_afstand  = sl_mult * atr
    if stop_afstand <= 0 or entry_prijs <= 0:
        return 0, entry_prijs, 0.0
    aandelen      = max(1, int(risico_eur / stop_afstand))
    stop_loss     = entry_prijs - stop_afstand
    max_verlies   = round(stop_afstand * aandelen, 2)
    return aandelen, stop_loss, max_verlies


def sizing_tekst(
    ticker:           str,
    richting:         str,
    prijs:            float,
    atr:              float,
    portfolio_waarde: float,
    sl_mult:          float,
    tp_mult:          float,
) -> str:
    """
    Genereert sizing-advies tekst voor Telegram.
    Gebruikt T+1 open schatting (prijs + slippage).
    """
    entry       = prijs * (1 + SLIPPAGE_PCT)   # geschatte open T+1
    aandelen, stop, max_loss = bereken_atr_positie(
        portfolio_waarde, entry, atr, sl_mult
    )
    tp          = entry + tp_mult * atr
    investering = round(entry * aandelen, 2)
    slip_est    = round(entry * SLIPPAGE_PCT * aandelen * 2, 2)
    kosten      = round(trade_cost(investering), 2)

    return (
        f"  📐 *Sizing ({richting}):*\n"
        f"  Entry geschat : EUR{entry:.2f} (+0.1% slip)\n"
        f"  Stop-Loss     : EUR{stop:.2f}  ({sl_mult}×ATR)\n"
        f"  Take-Profit   : EUR{tp:.2f}  ({tp_mult}×ATR)\n"
        f"  ATR(14)       : EUR{atr:.4f}\n"
        f"  Aandelen      : {aandelen} stuks\n"
        f"  Investering   : EUR{investering:,.2f}\n"
        f"  Max verlies   : EUR{max_loss:,.2f}  (10% portfolio)\n"
        f"  Slippage est. : EUR{slip_est:.2f}\n"
        f"  Transactiekosten: EUR{kosten:.2f}"
    )


# ============================================================
# DATA & INDICATOREN
# ============================================================

def download_history(
    tickers: List[str],
    start:   Optional[str] = None,
    end:     Optional[str] = None,
    period:  Optional[str] = "5y",
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    kwargs: Dict = dict(
        tickers=tickers,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if start and end:
        kwargs["start"] = start
        kwargs["end"]   = end
    else:
        kwargs["period"] = period

    frames = []

    def _normalise(df_raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        df = df_raw.dropna(how="all").copy()
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
        df["Ticker"] = ticker
        return df

    try:
        data = yf.download(**kwargs)
    except Exception as e:
        print(f"[WARN] Batch download mislukt ({e}), probeer 1-voor-1...")
        data = pd.DataFrame()

    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            available = data.columns.get_level_values(1).unique()
            for t in tickers:
                if t not in available:
                    print(f"[WARN] {t}: geen data in batch (mogelijk delisted), overgeslagen.")
                    continue
                try:
                    raw  = data.xs(t, axis=1, level=1).copy()
                    norm = _normalise(raw, t)
                    if norm is not None:
                        frames.append(norm)
                except Exception as e:
                    print(f"[WARN] {t}: fout bij verwerken ({e}), overgeslagen.")
        else:
            norm = _normalise(data, tickers[0])
            if norm is not None:
                frames.append(norm)

    if not frames:
        print(f"[INFO] Probeer {len(tickers)} tickers 1-voor-1...")
        for t in tickers:
            try:
                kw = dict(tickers=t, auto_adjust=True, progress=False)
                if start and end:
                    kw["start"] = start
                    kw["end"]   = end
                else:
                    kw["period"] = period
                raw = yf.download(**kw)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                norm = _normalise(raw, t)
                if norm is not None:
                    frames.append(norm)
                else:
                    print(f"[WARN] {t}: geen data, overgeslagen.")
                time.sleep(0.2)
            except Exception as e:
                print(f"[WARN] {t}: download mislukt ({e}), overgeslagen.")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    # ── Voeg Next_Open toe voor T+1 entry simulatie ← NIEUW v5.0
    df.sort_values(["Ticker", "Date"], inplace=True)
    df["Next_Open"] = df.groupby("Ticker")["Open"].shift(-1)

    df.reset_index(drop=True, inplace=True)
    return df


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.dropna()
    if len(valid) < period:
        return result
    first_idx         = valid.index[period - 1]
    result[first_idx] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        idx         = valid.index[i]
        prev_idx    = valid.index[i - 1]
        result[idx] = result[prev_idx] * (period - 1) / period + valid.iloc[i] / period
    return result


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        ticker_val = group["Ticker"].iloc[0] if "Ticker" in group.columns else None
        next_open  = group["Next_Open"].values if "Next_Open" in group.columns else None
        g = group.drop(columns=["Ticker", "Next_Open"], errors="ignore").copy().reset_index(drop=True)
        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]

        g["MA20"]  = close.rolling(20).mean()
        g["MA50"]  = close.rolling(50).mean()
        g["MA200"] = close.rolling(200).mean()

        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = _wilder_smooth(gain, 14)
        avg_loss = _wilder_smooth(loss, 14)
        rs       = avg_gain / (avg_loss + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"] = _wilder_smooth(tr, 14)

        g["IBS"] = (close - low) / (high - low + 1e-9)

        up_move   = high.diff()
        down_move = (-low.diff())
        plus_dm   = np.where((up_move > down_move)  & (up_move > 0),   up_move,   0.0)
        minus_dm  = np.where((down_move > up_move)  & (down_move > 0), down_move, 0.0)
        s_plus_dm  = _wilder_smooth(pd.Series(plus_dm,  index=g.index), 14)
        s_minus_dm = _wilder_smooth(pd.Series(minus_dm, index=g.index), 14)
        s_tr       = _wilder_smooth(tr, 14)
        plus_di    = 100 * s_plus_dm  / (s_tr + 1e-9)
        minus_di   = 100 * s_minus_dm / (s_tr + 1e-9)
        dx         = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        g["ADX14"] = _wilder_smooth(dx, 14)

        if ticker_val is not None:
            g["Ticker"] = ticker_val
        if next_open is not None:
            g["Next_Open"] = next_open
        return g

    return df.groupby("Ticker", group_keys=False).apply(_calc)


# ============================================================
# STRATEGIE-SIGNALEN
# ============================================================

@dataclass
class Signal:
    ticker:          str
    date:            dt.date
    strategy:        str
    direction:       str
    reason:          str
    price:           float
    atr:             float        = 0.0   # ← NIEUW v5.0
    sl:              Optional[float] = None
    tp:              Optional[float] = None
    rr_ratio:        float           = 0.0
    next_open:       Optional[float] = None   # ← NIEUW v5.0 T+1 entry


def _calc_rr(price: float, sl: Optional[float], tp: Optional[float]) -> float:
    if sl is None or tp is None:
        return 0.0
    risk   = abs(price - sl)
    reward = abs(tp - price)
    return (reward / risk) if risk > 1e-9 else 0.0


def generate_signals_for_day(df: pd.DataFrame, date: dt.date) -> List[Signal]:
    signals: List[Signal] = []
    day_df = df[df["Date"] == pd.Timestamp(date)].copy()
    if day_df.empty:
        return signals

    for _, row in day_df.iterrows():
        t         = row["Ticker"]
        close     = safe_float(row.get("Close"))
        ma20      = safe_float(row.get("MA20"))
        ma50      = safe_float(row.get("MA50"))
        ma200     = safe_float(row.get("MA200"))
        rsi       = safe_float(row.get("RSI14"))
        ibs       = safe_float(row.get("IBS"))
        atr       = safe_float(row.get("ATR14"))
        adx       = safe_float(row.get("ADX14"))
        next_open = safe_float(row.get("Next_Open"), default=close)

        if math.isnan(close) or close <= 0 or math.isnan(atr) or atr <= 0:
            continue

        def make(strategy: str, reason: str) -> Signal:
            cfg    = STRAT_CONFIG.get(strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
            sl     = close - cfg["sl_mult"] * atr
            tp     = close + cfg["tp_mult"] * atr
            return Signal(
                ticker=t, date=date, strategy=strategy,
                direction="BUY", reason=reason, price=close,
                atr=atr, sl=sl, tp=tp,
                rr_ratio=_calc_rr(close, sl, tp),
                next_open=next_open,
            )

        if not math.isnan(ma50) and not math.isnan(ma200) and not math.isnan(adx):
            if ma50 > ma200 and close > ma50 and adx > 15:
                signals.append(make("Traag", "MA50>MA200 & Close>MA50 & ADX>15"))

        if not math.isnan(ma20) and not math.isnan(ma50) and not math.isnan(adx):
            if ma20 > ma50 and close > ma20 and adx > 15:
                signals.append(make("Snel", "MA20>MA50 & Close>MA20 & ADX>15"))

        if not math.isnan(ma50) and not math.isnan(ma200) and not math.isnan(adx) and not math.isnan(rsi):
            if ma50 > ma200 and close > ma50 and adx > 20 and rsi > 55:
                signals.append(make("Hyper Trend", "ADX>20 & RSI>55 & MA50>MA200"))

        if not math.isnan(rsi) and not math.isnan(ibs):
            if rsi < 30 and ibs < 0.2:
                signals.append(make("Hyper Scalp", "RSI<30 & IBS<0.2"))

        if not math.isnan(rsi) and not math.isnan(ibs):
            if rsi < 35 and ibs < 0.3:
                signals.append(make("MRA Snel", "RSI<35 & IBS<0.3"))

        if not math.isnan(rsi) and not math.isnan(ibs):
            if rsi < 40 and ibs < 0.4:
                signals.append(make("MRA Traag", "RSI<40 & IBS<0.4"))

    signals.sort(key=lambda s: s.rr_ratio, reverse=True)
    return signals


# ============================================================
# LIVE PORTEFEUILLE
# ============================================================

@dataclass
class LivePosition:
    ticker:      str
    strategy:    str
    entry_date:  str
    entry_price: float
    size:        int
    cost:        float
    sl:          Optional[float]
    tp:          Optional[float]
    atr:         float = 0.0
    days_open:   int   = 0


class LivePortfolio:
    def __init__(self, start_capital: float):
        self.cash       = start_capital
        self.positions: Dict[str, LivePosition] = {}
        self.load_state()

    def load_state(self):
        if os.path.exists(LIVE_PORTFOLIO_FILE):
            df = pd.read_csv(LIVE_PORTFOLIO_FILE)
            if not df.empty:
                self.cash = float(df.iloc[-1]["cash"])
        if os.path.exists(LIVE_POSITIONS_FILE):
            dfp = pd.read_csv(LIVE_POSITIONS_FILE)
            for _, r in dfp.iterrows():
                self.positions[r["ticker"]] = LivePosition(
                    ticker      = r["ticker"],
                    strategy    = r["strategy"],
                    entry_date  = r["entry_date"],
                    entry_price = float(r["entry_price"]),
                    size        = int(r["size"]),
                    cost        = float(r["cost"]),
                    sl          = float(r["sl"])  if not pd.isna(r["sl"])  else None,
                    tp          = float(r["tp"])  if not pd.isna(r["tp"])  else None,
                    atr         = float(r.get("atr", 0.0)),
                    days_open   = int(r.get("days_open", 0)),
                )

    def save_state(self, date: str, prices: Dict[str, float]):
        ensure_csv_header(LIVE_POSITIONS_FILE,
            ["ticker","strategy","entry_date","entry_price","size","cost","sl","tp","atr","days_open"])
        with open(LIVE_POSITIONS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker","strategy","entry_date","entry_price","size","cost","sl","tp","atr","days_open"])
            for p in self.positions.values():
                w.writerow([
                    p.ticker, p.strategy, p.entry_date, p.entry_price,
                    p.size, p.cost,
                    p.sl if p.sl is not None else "",
                    p.tp if p.tp is not None else "",
                    p.atr, p.days_open,
                ])
        ensure_csv_header(LIVE_PORTFOLIO_FILE, ["date","cash","positions_value","total"])
        pos_val = sum(prices.get(t, p.entry_price) * p.size for t, p in self.positions.items())
        with open(LIVE_PORTFOLIO_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([date, self.cash, pos_val, self.cash + pos_val])

    def current_total_value(self, prices: Dict[str, float]) -> float:
        return self.cash + sum(
            prices.get(t, p.entry_price) * p.size for t, p in self.positions.items()
        )

    def can_open_new_position(self, prices: Dict[str, float], investering: float) -> bool:
        if len(self.positions) >= MAX_POSITIONS:
            return False
        total    = self.current_total_value(prices)
        min_cash = total * MIN_CASH_RATIO
        return (self.cash - investering) >= min_cash and investering <= self.cash

    def open_position(self, sig: Signal, prices: Dict[str, float]) -> Optional[LivePosition]:
        """
        v5.0: entry op next_open met slippage, ATR-gebaseerde sizing.
        """
        if sig.ticker in self.positions:
            return None
        total        = self.current_total_value(prices)
        cfg          = STRAT_CONFIG.get(sig.strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
        entry_prijs  = (sig.next_open or sig.price) * (1 + SLIPPAGE_PCT)
        aandelen, stop, max_loss = bereken_atr_positie(
            total, entry_prijs, sig.atr, cfg["sl_mult"]
        )
        if aandelen <= 0:
            return None
        investering = entry_prijs * aandelen
        cost        = trade_cost(investering)
        if not self.can_open_new_position(prices, investering + cost):
            return None

        tp = entry_prijs + cfg["tp_mult"] * sig.atr

        self.cash -= investering + cost
        p = LivePosition(
            ticker=sig.ticker, strategy=sig.strategy,
            entry_date=sig.date.isoformat(), entry_price=round(entry_prijs, 4),
            size=aandelen, cost=cost, sl=round(stop, 4), tp=round(tp, 4),
            atr=sig.atr, days_open=0,
        )
        self.positions[sig.ticker] = p
        self.log_trade(
            sig.date.isoformat(), sig.ticker, sig.strategy,
            "BUY", entry_prijs, aandelen, cost, 0.0, 0.0, 0.0,
        )
        return p

    def close_position(self, ticker: str, date: str, exit_price: float, reason: str):
        if ticker not in self.positions:
            return
        p           = self.positions[ticker]
        exit_slip   = exit_price * (1 - SLIPPAGE_PCT)   # slippage bij verkoop
        gross       = exit_slip * p.size
        cost        = trade_cost(gross)
        pnl         = gross - cost - (p.entry_price * p.size + p.cost)
        tax         = pnl * TAX_RATE if pnl > 0 else 0.0
        self.cash  += gross - cost - tax
        self.log_trade(date, ticker, p.strategy, "SELL",
                       exit_slip, p.size, cost, pnl, tax, pnl - tax, reason)
        del self.positions[ticker]

    def log_trade(self, date, ticker, strategy, side, price, size,
                  cost, pnl, tax, net, reason=""):
        ensure_csv_header(LIVE_TRADES_FILE,
            ["date","ticker","strategy","side","price","size","cost","pnl","tax","net","reason"])
        with open(LIVE_TRADES_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [date, ticker, strategy, side, price, size, cost, pnl, tax, net, reason]
            )


# ============================================================
# EXIT ENGINE
# ============================================================

def generate_exit_signals(
    portfolio: LivePortfolio,
    df:        pd.DataFrame,
    date:      dt.date,
) -> List[Signal]:
    signals: List[Signal] = []
    day_df  = df[df["Date"] == pd.Timestamp(date)].copy()
    if day_df.empty:
        return signals
    day_map = {row["Ticker"]: row for _, row in day_df.iterrows()}

    for t, p in list(portfolio.positions.items()):
        if t not in day_map:
            continue
        row   = day_map[t]
        close = safe_float(row.get("Close"))
        ma20  = safe_float(row.get("MA20"))
        rsi   = safe_float(row.get("RSI14"))
        if math.isnan(close):
            continue

        reason: Optional[str] = None
        if p.sl is not None and close <= p.sl:
            reason = f"Stoploss geraakt (SL={p.sl:.2f})"
        elif p.tp is not None and close >= p.tp:
            reason = f"Take Profit geraakt (TP={p.tp:.2f})"
        elif not math.isnan(ma20) and close < ma20 and p.strategy in ("Traag", "Snel", "Hyper Trend"):
            reason = f"Trend exit: Close<MA20 ({close:.2f}<{ma20:.2f})"
        elif not math.isnan(rsi) and rsi > 70 and p.strategy in ("MRA Snel", "MRA Traag", "Hyper Scalp"):
            reason = f"RSI exit: RSI>70 ({rsi:.1f})"
        elif p.days_open >= MAX_HOLD_DAYS:
            reason = f"Time exit: {p.days_open} dagen open"

        if reason:
            signals.append(Signal(
                ticker=t, date=date, strategy=p.strategy,
                direction="SELL", reason=reason, price=close,
                atr=p.atr,
            ))

    return signals


# ============================================================
# BACKTEST ENGINE  ← v5.0: T+1 open entry + slippage
# ============================================================

@dataclass
class BTPosition:
    ticker:      str
    strategy:    str
    entry_date:  dt.date
    entry_price: float
    size:        int
    cost:        float
    sl:          Optional[float]
    tp:          Optional[float]
    atr:         float = 0.0
    days_open:   int   = 0


class BacktestPortfolio:
    def __init__(self, start_capital: float):
        self.cash             = start_capital
        self.positions:       Dict[str, BTPosition] = {}
        self.closed_trades:   List[Dict]            = []
        self.daily_snapshots: List[Dict]            = []

    def current_total_value(self, prices: Dict[str, float]) -> float:
        return self.cash + sum(
            prices.get(t, p.entry_price) * p.size for t, p in self.positions.items()
        )

    def can_open(self, prices: Dict[str, float], investering: float) -> bool:
        if len(self.positions) >= MAX_POSITIONS:
            return False
        total    = self.current_total_value(prices)
        min_cash = total * MIN_CASH_RATIO
        return (self.cash - investering) >= min_cash and investering <= self.cash

    def open_position(self, sig: Signal, prices: Dict[str, float]) -> bool:
        """
        v5.0: entry = next_open × (1 + slippage), ATR-sizing.
        """
        if sig.ticker in self.positions:
            return False
        total       = self.current_total_value(prices)
        cfg         = STRAT_CONFIG.get(sig.strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
        entry       = (sig.next_open or sig.price) * (1 + SLIPPAGE_PCT)
        aandelen, stop, _ = bereken_atr_positie(total, entry, sig.atr, cfg["sl_mult"])
        if aandelen <= 0:
            return False
        gross = entry * aandelen
        cost  = trade_cost(gross)
        if not self.can_open(prices, gross + cost):
            return False
        tp = entry + cfg["tp_mult"] * sig.atr
        self.cash -= gross + cost
        self.positions[sig.ticker] = BTPosition(
            ticker=sig.ticker, strategy=sig.strategy, entry_date=sig.date,
            entry_price=round(entry, 4), size=aandelen, cost=cost,
            sl=round(stop, 4), tp=round(tp, 4), atr=sig.atr, days_open=0,
        )
        return True

    def close_position(self, ticker: str, date: dt.date, exit_price: float, reason: str):
        if ticker not in self.positions:
            return
        p          = self.positions[ticker]
        exit_slip  = exit_price * (1 - SLIPPAGE_PCT)
        gross      = exit_slip * p.size
        cost       = trade_cost(gross)
        pnl        = gross - cost - (p.entry_price * p.size + p.cost)
        tax        = pnl * TAX_RATE if pnl > 0 else 0.0
        self.cash += gross - cost - tax
        self.closed_trades.append({
            "entry_date":  p.entry_date.isoformat(),
            "exit_date":   date.isoformat(),
            "ticker":      ticker,
            "strategy":    p.strategy,
            "entry_price": p.entry_price,
            "exit_price":  round(exit_slip, 4),
            "size":        p.size,
            "atr":         p.atr,
            "pnl":         round(pnl, 2),
            "tax":         round(tax, 2),
            "net":         round(pnl - tax, 2),
            "reason":      reason,
            "days_open":   p.days_open,
        })
        del self.positions[ticker]

    def snapshot(self, date: dt.date, prices: Dict[str, float]):
        pos_val = sum(
            prices.get(t, p.entry_price) * p.size for t, p in self.positions.items()
        )
        self.daily_snapshots.append({
            "date":            date.isoformat(),
            "cash":            round(self.cash, 2),
            "positions_value": round(pos_val, 2),
            "total":           round(self.cash + pos_val, 2),
            "n_positions":     len(self.positions),
        })


def run_backtest():
    print("=" * 60)
    print("BACKTEST GLOBAL v5.0  —  ATR Sizing + T+1 Slippage")
    print(f"Periode  : {BACKTEST_START} -> {BACKTEST_END}")
    print(f"Kapitaal : EUR{START_CAPITAL:,.0f}")
    print(f"Risico%  : {int(RISICO_PCT_PER_TRADE*100)}% per trade  |  Slippage: {SLIPPAGE_PCT*100:.1f}%")
    print("=" * 60)

    all_tickers: List[str] = []
    for path in EXCHANGES.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[WARN] Geen tickerbestanden gevonden, gebruik fallback tickers.")
        for tlist in FALLBACK_TICKERS.values():
            all_tickers.extend(tlist)
        all_tickers = sorted(set(all_tickers))
    print(f"Tickers  : {len(all_tickers)}")

    print("Data downloaden...")
    df = download_history(all_tickers, start=BACKTEST_START, end=BACKTEST_END)
    if df.empty:
        print("Geen data. Gestopt.")
        return

    print("Indicatoren berekenen...")
    df = add_indicators(df)

    all_dates = sorted(df["Date"].dt.date.unique())
    print(f"Handelsdagen: {len(all_dates)}")

    bt = BacktestPortfolio(START_CAPITAL)

    for date in all_dates:
        day_df = df[df["Date"] == pd.Timestamp(date)].copy()
        prices = {
            row["Ticker"]: safe_float(row["Close"])
            for _, row in day_df.iterrows()
            if not math.isnan(safe_float(row.get("Close")))
        }

        # Days_open
        for p in bt.positions.values():
            p.days_open += 1

        # Exit
        for ticker, pos in list(bt.positions.items()):
            if ticker not in prices:
                continue
            close = prices[ticker]
            row   = day_df[day_df["Ticker"] == ticker]
            if row.empty:
                continue
            r    = row.iloc[0]
            ma20 = safe_float(r.get("MA20"))
            rsi  = safe_float(r.get("RSI14"))

            reason: Optional[str] = None
            if pos.sl is not None and close <= pos.sl:
                reason = f"Stoploss ({pos.sl:.2f})"
            elif pos.tp is not None and close >= pos.tp:
                reason = f"Take Profit ({pos.tp:.2f})"
            elif (not math.isnan(ma20) and close < ma20
                  and pos.strategy in ("Traag", "Snel", "Hyper Trend")):
                reason = "Trend exit MA20"
            elif (not math.isnan(rsi) and rsi > 70
                  and pos.strategy in ("MRA Snel", "MRA Traag", "Hyper Scalp")):
                reason = f"RSI exit ({rsi:.1f})"
            elif pos.days_open >= MAX_HOLD_DAYS:
                reason = f"Time exit ({pos.days_open}d)"

            if reason:
                bt.close_position(ticker, date, close, reason)

        # Koop
        buy_signals = generate_signals_for_day(day_df, date)
        for sig in buy_signals:
            total = bt.current_total_value(prices)
            cfg   = STRAT_CONFIG.get(sig.strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
            entry = (sig.next_open or sig.price) * (1 + SLIPPAGE_PCT)
            n, _, _ = bereken_atr_positie(total, entry, sig.atr, cfg["sl_mult"])
            est_inv = entry * n + trade_cost(entry * n)
            if not bt.can_open(prices, est_inv):
                continue
            bt.open_position(sig, prices)

        bt.snapshot(date, prices)

    if bt.closed_trades:
        trades_df = pd.DataFrame(bt.closed_trades)
        trades_df.to_csv("backtest_trades_v5.csv", index=False, encoding="utf-8")
        print(f"\nTrades: backtest_trades_v5.csv ({len(bt.closed_trades)} trades)")

    snap_df = pd.DataFrame(bt.daily_snapshots)
    snap_df.to_csv("backtest_portfolio_v5.csv", index=False, encoding="utf-8")
    print("Portfolio: backtest_portfolio_v5.csv")

    _print_stats(bt, snap_df)


def _print_stats(bt: BacktestPortfolio, snap_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BACKTEST RESULTATEN v5.0")
    print("=" * 60)

    if snap_df.empty:
        print("Geen data.")
        return

    start_val  = START_CAPITAL
    end_val    = snap_df.iloc[-1]["total"]
    total_ret  = (end_val - start_val) / start_val * 100
    start_date = pd.to_datetime(snap_df.iloc[0]["date"])
    end_date   = pd.to_datetime(snap_df.iloc[-1]["date"])
    years      = max((end_date - start_date).days / 365.25, 1e-6)
    cagr       = ((end_val / start_val) ** (1 / years) - 1) * 100

    snap_df["peak"]      = snap_df["total"].cummax()
    snap_df["drawdown"]  = (snap_df["total"] - snap_df["peak"]) / snap_df["peak"] * 100
    max_dd               = snap_df["drawdown"].min()

    snap_df["daily_ret"] = snap_df["total"].pct_change()
    avg_d  = snap_df["daily_ret"].mean()
    std_d  = snap_df["daily_ret"].std()
    sharpe = (avg_d / std_d * math.sqrt(252)) if std_d > 1e-9 else 0.0

    print(f"Startkapitaal    : EUR{start_val:>12,.2f}")
    print(f"Eindkapitaal     : EUR{end_val:>12,.2f}")
    print(f"Totaal rendement : {total_ret:>+.1f}%")
    print(f"CAGR             : {cagr:>+.1f}%")
    print(f"Max Drawdown     : {max_dd:>.1f}%")
    print(f"Sharpe Ratio     : {sharpe:>.2f}")

    if bt.closed_trades:
        tdf      = pd.DataFrame(bt.closed_trades)
        n        = len(tdf)
        n_win    = (tdf["net"] > 0).sum()
        n_loss   = (tdf["net"] <= 0).sum()
        win_rate = n_win / n * 100
        avg_win  = tdf.loc[tdf["net"] > 0,  "net"].mean() if n_win  else 0.0
        avg_loss = tdf.loc[tdf["net"] <= 0, "net"].mean() if n_loss else 0.0
        pf_denom = abs(tdf.loc[tdf["net"] <= 0, "net"].sum())
        pf       = abs(tdf.loc[tdf["net"] > 0,  "net"].sum()) / max(pf_denom, 1e-9)

        print(f"\nAantal trades    : {n}")
        print(f"Winnaars         : {n_win} ({win_rate:.1f}%)")
        print(f"Verliezers       : {n_loss}")
        print(f"Gem. winst       : EUR{avg_win:>+.2f}")
        print(f"Gem. verlies     : EUR{avg_loss:>+.2f}")
        print(f"Profit factor    : {pf:.2f}")
        print(f"Betaalde bel.    : EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days_open'].mean():.1f} dagen")

        # ── PER STRATEGIE met Sharpe ← NIEUW v5.0
        print(f"\n{'─'*72}")
        print(f"{'Strategie':<15} {'#':>4} {'Win%':>6} {'Net PnL':>10} {'Avg':>9} {'Sharpe':>7}")
        print(f"{'─'*72}")
        for strat, grp in tdf.groupby("strategy"):
            wr    = (grp["net"] > 0).sum() / len(grp) * 100
            net   = grp["net"].sum()
            avg   = grp["net"].mean()
            std_s = grp["net"].std()
            sh_s  = (avg / std_s) if std_s > 1e-9 else 0.0
            print(f"{strat:<15} {len(grp):>4} {wr:>5.1f}% {net:>+10.2f} {avg:>+9.2f} {sh_s:>+7.2f}")

        print(f"\nEXIT-REDENEN:")
        rkey = tdf["reason"].str.split("(").str[0].str.strip()
        for rk, grp in tdf.groupby(rkey):
            wr  = (grp["net"] > 0).sum() / len(grp) * 100
            net = grp["net"].sum()
            print(f"  {rk:<26} #{len(grp):>3}  win={wr:>4.0f}%  net=EUR{net:>+,.0f}")

    print("=" * 60)


# ============================================================
# TELEGRAM OUTPUT  ← v5.0: sizing-advies in elk signaal
# ============================================================

def _yahoo_link(ticker: str) -> str:
    return f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"


def _strat_emoji(strategy: str) -> str:
    return {
        "Traag":       "🐢",
        "Snel":        "⚡",
        "Hyper Trend": "🚀",
        "Hyper Scalp": "🔥",
        "MRA Snel":    "🛡️",
        "MRA Traag":   "🐢",
    }.get(strategy, "📊")


def format_signals_per_exchange(
    exchange_name:   str,
    buy_signals:     List[Signal],
    sell_signals:    List[Signal],
    portfolio:       LivePortfolio,
    portfolio_waarde: float,
) -> Tuple[str, str]:

    nu = today_str()

    def koop_blok(signals: List[Signal]) -> str:
        if not signals:
            return "_Geen signalen_"
        lines = []
        for s in signals:
            cfg = STRAT_CONFIG.get(s.strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
            lines.append(
                f"• `{s.ticker}` {_strat_emoji(s.strategy)} *{s.strategy}* | "
                f"EUR{s.price:.2f} | R/R:{s.rr_ratio:.1f} | {_yahoo_link(s.ticker)}\n"
                + sizing_tekst(
                    s.ticker, "LONG", s.price, s.atr,
                    portfolio_waarde, cfg["sl_mult"], cfg["tp_mult"]
                )
            )
        return "\n\n".join(lines)

    def verkoop_blok(signals: List[Signal]) -> str:
        if not signals:
            return "_Geen signalen_"
        lines = []
        for s in signals:
            pos  = portfolio.positions.get(s.ticker)
            size = pos.size if pos else 0
            pnl_str = ""
            if pos:
                pnl = (s.price * (1 - SLIPPAGE_PCT) - pos.entry_price) * size
                pnl_str = f" | PnL est: EUR{pnl:+.2f}"
            lines.append(
                f"• `{s.ticker}` {_strat_emoji(s.strategy)} *VERKOOP*\n"
                f"  Reden : {s.reason}\n"
                f"  Prijs : EUR{s.price:.2f}{pnl_str}\n"
                f"  Commando: `/sell {s.ticker} {s.price:.2f} {size}`"
            )
        return "\n\n".join(lines)

    traag_sig  = [s for s in buy_signals if s.strategy == "Traag"]
    snel_sig   = [s for s in buy_signals if s.strategy == "Snel"]
    htrend_sig = [s for s in buy_signals if s.strategy == "Hyper Trend"]
    hscalp_sig = [s for s in buy_signals if s.strategy == "Hyper Scalp"]
    mras_sig   = [s for s in buy_signals if s.strategy == "MRA Snel"]
    mrat_sig   = [s for s in buy_signals if s.strategy == "MRA Traag"]

    pos_count = len(portfolio.positions)
    cash_str  = f"EUR{portfolio.cash:,.2f}"

    deel1 = "\n\n".join([
        f"📊 *{exchange_name} — GLOBAL v5.0*",
        f"_{nu} | Portefeuille: EUR{portfolio_waarde:,.0f} | Risico: 10%/trade_",
        "─────────────────────────────",
        "🐢 *TRAAG (50/200 trend):*",
        koop_blok(traag_sig),
        "⚡ *SNEL (20/50 trend):*",
        koop_blok(snel_sig),
        "🔴 *VERKOOPSIGNALEN:*",
        verkoop_blok(sell_signals),
    ])

    deel2_lines = [
        f"📊 *{exchange_name} — v5.0 (2/2)*",
        "",
        "🚀 *HYPER TREND:*",
        koop_blok(htrend_sig),
        "",
        "🔥 *HYPER SCALP:*",
        koop_blok(hscalp_sig),
        "",
        "🛡️ *MRA SNEL:*",
        koop_blok(mras_sig),
        "",
        "🐢 *MRA TRAAG:*",
        koop_blok(mrat_sig),
        "",
        f"💼 *PORTFOLIO:* {pos_count}/{MAX_POSITIONS} posities | Cash: {cash_str}",
    ]

    if portfolio.positions:
        for t, pos in portfolio.positions.items():
            deel2_lines.append(
                f"  • `{t}`: {pos.size}x @ EUR{pos.entry_price:.2f} "
                f"({pos.strategy}, {pos.days_open}d | SL:{format_price(pos.sl)} TP:{format_price(pos.tp)})"
            )

    deel2_lines += [
        "",
        "⚙️ *PARAMETERS v5.0:*",
        f"_ATR-sizing: 10% risico/trade | Slippage: 0.1%_",
        f"_Entry: open T+1 | SL: 2×ATR | TP: 3-5×ATR_",
        f"_ADX>15 filter | Wilder smoothing | Time-exit: {MAX_HOLD_DAYS}d_",
    ]

    return deel1, "\n".join(deel2_lines)


# ============================================================
# /BUY EN /SELL COMMANDO'S
# ============================================================

def apply_telegram_commands(portfolio: LivePortfolio, commands_file: str):
    if not os.path.exists(commands_file):
        return
    with open(commands_file, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    if not lines:
        return
    today = today_str()
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        cmd, ticker = parts[0].lower(), parts[1]
        try:
            price, size = float(parts[2]), int(parts[3])
        except ValueError:
            continue
        if cmd == "/buy":
            cost = trade_cost(price * size)
            portfolio.cash -= price * size + cost
            portfolio.positions[ticker] = LivePosition(
                ticker=ticker, strategy="MANUAL", entry_date=today,
                entry_price=price, size=size, cost=cost,
                sl=None, tp=None, atr=0.0, days_open=0,
            )
            portfolio.log_trade(today, ticker, "MANUAL", "BUY",
                                price, size, cost, 0.0, 0.0, 0.0, "Manual /buy")
        elif cmd == "/sell":
            portfolio.close_position(ticker, today, price, "Manual /sell")
    with open(commands_file, "w", encoding="utf-8") as f:
        f.write("")


# ============================================================
# MAIN: LIVE ENGINE
# ============================================================

def run_live_engine():
    all_tickers: List[str] = []
    exchange_tickers: Dict[str, List[str]] = {}
    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        exchange_tickers[ex_name] = tlist
        all_tickers.extend(tlist)
    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[WARN] Geen tickerbestanden gevonden, gebruik fallback tickers.")
        for ex_name, tlist in FALLBACK_TICKERS.items():
            exchange_tickers[ex_name] = tlist
            all_tickers.extend(tlist)
        all_tickers = sorted(set(all_tickers))

    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data. Bot gestopt.")
        return

    n_ok = df["Ticker"].nunique()
    n_total = len(all_tickers)
    if n_ok < n_total:
        print(f"[WARN] Data beschikbaar voor {n_ok}/{n_total} tickers.")

    df = add_indicators(df)

    last_date = df["Date"].max().date()
    portfolio = LivePortfolio(START_CAPITAL)

    day_df    = df[df["Date"] == pd.Timestamp(last_date)].copy()
    price_map = {
        row["Ticker"]: safe_float(row["Close"])
        for _, row in day_df.iterrows()
        if not math.isnan(safe_float(row.get("Close")))
    }

    # ── days_open correct bijhouden ← FIX v5.0
    for p in portfolio.positions.values():
        p.days_open += 1

    portfolio_waarde = portfolio.current_total_value(price_map)
    exit_signals_all = generate_exit_signals(portfolio, df, last_date)

    for ex_name, tlist in exchange_tickers.items():
        if not tlist:
            continue
        df_ex       = df[df["Ticker"].isin(tlist)].copy()
        buy_signals = generate_signals_for_day(df_ex, last_date)

        filtered_buys: List[Signal] = []
        for sig in buy_signals:
            cfg   = STRAT_CONFIG.get(sig.strategy, {"sl_mult": 2.0, "tp_mult": 3.0})
            entry = (sig.next_open or sig.price) * (1 + SLIPPAGE_PCT)
            n, _, _ = bereken_atr_positie(portfolio_waarde, entry, sig.atr, cfg["sl_mult"])
            est_inv = entry * n + trade_cost(entry * n)
            if portfolio.can_open_new_position(price_map, est_inv):
                filtered_buys.append(sig)
                if len(filtered_buys) + len(portfolio.positions) >= MAX_POSITIONS:
                    break

        exit_ex = [s for s in exit_signals_all if s.ticker in tlist]
        deel1, deel2 = format_signals_per_exchange(
            ex_name, filtered_buys, exit_ex, portfolio, portfolio_waarde
        )
        send_telegram_message(deel1)
        time.sleep(1)
        send_telegram_message(deel2)

    portfolio.save_state(last_date.isoformat(), price_map)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    elif mode == "apply":
        p = LivePortfolio(START_CAPITAL)
        apply_telegram_commands(p, "commands.txt")
        p.save_state(today_str(), {})
    else:
        run_live_engine()
