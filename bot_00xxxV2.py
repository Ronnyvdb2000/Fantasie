#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL ENGINE v3.0 — Realistische Portefeuille + Signaal Bot

Features:
- 1x per dag run (GitHub Actions om 22u)
- Backtest-engine (optioneel, via flag)
- Live signaal-engine (Telegram berichten)
- Live portefeuille-tracker (CSV-bestanden)
- Max 10 posities, max 10% cash
- Inzet 2500 → 3000 bij groei
- Alle strategieën: Traag, Snel, Hyper Trend, Hyper Scalp, MRA Snel, MRA Traag
- Volledige exit-engine: SL, TP, MA-exit, RSI-exit, Time-exit
- Verkoopsignalen per beurs en per ticker
"""

import os
import sys
import math
import csv
import json
import time
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL = 50_000.0
MAX_POSITIONS = 10
MAX_CASH_RATIO = 0.10  # max 10% cash
BASE_POSITION_SIZE = 2500.0
MAX_POSITION_SIZE = 3000.0

TRADE_COST_FIXED = 15.0
TRADE_COST_PCT = 0.0035
TAX_RATE = 0.10  # 10% op meerwaarde

MAX_HOLD_DAYS = 20  # time-exit

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Bestanden
LIVE_TRADES_FILE = "trades_live.csv"
LIVE_POSITIONS_FILE = "positions_live.csv"
LIVE_PORTFOLIO_FILE = "portfolio_live.csv"

# Ticker-lijsten per beurs (voorbeeld)
EXCHANGES = {
    "041 Benelux": "tickers_041.txt",
    "048 Nasdaq/NYSE": "tickers_048.txt",
    # voeg hier andere beurzen toe
}

# ============================================================
# HULPFUNCTIES
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT


def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")


def load_tickers_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    return [x for x in lines if x and not x.startswith("#")]


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram niet geconfigureerd, bericht niet verzonden.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram fout: {e}")
        print(text)


def ensure_csv_header(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


# ============================================================
# DATA & INDICATOREN
# ============================================================

def download_history(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True
    )
    # Normaliseer naar MultiIndex: (date, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t not in data.columns.levels[1]:
                continue
            df_t = data.xs(t, axis=1, level=1).copy()
            df_t["Ticker"] = t
            frames.append(df_t)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
    else:
        df = data.copy()
        df["Ticker"] = tickers[0]
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Verwacht kolommen: Date, Open, High, Low, Close, Adj Close, Volume, Ticker
    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        close = g["Close"]

        g["MA20"] = close.rolling(20).mean()
        g["MA50"] = close.rolling(50).mean()
        g["MA200"] = close.rolling(200).mean()

        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain).rolling(14).mean()
        roll_loss = pd.Series(loss).rolling(14).mean()
        rs = roll_gain / (roll_loss + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift()).abs()
        tr3 = (g["Low"] - g["Close"].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["ATR14"] = tr.rolling(14).mean()

        g["IBS"] = (g["Close"] - g["Low"]) / (g["High"] - g["Low"] + 1e-9)

        # ADX (simplified)
        up_move = g["High"].diff()
        down_move = g["Low"].diff() * -1
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = g["ATR14"]
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum() / (atr.rolling(14).sum() + 1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / (atr.rolling(14).sum() + 1e-9)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        g["ADX14"] = dx.rolling(14).mean()

        return g

    return df.groupby("Ticker", group_keys=False).apply(_calc)


# ============================================================
# STRATEGIE-SIGNALEN
# ============================================================

@dataclass
class Signal:
    ticker: str
    date: dt.date
    strategy: str
    direction: str  # "BUY" of "SELL"
    reason: str
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    extra: Dict = field(default_factory=dict)


def generate_signals_for_day(df: pd.DataFrame, date: dt.date) -> List[Signal]:
    """Genereer KOOP-signalen voor een bepaalde dag (EOD)."""
    signals: List[Signal] = []
    day_df = df[df["Date"] == pd.Timestamp(date)].copy()
    if day_df.empty:
        return signals

    for _, row in day_df.iterrows():
        t = row["Ticker"]
        close = row["Close"]
        ma50 = row["MA50"]
        ma200 = row["MA200"]
        rsi = row["RSI14"]
        ibs = row["IBS"]
        atr = row["ATR14"]
        adx = row["ADX14"]

        # Traag (50/200 trend)
        if not math.isnan(ma50) and not math.isnan(ma200):
            if ma50 > ma200 and close > ma50 and adx > 15:
                signals.append(Signal(
                    ticker=t,
                    date=date,
                    strategy="Traag",
                    direction="BUY",
                    reason="MA50>MA200 & Close>MA50 & ADX>15",
                    price=float(close),
                    sl=float(close - 2 * atr) if not math.isnan(atr) else None,
                    tp=float(close + 4 * atr) if not math.isnan(atr) else None,
                ))

        # Snel (20/50 trend)
        ma20 = row["MA20"]
        if not math.isnan(ma20) and not math.isnan(ma50):
            if ma20 > ma50 and close > ma20 and adx > 15:
                signals.append(Signal(
                    ticker=t,
                    date=date,
                    strategy="Snel",
                    direction="BUY",
                    reason="MA20>MA50 & Close>MA20 & ADX>15",
                    price=float(close),
                    sl=float(close - 2 * atr) if not math.isnan(atr) else None,
                    tp=float(close + 3 * atr) if not math.isnan(atr) else None,
                ))

        # Hyper Trend (strenger)
        if not math.isnan(ma50) and not math.isnan(ma200):
            if ma50 > ma200 and close > ma50 and adx > 20 and rsi > 55:
                signals.append(Signal(
                    ticker=t,
                    date=date,
                    strategy="Hyper Trend",
                    direction="BUY",
                    reason="Sterke trend (ADX>20, RSI>55)",
                    price=float(close),
                    sl=float(close - 2.5 * atr) if not math.isnan(atr) else None,
                    tp=float(close + 5 * atr) if not math.isnan(atr) else None,
                ))

        # Hyper Scalp (mean reversion kort)
        if rsi < 30 and ibs < 0.2 and not math.isnan(atr):
            signals.append(Signal(
                ticker=t,
                date=date,
                strategy="Hyper Scalp",
                direction="BUY",
                reason="RSI<30 & IBS<0.2",
                price=float(close),
                sl=float(close - 1.5 * atr),
                tp=float(close + 2.5 * atr),
            ))

        # MRA Snel
        if rsi < 35 and ibs < 0.3 and not math.isnan(atr):
            signals.append(Signal(
                ticker=t,
                date=date,
                strategy="MRA Snel",
                direction="BUY",
                reason="RSI<35 & IBS<0.3",
                price=float(close),
                sl=float(close - 2 * atr),
                tp=float(close + 3 * atr),
            ))

        # MRA Traag
        if rsi < 40 and ibs < 0.4 and not math.isnan(atr):
            signals.append(Signal(
                ticker=t,
                date=date,
                strategy="MRA Traag",
                direction="BUY",
                reason="RSI<40 & IBS<0.4",
                price=float(close),
                sl=float(close - 2.5 * atr),
                tp=float(close + 4 * atr),
            ))

    return signals


# ============================================================
# LIVE PORTEFEUILLE-TRACKER
# ============================================================

@dataclass
class LivePosition:
    ticker: str
    strategy: str
    entry_date: str
    entry_price: float
    size: int
    cost: float
    sl: Optional[float]
    tp: Optional[float]
    days_open: int = 0


class LivePortfolio:
    def __init__(self, start_capital: float):
        self.cash = start_capital
        self.positions: Dict[str, LivePosition] = {}
        self.load_state()

    # ---------- CSV STATE ----------
    def load_state(self):
        # portfolio
        if os.path.exists(LIVE_PORTFOLIO_FILE):
            df = pd.read_csv(LIVE_PORTFOLIO_FILE)
            if not df.empty:
                last = df.iloc[-1]
                self.cash = float(last["cash"])
        # positions
        if os.path.exists(LIVE_POSITIONS_FILE):
            dfp = pd.read_csv(LIVE_POSITIONS_FILE)
            for _, r in dfp.iterrows():
                self.positions[r["ticker"]] = LivePosition(
                    ticker=r["ticker"],
                    strategy=r["strategy"],
                    entry_date=r["entry_date"],
                    entry_price=float(r["entry_price"]),
                    size=int(r["size"]),
                    cost=float(r["cost"]),
                    sl=float(r["sl"]) if not pd.isna(r["sl"]) else None,
                    tp=float(r["tp"]) if not pd.isna(r["tp"]) else None,
                    days_open=int(r.get("days_open", 0)),
                )

    def save_state(self, date: str, prices: Dict[str, float]):
        # positions
        ensure_csv_header(LIVE_POSITIONS_FILE,
                          ["ticker", "strategy", "entry_date", "entry_price", "size", "cost", "sl", "tp", "days_open"])
        with open(LIVE_POSITIONS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "strategy", "entry_date", "entry_price", "size", "cost", "sl", "tp", "days_open"])
            for p in self.positions.values():
                w.writerow([
                    p.ticker, p.strategy, p.entry_date, p.entry_price, p.size,
                    p.cost, p.sl if p.sl is not None else "", p.tp if p.tp is not None else "", p.days_open
                ])

        # portfolio
        ensure_csv_header(LIVE_PORTFOLIO_FILE, ["date", "cash", "positions_value", "total"])
        positions_value = 0.0
        for t, p in self.positions.items():
            price = prices.get(t, p.entry_price)
            positions_value += price * p.size
        total = self.cash + positions_value
        with open(LIVE_PORTFOLIO_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([date, self.cash, positions_value, total])

    # ---------- LOGIC ----------
    def current_total_value(self, prices: Dict[str, float]) -> float:
        v = self.cash
        for t, p in self.positions.items():
            price = prices.get(t, p.entry_price)
            v += price * p.size
        return v

    def dynamic_position_size(self, prices: Dict[str, float]) -> float:
        total = self.current_total_value(prices)
        if total >= 60_000:
            return MAX_POSITION_SIZE
        return BASE_POSITION_SIZE

    def can_open_new_position(self, prices: Dict[str, float]) -> bool:
        if len(self.positions) >= MAX_POSITIONS:
            return False
        total = self.current_total_value(prices)
        max_cash = total * MAX_CASH_RATIO
        if self.cash <= max_cash:
            return False
        pos_size = self.dynamic_position_size(prices)
        return self.cash >= pos_size

    def open_position(self, sig: Signal, prices: Dict[str, float]) -> Optional[LivePosition]:
        if sig.ticker in self.positions:
            return None
        if not self.can_open_new_position(prices):
            return None
        pos_size_eur = self.dynamic_position_size(prices)
        entry_price = sig.price
        if entry_price <= 0:
            return None
        size = int(pos_size_eur // entry_price)
        if size <= 0:
            return None
        gross = size * entry_price
        cost = trade_cost(gross)
        total_out = gross + cost
        if total_out > self.cash:
            return None
        self.cash -= total_out
        p = LivePosition(
            ticker=sig.ticker,
            strategy=sig.strategy,
            entry_date=sig.date.isoformat(),
            entry_price=entry_price,
            size=size,
            cost=cost,
            sl=sig.sl,
            tp=sig.tp,
            days_open=0
        )
        self.positions[sig.ticker] = p
        self.log_trade(sig.date.isoformat(), sig.ticker, sig.strategy,
                       "BUY", entry_price, size, cost, 0.0, 0.0, 0.0)
        return p

    def close_position(self, ticker: str, date: str, exit_price: float, reason: str):
        if ticker not in self.positions:
            return
        p = self.positions[ticker]
        gross = exit_price * p.size
        cost = trade_cost(gross)
        pnl = gross - cost - (p.entry_price * p.size + p.cost)
        tax = 0.0
        if pnl > 0:
            tax = pnl * TAX_RATE
        net = pnl - tax
        self.cash += gross - cost - tax
        self.log_trade(date, ticker, p.strategy, "SELL", exit_price, p.size, cost, pnl, tax, net, reason)
        del self.positions[ticker]

    def log_trade(self, date: str, ticker: str, strategy: str, side: str,
                  price: float, size: int, cost: float, pnl: float,
                  tax: float, net: float, reason: str = ""):
        ensure_csv_header(LIVE_TRADES_FILE,
                          ["date", "ticker", "strategy", "side", "price", "size",
                           "cost", "pnl", "tax", "net", "reason"])
        with open(LIVE_TRADES_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([date, ticker, strategy, side, price, size, cost, pnl, tax, net, reason])


# ============================================================
# EXIT-ENGINE (SL / TP / MA / RSI / TIME)
# ============================================================

def generate_exit_signals(
    portfolio: LivePortfolio,
    df: pd.DataFrame,
    date: dt.date
) -> List[Signal]:
    signals: List[Signal] = []
    day_df = df[df["Date"] == pd.Timestamp(date)].copy()
    if day_df.empty:
        return signals
    day_map = {row["Ticker"]: row for _, row in day_df.iterrows()}

    for t, p in list(portfolio.positions.items()):
        if t not in day_map:
            continue
        row = day_map[t]
        close = float(row["Close"])
        ma20 = float(row["MA20"])
        rsi = float(row["RSI14"])
        atr = float(row["ATR14"]) if not math.isnan(row["ATR14"]) else None

        reason = None

        # 1) Stoploss
        if p.sl is not None and close <= p.sl:
            reason = f"Stoploss geraakt (SL={p.sl:.2f})"

        # 2) Take profit
        if reason is None and p.tp is not None and close >= p.tp:
            reason = f"Take Profit geraakt (TP={p.tp:.2f})"

        # 3) MA-exit (trendbreuk)
        if reason is None and not math.isnan(ma20):
            if close < ma20 and p.strategy in ["Traag", "Snel", "Hyper Trend"]:
                reason = f"Trend exit: Close < MA20 ({close:.2f} < {ma20:.2f})"

        # 4) RSI-exit (overbought)
        if reason is None and not math.isnan(rsi):
            if rsi > 70 and p.strategy in ["MRA Snel", "MRA Traag", "Hyper Scalp"]:
                reason = f"RSI exit: RSI>70 ({rsi:.1f})"

        # 5) Time-exit
        if reason is None:
            if p.days_open >= MAX_HOLD_DAYS:
                reason = f"Time exit: {p.days_open} dagen open"

        if reason is not None:
            signals.append(Signal(
                ticker=t,
                date=date,
                strategy=p.strategy,
                direction="SELL",
                reason=reason,
                price=close
            ))

    return signals


# ============================================================
# TELEGRAM-OUTPUT (PER BEURS, PER TICKER)
# ============================================================

def format_signals_per_exchange(
    exchange_name: str,
    buy_signals: List[Signal],
    sell_signals: List[Signal],
    portfolio: LivePortfolio
) -> str:
    lines = []
    today = today_str()
    header = f"📊 {exchange_name} — GLOBAL v3.0\n{today}\n----------------------------------"
    lines.append(header)

    # BUY-signalen per strategie
    if buy_signals:
        lines.append("")
        lines.append("🟢 KOOPSIGNALEN:")
        # groepeer per ticker
        by_ticker: Dict[str, List[Signal]] = {}
        for s in buy_signals:
            by_ticker.setdefault(s.ticker, []).append(s)
        for ticker in sorted(by_ticker.keys()):
            lines.append(f"\n**{ticker}**")
            for s in by_ticker[ticker]:
                lines.append(
                    f"• {s.ticker}: {s.strategy} | €{s.price:.2f} | SL {s.sl:.2f if s.sl else 'n/a'} | TP {s.tp:.2f if s.tp else 'n/a'}"
                )
    else:
        lines.append("")
        lines.append("🟢 KOOPSIGNALEN:")
        lines.append("Geen koopsignalen")

    # SELL-signalen per ticker
    if sell_signals:
        lines.append("")
        lines.append("🔴 VERKOOPSIGNALEN:")
        by_ticker: Dict[str, List[Signal]] = {}
        for s in sell_signals:
            by_ticker.setdefault(s.ticker, []).append(s)
        for ticker in sorted(by_ticker.keys()):
            lines.append(f"\n**{ticker}**")
            for s in by_ticker[ticker]:
                # we kennen de size uit portfolio
                size = portfolio.positions.get(ticker).size if ticker in portfolio.positions else 0
                lines.append(
                    f"- Reden: {s.reason}\n"
                    f"  Strategie: {s.strategy}\n"
                    f"  Slotprijs: €{s.price:.2f}\n"
                    f"  Commando:\n"
                    f"  `/sell {ticker} {s.price:.2f} {size}`"
                )
    else:
        lines.append("")
        lines.append("🔴 VERKOOPSIGNALEN:")
        lines.append("Geen verkoopsignalen")

    return "\n".join(lines)


# ============================================================
# PARSEN VAN /buy EN /sell COMMANDO'S (VIA BESTAND)
# ============================================================

def apply_telegram_commands(portfolio: LivePortfolio, commands_file: str):
    """
    Simpele manier om /buy en /sell te verwerken:
    - Jij kopieert de Telegram-commando's naar bv. commands.txt
    - Script wordt met argument --apply-commands gedraaid
    - Deze functie verwerkt de commando's en leegt het bestand
    """
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
        cmd = parts[0].lower()
        ticker = parts[1]
        try:
            price = float(parts[2])
            size = int(parts[3])
        except ValueError:
            continue

        if cmd == "/buy":
            # we log alleen, echte open gebeurt via signaal-engine
            cost = trade_cost(price * size)
            portfolio.cash -= (price * size + cost)
            portfolio.positions[ticker] = LivePosition(
                ticker=ticker,
                strategy="MANUAL",
                entry_date=today,
                entry_price=price,
                size=size,
                cost=cost,
                sl=None,
                tp=None,
                days_open=0
            )
            portfolio.log_trade(today, ticker, "MANUAL", "BUY", price, size, cost, 0.0, 0.0, 0.0, "Manual /buy")
        elif cmd == "/sell":
            portfolio.close_position(ticker, today, price, "Manual /sell")

    # bestand leegmaken
    with open(commands_file, "w", encoding="utf-8") as f:
        f.write("")


# ============================================================
# MAIN: LIVE SIGNAL ENGINE
# ============================================================

def run_live_engine():
    # 1) Laad tickers per beurs
    all_tickers = []
    exchange_tickers: Dict[str, List[str]] = {}
    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        exchange_tickers[ex_name] = tlist
        all_tickers.extend(tlist)
    all_tickers = sorted(list(set(all_tickers)))
    if not all_tickers:
        print("Geen tickers gevonden.")
        return

    # 2) Download data + indicators
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("Geen data.")
        return
    df = add_indicators(df)

    # 3) Bepaal datum (laatste dag)
    last_date = df["Date"].max().date()

    # 4) Laad live portefeuille
    portfolio = LivePortfolio(START_CAPITAL)

    # 5) Maak prijs-map voor vandaag
    day_df = df[df["Date"] == pd.Timestamp(last_date)].copy()
    price_map = {row["Ticker"]: float(row["Close"]) for _, row in day_df.iterrows()}

    # 6) Verhoog days_open
    for p in portfolio.positions.values():
        p.days_open += 1

    # 7) Genereer exit-signalen
    exit_signals_all = generate_exit_signals(portfolio, df, last_date)

    # 8) Genereer koop-signalen per beurs
    for ex_name, tlist in exchange_tickers.items():
        if not tlist:
            continue
        df_ex = df[df["Ticker"].isin(tlist)].copy()
        buy_signals = generate_signals_for_day(df_ex, last_date)

        # Filter koop-signalen op basis van portefeuille (max posities, max cash)
        filtered_buys: List[Signal] = []
        # sorteer bv. op RSI (laagste eerst) + ATR (hoogste eerst) als mix
        buy_signals_sorted = sorted(
            buy_signals,
            key=lambda s: (s.strategy, s.price)
        )
        for sig in buy_signals_sorted:
            if portfolio.can_open_new_position(price_map):
                # we openen NIET automatisch, maar we tonen voorstel
                filtered_buys.append(sig)

        # Exit-signalen voor deze beurs
        exit_signals_ex = [s for s in exit_signals_all if s.ticker in tlist]

        # 9) Telegram-bericht
        msg = format_signals_per_exchange(ex_name, filtered_buys, exit_signals_ex, portfolio)
        send_telegram_message(msg)

    # 10) State opslaan
    portfolio.save_state(last_date.isoformat(), price_map)


# ============================================================
# (OPTIONEEL) BACKTEST-ENGINE — skeleton
# ============================================================

def run_backtest():
    """
    Skeleton: hier zou je dezelfde signalen + een aparte Portfolio-klasse
    gebruiken om volledig automatisch te backtesten.
    Voor nu laten we dit leeg of minimal.
    """
    print("Backtest-engine skeleton — nog te vullen indien gewenst.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    # Gebruik:
    #   python bot_00xxxV3.py           -> live signaal-engine
    #   python bot_00xxxV3.py backtest  -> backtest skeleton
    #   python bot_00xxxV3.py apply     -> verwerk commands.txt (/buy /sell)
    mode = "live"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "backtest":
        run_backtest()
    elif mode == "apply":
        p = LivePortfolio(START_CAPITAL)
        apply_telegram_commands(p, "commands.txt")
        # prijzen onbekend hier, dus we loggen alleen cash/positions
        p.save_state(today_str(), {})
    else:
        run_live_engine()
