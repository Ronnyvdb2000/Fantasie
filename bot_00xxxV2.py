import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import os
import csv
from datetime import datetime

# ===========================
# 📦 DATA ENGINE (v3.2)
# ===========================

def load_data(tickers):
    data = {}
    for t in tickers:
        try:
            df = yf.download(
                t,
                period="1y",
                interval="1d",
                auto_adjust=True,   # FIX: splits/dividenden correct
                progress=False
            )
            if df.empty:
                print(f"[WARN] Geen data voor {t}")
                continue

            # FIX: yfinance geeft soms MultiIndex terug bij 1 ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.dropna(inplace=True)
            data[t] = df

            time.sleep(0.25)
        except Exception as e:
            print(f"[ERROR] Download mislukt voor {t}: {e}")

    return data

# ===========================
# ⚡ INDICATORS
# ===========================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()   # Wilder
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ADX(df, period=14):
    df = df.copy()
    df["H-L"]  = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"]  - df["Close"].shift(1)).abs()
    df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    up   = df["High"].diff()
    down = (-df["Low"].diff())
    df["+DM"] = np.where((up > down) & (up > 0), up, 0.0)
    df["-DM"] = np.where((down > up) & (down > 0), down, 0.0)

    tr_sum      = df["TR"].rolling(period).sum()
    df["+DI"]   = 100 * df["+DM"].rolling(period).sum() / (tr_sum + 1e-9)
    df["-DI"]   = 100 * df["-DM"].rolling(period).sum() / (tr_sum + 1e-9)
    df["DX"]    = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"] + 1e-9)) * 100
    return df["DX"].rolling(period).mean()

def ATR(df, period=14):
    df = df.copy()
    hl  = df["High"] - df["Low"]
    hcp = (df["High"] - df["Close"].shift(1)).abs()
    lcp = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def scalar(val):
    """
    FIX KERN: haal altijd een Python scalar op uit een pandas waarde.
    Voorkomt 'The truth value of a Series is ambiguous'.
    """
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

# ===========================
# ⚡ SIGNAL ENGINE (v3.2)
# ===========================

def strategy_slow(df):
    df = df.copy()
    df["EMA50"]  = EMA(df["Close"], 50)
    df["EMA200"] = EMA(df["Close"], 200)
    df["ADX"]    = ADX(df)
    df["RSI"]    = RSI(df["Close"])
    df["Vol20"]  = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]

    ema50  = scalar(last["EMA50"])
    ema200 = scalar(last["EMA200"])
    adx    = scalar(last["ADX"])
    rsi    = scalar(last["RSI"])
    vol    = scalar(last["Volume"])
    vol20  = scalar(last["Vol20"])

    return (
        ema50 > ema200 and
        adx > 18 and
        45 < rsi < 60 and
        vol > vol20
    )

def strategy_fast(df):
    df = df.copy()
    df["EMA20"] = EMA(df["Close"], 20)
    df["EMA50"] = EMA(df["Close"], 50)
    df["RSI"]   = RSI(df["Close"])
    df["Vol10"] = df["Volume"].rolling(10).mean()

    last = df.iloc[-1]

    ema20 = scalar(last["EMA20"])
    ema50 = scalar(last["EMA50"])
    rsi   = scalar(last["RSI"])
    vol   = scalar(last["Volume"])
    vol10 = scalar(last["Vol10"])

    return (
        ema20 > ema50 and
        rsi > 55 and
        vol > vol10
    )

def strategy_hypertrend(df):
    df = df.copy()
    df["EMA10"] = EMA(df["Close"], 10)
    df["EMA20"] = EMA(df["Close"], 20)
    df["EMA50"] = EMA(df["Close"], 50)
    df["ADX"]   = ADX(df)
    df["RSI"]   = RSI(df["Close"])

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    ema10     = scalar(last["EMA10"])
    ema20     = scalar(last["EMA20"])
    ema50     = scalar(last["EMA50"])
    adx       = scalar(last["ADX"])
    rsi       = scalar(last["RSI"])
    close     = scalar(last["Close"])
    close_prv = scalar(prev["Close"])

    return (
        ema10 > ema20 > ema50 and
        adx > 22 and
        rsi > 60 and
        close > close_prv
    )

def strategy_scalp(df):
    df = df.copy()
    df["RSI"]  = RSI(df["Close"])
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Vol5"] = df["Volume"].rolling(5).mean()

    last = df.iloc[-1]

    rsi   = scalar(last["RSI"])
    close = scalar(last["Close"])
    ma20  = scalar(last["MA20"])
    vol   = scalar(last["Volume"])
    vol5  = scalar(last["Vol5"])

    return (
        rsi < 35 and
        close > ma20 and
        vol > vol5
    )

def generate_buy_signals(data):
    signals = []

    for ticker, df in data.items():
        try:
            if len(df) < 200:
                print(f"[WARN] Te weinig data voor {ticker} ({len(df)} rijen), overgeslagen.")
                continue

            if strategy_slow(df):
                signals.append((ticker, "🐢 Traag"))

            if strategy_fast(df):
                signals.append((ticker, "⚡ Snel"))

            if strategy_hypertrend(df):
                signals.append((ticker, "🚀 Hyper Trend"))

            if strategy_scalp(df):
                signals.append((ticker, "🔥 Hyper Scalp"))

        except Exception as e:
            print(f"[ERROR] Strategie fout bij {ticker}: {e}")

    return signals

# ===========================
# 🔥 EXIT ENGINE (v3.2)
# ===========================

def generate_sell_signals(data, positions):
    sells = []

    for ticker, pos in positions.items():
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI"]  = RSI(df["Close"])
        df["ATR"]  = ATR(df)

        last = df.iloc[-1]

        close       = scalar(last["Close"])
        ma20        = scalar(last["MA20"])
        rsi         = scalar(last["RSI"])
        atr         = scalar(last["ATR"])
        entry_price = float(pos.get("entry_price", close))
        max_price   = float(pos.get("max_price", entry_price))
        days_in_pos = int(pos.get("days", 0))

        # Trailing max bijhouden
        if close > max_price:
            max_price = close
            pos["max_price"] = max_price

        stop_level = entry_price - 2 * atr
        gain_pct   = (close / entry_price - 1) * 100

        if close < stop_level:
            sells.append((ticker, "Stoploss"))
            continue

        if gain_pct >= 12:
            sells.append((ticker, "Take Profit 12%"))
            continue

        if gain_pct >= 8 and rsi > 65:
            sells.append((ticker, "Take Profit 8%+RSI"))
            continue

        if close < ma20:
            sells.append((ticker, "MA20 Exit"))
            continue

        if rsi > 70:
            sells.append((ticker, "RSI > 70"))
            continue

        if days_in_pos > 20:
            sells.append((ticker, "Time Exit > 20d"))
            continue

    return sells

# ===========================
# 💼 PORTFOLIO ENGINE (v3.2)
# ===========================

PORTFOLIO_FILE = "portfolio_live.csv"

def load_portfolio():
    portfolio = {
        "cash": 50000.0,
        "positions": {}
    }

    if not os.path.exists(PORTFOLIO_FILE):
        return portfolio

    with open(PORTFOLIO_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "META":
                portfolio["cash"] = float(row["cash"])
            elif row["type"] == "POS":
                ticker = row["ticker"]
                portfolio["positions"][ticker] = {
                    "entry_price": float(row["entry_price"]),
                    "size":        float(row["size"]),
                    "strategy":    row["strategy"],
                    "days":        int(row["days"]),
                    "max_price":   float(row["max_price"])
                }

    return portfolio

def save_portfolio(portfolio):
    fieldnames = ["type", "ticker", "entry_price", "size", "strategy", "days", "max_price", "cash"]
    with open(PORTFOLIO_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            "type": "META", "ticker": "", "entry_price": "",
            "size": "", "strategy": "", "days": "",
            "max_price": "", "cash": portfolio["cash"]
        })

        for ticker, pos in portfolio["positions"].items():
            writer.writerow({
                "type":        "POS",
                "ticker":      ticker,
                "entry_price": pos["entry_price"],
                "size":        pos["size"],
                "strategy":    pos["strategy"],
                "days":        pos["days"],
                "max_price":   pos.get("max_price", pos["entry_price"]),
                "cash":        ""
            })

def update_portfolio(buys, sells, portfolio, data):
    # Verkopen verwerken
    for ticker, reason in sells:
        if ticker in portfolio["positions"]:
            pos        = portfolio["positions"][ticker]
            last_price = scalar(data[ticker]["Close"].iloc[-1])
            value      = float(pos["size"]) * last_price
            portfolio["cash"] += value
            del portfolio["positions"][ticker]
            print(f"[SELL] {ticker} @ {last_price:.2f} | reden: {reason}")

    max_positions     = 10
    current_positions = len(portfolio["positions"])

    # Aankopen verwerken
    for ticker, strategy in buys:
        if current_positions >= max_positions:
            break
        if ticker in portfolio["positions"]:
            continue

        base_size = 3000 if portfolio["cash"] > 60000 else 2500

        if portfolio["cash"] < base_size:
            continue

        price = scalar(data[ticker]["Close"].iloc[-1])
        if price <= 0:
            continue

        size = base_size / price
        portfolio["cash"] -= base_size
        portfolio["positions"][ticker] = {
            "entry_price": price,
            "size":        size,
            "strategy":    strategy,
            "days":        0,
            "max_price":   price
        }
        current_positions += 1
        print(f"[BUY]  {ticker} @ {price:.2f} | strategie: {strategy}")

    # Days_open verhogen
    for pos in portfolio["positions"].values():
        pos["days"] += 1

    return portfolio

# ===========================
# 📲 TELEGRAM ENGINE
# ===========================

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Geen TELEGRAM_TOKEN of TELEGRAM_CHAT_ID ingesteld.")
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
    except Exception as e:
        print(f"[ERROR] Telegram fout: {e}")
        print(text)

def build_report(buys, sells, portfolio):
    lines = []
    lines.append("📊 GLOBAL ENGINE v3.2")
    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("")

    if buys:
        lines.append("🟢 BUY SIGNALEN:")
        for t, s in buys:
            lines.append(f"- {t} ({s})")
    else:
        lines.append("🟢 Geen nieuwe BUY signalen.")
    lines.append("")

    if sells:
        lines.append("🔴 SELL SIGNALEN:")
        for t, r in sells:
            lines.append(f"- {t} ({r})")
    else:
        lines.append("🔴 Geen SELL signalen.")
    lines.append("")

    lines.append("📂 PORTFOLIO:")
    lines.append(f"Cash: €{portfolio['cash']:.2f}")
    if portfolio["positions"]:
        for t, pos in portfolio["positions"].items():
            lines.append(
                f"- {t}: {pos['size']:.4f} @ €{pos['entry_price']:.2f} "
                f"({pos['strategy']}, {pos['days']}d)"
            )
    else:
        lines.append("- Geen open posities.")

    return "\n".join(lines)

# ===========================
# 🧠 MAIN WORKFLOW
# ===========================

def load_ticker_list():
    return ["AAPL", "MSFT", "NVDA", "META", "GOOGL"]

def main():
    print("🚀 GLOBAL ENGINE v3.2 start...")

    tickers = load_ticker_list()
    if not tickers:
        print("Geen tickers gevonden.")
        return

    data = load_data(tickers)
    if not data:
        print("Geen marktdata beschikbaar.")
        return

    portfolio = load_portfolio()

    buys  = generate_buy_signals(data)
    sells = generate_sell_signals(data, portfolio["positions"])

    portfolio = update_portfolio(buys, sells, portfolio, data)
    save_portfolio(portfolio)

    report = build_report(buys, sells, portfolio)
    send_telegram_message(report)

    print("✅ Klaar.")

if __name__ == "__main__":
    main()
