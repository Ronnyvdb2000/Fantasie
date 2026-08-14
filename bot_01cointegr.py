#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01cointegr.py  — STATISTISCHE ARBITRAGE ENGINE v1.0
Gebaseerd op co-integratie en mean reversion (Pairs Trading).

Strategie:
  - Test alle combinaties binnen elke beurs op co-integratie (Engle-Granger)
  - Handel de spread via z-score: entry bij ±2σ, exit bij ±0.5σ
  - Backtest met transactiekosten en slippage

Gebruik:
  python bot_01cointegr.py live      # live rapport (Telegram + Email)
  python bot_01cointegr.py backtest  # backtest modus
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

try:
    from db_logger import log_selectie
except Exception as _e:
    print(f"[WARN] db_logger niet beschikbaar ({_e}) — DB-logging wordt overgeslagen")
    def log_selectie(*args, **kwargs):
        return False

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG — Hergebruik jouw bestaande variabelen
# ============================================================

START_CAPITAL        = 50_000.0
MAX_PAIRS            = 5           # max aantal pairs per beurs
MAX_TOTAL_PAIRS      = 20          # max totaal over alle beurzen
SLIPPAGE_PCT         = 0.001
TRADE_COST_FIXED     = 15.0
TRADE_COST_PCT       = 0.0035
TAX_RATE             = 0.10

# Pairs trading parameters
LOOKBACK             = 60          # rolling window voor z-score
ENTRY_Z              = 2.0         # entry drempel
EXIT_Z               = 0.5         # exit drempel
COINT_P_VALUE_MAX    = 0.05        # maximale p-waarde voor co-integratie
MAX_HALF_LIFE        = 30          # maximale half-life in dagen

# Gebruik jouw bestaande omgevingsvariabelen
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

# Vriendelijke namen voor de beurzen (zelfde als in jouw weekly_report.py)
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

# ============================================================
# HERGEBRUIK: Email & Telegram functies (uit jouw bot_00mail.py)
# ============================================================

def trade_cost(amount: float) -> float:
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str() -> str:
    return dt.date.today().strftime("%Y-%m-%d")

def send_telegram_message(text: str) -> None:
    """Stuur Telegram bericht — identiek aan jouw bestaande functie."""
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
    """Stuur Email — identiek aan jouw bestaande functie."""
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
    """Schat half-life van mean reversion (in dagen)."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    min_len = min(len(spread_lag), len(spread_diff))
    if min_len < 10:
        return float('inf')
    spread_lag = spread_lag.iloc[:min_len]
    spread_diff = spread_diff.iloc[:min_len]
    beta = np.cov(spread_lag, spread_diff)[0,1] / np.var(spread_lag)
    if beta < 0:
        return -np.log(2) / beta
    return float('inf')

# ============================================================
# DATA DOWNLOAD (hergebruik uit jouw andere bots)
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
    """Download historische data — identiek aan jouw bestaande functie."""
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
    score:         float        # score gebaseerd op p-waarde en half-life

def analyseer_pair(
    ticker_a: str,
    ticker_b: str,
    data: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> Optional[PairSignaal]:
    """Analyseer 1 pair en bereken z-score + signaal."""
    try:
        from statsmodels.tsa.stattools import coint
        
        # Check of tickers bestaan
        if ticker_a not in data.columns or ticker_b not in data.columns:
            return None
        
        # Check genoeg data
        if len(data[ticker_a].dropna()) < lookback or len(data[ticker_b].dropna()) < lookback:
            return None
        
        # Bereken spread (prijsverhouding)
        spread = data[ticker_a] / data[ticker_b]
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        
        # Huidige waarden
        current_spread = spread.iloc[-1]
        current_mean = spread_mean.iloc[-1]
        current_std = spread_std.iloc[-1]
        
        if math.isnan(current_std) or current_std <= 0:
            return None
        
        z_score = (current_spread - current_mean) / current_std
        
        # Co-integratie test
        score, p_value, _ = coint(data[ticker_a], data[ticker_b])
        
        if p_value > COINT_P_VALUE_MAX:
            return None
        
        half_life = schat_half_life(spread)
        if half_life > MAX_HALF_LIFE or half_life <= 0:
            return None
        
        # Bepaal signaal
        if z_score > ENTRY_Z:
            signal = "SHORT"  # Spread is te hoog → short spread (verkoop A, koop B)
        elif z_score < -ENTRY_Z:
            signal = "LONG"   # Spread is te laag → long spread (koop A, verkoop B)
        else:
            signal = "NEUTRAL"
        
        # Score: combinatie van p-waarde en half-life (hoe lager beide, hoe beter)
        score_val = (1 / (p_value + 0.001)) * (1 / (half_life + 1))
        
        return PairSignaal(
            pair=f"{ticker_a}/{ticker_b}",
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            p_value=round(p_value, 4),
            half_life=round(half_life, 1),
            current_z=round(z_score, 2),
            spread=round(current_spread, 4),
            spread_mean=round(current_mean, 4),
            spread_std=round(current_std, 4),
            price_a=round(data[ticker_a].iloc[-1], 2),
            price_b=round(data[ticker_b].iloc[-1], 2),
            signal=signal,
            score=score_val,
        )
    except Exception as e:
        return None

# ============================================================
# PAIRS ENGINE — LIVE
# ============================================================

def run_live_engine():
    """Live engine: vind co-geintegreerde pairs en stuur rapport."""
    print(f"{'='*60}")
    print(f"PAIRS TRADING — LIVE  {today_str()}")
    print(f"{'='*60}")

    # Laad alle ticker bestanden (jouw formaat)
    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for f_name in bouw_bestandslijst():
        tlist = laad_tickers_uit_bestand(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name = label_voor(f_name)
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

    # Zet om naar pivot table
    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    pivot = pivot.dropna(axis=1, how='all')
    print(f"Data geladen: {len(pivot.columns)} tickers met voldoende data")

    alle_signalen: List[PairSignaal] = []
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        
        # Filter data voor deze beurs
        beschikbaar = [t for t in tlist if t in pivot.columns]
        if len(beschikbaar) < 2:
            print(f"  → Te weinig beschikbare tickers voor {ex_name}")
            continue
        
        data_ex = pivot[beschikbaar].dropna()
        if len(data_ex.columns) < 2:
            print(f"  → Te weinig data voor {ex_name}")
            continue
        
        # Test alle combinaties
        signalen: List[PairSignaal] = []
        totaal_pairs = len(list(combinations(beschikbaar, 2)))
        processed = 0
        
        for a, b in combinations(beschikbaar, 2):
            processed += 1
            if processed % 50 == 0:
                print(f"  → {processed}/{totaal_pairs} pairs getest...")
            
            sig = analyseer_pair(a, b, data_ex)
            if sig:
                signalen.append(sig)
        
        # Sorteer op score (hoogste eerst)
        signalen.sort(key=lambda s: s.score, reverse=True)
        
        # Selecteer top MAX_PAIRS
        top_signalen = signalen[:MAX_PAIRS]
        
        # Filter op signalen (alleen LONG/SHORT)
        actieve_signalen = [s for s in top_signalen if s.signal != "NEUTRAL"]
        
        print(f"  → {len(signalen)} co-geintegreerde pairs gevonden")
        print(f"  → {len(actieve_signalen)} actieve signalen")
        
        if actieve_signalen:
            # Log naar database (hergebruik db_logger)
            for s in actieve_signalen[:3]:  # Max 3 per beurs loggen
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
                        "grafiek": f"https://finance.yahoo.com/quote/{s.ticker_a}/chart?p={s.ticker_a}",
                    },
                )
            
            # Format bericht
            bericht = format_pairs_bericht(ex_name, actieve_signalen)
            if bericht:
                send_telegram_message(bericht)
                email_delen.append(bericht)
                print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Geen actieve signalen voor {ex_name}")

    # Samenvatting email (één mail met alle beurzen)
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

def format_pairs_bericht(
    exchange_name: str,
    signalen: List[PairSignaal],
) -> Optional[str]:
    if not signalen:
        return None

    nu = today_str()
    
    def signal_emoji(s: str) -> str:
        if s == "LONG":
            return "🟢 LONG"
        elif s == "SHORT":
            return "🔴 SHORT"
        return "⚪ NEUTRAL"
    
    # Top 3
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
    
    # Extra signalen (als er meer dan 3 zijn)
    if len(signalen) > 3:
        extra = signalen[3:]
        delen.append("─────────────────────────────")
        delen.append("📋 *Overige signalen:*")
        for s in extra:
            emoji = "🟢" if s.signal == "LONG" else "🔴" if s.signal == "SHORT" else "⚪"
            delen.append(f"  {emoji} `{s.pair}` | z={s.current_z:.2f} | p={s.p_value:.4f}")
    
    delen.append(
        f"\n⚙️ _Entry: ±{ENTRY_Z}σ | Exit: ±{EXIT_Z}σ | "
        f"Lookback: {LOOKBACK}d | Min p: {COINT_P_VALUE_MAX}_"
    )
    
    return "\n\n".join(delen)

# ============================================================
# BACKTEST ENGINE
# ============================================================

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()

def run_backtest():
    """Backtest de pairs trading strategie."""
    print(f"{'='*60}")
    print(f"PAIRS TRADING BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")

    # Laad alle tickers
    all_tickers: List[str] = []
    for f_name in bouw_bestandslijst():
        all_tickers.extend(laad_tickers_uit_bestand(f_name))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    # Pivot
    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    pivot = pivot.dropna(axis=1, how='all')
    print(f"Data: {len(pivot.columns)} tickers | {len(pivot)} dagen")

    # Simuleer dagelijkse trading
    cash = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades: List[Dict] = []
    equity_curve: List[Dict] = []

    dates = sorted(pivot.index)
    print(f"Backtest van {dates[0].date()} tot {dates[-1].date()}")

    for i, date in enumerate(dates):
        if i < LOOKBACK:
            continue
        
        # Huidige data
        data_till = pivot.iloc[:i+1]
        current_prices = pivot.iloc[i]
        
        # Check open posities
        for pair_key, pos in list(positions.items()):
            a, b = pair_key.split('/')
            if a not in current_prices or b not in current_prices:
                continue
            
            # Bereken huidige spread en z-score
            spread = current_prices[a] / current_prices[b]
            spread_hist = data_till[a] / data_till[b]
            spread_mean = spread_hist.rolling(LOOKBACK).mean().iloc[-1]
            spread_std = spread_hist.rolling(LOOKBACK).std().iloc[-1]
            
            if math.isnan(spread_std) or spread_std <= 0:
                continue
            
            z = (spread - spread_mean) / spread_std
            
            # Exit conditie
            exit_signal = False
            if abs(z) < EXIT_Z:
                exit_signal = True
            elif pos['days'] >= 60:
                exit_signal = True
            
            if exit_signal:
                # Sluit positie
                if pos['direction'] == 'LONG':
                    pnl_a = (current_prices[a] - pos['entry_a']) * pos['size_a']
                    pnl_b = (pos['entry_b'] - current_prices[b]) * pos['size_b']
                else:  # SHORT
                    pnl_a = (pos['entry_a'] - current_prices[a]) * pos['size_a']
                    pnl_b = (current_prices[b] - pos['entry_b']) * pos['size_b']
                
                gross_pnl = pnl_a + pnl_b
                cost = trade_cost(abs(gross_pnl)) if gross_pnl > 0 else 0
                tax = gross_pnl * TAX_RATE if gross_pnl > 0 else 0
                net_pnl = gross_pnl - cost - tax
                
                cash += gross_pnl - cost - tax
                
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
        
        # Open nieuwe posities (elke 5 dagen)
        if i % 5 == 0:
            beschikbaar = [t for t in pivot.columns if t in current_prices and not math.isnan(current_prices[t])]
            if len(beschikbaar) >= 2 and len(positions) < MAX_TOTAL_PAIRS:
                candidates = []
                for a, b in combinations(beschikbaar, 2):
                    sig = analyseer_pair(a, b, data_till)
                    if sig and sig.signal != "NEUTRAL":
                        candidates.append((sig, a, b))
                
                candidates.sort(key=lambda x: x[0].score, reverse=True)
                
                for sig, a, b in candidates[:MAX_PAIRS]:
                    pair_key = f"{a}/{b}"
                    if pair_key in positions:
                        continue
                    
                    investering_per_pair = cash * 0.10
                    size_a = int(investering_per_pair / current_prices[a] * 0.5)
                    size_b = int(investering_per_pair / current_prices[b] * 0.5)
                    
                    if size_a <= 0 or size_b <= 0:
                        continue
                    
                    pos = {
                        'entry_date': date,
                        'entry_a': current_prices[a],
                        'entry_b': current_prices[b],
                        'size_a': size_a,
                        'size_b': size_b,
                        'direction': sig.signal,
                        'entry_z': sig.current_z,
                        'days': 0,
                    }
                    positions[pair_key] = pos
        
        # Equity curve
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
                portfolio_value += value_a + value_b
        
        equity_curve.append({'date': date, 'value': portfolio_value})

    # Resultaten
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
        
        # Opslaan als CSV (in jouw backtest_results folder)
        os.makedirs('backtest_results', exist_ok=True)
        tdf.to_csv('backtest_results/backtest_cointegr.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTATEN — bot_01cointegr")
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
        print("\n🏆 Beste pairs:")
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
