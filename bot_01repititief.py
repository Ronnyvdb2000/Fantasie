#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01repititief.py  —  SEIZOENSEFFECTEN ENGINE v1.0
Combineert weekdag-, maand- en kwartaaleffecten per ticker tot één
seizoensscore, met live signalen (Telegram/mail) én de onderliggende
historische statistiek (gemiddelde, win rate, p-waarde) mee in het bericht.

Opgebouwd 1-op-1 naar het patroon van bot_00kr.py: zelfde bestandslijst-
opbouw, zelfde download_history/telegram/email-hulpfuncties, zelfde
db_logger-integratie, zelfde live/backtest CLI.

Gebruik:
  python bot_01repititief.py live      # live rapport
  python bot_01repititief.py backtest  # walk-forward backtest (maand-effect)
"""

import os
import sys
import math
import warnings
import datetime as dt
import time
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy import stats

try:
    from db_logger import log_selectie
except Exception as _e:
    print(f"[WARN] db_logger niet beschikbaar ({_e}) — DB-logging wordt overgeslagen")
    def log_selectie(*args, **kwargs):
        return False

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG
# ============================================================

START_CAPITAL    = 50_000.0
SLIPPAGE_PCT     = 0.001
TRADE_COST_FIXED = 15.0
TRADE_COST_PCT   = 0.0035
TAX_RATE         = 0.10

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

# Zelfde namenlijst/bestandsopbouw als bot_00kr.py
BEURS_NAMEN = {
    "tickers_041x.txt": "041 Benelux Ierland",
    "tickers_042x.txt": "042 Parijs",
    "tickers_043x.txt": "043 Frankfurt",
    "tickers_044x.txt": "044 Spanje/Portugal",
    "tickers_045x.txt": "045 Londen",
    "tickers_046x.txt": "046 Milaan",
    "tickers_047x.txt": "047 Toronto",
    "tickers_048x.txt": "048 Nasdaq/NYSE",
    "tickers_049x.txt": "049 Stockholm",
    "tickers_050x.txt": "050 Zurich",
    "tickers_051x.txt": "051 Warschau",
    "tickers_052x.txt": "052 Oslo",
    "tickers_053x.txt": "053 Kopenhagen",
    "tickers_054x.txt": "054 Helsinki",
    "tickers_055x.txt": "055 CBoe",
    "tickers_056x.txt": "056 NYSE int",
    "tickers_057x.txt": "057 NYSE",
    "tickers_058x.txt": "058 TSXV",
    "tickers_059x.txt": "059 Oostenrijk Slovenie Slovakije",
}

def bouw_bestandslijst() -> List[str]:
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

SZ_CFG = {
    "lookback_years": 15,   # hoeveel historiek max wordt opgehaald
    "min_jaren":      5,    # minimum aantal jaar data om een ticker mee te nemen
    "p_drempel":      0.10, # significantiedrempel (tweezijdige t-test tegen 0)
    "top_n":          8,    # aantal tickers per beurs in het Telegram-bericht
}

WEEKDAGEN = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag"]
MAANDEN   = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]

BACKTEST_START      = "2021-01-01"
BACKTEST_END        = dt.date.today().isoformat()
BACKTEST_MIN_TRAIN  = 5   # jaar trainingsdata vereist vóór een testjaar meetelt


# ============================================================
# HULPFUNCTIES — 1-op-1 uit bot_00kr.py
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
# DATA DOWNLOAD — 1-op-1 uit bot_00kr.py
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

def download_history(tickers: List[str], period: str = "15y") -> pd.DataFrame:
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
# SEIZOENSANALYSE
# ============================================================

@dataclass
class SeizoenSignaal:
    ticker:             str
    price:              float
    score:              float   # gewogen som van significante gemiddelde rendementen
    weekdag:            Optional[dict]
    maand:              Optional[dict]
    kwartaal:           Optional[dict]
    jaren_data:         float
    aantal_significant: int

def bereken_bucket_stats(rendementen: np.ndarray) -> Optional[dict]:
    """(gemiddelde, win_rate, p_waarde, n) voor een reeks dagrendementen."""
    n = len(rendementen)
    if n < 5:
        return None
    gemiddelde = float(np.mean(rendementen))
    win_rate   = float(np.mean(rendementen > 0))
    _, p_waarde = stats.ttest_1samp(rendementen, 0.0)
    return {"gemiddelde": gemiddelde, "win_rate": win_rate, "p_waarde": float(p_waarde), "n": n}

def analyseer_ticker(ticker: str, df_ticker: pd.DataFrame) -> Optional[SeizoenSignaal]:
    try:
        g = df_ticker.sort_values("Date").copy()
        if len(g) < 250 * SZ_CFG["min_jaren"]:
            return None

        g = g.set_index("Date")
        rendement = g["Close"].pct_change().dropna()
        aantal_jaren = (rendement.index[-1] - rendement.index[0]).days / 365.25
        if aantal_jaren < SZ_CFG["min_jaren"]:
            return None

        current_price = safe_float(g["Close"].iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        vandaag = dt.datetime.now(dt.timezone.utc)
        huidige_weekdag = vandaag.weekday()               # 0=maandag ... 4=vrijdag
        huidige_maand   = vandaag.month
        huidig_kwartaal = (vandaag.month - 1) // 3 + 1

        weekdag_stats  = bereken_bucket_stats(rendement[rendement.index.weekday == huidige_weekdag].values)
        maand_stats    = bereken_bucket_stats(rendement[rendement.index.month == huidige_maand].values)
        kwartaal_stats = bereken_bucket_stats(rendement[rendement.index.quarter == huidig_kwartaal].values)

        onderdelen  = {"weekdag": weekdag_stats, "maand": maand_stats, "kwartaal": kwartaal_stats}
        significant = {k: v for k, v in onderdelen.items() if v and v["p_waarde"] < SZ_CFG["p_drempel"]}

        if not significant:
            return None  # geen enkel significant seizoenspatroon -> niet meenemen

        score = sum(v["gemiddelde"] * (1 - v["p_waarde"]) for v in significant.values())

        return SeizoenSignaal(
            ticker=ticker, price=round(current_price, 2), score=score,
            weekdag=weekdag_stats, maand=maand_stats, kwartaal=kwartaal_stats,
            jaren_data=round(aantal_jaren, 1), aantal_significant=len(significant),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# TELEGRAM + EMAIL OUTPUT — één bericht per exchange
# ============================================================

def _bucket_regel(key: str, stat: Optional[dict]) -> Optional[str]:
    if not stat or stat["p_waarde"] >= SZ_CFG["p_drempel"]:
        return None
    vandaag = dt.datetime.now(dt.timezone.utc)
    if key == "weekdag":
        kort, label = "wk", WEEKDAGEN[vandaag.weekday()]
    elif key == "maand":
        kort, label = "mnd", MAANDEN[vandaag.month - 1]
    else:
        kort, label = "kw", f"Q{(vandaag.month - 1)//3 + 1}"
    return (f"{kort} {label}: {stat['gemiddelde']*100:+.2f}% "
            f"(winrate {stat['win_rate']*100:.0f}%, n={stat['n']}, p={stat['p_waarde']:.3f})")

def sig_regel(s: SeizoenSignaal) -> str:
    onderbouwing = [r for r in (
        _bucket_regel("weekdag", s.weekdag),
        _bucket_regel("maand", s.maand),
        _bucket_regel("kwartaal", s.kwartaal),
    ) if r]
    return (
        f"• `{s.ticker}` score {s.score*100:+.2f}% | EUR{s.price:.2f} | "
        f"{s.jaren_data:.0f}j data | {_yahoo_link(s.ticker)}\n"
        f"  {' | '.join(onderbouwing)}"
    )

def format_bericht(exchange_name: str, top: List[SeizoenSignaal], alle: List[SeizoenSignaal]) -> Optional[str]:
    """Eén bericht per exchange. Lege exchanges -> None."""
    if not alle:
        return None
    nu = today_str()
    delen = [
        f"📅 *SEIZOENSEFFECTEN — {exchange_name}*",
        f"_{nu} | {len(alle)} geanalyseerd | {len(top)} met significant seizoenspatroon_",
        "─────────────────────────────",
        "\n\n".join(sig_regel(s) for s in top),
        "─────────────────────────────",
        f"⚙️ _Weekdag/maand/kwartaal t.o.v. eigen historiek (max {SZ_CFG['lookback_years']}j) | "
        f"p<{SZ_CFG['p_drempel']:.2f} | score = gewogen som significante gemiddeldes_",
    ]
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"SEIZOENSEFFECTEN — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for f_name in bouw_bestandslijst():
        tlist = load_tickers_from_file(f_name)
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
    print(f"Data downloaden ({SZ_CFG['lookback_years']} jaar)...")
    df = download_history(all_tickers, period=f"{SZ_CFG['lookback_years']}y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers, {len(df)} rijen")
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        alle: List[SeizoenSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyseer_ticker(ticker, group)
            if sig is not None:
                alle.append(sig)
                print(f"  ✓ {ticker}: score {sig.score*100:+.2f}% | "
                      f"{sig.aantal_significant} significante bucket(s)")

        if not alle:
            print(f"  → Overgeslagen: {ex_name} (geen significante patronen)")
            continue

        alle.sort(key=lambda s: s.score, reverse=True)
        top = alle[:SZ_CFG["top_n"]]

        for s in top:
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01repititief",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                    "jaren_data": s.jaren_data,
                    "weekdag_gemiddelde": s.weekdag["gemiddelde"] if s.weekdag else None,
                    "weekdag_winrate":    s.weekdag["win_rate"] if s.weekdag else None,
                    "weekdag_p":          s.weekdag["p_waarde"] if s.weekdag else None,
                    "maand_gemiddelde":   s.maand["gemiddelde"] if s.maand else None,
                    "maand_winrate":      s.maand["win_rate"] if s.maand else None,
                    "maand_p":            s.maand["p_waarde"] if s.maand else None,
                    "kwartaal_gemiddelde": s.kwartaal["gemiddelde"] if s.kwartaal else None,
                    "kwartaal_winrate":    s.kwartaal["win_rate"] if s.kwartaal else None,
                    "kwartaal_p":          s.kwartaal["p_waarde"] if s.kwartaal else None,
                },
            )

        bericht = format_bericht(ex_name, top, alle)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")

    if email_delen:
        send_email(
            f"Seizoenseffecten rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE — walk-forward op het maand-effect
# ============================================================
# Enkel het maand-effect wordt hier effectief "verhandeld": voor elk testjaar
# wordt de maand-bucket-statistiek herberekend op basis van UITSLUITEND de
# jaren die vóór dat testjaar liggen (expanding window, geen look-ahead).
# Is de maand voor een ticker significant positief op basis van die
# trainingsdata, dan wordt een positie geopend op de eerste handelsdag van
# de maand en gesloten op de laatste, inclusief slippage/kosten/taks.
# Weekdag- en kwartaaleffect worden in live-mode wél getoond als extra
# onderbouwing, maar hier niet apart backtest — laat het weten als je die
# ook wil, dat vergt een vergelijkbare tweede walk-forward-loop.

def _train_maand_stats(rendement: pd.Series, test_jaar: int) -> Dict[int, dict]:
    train = rendement[rendement.index.year < test_jaar]
    if train.empty:
        return {}
    jaren_train = (train.index[-1] - train.index[0]).days / 365.25
    if jaren_train < BACKTEST_MIN_TRAIN:
        return {}
    resultaat = {}
    for maand in range(1, 13):
        stat = bereken_bucket_stats(train[train.index.month == maand].values)
        if stat and stat["p_waarde"] < SZ_CFG["p_drempel"] and stat["gemiddelde"] > 0:
            resultaat[maand] = stat
    return resultaat

def run_backtest():
    print(f"{'='*60}")
    print(f"SEIZOENSEFFECTEN BACKTEST (maand-effect)  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for f_name in bouw_bestandslijst():
        all_tickers.extend(load_tickers_from_file(f_name))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden ({SZ_CFG['lookback_years']}y)...")
    df = download_history(all_tickers, period=f"{SZ_CFG['lookback_years']}y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    cash = START_CAPITAL
    trades: List[Dict] = []
    testjaren = range(pd.Timestamp(BACKTEST_START).year, pd.Timestamp(BACKTEST_END).year + 1)

    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date").set_index("Date")
        rendement = g["Close"].pct_change().dropna()
        if rendement.empty:
            continue

        for test_jaar in testjaren:
            kansrijke_maanden = _train_maand_stats(rendement, test_jaar)
            if not kansrijke_maanden:
                continue

            for maand, stat in kansrijke_maanden.items():
                maand_data = g[(g.index.year == test_jaar) & (g.index.month == maand)]
                if len(maand_data) < 5:
                    continue

                entry_price = safe_float(maand_data["Close"].iloc[0]) * (1 + SLIPPAGE_PCT)
                exit_price  = safe_float(maand_data["Close"].iloc[-1]) * (1 - SLIPPAGE_PCT)
                if math.isnan(entry_price) or math.isnan(exit_price) or entry_price <= 0:
                    continue

                risico_bedrag = cash * 0.05
                aandelen = max(1, int(risico_bedrag / entry_price))

                investering = entry_price * aandelen + trade_cost(entry_price * aandelen)
                gross = exit_price * aandelen
                cost  = trade_cost(gross)
                pnl   = gross - cost - investering
                tax   = pnl * TAX_RATE if pnl > 0 else 0.0
                net   = pnl - tax

                trades.append({
                    "ticker": ticker, "jaar": test_jaar, "maand": maand,
                    "train_gemiddelde": stat["gemiddelde"], "train_p": stat["p_waarde"],
                    "entry_price": round(entry_price, 4), "exit_price": round(exit_price, 4),
                    "size": aandelen, "pnl": round(pnl, 2), "tax": round(tax, 2),
                    "net": round(net, 2),
                })

    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("seizoen_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        totaal_net = tdf["net"].sum()
        print(f"\n{'='*60}")
        print(f"Trades (maand-effect) : {n}")
        print(f"Winnaars              : {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor         : {pf:.2f}")
        print(f"Netto resultaat       : EUR{totaal_net:,.2f} (som over alle posities, geen samengesteld kapitaal)")
        print(f"Belasting             : EUR{tdf['tax'].sum():,.2f}")
        print(f"{'='*60}")
        print("Opgeslagen: seizoen_backtest_trades.csv")
    else:
        print("Geen trades gegenereerd (geen enkele ticker/maand-combinatie was significant én out-of-sample bruikbaar).")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
