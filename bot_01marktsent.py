#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01marktsent.py  —  MARKTSENTIMENT ENGINE v1.0

Combineert nieuwssentiment (VADER, via NewsAPI) met technische indicatoren
(RSI, MACD, ATR) tot één score per ticker, volgens hetzelfde patroon als
bot_00kr.py: dynamische 041x-059x scan, één Telegram-bericht per beurs,
één e-mailsamenvatting, Supabase-logging via db_logger, geen CSV.

BELANGRIJKE BEPERKING (lees dit voor je vertrouwt op de backtest-cijfers):
  NewsAPI's gratis "Developer"-tier levert maximaal 100 requests/dag en
  geeft via /v2/everything geen artikelen ouder dan ~1 maand. Historisch
  nieuwssentiment over meerdere jaren is dus niet beschikbaar zonder
  betaald abonnement. `run_backtest()` hieronder test daarom ALLEEN de
  technische component (RSI/MACD/ATR, walk-forward, met dezelfde
  kosten/fiscaliteit als bot_00kr). De sentiment-component wordt enkel
  LIVE toegepast, als filter/tiebreaker bovenop een reeds op zichzelf
  geteste technische score — niet als bewezen losse alpha-bron. Elke
  live-selectie wordt wel naar Supabase gelogd (incl. sentiment_score),
  zodat er na verloop van tijd een eigen forward-test-dataset ontstaat
  om de toegevoegde waarde van sentiment achteraf te evalueren.

  Wegens de 100 req/dag-limiet wordt ALLEEN gescand op de x-lijsten
  (tickers_041x.txt .. tickers_059x.txt, de reeds gescreende kwaliteits-
  aandelen), niet op de volledige a-lijsten — anders is de dagquota in
  een handomdraai op.

Gebruik:
  python bot_01marktsent.py live       # live rapport (technisch + sentiment)
  python bot_01marktsent.py backtest   # walk-forward backtest (technisch only)
"""

import os
import sys
import math
import time
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

START_CAPITAL        = 50_000.0
MAX_POSITIONS        = 10
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
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY", "")

NEWSAPI_DAILY_BUDGET = 95          # marge onder de 100/dag-limiet
_newsapi_calls_used   = 0
_newsapi_rate_limited = False  # blijft True voor de rest van de run na 1x HTTP 429

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
    """Zelfde dynamische patroon als bot_00kr/weekly_report: 041x t/m 059x,
    ontbrekende bestanden worden verderop overgeslagen."""
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

MS_CFG = {
    "rsi_period":         14,
    "rsi_oversold":       35.0,   # oversold-bounce kandidaat
    "rsi_momentum_low":   55.0,   # gezonde momentumzone: [low, high]
    "rsi_momentum_high":  70.0,   # boven deze grens = overbought, geen punt meer
    "macd_fast":          12,
    "macd_slow":          26,
    "macd_signal":        9,
    "macd_cross_lookback": 5,     # crossover moet binnen N dagen liggen (vers signaal)
    "sma_trend_period":   200,
    "sma_short_period":   50,
    "max_pct_above_sma_short": 15.0,  # max. % boven SMA50 -- filtert parabolische spikes
    "atr_period":         14,
    "sentiment_lookback_days": 7,
    "sentiment_min_articles":  2,
    # eindscore = technische score (0-4) + sentiment-bonus (-1..+1)
    # min_score=4.0 -> vereist ALLE vier technische criteria (RSI-zone,
    # verse MACD-crossover, boven SMA200, niet overextended t.o.v. SMA50).
    # Bij 3.0 kwalificeerde bijna elk aandeel in een brede uptrend al
    # (bv. 140 van de 368 Nasdaq/NYSE-tickers in één run) -- te weinig
    # onderscheidend. Sentiment kan een 3.0-kandidaat nog over de drempel
    # tillen (bonus tot +1.0), dus die blijft een geldige route naar 4.0.
    "min_score":        4.0,
}

BACKTEST_START = "2021-01-01"
BACKTEST_END   = dt.date.today().isoformat()


# ============================================================
# HULPFUNCTIES (identiek aan bot_00kr.py)
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


# ============================================================
# TECHNISCHE INDICATOREN (Wilder-smoothing, zelfde als bot_00kr)
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
# DATA DOWNLOAD (identiek aan bot_00kr.py)
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
# NIEUWSSENTIMENT (NewsAPI + VADER) — LIVE ONLY
# ============================================================

_vader = SentimentIntensityAnalyzer()
_company_name_cache: Dict[str, str] = {}

def resolve_company_name(ticker: str) -> str:
    """
    Zoekterm voor NewsAPI mag geen kale Yahoo-tickercode zijn (bv. 'PHIA.AS')
    -- die string komt vrijwel nooit letterlijk in nieuwsartikelen voor.
    Haalt de korte bedrijfsnaam op via yfinance (bv. 'Koninklijke Philips'),
    met een in-memory cache zodat dit maar 1x per ticker per run gebeurt.
    Valt terug op het deel vóór de beurssuffix als yfinance niets teruggeeft.
    """
    if ticker in _company_name_cache:
        return _company_name_cache[ticker]
    naam = ticker.split(".")[0]
    try:
        info = yf.Ticker(ticker).get_info()
        kandidaat = info.get("shortName") or info.get("longName")
        if kandidaat:
            naam = kandidaat
    except Exception as e:
        print(f"[WARN] Kon bedrijfsnaam voor {ticker} niet ophalen ({e}), val terug op '{naam}'")
    _company_name_cache[ticker] = naam
    return naam

def fetch_news_sentiment(ticker: str, company_hint: str = "") -> Tuple[Optional[float], int]:
    """
    Haalt recent nieuws op via NewsAPI /v2/everything en scoort met VADER.
    Retourneert (gemiddelde_compound_score, aantal_artikelen).
    Geeft (None, 0) terug als de dagquota op is, geen key is ingesteld,
    of er onvoldoende artikelen zijn (< MS_CFG["sentiment_min_articles"]).

    Let op: houdt zelf de dagquota bij (NEWSAPI_DAILY_BUDGET) — daarom
    ALLEEN aanroepen voor tickers die al door de technische score komen,
    nooit voor de volledige scanlijst.
    """
    global _newsapi_calls_used, _newsapi_rate_limited

    if not NEWSAPI_KEY:
        return None, 0
    if _newsapi_calls_used >= NEWSAPI_DAILY_BUDGET:
        return None, 0
    if _newsapi_rate_limited:
        # Al 1x een 429 gehad deze run -> het 24u-rolling-window-budget is
        # elders al verbruikt (bv. door een eerdere handmatige testrun
        # dezelfde dag). Verdere pogingen zijn zinloos en kosten alleen
        # runtime, dus we stoppen meteen voor de rest van deze run.
        return None, 0

    query   = company_hint or ticker
    since   = (dt.date.today() - dt.timedelta(days=MS_CFG["sentiment_lookback_days"])).isoformat()
    url     = "https://newsapi.org/v2/everything"
    params  = {
        "q": query, "from": since, "sortBy": "publishedAt",
        "language": "en", "pageSize": 20, "apiKey": NEWSAPI_KEY,
    }
    try:
        _newsapi_calls_used += 1
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            _newsapi_rate_limited = True
            print(f"[WARN] NewsAPI 429 (rate limited) -- 24u-quota elders al verbruikt. "
                  f"Stop met verdere sentiment-calls voor de rest van deze run.")
            return None, 0
        if resp.status_code != 200:
            print(f"[WARN] NewsAPI {resp.status_code} voor {ticker} (query='{query}'): {resp.text[:150]}")
            return None, 0
        articles = resp.json().get("articles", [])
    except Exception as e:
        print(f"[WARN] NewsAPI fout voor {ticker} (query='{query}'): {e}")
        return None, 0

    if len(articles) < MS_CFG["sentiment_min_articles"]:
        return None, len(articles)

    scores = []
    for art in articles:
        tekst = " ".join(filter(None, [art.get("title"), art.get("description")]))
        if not tekst.strip():
            continue
        scores.append(_vader.polarity_scores(tekst)["compound"])

    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)

def sentiment_bonus(score: Optional[float]) -> float:
    """Zet VADER-compound (-1..+1) om in een score-bonus (-1..+1),
    met een neutrale zone tussen -0.15 en +0.15 die geen bonus geeft."""
    if score is None:
        return 0.0
    if score >= 0.15:
        return min(1.0, score)
    if score <= -0.15:
        return max(-1.0, score)
    return 0.0


# ============================================================
# ANALYSE
# ============================================================

@dataclass
class MSSignaal:
    ticker: str
    price: float
    rsi_monthly: float
    macd_label: str
    tech_score: float
    sentiment_score: Optional[float]
    sentiment_n: int
    score: float
    atr: float
    stop: float

def analyse_ticker(ticker: str, df_ticker: pd.DataFrame, met_sentiment: bool = False) -> Optional[MSSignaal]:
    df = df_ticker.sort_values("Date").reset_index(drop=True)
    # Minimaal genoeg historiek voor een betrouwbare SMA200/252d-hoogtepunt.
    if len(df) < 210:
        return None

    close = df["Close"]
    high  = df["High"] if "High" in df.columns else close
    low   = df["Low"] if "Low" in df.columns else close

    rsi_series = compute_rsi_series(close, MS_CFG["rsi_period"])
    macd_line, signal_line, hist = compute_macd_series(
        close, MS_CFG["macd_fast"], MS_CFG["macd_slow"], MS_CFG["macd_signal"])
    atr_series = compute_atr_series(high, low, close, MS_CFG["atr_period"])
    sma_trend  = close.rolling(MS_CFG["sma_trend_period"], min_periods=MS_CFG["sma_trend_period"]).mean()
    sma_short  = close.rolling(MS_CFG["sma_short_period"], min_periods=MS_CFG["sma_short_period"]).mean()

    rsi    = safe_float(rsi_series.iloc[-1])
    macd   = safe_float(macd_line.iloc[-1])
    sig    = safe_float(signal_line.iloc[-1])
    h_now  = safe_float(hist.iloc[-1])
    h_prev = safe_float(hist.iloc[-2]) if len(hist) > 1 else float("nan")
    atr    = safe_float(atr_series.iloc[-1])
    price  = safe_float(close.iloc[-1])
    sma    = safe_float(sma_trend.iloc[-1])
    sma50  = safe_float(sma_short.iloc[-1])

    if math.isnan(rsi) or math.isnan(price) or math.isnan(atr):
        return None

    tech_score = 0.0

    # 1. RSI in gezonde zone: oversold-bounce OF gematigd momentum
    #    (bewust NIET meer "of overbought" -- dat sloot juist niets uit).
    if rsi <= MS_CFG["rsi_oversold"] or (MS_CFG["rsi_momentum_low"] <= rsi <= MS_CFG["rsi_momentum_high"]):
        tech_score += 1.0

    # 2. VERSE MACD bullish crossover (binnen macd_cross_lookback dagen),
    #    niet "staat toevallig al boven signal" -- dat is een blijvende
    #    toestand, geen gebeurtenis, en dus veel minder onderscheidend.
    diff = (macd_line - signal_line).tail(MS_CFG["macd_cross_lookback"] + 1)
    diff_prev = diff.shift(1)
    fresh_cross = bool(((diff_prev <= 0) & (diff > 0)).any())
    if fresh_cross:
        tech_score += 1.0
    macd_label = "Verse crossover" if fresh_cross else ("Bullish" if macd > sig else "Bearish")

    # 3. Onafhankelijke langetermijntrend: prijs boven SMA200.
    if not math.isnan(sma) and price > sma:
        tech_score += 1.0

    # 4. Niet overextended: prijs niet te ver boven het (traag bewegende)
    #    SMA50 -- een SMA reageert nauwelijks op één spike-dag, dus dit
    #    vangt parabolische/blow-off-bewegingen wel degelijk, in
    #    tegenstelling tot een "afstand tot 252d-hoogtepunt"-criterium
    #    (dat zichzelf als hoogtepunt meetelt en dus altijd >=0 geeft).
    if not math.isnan(sma50) and sma50 > 0:
        pct_above_sma50 = (price - sma50) / sma50 * 100.0
        if pct_above_sma50 <= MS_CFG["max_pct_above_sma_short"]:
            tech_score += 1.0

    sent_score, sent_n = (None, 0)
    if met_sentiment and NEWSAPI_KEY and not _newsapi_rate_limited:
        naam = resolve_company_name(ticker)
        sent_score, sent_n = fetch_news_sentiment(ticker, company_hint=naam)

    eind_score = tech_score + sentiment_bonus(sent_score)
    stop = price - ATR_STOP_MULT * atr

    return MSSignaal(
        ticker=ticker, price=price, rsi_monthly=rsi, macd_label=macd_label,
        tech_score=tech_score, sentiment_score=sent_score, sentiment_n=sent_n,
        score=eind_score, atr=atr, stop=stop,
    )


# ============================================================
# RAPPORTAGE
# ============================================================

def format_bericht(ex_name: str, signalen: List[MSSignaal]) -> str:
    if not signalen:
        return ""
    regels = [f"*MARKTSENTIMENT — {ex_name}*  ({today_str()})\n"]
    for s in signalen[:15]:
        sent_txt = f"{s.sentiment_score:+.2f} ({s.sentiment_n}art.)" if s.sentiment_score is not None else f"n/a ({s.sentiment_n}art.)"
        regels.append(
            f"• *{s.ticker}* — score {s.score:.1f} | RSI {s.rsi_monthly:.0f} | "
            f"MACD {s.macd_label} | sentiment {sent_txt} | EUR{s.price:.2f}"
        )
    return "\n".join(regels)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"MARKTSENTIMENT — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for f_name in bouw_bestandslijst():
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            continue
        ex_name = label_voor(f_name)
        exchange_tickers[ex_name] = tlist
        all_tickers.extend(tlist)
        print(f"  {ex_name}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden (verwacht tickers_041x.txt .. tickers_059x.txt).")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke kwaliteitstickers")
    print("Koersdata downloaden (2 jaar, nodig voor SMA200/252d-hoogtepunt)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen koersdata.")
        return

    email_delen: List[str] = []

    # ------------------------------------------------------------------
    # Stap 1: technische score voor ALLE beurzen eerst bepalen (gratis,
    # geen NewsAPI-call). We bouwen één globale kandidatenlijst met de
    # beurs erbij, i.p.v. per beurs apart te verwerken — anders wordt het
    # NewsAPI-dagbudget (95) al verbruikt door de eerste paar beurzen in
    # bestandsvolgorde (041, 042, ...) en blijft er niets over voor
    # latere beurzen zoals 057 NYSE, ook al zitten daar sterkere
    # kandidaten tussen.
    # ------------------------------------------------------------------
    globale_kandidaten: List[Tuple[str, MSSignaal]] = []  # (ex_name, signaal)
    for ex_name, tlist in exchange_tickers.items():
        df_ex = df[df["Ticker"].isin(tlist)].copy()
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group, met_sentiment=False)
            if sig is not None and sig.tech_score >= 2.0:
                globale_kandidaten.append((ex_name, sig))

    print(f"\nTechnische voorselectie klaar: {len(globale_kandidaten)} kandidaten over "
          f"{len(exchange_tickers)} beurzen (tech_score >= 2.0).")

    # ------------------------------------------------------------------
    # Stap 2: NewsAPI-budget gericht toewijzen. Grensgevallen
    # (tech_score == 2.0) hebben de sentiment-bonus NODIG om de
    # min_score-drempel van 3.0 te halen -- die krijgen voorrang op het
    # budget. Kandidaten die al op techniek alleen kwalificeren
    # (tech_score >= 3.0) krijgen sentiment als extra context, met wat
    # budget overblijft.
    # ------------------------------------------------------------------
    grensgevallen = [t for t in globale_kandidaten if t[1].tech_score < MS_CFG["min_score"]]
    al_gekwalificeerd = [t for t in globale_kandidaten if t[1].tech_score >= MS_CFG["min_score"]]
    al_gekwalificeerd.sort(key=lambda t: t[1].tech_score, reverse=True)
    verwerkingsvolgorde = grensgevallen + al_gekwalificeerd

    per_beurs: Dict[str, List[MSSignaal]] = {ex: [] for ex in exchange_tickers}
    for ex_name, kand in verwerkingsvolgorde:
        df_ex_ticker = df[(df["Ticker"] == kand.ticker) & (df["Ticker"].isin(exchange_tickers[ex_name]))]
        sig = analyse_ticker(kand.ticker, df_ex_ticker, met_sentiment=True)
        if sig is not None and sig.score >= MS_CFG["min_score"]:
            per_beurs[ex_name].append(sig)

    print(f"NewsAPI-calls gebruikt: {_newsapi_calls_used}/{NEWSAPI_DAILY_BUDGET} "
          f"({len(grensgevallen)} grensgevallen voorrang gegeven)\n")

    for ex_name, tlist in exchange_tickers.items():
        signalen = per_beurs[ex_name]
        signalen.sort(key=lambda s: s.score, reverse=True)
        if not signalen:
            continue
        print(f"Analyseren: {ex_name} ({len(tlist)} tickers)... "
              f"→ {len(signalen)} kandidaten (min_score {MS_CFG['min_score']})")
        for s in signalen:
            print(f"  ✓ {s.ticker}: score {s.score:.1f} | RSI={s.rsi_monthly:.1f} | "
                  f"sentiment={s.sentiment_score}")

        for s in signalen:
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_01marktsent",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "rsi_monthly": s.rsi_monthly,
                    "macd_label": s.macd_label,
                    "stop": s.stop,
                    "atr": s.atr,
                    "sentiment_score": s.sentiment_score,
                    "sentiment_n": s.sentiment_n,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, signalen)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd")

    if email_delen:
        send_email(
            f"Marktsentiment rapport {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )
    else:
        print("\nGeen kandidaten vandaag, geen berichten verstuurd.")

    print(f"{'='*60}")
    print("Klaar.")


# ============================================================
# WALK-FORWARD BACKTEST — TECHNISCHE COMPONENT (zie beperking bovenaan)
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"MARKTSENTIMENT BACKTEST (technisch, walk-forward)  {BACKTEST_START} -> {BACKTEST_END}")
    print("LET OP: sentiment zit hier NIET in — geen historisch nieuwsarchief")
    print("beschikbaar op de gratis NewsAPI-tier. Zie docstring bovenaan.")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for f_name in bouw_bestandslijst():
        all_tickers.extend(load_tickers_from_file(f_name))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    all_dates = sorted(df["Date"].dt.date.unique())
    cash      = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []

    # Maandelijkse scanmomenten, zelfde walk-forward-ritme als bot_00kr.
    scan_dates = []
    prev_month = None
    for d in all_dates:
        if d.month != prev_month:
            scan_dates.append(d)
            prev_month = d.month
    print(f"Scanmomenten: {len(scan_dates)} (maandelijks, out-of-sample t.o.v. elk scanmoment)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group, met_sentiment=False)
            if not sig or sig.tech_score < MS_CFG["min_score"]:
                continue
            entry = sig.price * (1 + SLIPPAGE_PCT)
            risico_eur   = cash * RISICO_PCT_PER_TRADE
            stop_afstand = ATR_STOP_MULT * sig.atr
            if math.isnan(stop_afstand) or stop_afstand <= 0:
                continue
            aandelen    = max(1, int(risico_eur / stop_afstand))
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue
            cash -= investering
            positions[ticker] = {
                "entry_date": scan_date, "entry_price": round(entry, 4),
                "size": aandelen, "stop": sig.stop, "days": 0,
                "cost": trade_cost(investering), "score": sig.tech_score,
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
                    "entry_price": pos["entry_price"], "exit_price": round(exit_slip, 4),
                    "size": pos["size"], "pnl": round(pnl, 2), "tax": round(tax, 2),
                    "net": round(pnl - tax, 2), "reason": reason, "days": pos["days"],
                })
                del positions[ticker]

    if trades:
        tdf   = pd.DataFrame(trades)
        n     = len(tdf)
        nwin  = (tdf["net"] > 0).sum()
        pf    = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
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
        print("(Dit is de technische score zonder sentiment — zie docstring.)")
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
