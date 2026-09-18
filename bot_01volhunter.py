#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_01volhunter.py  —  HOGE-ATR MEAN REVERSION SCANNER  v1.0

Doel: uit het VOLLEDIGE, ongefilterde tickeruniversum (tickers_0NNa.txt,
niet de kwaliteitsgescreende x-lijsten) per beurs de 3 aandelen selecteren
die (a) voldoende volatiel zijn om interessante bewegingen te maken, EN
(b) op dit moment tijdelijk laag staan binnen die eigen volatiliteit --
de "laag kopen"-kant van een laag-kopen/hoog-verkopen-aanpak op volatiele
aandelen.

TWEE FASES, in die volgorde (bewust -- eerst het universum versmallen,
dan pas de instapregel toepassen, i.p.v. omgekeerd zoals a_trade_combi
volatiliteit als bijkomstige filter gebruikt):

  FASE 1 — HOGE-ATR UNIVERSUM
    ATR14% = 14-daags Wilder-ATR / slotkoers x 100 (zelfde Wilder-smoothing
    als bot_00mr.py, voor consistentie binnen de repo). Enkel tickers met
    ATR14% >= MIN_ATR_PCT (default 6.0%) gaan door naar fase 2 -- dat is
    het "hoge-ATR universum" waar in het vorige gesprek naar gevraagd werd.
    Een zachte bovengrens (MAX_ATR_PCT, default 60%) sluit datafouten en
    quasi-gehalte/illiquide tickers uit.

  FASE 2 — INSTAPSIGNAAL BINNEN DAT UNIVERSUM (IBS + RSI3, zelfde
    formules als bot_00mr.py's IBS+RSI-systeem, hier toegepast op een
    volatielere, bredere ticker-pool i.p.v. diens vaste x-universum):
      - IBS (Internal Bar Strength) <= IBS_MAX (default 0.25): koers sloot
        dicht bij de daglaagte -- teken van verkoopdruk die kan keren.
      - RSI(3) <= RSI_MAX (default 15): kortetermijn oversold.
      - Trendfilter: koers > MA100 (default AAN, via VEREIST_UPTREND):
        enkel dips kopen binnen een bredere opwaartse trend, niet
        "vallende messen" in een structurele downtrend opvangen.

  SCORE = ATR14_pct * (1 - IBS) -- beloont zowel hogere volatiliteit
  (meer bewegingsruimte naar boven) als een dieper doorgezakte bar
  (dichter bij de laagte). Top 3 per beurs, ongeacht overige budgetten.

  STOP / TARGET (informatief, geen automatische orders):
    stop   = koers - 1.5 x ATR14
    target = koers + 2.5 x ATR14   (R-multiple ~1.67)

BELANGRIJKE BEPERKING: dit is, zoals de meeste "01"-scanners in deze repo,
een PURE SIGNAALGENERATOR -- hij houdt geen eigen open-posities/portfolio
bij en genereert dus ook geen automatische exit-signalen zodra de "hoog
verkopen"-kant bereikt is. De TP/stop hierboven zijn een eenmalige
richtwaarde bij instap, geen lopende trailing-berekening. Als er nadien
behoefte is aan een bot die dat wél doet (zoals bot_00mr.py voor zijn
eigen IBS+RSI-systeem), kan dat als aparte, tweede stap gebouwd worden.

BULK DOWNLOAD (i.p.v. per-ticker live calls): per beurs wordt de
tickerlijst in batches van BATCH_SIZE (default 150) via yf.download(...,
group_by="ticker") in bulk opgehaald -- dezelfde aanpak als bot_00mr.py's
download_eod(), met per-ticker fallback als de batch faalt. Dit is
bewust zo gekozen na de rate-limit-lessen gedocumenteerd in
a_trade_combi.py (voorheen a_trade_volatiel.py): op de a-lijsten
(~13.400 tickers) zou losse live-calls per ticker de sessie binnen
enkele minuten laten blokkeren door Yahoo Finance.

Draait wekelijks (zondag), NIET op zaterdag, om niet te overlappen met
de andere twee zware a-lijst-scans die al op zaterdag draaien
(bot_01beurssig_full 09:00 UTC, a_trade_combi 06:00 UTC) -- elke scan
krijgt zo zijn eigen dag om Yahoo's rate limits te laten resetten.

Env vars (zelfde secrets als de rest van de Fantasie-repo):
  SUPABASE_DB_URL, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
  EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER
  MIN_ATR_PCT       - ondergrens hoge-ATR universum (default 6.0)
  MAX_ATR_PCT       - bovengrens, sluit datafouten uit (default 60.0)
  IBS_MAX           - max Internal Bar Strength voor instap (default 0.25)
  RSI_MAX           - max RSI(3) voor instap (default 15.0)
  VEREIST_UPTREND   - "1"/"0", koers > MA100 vereist (default 1)
  TOP_N             - aantal picks per beurs (default 3)
  BATCH_SIZE        - tickers per bulk yf.download-call (default 150)

Gebruik:
  python bot_01volhunter.py live
"""

import os
import sys
import math
import time
import warnings
import datetime as dt
import smtplib
from dataclasses import dataclass
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
# CONFIG
# ============================================================

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_PASS       = os.getenv("EMAIL_PASS", "")
EMAIL_RECEIVER   = os.getenv("EMAIL_RECEIVER", "")

MIN_ATR_PCT     = float(os.getenv("MIN_ATR_PCT", "6.0"))
MAX_ATR_PCT     = float(os.getenv("MAX_ATR_PCT", "60.0"))
IBS_MAX         = float(os.getenv("IBS_MAX", "0.25"))
RSI_MAX         = float(os.getenv("RSI_MAX", "15.0"))
VEREIST_UPTREND = os.getenv("VEREIST_UPTREND", "1") == "1"
TOP_N           = int(os.getenv("TOP_N", "3"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "150"))

RSI_PERIOD = 3
ATR_PERIOD = 14
MA_TREND   = 100

BEURS_NAMEN = {
    "041": "041 Benelux Ierland",
    "042": "042 Parijs",
    "043": "043 Frankfurt",
    "044": "044 Spanje/Portugal",
    "045": "045 Londen",
    "046": "046 Milaan",
    "047": "047 Toronto",
    "048": "048 Nasdaq/NYSE",
    "049": "049 Stockholm",
    "050": "050 Zurich",
    "051": "051 Warschau",
    "052": "052 Oslo",
    "053": "053 Kopenhagen",
    "054": "054 Helsinki",
    "055": "055 CBoe",
    "056": "056 NYSE int",
    "057": "057 NYSE",
    "058": "058 TSXV",
    "059": "059 Oostenrijk Slovenie Slovakije",
}

def bouw_bestandslijst() -> List[str]:
    return [f"tickers_{n:03d}a.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    getal = f_name.replace("tickers_", "")[:3]
    return BEURS_NAMEN.get(getal, f_name.replace(".txt", ""))


# ============================================================
# HULPFUNCTIES
# ============================================================

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
    return f"https://finance.yahoo.com/quote/{ticker}"


# ============================================================
# INDICATOREN — Wilder-smoothing, zelfde formules als bot_00mr.py
# ============================================================

def _wilder(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period
            + valid.iloc[i] / period
        )
    return result

def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    close, high, low = g["Close"], g["High"], g["Low"]

    g["IBS"] = (close - low) / (high - low + 1e-9)

    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    rs    = _wilder(gain, RSI_PERIOD) / (_wilder(loss, RSI_PERIOD) + 1e-9)
    g["RSI3"] = 100 - (100 / (1 + rs))

    hl  = high - low
    hcp = (high - close.shift()).abs()
    lcp = (low - close.shift()).abs()
    tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    g["ATR14"] = _wilder(tr, ATR_PERIOD)
    g["ATR14_PCT"] = g["ATR14"] / close * 100

    g["MA100"] = close.rolling(MA_TREND).mean()
    g["HOOG20"] = high.rolling(20).max()
    return g


# ============================================================
# BULK DOWNLOAD — zelfde aanpak als bot_00mr.py's download_eod()
# ============================================================

def _normalise(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.reset_index() if df.index.name in ("Date", "Datetime") else df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None
    df["Ticker"] = ticker
    return df

def download_batch(tickers: List[str], period: str = "8mo") -> Dict[str, pd.DataFrame]:
    """Downloadt een batch tickers in bulk en geeft {ticker: dataframe} terug."""
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    try:
        data = yf.download(tickers, auto_adjust=True, group_by="ticker",
                            progress=False, threads=True, period=period)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                for t in tickers:
                    try:
                        norm = _normalise(data.xs(t, axis=1, level=0).copy(), t)
                        if norm is not None and len(norm) >= MA_TREND:
                            out[t] = norm
                    except Exception:
                        pass
            elif len(tickers) == 1:
                norm = _normalise(data, tickers[0])
                if norm is not None and len(norm) >= MA_TREND:
                    out[tickers[0]] = norm
    except Exception as e:
        print(f"  [WARN] batch-download mislukt ({len(tickers)} tickers): {e}")

    ontbrekend = [t for t in tickers if t not in out]
    for t in ontbrekend:
        try:
            raw = yf.download(t, period=period, auto_adjust=True, progress=False)
            norm = _normalise(raw, t)
            if norm is not None and len(norm) >= MA_TREND:
                out[t] = norm
            time.sleep(0.2)
        except Exception:
            continue
    return out


# ============================================================
# SIGNAAL
# ============================================================

@dataclass
class VolSignaal:
    ticker:     str
    price:      float
    atr:        float
    atr_pct:    float
    ibs:        float
    rsi3:       float
    ma100:      float
    pct_from_20d_high: float
    stop:       float
    tp:         float
    score:      float

def analyse_ticker_df(ticker: str, df: pd.DataFrame) -> Optional[VolSignaal]:
    try:
        g = add_indicators(df)
        row = g.iloc[-1]

        close  = safe_float(row["Close"])
        atr    = safe_float(row["ATR14"])
        atrp   = safe_float(row["ATR14_PCT"])
        ibs    = safe_float(row["IBS"])
        rsi3   = safe_float(row["RSI3"])
        ma100  = safe_float(row["MA100"])
        hoog20 = safe_float(row["HOOG20"])

        if any(math.isnan(x) for x in [close, atr, atrp, ibs, rsi3]) or close <= 0:
            return None

        # FASE 1 — hoge-ATR universum
        if not (MIN_ATR_PCT <= atrp <= MAX_ATR_PCT):
            return None

        # FASE 2 — instapsignaal
        if ibs > IBS_MAX or rsi3 > RSI_MAX:
            return None
        if VEREIST_UPTREND and not (not math.isnan(ma100) and close > ma100):
            return None

        pct_from_high = (close / hoog20 - 1) * 100 if not math.isnan(hoog20) and hoog20 > 0 else 0.0
        score = atrp * (1 - ibs)

        return VolSignaal(
            ticker=ticker, price=round(close, 2), atr=round(atr, 3),
            atr_pct=round(atrp, 2), ibs=round(ibs, 3), rsi3=round(rsi3, 1),
            ma100=round(ma100, 2) if not math.isnan(ma100) else 0.0,
            pct_from_20d_high=round(pct_from_high, 1),
            stop=round(close - 1.5 * atr, 2), tp=round(close + 2.5 * atr, 2),
            score=round(score, 2),
        )
    except Exception:
        return None


# ============================================================
# OUTPUT
# ============================================================

def format_bericht(exchange_name: str, signalen: List[VolSignaal], universum_grootte: int, hoge_atr_grootte: int) -> str:
    nu = today_str()
    delen = [
        f"🌊 *VOLATIELE DIP-SCAN — {exchange_name}*",
        f"_{nu} | {universum_grootte} tickers gescand | {hoge_atr_grootte} met ATR%>={MIN_ATR_PCT:.0f} | {len(signalen)} kandidaten_",
        "─────────────────────────────",
    ]
    for i, s in enumerate(signalen, start=1):
        delen.append(
            f"{i}. `{s.ticker}` | {s.price:.2f} | ATR%:{s.atr_pct:.1f} | "
            f"IBS:{s.ibs:.2f} | RSI3:{s.rsi3:.1f} | vs 20d-high:{s.pct_from_20d_high:+.1f}%\n"
            f"   🎯 richtdoel: {s.tp:.2f} | 🛑 richt-stop: {s.stop:.2f} | "
            f"[Grafiek]({_yahoo_link(s.ticker)})"
        )
    delen.append(
        f"⚙️ _Hoge-ATR universum: {MIN_ATR_PCT:.0f}-{MAX_ATR_PCT:.0f}% ATR14 | "
        f"Instap: IBS<={IBS_MAX:.2f} + RSI3<={RSI_MAX:.0f}"
        + (" + koers>MA100" if VEREIST_UPTREND else "") + "_"
    )
    delen.append(
        "⚠️ _Pure instapsignaal — geen automatische exit-tracking. "
        "Richtdoel/richt-stop zijn een eenmalige ATR-gebaseerde inschatting bij instap._"
    )
    return "\n\n".join(delen)


# ============================================================
# ENGINE
# ============================================================

def run_live():
    print("=" * 60)
    print(f"VOLATIELE DIP-SCAN (hoge-ATR mean reversion)  {today_str()}")
    print(f"MIN_ATR_PCT={MIN_ATR_PCT} MAX_ATR_PCT={MAX_ATR_PCT} "
          f"IBS_MAX={IBS_MAX} RSI_MAX={RSI_MAX} VEREIST_UPTREND={VEREIST_UPTREND}")
    print("=" * 60)

    bestanden = bouw_bestandslijst()
    email_delen: List[str] = []

    for f_name in bestanden:
        tlist = load_tickers_from_file(f_name)
        if not tlist:
            print(f"Bestand {f_name} niet gevonden of leeg, overslaan.")
            continue
        ex_name = label_voor(f_name)
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")

        alle_signalen: List[VolSignaal] = []
        hoge_atr_teller = 0

        for i in range(0, len(tlist), BATCH_SIZE):
            batch = tlist[i:i + BATCH_SIZE]
            data = download_batch(batch)
            for ticker, df in data.items():
                sig = analyse_ticker_df(ticker, df)
                if sig is not None:
                    alle_signalen.append(sig)
                    hoge_atr_teller += 1  # sig bestaat enkel als ATR-filter al gehaald is
            print(f"  batch {i // BATCH_SIZE + 1}/{(len(tlist) - 1) // BATCH_SIZE + 1} "
                  f"verwerkt ({len(data)}/{len(batch)} opgehaald)")

        alle_signalen.sort(key=lambda s: s.score, reverse=True)
        top = alle_signalen[:TOP_N]

        print(f"  → {len(top)} van {len(alle_signalen)} kandidaten in hoge-ATR universum")

        for rank, s in enumerate(top, start=1):
            log_selectie(
                ticker=s.ticker, datum=today_str(), strategie="bot_01volhunter",
                beurs=ex_name, koers=s.price,
                parameters={
                    "score": s.score, "rank": rank,
                    "atr": s.atr, "atr_pct": s.atr_pct,
                    "ibs": s.ibs, "rsi3": s.rsi3, "ma100": s.ma100,
                    "pct_from_high": s.pct_from_20d_high,
                    "stop": s.stop, "tp": s.tp,
                    "grafiek": _yahoo_link(s.ticker),
                },
            )

        if top:
            bericht = format_bericht(ex_name, top, len(tlist), hoge_atr_teller)
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print("  → Telegram verstuurd")
        else:
            print(f"  → Geen kandidaten, overgeslagen: {ex_name}")

    if email_delen:
        send_email(
            f"Volatiele Dip-Scan {today_str()}",
            "\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'=' * 60}\nKlaar.")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    run_live()
