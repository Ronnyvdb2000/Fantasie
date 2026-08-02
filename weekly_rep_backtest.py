#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weekly_rep_backtest.py

Backtest van de top 5 stijgers per beurs uit het wekelijkse
Hall-of-Fame-rapport (weekly_report.py): had de Trend Template + VCP +
momentum-logica (zoals gebruikt in bot_00super.py) deze bewegingen
vroeger al kunnen zien aankomen?

Werking:
  De tickerlijst komt niet meer hardcoded in dit bestand te staan, maar
  wordt ingelezen uit laatste_toppers.json — het bestand dat
  weekly_report.py bij elke run wegschrijft. Dat bestand wordt dus elke
  week volledig OVERSCHREVEN met de nieuwste top 5 per lijst; deze
  backtest test daardoor automatisch altijd de meest recente winnaars,
  nooit een oude/vaste lijst.

  Voor elke ticker wordt de koershistoriek gedownload en worden de
  laatste `--dagen-terug` handelsdagen weggeknipt VOORDAT de indicatoren
  worden berekend. Zo simuleer je wat de bot had gezien als hij enkele
  dagen vóór de grote beweging had gedraaid — in plaats van te "cheaten"
  met de koers van na de piek.

Belangrijke beperking t.o.v. bot_00super.py:
  De volledige RS-rating (percentiel-rank t.o.v. alle andere tickers op
  dezelfde beurs) vereist het downloaden van de hele beursuniverse
  (honderden tickers). Voor deze losse verkenning is dat overkill, dus
  dit script gebruikt een eenvoudigere momentum-proxy (% verandering
  over 3 en 12 maanden) in plaats van de RS-percentiel.

Vereist: yfinance, pandas, requests (met internettoegang naar Yahoo
Finance). Dit is NIET beschikbaar in de Claude-sandbox — draai dit lokaal.

Gebruik:
  python weekly_rep_backtest.py                  # default: 5 handelsdagen terug
  python weekly_rep_backtest.py --dagen-terug 15  # 3 weken terug
"""

import argparse
import json
import math
import os
import warnings

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TOPPERS_BESTAND = "laatste_toppers.json"


def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID:
        print("Telegram configuratie ontbreekt.")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram fout: {e}")


def stuur_telegram_lang(volledige_tekst, limiet=4000):
    """Zelfde chunking-logica als in weekly_report.py — nooit halverwege
    een sectie opknippen als het bericht toch te lang wordt."""
    if len(volledige_tekst) <= limiet:
        stuur_telegram(volledige_tekst)
        return

    secties = volledige_tekst.split("\n\n")
    huidig = ""
    for sectie in secties:
        kandidaat = (huidig + "\n\n" + sectie) if huidig else sectie
        if len(kandidaat) > limiet:
            if huidig:
                stuur_telegram(huidig)
            huidig = sectie
        else:
            huidig = kandidaat
    if huidig:
        stuur_telegram(huidig)


def laad_toppers():
    if not os.path.exists(TOPPERS_BESTAND):
        return None
    with open(TOPPERS_BESTAND, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["beurs"], d["ticker"], d["week_perf"]) for d in data]

CFG = {
    "vcp_lookback": 60,
    "vol_contraction_pct": 20.0,
    "volume_dry_pct": 30.0,
    "pivot_lookback": 20,
    "pivot_breakout_vol": 1.5,
}


def wilder_smooth(series, period):
    result = pd.Series(index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < period:
        return result
    result[valid.index[period - 1]] = valid.iloc[:period].mean()
    for i in range(period, len(valid)):
        result[valid.index[i]] = (
            result[valid.index[i - 1]] * (period - 1) / period + valid.iloc[i] / period
        )
    return result


def fmt(val, suffix=""):
    return "n.v.t." if val is None else f"{val:.1f}{suffix}"


def analyseer(ticker, dagen_terug):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None

        if dagen_terug > 0:
            df = df.iloc[:-dagen_terug]
        if len(df) < 260:
            return None

        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()
        ma200_slope = ma200.diff(20)
        vol_ma20 = vol.rolling(20).mean()
        high52 = close.rolling(252).max()
        low52 = close.rolling(252).min()

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = wilder_smooth(tr, 14)

        c = float(close.iloc[-1])

        def safe(x):
            try:
                v = float(x)
                return None if math.isnan(v) else v
            except Exception:
                return None

        s_ma50, s_ma150, s_ma200 = safe(ma50.iloc[-1]), safe(ma150.iloc[-1]), safe(ma200.iloc[-1])
        s_slope = safe(ma200_slope.iloc[-1])

        trend_ok = (
            s_ma50 is not None and s_ma150 is not None and s_ma200 is not None and s_slope is not None
            and c > s_ma150 and c > s_ma200 and s_ma150 > s_ma200 and s_slope > 0
            and s_ma50 > s_ma150 > s_ma200 and c > s_ma50
        )

        h52, l52 = safe(high52.iloc[-1]), safe(low52.iloc[-1])
        pct_from_high = round((h52 - c) / h52 * 100, 1) if h52 else None
        pct_from_low = round((c - l52) / l52 * 100, 1) if l52 else None

        mom_3m = round((c - close.iloc[-63]) / close.iloc[-63] * 100, 1) if len(close) > 63 else None
        mom_12m = round((c - close.iloc[-252]) / close.iloc[-252] * 100, 1) if len(close) > 252 else None

        # VCP-achtige score (0-4)
        vcp_score = 0
        lb = CFG["vcp_lookback"]
        atr_now, atr_start = safe(atr14.iloc[-1]), safe(atr14.iloc[-lb])
        if atr_now is not None and atr_start and atr_start > 0:
            contraction = (atr_start - atr_now) / atr_start * 100
            if contraction >= CFG["vol_contraction_pct"]:
                vcp_score += 1

        vol_now, vol_mean = safe(vol.iloc[-1]), safe(vol_ma20.iloc[-1])
        if vol_now is not None and vol_mean and vol_mean > 0:
            if (vol_now / vol_mean * 100) <= (100 - CFG["volume_dry_pct"]):
                vcp_score += 1

        recent = df.iloc[-lb:]
        n = len(recent)
        third = n // 3
        if third >= 5:
            r1, r2, r3 = recent.iloc[:third]["Close"], recent.iloc[third:2*third]["Close"], recent.iloc[2*third:]["Close"]
            range1, range2, range3 = float(r1.max()-r1.min()), float(r2.max()-r2.min()), float(r3.max()-r3.min())
            if range1 > 0 and range2 < range1 and range3 < range2:
                vcp_score += 1

        piv = CFG["pivot_lookback"]
        very_recent = df.iloc[-piv:]
        pivot_high = float(very_recent["Close"].iloc[:-1].max())
        vol_recent_mean = safe(vol_ma20.iloc[-piv])
        if c > pivot_high and vol_recent_mean and vol_recent_mean > 0:
            if vol.iloc[-1] >= vol_recent_mean * CFG["pivot_breakout_vol"]:
                vcp_score += 1

        return {
            "close": round(c, 2),
            "trend_ok": trend_ok,
            "vcp_score": vcp_score,
            "pct_from_high": pct_from_high,
            "pct_from_low": pct_from_low,
            "mom_3m": mom_3m,
            "mom_12m": mom_12m,
            "laatste_datum": df.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        print(f"Fout bij {ticker}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dagen-terug", type=int, default=5,
        help="Aantal handelsdagen weggeknipt vóór analyse, om de piekweek zelf "
             "niet mee te nemen (default: 5 = 1 handelsweek)."
    )
    args = parser.parse_args()

    toppers = laad_toppers()
    if not toppers:
        melding = (
            f"⚠️ *Backtest overgeslagen:* `{TOPPERS_BESTAND}` niet gevonden of leeg. "
            f"Draai eerst weekly_report.py."
        )
        print(melding)
        stuur_telegram(melding)
        return

    print(f"{'Beurs':<16}{'Ticker':<11}{'Weekperf':>9}  {'Trend':<6}{'VCP':<5}"
          f"{'%<high':>8}{'%>low':>8}{'Mom3m':>8}{'Mom12m':>8}  Analysedatum")
    print("-" * 105)

    regels_tg = []

    for beurs, ticker, week_perf in toppers:
        r = analyseer(ticker, args.dagen_terug)
        if r is None:
            print(f"{beurs:<16}{ticker:<11}{week_perf:>8.1f}%  geen data / te weinig historiek")
            regels_tg.append(f"• `{ticker}` ({beurs}): +{week_perf:.1f}% — geen data")
            continue
        print(
            f"{beurs:<16}{ticker:<11}{week_perf:>8.1f}%  "
            f"{'JA' if r['trend_ok'] else 'nee':<6}{r['vcp_score']}/4  "
            f"{fmt(r['pct_from_high'], '%'):>7}  {fmt(r['pct_from_low'], '%'):>7}  "
            f"{fmt(r['mom_3m'], '%'):>7}  {fmt(r['mom_12m'], '%'):>7}  {r['laatste_datum']}"
        )
        trend_icoon = "✅" if r["trend_ok"] else "❌"
        regels_tg.append(
            f"• `{ticker}` ({beurs}): +{week_perf:.1f}% | Trend:{trend_icoon} "
            f"VCP:{r['vcp_score']}/4 | Mom3m:{fmt(r['mom_3m'], '%')}"
        )

    print("\nLegende:")
    print("  Trend  = voldoet aan Trend Template (JA/nee), vóór de piekweek")
    print("  VCP    = VCP-achtige score 0-4 (ATR-contractie, volume-droogte, pullbacks, breakout)")
    print("  %<high = % onder 52w-high | %>low = % boven 52w-low")
    print("  Mom3m/12m = momentum-proxy (i.p.v. volledige RS-rank t.o.v. beursuniverse)")

    kop = (
        f"🔍 *BACKTEST — HADDEN WE DIT KUNNEN ZIEN?*\n"
        f"_Indicatoren berekend {args.dagen_terug} handelsdag(en) vóór de piekweek_\n"
        "==================================\n\n"
    )
    voet = (
        "\n\n💡 Trend✅ + hoge VCP-score = zat al vroeger in een gezonde setup. "
        "Trend❌ + lage VCP = puur nieuws-gedreven, niet te voorspellen via technische signalen."
    )
    bericht = kop + "\n".join(regels_tg) + voet
    stuur_telegram_lang(bericht)
    print("\nBacktest-samenvatting verzonden naar Telegram.")


if __name__ == "__main__":
    main()
