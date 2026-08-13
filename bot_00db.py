#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_00db.py  —  DARVAS BOX ENGINE v1.0
Nicolas Darvas' 'Box Theory' — dozen van opeenvolgende consolidatie.

Hoe Darvas Box werkt:
  Na een nieuwe koershoogte consolideert een aandeel binnen een vaste
  bandbreedte (de "doos"):
    - Box top    = de hoogste koers die N dagen niet meer doorbroken wordt
    - Box bottom = de laagste koers binnen diezelfde periode
    - Een doos is "bevestigd" zodra de koers minstens N handelsdagen
      binnen top en bottom blijft, zonder nieuwe high of lagere low
  Bij een opwaartse doorbraak boven de box top op verhoogd volume:
    → BREAKOUT, nieuwe doos begint te vormen op een hoger niveau
  Meerdere opeenvolgende, oplopende dozen ("staircase") zijn het
  sterkste signaal — vergelijkbaar met de VCP-trapsgewijze contracties,
  maar Darvas werkt met vaste horizontale niveaus i.p.v. procentuele
  contracties.

Score systeem (0-8):
  1. Minimum 1 bevestigde doos gedetecteerd
  2. Doos smal genoeg (top-bottom spreiding <= drempel)
  3. Volume droogt op binnen de doos t.o.v. periode ervoor
  4. Minimum 2 opeenvolgende dozen (staircase)
  5. Elke volgende doos hoger dan de vorige (oplopende staircase)
  6. Prijs binnen 10% van box top (net onder of op breakout-niveau)
  7. Stage 2 trend (Close > MA50 > MA150 > MA200)
  8. Breakout boven box top op verhoogd volume

Gebruik:
  python bot_00db.py live     # live rapport
  python bot_00db.py backtest # backtest modus

GitHub Actions: dagelijks om 19:55 UTC

Let op: dit bestand schrijft bewust GEEN CSV weg (in tegenstelling tot
oudere bots) — enkel Telegram + email, zelfde patroon als bot_00vcp.py.
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

# Vriendelijke namen voor de kwaliteits-lijsten (x-suffix), 041-059.
# Ontbrekende bestanden worden gewoon overgeslagen — dit is enkel voor labels.
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
    """
    Bouwt de volledige lijst van kwaliteits-tickerbestanden tickers_041x.txt
    t/m tickers_059x.txt. Nummers kunnen ontbreken; niet-bestaande bestanden
    worden verderop gewoon overgeslagen (zelfde patroon als weekly_report.py).
    """
    return [f"tickers_{n:03d}x.txt" for n in range(41, 60)]

def label_voor(f_name: str) -> str:
    return BEURS_NAMEN.get(f_name, f_name.replace(".txt", ""))

DB_CFG = {
    "min_box_days":          3,      # min. handelsdagen zonder nieuwe high/low om doos te bevestigen
    "max_box_days":          15,     # max. duur van een doos voordat hij "verouderd" is
    "max_box_pct":           12.0,   # max. spreiding (top-bottom)/top van een geldige doos
    "min_boxes":             2,      # min. aantal opeenvolgende dozen voor staircase-bonus
    "max_boxes":             5,      # max. aantal dozen dat we terugzoeken
    "vol_ma_period":         50,
    "vol_droogval_ratio":    0.85,   # volume in doos max 85% van volume vóór de doos
    "breakout_vol_mult":     1.5,    # breakout volume >= 1.5x gemiddelde
    "pivot_proximity_pct":   10.0,   # prijs binnen 10% van box top
    "ma_fast":               50,
    "ma_mid":                150,
    "ma_slow":               200,
    "atr_period":            14,
    "min_score":             4,
    "lookback_days":         150,
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
    for i in range(0, len(text), 4096):
        chunk = text[i:i + 4096]
        try:
            r = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                      "parse_mode": "Markdown", "disable_web_page_preview": True},
                timeout=10,
            )
            if r.status_code != 200:
                # Fallback zonder Markdown-opmaak bij parse-fouten
                requests.post(
                    url,
                    json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                          "disable_web_page_preview": True},
                    timeout=10,
                )
        except Exception as e:
            print(f"Telegram fout: {e}")
        if i + 4096 < len(text):
            time.sleep(1)

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

def sizing_tekst(ticker, prijs, stop, box_top, portfolio_waarde) -> str:
    entry       = prijs * (1 + SLIPPAGE_PCT)
    aandelen, max_loss = bereken_positie(portfolio_waarde, entry, stop)
    investering = round(entry * aandelen, 2)
    rr          = ((box_top - entry) / (entry - stop)) if (entry - stop) > 0 else 0
    return (
        f"  Entry: EUR{entry:.2f} | Stop: EUR{stop:.2f} | Box top: EUR{box_top:.2f}\n"
        f"  R/R: {rr:.1f}:1 | {aandelen} stuks | EUR{investering:,.2f} | Max verlies: EUR{max_loss:,.2f}"
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
# DARVAS BOX KERN: DOOS-DETECTIE
# ============================================================

@dataclass
class DarvasBox:
    nummer:    int
    top:       float
    bottom:    float
    pct:       float     # spreiding (top-bottom)/top in %
    duur:      int       # aantal handelsdagen bevestigd
    vol_gem:   float
    start_idx: int
    end_idx:   int

@dataclass
class DarvasResultaat:
    boxes:        List[DarvasBox]
    n_boxes:      int
    pct_smal_genoeg: bool
    vol_krimpt:   bool
    staircase_omhoog: bool   # elke volgende doos hoger dan de vorige
    huidige_box:  DarvasBox
    breakout:     bool
    breakout_vol: float
    near_top:     bool

def detect_darvas_boxes(
    close:  pd.Series,
    high:   pd.Series,
    low:    pd.Series,
    volume: pd.Series,
) -> Optional[DarvasResultaat]:
    n = len(close)
    if n < DB_CFG["lookback_days"] + 20:
        return None

    lb = DB_CFG["lookback_days"]
    c  = close.values[-lb:]
    h  = high.values[-lb:]
    l  = low.values[-lb:]
    v  = volume.values[-lb:]
    n_lb = len(c)

    vol_ma = pd.Series(v).rolling(DB_CFG["vol_ma_period"]).mean().values
    min_days = DB_CFG["min_box_days"]
    max_days = DB_CFG["max_box_days"]

    # Vind alle geldige dozen: een venster van [min_days..max_days] waarin
    # de koers een top (hoogste high) en bottom (laagste low) niet doorbreekt.
    boxes: List[DarvasBox] = []
    i = 0
    while i < n_lb - min_days and len(boxes) < DB_CFG["max_boxes"]:
        best_end = None
        best_top = h[i]
        best_bottom = l[i]
        # Rek het venster op zolang high/low binnen de eerste-dag-marge blijven
        for span in range(min_days, min(max_days, n_lb - i) + 1):
            window_h = h[i:i + span]
            window_l = l[i:i + span]
            top    = float(window_h.max())
            bottom = float(window_l.min())
            spread = (top - bottom) / top * 100 if top > 0 else 999
            if spread <= DB_CFG["max_box_pct"]:
                best_end = i + span - 1
                best_top = top
                best_bottom = bottom
            else:
                break
        if best_end is not None:
            spread = (best_top - best_bottom) / best_top * 100
            vol_gem = float(np.mean(v[i:best_end + 1]))
            boxes.append(DarvasBox(
                nummer=len(boxes) + 1,
                top=round(best_top, 4), bottom=round(best_bottom, 4),
                pct=round(spread, 2), duur=best_end - i + 1,
                vol_gem=round(vol_gem, 0),
                start_idx=i, end_idx=best_end,
            ))
            i = best_end + 1
        else:
            i += 1

    if not boxes:
        return None

    # Alleen de meest recente opeenvolgende dozen tellen mee voor de staircase
    recente_boxes = boxes[-DB_CFG["max_boxes"]:]

    pct_smal_genoeg = all(b.pct <= DB_CFG["max_box_pct"] for b in recente_boxes)
    vol_krimpt = True
    for idx in range(1, len(recente_boxes)):
        eerdere_vol = recente_boxes[idx - 1].vol_gem
        if eerdere_vol <= 0 or recente_boxes[idx].vol_gem > eerdere_vol * DB_CFG["vol_droogval_ratio"]:
            vol_krimpt = False
            break

    staircase_omhoog = len(recente_boxes) >= DB_CFG["min_boxes"] and all(
        recente_boxes[idx].bottom > recente_boxes[idx - 1].bottom
        for idx in range(1, len(recente_boxes))
    )

    huidige_box = boxes[-1]
    current_price = float(c[-1])
    current_vol   = float(v[-1])
    vol_ma_now    = safe_float(vol_ma[-1], 1.0)

    breakout     = current_price > huidige_box.top
    breakout_vol = (current_vol / vol_ma_now) if vol_ma_now > 0 else 0.0
    near_top     = ((huidige_box.top - current_price) / huidige_box.top * 100) <= DB_CFG["pivot_proximity_pct"]

    return DarvasResultaat(
        boxes=boxes, n_boxes=len(boxes),
        pct_smal_genoeg=pct_smal_genoeg, vol_krimpt=vol_krimpt,
        staircase_omhoog=staircase_omhoog, huidige_box=huidige_box,
        breakout=breakout, breakout_vol=round(breakout_vol, 2),
        near_top=near_top or breakout,
    )


# ============================================================
# STAGE 2 TREND CHECK
# ============================================================

def check_stage2(g: pd.DataFrame) -> Tuple[bool, str]:
    close = g["Close"]
    ma50  = close.rolling(DB_CFG["ma_fast"]).mean()
    ma150 = close.rolling(DB_CFG["ma_mid"]).mean()
    ma200 = close.rolling(DB_CFG["ma_slow"]).mean()

    c    = safe_float(close.iloc[-1])
    m50  = safe_float(ma50.iloc[-1])
    m150 = safe_float(ma150.iloc[-1])
    m200 = safe_float(ma200.iloc[-1])

    if any(math.isnan(x) for x in [c, m50, m150, m200]):
        return False, "onvoldoende data"

    ok = c > m50 > m150 > m200
    if ok:
        return True, f"✓ Close>MA{DB_CFG['ma_fast']}>MA{DB_CFG['ma_mid']}>MA{DB_CFG['ma_slow']}"
    else:
        return False, f"✗ Stage 2 vereist Close>MA{DB_CFG['ma_fast']}>MA{DB_CFG['ma_mid']}>MA{DB_CFG['ma_slow']}"


# ============================================================
# DARVAS SIGNAAL
# ============================================================

@dataclass
class DarvasSignaal:
    ticker:       str
    price:        float
    score:        int
    score_labels: List[str]
    db:           DarvasResultaat
    stage2:       bool
    stage2_label: str
    atr:          float
    stop:         float
    total_score:  float

def analyse_ticker(ticker: str, g: pd.DataFrame) -> Optional[DarvasSignaal]:
    try:
        g = g.sort_values("Date").copy()
        if len(g) < DB_CFG["ma_slow"] + DB_CFG["lookback_days"]:
            return None

        close  = g["Close"]
        high   = g["High"]
        low    = g["Low"]
        volume = g["Volume"]

        current_price = safe_float(close.iloc[-1])
        if current_price <= 0 or math.isnan(current_price):
            return None

        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([high - low, hcp, lcp], axis=1).max(axis=1)
        atr = safe_float(_wilder_smooth(tr, DB_CFG["atr_period"]).iloc[-1],
                         current_price * 0.02)

        db = detect_darvas_boxes(close, high, low, volume)
        if db is None:
            return None

        stage2, stage2_label = check_stage2(g)
        stop = db.huidige_box.bottom - (0.5 * atr)

        score  = 0
        labels = []

        def chk(ok: bool, ok_msg: str, fail_msg: str):
            nonlocal score
            if ok:
                score += 1
                labels.append(f"✓ {ok_msg}")
            else:
                labels.append(f"✗ {fail_msg}")

        chk(db.n_boxes >= 1,
            f"{db.n_boxes} doos/dozen gedetecteerd",
            "geen geldige doos gevonden")
        chk(db.pct_smal_genoeg,
            f"doos smal genoeg (<={DB_CFG['max_box_pct']:.0f}%)",
            "doos te breed")
        chk(db.vol_krimpt,
            "volume droogt op binnen de doos",
            "volume droogt NIET op")
        chk(db.n_boxes >= DB_CFG["min_boxes"],
            f"{db.n_boxes} opeenvolgende dozen (staircase, min {DB_CFG['min_boxes']})",
            f"slechts {db.n_boxes} doos/dozen")
        chk(db.staircase_omhoog,
            "elke doos hoger dan de vorige (staircase omhoog)",
            "geen oplopende staircase")
        chk(db.near_top,
            f"prijs nabij box top ({db.huidige_box.top:.2f})",
            f"prijs ver van box top ({db.huidige_box.top:.2f})")
        chk(stage2, stage2_label, stage2_label)
        chk(db.breakout and db.breakout_vol >= DB_CFG["breakout_vol_mult"],
            f"BREAKOUT boven {db.huidige_box.top:.2f} op {db.breakout_vol:.1f}x volume",
            f"geen breakout (vol={db.breakout_vol:.1f}x)")

        if score < DB_CFG["min_score"]:
            return None

        total_score = (
            score * 10
            + db.n_boxes * 5
            + (20 if db.breakout else 0)
            + (10 if stage2 else 0)
            + (5 if db.vol_krimpt else 0)
            + (5 if db.staircase_omhoog else 0)
        )

        return DarvasSignaal(
            ticker=ticker, price=round(current_price, 2),
            score=score, score_labels=labels,
            db=db, stage2=stage2, stage2_label=stage2_label,
            atr=round(atr, 4), stop=round(stop, 2),
            total_score=round(total_score, 1),
        )
    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 8) -> str:
    return "█" * score + "░" * (max_score - score) + f" {score}/{max_score}"

def format_bericht(
    exchange_name:    str,
    signalen:         List[DarvasSignaal],
    portfolio_waarde: float,
) -> Optional[str]:
    if not signalen:
        return None

    nu        = today_str()
    max_score = max(s.score for s in signalen)
    toon      = [s for s in signalen if s.score == max_score]
    top2      = signalen[:2]
    lbl       = {8: "🔥 PERFECT (8/8)", 7: "⭐ UITSTEKEND (7/8)",
                 6: "⚡ STERK (6/8)", 5: "📊 GOED (5/8)",
                 4: "📊 WATCHLIST (4/8)"}.get(max_score, "📊")

    delen = [
        f"📦 *Darvas Box — {exchange_name}*",
        f"_{nu} | {len(signalen)} kandidaten | {sum(s.db.breakout for s in signalen)} breakouts_",
        "─────────────────────────────",
        "🏆 *TOP 2:*",
    ]
    for s in top2:
        delen.append(
            f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} {_yahoo_link(s.ticker)}\n"
            f"  {s.db.n_boxes} doos/dozen | box {s.db.huidige_box.pct:.1f}% breed | "
            f"{'BREAKOUT' if s.db.breakout else 'setup'}\n"
            + sizing_tekst(s.ticker, s.price, s.stop, s.db.huidige_box.top, portfolio_waarde)
        )

    delen += ["─────────────────────────────", f"*{lbl}:*"]
    extra = [s for s in toon if s not in top2]
    if extra:
        for s in extra:
            delen.append(
                f"• `{s.ticker}` {_score_bar(s.score)} EUR{s.price:.2f} | "
                f"{s.db.n_boxes} doos/dozen | "
                f"{'BREAKOUT' if s.db.breakout else 'setup'}"
            )
    else:
        delen.append("_Zie top 2 hierboven_")

    delen.append(
        f"⚙️ _Min {DB_CFG['min_boxes']} dozen | "
        f"Breakout {DB_CFG['breakout_vol_mult']}x vol | Risico 5%_"
    )
    return "\n\n".join(delen)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"DARVAS BOX ENGINE — LIVE  {today_str()}")
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
    print("Data downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    print(f"Data geladen: {df['Ticker'].nunique()} tickers")
    portfolio_waarde = START_CAPITAL
    email_delen: List[str] = []

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name} ({len(tlist)} tickers)...")
        df_ex = df[df["Ticker"].isin(tlist)].copy()

        signalen: List[DarvasSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(ticker, group)
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: {sig.score}/8 | "
                    f"{sig.db.n_boxes} doos/dozen | "
                    f"box {sig.db.huidige_box.pct:.1f}% breed | "
                    f"{'BREAKOUT' if sig.db.breakout else 'setup'}"
                )

        signalen.sort(key=lambda s: s.total_score, reverse=True)
        print(f"  → {len(signalen)} Darvas kandidaten | {sum(s.db.breakout for s in signalen)} breakouts")

        for s in signalen:
            log_selectie(
                ticker=s.ticker,
                datum=today_str(),
                strategie="bot_00db",
                beurs=ex_name,
                koers=s.price,
                parameters={
                    "score": s.score,
                    "total_score": s.total_score,
                    "n_boxes": s.db.n_boxes,
                    "box_top": s.db.huidige_box.top,
                    "box_bottom": s.db.huidige_box.bottom,
                    "box_pct": s.db.huidige_box.pct,
                    "breakout": s.db.breakout,
                    "breakout_vol": s.db.breakout_vol,
                    "stage2": s.stage2,
                    "atr": s.atr,
                    "stop": s.stop,
                    "grafiek": f"https://finance.yahoo.com/quote/{s.ticker}",
                },
            )

        bericht = format_bericht(ex_name, signalen, portfolio_waarde)
        if bericht:
            send_telegram_message(bericht)
            email_delen.append(bericht)
            print(f"  → Telegram verstuurd: {ex_name}")
        else:
            print(f"  → Overgeslagen (geen signalen): {ex_name}")

    if email_delen:
        send_email(
            subject=f"Darvas Box rapport {today_str()}",
            body="\n\n" + ("=" * 40 + "\n\n").join(email_delen),
        )

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"DARVAS BOX BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
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

    all_dates  = sorted(df["Date"].dt.date.unique())
    cash       = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []
    price_map: Dict[str, float] = {}

    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(scan_date)].copy()

        price_map = {}
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
            elif close >= pos["tp"]:
                reason = f"TP ({pos['tp']:.2f})"
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
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   scan_date.isoformat(),
                    "ticker":      ticker,
                    "score":       pos["score"],
                    "n_boxes":     pos["n_boxes"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  round(exit_slip, 4),
                    "size":        pos["size"],
                    "pnl":         round(pnl, 2),
                    "tax":         round(tax, 2),
                    "net":         round(pnl - tax, 2),
                    "reason":      reason,
                    "days":        pos["days"],
                })
                del positions[ticker]

        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            sig = analyse_ticker(ticker, group)
            if not sig or not sig.db.breakout:
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
                "entry_date":  scan_date,
                "entry_price": round(entry, 4),
                "size":        aandelen,
                "stop":        sig.stop,
                "tp":          sig.db.huidige_box.top * 1.20,
                "score":       sig.score,
                "n_boxes":     sig.db.n_boxes,
                "days":        0,
                "cost":        trade_cost(investering),
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
