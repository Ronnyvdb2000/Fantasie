from __future__ import annotations

import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import logging
import time

# ---------------------------------------------------------------------------
# CONFIG & LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown"}, timeout=30)
        return r.status_code == 200
    except: return False

# ---------------------------------------------------------------------------
# INDICATOREN
# ---------------------------------------------------------------------------
def bereken_indicatoren(df: pd.DataFrame, s: int, t: int, is_hyper: bool) -> tuple:
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    # Moving Averages
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema100 = p.ewm(span=100, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    # RSI & ATR
    delta = p.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    tr = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    # MRS / Munger Specials
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    lower_b3 = ma20 - (2.2 * std20)  
    ibs = (p - l) / (h - l + 1e-10)  
    ma5 = p.rolling(5).mean()

    # ADX voor trendsterkte
    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    if is_hyper:
        p_rank = delta.rolling(100).apply(lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100 if len(x) > 0 else 50, raw=True)
        rsi_val = (rsi_val + p_rank) / 2

    return p, f_line, s_line, ema100, vol_ma, rsi_val, atr, adx, ibs, lower_b3, ma5

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_backtest(bestandsnaam: str, sector_naam: str) -> None:
    if not os.path.exists(bestandsnaam):
        logger.info(f"Bestand {bestandsnaam} niet gevonden. Overslaan.")
        return
    
    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    logger.info(f"Start analyse: {sector_naam} ({len(tickers)} tickers)")
    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except: return

    inzet = 2500.0
    res = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRS": 0.0}
    kosten = 15.0 + (inzet * 0.0035)

    for ticker in tickers:
        try:
            t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all') if len(tickers) > 1 else raw_df.dropna()
            if len(t_data) < 200: continue

            # --- 1. TREND STRATEGIEËN (T, S, HT, HS) ---
            configs = [("T", 50, 200, False), ("S", 20, 50, False), ("HT", 9, 21, True), ("HS", 5, 13, True)]
            for skey, sp, tp, ihyp in configs:
                p, f, s, e, vma, rsi, atr, adx, _, _, _ = bereken_indicatoren(t_data, sp, tp, ihyp)
                pb, fb, sb, ab, dxb = p.iloc[150:], f.iloc[150:], s.iloc[150:], atr.iloc[150:], adx.iloc[150:]
                pr, pos, ins, hi = 0.0, False, 0.0, 0.0
                for i in range(1, len(pb)):
                    cp = pb.iloc[i]
                    if not pos:
                        if fb.iloc[i] > sb.iloc[i] and fb.iloc[i-1] <= sb.iloc[i-1] and dxb.iloc[i] > 15:
                            ins, hi, pos = cp, cp, True
                            pr -= kosten
                    else:
                        hi = max(hi, cp)
                        if cp < (hi - 2.5 * ab.iloc[i]) or fb.iloc[i] < sb.iloc[i]:
                            pr += (inzet * (cp / ins) - inzet) - kosten
                            pos = False
                res[skey] += pr

            # --- 2. MRS STRATEGIE (Geoptimaliseerd) ---
            p, _, _, _, _, _, _, _, ibs, l_b3, ma5 = bereken_indicatoren(t_data, 20, 50, False)
            pb, ibsb, lbb, m5b = p.iloc[150:], ibs.iloc[150:], l_b3.iloc[150:], ma5.iloc[150:]
            pr_mrs, pos_mrs, ins_mrs = 0.0, False, 0.0
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos_mrs:
                    # VERBETERING: EMA filter weg (koop de echte dip) & IBS naar 0.30
                    if cp < lbb.iloc[i] and ibsb.iloc[i] < 0.30:
                        ins_mrs, pos_mrs = cp, True
                        pr_mrs -= kosten
                else:
                    # VERBETERING: Target naar 12% om kosten te verslaan
                    if cp > m5b.iloc[i] or cp > (ins_mrs * 1.12):
                        pr_mrs += (inzet * (cp / ins_mrs) - inzet) - kosten
                        pos_mrs = False
            res["MRS"] += pr_mrs

        except: continue

    # Resultaat formatteren en verzenden
    def f(n): return f"€{100000 + n:,.0f}"
    nu = datetime.now().strftime("%d/%m %H:%M")
    bericht = (f"📊 *RAPPORT: {sector_naam}*\n"
               f"Bestand: `{bestandsnaam}` | {nu}\n"
               f"----------------------------------\n"
               f"🐢 Traag: {f(res['T'])}\n⚡ Snel: {f(res['S'])}\n"
               f"🚀 Hyper: {f(res['HT'])}\n🔥 Scalp: {f(res['HS'])}\n"
               f"💎 *MRS:* {f(res['MRS'])}")
    stuur_telegram(bericht)

# ---------------------------------------------------------------------------
# MAIN SCRIPT (tickers_01.txt tot tickers_09.txt)
# ---------------------------------------------------------------------------
def main():
    sectoren = {
        "01": "Hoogland", "02": "Macrotrends", "03": "Beursbrink", 
        "04": "Benelux", "05": "Parijs", "06": "Power & AI", 
        "07": "Metalen", "08": "Defensie", "09": "Varia"
    }
    
    for nr, naam in sectoren.items():
        bestandsnaam = f"tickers_{nr}.txt"
        voer_backtest(bestandsnaam, naam)
        time.sleep(2) # Voorkom rate-limiting van Yahoo

if __name__ == "__main__":
    main()
