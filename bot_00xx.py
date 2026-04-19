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
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()
        return True
    except: return False

# ---------------------------------------------------------------------------
# INDICATOREN
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(df: pd.DataFrame, s: int, t: int, use_trend_filter: bool, is_hyper: bool) -> tuple:
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema100 = p.ewm(span=100, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    lower_b3 = ma20 - (2.2 * std20)  
    ibs = (p - l) / (h - l + 1e-10)  
    ma5 = p.rolling(5).mean()

    if is_hyper:
        p_rank = delta.rolling(100).apply(lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100 if len(x) > 0 else 50, raw=True)
        rsi_val = (rsi_val + p_rank) / 2

    tr = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema100, vol_ma, rsi_val, atr, adx, v, ibs, lower_b3, ma5

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> None:
    if not os.path.exists(bestandsnaam): return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")

    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except: return

    inzet = 2500.0
    res = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRA": 0.0}
    sig = {"T": [], "MRA": []}
    kosten = 15.0 + (inzet * 0.0035)

    for ticker in tickers:
        try:
            t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all') if len(tickers) > 1 else raw_df.dropna(how='all')
            if len(t_data) < 250: continue

            # --- TREND STRATS (Laatste Visie: StopLoss 3xATR) ---
            configs = [("T", 50, 200, True, False), ("S", 20, 50, True, False), ("HT", 9, 21, True, True), ("HS", 9, 21, False, True)]
            for skey, sp, tp, utr, ihyp in configs:
                p, f, sl, eb, vma, rsi, ab, dxb, vol, ibs_v, lbb, m5 = bereken_indicatoren_vectorized(t_data, sp, tp, utr, ihyp)
                pb, fb, sb, abb, dxb_b = p.iloc[200:], f.iloc[200:], sl.iloc[200:], ab.iloc[200:], dxb.iloc[200:]
                pr, pos, ins, hi = 0.0, False, 0.0, 0.0
                for i in range(1, len(pb)):
                    cp = pb.iloc[i]
                    if not pos:
                        if fb.iloc[i] > sb.iloc[i] and fb.iloc[i-1] <= sb.iloc[i-1] and dxb_b.iloc[i] > 15:
                            ins, hi, pos = cp, cp, True
                            pr -= kosten
                    else:
                        hi = max(hi, cp)
                        # LAATSTE VISIE: 3xATR
                        if cp < (hi - 3 * abb.iloc[i]) or fb.iloc[i] < sb.iloc[i]:
                            pr += (inzet * (cp / ins) - inzet) - kosten
                            pos = False
                res[skey] += pr
                if skey == "T":
                    cp = p.iloc[-1]; y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    if f.iloc[-1] > sl.iloc[-1] and f.iloc[-2] <= sl.iloc[-2] and dxb.iloc[-1] > 15:
                        sig["T"].append(f"• {ticker}: 🟢 *KOOP* | €{cp:.2f} | {y_l}")

            # --- MRA (Laatste Visie: Target 12%, IBS 0.30, Geen EMA) ---
            p, _, _, _, _, _, _, _, _, ibs, l_b3, ma5 = bereken_indicatoren_vectorized(t_data, 20, 50, False, False)
            pb, ibsb, lbb, m5b = p.iloc[200:], ibs.iloc[200:], l_b3.iloc[200:], ma5.iloc[200:]
            pr_mra, pos_mra, ins_mra = 0.0, False, 0.0
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos_mra:
                    if cp < lbb.iloc[i] and ibsb.iloc[i] < 0.30:
                        ins_mra, pos_mra = cp, True
                        pr_mra -= kosten
                else:
                    if cp > m5b.iloc[i] or cp > (ins_mra * 1.12):
                        pr_mra += (inzet * (cp / ins_mra) - inzet) - kosten
                        pos_mra = False
            res["MRA"] += pr_mra
            if p.iloc[-1] < l_b3.iloc[-1] and ibs.iloc[-1] < 0.30:
                sig["MRA"].append(f"• {ticker}: 🛡️ *DIP* | €{p.iloc[-1]:.2f}")

        except: continue

    # ---------------------------------------------------------------------------
    # RAPPORT LAYOUT (ZOALS BEZORGD OM 06:00 UUR)
    # ---------------------------------------------------------------------------
    def fmt(n): return f"€{100000 + n:,.0f}"
    
    rapport = [
        f"📊 *{label} {naam_sector} RAPPORT xx",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):* {fmt(res['T'])}",
        f"⚡ *Snel (20/50):* {fmt(res['S'])}",
        f"🚀 *Hyper Trend:* {fmt(res['HT'])}",
        f"🔥 *Hyper Scalp:* {fmt(res['HS'])}",
        f"💎 *{naam_sector} Mean Rev:* {fmt(res['MRA'])}",
        "",
        "🛡️ *SIGNALEN TRAAG:*",
        "\n".join(sig["T"]) if sig["T"] else "Geen actie",
        "",
        f"💎 *SIGNALEN {naam_sector.upper()} MEAN REV:*",
        "\n".join(sig["MRA"]) if sig["MRA"] else "Geen actie",
        "",
        "💡 *Instellingen:* StopLoss 3xATR, MRA Target 12%, IBS 0.30."
    ]
    stuur_telegram("\n".join(rapport))

def main():
    sectoren = {"01":"Hoogland", "02":"Macrotrends", "03":"Beursbrink", "04":"Benelux", "05":"Parijs", "06":"Power & AI", "07":"Metalen", "08":"Defensie", "09":"Varia"}
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)

if __name__ == "__main__": main()
