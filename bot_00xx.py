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
# INDICATOREN (ONGÉWIIJZIGD VOOR STRAT 1-4, VERBETERD VOOR 5)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(df: pd.DataFrame, s: int, t: int, use_trend_filter: bool, is_hyper: bool) -> tuple:
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    
    # 1. STANDAARD RSI (Wilder Smoothing)
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # 2. STRAT 5 SPECIALS (IBS & Extreme Bollinger)
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    lower_b3 = ma20 - (2.2 * std20)  # 2.2 Standard Deviations voor meer signalen (was oorspr 3)
    ibs = (p - l) / (h - l + 1e-10)  # Internal Bar Strength
    ma5 = p.rolling(5).mean()

    # 3. HYPER (Streak RSI)
    if is_hyper:
        change = np.sign(delta).fillna(0)
        streak = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta = streak.diff().fillna(0)
        s_gain = s_delta.where(s_delta > 0, 0.0).rolling(2).mean()
        s_loss = (-s_delta.where(s_delta < 0, 0.0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + s_gain / (s_loss + 1e-10))).fillna(50)
        rsi3_gain = delta.where(delta > 0, 0.0).ewm(alpha=1/3, adjust=False).mean()
        rsi3_loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/3, adjust=False).mean()
        rsi3 = 100 - (100 / (1 + rsi3_gain / (rsi3_loss + 1e-10)))
        p_rank = delta.rolling(100).apply(lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100 if len(x) > 0 else 50, raw=True)
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3

    # 4. CORRECTE ADX (Wilder)
    tr = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v, ibs, lower_b3, ma5
    
# ---------------------------------------------------------------------------
# SECTOR VERWERKING
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
    sig = {"T": [], "S": [], "HT": [], "HS": [], "MRA": []}
    STRATS = [("T", 50, 200, True, False), ("S", 20, 50, True, False), ("HT", 9, 21, True, True), ("HS", 9, 21, False, True)]

    for ticker in tickers:
        try:
            t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all') if len(tickers) > 1 else raw_df.dropna(how='all')
            if len(t_data) < 250: continue

            # --- DATA BEREKENEN ---
            p, f, sl, e200, v_ma, rsi, atr, adx, vol, ibs, l_b3, ma5 = bereken_indicatoren_vectorized(t_data, 50, 200, True, False)
            kosten = 15.0 + (inzet * 0.0035)

            # --- STRAT 1-4 BACKTEST (IDENTIEK) ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                pi, fi, sli, ei, vmai, rsii, atri, adxi, voli, _, _, _ = bereken_indicatoren_vectorized(t_data, s_p, t_p, utr, ihyp)
                pb = pi.iloc[200:]; fb = fi.iloc[200:]; sb = sli.iloc[200:]; eb = ei.iloc[200:]; vb = voli.iloc[200:]; vmb = vmai.iloc[200:]; ab = atri.iloc[200:]; dxb = adxi.iloc[200:]
                pr, pos, ins, hi = 0.0, False, 0.0, 0.0
                for i in range(1, len(pb)):
                    cp = pb.iloc[i]
                    if not pos:
                        if fb.iloc[i] > sb.iloc[i] and fb.iloc[i-1] <= sb.iloc[i-1] and dxb.iloc[i] > 15 and vb.iloc[i] > (vmb.iloc[i]*0.6) and ((not utr) or cp > eb.iloc[i]):
                            ins, hi, pos = cp, cp, True
                            pr -= kosten
                    else:
                        hi = max(hi, cp)
                        if cp < (hi - 2*ab.iloc[i]) or fb.iloc[i] < sb.iloc[i]:
                            pr += (inzet*(cp/ins)-inzet)-kosten
                            pos = False
                if pos: pr += (inzet*(pb.iloc[-1]/ins)-inzet)-kosten
                res[skey] += pr
                
                # Signaal Strat 1-4 (laatste dag)
                if skey == "T":
                    cp = pi.iloc[-1]; y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    if fi.iloc[-1] > sli.iloc[-1] and fi.iloc[-2] <= sli.iloc[-2] and adxi.iloc[-1] > 15 and voli.iloc[-1] > (vmb.iloc[-1]*0.6) and ((not utr) or cp > ei.iloc[-1]):
                        sig[skey].append(f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | {y_l}")
                    elif fi.iloc[-1] < sli.iloc[-1] and fi.iloc[-2] >= sli.iloc[-2]:
                        sig[skey].append(f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f}")

            # --- STRAT 5: IBS MEAN REVERSION BACKTEST (NEW & IMPROVED) ---
            pb, ibsb, lbb, eb, m5b = p.iloc[200:], ibs.iloc[200:], l_b3.iloc[200:], e200.iloc[200:], ma5.iloc[200:]
            pr5, pos5, ins5 = 0.0, False, 0.0
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos5:
                    if cp < lbb.iloc[i] and ibsb.iloc[i] < 0.2 and cp > eb.iloc[i]:
                        ins5, pos5 = cp, True
                        pr5 -= kosten
                else:
                    # Target: herstel naar MA5 of 6% winst
                    if cp > m5b.iloc[i] or cp > (ins5 * 1.06):
                        pr5 += (inzet*(cp/ins5)-inzet)-kosten
                        pos5 = False
            if pos5: pr5 += (inzet*(pb.iloc[-1]/ins5)-inzet)-kosten
            res["MRA"] += pr5

            # Signaal Strat 5
            if p.iloc[-1] < l_b3.iloc[-1] and ibs.iloc[-1] < 0.20 and p.iloc[-1] > e200.iloc[-1]:
                sig["MRA"].append(f"• `{ticker}`: 🛡️ *Munger Dip* | €{p.iloc[-1]:.2f}")

        except: continue

    def fmt(n): return f"€{100000 + n:,.0f}"
    rapport = [
        f"📊 *{label} {naam_sector} RAPPORTxx*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* {fmt(res['T'])}",
        f"⚡ *Snel (20/50):* {fmt(res['S'])}",
        f"🚀 *Hyper Trend:* {fmt(res['HT'])}",
        f"🔥 *Hyper Scalp:* {fmt(res['HS'])}",
        f"💎 *Power Mean Rev:* {fmt(res['MRA'])}",
        "", "🛡️ *SIGNALEN TRAAG:*", "\n".join(sig["T"]) if sig["T"] else "Geen actie",
        "", "💎 *SIGNALEN POWER MEAN REV:*", "\n".join(sig["MRA"]) if sig["MRA"] else "Geen actie",
        "", "💡 _Traag gebruikt 50/200 MA. MRA gebruikt IBS & Extreme Bands, Bollgier band naar 2.2, IBS naar 0.2._",
        "", "💡 _ATR %: <2% laag, >5% hoog. RSI: >70 overbought, <30 oversold. CRSI: >90 overbought, <10 oversold_"
    ]
    stuur_telegram("\n".join(rapport))

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    sectoren = {"01":"Hoogland", "02":"Macrotrends", "03":"Beursbrink", "04":"Benelux", "05":"Parijs", "06":"Power & AI", "07":"Metalen", "08":"Defensie", "09":"Varia"}
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)

if __name__ == "__main__": main()
