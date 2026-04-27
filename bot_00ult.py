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

# ---------------------------------------------------------------------------
# MRA PARAMETERS
# ---------------------------------------------------------------------------
MRA_BB_STD       = 2.2    # Bollinger Band breedte
MRA_IBS_MAX      = 0.30   # IBS (sluit in onderste 30% van dagrange)

# MRA SNEL — exact origineel (gaf €945.000)
MRA_SNEL_WINST   = 1.12   # +12% winstlimiet
MRA_SNEL_MA      = 5      # verkoop boven MA5

# MRA TRAAG — ruimere uitstap om grotere bewegingen mee te nemen
MRA_TRAAG_WINST  = 1.40   # +40% winstlimiet
MRA_TRAAG_ATR    = 3      # trailing stop 3x ATR

def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram fout: {e}")
        return False

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

    # RSI met correcte Wilder smoothing (EWM)
    delta = p.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Bollinger Band & IBS voor MRA
    ma20  = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    lower_bb = ma20 - (MRA_BB_STD * std20)
    ibs   = (p - l) / (h - l + 1e-10)
    ma5   = p.rolling(5).mean()
    ma20_line = ma20  # voor MRA Traag uitstap

    # CRSI voor Hyper strategieën
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

    # ATR & ADX met correcte Wilder smoothing
    tr = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up   = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_di  = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    adx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema100, vol_ma, rsi_val, atr, adx, v, ibs, lower_bb, ma5, ma20_line

# ---------------------------------------------------------------------------
# MRA INSTAP FILTER (gedeeld door Snel en Traag)
# ---------------------------------------------------------------------------
def mra_instap_ok(cp, lbb_i, ibs_i, vol_i, vma_i, rsi_i, ema100_i, ema100_prev) -> bool:
    """Originele MRA instap: enkel BB + IBS, geen extra filters."""
    return cp < lbb_i and ibs_i < MRA_IBS_MAX

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> None:
    if not os.path.exists(bestandsnaam):
        logger.warning(f"Bestand {bestandsnaam} niet gevonden.")
        return
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")

    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    logger.info(f"Start analyse voor {naam_sector} met {len(tickers)} tickers.")

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download fout: {e}")
        return

    inzet = 2500.0
    res       = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRAS": 0.0, "MRAT": 0.0}
    num_trades = {"T": 0,   "S": 0,   "HT": 0,   "HS": 0,   "MRAS": 0,   "MRAT": 0}
    sig       = {"T": [],   "S": [],   "HT": [],   "HS": [],   "MRAS": [],   "MRAT": []}

    STRATS = [
        ("T",  50,  200, True,  False),
        ("S",  20,   50, True,  False),
        ("HT",  9,   21, True,  True),
        ("HS",  9,   21, False, True),
    ]

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all')
            else:
                t_data = raw_df.dropna(how='all')

            if len(t_data) < 250: continue

            # Bereken basis indicatoren (50/200, niet-hyper) voor MRA
            p, f, sl, e100, v_ma, rsi, atr, adx, vol, ibs, l_bb, ma5, ma20 = bereken_indicatoren_vectorized(t_data, 50, 200, True, False)
            kosten = 15.0 + (inzet * 0.0035)

            # --- STRATEGIEËN T / S / HT / HS ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                pi, fi, sli, ei, vmai, rsii, atri, dxi, voli, _, _, _, _ = bereken_indicatoren_vectorized(t_data, s_p, t_p, utr, ihyp)

                pb  = pi.iloc[200:];  fb  = fi.iloc[200:];  sb  = sli.iloc[200:]
                eb  = ei.iloc[200:];  vb  = voli.iloc[200:]; vmb = vmai.iloc[200:]
                ab  = atri.iloc[200:]; dxb = dxi.iloc[200:]

                pr, pos, ins, hi = 0.0, False, 0.0, 0.0
                for i in range(1, len(pb)):
                    cp = pb.iloc[i]
                    if not pos:
                        if (fb.iloc[i] > sb.iloc[i] and fb.iloc[i-1] <= sb.iloc[i-1]
                                and dxb.iloc[i] > 15
                                and vb.iloc[i] > (vmb.iloc[i] * 0.6)
                                and ((not utr) or cp > eb.iloc[i])):
                            ins, hi, pos = cp, cp, True
                            pr -= kosten
                            num_trades[skey] += 1
                    else:
                        hi = max(hi, cp)
                        if cp < (hi - 2 * ab.iloc[i]) or fb.iloc[i] < sb.iloc[i]:
                            pr += (inzet * (cp / ins) - inzet) - kosten
                            pos = False
                if pos:
                    pr += (inzet * (pb.iloc[-1] / ins) - inzet) - kosten
                res[skey] += pr

                # Signalen
                cp    = pi.iloc[-1]
                catr  = atri.iloc[-1]
                crsi  = rsii.iloc[-1]
                y_l   = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                l_rsi = "💎 CRSI" if ihyp else "📊 RSI"

                if (fi.iloc[-1] > sli.iloc[-1] and fi.iloc[-2] <= sli.iloc[-2]
                        and dxi.iloc[-1] > 15
                        and voli.iloc[-1] > (vmai.iloc[-1] * 0.6)
                        and ((not utr) or cp > ei.iloc[-1])):
                    sig[skey].append(f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")
                elif fi.iloc[-1] < sli.iloc[-1] and fi.iloc[-2] >= sli.iloc[-2]:
                    sig[skey].append(f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")

            # ---------------------------------------------------------------
            # MRA SNEL — originele logica: BB + IBS instap, uitstap MA5 of +12%
            # Backtest venster identiek aan origineel: laatste 252 dagen
            # ---------------------------------------------------------------
            pb    = p.iloc[-252:];    ibsb  = ibs.iloc[-252:]
            lbb_b = l_bb.iloc[-252:]; m5b   = ma5.iloc[-252:]
            vb    = vol.iloc[-252:];  vmb   = v_ma.iloc[-252:]
            rsib  = rsi.iloc[-252:];  e100b = e100.iloc[-252:]
            atrb  = atr.iloc[-252:]

            pr_ms, pos_ms, ins_ms = 0.0, False, 0.0
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos_ms:
                    e100_prev = e100b.iloc[max(0, i - MRA_TREND_BARS)]
                    if mra_instap_ok(cp, lbb_b.iloc[i], ibsb.iloc[i], vb.iloc[i], vmb.iloc[i], rsib.iloc[i], e100b.iloc[i], e100_prev):
                        ins_ms, pos_ms = cp, True
                        pr_ms -= kosten
                        num_trades["MRAS"] += 1
                else:
                    if cp > m5b.iloc[i] or cp > (ins_ms * MRA_SNEL_WINST):
                        pr_ms += (inzet * (cp / ins_ms) - inzet) - kosten
                        pos_ms = False
            if pos_ms:
                pr_ms += (inzet * (pb.iloc[-1] / ins_ms) - inzet) - kosten
            res["MRAS"] += pr_ms

            # ---------------------------------------------------------------
            # MRA TRAAG — zelfde instap, uitstap via ATR trailing stop of +40%
            # Backtest venster identiek aan origineel: laatste 252 dagen
            # ---------------------------------------------------------------
            pr_mt, pos_mt, ins_mt, hi_mt, sl_mt = 0.0, False, 0.0, 0.0, 0.0
            for i in range(1, len(pb)):  # hergebruik pb/atrb van MRA Snel hierboven
                cp = pb.iloc[i]
                if not pos_mt:
                    e100_prev = e100b.iloc[max(0, i - MRA_TREND_BARS)]
                    if mra_instap_ok(cp, lbb_b.iloc[i], ibsb.iloc[i], vb.iloc[i], vmb.iloc[i], rsib.iloc[i], e100b.iloc[i], e100_prev):
                        ins_mt  = cp
                        hi_mt   = cp
                        sl_mt   = cp - (3 * atrb.iloc[i])  # ruimere stop dan trend strats (2x ATR)
                        pos_mt  = True
                        pr_mt  -= kosten
                        num_trades["MRAT"] += 1
                else:
                    hi_mt = max(hi_mt, cp)
                    sl_mt = max(sl_mt, hi_mt - (3 * atrb.iloc[i]))  # trailing stop 3x ATR
                    if cp < sl_mt or cp > (ins_mt * MRA_TRAAG_WINST):
                        pr_mt += (inzet * (cp / ins_mt) - inzet) - kosten
                        pos_mt = False
            if pos_mt:
                pr_mt += (inzet * (pb.iloc[-1] / ins_mt) - inzet) - kosten
            res["MRAT"] += pr_mt

            # ---------------------------------------------------------------
            # ACTUELE MRA SIGNALEN (gedeelde instapcheck)
            # ---------------------------------------------------------------
            cp_now     = p.iloc[-1]
            e100_prev  = e100.iloc[-1 - MRA_TREND_BARS] if len(e100) > MRA_TREND_BARS else e100.iloc[0]
            if mra_instap_ok(cp_now, l_bb.iloc[-1], ibs.iloc[-1], vol.iloc[-1], v_ma.iloc[-1], rsi.iloc[-1], e100.iloc[-1], e100_prev):
                crsi_now = rsi.iloc[-1]
                y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                sig["MRAS"].append(f"• `{ticker}`: 🛡️ *Munger Snel* | €{cp_now:.2f} | 📊 RSI: {crsi_now:.1f} | {y_l}")
                sig["MRAT"].append(f"• `{ticker}`: 🐢 *Munger Traag* | €{cp_now:.2f} | 📊 RSI: {crsi_now:.1f} | {y_l}")

        except Exception as e:
            logger.error(f"Fout bij ticker {ticker}: {e}")
            continue

    # ---------------------------------------------------------------------------
    # RAPPORT
    # ---------------------------------------------------------------------------
    def fmt(n): return f"€{100000 + n:,.0f}"
    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"

    rapport = [
        f"📊 *{label} {naam_sector} RAPPORT*",
        f"_{nu}_",
        "----------------------------------",
        f"🐢 *Traag (50/200):*  {fmt(res['T'])} ({num_trades['T']} trades)",
        f"⚡ *Snel (20/50):*    {fmt(res['S'])} ({num_trades['S']} trades)",
        f"🚀 *Hyper Trend:*     {fmt(res['HT'])} ({num_trades['HT']} trades)",
        f"🔥 *Hyper Scalp:*    {fmt(res['HS'])} ({num_trades['HS']} trades)",
        f"🛡️ *MRA Snel:*        {fmt(res['MRAS'])} ({num_trades['MRAS']} trades)",
        f"🐢 *MRA Traag:*       {fmt(res['MRAT'])} ({num_trades['MRAT']} trades)",
        "",
        "🛡️ *SIGNALEN TRAAG (RSI):*",
        get_s(sig["T"]),
        "",
        "🎯 *SIGNALEN SNEL (RSI):*",
        get_s(sig["S"]),
        "",
        "📈 *SIGNALEN HYPER TREND (CRSI):*",
        get_s(sig["HT"]),
        "",
        "⚡ *SIGNALEN HYPER SCALP (CRSI):*",
        get_s(sig["HS"]),
        "",
        "🛡️ *SIGNALEN MRA SNEL (Munger Dip):*",
        get_s(sig["MRAS"]),
        "",
        "🐢 *SIGNALEN MRA TRAAG (Munger Dip):*",
        get_s(sig["MRAT"]),
        "",
        "⚙️ *PARAMETERS TREND STRATS:*",
        f"_ADX>15 | Vol>0.6x MA20 | EMA100 trendfilter | ATR trailing stop (2x)_",
        f"_Traag: MA50/200 | Snel: MA20/50 | Hyper: EMA9/21 | Scalp: EMA9/21 geen trendfilter_",
        "",
        "⚙️ *PARAMETERS MRA:*",
        f"_Instap: BB {MRA_BB_STD}σ | IBS<{MRA_IBS_MAX}_",
        f"_MRA Snel: uitstap MA{MRA_SNEL_MA} of +{int((MRA_SNEL_WINST-1)*100)}% (origineel)_",
        f"_MRA Traag: trailing stop {MRA_TRAAG_ATR}x ATR of +{int((MRA_TRAAG_WINST-1)*100)}%_",
        f"_Inzet: €{inzet:.0f} | Kosten: €{kosten:.2f}/trade_"
    ]
    stuur_telegram("\n".join(rapport))

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    sectoren = {
        "01": "Hoogland",
        "02": "Macrotrends",
        "03": "Beursbrink",
        "04": "Benelux",
        "05": "Parijs",
        "06": "Power & AI",
        "07": "Metalen",
        "08": "Defensie",
        "09": "Varia",
    }
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)

if __name__ == "__main__":
    main()
