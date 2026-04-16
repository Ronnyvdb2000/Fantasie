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
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN   = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID:
        logger.error("Telegram TOKEN of CHAT_ID niet ingesteld.")
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": bericht,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        r.raise_for_status()
        time.sleep(1)
        return True
    except Exception as e:
        logger.error(f"Telegram fout: {e}")
    return False


# ---------------------------------------------------------------------------
# INDICATOREN (Bestaande logica + Strat 5 toevoeging)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(
    df: pd.DataFrame,
    s: int,
    t: int,
    use_trend_filter: bool,
    is_hyper: bool,
) -> tuple:
    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    # RSI(14)
    delta = p.diff()
    gain     = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi_val  = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # --- TOEVOEGING VOOR STRAT 5 ---
    rsi2_gain = delta.where(delta > 0, 0.0).rolling(2).mean()
    rsi2_loss = (-delta.where(delta < 0, 0.0)).rolling(2).mean()
    rsi2      = 100 - (100 / (1 + rsi2_gain / (rsi2_loss + 1e-10)))
    ma5       = p.rolling(window=5).mean()
    ma2       = p.rolling(window=2).mean()
    # ------------------------------

    if is_hyper:
        rsi3_gain = delta.where(delta > 0, 0.0).rolling(3).mean()
        rsi3_loss = (-delta.where(delta < 0, 0.0)).rolling(3).mean()
        rsi3      = 100 - (100 / (1 + rsi3_gain / (rsi3_loss + 1e-10)))

        change = np.sign(delta).fillna(0)
        groups = (change != change.shift()).cumsum()
        streak = change.groupby(groups).cumsum()
        s_delta  = streak.diff().fillna(0)
        s_gain   = s_delta.where(s_delta > 0, 0.0).rolling(2).mean()
        s_loss   = (-s_delta.where(s_delta < 0, 0.0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + s_gain / (s_loss + 1e-10)))

        p_rank = delta.rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100, raw=True
        )
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3

    tr  = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    up   = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    atr14      = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di    = 100 * up.ewm(alpha=1/14, adjust=False).mean()  / (atr14 + 1e-10)
    minus_di   = 100 * down.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx        = dx.ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v, rsi2, ma5, ma2


# ---------------------------------------------------------------------------
# SECTOR VERWERKING
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> None:
    if not os.path.exists(bestandsnaam):
        logger.warning(f"Bestand niet gevonden: {bestandsnaam}")
        return

    nu = datetime.now().strftime("%d/%m/%Y %H:%M")

    with open(bestandsnaam, 'r') as f:
        content = f.read().replace('\n', ',').replace('$', '')
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))

    if not tickers: return

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download mislukt: {e}")
        return

    inzet  = 2500.0
    res    = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRA": 0.0}
    sig    = {"T": [],  "S": [],  "HT": [],   "HS": [], "MRA": []}

    STRATEGIEEN = [
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

            if len(t_data) < 260: continue

            # --- BEREKENING STRAT 1-4 ---
            for strat_key, s_per, t_per, use_tr, is_hyp in STRATEGIEEN:
                p, f, s_line, e200, v_ma, rsi, atr, adx, vol, rsi2, ma5, ma2 = bereken_indicatoren_vectorized(
                    t_data, s_per, t_per, use_tr, is_hyp
                )

                # Backtest logic Strat 1-4
                p_bt = p.iloc[-252:]; f_bt = f.iloc[-252:]; s_bt = s_line.iloc[-252:]
                e_bt = e200.iloc[-252:]; v_bt = vol.iloc[-252:]; v_ma_bt = v_ma.iloc[-252:]
                atr_bt = atr.iloc[-252:]; adx_bt = adx.iloc[-252:]
                
                profit = 0.0; pos = False; instap = 0.0; high_p = 0.0; kosten = 15.0 + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp_i = p_bt.iloc[i]
                    if not pos:
                        if f_bt.iloc[i] > s_bt.iloc[i] and f_bt.iloc[i-1] <= s_bt.iloc[i-1] and adx_bt.iloc[i] > 15 and v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6) and ((not use_tr) or cp_i > e_bt.iloc[i]):
                            instap = cp_i; high_p = cp_i; pos = True; profit -= kosten
                    else:
                        high_p = max(high_p, cp_i)
                        if cp_i < (high_p - (2 * atr_bt.iloc[i])) or f_bt.iloc[i] < s_bt.iloc[i]:
                            profit += (inzet * (cp_i / instap) - inzet) - kosten
                            pos = False
                if pos: profit += (inzet * (p_bt.iloc[-1] / instap) - inzet) - kosten
                res[strat_key] += profit

                # Signalen Strat 1-4
                if strat_key == "T": # Alleen signalen toevoegen bij de laatste iteratie van de ticker per strat
                    cp = p.iloc[-1]; catr = atr.iloc[-1]; crsi = rsi.iloc[-1]
                    k_up = f.iloc[-1] > s_line.iloc[-1] and f.iloc[-2] <= s_line.iloc[-2]
                    k_down = f.iloc[-1] < s_line.iloc[-1] and f.iloc[-2] >= s_line.iloc[-2]
                    y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    l_rsi = "💎 CRSI" if is_hyp else "📊 RSI"
                    if k_up and adx.iloc[-1] > 15 and vol.iloc[-1] > (v_ma.iloc[-1]*0.6) and ((not use_tr) or cp > e200.iloc[-1]):
                        sig[strat_key].append(f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")
                    elif k_down:
                        sig[strat_key].append(f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}")

            # --- BEREKENING STRAT 5 (POWER MEAN REVERSION) ---
            # We gebruiken hier de data van de laatste indicator berekening (ma5, rsi2, ma2)
            pb = p.iloc[-252:]; r2b = rsi2.iloc[-252:]; m5b = ma5.iloc[-252:]; m2b = ma2.iloc[-252:]; e_st = e200.diff(5).iloc[-252:]
            p5 = 0.0; pos5 = False; ins5 = 0.0
            for i in range(1, len(pb)):
                cp = pb.iloc[i]
                if not pos5:
                    if r2b.iloc[i] < 5 and cp < (m5b.iloc[i]*0.95) and e_st.iloc[i] > 0:
                        ins5 = cp; pos5 = True; p5 -= kosten
                else:
                    if cp > m2b.iloc[i] or cp > (ins5 * 1.08):
                        p5 += (inzet*(cp/ins5)-inzet)-kosten; pos5 = False
            if pos5: p5 += (inzet*(pb.iloc[-1]/ins5)-inzet)-kosten
            res["MRA"] += p5

            # Signaal Strat 5
            if rsi2.iloc[-1] < 10 and p.iloc[-1] < (ma5.iloc[-1]*0.95) and e200.diff(5).iloc[-1] > 0:
                sig["MRA"].append(f"• `{ticker}`: 💎 *POWER BOUNCE* | €{p.iloc[-1]:.2f} | RSI2: {rsi2.iloc[-1]:.1f}")

        except Exception as e: continue

    # --- RAPPORT (Exacte weergave behouden) ---
    def get_s(lst: list) -> str: return "\n".join(lst) if lst else "Geen actie"

    rapport_lijst = [
        f"📊 *{label} {naam_sector} RAPPORT*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        f"💎 *Power Mean Rev:* €{100000 + res['MRA']:,.0f}",
        "", "🛡️ *SIGNALEN TRAAG (RSI):*",         get_s(sig["T"]),
        "", "🎯 *SIGNALEN SNEL (RSI):*",           get_s(sig["S"]),
        "", "📈 *SIGNALEN HYPER TREND (CRSI):*",   get_s(sig["HT"]),
        "", "⚡ *SIGNALEN HYPER SCALP (CRSI):*",   get_s(sig["HS"]),
        "", "💎 *SIGNALEN POWER MEAN REV:*",       get_s(sig["MRA"]),
        "", "💡 _ATR %: <2% laag, >5% hoog. RSI: >70 overbought, <30 oversold. CRSI: >90 overbought, <10 oversold_",
    ]
    stuur_telegram("\n".join(rapport_lijst))


# ---------------------------------------------------------------------------
# MAIN (Alle tickers van 01 tot 09)
# ---------------------------------------------------------------------------
def main() -> None:
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
        try:
            voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        except Exception as e:
            logger.error(f"Sector {naam} mislukt: {e}")
        time.sleep(2)

if __name__ == "__main__":
    main()
