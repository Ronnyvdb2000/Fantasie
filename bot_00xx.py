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
# LOGGING — vervangt stille except: pass
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
            timeout=20,
        )
        r.raise_for_status()
        time.sleep(1)
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Telegram HTTP fout: {e} | {r.text}")
    except requests.exceptions.ConnectionError:
        logger.error("Telegram: geen internetverbinding.")
    except requests.exceptions.Timeout:
        logger.error("Telegram: timeout.")
    except Exception as e:
        logger.error(f"Telegram: onverwachte fout – {e}")
    return False


# ---------------------------------------------------------------------------
# INDICATOREN (vectorized)
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(
    df: pd.DataFrame,
    s: int,
    t: int,
    use_trend_filter: bool,
    is_hyper: bool,
) -> tuple:
    """Berekent alle indicatoren vectorized op de volledige dataset."""

    p = df['Close'].ffill()
    h = df['High'].ffill()
    l = df['Low'].ffill()
    v = df['Volume'].ffill()

    # Moving averages
    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    # RSI(14)
    delta = p.diff()
    gain     = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi_val  = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # CRSI (alleen voor hyper-strategieën)
    if is_hyper:
        # 1. RSI(3)
        rsi3_gain = delta.where(delta > 0, 0.0).rolling(3).mean()
        rsi3_loss = (-delta.where(delta < 0, 0.0)).rolling(3).mean()
        rsi3      = 100 - (100 / (1 + rsi3_gain / (rsi3_loss + 1e-10)))

        # 2. Streak RSI(2) — vectorized, NaN-safe
        change = np.sign(delta).fillna(0)
        groups = (change != change.shift()).cumsum()
        streak = change.groupby(groups).cumsum()
        s_delta  = streak.diff().fillna(0)
        s_gain   = s_delta.where(s_delta > 0, 0.0).rolling(2).mean()
        s_loss   = (-s_delta.where(s_delta < 0, 0.0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + s_gain / (s_loss + 1e-10)))

        # 3. Percent Rank(100)
        p_rank = delta.rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100, raw=True
        )

        rsi_val = (rsi3 + streak_rsi + p_rank) / 3

    # ATR
    tr  = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ADX met Wilder smoothing (correcte berekening)
    up   = h.diff().clip(lower=0)
    down = (-l.diff()).clip(lower=0)
    plus_dm  = up.where(up > down, 0.0)
    minus_dm = down.where(down > up, 0.0)

    atr14      = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di    = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean()  / (atr14 + 1e-10)
    minus_di   = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / (atr14 + 1e-10)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx        = dx.ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema200, vol_ma, rsi_val, atr, adx, v


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

    if not tickers:
        logger.warning(f"{bestandsnaam}: geen tickers gevonden.")
        return

    logger.info(f"Start sector '{naam_sector}' | {len(tickers)} tickers")

    # Bulk download
    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download mislukt voor sector {naam_sector}: {e}")
        return

    if raw_df.empty:
        logger.warning(f"Geen data ontvangen voor sector {naam_sector}.")
        return

    inzet  = 2500.0
    res    = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0}
    sig    = {"T": [],  "S": [],  "HT": [],   "HS": []}

    STRATEGIEEN = [
        ("T",  50,  200, True,  False),
        ("S",  20,   50, True,  False),
        ("HT",  9,   21, True,  True),
        ("HS",  9,   21, False, True),
    ]

    for strat_key, s_per, t_per, use_tr, is_hyp in STRATEGIEEN:
        for ticker in tickers:
            try:
                # Ticker data ophalen uit bulk download
                if len(tickers) > 1:
                    t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how='all')
                else:
                    t_data = raw_df.dropna(how='all')

                if len(t_data) < 260:
                    logger.debug(f"{ticker}: onvoldoende data ({len(t_data)} rijen), overgeslagen.")
                    continue

                p, f, s_line, e200, v_ma, rsi, atr, adx, vol = bereken_indicatoren_vectorized(
                    t_data, s_per, t_per, use_tr, is_hyp
                )

                # --- BACKTEST (laatste 252 dagen) ---
                p_bt    = p.iloc[-252:]
                f_bt    = f.iloc[-252:]
                s_bt    = s_line.iloc[-252:]
                e_bt    = e200.iloc[-252:]
                v_bt    = vol.iloc[-252:]
                v_ma_bt = v_ma.iloc[-252:]
                atr_bt  = atr.iloc[-252:]
                adx_bt  = adx.iloc[-252:]

                profit  = 0.0
                pos     = False
                instap  = 0.0
                high_p  = 0.0
                sl_val  = 0.0
                kosten  = 15.0 + (inzet * 0.0035)

                for i in range(1, len(p_bt)):
                    cp_i = p_bt.iloc[i]

                    if not pos:
                        kruist_omhoog = (
                            f_bt.iloc[i]   > s_bt.iloc[i] and
                            f_bt.iloc[i-1] <= s_bt.iloc[i-1]
                        )
                        volume_ok  = v_bt.iloc[i] > (v_ma_bt.iloc[i] * 0.6)
                        adx_ok     = adx_bt.iloc[i] > 15
                        trend_ok   = (not use_tr) or (cp_i > e_bt.iloc[i])

                        if kruist_omhoog and adx_ok and volume_ok and trend_ok:
                            instap = cp_i
                            high_p = cp_i
                            sl_val = cp_i - (2 * atr_bt.iloc[i])
                            pos    = True
                            profit -= kosten
                    else:
                        high_p = max(high_p, cp_i)
                        sl_val = max(sl_val, high_p - (2 * atr_bt.iloc[i]))

                        stop_geraakt   = cp_i < sl_val
                        kruis_omlaag   = f_bt.iloc[i] < s_bt.iloc[i]

                        if stop_geraakt or kruis_omlaag:
                            profit += (inzet * (cp_i / instap) - inzet) - kosten
                            pos     = False

                # FIX: open positie aan einde backtest mark-to-market verrekenen
                if pos:
                    laatste = p_bt.iloc[-1]
                    profit += (inzet * (laatste / instap) - inzet) - kosten
                    logger.debug(f"{ticker}/{strat_key}: open positie verrekend @ {laatste:.2f}")

                res[strat_key] += profit

                # --- ACTUEEL SIGNAAL ---
                cp   = p.iloc[-1]
                catr = atr.iloc[-1]
                crsi = rsi.iloc[-1]

                kruist_nu_omhoog = f.iloc[-1] > s_line.iloc[-1] and f.iloc[-2] <= s_line.iloc[-2]
                kruist_nu_omlaag = f.iloc[-1] < s_line.iloc[-1] and f.iloc[-2] >= s_line.iloc[-2]
                vol_ok   = vol.iloc[-1] > (v_ma.iloc[-1] * 0.6)
                adx_ok   = adx.iloc[-1] > 15
                trend_ok = (not use_tr) or (cp > e200.iloc[-1])
                l_rsi    = "💎 CRSI" if is_hyp else "📊 RSI"
                y_l      = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"

                if kruist_nu_omhoog and adx_ok and vol_ok and trend_ok:
                    sig[strat_key].append(
                        f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | "
                        f"⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | "
                        f"🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"
                    )
                elif kruist_nu_omlaag:
                    sig[strat_key].append(
                        f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | "
                        f"⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | "
                        f"🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"
                    )

            except Exception as e:
                logger.warning(f"{ticker}/{strat_key}: overgeslagen – {e}")
                continue

    # --- RAPPORT (ongewijzigd) ---
    def get_s(lst: list) -> str:
        return "\n".join(lst) if lst else "Geen actie"

    rapport_lijst = [
        f"📊 *{label} {naam_sector} RAPPORT*", f"_{nu}_", "----------------------------------",
        f"🐢 *Traag (50/200):* €{100000 + res['T']:,.0f}",
        f"⚡ *Snel (20/50):* €{100000 + res['S']:,.0f}",
        f"🚀 *Hyper Trend:* €{100000 + res['HT']:,.0f}",
        f"🔥 *Hyper Scalp:* €{100000 + res['HS']:,.0f}",
        "", "🛡️ *SIGNALEN TRAAG (RSI):*",         get_s(sig["T"]),
        "", "🎯 *SIGNALEN SNEL (RSI):*",           get_s(sig["S"]),
        "", "📈 *SIGNALEN HYPER TREND (CRSI):*",   get_s(sig["HT"]),
        "", "⚡ *SIGNALEN HYPER SCALP (CRSI):*",   get_s(sig["HS"]),
        "", "💡 _ATR %: <2% laag, >5% hoog. RSI: >70 overbought, <30 oversold. CRSI: >90 overbought, <10 oversold_",
    ]
    stuur_telegram("\n".join(rapport_lijst))
    logger.info(f"Sector '{naam_sector}' klaar.")


# ---------------------------------------------------------------------------
# MAIN
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
