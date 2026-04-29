from __future__ import annotations

import yfinance as yf
import pandas as pd
import os
import requests
import numpy as np
import json
from dotenv import load_dotenv
from datetime import datetime, date
import logging
import time

# ---------------------------------------------------------------------------
# CONFIG & LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN   = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

POSITIES_BESTAND = "posities.json"  # bestand voor open posities

# ---------------------------------------------------------------------------
# MRA PARAMETERS
# ---------------------------------------------------------------------------
MRA_BB_STD      = 2.2
MRA_IBS_MAX     = 0.30

MRA_SNEL_WINST  = 1.12
MRA_SNEL_MA     = 5

MRA_TRAAG_WINST = 1.25
MRA_TRAAG_MA    = 10
MRA_TRAAG_HOLD  = 5

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID:
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
        return True
    except Exception as e:
        logger.error(f"Telegram fout: {e}")
        return False

# ---------------------------------------------------------------------------
# POSITIEBEHEER — lezen / schrijven / updaten
# ---------------------------------------------------------------------------
def laad_posities() -> dict:
    """Laad open posities uit JSON bestand."""
    if not os.path.exists(POSITIES_BESTAND):
        return {}
    try:
        with open(POSITIES_BESTAND, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fout bij laden posities: {e}")
        return {}

def sla_posities_op(posities: dict) -> None:
    """Sla posities op naar JSON bestand."""
    try:
        with open(POSITIES_BESTAND, "w") as f:
            json.dump(posities, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Fout bij opslaan posities: {e}")

def voeg_positie_toe(ticker: str, strategie: str, aankoopprijs: float) -> None:
    """Voeg een nieuwe positie toe."""
    posities = laad_posities()
    sleutel  = f"{ticker}_{strategie}"
    posities[sleutel] = {
        "ticker":       ticker,
        "strategie":    strategie,
        "aankoopprijs": round(aankoopprijs, 4),
        "datum":        date.today().isoformat(),
        "dagen":        0,
    }
    sla_posities_op(posities)
    logger.info(f"Positie toegevoegd: {sleutel} @ €{aankoopprijs:.2f}")

def verwijder_positie(sleutel: str) -> None:
    """Verwijder een gesloten positie."""
    posities = laad_posities()
    if sleutel in posities:
        del posities[sleutel]
        sla_posities_op(posities)
        logger.info(f"Positie verwijderd: {sleutel}")

# ---------------------------------------------------------------------------
# DAGELIJKS OVERZICHT OPEN POSITIES → Telegram
# ---------------------------------------------------------------------------
def stuur_overzicht_posities(prijzen: dict) -> None:
    """Stuur een overzicht van alle open posities naar Telegram."""
    posities = laad_posities()
    nu       = datetime.now().strftime("%d/%m/%Y %H:%M")

    if not posities:
        stuur_telegram(
            f"📂 *OPEN POSITIES*\n_{nu}_\n\nGeen open posities."
        )
        return

    regels = [f"📂 *OPEN POSITIES*", f"_{nu}_", ""]

    totaal_winst = 0.0
    for sleutel, pos in posities.items():
        ticker      = pos["ticker"]
        strategie   = pos["strategie"]
        aankoop     = pos["aankoopprijs"]
        dagen       = pos.get("dagen", 0)
        koers_nu    = prijzen.get(ticker)

        if koers_nu is None or pd.isna(koers_nu):
            regels.append(f"• `{ticker}` ({strategie}): ❓ Geen koers beschikbaar")
            continue

        pct = ((koers_nu / aankoop) - 1) * 100
        totaal_winst += pct
        pijl = "📈" if pct >= 0 else "📉"

        regels.append(
            f"• `{ticker}` ({strategie}): {pijl} {pct:+.1f}% | "
            f"€{aankoop:.2f} → €{koers_nu:.2f} | {dagen}d"
        )

    regels.append("")
    regels.append(f"_Totaal open: {len(posities)} posities_")

    stuur_telegram("\n".join(regels))

# ---------------------------------------------------------------------------
# VERKOOPSIGNALEN CHECKER → Telegram
# ---------------------------------------------------------------------------
def check_verkoop_signalen(prijzen: dict, ma_waarden: dict) -> None:
    """
    Controleer alle open posities op verkoopvoorwaarden.
    Stuurt een Telegram bericht met alle verkoop- en houdadviezen.
    Verwijdert automatisch gesloten posities uit het bestand.
    """
    posities = laad_posities()
    if not posities:
        return

    nu     = datetime.now().strftime("%d/%m/%Y %H:%M")
    verkoop_regels = []
    houd_regels    = []
    gesloten       = []

    for sleutel, pos in posities.items():
        ticker    = pos["ticker"]
        strategie = pos["strategie"]
        aankoop   = pos["aankoopprijs"]
        dagen     = pos.get("dagen", 0)

        koers_nu  = prijzen.get(ticker)
        if koers_nu is None or pd.isna(koers_nu):
            continue

        pct  = ((koers_nu / aankoop) - 1) * 100
        pijl = "📈" if pct >= 0 else "📉"

        # Haal MA waarden op
        ma5_nu  = ma_waarden.get(ticker, {}).get("ma5")
        ma10_nu = ma_waarden.get(ticker, {}).get("ma10")

        y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"

        # --- MRA SNEL uitstap ---
        if strategie == "MRAS":
            reden = None
            if ma5_nu and not pd.isna(ma5_nu) and koers_nu > ma5_nu:
                reden = f"MA5 doorbroken (MA5=€{ma5_nu:.2f})"
            elif koers_nu >= aankoop * MRA_SNEL_WINST:
                reden = f"+{int((MRA_SNEL_WINST-1)*100)}% winstlimiet bereikt"

            if reden:
                verkoop_regels.append(
                    f"• `{ticker}`: 🔴 *VERKOOP* (Snel)\n"
                    f"  {pijl} Koers: €{koers_nu:.2f} | Aankoop: €{aankoop:.2f} | {pct:+.1f}%\n"
                    f"  ✅ Reden: {reden} | {dagen}d | {y_l}"
                )
                gesloten.append(sleutel)
            else:
                # Houd vast — toon status
                ma5_info = f"MA5=€{ma5_nu:.2f}" if ma5_nu else "MA5 onbekend"
                verkoop_at = aankoop * MRA_SNEL_WINST
                houd_regels.append(
                    f"• `{ticker}` (Snel): ⏳ Houd vast\n"
                    f"  {pijl} {pct:+.1f}% | €{koers_nu:.2f} | {ma5_info} | "
                    f"Winst bij €{verkoop_at:.2f} | {dagen}d"
                )

        # --- MRA TRAAG uitstap ---
        elif strategie == "MRAT":
            reden = None
            if dagen >= MRA_TRAAG_HOLD:
                if ma10_nu and not pd.isna(ma10_nu) and koers_nu > ma10_nu:
                    reden = f"MA10 doorbroken (MA10=€{ma10_nu:.2f})"
                elif koers_nu >= aankoop * MRA_TRAAG_WINST:
                    reden = f"+{int((MRA_TRAAG_WINST-1)*100)}% winstlimiet bereikt"

            if reden:
                verkoop_regels.append(
                    f"• `{ticker}`: 🔴 *VERKOOP* (Traag)\n"
                    f"  {pijl} Koers: €{koers_nu:.2f} | Aankoop: €{aankoop:.2f} | {pct:+.1f}%\n"
                    f"  ✅ Reden: {reden} | {dagen}d | {y_l}"
                )
                gesloten.append(sleutel)
            else:
                # Nog niet verkopen — minimum houdperiode of MA nog niet bereikt
                ma10_info    = f"MA10=€{ma10_nu:.2f}" if ma10_nu else "MA10 onbekend"
                nog_dagen    = max(0, MRA_TRAAG_HOLD - dagen)
                verkoop_at   = aankoop * MRA_TRAAG_WINST
                if nog_dagen > 0:
                    extra = f"Nog {nog_dagen}d min. houdperiode"
                else:
                    extra = f"MA10 nog niet bereikt | Winst bij €{verkoop_at:.2f}"
                houd_regels.append(
                    f"• `{ticker}` (Traag): ⏳ Houd vast\n"
                    f"  {pijl} {pct:+.1f}% | €{koers_nu:.2f} | {ma10_info} | {extra} | {dagen}d"
                )

        # --- Update dagenteller ---
        posities[sleutel]["dagen"] = dagen + 1

    # Verwijder gesloten posities
    for sleutel in gesloten:
        if sleutel in posities:
            del posities[sleutel]
    sla_posities_op(posities)

    # Stuur Telegram bericht
    if not verkoop_regels and not houd_regels:
        return

    regels = [f"🔔 *MRA VERKOOP CHECK*", f"_{nu}_", ""]

    if verkoop_regels:
        regels.append("🔴 *VERKOPEN:*")
        regels.extend(verkoop_regels)
        regels.append("")

    if houd_regels:
        regels.append("⏳ *HOUD VAST:*")
        regels.extend(houd_regels)

    stuur_telegram("\n".join(regels))

# ---------------------------------------------------------------------------
# INDICATOREN
# ---------------------------------------------------------------------------
def bereken_indicatoren_vectorized(
    df: pd.DataFrame, s: int, t: int, use_trend_filter: bool, is_hyper: bool
) -> tuple:
    p = df["Close"].ffill()
    h = df["High"].ffill()
    l = df["Low"].ffill()
    v = df["Volume"].ffill()

    f_line = p.rolling(window=s).mean() if s >= 20 else p.ewm(span=s, adjust=False).mean()
    s_line = p.rolling(window=t).mean() if t >= 50 else p.ewm(span=t, adjust=False).mean()
    ema100 = p.ewm(span=100, adjust=False).mean()
    vol_ma = v.rolling(window=20).mean()

    delta = p.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rsi_val = 100 - (100 / (1 + gain / (loss + 1e-10)))

    ma20     = p.rolling(20).mean()
    std20    = p.rolling(20).std()
    lower_bb = ma20 - (MRA_BB_STD * std20)
    ibs      = (p - l) / (h - l + 1e-10)
    ma5      = p.rolling(5).mean()

    if is_hyper:
        change     = np.sign(delta).fillna(0)
        streak     = change.groupby((change != change.shift()).cumsum()).cumsum()
        s_delta    = streak.diff().fillna(0)
        s_gain     = s_delta.where(s_delta > 0, 0.0).rolling(2).mean()
        s_loss     = (-s_delta.where(s_delta < 0, 0.0)).rolling(2).mean()
        streak_rsi = 100 - (100 / (1 + s_gain / (s_loss + 1e-10))).fillna(50)
        rsi3_gain  = delta.where(delta > 0, 0.0).ewm(alpha=1/3, adjust=False).mean()
        rsi3_loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/3, adjust=False).mean()
        rsi3       = 100 - (100 / (1 + rsi3_gain / (rsi3_loss + 1e-10)))
        p_rank     = delta.rolling(100).apply(
            lambda x: (x[:-1] < x[-1]).sum() / 99.0 * 100 if len(x) > 0 else 50, raw=True
        )
        rsi_val = (rsi3 + streak_rsi + p_rank) / 3

    tr       = pd.concat([h - l, (h - p.shift()).abs(), (l - p.shift()).abs()], axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/14, adjust=False).mean()
    up       = h.diff().clip(lower=0)
    down     = (-l.diff()).clip(lower=0)
    plus_di  = 100 * (up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    adx      = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    return p, f_line, s_line, ema100, vol_ma, rsi_val, atr, adx, v, ibs, lower_bb, ma5

# ---------------------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------------------
def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> tuple[dict, dict]:
    """
    Verwerkt een tickerlijst. Geeft (prijzen, ma_waarden) terug voor
    gebruik in verkoop-check en positie-overzicht.
    """
    prijzen    = {}
    ma_waarden = {}

    if not os.path.exists(bestandsnaam):
        logger.warning(f"Bestand {bestandsnaam} niet gevonden.")
        return prijzen, ma_waarden

    nu = datetime.now().strftime("%d/%m/%Y %H:%M")

    with open(bestandsnaam, "r") as f:
        content = f.read().replace("\n", ",").replace("$", "")
        tickers = sorted(list(set([t.strip().upper() for t in content.split(",") if t.strip()])))
    if not tickers:
        return prijzen, ma_waarden

    logger.info(f"Start analyse voor {naam_sector} met {len(tickers)} tickers.")

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download fout: {e}")
        return prijzen, ma_waarden

    inzet  = 2500.0
    res        = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRAS": 0.0, "MRAT": 0.0}
    num_trades = {"T": 0,   "S": 0,   "HT": 0,   "HS": 0,   "MRAS": 0,   "MRAT": 0}
    sig        = {"T": [],  "S": [],  "HT": [],   "HS": [],  "MRAS": [],  "MRAT": []}

    STRATS = [
        ("T",  50, 200, True,  False),
        ("S",  20,  50, True,  False),
        ("HT",  9,  21, True,  True),
        ("HS",  9,  21, False, True),
    ]

    for ticker in tickers:
        try:
            if len(tickers) > 1:
                t_data = raw_df.xs(ticker, axis=1, level=1).dropna(how="all")
            else:
                t_data = raw_df.dropna(how="all")

            if len(t_data) < 250:
                continue

            p, f, sl, e100, v_ma, rsi, atr, adx, vol, ibs, l_bb, ma5 = bereken_indicatoren_vectorized(
                t_data, 50, 200, True, False
            )
            kosten = 15.0 + (inzet * 0.0035)

            # Sla actuele koers & MA's op voor positiebeheer
            prijzen[ticker]    = float(p.iloc[-1])
            ma10_serie         = p.rolling(MRA_TRAAG_MA).mean()
            ma_waarden[ticker] = {
                "ma5":  float(ma5.iloc[-1])      if not pd.isna(ma5.iloc[-1])      else None,
                "ma10": float(ma10_serie.iloc[-1]) if not pd.isna(ma10_serie.iloc[-1]) else None,
            }

            # --- STRATEGIEËN T / S / HT / HS ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                pi, fi, sli, ei, vmai, rsii, atri, dxi, voli, _, _, _ = bereken_indicatoren_vectorized(
                    t_data, s_p, t_p, utr, ihyp
                )

                pb  = pi.iloc[200:];  fb  = fi.iloc[200:];  sb  = sli.iloc[200:]
                eb  = ei.iloc[200:];  vb  = voli.iloc[200:]; vmb = vmai.iloc[200:]
                ab  = atri.iloc[200:]; dxb = dxi.iloc[200:]

                pr, pos, ins, hi = 0.0, False, 0.0, 0.0
                for i in range(1, len(pb)):
                    cp = pb.iloc[i]
                    if not pos:
                        if (
                            fb.iloc[i] > sb.iloc[i]
                            and fb.iloc[i-1] <= sb.iloc[i-1]
                            and dxb.iloc[i] > 15
                            and vb.iloc[i] > (vmb.iloc[i] * 0.6)
                            and ((not utr) or cp > eb.iloc[i])
                        ):
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

                if (
                    fi.iloc[-1] > sli.iloc[-1]
                    and fi.iloc[-2] <= sli.iloc[-2]
                    and dxi.iloc[-1] > 15
                    and voli.iloc[-1] > (vmai.iloc[-1] * 0.6)
                    and ((not utr) or cp > ei.iloc[-1])
                ):
                    sig[skey].append(
                        f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | "
                        f"{l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"
                    )
                elif fi.iloc[-1] < sli.iloc[-1] and fi.iloc[-2] >= sli.iloc[-2]:
                    sig[skey].append(
                        f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | "
                        f"{l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f} | {y_l}"
                    )

            # --- MRA SNEL backtest ---
            pb_mra  = p.iloc[-252:].reset_index(drop=True)
            lbb_mra = l_bb.iloc[-252:].reset_index(drop=True)
            ibs_mra = ibs.iloc[-252:].reset_index(drop=True)
            m5_mra  = ma5.iloc[-252:].reset_index(drop=True)

            pr_ms, pos_ms, ins_ms = 0.0, False, 0.0
            for i in range(1, len(pb_mra)):
                cp    = pb_mra.iloc[i]
                lbb_i = lbb_mra.iloc[i]
                ibs_i = ibs_mra.iloc[i]
                if pd.isna(cp) or pd.isna(lbb_i) or pd.isna(ibs_i):
                    continue
                if not pos_ms:
                    if cp < lbb_i and ibs_i < MRA_IBS_MAX:
                        ins_ms, pos_ms = cp, True
                        pr_ms -= kosten
                        num_trades["MRAS"] += 1
                else:
                    m5_i = m5_mra.iloc[i]
                    if pd.isna(m5_i):
                        continue
                    if cp > m5_i or cp > (ins_ms * MRA_SNEL_WINST):
                        pr_ms += (inzet * (cp / ins_ms) - inzet) - kosten
                        pos_ms = False
            if pos_ms:
                pr_ms += (inzet * (pb_mra.iloc[-1] / ins_ms) - inzet) - kosten
            res["MRAS"] += pr_ms

            # --- MRA TRAAG backtest ---
            ma10_mra = p.rolling(MRA_TRAAG_MA).mean().iloc[-252:].reset_index(drop=True)

            pr_mt, pos_mt, ins_mt, hold_mt = 0.0, False, 0.0, 0
            for i in range(1, len(pb_mra)):
                cp    = pb_mra.iloc[i]
                lbb_i = lbb_mra.iloc[i]
                ibs_i = ibs_mra.iloc[i]
                if pd.isna(cp) or pd.isna(lbb_i) or pd.isna(ibs_i):
                    continue
                if not pos_mt:
                    if cp < lbb_i and ibs_i < MRA_IBS_MAX:
                        ins_mt, pos_mt, hold_mt = cp, True, 0
                        pr_mt -= kosten
                        num_trades["MRAT"] += 1
                else:
                    hold_mt += 1
                    ma10_i = ma10_mra.iloc[i]
                    if pd.isna(ma10_i):
                        continue
                    if hold_mt >= MRA_TRAAG_HOLD:
                        if cp > ma10_i or cp > (ins_mt * MRA_TRAAG_WINST):
                            pr_mt += (inzet * (cp / ins_mt) - inzet) - kosten
                            pos_mt = False
            if pos_mt:
                pr_mt += (inzet * (pb_mra.iloc[-1] / ins_mt) - inzet) - kosten
            res["MRAT"] += pr_mt

            # --- Actuele MRA KOOP signalen + automatisch positie registreren ---
            cp_now  = p.iloc[-1]
            lbb_now = l_bb.iloc[-1]
            ibs_now = ibs.iloc[-1]
            if not pd.isna(cp_now) and not pd.isna(lbb_now) and not pd.isna(ibs_now):
                if cp_now < lbb_now and ibs_now < MRA_IBS_MAX:
                    y_l = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    sig["MRAS"].append(
                        f"• `{ticker}`: 🛡️ *Munger Snel* | €{cp_now:.2f} | "
                        f"📊 RSI: {rsi.iloc[-1]:.1f} | {y_l}"
                    )
                    sig["MRAT"].append(
                        f"• `{ticker}`: 🐢 *Munger Traag* | €{cp_now:.2f} | "
                        f"📊 RSI: {rsi.iloc[-1]:.1f} | {y_l}"
                    )
                    # ✅ Automatisch positie registreren
                    voeg_positie_toe(ticker, "MRAS", float(cp_now))
                    voeg_positie_toe(ticker, "MRAT", float(cp_now))

        except Exception as e:
            logger.error(f"Fout bij ticker {ticker}: {e}")
            continue

    # --- RAPPORT ---
    def fmt(n):
        return f"€{100000 + n:,.0f}"

    def get_s(lst):
        return "\n".join(lst) if lst else "Geen actie"

    deel1 = [
        f"📊 *{label} {naam_sector} RAPPORT ult*",
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
    ]

    deel2 = [
        f"📊 *{label} {naam_sector} (2/2)*",
        "",
        "📈 *SIGNALEN HYPER TREND (CRSI):*",
        get_s(sig["HT"]),
        "",
        "⚡ *SIGNALEN HYPER SCALP (CRSI):*",
        get_s(sig["HS"]),
        "",
        "🛡️ *SIGNALEN MRA SNEL:*",
        get_s(sig["MRAS"]),
        "",
        "🐢 *SIGNALEN MRA TRAAG:*",
        get_s(sig["MRAT"]),
        "",
        "⚙️ *PARAMETERS:*",
        f"_Trend: ADX>15 | Vol>0.6x MA20 | EMA100 filter | Trailing stop 2x ATR_",
        f"_MRA instap: BB {MRA_BB_STD}σ | IBS<{MRA_IBS_MAX} (geen extra filters)_",
        f"_MRA Snel: uitstap MA{MRA_SNEL_MA} of +{int((MRA_SNEL_WINST-1)*100)}%_",
        f"_MRA Traag: min {MRA_TRAAG_HOLD}d, uitstap MA{MRA_TRAAG_MA} of +{int((MRA_TRAAG_WINST-1)*100)}%_",
        f"_Inzet: €{inzet:.0f} | Kosten: €{kosten:.2f}/trade_",
        "",
        "💾 _Posities automatisch bijgehouden in posities.json_",
    ]

    stuur_telegram("\n".join(deel1))
    time.sleep(1)
    stuur_telegram("\n".join(deel2))

    return prijzen, ma_waarden

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

    # Verzamel alle prijzen & MA's over alle sectoren
    alle_prijzen    = {}
    alle_ma_waarden = {}

    for nr, naam in sectoren.items():
        p, m = voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        alle_prijzen.update(p)
        alle_ma_waarden.update(m)
        time.sleep(2)

    # ✅ Na alle sectoren: check verkoopsignalen & stuur overzicht
    time.sleep(1)
    check_verkoop_signalen(alle_prijzen, alle_ma_waarden)
    time.sleep(1)
    stuur_overzicht_posities(alle_prijzen)


if __name__ == "__main__":
    main()
