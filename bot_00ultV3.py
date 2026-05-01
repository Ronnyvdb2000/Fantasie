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
import re

# ---------------------------------------------------------------------------
# CONFIG & LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ---------------------------------------------------------------------------
# MRA PARAMETERS — geen filters, puur BB + IBS (origineel bewezen €945.000)
# ---------------------------------------------------------------------------
MRA_BB_STD      = 2.2   # Bollinger Band breedte
MRA_IBS_MAX     = 0.30  # IBS (sluit in onderste 30% van dagrange)

# MRA SNEL uitstap — origineel
MRA_SNEL_WINST  = 1.12  # +12% winstlimiet
MRA_SNEL_MA     = 5     # verkoop boven MA5

# MRA TRAAG uitstap — minimum houdperiode, dan MA10 of +25%
MRA_TRAAG_WINST = 1.25  # +25% winstlimiet
MRA_TRAAG_MA    = 10    # verkoop boven MA10
MRA_TRAAG_HOLD  = 5     # minimum 5 dagen vasthouden

# ---------------------------------------------------------------------------
# TICKER CORRECTIE - lost suffix problemen op
# ---------------------------------------------------------------------------
def corrigeer_ticker(ticker: str) -> str:
    """
    Corrigeert veelvoorkomende ticker fouten:
    - AS: Belgische aandelen op Euronext Brussels
    - BR: Brussels (alternatief)
    - PA: Euronext Paris
    - DE: Xetra (Duitsland)
    """
    ticker_orig = ticker.upper().strip()
    
    # Bekende problematische tickers met land-suffix
    correcties = {
        # België (.AS is correct voor Euronext Brussels)
        "UCB": "UCB.BR",      # UCB op Brussels
        "KBC": "KBC.BR",       # KBC Bank
        "SOLB": "SOLB.BR",     # Solvay
        "ACKB": "ACKB.BR",     # Ackermans & van Haaren
        "COLR": "COLR.BR",     # Colruyt
        "PROX": "PROX.BR",     # Proximus
        "AGFB": "AGFB.BR",     # Ageas
        "ELI": "ELI.BR",       # Elis (let op: kan ook Frans zijn)
        
        # Frankrijk (.PA is correct voor Euronext Paris)
        "AIR": "AIR.PA",       # Airbus
        "SU": "SU.PA",         # Schneider Electric
        "TTE": "TTE.PA",       # TotalEnergies
        "SAN": "SAN.PA",       # Sanofi
        "BNP": "BNP.PA",       # BNP Paribas
        
        # Duitsland (.DE voor Xetra)
        "SAP": "SAP.DE",       # SAP
        "SIEGY": "SIE.DE",     # Siemens
        "BAS": "BAS.DE",       # BASF
        "BMW": "BMW.DE",       # BMW
        
        # VS (geen suffix)
        "BRK.B": "BRK-B",      # Berkshire Hathaway (punt wordt streepje)
    }
    
    # Check of ticker in correcties staat
    if ticker_orig in correcties:
        nieuwe = correcties[ticker_orig]
        logger.info(f"Ticker {ticker_orig} gecorrigeerd naar {nieuwe}")
        return nieuwe
    
    # Check of ticker al een suffix heeft (.AS, .BR, .PA, .DE)
    heeft_suffix = re.search(r'\.(AS|BR|PA|DE|L|TO|V)$', ticker_orig)
    if heeft_suffix:
        return ticker_orig
    
    # Geen suffix: probeer te raden op basis van ticker lengte/patroon
    # (standaard: US tickers hebben geen suffix nodig)
    return ticker_orig

def valideer_ticker(ticker: str) -> tuple:
    """
    Checkt of ticker valide is door 1 dag data op te halen.
    Retourneert (is_valide, gecorrigeerde_ticker, error_msg)
    """
    try:
        # Probeer ticker zoals hij is
        test = yf.Ticker(ticker)
        hist = test.history(period="5d", progress=False)
        
        if not hist.empty and len(hist) >= 2:
            return True, ticker, None
        
        # Probeer met .AS suffix (België)
        if not ticker.endswith('.AS'):
            test_as = yf.Ticker(f"{ticker}.AS")
            hist_as = test_as.history(period="5d", progress=False)
            if not hist_as.empty and len(hist_as) >= 2:
                logger.info(f"Ticker {ticker} werkt met .AS suffix")
                return True, f"{ticker}.AS", None
        
        # Probeer met .BR suffix (Brussels alternatief)
        if not ticker.endswith('.BR'):
            test_br = yf.Ticker(f"{ticker}.BR")
            hist_br = test_br.history(period="5d", progress=False)
            if not hist_br.empty and len(hist_br) >= 2:
                logger.info(f"Ticker {ticker} werkt met .BR suffix")
                return True, f"{ticker}.BR", None
        
        # Probeer met .PA suffix (Parijs)
        if not ticker.endswith('.PA'):
            test_pa = yf.Ticker(f"{ticker}.PA")
            hist_pa = test_pa.history(period="5d", progress=False)
            if not hist_pa.empty and len(hist_pa) >= 2:
                logger.info(f"Ticker {ticker} werkt met .PA suffix")
                return True, f"{ticker}.PA", None
        
        return False, ticker, f"Geen data gevonden voor {ticker}"
        
    except Exception as e:
        return False, ticker, f"Fout bij valideren {ticker}: {e}"

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

    return p, f_line, s_line, ema100, vol_ma, rsi_val, atr, adx, v, ibs, lower_bb, ma5

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
        ruwe_tickers = list(set([t.strip().upper() for t in content.split(',') if t.strip()]))
    
    # VALIDEER EN CORRIGEER TICKERS
    geldige_tickers = []
    for ticker in ruwe_tickers:
        is_valide, gecorrigeerd, error = valideer_ticker(ticker)
        if is_valide:
            geldige_tickers.append(gecorrigeerd)
        else:
            logger.warning(f"Ongeldige ticker overgeslagen: {ticker} - {error}")
            stuur_telegram(f"⚠️ *Waarschuwing:* Ticker `{ticker}` in {naam_sector} werkt niet.\nReden: {error}")
    
    if not geldige_tickers:
        logger.warning(f"Geen geldige tickers in {bestandsnaam}")
        return
    
    tickers = sorted(geldige_tickers)
    logger.info(f"Start analyse voor {naam_sector} met {len(tickers)} tickers (van {len(ruwe_tickers)} origineel).")

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download fout: {e}")
        return

    inzet = 2500.0
    res        = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRAS": 0.0, "MRAT": 0.0}
    num_trades = {"T": 0,   "S": 0,   "HT": 0,   "HS": 0,   "MRAS": 0,   "MRAT": 0}
    sig        = {"T": [],  "S": [],  "HT": [],   "HS": [],  "MRAS": [],  "MRAT": []}

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

            if len(t_data) < 250: 
                logger.info(f"{ticker}: onvoldoende data ({len(t_data)} dagen)")
                continue

            # Bereken basis indicatoren voor MRA
            p, f, sl, e100, v_ma, rsi, atr, adx, vol, ibs, l_bb, ma5 = bereken_indicatoren_vectorized(t_data, 50, 200, True, False)
            kosten = 15.0 + (inzet * 0.0035)

            # --- STRATEGIEËN T / S / HT / HS (ongewijzigd) ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                pi, fi, sli, ei, vmai, rsii, atri, dxi, voli, _, _, _ = bereken_indicatoren_vectorized(t_data, s_p, t_p, utr, ihyp)

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

                # Signalen met links
                cp    = pi.iloc[-1]
                catr  = atri.iloc[-1]
                crsi  = rsii.iloc[-1]
                
                # LINKS naar Yahoo Finance (geen API calls, alleen URLs)
                yl_grafiek = f"[📈](https://finance.yahoo.com/quote/{ticker})"
                yl_fund = f"[💰](https://finance.yahoo.com/quote/{ticker}/financials)"
                yl_stats = f"[📊](https://finance.yahoo.com/quote/{ticker}/key-statistics)"
                l_rsi = "💎 CRSI" if ihyp else "📊 RSI"

                if (fi.iloc[-1] > sli.iloc[-1] and fi.iloc[-2] <= sli.iloc[-2]
                        and dxi.iloc[-1] > 15
                        and voli.iloc[-1] > (vmai.iloc[-1] * 0.6)
                        and ((not utr) or cp > ei.iloc[-1])):
                    sig[skey].append(f"• `{ticker}`: 🟢 *KOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f}\n  └─ {yl_grafiek} Grafiek | {yl_fund} Winst/Div | {yl_stats} Stats")
                elif fi.iloc[-1] < sli.iloc[-1] and fi.iloc[-2] >= sli.iloc[-2]:
                    sig[skey].append(f"• `{ticker}`: 🔴 *VERKOOP* | €{cp:.2f} | ⚡ ATR: {catr:.2f} | {l_rsi}: {crsi:.1f} | 🛡️ SL: €{cp-(2*catr):.2f}\n  └─ {yl_grafiek} Grafiek | {yl_fund} Winst/Div | {yl_stats} Stats")

            # ---------------------------------------------------------------
            # MRA SNEL backtest
            # ---------------------------------------------------------------
            pb_mra  = p.iloc[-252:].reset_index(drop=True)
            lbb_mra = l_bb.iloc[-252:].reset_index(drop=True)
            ibs_mra = ibs.iloc[-252:].reset_index(drop=True)
            m5_mra  = ma5.iloc[-252:].reset_index(drop=True)

            pr_ms, pos_ms, ins_ms = 0.0, False, 0.0
            for i in range(1, len(pb_mra)):
                cp    = pb_mra.iloc[i]
                lbb_i = lbb_mra.iloc[i]
                ibs_i = ibs_mra.iloc[i]
                if pd.isna(cp) or pd.isna(lbb_i) or pd.isna(ibs_i): continue
                if not pos_ms:
                    if cp < lbb_i and ibs_i < MRA_IBS_MAX:
                        ins_ms, pos_ms = cp, True
                        pr_ms -= kosten
                        num_trades["MRAS"] += 1
                else:
                    m5_i = m5_mra.iloc[i]
                    if pd.isna(m5_i): continue
                    if cp > m5_i or cp > (ins_ms * MRA_SNEL_WINST):
                        pr_ms += (inzet * (cp / ins_ms) - inzet) - kosten
                        pos_ms = False
            if pos_ms:
                pr_ms += (inzet * (pb_mra.iloc[-1] / ins_ms) - inzet) - kosten
            res["MRAS"] += pr_ms

            # ---------------------------------------------------------------
            # MRA TRAAG backtest
            # ---------------------------------------------------------------
            ma10_mra = p.rolling(MRA_TRAAG_MA).mean().iloc[-252:].reset_index(drop=True)

            pr_mt, pos_mt, ins_mt, hold_mt = 0.0, False, 0.0, 0
            for i in range(1, len(pb_mra)):
                cp    = pb_mra.iloc[i]
                lbb_i = lbb_mra.iloc[i]
                ibs_i = ibs_mra.iloc[i]
                if pd.isna(cp) or pd.isna(lbb_i) or pd.isna(ibs_i): continue
                if not pos_mt:
                    if cp < lbb_i and ibs_i < MRA_IBS_MAX:
                        ins_mt, pos_mt, hold_mt = cp, True, 0
                        pr_mt -= kosten
                        num_trades["MRAT"] += 1
                else:
                    hold_mt += 1
                    ma10_i = ma10_mra.iloc[i]
                    if pd.isna(ma10_i): continue
                    if hold_mt >= MRA_TRAAG_HOLD:
                        if cp > ma10_i or cp > (ins_mt * MRA_TRAAG_WINST):
                            pr_mt += (inzet * (cp / ins_mt) - inzet) - kosten
                            pos_mt = False
            if pos_mt:
                pr_mt += (inzet * (pb_mra.iloc[-1] / ins_mt) - inzet) - kosten
            res["MRAT"] += pr_mt

            # ---------------------------------------------------------------
            # ACTUELE MRA SIGNALEN met LINKS naar fundamentele data
            # ---------------------------------------------------------------
            cp_now  = p.iloc[-1]
            lbb_now = l_bb.iloc[-1]
            ibs_now = ibs.iloc[-1]
            if not pd.isna(cp_now) and not pd.isna(lbb_now) and not pd.isna(ibs_now):
                if cp_now < lbb_now and ibs_now < MRA_IBS_MAX:
                    # LINKS (geen API calls, alleen URLs)
                    yl_grafiek = f"[📈](https://finance.yahoo.com/quote/{ticker})"
                    yl_fund = f"[💰](https://finance.yahoo.com/quote/{ticker}/financials)"
                    yl_stats = f"[📊](https://finance.yahoo.com/quote/{ticker}/key-statistics)"
                    
                    sig["MRAS"].append(
                        f"• `{ticker}`: 🛡️ *Munger Snel* | €{cp_now:.2f} | 📊 RSI: {rsi.iloc[-1]:.1f}\n"
                        f"  └─ {yl_grafiek} Grafiek | {yl_fund} Winst/Div | {yl_stats} Stats"
                    )
                    sig["MRAT"].append(
                        f"• `{ticker}`: 🐢 *Munger Traag* | €{cp_now:.2f} | 📊 RSI: {rsi.iloc[-1]:.1f}\n"
                        f"  └─ {yl_grafiek} Grafiek | {yl_fund} Winst/Div | {yl_stats} Stats"
                    )

        except Exception as e:
            logger.error(f"Fout bij ticker {ticker}: {e}")
            continue

    # ---------------------------------------------------------------------------
    # RAPPORT — gesplitst in 2 berichten (Telegram max 4096 tekens)
    # ---------------------------------------------------------------------------
    def fmt(n): return f"€{100000 + n:,.0f}"
    def get_s(lst): return "\n".join(lst) if lst else "Geen actie"

    # Deel 1: resultaten + trend signalen
    deel1 = [
        f"📊 *{label} {naam_sector} RAPPORT ult V3*",
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

    # Deel 2: hyper + MRA signalen + parameters
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
    ]

    stuur_telegram("\n".join(deel1))
    time.sleep(1)
    stuur_telegram("\n".join(deel2))

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
