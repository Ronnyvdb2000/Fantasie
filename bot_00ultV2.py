from __future__ import annotations

import yfinance as yf
import pandas as pd
import os
import json
import requests
import smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
TOKEN          = os.getenv('TELEGRAM_TOKEN')
CHAT_ID        = os.getenv('TELEGRAM_CHAT_ID')
EMAIL_USER     = os.getenv('EMAIL_USER')
EMAIL_PASS     = os.getenv('EMAIL_PASS')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')      # ronnybot e-mailadres

PORTFOLIO_FILE = "portfolio.json"  # lokaal bestand met open posities

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
# E-MAIL
# ---------------------------------------------------------------------------
def stuur_mail(onderwerp: str, inhoud: str) -> bool:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        logger.warning("E-mail niet geconfigureerd.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From']    = EMAIL_USER
        msg['To']      = EMAIL_RECEIVER
        msg['Subject'] = onderwerp
        schoon = inhoud.replace('*', '').replace('`', '').replace('_', '').replace('•', '-')
        msg.attach(MIMEText(schoon, 'plain', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        logger.info(f"E-mail verzonden naar {EMAIL_RECEIVER}")
        return True
    except Exception as e:
        logger.error(f"E-mail fout: {e}")
        return False

# ---------------------------------------------------------------------------
# PORTFOLIO BEHEER
# Structuur portfolio.json:
# {
#   "TICKER": {
#     "strategie":    "MRAS" | "MRAT" | "T" | "S" | "HT" | "HS",
#     "datum_koop":   "2025-04-01",
#     "prijs_koop":   18.50,
#     "inzet":        2500.0,
#     "aantal_dagen": 0
#   }, ...
# }
# ---------------------------------------------------------------------------
def laad_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Portfolio laden fout: {e}")
        return {}

def sla_portfolio_op(portfolio: dict) -> None:
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
    except Exception as e:
        logger.error(f"Portfolio opslaan fout: {e}")

def voeg_positie_toe(ticker: str, strategie: str, prijs_koop: float, inzet: float) -> None:
    """Voeg nieuwe positie toe aan portfolio na koopsignaal."""
    portfolio = laad_portfolio()
    portfolio[ticker] = {
        "strategie":    strategie,
        "datum_koop":   datetime.now().strftime("%Y-%m-%d"),
        "prijs_koop":   round(prijs_koop, 4),
        "inzet":        inzet,
        "aantal_dagen": 0
    }
    sla_portfolio_op(portfolio)
    logger.info(f"Positie toegevoegd: {ticker} @ {prijs_koop:.2f} ({strategie})")

def verwijder_positie(ticker: str) -> None:
    """Verwijder positie uit portfolio na verkoopsignaal."""
    portfolio = laad_portfolio()
    if ticker in portfolio:
        del portfolio[ticker]
        sla_portfolio_op(portfolio)
        logger.info(f"Positie verwijderd: {ticker}")

def update_portfolio_en_rapport() -> str:
    """
    Haal actuele koersen op voor alle open posities.
    Controleer verkoopconditie per strategie.
    Geef rapport terug als string.
    Gesorteerd alfabetisch op ticker.
    """
    portfolio = laad_portfolio()
    if not portfolio:
        return "📂 *PORTFOLIO* — Geen open posities."

    tickers = sorted(portfolio.keys())
    nu      = datetime.now().strftime("%d/%m/%Y %H:%M")
    datum   = datetime.now().strftime("%Y-%m-%d")

    # Haal actuele koersen op in bulk
    try:
        raw = yf.download(tickers, period="30d", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Portfolio download fout: {e}")
        return "⚠️ Portfolio update mislukt — download fout."

    regels       = []
    verkoop_tips = []
    totaal_kost  = 0.0
    totaal_waarde = 0.0

    for ticker in tickers:
        pos = portfolio[ticker]
        try:
            if len(tickers) > 1:
                t_data = raw.xs(ticker, axis=1, level=1).dropna(how='all')
            else:
                t_data = raw.dropna(how='all')

            p_serie  = t_data['Close'].ffill()
            ma5      = p_serie.rolling(5).mean()
            ma10     = p_serie.rolling(10).mean()
            cp       = p_serie.iloc[-1]
            ma5_nu   = ma5.iloc[-1]
            ma10_nu  = ma10.iloc[-1]

            prijs_koop    = pos['prijs_koop']
            inzet         = pos['inzet']
            strategie     = pos['strategie']
            datum_koop    = pos['datum_koop']
            aantal_dagen  = pos.get('aantal_dagen', 0) + 1

            # Update houdperiode
            portfolio[ticker]['aantal_dagen'] = aantal_dagen

            # Bereken P&L
            waarde_nu  = inzet * (cp / prijs_koop)
            pnl        = waarde_nu - inzet
            pnl_pct    = ((cp / prijs_koop) - 1) * 100
            pijl       = "🟢" if pnl >= 0 else "🔴"

            totaal_kost   += inzet
            totaal_waarde += waarde_nu

            # Verkoopconditie per strategie
            verkoop = False
            reden   = ""
            if strategie == "MRAS":
                if cp > ma5_nu:
                    verkoop, reden = True, "boven MA5"
                elif cp > prijs_koop * MRA_SNEL_WINST:
                    verkoop, reden = True, f"+{int((MRA_SNEL_WINST-1)*100)}% bereikt"
            elif strategie == "MRAT":
                if aantal_dagen >= MRA_TRAAG_HOLD and cp > ma10_nu:
                    verkoop, reden = True, f"boven MA10 na {aantal_dagen}d"
                elif cp > prijs_koop * MRA_TRAAG_WINST:
                    verkoop, reden = True, f"+{int((MRA_TRAAG_WINST-1)*100)}% bereikt"
            else:  # Trend strats: verkoop bij crossover (simpele check)
                if cp < prijs_koop * 0.92:
                    verkoop, reden = True, "SL -8% geraakt"

            y_l = f"https://finance.yahoo.com/quote/{ticker}"

            regel = (
                f"• `{ticker}` [{strategie}] | Koop: {datum_koop} @ €{prijs_koop:.2f} | "
                f"Nu: €{cp:.2f} | {pijl} {pnl_pct:+.1f}% (€{pnl:+.0f}) | "
                f"Waarde: €{waarde_nu:.0f} | Dagen: {aantal_dagen}"
            )
            regels.append(regel)

            if verkoop:
                verkoop_tips.append(
                    f"⚠️ VERKOOP `{ticker}` [{strategie}] — {reden} | "
                    f"Limiet: €{cp*0.99:.2f} | P&L: {pnl_pct:+.1f}%"
                )

        except Exception as e:
            logger.error(f"Portfolio fout {ticker}: {e}")
            regels.append(f"• `{ticker}` — data fout")

    sla_portfolio_op(portfolio)

    totaal_pnl     = totaal_waarde - totaal_kost
    totaal_pnl_pct = ((totaal_waarde / totaal_kost) - 1) * 100 if totaal_kost > 0 else 0
    pijl_totaal    = "🟢" if totaal_pnl >= 0 else "🔴"

    rapport = [
        f"📂 *PORTFOLIO OVERZICHT* — {nu}",
        "----------------------------------",
        *regels,
        "",
        f"💰 *Totaal geïnvesteerd:* €{totaal_kost:.0f}",
        f"{pijl_totaal} *Totaal P&L:* {totaal_pnl_pct:+.1f}% (€{totaal_pnl:+.0f})",
        f"💼 *Totaal waarde:* €{totaal_waarde:.0f}",
    ]

    if verkoop_tips:
        rapport += ["", "🔔 *VERKOOPSIGNALEN:*", *verkoop_tips]

    return "\n".join(rapport)

# ---------------------------------------------------------------------------
# INDICATOREN
# FIX: ma20_line alias verwijderd — ma20 wordt lokaal berekend in MRA sectie
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
    # FIX: ma20_line alias verwijderd — was identiek aan ma20, overbodig

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

    # FIX: 12 return waarden ipv 13 (ma20_line verwijderd)
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
        tickers = sorted(list(set([t.strip().upper() for t in content.split(',') if t.strip()])))
    if not tickers: return

    logger.info(f"Start analyse voor {naam_sector} met {len(tickers)} tickers.")

    try:
        raw_df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Download fout: {e}")
        return

    inzet        = 2500.0
    BEURSTAKS    = 0.0035  # 0.35% Belgische TOB op aan- én verkoop
    MEERWAARDE   = 0.10    # 10% meerwaardebelasting op netto winst
    BROKER_PCT   = 0.0035  # 0.35% broker commissie
    BROKER_VAST  = 15.0    # €15 vaste broker kost per transactie

    def kosten_koop(n: float) -> float:
        """Totale aankoopkosten: vast + broker% + TOB%"""
        return BROKER_VAST + (n * BROKER_PCT) + (n * BEURSTAKS)

    def kosten_verk(n: float) -> float:
        """Totale verkoopkosten: vast + broker% + TOB%"""
        return BROKER_VAST + (n * BROKER_PCT) + (n * BEURSTAKS)

    def bereken_winst(inzet: float, instap: float, uitstap: float) -> float:
        """
        Netto winst per trade na alle kosten en belastingen:
        1. Bruto winst/verlies
        2. Verkoopkosten (broker + TOB)
        3. 10% meerwaardebelasting op netto winst (enkel bij winst)
        """
        eindwaarde = inzet * (uitstap / instap)
        bruto      = eindwaarde - inzet
        netto      = bruto - kosten_verk(eindwaarde)
        if netto > 0:
            netto -= netto * MEERWAARDE
        return netto

    kosten = kosten_koop(inzet)  # aankoopkosten bij instap

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

            if len(t_data) < 250: continue

            # Bereken basis indicatoren voor MRA (FIX: 12 return waarden)
            p, f, sl, e100, v_ma, rsi, atr, adx, vol, ibs, l_bb, ma5 = bereken_indicatoren_vectorized(t_data, 50, 200, True, False)

            # --- STRATEGIEËN T / S / HT / HS (ongewijzigd) ---
            for skey, s_p, t_p, utr, ihyp in STRATS:
                # FIX: 12 return waarden unpacking
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
                            pr -= kosten  # koop kosten
                            num_trades[skey] += 1
                    else:
                        hi = max(hi, cp)
                        if cp < (hi - 2 * ab.iloc[i]) or fb.iloc[i] < sb.iloc[i]:
                            pr += bereken_winst(inzet, ins, cp)
                            pos = False
                if pos:
                    pr += bereken_winst(inzet, ins, pb.iloc[-1])
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
            # MRA SNEL — origineel: BB + IBS, geen filters, uitstap MA5 of +12%
            # FIX backtest venster: iloc[-252:] + reset_index voor NaN-veiligheid
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
                    # FIX: enkel BB + IBS, geen extra filters
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
            # MRA TRAAG — BB + IBS instap, min 5 dagen, uitstap MA10 of +25%
            # FIX backtest venster: iloc[-252:] + reset_index voor NaN-veiligheid
            # FIX uitstap: MA10 ipv MA20 (MA20 werkt niet voor miners)
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
            # ACTUELE MRA SIGNALEN — enkel BB + IBS, geen filters
            # SL = -8% onder instap | Limiet = slot × 1.01
            # AUTO: positie toevoegen aan portfolio bij koopsignaal
            # ---------------------------------------------------------------
            cp_now   = p.iloc[-1]
            lbb_now  = l_bb.iloc[-1]
            ibs_now  = ibs.iloc[-1]
            atr_now  = atr.iloc[-1]
            rsi_now  = rsi.iloc[-1]
            portfolio = laad_portfolio()
            if not pd.isna(cp_now) and not pd.isna(lbb_now) and not pd.isna(ibs_now):
                if cp_now < lbb_now and ibs_now < MRA_IBS_MAX:
                    limiet = cp_now * 1.01
                    sl     = cp_now * 0.92
                    y_l    = f"[Grafiek](https://finance.yahoo.com/quote/{ticker})"
                    sig["MRAS"].append(
                        f"• `{ticker}`: 🛡️ *Munger Snel* | Slot: €{cp_now:.2f} | "
                        f"📋 Limiet: €{limiet:.2f} | 🛑 SL: €{sl:.2f} | "
                        f"🎯 TP: €{cp_now*MRA_SNEL_WINST:.2f} | "
                        f"⚡ ATR: {atr_now:.2f} | 📊 RSI: {rsi_now:.1f} | {y_l}"
                    )
                    sig["MRAT"].append(
                        f"• `{ticker}`: 🐢 *Munger Traag* | Slot: €{cp_now:.2f} | "
                        f"📋 Limiet: €{limiet:.2f} | 🛑 SL: €{sl:.2f} | "
                        f"🎯 TP: €{cp_now*MRA_TRAAG_WINST:.2f} | "
                        f"⚡ ATR: {atr_now:.2f} | 📊 RSI: {rsi_now:.1f} | {y_l}"
                    )
                    # AUTO: toevoegen aan portfolio als nog niet aanwezig
                    if ticker not in portfolio:
                        portfolio[ticker] = {
                            "strategie":    "MRAT",  # standaard Traag want beste resultaat
                            "datum_koop":   datetime.now().strftime("%Y-%m-%d"),
                            "prijs_koop":   round(float(cp_now), 4),
                            "inzet":        inzet,
                            "aantal_dagen": 0
                        }
                        sla_portfolio_op(portfolio)
                        logger.info(f"Auto portfolio: {ticker} toegevoegd @ {cp_now:.2f}")

            # AUTO: trend koopsignalen ook toevoegen aan portfolio
            for skey, s_p, t_p, utr, ihyp in STRATS:
                pi2, fi2, sli2, ei2, vmai2, rsii2, atri2, dxi2, voli2, _, _, _ = bereken_indicatoren_vectorized(t_data, s_p, t_p, utr, ihyp)
                cp2 = pi2.iloc[-1]
                if (fi2.iloc[-1] > sli2.iloc[-1] and fi2.iloc[-2] <= sli2.iloc[-2]
                        and dxi2.iloc[-1] > 15
                        and voli2.iloc[-1] > (vmai2.iloc[-1] * 0.6)
                        and ((not utr) or cp2 > ei2.iloc[-1])):
                    portfolio = laad_portfolio()
                    if ticker not in portfolio:
                        portfolio[ticker] = {
                            "strategie":    skey,
                            "datum_koop":   datetime.now().strftime("%Y-%m-%d"),
                            "prijs_koop":   round(float(cp2), 4),
                            "inzet":        inzet,
                            "aantal_dagen": 0
                        }
                        sla_portfolio_op(portfolio)
                        logger.info(f"Auto portfolio: {ticker} toegevoegd @ {cp2:.2f} ({skey})")

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
        f"_Inzet: €{inzet:.0f} | Broker: €{BROKER_VAST:.0f}+{BROKER_PCT*100:.2f}% | TOB: {BEURSTAKS*100:.2f}% | Meerwaarde: {MEERWAARDE*100:.0f}%_",
    ]

    stuur_telegram("\n".join(deel1))
    time.sleep(1)
    stuur_telegram("\n".join(deel2))

# ---------------------------------------------------------------------------
# MAIN — scheduler: dagelijks om 22:30 (na sluit NYSE + Euronext)
# ---------------------------------------------------------------------------
def run_alle_sectoren():
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

    logger.info("Start dagelijkse analyse run.")
    verzamel = []

    # 1. Portfolio update — eerst
    portfolio_rapport = update_portfolio_en_rapport()
    stuur_telegram(portfolio_rapport)
    verzamel.append(portfolio_rapport)
    time.sleep(1)

    # 2. Alle sectoren analyseren
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        verzamel.append(f"\n{'='*30}\n")
        time.sleep(2)

    # 3. E-mail met portfolio + samenvatting
    datum = datetime.now().strftime("%d-%m-%Y")
    stuur_mail(
        onderwerp=f"Trading Rapport {datum}",
        inhoud="\n".join(verzamel)
    )

    logger.info("Dagelijkse analyse run voltooid.")

# ---------------------------------------------------------------------------
# MAIN — 1x uitvoeren en stoppen (scheduler via GitHub Actions cron)
# ---------------------------------------------------------------------------
def main():
    logger.info(f"Bot gestart om {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    run_alle_sectoren()
    logger.info("Bot klaar.")

if __name__ == "__main__":
    main()
