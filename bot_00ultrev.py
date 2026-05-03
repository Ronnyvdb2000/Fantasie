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
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')

PORTFOLIO_FILE = "portfolio.json"

# MRA PARAMETERS
MRA_BB_STD      = 2.2
MRA_IBS_MAX     = 0.30
MRA_SNEL_WINST  = 1.12
MRA_SNEL_MA     = 5
MRA_TRAAG_WINST = 1.25
MRA_TRAAG_MA    = 10
MRA_TRAAG_HOLD  = 5

# FISCALE PARAMETERS
INZET          = 2500.0
BEURSTAKS_PCT  = 0.0035  # 0.35% TOB
TOB_MAX        = 1600.0
MEERWAARDE     = 0.10    # 10% netto winst belasting
BROKER_PCT     = 0.0035  # 0.35%
BROKER_VAST    = 15.0    # €15 vaste kost

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def bereken_kosten_totaal(bedrag: float) -> float:
    tob = min(bedrag * BEURSTAKS_PCT, TOB_MAX)
    broker = BROKER_VAST + (bedrag * BROKER_PCT)
    return tob + broker

def stuur_telegram(bericht: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram fout: {e}")
        return False

def stuur_mail(onderwerp: str, inhoud: str) -> bool:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER: return False
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = EMAIL_USER, EMAIL_RECEIVER, onderwerp
        schoon = inhoud.replace('*', '').replace('`', '').replace('_', '').replace('•', '-')
        msg.attach(MIMEText(schoon, 'plain', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"E-mail fout: {e}")
        return False

# ---------------------------------------------------------------------------
# PORTFOLIO BEHEER
# ---------------------------------------------------------------------------
def laad_portfolio() -> dict:
    if not os.path.exists(PORTFOLIO_FILE): return {}
    try:
        with open(PORTFOLIO_FILE, 'r') as f: return json.load(f)
    except: return {}

def sla_portfolio_op(portfolio: dict) -> None:
    with open(PORTFOLIO_FILE, 'w') as f: json.dump(portfolio, f, indent=2)

def update_portfolio_en_rapport() -> str:
    portfolio = laad_portfolio()
    if not portfolio: return "📂 *PORTFOLIO* — Geen open posities."

    tickers = sorted(portfolio.keys())
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    try:
        raw = yf.download(tickers, period="65d", progress=False, auto_adjust=True)
    except Exception as e:
        return f"⚠️ Portfolio update mislukt: {e}"

    regels, verkoop_tips = [], []
    t_kost, t_waarde = 0.0, 0.0

    for ticker in tickers:
        pos = portfolio[ticker]
        try:
            df = raw.xs(ticker, axis=1, level=1).dropna(how='all') if len(tickers) > 1 else raw.dropna(how='all')
            cp = df['Close'].iloc[-1]
            
            # ATR Berekening voor dynamische SL
            h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
            tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
            atr_nu = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
            
            ma5, ma10 = df['Close'].rolling(5).mean().iloc[-1], df['Close'].rolling(10).mean().iloc[-1]
            
            # Update hoogste koers en dagen
            hi_since = max(pos.get('hi_since', cp), cp)
            pos['hi_since'] = hi_since
            pos['aantal_dagen'] = pos.get('aantal_dagen', 0) + 1
            sl_dyn = hi_since - (2 * atr_nu)

            waarde_nu = pos['inzet'] * (cp / pos['prijs_koop'])
            pnl_pct = ((cp / pos['prijs_koop']) - 1) * 100
            t_kost += pos['inzet']
            t_waarde += waarde_nu

            # Verkoopcheck
            verkoop, reden = False, ""
            if pos['strategie'] == "MRAS":
                if cp > ma5 or cp > pos['prijs_koop'] * MRA_SNEL_WINST: verkoop, reden = True, "Target/MA5"
            elif pos['strategie'] == "MRAT":
                if (pos['aantal_dagen'] >= MRA_TRAAG_HOLD and cp > ma10) or cp > pos['prijs_koop'] * MRA_TRAAG_WINST: verkoop, reden = True, "Target/MA10"
            else: # Trend
                if cp < sl_dyn: verkoop, reden = True, f"Trailing SL (€{sl_dyn:.2f})"

            pijl = "🟢" if pnl_pct >= 0 else "🔴"
            regels.append(f"• `{ticker}` [{pos['strategie']}] | Nu: €{cp:.2f} | {pijl} {pnl_pct:+.1f}% | 🛡️ SL: €{sl_dyn:.2f}")
            if verkoop: verkoop_tips.append(f"⚠️ *VERKOOP* `{ticker}`: {reden}")

        except Exception as e:
            logger.error(f"Fout bij {ticker}: {e}")

    sla_portfolio_op(portfolio)
    t_pnl = t_waarde - t_kost
    rapport = [f"📂 *PORTFOLIO OVERZICHT* — {nu}", "---", *regels, "", f"💰 *Totaal:* €{t_waarde:.0f} ({t_pnl:+.0f})", "", *verkoop_tips]
    return "\n".join(rapport)

# ---------------------------------------------------------------------------
# CORE ENGINE & INDICATOREN
# ---------------------------------------------------------------------------
def bereken_indicatoren(df: pd.DataFrame, s: int, t: int, is_hyper: bool) -> tuple:
    p, h, l, v = df['Close'].ffill(), df['High'].ffill(), df['Low'].ffill(), df['Volume'].ffill()
    f_line = p.rolling(s).mean() if s >= 20 else p.ewm(span=s).mean()
    s_line = p.rolling(t).mean() if t >= 50 else p.ewm(span=t).mean()
    ema100, v_ma = p.ewm(span=100).mean(), v.rolling(20).mean()
    
    # RSI / CRSI
    delta = p.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR & ADX
    tr = pd.concat([h-l, (h-p.shift()).abs(), (l-p.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14).mean()
    up, down = h.diff().clip(lower=0), (-l.diff()).clip(lower=0)
    p_di = 100 * (up.where((up > down) & (up > 0), 0).ewm(alpha=1/14).mean() / (atr + 1e-10))
    m_di = 100 * (down.where((down > up) & (down > 0), 0).ewm(alpha=1/14).mean() / (atr + 1e-10))
    adx = (100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)).ewm(alpha=1/14).mean()

    # MRA specifiek
    ma20 = p.rolling(20).mean()
    l_bb = ma20 - (MRA_BB_STD * p.rolling(20).std())
    ibs = (p - l) / (h - l + 1e-10)
    
    return p, f_line, s_line, ema100, v_ma, rsi, atr, adx, vol, ibs, l_bb

def voer_lijst_uit(bestandsnaam: str, label: str, naam_sector: str) -> None:
    if not os.path.exists(bestandsnaam): return
    with open(bestandsnaam, 'r') as f:
        tickers = sorted(list(set([t.strip().upper() for t in f.read().replace('\n', ',').split(',') if t.strip()])))
    
    try: raw_df = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
    except: return

    res = {"T": 0.0, "S": 0.0, "HT": 0.0, "HS": 0.0, "MRAS": 0.0, "MRAT": 0.0}
    sig = {k: [] for k in res.keys()}
    portfolio = laad_portfolio()

    for ticker in tickers:
        try:
            df = raw_df.xs(ticker, axis=1, level=1).dropna() if len(tickers) > 1 else raw_df.dropna()
            if len(df) < 150: continue
            
            p, f, sl, e100, vma, rsi, atr, adx, vol, ibs, lbb = bereken_indicatoren(df, 50, 200, False)
            
            # Check MRA Signalen (Actueel)
            cp, lbb_n, ibs_n = p.iloc[-1], lbb.iloc[-1], ibs.iloc[-1]
            if cp < lbb_n and ibs_n < MRA_IBS_MAX:
                msg = f"• `{ticker}`: 🛡️ *MRA* | Slot: €{cp:.2f} | RSI: {rsi.iloc[-1]:.1f}"
                sig["MRAS"].append(msg)
                if ticker not in portfolio:
                    portfolio[ticker] = {"strategie": "MRAT", "prijs_koop": round(float(cp), 4), "inzet": INZET, "aantal_dagen": 0, "hi_since": float(cp)}

            # Trend Signalen (Snel 20/50)
            p2, f2, sl2, _, vma2, rsi2, _, adx2, _, _, _ = bereken_indicatoren(df, 20, 50, False)
            if f2.iloc[-1] > sl2.iloc[-1] and f2.iloc[-2] <= sl2.iloc[-2] and adx2.iloc[-1] > 15:
                sig["S"].append(f"• `{ticker}`: 🟢 *Trend Koop* | €{cp:.2f}")

        except: continue

    sla_portfolio_op(portfolio)
    stuur_telegram(f"📊 *Sector {label}: {naam_sector}*\n" + "\n".join(sig["MRAS"] + sig["S"]))

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    logger.info("Bot gestart.")
    sectoren = {"01": "Hoogland", "02": "Macro", "03": "Brink", "04": "Benelux", "05": "Parijs"}
    
    # 1. Update Portfolio
    port_rapport = update_portfolio_en_rapport()
    stuur_telegram(port_rapport)
    
    # 2. Analyse Sectoren
    for nr, naam in sectoren.items():
        voer_lijst_uit(f"tickers_{nr}.txt", nr, naam)
        time.sleep(2)
    
    logger.info("Bot klaar.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
