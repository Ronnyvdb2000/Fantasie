#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_parameter_ic.py
========================
Bereken voor ELKE parameter de voorspellende waarde (Spearman IC) voor ELKE horizon
(5d, 10d, 20d, 30d, 60d), inclusief:
  - Newey-West HAC t-statistieken (gecorrigeerd voor overlappende returns)
  - Benjamini-Hochberg (BH-FDR) correctie per horizon voor valse ontdekkingen
  - Automatische Telegram- en E-mailnotificaties met de resultaten

Env:
  SUPABASE_DB_URL, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
  EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER
Output:
  results/parameter_ic_matrix.csv   (parameters x horizons full stats)
  Telegram & Email samenvatting
"""

import os
import sys
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr
import statsmodels.api as sm
import requests
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Notificatie credentials
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

HORIZONS = ["5d", "10d", "20d", "30d", "60d"]
HORIZON_DAYS = {"5d": 5, "10d": 10, "20d": 20, "30d": 30, "60d": 60}
TARGET_KOLOM = {h: f"fwd_ret_{h}" for h in HORIZONS}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_TICKERS_PER_DATUM = 20
MIN_DATUMS = 10

NIET_FACTOR = {
    "ticker", "datum", "beurs", "strategie", "bijgewerkt_op", "id",
    "entry_koers", "high52w", "ma50", "ma200", "market_cap",
    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d", "fwd_ret_60d",
    "fwd_close_5d", "fwd_close_10d", "fwd_close_20d",
    "fwd_close_30d", "fwd_close_60d",
}


def stuur_telegram(bericht):
    if not TOKEN or not CHAT_ID:
        logger.info("Telegram-secrets ontbreken, overslaan.")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": bericht, "parse_mode": "HTML"},
            timeout=30,
        )
        if not resp.ok:
            logger.error("Telegram-fout: %s %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Telegram-fout (exception): %s", exc)


def stuur_email(html_body: str, onderwerp: str):
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        logger.info("Email-secrets ontbreken, overslaan.")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = onderwerp
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
    except Exception as exc:
        logger.error("Email-fout: %s", exc)


def get_data(conn) -> pd.DataFrame:
    query = """
    WITH tech AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, atr14, atr14_pct, rsi14, ibs,
            ma50, ma200, pct_from_ma50, pct_from_ma200,
            vol_ratio_20d, high52w, pct_from_high52w,
            macd, macd_signaal, macd_hist, bb_breedte, bb_percent_b,
            stoch_k, stoch_d, adx14, rel_sterkte_20d, hv20, hv60,
            dagen_sinds_low52w, vol_ratio_50d
        FROM generieke_technicals
        ORDER BY ticker, datum
    ),
    fund AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum, market_cap, trailing_pe, price_to_book,
            dividend_yield, current_ratio, revenue_growth_pct, fcf_yield,
            net_debt_ebitda, payout_pct, analisten_count,
            piotroski_score, eps_growth_pct, eps_cagr_pct, peg_ratio
        FROM generieke_fundamentals
        ORDER BY ticker, datum DESC
    ),
    fr AS (
        SELECT DISTINCT ON (ticker, datum)
            ticker, datum,
            fwd_ret_5d, fwd_ret_10d, fwd_ret_20d, fwd_ret_30d, fwd_ret_60d
        FROM forward_returns
        ORDER BY ticker, datum
    )
    SELECT
        t.ticker, t.datum,
        t.atr14, t.atr14_pct, t.rsi14, t.ibs,
        t.ma50, t.ma200, t.pct_from_ma50, t.pct_from_ma200,
        t.vol_ratio_20d, t.high52w, t.pct_from_high52w,
        t.macd, t.macd_signaal, t.macd_hist, t.bb_breedte, t.bb_percent_b,
        t.stoch_k, t.stoch_d, t.adx14, t.rel_sterkte_20d, t.hv20, t.hv60,
        t.dagen_sinds_low52w, t.vol_ratio_50d,
        f.market_cap, f.trailing_pe, f.price_to_book, f.dividend_yield,
        f.current_ratio, f.revenue_growth_pct, f.fcf_yield,
        f.net_debt_ebitda, f.payout_pct, f.analisten_count,
        f.piotroski_score, f.eps_growth_pct, f.eps_cagr_pct, f.peg_ratio,
        fr.fwd_ret_5d, fr.fwd_ret_10d, fr.fwd_ret_20d,
        fr.fwd_ret_30d, fr.fwd_ret_60d
    FROM tech t
    LEFT JOIN fund f ON f.ticker = t.ticker AND f.datum = t.datum
    LEFT JOIN fr ON fr.ticker = t.ticker AND fr.datum = t.datum;
    """
    return pd.read_sql(query, conn)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p_values = np.asarray(p_values)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    significant_sorted = sorted_p <= thresholds
    
    if np.any(significant_sorted):
        max_sig_idx = np.max(np.where(significant_sorted)[0])
        significant_sorted[:max_sig_idx + 1] = True
    else:
        significant_sorted[:] = False

    reject = np.empty(n, dtype=bool)
    reject[sorted_idx] = significant_sorted
    return reject


def bereken_ic_hac(df: pd.DataFrame, parameter: str, target: str, max_lag: int) -> dict:
    sub = df[["datum", parameter, target]].dropna()
    if len(sub) < MIN_TICKERS_PER_DATUM:
        return None

    date_ics = []
    for datum, groep in sub.groupby("datum"):
        if len(groep) < MIN_TICKERS_PER_DATUM:
            continue
        rho, _ = spearmanr(groep[parameter], groep[target])
        if not np.isnan(rho):
            date_ics.append({"datum": datum, "ic": rho})

    if len(date_ics) < MIN_DATUMS:
        return None

    ic_df = pd.DataFrame(date_ics).sort_values("datum")
    ics = ic_df["ic"].values

    ic_gem = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1))
    ic_ir = ic_gem / ic_std if ic_std > 1e-9 else 0.0
    hit = float(np.mean(ics > 0)) if ic_gem > 0 else float(np.mean(ics < 0))

    X = np.ones(len(ics))
    try:
        model = sm.OLS(ics, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        t_stat = float(model.tvalues[0])
        p_val = float(model.pvalues[0])
    except Exception:
        t_stat = 0.0
        p_val = 1.0

    return {
        "ic": round(ic_gem, 4),
        "ic_ir": round(ic_ir, 3),
        "t_stat": round(t_stat, 3),
        "p_value": p_val,
        "hit": round(hit, 3),
        "n_datums": len(ics),
        "n_obs": len(sub),
    }


def bouw_html_rapport(matrix: pd.DataFrame, stats_summary: dict) -> str:
    """Bouwt een overzichtelijk HTML e-mailrapport met resultaten."""
    return f"""
    <html><body style="font-family:Arial,sans-serif;font-size:13px;">
    <h2>📊 Parameter IC & Newey-West Analyse</h2>
    <p><b>Samenvatting significante parameters na Benjamini-Hochberg FDR-correctie (alpha=0.05):</b></p>
    <ul>
        <li>5d: {stats_summary.get('5d', 0)} significant</li>
        <li>10d: {stats_summary.get('10d', 0)} significant</li>
        <li>20d: {stats_summary.get('20d', 0)} significant</li>
        <li>30d: {stats_summary.get('30d', 0)} significant</li>
        <li>60d: {stats_summary.get('60d', 0)} significant</li>
    </ul>
    <p>Volledige datamatrix is opgeslagen als CSV-artifact in de run.</p>
    </body></html>
    """


def main():
    try:
        logger.info("Data ophalen uit Supabase...")
        with psycopg2.connect(os.environ["SUPABASE_DB_URL"]) as conn:
            df = get_data(conn)
        logger.info(f"{len(df):,} rijen | {df['datum'].nunique()} datums | {df['ticker'].nunique()} tickers")

        params = [
            c for c in df.columns
            if c not in NIET_FACTOR and pd.api.types.is_numeric_dtype(df[c])
        ]

        results = {h: {} for h in HORIZONS}
        for p in params:
            for h in HORIZONS:
                res = bereken_ic_hac(df, p, TARGET_KOLOM[h], max_lag=HORIZON_DAYS[h])
                if res:
                    results[h][p] = res

        bh_significance = {h: {} for h in HORIZONS}
        stats_summary = {}
        for h in HORIZONS:
            p_names = [p for p in params if p in results[h]]
            if not p_names:
                stats_summary[h] = 0
                continue
            p_vals = [results[h][p]["p_value"] for p in p_names]
            rejects = benjamini_hochberg(p_vals, alpha=0.05)
            for p, rej in zip(p_names, rejects):
                bh_significance[h][p] = bool(rej)
            stats_summary[h] = int(sum(rejects))

        rijen = []
        for p in params:
            rij = {"parameter": p}
            for h in HORIZONS:
                r = results[h].get(p)
                rij[f"ic_{h}"] = r["ic"] if r else np.nan
                rij[f"ir_{h}"] = r["ic_ir"] if r else np.nan
                rij[f"tstat_{h}"] = r["t_stat"] if r else np.nan
                rij[f"pval_{h}"] = r["p_value"] if r else np.nan
                rij[f"sig_bh_{h}"] = bh_significance[h].get(p, False) if r else False
            rijen.append(rij)

        matrix = pd.DataFrame(rijen)
        out = os.path.join(RESULTS_DIR, "parameter_ic_matrix.csv")
        matrix.to_csv(out, index=False)
        logger.info(f"Opgeslagen: {out}")

        # Telegram samenvatting opbouwen
        tg_bericht = (
            f"📊 <b>Parameter IC Analyse Voltooid</b>\n\n"
            f"<b>Aantal significante parameters (BH-FDR, alpha=0.05):</b>\n"
            f"• 5d: {stats_summary.get('5d', 0)}\n"
            f"• 10d: {stats_summary.get('10d', 0)}\n"
            f"• 20d: {stats_summary.get('20d', 0)}\n"
            f"• 30d: {stats_summary.get('30d', 0)}\n"
            f"• 60d: {stats_summary.get('60d', 0)}\n\n"
            f"<i>Zie e-mail of CSV-artifact voor details.</i>"
        )

        stuur_telegram(tg_bericht)
        stuur_email(
            bouw_html_rapport(matrix, stats_summary),
            "Parameter IC & Newey-West Analyse Rapport",
        )
        logger.info("Notificaties succesvol verzonden.")

    except Exception as e:
        logger.error("Kritieke fout in main(): %s", e, exc_info=True)
        stuur_telegram(f"❌ <b>Parameter IC Analyse gecrasht:</b> {str(e)}")
        raise e


if __name__ == "__main__":
    main()
