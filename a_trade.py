#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_trade.py
=============
Leest de Supabase `selecties`-tabel uit (gevuld door de andere Fantasie-bots),
pikt daar het "beste" signaal uit en verrijkt dat met:

  1. XGBoost-scores uit `xgboost_scores` (beste rang per ticker)
  2. Kasstroom-onderwaardering signaal (bot_01kasstr in selecties)

Ranking-logica (in volgorde):
  1. Cross-strategie overlap (hoeveel strategieën kozen dezelfde ticker)
  2. XGBoost-rang (lagere rang = hoger in de ranking)
  3. Kasstr-vlag (bonus voor tickers die bot_01kasstr selecteerde)
  4. Gemiddelde score binnen de overlap
  5. Meest recente datum

Env vars (zelfde als de rest van de Fantasie-repo):
  SUPABASE_DB_URL, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
  EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER
  LOOKBACK_DAYS       (default 3)
  BESCHIKBAAR_KAPITAAL (default 2500)
  TRANSACTIE_BEDRAG    (default 2500)
  AANTAL_PICKS         (default 5)
  XGB_HORIZON          (default 10d)
  XGB_MODEL_VERSIE     (default leeg = nieuwste)
"""

import os
import sys
import html
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict

import psycopg2
import psycopg2.extras
import requests


# --------------------------------------------------------------------------
# Configuratie
# --------------------------------------------------------------------------
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "3"))

BESCHIKBAAR_KAPITAAL = float(os.environ.get("BESCHIKBAAR_KAPITAAL", "2500"))
TRANSACTIE_BEDRAG = float(os.environ.get("TRANSACTIE_BEDRAG", "2500"))
AANTAL_PICKS = max(1, int(os.environ.get("AANTAL_PICKS", "5")))
TOP_N = AANTAL_PICKS

# XGBoost-integratie
XGB_HORIZON = os.environ.get("XGB_HORIZON", "10d")
XGB_MODEL_VERSIE = os.environ.get("XGB_MODEL_VERSIE", "")  # leeg = nieuwste

# Fiscale/kostenparameters
VASTE_KOST = 15.0
VARIABELE_KOST_PCT = 0.35
TOB_PCT = 0.35

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")


# --------------------------------------------------------------------------
# Data ophalen
# --------------------------------------------------------------------------
def haal_selecties_op(lookback_days: int):
    """Haalt alle selecties op van de laatste `lookback_days` dagen."""
    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    query = """
        SELECT ticker, beurs, strategie, datum, koers, score, grafiek
        FROM selecties
        WHERE datum >= %s
        ORDER BY datum DESC;
    """

    with psycopg2.connect(SUPABASE_DB_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (since,))
            rows = cur.fetchall()

    return rows


def haal_xgboost_scores_op(lookback_days: int):
    """
    Haalt de beste XGBoost-score per ticker op binnen de lookback-periode.

    Retourneert een dict {ticker: {"score": float, "beste_rang": int, "n_datums": int}}
    Leeg als de query faalt of geen rijen oplevert.
    """
    if not SUPABASE_DB_URL:
        return {}

    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Als geen versie opgegeven: pak de meest recente versie in de tabel
    versie_filter = ""
    params = [since, XGB_HORIZON]
    if XGB_MODEL_VERSIE:
        versie_filter = "AND model_versie = %s"
        params.append(XGB_MODEL_VERSIE)

    query = f"""
        SELECT
            ticker,
            MAX(score)              AS xgb_score,
            MIN(rang)               AS beste_rang,
            COUNT(DISTINCT datum)   AS n_datums
        FROM xgboost_scores
        WHERE datum >= %s
          AND horizon = %s
          {versie_filter}
        GROUP BY ticker;
    """

    try:
        with psycopg2.connect(SUPABASE_DB_URL) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return {
            r["ticker"]: {
                "score": float(r["xgb_score"]) if r["xgb_score"] is not None else 0.0,
                "beste_rang": int(r["beste_rang"]) if r["beste_rang"] is not None else 9999,
                "n_datums": int(r["n_datums"]) if r["n_datums"] is not None else 0,
            }
            for r in rows
        }
    except Exception as e:
        print(f"[WARN] XGBoost-scores niet op te halen: {e}", file=sys.stderr)
        return {}


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def bouw_ranking(rows, xgb_scores=None):
    """
    Groepeert per (ticker, beurs), berekent overlap + gemiddelde score,
    en verrijkt met XGBoost-info + kasstr-vlag.
    """
    xgb_scores = xgb_scores or {}

    groepen = defaultdict(lambda: {
        "strategieen": set(),
        "scores": [],
        "koers": None,
        "laatste_datum": None,
        "grafiek": None,
        "details": [],
    })

    for row in rows:
        key = (row["ticker"], row["beurs"])
        g = groepen[key]
        g["strategieen"].add(row["strategie"])

        score = row.get("score")
        if score is not None:
            try:
                g["scores"].append(float(score))
            except (TypeError, ValueError):
                pass

        if row.get("grafiek"):
            g["grafiek"] = row["grafiek"]

        if row["koers"] is not None:
            g["koers"] = row["koers"]

        if g["laatste_datum"] is None or row["datum"] > g["laatste_datum"]:
            g["laatste_datum"] = row["datum"]

        g["details"].append((row["strategie"], row["datum"], score))

    ranking = []
    for (ticker, beurs), g in groepen.items():
        overlap = len(g["strategieen"])
        avg_score = sum(g["scores"]) / len(g["scores"]) if g["scores"] else 0.0

        xgb_info = xgb_scores.get(ticker, {})
        xgb_rang = xgb_info.get("beste_rang", 9999)
        xgb_score = xgb_info.get("score", 0.0)
        xgb_datums = xgb_info.get("n_datums", 0)

        is_kasstr = "bot_01kasstr" in g["strategieen"]

        ranking.append({
            "ticker": ticker,
            "beurs": beurs,
            "overlap": overlap,
            "strategieen": sorted(g["strategieen"]),
            "avg_score": avg_score,
            "koers": g["koers"],
            "grafiek": g["grafiek"],
            "laatste_datum": g["laatste_datum"],
            "xgb_rang": xgb_rang,
            "xgb_score": xgb_score,
            "xgb_datums": xgb_datums,
            "heeft_xgb": xgb_rang < 9999,
            "is_kasstr": is_kasstr,
        })

    # Alle criteria op "hoog = beter" brengen, dan reverse=True sorteren
    #  - overlap: hoog = beter
    #  - xgb_prioriteit: -xgb_rang (rang 1 -> -1; niet aanwezig -> -9999)
    #  - kasstr: 1 = beter
    #  - avg_score: hoog = beter
    #  - laatste_datum: recenter = beter
    ranking.sort(
        key=lambda r: (
            r["overlap"],
            -r["xgb_rang"] if r["heeft_xgb"] else -9999,
            int(r["is_kasstr"]),
            r["avg_score"],
            r["laatste_datum"],
        ),
        reverse=True,
    )
    return ranking


# --------------------------------------------------------------------------
# Kosteninschatting
# --------------------------------------------------------------------------
def bereken_kosten(bedrag: float):
    variabele_kost = bedrag * VARIABELE_KOST_PCT / 100
    tob = bedrag * TOB_PCT / 100
    totaal = VASTE_KOST + variabele_kost + tob
    return totaal, (totaal / bedrag * 100)


# --------------------------------------------------------------------------
# Berichten opbouwen
# --------------------------------------------------------------------------
def _esc(s):
    return html.escape(str(s))


def _xgb_markering(r, top_drempel=10):
    """Geeft een korte string met XGBoost-status voor een rij."""
    if not r["heeft_xgb"]:
        return ""
    if r["xgb_rang"] <= top_drempel:
        return f" 🎯 XGBoost #{r['xgb_rang']}"
    return f" 🎯 XGBoost rang {r['xgb_rang']}"


def maak_telegram_bericht(ranking, lookback_days, totaal_strategieen):
    vandaag = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not ranking:
        return (
            f"⚠️ <b>Beste Signaal Bot — {vandaag}</b>\n"
            f"Geen selecties gevonden in de laatste {lookback_days} dagen."
        )

    top = ranking[:TOP_N]
    kost, kost_pct = bereken_kosten(TRANSACTIE_BEDRAG)
    medailles = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    lijnen = [
        f"🚨🚨🚨 <b>BESTE SIGNAAL — {vandaag}</b> 🚨🚨🚨",
        f"<i>Analyse van laatste {lookback_days} dagen, "
        f"{sum(r['overlap'] for r in ranking)} selecties totaal</i>",
        f"<i>Aantal picks: {TOP_N} (van €{TRANSACTIE_BEDRAG:,.0f} elk)</i>",
        f"<i>Verrijkt met XGBoost ({XGB_HORIZON}) + kasstr-signaal</i>",
    ]

    for i, r in enumerate(top):
        medaille = medailles[i] if i < len(medailles) else "▫️"
        strategieen_str = _esc(", ".join(r["strategieen"]))
        xgb_str = _xgb_markering(r)
        kasstr_str = " 💰 Kasstr" if r["is_kasstr"] else ""

        lijnen.append("")
        lijnen.append(f"{medaille} <b>{_esc(r['ticker'])}</b> ({_esc(r['beurs'])})"
                      f"{_esc(xgb_str)}{_esc(kasstr_str)}")
        lijnen.append(f"✅ Overlap: <b>{r['overlap']}/{totaal_strategieen}</b> — "
                      f"{strategieen_str}")
        if r["heeft_xgb"]:
            lijnen.append(
                f"🎯 XGBoost: rang <b>#{r['xgb_rang']}</b> "
                f"(score {r['xgb_score']:.3f}, {r['xgb_datums']} datums)"
            )
        if r["avg_score"]:
            lijnen.append(f"📊 Gem. score: <b>{r['avg_score']:.2f}</b>")
        if r["koers"] is not None:
            lijnen.append(f"💶 Laatste koers: {_esc(r['koers'])}")
        lijnen.append(f"💸 Kost bij €{TRANSACTIE_BEDRAG:,.0f}: ~€{kost:.2f} ({kost_pct:.2f}%)")
        if r["grafiek"]:
            lijnen.append(f'📈 <a href="{_esc(r["grafiek"])}">Grafiek</a>')

    return "\n".join(lijnen)


def maak_email_html(ranking, lookback_days, totaal_strategieen):
    if not ranking:
        return (
            f"<h2>⚠️ Beste Signaal Bot</h2>"
            f"<p>Geen selecties gevonden in de laatste {lookback_days} dagen.</p>"
        )

    top = ranking[:TOP_N]
    beste = top[0]
    kost, kost_pct = bereken_kosten(TRANSACTIE_BEDRAG)

    strategieen_html = ", ".join(beste["strategieen"])
    grafiek_html = (
        f'<p><a href="{beste["grafiek"]}">📈 Bekijk grafiek</a></p>'
        if beste["grafiek"] else ""
    )

    xgb_html = ""
    if beste["heeft_xgb"]:
        xgb_html = (
            f'<p style="font-size:16px;">🎯 XGBoost: rang '
            f'<b>#{beste["xgb_rang"]}</b> '
            f'(score {beste["xgb_score"]:.3f}, {beste["xgb_datums"]} datums)</p>'
        )

    kasstr_html = (
        '<p style="font-size:16px;">💰 <b>Kasstr-signaal aanwezig</b> '
        '(FCF-onderwaardering + kwaliteit)</p>'
        if beste["is_kasstr"] else ""
    )

    # Overige rijen
    overige_html = ""
    if len(top) > 1:
        rijen = "".join(
            f"<tr>"
            f"<td>{r['ticker']}</td>"
            f"<td>{r['beurs']}</td>"
            f"<td>{r['overlap']}/{totaal_strategieen}</td>"
            f"<td>#{r['xgb_rang'] if r['heeft_xgb'] else '-'}</td>"
            f"<td>{'✓' if r['is_kasstr'] else ''}</td>"
            f"<td>{r['avg_score']:.2f}</td>"
            f"</tr>"
            for r in top[1:]
        )
        overige_html = f"""
        <h3>Overige kanshebbers (elk €{TRANSACTIE_BEDRAG:,.0f})</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
          <tr style="background:#eee;">
            <th>Ticker</th><th>Beurs</th><th>Overlap</th>
            <th>XGB rang</th><th>Kasstr</th><th>Score</th>
          </tr>
          {rijen}
        </table>
        """

    return f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <div style="background:#fff3cd; border:3px solid #ff9800; border-radius:10px;
                    padding:20px; margin-bottom:20px;">
          <h1 style="color:#e65100; margin-top:0;">🚨 BESTE SIGNAAL 🚨</h1>
          <p style="color:#555;">Analyse van de laatste {lookback_days} dagen
             ({sum(r['overlap'] for r in ranking)} selecties totaal)</p>
          <p style="color:#555;">Aantal picks: <b>{TOP_N}</b> (van €{TRANSACTIE_BEDRAG:,.0f} elk)</p>
          <p style="color:#555;">Verrijkt met XGBoost ({XGB_HORIZON}) + kasstr-signaal</p>
          <h2 style="font-size:28px; margin-bottom:5px;">🥇 {beste['ticker']} ({beste['beurs']})</h2>
          <p style="font-size:18px;">✅ Overlap: <b>{beste['overlap']}/{totaal_strategieen} strategieën</b>
             — {strategieen_html}</p>
          {xgb_html}
          {kasstr_html}
          <p style="font-size:16px;">📊 Gemiddelde score: <b>{beste['avg_score']:.2f}</b></p>
          <p style="font-size:16px;">💶 Laatste koers: <b>{beste['koers']}</b></p>
          <p style="font-size:16px;">💸 Kost bij €{TRANSACTIE_BEDRAG:,.0f}: ~€{kost:.2f} ({kost_pct:.2f}%)</p>
          {grafiek_html}
        </div>
        {overige_html}
      </body>
    </html>
    """


# --------------------------------------------------------------------------
# Versturen
# --------------------------------------------------------------------------
def stuur_telegram(tekst: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram-secrets ontbreken, overslaan.", file=sys.stderr)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": tekst,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    resp = requests.post(url, json=payload, timeout=30)
    if not resp.ok:
        print(f"Telegram-fout: {resp.status_code} {resp.text}", file=sys.stderr)


def stuur_email(html_body: str, heeft_top_signaal: bool):
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        print("Email-secrets ontbreken, overslaan.", file=sys.stderr)
        return

    onderwerp = (
        "🚨 BESTE SIGNAAL vandaag — actie vereist?"
        if heeft_top_signaal
        else "Beste Signaal Bot — geen resultaten"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = onderwerp
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    rows = haal_selecties_op(LOOKBACK_DAYS)
    xgb_scores = haal_xgboost_scores_op(LOOKBACK_DAYS)

    print(f"Selecties opgehaald: {len(rows)}")
    print(f"XGBoost-scores opgehaald: {len(xgb_scores)} tickers")

    ranking = bouw_ranking(rows, xgb_scores)
    totaal_strategieen = len({row["strategie"] for row in rows}) or 1

    # Diagnostiek
    n_met_xgb = sum(1 for r in ranking if r["heeft_xgb"])
    n_met_kasstr = sum(1 for r in ranking if r["is_kasstr"])
    n_beide = sum(1 for r in ranking if r["heeft_xgb"] and r["is_kasstr"])
    print(f"Tickers met XGBoost-signaal : {n_met_xgb}")
    print(f"Tickers met kasstr-signaal  : {n_met_kasstr}")
    print(f"Tickers met BEIDE signalen  : {n_beide}")

    telegram_tekst = maak_telegram_bericht(ranking, LOOKBACK_DAYS, totaal_strategieen)
    email_html = maak_email_html(ranking, LOOKBACK_DAYS, totaal_strategieen)

    stuur_telegram(telegram_tekst)
    stuur_email(email_html, heeft_top_signaal=bool(ranking))

    print(f"Klaar. {len(ranking)} unieke ticker/beurs-combinaties geanalyseerd.")


if __name__ == "__main__":
    main()
