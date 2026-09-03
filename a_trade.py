"""
a_trade.py
=============
Leest de Supabase `selecties`-tabel uit (gevuld door de andere Fantasie-bots:
bot_00kr, bot_00ms, bot_00db, bot_00cs, bot_00vcp, ...), en pikt daar het
"beste" signaal uit uit de laatste LOOKBACK_DAYS dagen.

Ranking-logica:
  1. Cross-strategie overlap (hoeveel verschillende strategieën kozen
     dezelfde ticker+beurs in de lookback-periode) -- zwaarst gewogen.
  2. Gemiddelde score binnen die overlap, als tiebreaker.
  3. Meest recente datum, als laatste tiebreaker.

Aantal getoonde picks wordt bepaald door AANTAL_PICKS, los van het budget.
BESCHIKBAAR_KAPITAAL/TRANSACTIE_BEDRAG worden enkel nog gebruikt om de
kosteninschatting per positie te berekenen. Stuurt de resultaten opvallend
naar Telegram (HTML, emojis) en naar e-mail (HTML, uitgelicht blok voor het
topsignaal), inclusief een kosteninschatting per positie (vaste kost +
variabele kost + TOB).

Env vars (zelfde secrets als de rest van de Fantasie-repo):
  SUPABASE_DB_URL     - Postgres connectiestring naar Supabase
  TELEGRAM_TOKEN
  TELEGRAM_CHAT_ID
  EMAIL_USER
  EMAIL_PASS
  EMAIL_RECEIVER
  BESCHIKBAAR_KAPITAAL - totaal beschikbaar bedrag in euro (default 2500)
  TRANSACTIE_BEDRAG    - bedrag per aankoop in euro (default 2500)
  AANTAL_PICKS         - aantal picks dat getoond wordt, los van budget
                         (default 5)
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
# Aantal getoonde picks is losgekoppeld van het budget -- het budget wordt
# enkel nog gebruikt om de kost per positie te berekenen, niet meer om het
# aantal picks te beperken.
AANTAL_PICKS = max(1, int(os.environ.get("AANTAL_PICKS", "5")))
TOP_N = AANTAL_PICKS

# Fiscale/kostenparameters (zelfde als de rest van de Fantasie-repo)
VASTE_KOST = 15.0
VARIABELE_KOST_PCT = 0.35  # %
TOB_PCT = 0.35  # %

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
    """Haalt alle selecties op van de laatste `lookback_days` dagen.

    Let op: de kolom `datum` is van het type text, in ISO-formaat
    (YYYY-MM-DD, geen tijdstip) -- dus we vergelijken met een string in
    hetzelfde formaat, geen datetime-object (dat geeft anders een
    type-mismatch-fout in Postgres).
    """
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


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def bouw_ranking(rows):
    """
    Groepeert per (ticker, beurs) en berekent overlap + gemiddelde score.
    Geeft een gesorteerde lijst terug, beste eerst.
    """
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
        ranking.append({
            "ticker": ticker,
            "beurs": beurs,
            "overlap": overlap,
            "strategieen": sorted(g["strategieen"]),
            "avg_score": avg_score,
            "koers": g["koers"],
            "grafiek": g["grafiek"],
            "laatste_datum": g["laatste_datum"],
        })

    ranking.sort(key=lambda r: (r["overlap"], r["avg_score"], r["laatste_datum"]), reverse=True)
    return ranking


# --------------------------------------------------------------------------
# Kosteninschatting
# --------------------------------------------------------------------------
def bereken_kosten(bedrag: float):
    """Vaste + variabele transactiekost + TOB voor een positie van `bedrag` euro."""
    variabele_kost = bedrag * VARIABELE_KOST_PCT / 100
    tob = bedrag * TOB_PCT / 100
    totaal = VASTE_KOST + variabele_kost + tob
    return totaal, (totaal / bedrag * 100)


# --------------------------------------------------------------------------
# Berichten opbouwen
# --------------------------------------------------------------------------
def _esc(s):
    """Escaped voor Telegram HTML parse_mode (&, <, > moeten geëscaped)."""
    return html.escape(str(s))


def maak_telegram_bericht(ranking, lookback_days):
    if not ranking:
        return (
            f"⚠️ <b>Beste Signaal Bot</b>\n"
            f"Geen selecties gevonden in de laatste {lookback_days} dagen."
        )

    top = ranking[:TOP_N]
    beste = top[0]
    kost, kost_pct = bereken_kosten(TRANSACTIE_BEDRAG)

    medailles = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    strategieen_str = _esc(", ".join(beste["strategieen"]))
    lijnen = [
        "🚨🚨🚨 <b>BESTE SIGNAAL</b> 🚨🚨🚨",
        f"<i>Analyse van laatste {lookback_days} dagen, {sum(r['overlap'] for r in ranking)} selecties totaal</i>",
        f"<i>Aantal picks: {TOP_N} (van €{TRANSACTIE_BEDRAG:,.0f} elk)</i>",
        "",
        f"{medailles[0]} <b>{_esc(beste['ticker'])}</b> ({_esc(beste['beurs'])})",
        f"✅ Overlap: <b>{beste['overlap']}/7</b> strategieën — {strategieen_str}",
    ]
    if beste["avg_score"]:
        lijnen.append(f"📊 Gem. score: <b>{beste['avg_score']:.2f}</b>")
    if beste["koers"] is not None:
        lijnen.append(f"💶 Laatste koers: {_esc(beste['koers'])}")
    lijnen.append(f"💸 Kost bij €{TRANSACTIE_BEDRAG:,.0f}: ~€{kost:.2f} ({kost_pct:.2f}%)")
    if beste["grafiek"]:
        lijnen.append(f'📈 <a href="{_esc(beste["grafiek"])}">Grafiek</a>')

    if len(top) > 1:
        lijnen.append("")
        lijnen.append(f"Overige kanshebbers (elk €{TRANSACTIE_BEDRAG:,.0f}):")
        for i, r in enumerate(top[1:], start=1):
            medaille = medailles[i] if i < len(medailles) else "▫️"
            lijnen.append(
                f"{medaille} {_esc(r['ticker'])} ({_esc(r['beurs'])}) — overlap {r['overlap']}/7, "
                f"score {r['avg_score']:.2f}"
            )

    return "\n".join(lijnen)


def maak_email_html(ranking, lookback_days):
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

    overige_html = ""
    if len(top) > 1:
        rijen = "".join(
            f"<tr><td>{r['ticker']}</td><td>{r['beurs']}</td>"
            f"<td>{r['overlap']}/7</td><td>{r['avg_score']:.2f}</td></tr>"
            for r in top[1:]
        )
        overige_html = f"""
        <h3>Overige kanshebbers (elk €{TRANSACTIE_BEDRAG:,.0f})</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
          <tr style="background:#eee;"><th>Ticker</th><th>Beurs</th><th>Overlap</th><th>Score</th></tr>
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
          <h2 style="font-size:28px; margin-bottom:5px;">🥇 {beste['ticker']} ({beste['beurs']})</h2>
          <p style="font-size:18px;">✅ Overlap: <b>{beste['overlap']}/7 strategieën</b>
             — {strategieen_html}</p>
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
    ranking = bouw_ranking(rows)

    telegram_tekst = maak_telegram_bericht(ranking, LOOKBACK_DAYS)
    email_html = maak_email_html(ranking, LOOKBACK_DAYS)

    stuur_telegram(telegram_tekst)
    stuur_email(email_html, heeft_top_signaal=bool(ranking))

    print(f"Klaar. {len(ranking)} unieke ticker/beurs-combinaties geanalyseerd.")


if __name__ == "__main__":
    main()
