"""
vers_signaal.py
================
Zusterbot van a_trade.py, met een andere denkwijze: waar a_trade.py overlap
telt over het HELE LOOKBACK_DAYS-venster (en dus wacht tot een ticker over
meerdere dagen door meerdere strategieën bevestigd is -- vaak te laat),
telt deze bot overlap ENKEL op de meest recente datum in het venster
("vandaag"). Doel: instappen op het moment dat de consensus vers ontstaat,
niet nadat de move al een paar dagen bezig is.

Ranking-logica:
  1. Overlap VANDAAG (aantal verschillende strategieën die exact op de
     meest recente datum dezelfde ticker+beurs kozen) -- zwaarst gewogen.
  2. Cumulatieve overlap over het volledige venster, als tiebreaker
     (puur informatief/tiebreak, telt niet mee als hoofdcriterium).
  3. Gemiddelde score, als laatste tiebreaker.

Aantal getoonde picks wordt bepaald door AANTAL_PICKS, los van het budget.
Rest van de architectuur (kosteninschatting, Telegram HTML + email HTML)
is identiek aan a_trade.py.

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
  LOOKBACK_DAYS        - hoeveel dagen data opgehaald wordt om de meest
                         recente datum in te bepalen (default 3; de
                         overlap-berekening zelf kijkt enkel naar die
                         meest recente datum, niet naar het hele venster)
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

    `datum` is text in ISO-formaat (YYYY-MM-DD, geen tijdstip) -- vergelijk
    dus met een string in hetzelfde formaat, geen datetime-object.
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
# Ranking -- kern van het verschil met a_trade.py
# --------------------------------------------------------------------------
def bouw_ranking(rows):
    """
    Groepeert per (ticker, beurs). Overlap wordt ENKEL geteld op de meest
    recente datum die voor die ticker/beurs voorkomt in het venster --
    niet over het hele venster heen zoals in a_trade.py.
    """
    # Eerste pas: bepaal per (ticker, beurs) de meest recente datum.
    laatste_datum_per_key = {}
    for row in rows:
        key = (row["ticker"], row["beurs"])
        d = row["datum"]
        if key not in laatste_datum_per_key or d > laatste_datum_per_key[key]:
            laatste_datum_per_key[key] = d

    groepen = defaultdict(lambda: {
        "strategieen_vandaag": set(),
        "strategieen_totaal": set(),
        "scores_vandaag": [],
        "koers": None,
        "laatste_datum": None,
        "grafiek": None,
    })

    for row in rows:
        key = (row["ticker"], row["beurs"])
        g = groepen[key]
        laatste_datum = laatste_datum_per_key[key]
        g["laatste_datum"] = laatste_datum

        g["strategieen_totaal"].add(row["strategie"])

        if row["datum"] == laatste_datum:
            g["strategieen_vandaag"].add(row["strategie"])

            score = row.get("score")
            if score is not None:
                try:
                    g["scores_vandaag"].append(float(score))
                except (TypeError, ValueError):
                    pass

            if row.get("grafiek"):
                g["grafiek"] = row["grafiek"]
            if row["koers"] is not None:
                g["koers"] = row["koers"]

    ranking = []
    for (ticker, beurs), g in groepen.items():
        overlap_vandaag = len(g["strategieen_vandaag"])
        # Enkel tickers tonen die vandaag door minstens 2 strategieën
        # samen gekozen zijn -- een overlap van 1 is geen "vers signaal",
        # gewoon een normale eenmalige selectie.
        if overlap_vandaag < 2:
            continue

        overlap_totaal = len(g["strategieen_totaal"])
        avg_score = (
            sum(g["scores_vandaag"]) / len(g["scores_vandaag"])
            if g["scores_vandaag"] else 0.0
        )
        ranking.append({
            "ticker": ticker,
            "beurs": beurs,
            "overlap": overlap_vandaag,
            "overlap_totaal": overlap_totaal,
            "strategieen": sorted(g["strategieen_vandaag"]),
            "avg_score": avg_score,
            "koers": g["koers"],
            "grafiek": g["grafiek"],
            "laatste_datum": g["laatste_datum"],
        })

    ranking.sort(
        key=lambda r: (r["overlap"], r["overlap_totaal"], r["avg_score"]),
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


def maak_telegram_bericht(ranking, lookback_days):
    vandaag = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not ranking:
        return (
            f"⚠️ <b>Vers Signaal Bot — {vandaag}</b>\n"
            f"Geen verse overlap (2+ strategieën, zelfde dag) gevonden "
            f"binnen de laatste {lookback_days} dagen."
        )

    top = ranking[:TOP_N]
    kost, kost_pct = bereken_kosten(TRANSACTIE_BEDRAG)
    medailles = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    lijnen = [
        f"🌱🌱🌱 <b>VERS SIGNAAL — {vandaag}</b> 🌱🌱🌱",
        f"<i>Overlap telkens geteld op de meest recente datum per ticker, niet over meerdere dagen</i>",
        f"<i>Aantal picks: {TOP_N} (van €{TRANSACTIE_BEDRAG:,.0f} elk)</i>",
    ]

    for i, r in enumerate(top):
        medaille = medailles[i] if i < len(medailles) else "▫️"
        strategieen_str = _esc(", ".join(r["strategieen"]))
        lijnen.append("")
        lijnen.append(f"{medaille} <b>{_esc(r['ticker'])}</b> ({_esc(r['beurs'])}) — {_esc(r['laatste_datum'])}")
        lijnen.append(f"✅ Overlap vandaag: <b>{r['overlap']}</b> strategieën — {strategieen_str}")
        lijnen.append(f"ℹ️ Cumulatieve overlap venster: {r['overlap_totaal']}")
        if r["avg_score"]:
            lijnen.append(f"📊 Gem. score: <b>{r['avg_score']:.2f}</b>")
        if r["koers"] is not None:
            lijnen.append(f"💶 Laatste koers: {_esc(r['koers'])}")
        lijnen.append(f"💸 Kost bij €{TRANSACTIE_BEDRAG:,.0f}: ~€{kost:.2f} ({kost_pct:.2f}%)")
        if r["grafiek"]:
            lijnen.append(f'📈 <a href="{_esc(r["grafiek"])}">Grafiek</a>')

    return "\n".join(lijnen)


def maak_email_html(ranking, lookback_days):
    if not ranking:
        return (
            f"<h2>⚠️ Vers Signaal Bot</h2>"
            f"<p>Geen verse overlap (2+ strategieën, zelfde dag) gevonden "
            f"binnen de laatste {lookback_days} dagen.</p>"
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
            f"<td>{r['overlap']}</td><td>{r['avg_score']:.2f}</td></tr>"
            for r in top[1:]
        )
        overige_html = f"""
        <h3>Overige kanshebbers (elk €{TRANSACTIE_BEDRAG:,.0f})</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
          <tr style="background:#eee;"><th>Ticker</th><th>Beurs</th><th>Overlap vandaag</th><th>Score</th></tr>
          {rijen}
        </table>
        """

    return f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <div style="background:#e8f5e9; border:3px solid #43a047; border-radius:10px;
                    padding:20px; margin-bottom:20px;">
          <h1 style="color:#2e7d32; margin-top:0;">🌱 VERS SIGNAAL 🌱</h1>
          <p style="color:#555;">Overlap geteld op {beste['laatste_datum']} zelf
             (niet gesommeerd over het venster van {lookback_days} dagen)</p>
          <p style="color:#555;">Aantal picks: <b>{TOP_N}</b> (van €{TRANSACTIE_BEDRAG:,.0f} elk)</p>
          <h2 style="font-size:28px; margin-bottom:5px;">🥇 {beste['ticker']} ({beste['beurs']})</h2>
          <p style="font-size:18px;">✅ Overlap vandaag: <b>{beste['overlap']} strategieën</b>
             — {strategieen_html}</p>
          <p style="font-size:14px; color:#777;">Cumulatieve overlap venster: {beste['overlap_totaal']}</p>
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
        "🌱 VERS SIGNAAL vandaag — actie vereist?"
        if heeft_top_signaal
        else "Vers Signaal Bot — geen resultaten"
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

    print(f"Klaar. {len(ranking)} verse overlap-kandidaten (2+ strategieën, zelfde dag) gevonden.")


if __name__ == "__main__":
    main()
