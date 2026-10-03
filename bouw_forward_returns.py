#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bouw_forward_returns.py  —  FEATURE STORE: selecties -> forward-rendement  v1.6

DOEL
====
De `selecties`-tabel bevat al, per dag/aandeel/strategie, een volledige set
indicatorwaarden (RSI, ATR%, VWAP-afstand, kwaliteitsscores, overlap-
tellingen, ...). Wat ontbreekt is het LABEL: wat deed dat aandeel nadien
werkelijk? Dit script vult die kloof op een incrementele, idempotente
manier -- geen jaren wachten per losse strategie, maar de duizenden reeds
gelogde rijen in `selecties` retroactief van een label voorzien zodra er
genoeg tijd verstreken is.

Voor elke (ticker, datum, strategie)-combinatie in `selecties` waarvoor nog
geen (volledig) label bestaat in `forward_returns`, wordt via yfinance het
koersverloop na `datum` opgehaald en worden de horizons uit HORIZONS
berekend (bv. 5/10/20/30/60 handelsdagen) t.o.v. de reeds gelogde `koers`
(entry-koers op selectiemoment) -- niet t.o.v. een nieuw opgehaalde
slotkoers op diezelfde dag, om consistent te blijven met wat de bots zelf
als instapkoers rapporteerden.

INCREMENTEEL EN IDEMPOTENT
===========================
- Een horizon wordt pas ingevuld zodra er ECHT genoeg toekomstige
  handelsdagen bestaan (geen NaN/placeholder-vulling voor wat nog niet
  gebeurd is). Een rij kan dus na een eerste run enkel fwd_ret_5d hebben,
  en een latere run vult fwd_ret_10d/20d/30d/60d aan -- vandaar de upsert
  (INSERT ... ON CONFLICT ... DO UPDATE) i.p.v. gewone INSERT.
- Tickers worden gegroepeerd: één yfinance-download per ticker dekt ALLE
  openstaande (datum, strategie)-rijen van die ticker in dit run, i.p.v.
  een aparte call per rij.
- Enkel tickers met effectief nog onvolledige forward_returns-rijen (of nog
  helemaal geen rij) worden opnieuw bevraagd -- reeds volledig ingevulde
  combinaties worden overgeslagen.
- PER-HORIZON cutoff (v1.1, 2026-09-26): een rij wordt al opgepikt zodra
  ÉÉN horizon oud genoeg is EN nog een ontbrekend label heeft voor die
  horizon -- niet pas wanneer de LANGSTE horizon (bv. 60d) matuur is. Zo
  hoeft fwd_ret_30d niet nodeloos ~48 dagen langer te wachten dan nodig,
  enkel omdat er ook een 60d-horizon in de lijst staat.

KWALITEIT VAN HET LABEL (v1.2 - v1.4)
=====================================
- v1.2 ONBRUIKBARE KOERS = ONTBREKEND: een forward-koers die NaN/inf is of
  <= 0 wordt als None (NULL) weggeschreven. In v1.1 liep zo'n waarde
  ongehinderd door: NaN kwam in forward_returns terecht, en omdat de
  selectie op `IS NULL` filtert (NaN is geen NULL) werd zo'n rij daarna nooit
  meer opgepikt.
- v1.3 RUWE KOERSEN: history(auto_adjust=False). entry_koers is de
  onaangepaste koers op selectiemoment; auto_adjust=True corrigeert achteraf
  voor dividenden (Yahoo gaf voor MOL.WA hierdoor negatieve koersen).
- v1.4 SPLIT-CORRECTIE: 'Close' is bij auto_adjust=False nog WEL achteraf
  gecorrigeerd voor splits, terwijl entry_koers de koers van toen is. Een
  split tussen selectiedatum en vandaag gaf dus een nep-rendement (APH, 2:1
  op 2026-09-03: -50% op alle rijen van daarvoor). Nu wordt de entry-koers
  gedeeld door het product van alle split-ratio's met ex-datum NA de
  prijsdatum van de entry. Splits met ex-datum <= prijsdatum veranderen
  niets (de entry-koers is dan al de nieuwe koers). `entry_koers` in de
  tabel blijft de gelogde koers; `fwd_ret_<h>d` is de gezaghebbende waarde.
- v1.4 EENHEIDSBREUK: Yahoo kan een reeks plots in een andere eenheid geven
  (BCG.L: pence tot 2026-08-21, ponden daarna: factor 100). Een
  forward-koers waarvan de verhouding tot de (split-gecorrigeerde) entry
  tussen 70x en 150x of tussen 1/150 en 1/70 ligt, wordt met 100 vermenigvuldigd
  of gedeeld. Zulke verhoudingen komen in echte koersen niet voor.
  De opgeslagen fwd_close_<h>d staat dan in de eenheid van entry_koers.

- v1.5 DUBBELE SELECTIES: `selecties` bevat meerdere rijen voor dezelfde
  (ticker, datum, strategie) met verschillende koers (691 van ~29.600
  combinaties; mediaan 0,76%, max 9,7%; vooral marktsent, kasstr, db,
  graham, hoogl). Bij de upsert won de laatst verwerkte rij, en dat was
  willekeurig. Nu wordt per combinatie EEN rij gekozen (DISTINCT ON): de
  eerst gelogde (created_at, id), want dat is de koers waarop je het signaal
  had kunnen handelen. Met DUBBELE_SELECTIE=laatste kies je de laatst
  gelogde. Een bijkomend voordeel: elke combinatie wordt nog maar een keer
  berekend.

MELDINGEN (v1.6)
================
- v1.6 TELEGRAM + MAIL ZOALS a_trade.py: het bericht gaat nu als HTML
  (parse_mode=HTML, alles geescaped met html.escape) naar Telegram EN per
  mail (Gmail SMTP). In v1.5 ging het bericht met legacy Markdown de deur uit;
  'forward_returns' bevat een losse underscore, waardoor Telegram het met
  HTTP 400 weigerde, terwijl de statuscode niet gecontroleerd werd (dus
  niets in de log). Nu wordt de status altijd gecontroleerd en geprint.
- Het bericht wordt ALTIJD verstuurd: ook als er niets te doen was (0
  openstaande rijen), en ook als het script crasht (foutmelding per Telegram
  en mail, daarna wordt de fout opnieuw opgegooid zodat de workflow rood wordt).
- Waarschuwingen (bv. 'HLX: geen koersdata') worden verzameld en in het
  bericht getoond.

GEBRUIK
=======
  python bouw_forward_returns.py build

Env vars:
  SUPABASE_DB_URL     - Postgres connectiestring
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  - optioneel, stuurt een samenvatting
                        naar Telegram; als afwezig wordt dat overgeslagen
  EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER - optioneel, stuurt dezelfde
                        samenvatting per mail (Gmail SMTP); als afwezig
                        wordt dat overgeslagen
  MAX_TICKERS_PER_RUN - veiligheidslimiet tegen te lange/rate-limited
                        runs (default 400)
  HORIZONS            - komma-gescheiden lijst handelsdagen (default
                        "5,10,20")
  DUBBELE_SELECTIE    - "eerste" (default) of "laatste": welke rij van
                        dubbele selecties-rijen (zelfde ticker/datum/
                        strategie) de entry-koers levert
"""

import os
import sys
import math
import time
import html
import smtplib
import traceback
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import yfinance as yf
import pandas as pd
import requests

SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# v1.6: mail, zelfde secrets als a_trade.py
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")
MAX_TICKERS_PER_RUN = int(os.environ.get("MAX_TICKERS_PER_RUN", "400"))
# LET OP: als je dit wijzigt, moet forward_returns's schema mee-veranderen
# (fwd_close_<h>d/fwd_ret_<h>d-kolommen moeten al bestaan voor elke horizon
# in deze lijst) -- de kolomnamen hier worden dynamisch opgebouwd, maar
# de tabel zelf niet automatisch.
HORIZONS = sorted(int(h) for h in os.environ.get("HORIZONS", "5,10,20").split(","))

# Eenheidsbreuk (pence <-> ponden e.d.): verhouding fwd_close / entry buiten
# elke realistische koersbeweging, maar vlak bij een factor 100.
EENHEID_RATIO_MIN = 70.0
EENHEID_RATIO_MAX = 150.0

# v1.5: welke rij bij dubbele selecties-rijen (zelfde ticker/datum/strategie).
DUBBELE_SELECTIE = os.environ.get("DUBBELE_SELECTIE", "eerste").strip().lower()
if DUBBELE_SELECTIE not in ("eerste", "laatste"):
    raise SystemExit(f"DUBBELE_SELECTIE moet 'eerste' of 'laatste' zijn, niet {DUBBELE_SELECTIE!r}")

# v1.6: waarschuwingen van dit run, getoond in het bericht.
WAARSCHUWINGEN: List[str] = []
MAX_WAARSCHUWINGEN_IN_BERICHT = 10


def vandaag() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _esc(s) -> str:
    """Escaped voor Telegram HTML parse_mode (&, <, > moeten geescaped)."""
    return html.escape(str(s))


def waarschuw(tekst: str) -> None:
    """Print een waarschuwing en bewaar ze voor het eindbericht."""
    print(f"  [WARN] {tekst}")
    WAARSCHUWINGEN.append(tekst)


# --------------------------------------------------------------------------
# Versturen (v1.6: Telegram HTML + mail, zoals a_trade.py)
# --------------------------------------------------------------------------
# v1.5 (Markdown, status niet gecontroleerd: een 400 bleef onzichtbaar):
# def send_telegram(tekst: str) -> None:
#     if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
#         print(tekst)
#         return
#     try:
#         requests.post(
#             f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#             json={"chat_id": TELEGRAM_CHAT_ID, "text": tekst, "parse_mode": "Markdown"},
#             timeout=10,
#         )
#     except Exception as e:
#         print(f"Telegram fout: {e}")
def send_telegram(tekst_html: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram-secrets ontbreken, overslaan.", file=sys.stderr)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": tekst_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.ok:
            print("Telegram verstuurd.")
        else:
            print(f"Telegram-fout: {resp.status_code} {resp.text}", file=sys.stderr)
    except Exception as e:
        print(f"Telegram-fout: {e}", file=sys.stderr)


def send_email(onderwerp: str, html_body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASS or not EMAIL_RECEIVER:
        print("Email-secrets ontbreken, overslaan.", file=sys.stderr)
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = onderwerp
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        print("Mail verstuurd.")
    except Exception as e:
        print(f"Mail-fout: {e}", file=sys.stderr)


def maak_telegram_bericht(open_n: int, bijgewerkt: int, compleet: int,
                          tickers_n: int, overgeslagen: int, nog_wachtend: int) -> str:
    lijnen = [f"📊 <b>Forward-Returns Feature Store — {vandaag()}</b>", ""]
    if open_n == 0:
        lijnen.append("✅ Niets te doen: geen openstaande rijen om te labelen.")
    else:
        lijnen.append(f"🔄 <b>{bijgewerkt}</b> rijen bijgewerkt in forward_returns")
        lijnen.append(f"🏁 {compleet} daarvan nu volledig (alle {HORIZONS[-1]} handelsdagen ingevuld)")
        lijnen.append(f"📥 {open_n} rijen waren openstaand, {nog_wachtend} blijven nog wachten")
        lijnen.append(f"🧾 {tickers_n} unieke tickers verwerkt dit run"
                      + (f" ({overgeslagen} nog te gaan)" if overgeslagen > 0 else ""))
    if WAARSCHUWINGEN:
        lijnen.append("")
        lijnen.append(f"⚠️ <b>{len(WAARSCHUWINGEN)} waarschuwing(en)</b>")
        for w in WAARSCHUWINGEN[:MAX_WAARSCHUWINGEN_IN_BERICHT]:
            lijnen.append(f"• {_esc(w)}")
        if len(WAARSCHUWINGEN) > MAX_WAARSCHUWINGEN_IN_BERICHT:
            lijnen.append(f"… en {len(WAARSCHUWINGEN) - MAX_WAARSCHUWINGEN_IN_BERICHT} meer (zie log)")
    return "\n".join(lijnen)


def maak_email_html(open_n: int, bijgewerkt: int, compleet: int,
                    tickers_n: int, overgeslagen: int, nog_wachtend: int) -> str:
    if open_n == 0:
        kern = "<p>✅ Niets te doen: geen openstaande rijen om te labelen.</p>"
    else:
        kern = (
            f"<p>🔄 <b>{bijgewerkt}</b> rijen bijgewerkt in forward_returns</p>"
            f"<p>🏁 {compleet} daarvan nu volledig (alle {HORIZONS[-1]} handelsdagen ingevuld)</p>"
            f"<p>📥 {open_n} rijen waren openstaand, {nog_wachtend} blijven nog wachten</p>"
            f"<p>🧾 {tickers_n} unieke tickers verwerkt dit run"
            + (f" ({overgeslagen} nog te gaan)" if overgeslagen > 0 else "")
            + "</p>"
        )
    warn_html = ""
    if WAARSCHUWINGEN:
        items = "".join(f"<li>{_esc(w)}</li>" for w in WAARSCHUWINGEN)
        warn_html = f"<h3>⚠️ {len(WAARSCHUWINGEN)} waarschuwing(en)</h3><ul>{items}</ul>"
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <div style="background:#e8f4fd; border:2px solid #2196f3; border-radius:10px; padding:20px;">
          <h2 style="margin-top:0;">📊 Forward-Returns Feature Store — {vandaag()}</h2>
          {kern}
        </div>
        {warn_html}
      </body>
    </html>
    """


def meld_fout(fout_tekst: str) -> None:
    """Foutmelding per Telegram en mail (v1.6: ook bij een crash een bericht)."""
    tg = (f"❌ <b>Forward-Returns Feature Store — {vandaag()}</b>\n\n"
          f"Het script is gecrasht:\n<pre>{_esc(fout_tekst[-1500:])}</pre>")
    mail = (f"<html><body style='font-family: Arial, sans-serif;'>"
            f"<h2>❌ Forward-Returns Feature Store — {vandaag()}</h2>"
            f"<p>Het script is gecrasht:</p><pre>{_esc(fout_tekst)}</pre></body></html>")
    send_telegram(tg)
    send_email(f"❌ Forward-Returns FOUT — {vandaag()}", mail)


# --------------------------------------------------------------------------
# Stap 1: welke (ticker, datum, strategie)-rijen hebben nog een onvolledig
# of ontbrekend label, en zijn oud genoeg voor MINSTENS ÉÉN horizon?
# --------------------------------------------------------------------------
def haal_openstaande_rijen(conn) -> List[dict]:
    nu = datetime.now(timezone.utc)
    # Per horizon een eigen cutoff -- een rij wordt al opgepikt zodra ÉÉN
    # horizon oud genoeg is en nog een ontbrekend label heeft, niet pas
    # wanneer de LANGSTE horizon (bv. 60d) matuur is. Zo hoeft fwd_ret_30d
    # niet nodeloos ~48 dagen te wachten tot de rij ook oud genoeg is voor 60d.
    voorwaarden = []
    params: Dict[str, str] = {}
    for h in HORIZONS:
        kolom = f"fwd_ret_{h}d"
        param_naam = f"cutoff_{h}"
        # ~1.5x buffer voor weekends/feestdagen om h handelsdagen te dekken
        params[param_naam] = (nu - timedelta(days=int(h * 1.6) + 3)).strftime("%Y-%m-%d")
        voorwaarden.append(
            f"(s.datum <= %({param_naam})s AND (f.ticker IS NULL OR f.{kolom} IS NULL))"
        )
    horizon_filter = " OR ".join(voorwaarden)

    # ticker NOT LIKE '%/%' sluit bot_01cointegr.py's samengestelde
    # "TICKER_A/TICKER_B"-paarstrings uit (zie bouw_generieke_technicals.py
    # voor dezelfde fix en de uitleg) -- kunnen nooit via yfinance opgelost
    # worden, en koers is voor die rijen sowieso een spread-ratio, geen
    # echte prijs, dus een forward-rendement erop zou toch niet zinvol zijn.
    # De %% (i.p.v. losse %) is nodig omdat psycopg2 een bare % anders
    # verwart met zijn eigen %(naam)s-placeholder-syntax.
    # v1.5: per (ticker, datum, strategie) EEN selecties-rij (DISTINCT ON),
    # deterministisch gekozen op created_at/id. NULLS LAST zodat een rij zonder
    # created_at nooit voor een rij met tijdstip wordt gekozen.
    if DUBBELE_SELECTIE == "eerste":
        volgorde_dubbel = "created_at ASC NULLS LAST, id ASC"
    else:
        volgorde_dubbel = "created_at DESC NULLS LAST, id DESC"

    # v1.4 (alle selecties-rijen; bij dubbelen won de laatst verwerkte rij):
    # query = f"""
    #     SELECT s.ticker, s.datum, s.strategie, s.beurs, s.koers
    #     FROM selecties s
    #     LEFT JOIN forward_returns f
    #       ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
    #     WHERE s.koers IS NOT NULL
    #       AND s.ticker NOT LIKE '%%/%%'
    #       AND ({horizon_filter})
    #     ORDER BY s.ticker, s.datum;
    # """
    query = f"""
        WITH s AS (
            SELECT DISTINCT ON (ticker, datum, strategie)
                   ticker, datum, strategie, beurs, koers
            FROM selecties
            WHERE koers IS NOT NULL
              AND ticker NOT LIKE '%%/%%'
            ORDER BY ticker, datum, strategie, {volgorde_dubbel}
        )
        SELECT s.ticker, s.datum, s.strategie, s.beurs, s.koers
        FROM s
        LEFT JOIN forward_returns f
          ON s.ticker = f.ticker AND s.datum = f.datum AND s.strategie = f.strategie
        WHERE ({horizon_filter})
        ORDER BY s.ticker, s.datum;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        return cur.fetchall()


# --------------------------------------------------------------------------
# Hulpfuncties voor de kwaliteit van het label (v1.4)
# --------------------------------------------------------------------------
def split_factor_na(splits: pd.Series, prijsdatum: pd.Timestamp) -> float:
    """Product van alle split-ratio's met ex-datum STRIKT NA prijsdatum.
    splits: Series (index = ex-datum, waarde = ratio, bv. 2.0 voor 2:1,
    0.1 voor een omgekeerde 1:10). Lege reeks -> 1.0."""
    if splits is None or splits.empty:
        return 1.0
    na = splits[splits.index > prijsdatum]
    if na.empty:
        return 1.0
    return float(na.prod())


def corrigeer_eenheid(fwd_close: float, entry_adj: float) -> Tuple[float, bool]:
    """Corrigeert een factor-100 eenheidsbreuk (pence <-> ponden).
    Geeft (gecorrigeerde_koers, is_gecorrigeerd) terug."""
    ratio = fwd_close / entry_adj
    if EENHEID_RATIO_MIN <= ratio <= EENHEID_RATIO_MAX:
        return fwd_close / 100.0, True
    if (1.0 / EENHEID_RATIO_MAX) <= ratio <= (1.0 / EENHEID_RATIO_MIN):
        return fwd_close * 100.0, True
    return fwd_close, False


# --------------------------------------------------------------------------
# Stap 2: per ticker het koersverloop ophalen en de horizons berekenen
# --------------------------------------------------------------------------
def bereken_labels_voor_ticker(ticker: str, rijen: List[dict]) -> List[dict]:
    """rijen = alle openstaande (datum, strategie, beurs, koers)-combinaties
    voor deze ene ticker. Geeft een lijst dicts terug, klaar voor upsert.

    v1.2: een forward-koers die NaN/inf is of <= 0 wordt behandeld als
    ONTBREKEND (None -> NULL).
    v1.3: ruwe slotkoersen (auto_adjust=False), consistent met entry_koers.
    v1.4: entry_koers wordt gecorrigeerd voor splits na de prijsdatum, en
    een factor-100 eenheidsbreuk in de koersreeks wordt hersteld.
    v1.6: waarschuwingen worden ook verzameld voor het eindbericht."""
    vroegste = min(r["datum"] for r in rijen)
    try:
        # v1.2: hist = yf.Ticker(ticker).history(start=vroegste, auto_adjust=True)
        hist = yf.Ticker(ticker).history(start=vroegste, auto_adjust=False)
    except Exception as e:
        # v1.5: print(f"  [WARN] {ticker}: download mislukt ({e})")
        waarschuw(f"{ticker}: download mislukt ({e})")
        return []

    if hist is None or hist.empty or "Close" not in hist.columns:
        # v1.5: print(f"  [WARN] {ticker}: geen koersdata")
        waarschuw(f"{ticker}: geen koersdata")
        return []

    closes = hist["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    datums = closes.index

    # v1.4: splits (ex-datum -> ratio). Ontbreekt de kolom, dan geen correctie.
    splits = None
    if "Stock Splits" in hist.columns:
        sp = hist["Stock Splits"].copy()
        sp.index = pd.to_datetime(sp.index).tz_localize(None)
        splits = sp[sp > 0]

    n_split_gecorr = 0
    n_eenheid_gecorr = 0

    resultaten = []
    for r in rijen:
        try:
            doel = pd.Timestamp(r["datum"])
            pos_kandidaten = datums.searchsorted(doel)
            if pos_kandidaten >= len(datums):
                continue  # datum ligt na alle beschikbare koersdata, niets te doen
            entry_koers = float(r["koers"])
            if not entry_koers or math.isnan(entry_koers) or entry_koers <= 0:
                continue

            # v1.4: prijsdatum = laatste handelsdag <= datum (een weekend-
            # selectie logt de slotkoers van vrijdag). Ligt er geen
            # handelsdag voor `datum` in de reeks, dan gebruiken we `datum`.
            pos_vorige = datums.searchsorted(doel, side="right") - 1
            prijsdatum = datums[pos_vorige] if pos_vorige >= 0 else doel

            factor = split_factor_na(splits, prijsdatum)
            entry_adj = entry_koers / factor
            if factor != 1.0:
                n_split_gecorr += 1

            rij = {
                "ticker": ticker, "datum": r["datum"], "strategie": r["strategie"],
                "beurs": r["beurs"], "entry_koers": entry_koers,
            }
            for h in HORIZONS:
                idx = pos_kandidaten + h
                if idx < len(closes):
                    fwd_close = float(closes.iloc[idx])

                    # v1.2: onbruikbare koers (NaN/inf of <= 0) = ontbrekend (NULL).
                    if not math.isfinite(fwd_close) or fwd_close <= 0:
                        rij[f"fwd_close_{h}d"] = None
                        rij[f"fwd_ret_{h}d"] = None
                        continue

                    # v1.4: factor-100 eenheidsbreuk herstellen.
                    fwd_close, hersteld = corrigeer_eenheid(fwd_close, entry_adj)
                    if hersteld:
                        n_eenheid_gecorr += 1

                    # v1.3 (zonder split-correctie):
                    # rij[f"fwd_ret_{h}d"] = round((fwd_close / entry_koers - 1) * 100, 3)
                    rij[f"fwd_close_{h}d"] = fwd_close
                    rij[f"fwd_ret_{h}d"] = round((fwd_close / entry_adj - 1) * 100, 3)
                else:
                    rij[f"fwd_close_{h}d"] = None
                    rij[f"fwd_ret_{h}d"] = None
            resultaten.append(rij)
        except Exception as e:
            # v1.5: print(f"  [WARN] {ticker} {r['datum']}/{r['strategie']}: {e}")
            waarschuw(f"{ticker} {r['datum']}/{r['strategie']}: {e}")
            continue

    if n_split_gecorr or n_eenheid_gecorr:
        print(f"  [INFO] {ticker}: {n_split_gecorr} rijen split-gecorrigeerd, "
              f"{n_eenheid_gecorr} koersen eenheid-gecorrigeerd")

    return resultaten


# --------------------------------------------------------------------------
# Stap 3: upsert naar forward_returns
# --------------------------------------------------------------------------
def upsert_labels(conn, rijen: List[dict]) -> int:
    if not rijen:
        return 0
    kolommen = ["ticker", "datum", "strategie", "beurs", "entry_koers"]
    for h in HORIZONS:
        kolommen += [f"fwd_close_{h}d", f"fwd_ret_{h}d"]

    kolom_lijst = ", ".join(kolommen)
    placeholders = ", ".join(f"%({k})s" for k in kolommen)
    update_lijst = ", ".join(f"{k} = EXCLUDED.{k}" for k in kolommen if k not in ("ticker", "datum", "strategie"))
    update_lijst += ", bijgewerkt_op = now()"

    query = f"""
        INSERT INTO forward_returns ({kolom_lijst})
        VALUES ({placeholders})
        ON CONFLICT (ticker, datum, strategie)
        DO UPDATE SET {update_lijst};
    """
    with conn.cursor() as cur:
        for rij in rijen:
            volledige_rij = {k: rij.get(k) for k in kolommen}
            cur.execute(query, volledige_rij)
    conn.commit()
    return len(rijen)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_build():
    if not SUPABASE_DB_URL:
        print("FOUT: SUPABASE_DB_URL ontbreekt.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        open_rijen = haal_openstaande_rijen(conn)
        print(f"{len(open_rijen)} (ticker, datum, strategie)-rijen wachten op een (bijgewerkt) label.")

        per_ticker: Dict[str, List[dict]] = {}
        for r in open_rijen:
            per_ticker.setdefault(r["ticker"], []).append(r)

        # Prioriteer tickers op hun OUDSTE nog openstaande datum (langst
        # wachtend eerst) i.p.v. alfabetisch -- anders blijft een ticker als
        # AAPL (bijna dagelijks opnieuw geselecteerd door meerdere
        # strategieën) elke run in de eerste MAX_TICKERS_PER_RUN vallen, en
        # komen alfabetisch latere tickers structureel nooit aan de beurt.
        oudste_datum_per_ticker = {
            t: min(r["datum"] for r in rijen) for t, rijen in per_ticker.items()
        }
        alle_tickers_gesorteerd = sorted(per_ticker.keys(), key=lambda t: oudste_datum_per_ticker[t])
        tickers = alle_tickers_gesorteerd[:MAX_TICKERS_PER_RUN]
        overgeslagen = len(per_ticker) - len(tickers)
        print(f"{len(tickers)} unieke tickers te verwerken dit run"
              + (f" ({overgeslagen} tickers volgen in een volgend run, MAX_TICKERS_PER_RUN bereikt)" if overgeslagen > 0 else ""))

        totaal_bijgewerkt = 0
        totaal_compleet = 0
        for i, ticker in enumerate(tickers, start=1):
            labels = bereken_labels_voor_ticker(ticker, per_ticker[ticker])
            aantal = upsert_labels(conn, labels)
            totaal_bijgewerkt += aantal
            totaal_compleet += sum(1 for l in labels if l.get(f"fwd_ret_{HORIZONS[-1]}d") is not None)
            if i % 25 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} tickers verwerkt...")
            time.sleep(0.1)

        nog_wachtend = len(open_rijen) - totaal_bijgewerkt

        # v1.5 (Markdown, enkel Telegram):
        # bericht = (
        #     f"📊 *Forward-Returns Feature Store — {vandaag()}*\n\n"
        #     f"{totaal_bijgewerkt} rijen bijgewerkt in forward_returns "
        #     f"({totaal_compleet} daarvan nu volledig, alle {HORIZONS[-1]} handelsdagen ingevuld)\n"
        #     f"{len(tickers)} unieke tickers verwerkt dit run"
        #     + (f" ({overgeslagen} tickers nog te gaan)" if overgeslagen > 0 else "")
        # )
        # send_telegram(bericht)

        # v1.6: ALTIJD Telegram + mail, ook als er niets te doen was.
        argumenten = (len(open_rijen), totaal_bijgewerkt, totaal_compleet,
                      len(tickers), overgeslagen, nog_wachtend)
        send_telegram(maak_telegram_bericht(*argumenten))
        send_email(f"📊 Forward-Returns Feature Store — {vandaag()}", maak_email_html(*argumenten))

        print(f"\nKlaar. {totaal_bijgewerkt} rijen bijgewerkt.")
    finally:
        conn.close()


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "build"
    try:
        run_build()
    except SystemExit:
        raise
    except Exception:
        # v1.6: ook bij een crash een bericht, daarna opnieuw opgooien
        # zodat de workflow rood wordt.
        fout = traceback.format_exc()
        print(fout, file=sys.stderr)
        meld_fout(fout)
        raise
