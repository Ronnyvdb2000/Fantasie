"""
analyseer_weekly_correlaties.py
================================
Ad hoc analysescript (workflow_dispatch, GEEN cron) dat na verloop van tijd
op de data in weekly_toppers + weekly_topper_parameters een correlatie
zoekt tussen de weekprestatie (week_perf) en de parameters, opgesplitst in
drie families:

  - "evolutie (delta binnen de week)": de VERANDERING van elke dagaccurate
    Groep A-parameter over de week (dag_index max - dag_index min).
    LET OP: voor prijs-afgeleide parameters (close, rsi, macd_hist,
    dist_sma*_pct, ...) is deze evolutie mechanisch verweven met week_perf
    zelf (delta close IS letterlijk de teller van week_perf) -- een sterke
    correlatie hier is dus verwacht en weinig informatief op zich, tenzij
    het om DIVERGENTIE gaat (prijs op, indicator neer). Toch mee opgenomen
    omdat dit letterlijk was wat gevraagd werd ("evolutie van 1 of meer
    parameters").
  - "startniveau (dag 1, voorspellend)": de waarde van diezelfde Groep
    A-parameters op de EERSTE dag van de week, dus vóór het grootste deel
    van de weekbeweging plaatsvond. Dit is de forward-looking/voorspellende
    variant en dus de familie die het meest interessant is als er ooit
    iets bruikbaars uitkomt.
  - "fundamenteel/rank": Groep B (fundamenteel, quasi-constant binnen de
    week) en Groep C (cross-sectionele rank/score, bevroren herhaald)
    parameters vs week_perf.

Elke familie krijgt een EIGEN Benjamini-Hochberg FDR-correctie
(alpha=0.05, zelfde drempel als bot_01repititief), zodat het gelijktijdig
testen van tientallen parameters niet vanzelf "significante"
toevalstreffers oplevert. Gebruikt Spearman-correlatie (robuuster tegen de
elders in dit project al gedocumenteerde yfinance-outliers/extreme
ratio's) i.p.v. Pearson.

Draait bewust NIET op een schema -- pas zinvol na een aantal weken
accumulatie van data. Trigger handmatig via workflow_dispatch in
analyse_correlaties.yml wanneer er genoeg data verzameld is.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import requests
from dotenv import load_dotenv

import weekly_db_opvolg as db

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

MIN_N = 30        # minimum aantal geldige (niet-NaN) paren vooraleer een correlatie berekend wordt
FDR_ALPHA = 0.05  # zelfde drempel als bot_01repititief

SLEUTEL = ["ticker", "beurs", "week_startdatum"]

# Groep A: dagaccuraat -- zowel evolutie (delta) als startniveau worden getest
GROEP_A_KOLOMMEN = [
    "close", "rsi", "macd_hist", "atr_pct", "dist_sma50_pct", "dist_sma200_pct",
    "support", "resistance", "stop",
    "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "fcf_yield",
    "dividend_yield", "market_cap",
]

# Groep B + C: quasi-constant binnen de week -- enkel niveau getest (delta is per constructie ~0)
GROEP_BC_KOLOMMEN = [
    "roe_pct", "current_ratio", "revenue_growth_pct", "eps_growth_pct", "debt_to_ebitda",
    "piotroski_score", "combined_rank", "roc_rank", "ey_rank", "vc2_score", "total_score",
]

# Families waarvoor een significant resultaat ook effectief betekenisvol is
# (dus expliciet UITGESLOTEN: "evolutie (delta binnen de week)", want die
# correleert mechanisch bijna altijd sterk met week_perf zelf -- zie de
# module-docstring. Die familie blijft wel in het volledige CSV-rapport
# staan, maar wordt niet apart uitgelicht in Telegram/e-mail.)
NOEMENSWAARDIGE_FAMILIES = ["startniveau (dag 1, voorspellend)", "fundamenteel/rank"]


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
            # Dit ontbrak eerder: zonder deze check faalt een Telegram-fout
            # (bv. verkeerde chat_id, ongeldige HTML-entity) volledig stil --
            # geen exception, geen log, gewoon geen bericht dat aankomt.
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


def haal_data():
    """Haalt weekly_toppers en weekly_topper_parameters volledig op als DataFrames."""
    conn = db._get_connection()
    try:
        df_toppers = pd.read_sql(
            "SELECT week_startdatum, lijst, ticker, beurs, type, rang, week_perf "
            "FROM weekly_toppers",
            conn,
        )
        df_params = pd.read_sql(
            "SELECT week_startdatum, ticker, beurs, datum, dag_index, "
            + ", ".join(GROEP_A_KOLOMMEN + GROEP_BC_KOLOMMEN)
            + " FROM weekly_topper_parameters",
            conn,
        )
    finally:
        conn.close()
    return df_toppers, df_params


def bouw_analysetabel(df_toppers, df_params):
    """
    Combineert beide tabellen tot 1 rij per (ticker, beurs, week_startdatum)
    met: week_perf, begin_<param> (de volledige rij bij dag_index=min) en
    eind_<param> (de volledige rij bij dag_index=max) voor elke Groep
    A-kolom, en niveau_<param> voor elke Groep B/C-kolom.

    Gebruikt bewust idxmin/idxmax i.p.v. groupby().first()/.last(): die
    laatste kiezen namelijk PER KOLOM de eerste/laatste niet-NaN waarde,
    wat zou kunnen leiden tot een "begin"-rij die kolommen uit verschillende
    dagen door elkaar mengt zodra 1 kolom toevallig NaN is op dag 1.
    idxmin/idxmax selecteert steeds de volledige rij van exact dezelfde dag.
    """
    if df_params.empty or df_toppers.empty:
        return pd.DataFrame()

    groep = df_params.groupby(SLEUTEL, dropna=False)["dag_index"]
    idx_begin = groep.idxmin()
    idx_eind = groep.idxmax()

    df_begin = df_params.loc[idx_begin].reset_index(drop=True)
    df_eind = df_params.loc[idx_eind].reset_index(drop=True)

    df_begin_a = df_begin[SLEUTEL + GROEP_A_KOLOMMEN].rename(
        columns={k: f"begin_{k}" for k in GROEP_A_KOLOMMEN}
    )
    df_eind_a = df_eind[SLEUTEL + GROEP_A_KOLOMMEN].rename(
        columns={k: f"eind_{k}" for k in GROEP_A_KOLOMMEN}
    )
    df_eind_bc = df_eind[SLEUTEL + GROEP_BC_KOLOMMEN].rename(
        columns={k: f"niveau_{k}" for k in GROEP_BC_KOLOMMEN}
    )

    df = df_toppers.merge(df_begin_a, on=SLEUTEL, how="inner")
    df = df.merge(df_eind_a, on=SLEUTEL, how="inner")
    df = df.merge(df_eind_bc, on=SLEUTEL, how="inner")

    for kolom in GROEP_A_KOLOMMEN:
        df[f"delta_{kolom}"] = df[f"eind_{kolom}"] - df[f"begin_{kolom}"]

    return df


def bh_correctie(p_waarden, alpha=FDR_ALPHA):
    """
    Benjamini-Hochberg FDR-correctie. p_waarden: array-like van p-waardes,
    mag NaN bevatten (die worden genegeerd voor de correctie en blijven NaN
    in het resultaat). Retourneert (aangepaste_p, significant) in de
    oorspronkelijke volgorde.
    """
    p_waarden = np.asarray(p_waarden, dtype=float)
    geldig = ~np.isnan(p_waarden)
    aangepast = np.full_like(p_waarden, np.nan)
    significant = np.zeros_like(p_waarden, dtype=bool)

    m = int(geldig.sum())
    if m == 0:
        return aangepast, significant

    idx_geldig = np.where(geldig)[0]
    p_geldig = p_waarden[idx_geldig]
    volgorde = np.argsort(p_geldig)
    gesorteerd = p_geldig[volgorde]
    ranks = np.arange(1, m + 1)
    aangepast_gesorteerd = gesorteerd * m / ranks
    # Cumulatief minimum van achter naar voren -> monotoon dalende adjusted p-waardes
    aangepast_gesorteerd = np.minimum.accumulate(aangepast_gesorteerd[::-1])[::-1]
    aangepast_gesorteerd = np.clip(aangepast_gesorteerd, 0, 1)

    aangepast_geldig = np.empty(m)
    aangepast_geldig[volgorde] = aangepast_gesorteerd
    aangepast[idx_geldig] = aangepast_geldig
    significant[idx_geldig] = aangepast_geldig <= alpha
    return aangepast, significant


def analyseer_familie(df, kolomnamen, familienaam):
    """
    Berekent Spearman-correlatie tussen week_perf en elke kolom in
    kolomnamen, met een eigen BH-correctie binnen deze familie.
    Retourneert een lijst van dicts (1 per parameter).
    """
    resultaten = []
    for kolom in kolomnamen:
        if kolom not in df.columns:
            continue
        subset = df[["week_perf", kolom]].dropna()
        n = len(subset)
        if n < MIN_N:
            resultaten.append({
                "familie": familienaam, "parameter": kolom, "n": n,
                "rho": None, "p_waarde": None,
            })
            continue
        rho, p = spearmanr(subset["week_perf"], subset[kolom])
        resultaten.append({
            "familie": familienaam, "parameter": kolom, "n": n,
            "rho": float(rho), "p_waarde": float(p),
        })

    p_waarden = [r["p_waarde"] if r["p_waarde"] is not None else np.nan for r in resultaten]
    aangepast, significant = bh_correctie(p_waarden)
    for r, p_adj, sig in zip(resultaten, aangepast, significant):
        r["p_aangepast"] = None if np.isnan(p_adj) else float(p_adj)
        r["significant"] = bool(sig)
    return resultaten


def bouw_html_rapport(rapport: pd.DataFrame, aantal_weken: int) -> str:
    """Bouwt de volledige tabel als HTML e-mailbody (alle rijen, alle families)."""
    def fmt(v, decimalen=3):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.{decimalen}f}"

    rijen_html = []
    for r in rapport.itertuples():
        stijl = "font-weight:bold;background:#eaffea;" if r.significant else ""
        rijen_html.append(
            f"<tr style='{stijl}'>"
            f"<td>{r.familie}</td><td>{r.parameter}</td><td>{r.n}</td>"
            f"<td>{fmt(r.rho)}</td><td>{fmt(r.p_waarde, 5)}</td>"
            f"<td>{fmt(r.p_aangepast, 5)}</td><td>{'JA' if r.significant else ''}</td>"
            f"</tr>"
        )

    return f"""
    <html><body style="font-family:Arial,sans-serif;font-size:13px;">
    <h2>📊 Correlatie-analyse Weekly Toppers</h2>
    <p>{aantal_weken} ticker-weken geanalyseerd. FDR-correctie (Benjamini-Hochberg,
    alpha={FDR_ALPHA}) toegepast per familie afzonderlijk.</p>
    <p><b>Let op:</b> de familie "evolutie (delta binnen de week)" correleert
    voor prijs-afgeleide parameters bijna altijd mechanisch sterk met
    week_perf zelf (delta close IS de teller van week_perf) -- significantie
    daar is verwacht en weinig informatief op zich. De families
    "startniveau (dag 1, voorspellend)" en "fundamenteel/rank" zijn de
    families waar een significant resultaat effectief iets zou kunnen
    betekenen.</p>
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;">
    <tr style="background:#ddd;">
        <th>Familie</th><th>Parameter</th><th>n</th><th>rho</th>
        <th>p-waarde</th><th>p (aangepast)</th><th>Significant</th>
    </tr>
    {"".join(rijen_html)}
    </table>
    </body></html>
    """


def main():
    logger.info("Data ophalen uit weekly_toppers en weekly_topper_parameters...")
    df_toppers, df_params = haal_data()
    df = bouw_analysetabel(df_toppers, df_params)

    if df.empty:
        logger.info("Nog geen (voldoende) data beschikbaar.")
        stuur_telegram(
            "📊 Correlatie-analyse: nog geen data beschikbaar in "
            "weekly_toppers/weekly_topper_parameters."
        )
        stuur_email(
            "<p>Nog geen data beschikbaar in weekly_toppers/weekly_topper_parameters.</p>",
            "Correlatie-analyse — geen data",
        )
        return

    logger.info("%d ticker-weken beschikbaar voor analyse.", len(df))

    alle_resultaten = []
    alle_resultaten += analyseer_familie(
        df, [f"delta_{k}" for k in GROEP_A_KOLOMMEN], "evolutie (delta binnen de week)"
    )
    alle_resultaten += analyseer_familie(
        df, [f"begin_{k}" for k in GROEP_A_KOLOMMEN], "startniveau (dag 1, voorspellend)"
    )
    alle_resultaten += analyseer_familie(
        df, [f"niveau_{k}" for k in GROEP_BC_KOLOMMEN], "fundamenteel/rank"
    )

    rapport = pd.DataFrame(alle_resultaten)
    rapport = rapport.sort_values(by=["familie", "p_aangepast"], na_position="last")

    print(rapport.to_string(index=False))

    output_pad = "correlatie_rapport.csv"
    rapport.to_csv(output_pad, index=False)
    logger.info("Volledig rapport weggeschreven naar %s", output_pad)

    # Telegram: kort en enkel de NOEMENSWAARDIGE families (evolutie-familie
    # is bijna altijd (bijna) volledig significant om mechanische redenen --
    # dat zou het bericht enkel vervuilen, zie module-docstring).
    noemenswaardig = rapport[
        (rapport["significant"] == True) & (rapport["familie"].isin(NOEMENSWAARDIGE_FAMILIES))
    ]
    aantal_evolutie_sig = int(
        (rapport["significant"] & (rapport["familie"] == "evolutie (delta binnen de week)")).sum()
    )

    if noemenswaardig.empty:
        telegram_bericht = (
            f"📊 <b>Correlatie-analyse</b> ({len(df)} ticker-weken)\n"
            f"Geen significante bevindingen in de betekenisvolle families "
            f"(startniveau/fundamenteel) na FDR-correctie."
        )
    else:
        regels = "\n".join(
            f"• {r.familie} — {r.parameter}: rho={r.rho:.3f}, p_adj={r.p_aangepast:.4f}, n={r.n}"
            for r in noemenswaardig.itertuples()
        )
        telegram_bericht = (
            f"📊 <b>Correlatie-analyse</b> ({len(df)} ticker-weken)\n"
            f"{len(noemenswaardig)} noemenswaardige bevinding(en) na FDR-correctie:\n{regels}"
        )
    if aantal_evolutie_sig:
        telegram_bericht += (
            f"\n\n(+{aantal_evolutie_sig} significant in 'evolutie' -- "
            f"grotendeels verwacht/mechanisch, zie volledig rapport)"
        )
    telegram_bericht += "\n\nVolledig rapport: zie e-mail of de CSV-artifact van deze run."

    stuur_telegram(telegram_bericht)
    stuur_email(
        bouw_html_rapport(rapport, len(df)),
        f"Correlatie-analyse Weekly Toppers ({len(df)} ticker-weken)",
    )


if __name__ == "__main__":
    main()
