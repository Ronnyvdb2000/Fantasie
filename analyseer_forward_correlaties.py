"""
analyseer_forward_correlaties.py
==================================
Ad hoc analysescript (workflow_dispatch, GEEN cron), zusterscript van
analyseer_weekly_correlaties.py. Test dezelfde begin_-parameters (dag 1
van de week) en niveau_-parameters (fundamenteel/rank, cross-sectioneel)
uit `weekly_topper_parameters`, maar tegen een LANGERE, zelf berekende
horizon i.p.v. het ingebouwde 1-weekse `week_perf` uit `weekly_toppers`.

Aanleiding (2026-09-22): de bestaande 1-weekse analyse toont enkel zwakke
correlaties (rho 0,05-0,13) voor begin_pb_ratio/begin_dividend_yield/
begin_pe_ratio/niveau_total_score. Vermoeden: sommige fundamentele
signalen hebben meer dan 1 week nodig om zich in de koers te vertalen.
Dit script berekent daarom zelf, via yfinance, het rendement over
--weken weken (default 3) vanaf dag 1 van elke ticker-week, volledig
onafhankelijk van het bestaande, reeds 1 week vooruit berekende
`week_perf`.

Belangrijk: enkel ticker-weken die al minstens --weken weken oud zijn
worden meegenomen (anders is de outcome nog niet meetbaar) -- zie
bereken_forward_rendement(). Met `weekly_topper_parameters` momenteel
maar ~4 weken data (2026-08-24 e.v.), zal dit bij --weken 3 initieel
weinig/geen bruikbare rijen opleveren; dat groeit vanzelf mee met de
tijd, net als bij de rank-kolommen (zie fantasie-trading-bots.md).

Zowel begin_ als eind_/target-prijs worden VERS via yfinance opgehaald
(niet de reeds opgeslagen begin_close hergebruikt), zodat beide
prijspunten uit dezelfde download/adjustment-instellingen komen --
zelfde voorzichtigheid als analyse_selecties.py.

Vereist env var: SUPABASE_DB_URL (zelfde secret als de andere scripts).
Optioneel voor --telegram: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
Optioneel voor --email: EMAIL_USER, EMAIL_PASS, EMAIL_RECEIVER

Installatie:
    pip install psycopg2-binary yfinance pandas numpy scipy requests --break-system-packages

Gebruik:
    python analyseer_forward_correlaties.py                 # default 3 weken
    python analyseer_forward_correlaties.py --weken 4
    python analyseer_forward_correlaties.py --csv rapport.csv
    python analyseer_forward_correlaties.py --telegram --email
"""

import argparse
import os
import smtplib
import sys
from datetime import date
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests
import yfinance as yf
from scipy.stats import spearmanr

MIN_N = 30
FDR_ALPHA = 0.05
SLEUTEL = ["ticker", "beurs", "week_startdatum"]

# Zelfde twee kolomgroepen als in analyseer_weekly_correlaties.py.
GROEP_A_KOLOMMEN = [
    "close", "rsi", "macd_hist", "atr_pct", "dist_sma50_pct", "dist_sma200_pct",
    "support", "resistance", "stop",
    "pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "fcf_yield",
    "dividend_yield", "market_cap",
]
GROEP_BC_KOLOMMEN = [
    "roe_pct", "current_ratio", "revenue_growth_pct", "eps_growth_pct", "debt_to_ebitda",
    "piotroski_score", "combined_rank", "roc_rank", "ey_rank", "vc2_score", "total_score",
]

# Zelfde reden als in analyseer_weekly_correlaties.py: "evolutie" correleert
# mechanisch met een 1-weekse uitkomst, maar heeft die eigenschap NIET meer
# t.o.v. een outcome die pas N weken later gemeten wordt -- daarom hier
# gewoon niet meegenomen, enkel de 2 betekenisvolle families.
FAMILIES = ["startniveau (dag 1, voorspellend)", "fundamenteel/rank"]


def _get_connection():
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip().strip('"').strip("'")
    if not db_url:
        sys.exit("Fout: env var SUPABASE_DB_URL is niet gezet.")
    return psycopg2.connect(db_url)


def haal_parameters(conn) -> pd.DataFrame:
    kolommen = ", ".join(GROEP_A_KOLOMMEN + GROEP_BC_KOLOMMEN)
    query = (
        f"SELECT week_startdatum, ticker, beurs, datum, dag_index, {kolommen} "
        f"FROM weekly_topper_parameters"
    )
    df = pd.read_sql(query, conn)
    if df.empty:
        return df
    df["datum"] = pd.to_datetime(df["datum"]).dt.date
    return df


def bouw_analysetabel(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Analoog aan bouw_analysetabel() in analyseer_weekly_correlaties.py:
    gebruikt idxmin/idxmax op dag_index om de volledige dag-1-rij
    (begin_*, + de kalenderdatum) en de laatste-dag-rij (niveau_* voor de
    quasi-constante Groep B/C-parameters) per ticker-week te selecteren.
    """
    if df_params.empty:
        return pd.DataFrame()

    groep = df_params.groupby(SLEUTEL, dropna=False)["dag_index"]
    idx_begin = groep.idxmin()
    idx_eind = groep.idxmax()

    df_begin = df_params.loc[idx_begin].reset_index(drop=True)
    df_eind = df_params.loc[idx_eind].reset_index(drop=True)

    df_begin_a = df_begin[SLEUTEL + ["datum"] + GROEP_A_KOLOMMEN].rename(
        columns={**{k: f"begin_{k}" for k in GROEP_A_KOLOMMEN}, "datum": "begin_datum"}
    )
    df_eind_bc = df_eind[SLEUTEL + GROEP_BC_KOLOMMEN].rename(
        columns={k: f"niveau_{k}" for k in GROEP_BC_KOLOMMEN}
    )

    return df_begin_a.merge(df_eind_bc, on=SLEUTEL, how="inner")


def bereken_forward_rendement(df: pd.DataFrame, weken: int) -> pd.DataFrame:
    """
    Berekent voor elke ticker-week het rendement over `weken` weken vanaf
    begin_datum, via yfinance. Zelfde voorzichtigheidspatroon als
    bereken_rendementen() in analyse_selecties.py: caching per ticker,
    eerste beschikbare close op/na een datum, NaN-guard.
    """
    horizon_dagen = weken * 7
    vandaag = date.today()

    df = df.copy()
    df["target_datum"] = df["begin_datum"].apply(lambda d: d + pd.Timedelta(days=horizon_dagen).to_pytimedelta())

    meetbaar = df[df["target_datum"] <= vandaag].copy()
    te_jong = len(df) - len(meetbaar)
    if te_jong:
        print(f"  {te_jong} ticker-weken overgeslagen: horizon van {weken} weken nog niet bereikt.")
    if meetbaar.empty:
        return pd.DataFrame()

    tickers = meetbaar["ticker"].unique()
    print(f"Prijsdata ophalen voor {len(tickers)} tickers via yfinance (horizon {weken} weken)...")

    hist_cache = {}
    for i, ticker in enumerate(tickers, 1):
        subset = meetbaar[meetbaar["ticker"] == ticker]
        start = subset["begin_datum"].min()
        eind = subset["target_datum"].max() + pd.Timedelta(days=7)  # buffer voor weekend/feestdagen
        try:
            hist = yf.download(ticker, start=start, end=eind, progress=False, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist_cache[ticker] = hist
        except Exception as e:
            print(f"  waarschuwing: kon {ticker} niet ophalen ({e})")
            hist_cache[ticker] = None

        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} tickers verwerkt")

    resultaten = []
    overgeslagen_leeg = 0
    overgeslagen_nan = 0
    for _, row in meetbaar.iterrows():
        ticker = row["ticker"]
        hist = hist_cache.get(ticker)
        if hist is None or hist.empty:
            overgeslagen_leeg += 1
            continue

        na_begin = hist[hist.index.date >= row["begin_datum"]]
        na_target = hist[hist.index.date >= row["target_datum"]]
        if na_begin.empty or na_target.empty:
            overgeslagen_leeg += 1
            continue

        begin_prijs = float(na_begin["Close"].iloc[0])
        eind_prijs = float(na_target["Close"].iloc[0])

        if pd.isna(begin_prijs) or pd.isna(eind_prijs) or begin_prijs <= 0:
            overgeslagen_nan += 1
            continue

        rendement_pct = (eind_prijs - begin_prijs) / begin_prijs * 100
        nieuwe_rij = row.to_dict()
        nieuwe_rij[f"perf_{weken}w"] = rendement_pct
        resultaten.append(nieuwe_rij)

    if overgeslagen_leeg:
        print(f"  waarschuwing: {overgeslagen_leeg} rijen overgeslagen (geen prijsdata rond begin/target-datum)")
    if overgeslagen_nan:
        print(f"  waarschuwing: {overgeslagen_nan} rijen overgeslagen wegens NaN/ongeldige koersdata")

    return pd.DataFrame(resultaten)


def bh_correctie(p_waarden, alpha=FDR_ALPHA):
    """Zelfde Benjamini-Hochberg-implementatie als analyseer_weekly_correlaties.py."""
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
    aangepast_gesorteerd = np.minimum.accumulate(aangepast_gesorteerd[::-1])[::-1]
    aangepast_gesorteerd = np.clip(aangepast_gesorteerd, 0, 1)

    aangepast_geldig = np.empty(m)
    aangepast_geldig[volgorde] = aangepast_gesorteerd
    aangepast[idx_geldig] = aangepast_geldig
    significant[idx_geldig] = aangepast_geldig <= alpha
    return aangepast, significant


def analyseer_familie(df, uitkomst_kolom, kolomnamen, familienaam):
    """Zelfde als analyseer_weekly_correlaties.py, maar met de uitkomstkolom
    als parameter i.p.v. hardcoded 'week_perf'."""
    resultaten = []
    for kolom in kolomnamen:
        if kolom not in df.columns:
            continue
        subset = df[[uitkomst_kolom, kolom]].dropna()
        n = len(subset)
        if n < MIN_N:
            resultaten.append({
                "familie": familienaam, "parameter": kolom, "n": n,
                "rho": None, "p_waarde": None,
            })
            continue
        rho, p = spearmanr(subset[uitkomst_kolom], subset[kolom])
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


def bouw_telegram_samenvatting(rapport: pd.DataFrame, n_totaal: int, weken: int) -> str:
    regels = [f"*Forward-correlatie-analyse ({weken} weken)* ({date.today().isoformat()})", ""]
    noemenswaardig = rapport[(rapport["significant"] == True) & (rapport["familie"].isin(FAMILIES))]
    if noemenswaardig.empty:
        regels.append(f"{n_totaal} ticker-weken geanalyseerd. Geen significante bevindingen na FDR-correctie.")
    else:
        regels.append(f"{n_totaal} ticker-weken geanalyseerd. {len(noemenswaardig)} significante bevinding(en):")
        for r in noemenswaardig.itertuples():
            regels.append(f"  {r.familie} — {r.parameter}: rho={r.rho:.3f}, p_adj={r.p_aangepast:.4f}, n={r.n}")
    return "\n".join(regels)


def bouw_html_rapport(rapport: pd.DataFrame, n_totaal: int, weken: int) -> str:
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
    <h2>📊 Forward-correlatie-analyse ({weken} weken)</h2>
    <p>{n_totaal} ticker-weken geanalyseerd, enkel ticker-weken die al minstens
    {weken} weken oud zijn. FDR-correctie (Benjamini-Hochberg, alpha={FDR_ALPHA})
    per familie afzonderlijk.</p>
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;">
    <tr style="background:#ddd;">
        <th>Familie</th><th>Parameter</th><th>n</th><th>rho</th>
        <th>p-waarde</th><th>p (aangepast)</th><th>Significant</th>
    </tr>
    {"".join(rijen_html)}
    </table>
    </body></html>
    """


def verstuur_telegram(tekst: str):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram-secrets ontbreken, overslaan.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 4000
    for i in range(0, len(tekst), max_len):
        chunk = tekst[i:i + max_len]
        resp = requests.post(url, data={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
        if resp.status_code != 200:
            print(f"waarschuwing: Telegram-verzending mislukt ({resp.status_code}): {resp.text}")


def verstuur_email(onderwerp: str, html_body: str):
    user = os.environ.get("EMAIL_USER")
    wachtwoord = os.environ.get("EMAIL_PASS")
    ontvanger = os.environ.get("EMAIL_RECEIVER")
    if not user or not wachtwoord or not ontvanger:
        print("Email-secrets ontbreken, overslaan.")
        return
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = onderwerp
    msg["From"] = user
    msg["To"] = ontvanger
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, wachtwoord)
        server.sendmail(user, [ontvanger], msg.as_string())


def main():
    parser = argparse.ArgumentParser(
        description="Test begin_/niveau_-parameters uit weekly_topper_parameters tegen een "
                    "zelf berekende N-weekse forward return (default 3 weken)."
    )
    parser.add_argument("--weken", type=int, default=3,
                        help="Horizon in weken voor het forward-rendement (default 3)")
    parser.add_argument("--csv", help="Optioneel: schrijf het volledige rapport weg naar dit csv-pad")
    parser.add_argument("--telegram", action="store_true", help="Stuur beknopte samenvatting via Telegram")
    parser.add_argument("--email", action="store_true", help="Stuur volledig rapport via e-mail")
    args = parser.parse_args()

    if args.weken < 1:
        sys.exit("Fout: --weken moet minstens 1 zijn.")

    conn = _get_connection()
    try:
        df_params = haal_parameters(conn)
    finally:
        conn.close()

    if df_params.empty:
        sys.exit("Geen data gevonden in weekly_topper_parameters.")

    df = bouw_analysetabel(df_params)
    if df.empty:
        sys.exit("Geen ticker-weken kunnen samenstellen (join tussen dag-1- en laatste-dag-rijen leverde niets op).")

    df = bereken_forward_rendement(df, args.weken)
    uitkomst_kolom = f"perf_{args.weken}w"
    if df.empty:
        sys.exit(f"Geen enkele ticker-week is al oud genoeg voor een horizon van {args.weken} weken.")

    print(f"\n{len(df)} ticker-weken beschikbaar met een gemeten {args.weken}-weekse forward return.")

    alle_resultaten = []
    alle_resultaten += analyseer_familie(
        df, uitkomst_kolom, [f"begin_{k}" for k in GROEP_A_KOLOMMEN], "startniveau (dag 1, voorspellend)"
    )
    alle_resultaten += analyseer_familie(
        df, uitkomst_kolom, [f"niveau_{k}" for k in GROEP_BC_KOLOMMEN], "fundamenteel/rank"
    )

    rapport = pd.DataFrame(alle_resultaten)
    rapport = rapport.sort_values(by=["familie", "p_aangepast"], na_position="last")
    print("\n" + rapport.to_string(index=False))

    if args.csv:
        rapport.to_csv(args.csv, index=False)
        print(f"\nRapport weggeschreven naar {args.csv}")

    if args.telegram:
        verstuur_telegram(bouw_telegram_samenvatting(rapport, len(df), args.weken))
        print("Samenvatting verstuurd via Telegram.")

    if args.email:
        verstuur_email(
            f"Forward-correlatie-analyse ({args.weken} weken, n={len(df)})",
            bouw_html_rapport(rapport, len(df), args.weken),
        )
        print("Rapport verstuurd via e-mail.")


if __name__ == "__main__":
    main()
