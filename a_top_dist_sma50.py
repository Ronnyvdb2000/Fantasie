"""
top_dist_sma50.py
===================
Ad hoc, read-only scriptje: toont de tickers met de laagste (meest
negatieve) `pct_from_ma50` uit `generieke_technicals`, gemeten binnen de
laatste --dagen dagen (default 15). Dit is exact het signaal dat in
analyseer_forward_correlaties.py naar voren kwam als sterkste (nog steeds
zwakke) 3-wekense voorspeller: begin_dist_sma50_pct, rho=-0,159,
p_aangepast=0,0001, n=797 (2026-09-22-run). Negatief rho betekent: hoe
verder ONDER het 50-daags gemiddelde bij meting, hoe iets beter het
rendement over de volgende 3 weken -- vandaar dat dit script AFLOPEND
sorteert op de meest negatieve pct_from_ma50 (verst onder het gemiddelde).

BELANGRIJK, lees dit voor gebruik: dit is GEEN koopadvies en geen
gevalideerde stock-picker. Het onderliggende signaal verklaart ~2,5% van
de variantie (R² = rho^2) in 3-wekenrendement, 1 keer gemeten op 797
ticker-weken. Op individueel-aandeelniveau is dit nog steeds overwegend
ruis -- dit script toont enkel WELKE tickers vandaag toevallig aan het
criterium voldoen, niet dat ze "goed gaan presteren". Zie
fantasie-trading-bots.md voor de volledige context en kanttekeningen.

Toont ook, ter context, welke strategieen (uit `selecties`) deze ticker
in dezelfde periode zelf ook al selecteerden -- zo zie je meteen overlap
met a_trade/a_trade_combi's eigen logica.

Vereist env var: SUPABASE_DB_URL

Gebruik:
    python top_dist_sma50.py                  # default: laatste 15 dagen, top 10
    python top_dist_sma50.py --dagen 10 --top 20
    python top_dist_sma50.py --csv resultaat.csv
"""

import argparse
import os
import smtplib
import sys
from datetime import date, timedelta
from email.mime.text import MIMEText

import pandas as pd
import psycopg2
import requests


def _get_connection():
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip().strip('"').strip("'")
    if not db_url:
        sys.exit("Fout: env var SUPABASE_DB_URL is niet gezet.")
    return psycopg2.connect(db_url)


def haal_recente_technicals(conn, dagen: int) -> pd.DataFrame:
    """Per ticker de meest recente rij uit generieke_technicals binnen de
    laatste `dagen` dagen (DISTINCT ON, nieuwste datum per ticker wint).

    `datum` staat, net als in `selecties`, als ISO-tekst in de databank
    (geen native date-type) -- vandaar .isoformat() i.p.v. het date-object
    zelf door te geven. Werkt correct omdat ISO-datumstrings (YYYY-MM-DD)
    ook lexicografisch correct sorteren/vergelijken.
    """
    sinds = (date.today() - timedelta(days=dagen)).isoformat()
    query = """
        SELECT DISTINCT ON (ticker)
            ticker, datum, pct_from_ma50, pct_from_ma200, rsi14, atr14_pct,
            ibs, vol_ratio_20d, pct_from_high52w
        FROM generieke_technicals
        WHERE datum >= %s AND pct_from_ma50 IS NOT NULL
        ORDER BY ticker, datum DESC
    """
    return pd.read_sql(query, conn, params=(sinds,))


def haal_recente_strategieen(conn, tickers, dagen: int) -> dict:
    """Per ticker: welke strategieen deze ticker binnen dezelfde periode
    zelf ook al selecteerden, ter context (overlap met a_trade e.a.)."""
    if not tickers:
        return {}
    sinds = date.today() - timedelta(days=dagen)
    query = """
        SELECT DISTINCT ticker, strategie
        FROM selecties
        WHERE ticker = ANY(%s) AND datum >= %s
    """
    df = pd.read_sql(query, conn, params=(list(tickers), sinds.isoformat()))
    resultaat = {}
    for ticker, groep in df.groupby("ticker"):
        resultaat[ticker] = sorted(groep["strategie"].tolist())
    return resultaat


def bouw_telegram_bericht(df: pd.DataFrame, strategieen: dict, dagen: int) -> str:
    regels = [
        f"*Top {len(df)} tickers -- laagste pct_from_ma50* (laatste {dagen} dagen)",
        "_Signaal uit analyseer_forward_correlaties.py: rho=-0,159 op 3 weken, "
        "R2~2,5%. GEEN koopadvies, enkel een filter._",
        "",
    ]
    for _, r in df.iterrows():
        strat_lijst = ", ".join(strategieen.get(r["ticker"], [])) or "-"
        regels.append(
            f"*{r['ticker']}*  ({r['datum']})\n"
            f"  pct_from_ma50: {r['pct_from_ma50']:+.2f}%  RSI14: {r['rsi14']:.1f}  "
            f"IBS: {r['ibs']:.2f}  ATR%: {r['atr14_pct']:.2f}\n"
            f"  strategieën: {strat_lijst}"
        )
    return "\n".join(regels)


def bouw_html_rapport(df: pd.DataFrame, strategieen: dict, dagen: int) -> str:
    rijen_html = []
    for _, r in df.iterrows():
        strat_lijst = ", ".join(strategieen.get(r["ticker"], [])) or "-"
        rijen_html.append(
            f"<tr>"
            f"<td>{r['ticker']}</td><td>{r['datum']}</td>"
            f"<td>{r['pct_from_ma50']:+.2f}%</td><td>{r['rsi14']:.1f}</td>"
            f"<td>{r['ibs']:.2f}</td><td>{r['atr14_pct']:.2f}</td>"
            f"<td>{strat_lijst}</td>"
            f"</tr>"
        )
    return f"""
    <html><body style="font-family:Arial,sans-serif;font-size:13px;">
    <h2>📉 Top {len(df)} tickers — laagste pct_from_ma50 (laatste {dagen} dagen)</h2>
    <p>Signaal uit <code>analyseer_forward_correlaties.py</code>: rho=-0,159 op een
    3-wekenhorizon, R²~2,5%. <b>Geen koopadvies</b>, enkel een filter op wie vandaag
    toevallig het verst onder zijn 50-daags gemiddelde noteert.</p>
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;">
    <tr style="background:#ddd;">
        <th>Ticker</th><th>Datum</th><th>pct_from_ma50</th><th>RSI14</th>
        <th>IBS</th><th>ATR%</th><th>Recent geselecteerd door</th>
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
        description="Toont de tickers met de laagste pct_from_ma50 uit generieke_technicals "
                    "binnen de laatste N dagen (het sterkste, nog steeds zwakke, signaal uit "
                    "analyseer_forward_correlaties.py). Geen koopadvies."
    )
    parser.add_argument("--dagen", type=int, default=15,
                        help="Hoe recent de technicals-meting moet zijn (default 15)")
    parser.add_argument("--top", type=int, default=10, help="Aantal tickers om te tonen (default 10)")
    parser.add_argument("--csv", help="Optioneel: schrijf het resultaat weg naar dit csv-pad")
    parser.add_argument("--telegram", action="store_true", help="Stuur beknopte samenvatting via Telegram")
    parser.add_argument("--email", action="store_true", help="Stuur volledig rapport via e-mail")
    args = parser.parse_args()

    conn = _get_connection()
    try:
        df = haal_recente_technicals(conn, args.dagen)
        if df.empty:
            sys.exit(f"Geen data gevonden in generieke_technicals binnen de laatste {args.dagen} dagen.")

        df = df.sort_values("pct_from_ma50").head(args.top).reset_index(drop=True)
        strategieen = haal_recente_strategieen(conn, df["ticker"].tolist(), args.dagen)
    finally:
        conn.close()

    print(f"\nTop {len(df)} tickers, laagste pct_from_ma50, binnen de laatste {args.dagen} dagen:")
    print(
        "(negatief = onder het 50-daags gemiddelde -- signaal uit analyseer_forward_correlaties.py, "
        "rho=-0,159 op 3-wekenhorizon, R2~2,5%. GEEN koopadvies, enkel een filter.)\n"
    )
    for _, r in df.iterrows():
        strat_lijst = ", ".join(strategieen.get(r["ticker"], [])) or "-"
        print(
            f"  {r['ticker']:>10}  datum={r['datum']}  "
            f"pct_from_ma50={r['pct_from_ma50']:+.2f}%  "
            f"RSI14={r['rsi14']:.1f}  IBS={r['ibs']:.2f}  "
            f"ATR%={r['atr14_pct']:.2f}  "
            f"recent geselecteerd door: {strat_lijst}"
        )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nWeggeschreven naar {args.csv}")

    if args.telegram:
        verstuur_telegram(bouw_telegram_bericht(df, strategieen, args.dagen))
        print("Bericht verstuurd via Telegram.")

    if args.email:
        verstuur_email(
            f"Top {len(df)} tickers — laagste pct_from_ma50 ({args.dagen} dagen)",
            bouw_html_rapport(df, strategieen, args.dagen),
        )
        print("Rapport verstuurd via e-mail.")


if __name__ == "__main__":
    main()
