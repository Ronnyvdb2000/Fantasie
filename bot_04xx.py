# --- STRATEGIE 5: POWER REVERSION ALPHA V2 (VERBETERDE VERSIE + TELEGRAM) ---
#
# Verbeteringen t.o.v. origineel:
# 1. Look-ahead bias opgelost
# 2. Backtest venster uitgebreid naar volledige 2 jaar
# 3. Entry-drempel in signaal consistent met backtest (RSI2 < 5)
# 4. Open positie aan einde van backtest wordt verrekend (mark-to-market)
# 5. Stop-loss toegevoegd (-8%)
# 6. Foutafhandeling verbeterd
# 7. Indicatoren berekend op volledige dataset
# 8. FIX: __future__ annotations voor Python < 3.10
# 9. NIEUW: Telegram notificaties toegevoegd

from __future__ import annotations

import logging
import os
import requests
import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TELEGRAM CONFIGURATIE
# Stel deze in als omgevingsvariabelen (GitHub Secrets / .env):
#   TELEGRAM_TOKEN   = "123456:ABCdef..."
#   TELEGRAM_CHAT_ID = "-100123456789"  (groep) of "123456789" (persoonlijk)
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def stuur_telegram(bericht: str) -> bool:
    """
    Verstuurt een bericht naar Telegram.
    Geeft True terug bij succes, False bij fout.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error(
            "Telegram TOKEN of CHAT_ID niet ingesteld! "
            "Controleer je omgevingsvariabelen (GitHub Secrets)."
        )
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": bericht,
        "parse_mode": "HTML",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Telegram bericht verstuurd: {bericht[:60]}...")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Telegram HTTP fout: {e} | Antwoord: {response.text}")
    except requests.exceptions.ConnectionError:
        logger.error("Telegram: geen internetverbinding.")
    except requests.exceptions.Timeout:
        logger.error("Telegram: timeout bij versturen.")
    except Exception as e:
        logger.error(f"Telegram: onverwachte fout – {e}")
    return False


# ---------------------------------------------------------------------------
# STRATEGIE
# ---------------------------------------------------------------------------
def bereken_mean_reversion_alpha(ticker: str, inzet: float) -> tuple[float, str | None]:
    """
    Mean reversion strategie gebaseerd op RSI(2) + prijsstretch + EMA200 trendfilter.

    Entry:
      - RSI(2) < 5 (extreem oververkocht)
      - Prijs > 5% onder MA(5) (de 'stretch')
      - EMA(200) is stijgend (opwaartse trend)

    Exit:
      - Prijs sluit boven MA(2)  → bounce voltooid
      - Prijs bereikt Upper Bollinger Band (MA20 + 2*std)
      - Stop-loss: prijs daalt > 8% onder instapkoers
    """
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)

        if df.empty or len(df) < 200:
            logger.warning(f"{ticker}: onvoldoende data ({len(df)} rijen)")
            return 0, None

        # Prijsserie ophalen (MultiIndex-safe)
        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df["Close"].columns:
                logger.warning(f"{ticker}: niet gevonden in MultiIndex kolommen")
                return 0, None
            p = df["Close"][ticker].dropna().astype(float)
        else:
            p = df["Close"].dropna().astype(float)

        if len(p) < 200:
            return 0, None

        # Indicatoren
        ma2        = p.rolling(window=2).mean()
        ma5        = p.rolling(window=5).mean()
        ma20       = p.rolling(window=20).mean()
        std20      = p.rolling(window=20).std()
        upper_band = ma20 + (2.0 * std20)

        ema200          = p.ewm(span=200, adjust=False).mean()
        ema200_stijgend = ema200.diff(5) > 0

        # RSI(2)
        delta = p.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(window=2).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(window=2).mean()
        rsi2  = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # Backtest
        WARMUP    = 200
        STOP_LOSS = 0.08
        kosten    = 15.0 + (inzet * 0.0035)

        profit = 0.0
        pos    = False
        instap = 0.0

        for i in range(WARMUP, len(p)):
            cp = p.iloc[i]

            if not pos:
                entry_conditie = (
                    rsi2.iloc[i]            < 5  and
                    cp                      < ma5.iloc[i] * 0.95 and
                    ema200_stijgend.iloc[i]
                )
                if entry_conditie:
                    instap  = cp
                    pos     = True
                    profit -= kosten
            else:
                stop_geraakt  = cp < instap * (1 - STOP_LOSS)
                bounce_klaar  = cp > ma2.iloc[i]
                upper_geraakt = cp > upper_band.iloc[i]

                if stop_geraakt or bounce_klaar or upper_geraakt:
                    profit += (inzet * (cp / instap)) - inzet - kosten
                    pos     = False

        # Open positie mark-to-market
        if pos:
            laatste_prijs = p.iloc[-1]
            profit += (inzet * (laatste_prijs / instap)) - inzet - kosten
            logger.info(f"{ticker}: open positie verrekend @ {laatste_prijs:.2f}")

        # Huidig signaal
        signaal = None
        if (
            rsi2.iloc[-1] < 5 and
            p.iloc[-1]    < ma5.iloc[-1] * 0.95 and
            ema200_stijgend.iloc[-1]
        ):
            signaal = (
                f"💥 <b>ALPHA BOUNCE</b> | <b>{ticker}</b>\n"
                f"💶 Koers:    €{p.iloc[-1]:.2f}\n"
                f"📉 RSI(2):   {rsi2.iloc[-1]:.1f}\n"
                f"🛑 Stop:     €{p.iloc[-1] * (1 - STOP_LOSS):.2f}\n"
                f"📊 Backtest: €{round(profit, 2):+.2f}"
            )

        return round(profit, 2), signaal

    except Exception as e:
        logger.error(f"{ticker}: fout – {e}", exc_info=True)
        return 0, None


# ---------------------------------------------------------------------------
# MAIN: scan tickers en stuur Telegram bij signaal
# ---------------------------------------------------------------------------
def main():
    # Pas deze lijst aan naar jouw gewenste tickers
    TICKERS = [
        "ASML.AS", "ADYEN.AS", "INGA.AS", "PHIA.AS", "HEIA.AS",
        "URW.AS",  "WKL.AS",  "NN.AS",   "RAND.AS", "BESI.AS",
        "TKWY.AS", "SBMO.AS", "AKZA.AS", "DSM.AS",  "KPN.AS",
    ]
    INZET = 1000.0  # euro per trade (voor backtest P&L berekening)

    logger.info(f"=== Bot gestart | {len(TICKERS)} tickers ===")
    signalen_gevonden = 0

    for ticker in TICKERS:
        logger.info(f"Analyseer: {ticker}")
        profit, signaal = bereken_mean_reversion_alpha(ticker, INZET)

        if signaal:
            signalen_gevonden += 1
            logger.info(f"SIGNAAL gevonden voor {ticker}")
            stuur_telegram(signaal)
        else:
            logger.info(f"{ticker}: geen signaal | Backtest P&L: €{profit:.2f}")

    # Altijd een samenvattingsbericht sturen zodat je weet dat de bot heeft gedraaid
    if signalen_gevonden == 0:
        stuur_telegram(
            f"🤖 <b>Power Reversion Alpha</b>\n"
            f"✅ Scan voltooid — geen signalen gevonden\n"
            f"📋 {len(TICKERS)} tickers geanalyseerd"
        )
    else:
        stuur_telegram(
            f"✅ <b>Scan klaar</b> | {signalen_gevonden} signaal(en) verstuurd"
        )

    logger.info(f"=== Bot klaar | {signalen_gevonden} signalen gevonden ===")


if __name__ == "__main__":
    main()
