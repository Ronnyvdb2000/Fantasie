# --- STRATEGIE 5: POWER REVERSION ALPHA V2 (VERBETERDE VERSIE) ---
#
# Verbeteringen t.o.v. origineel:
# 1. Look-ahead bias opgelost: geen handmatige index-offset meer
# 2. Backtest venster uitgebreid naar volledige 2 jaar
# 3. Entry-drempel in signaal consistent met backtest (RSI2 < 5)
# 4. Open positie aan einde van backtest wordt verrekend (mark-to-market)
# 5. Stop-loss toegevoegd (-8%)
# 6. Foutafhandeling verbeterd: specifieke Exception logging
# 7. Indicatoren berekend op volledige dataset, backtest indexeert correct
# 8. FIX: 'from __future__ import annotations' toegevoegd voor Python 3.7/3.8/3.9 compatibiliteit

from __future__ import annotations  # FIX: maakt str | None en tuple[...] syntax compatibel met Python < 3.10

import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbool (bijv. "ASML.AS")
    inzet  : float
        Bedrag in euro's per trade

    Returns
    -------
    profit : float
        Gesimuleerde winst/verlies over de backtest-periode
    signaal : str | None
        Huidig koopsignaal als aan alle entry-condities is voldaan, anders None
    """
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)

        if df.empty or len(df) < 200:
            logger.warning(f"{ticker}: onvoldoende data ({len(df)} rijen)")
            return 0, None

        # --- Prijsserie ophalen (MultiIndex-safe) ---
        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df["Close"].columns:
                logger.warning(f"{ticker}: ticker niet gevonden in MultiIndex kolommen")
                return 0, None
            p = df["Close"][ticker].dropna().astype(float)
        else:
            p = df["Close"].dropna().astype(float)

        if len(p) < 200:
            return 0, None

        # --- Indicatoren berekend op volledige reeks ---
        ma2        = p.rolling(window=2).mean()
        ma5        = p.rolling(window=5).mean()
        ma20       = p.rolling(window=20).mean()
        std20      = p.rolling(window=20).std()
        upper_band = ma20 + (2.0 * std20)

        ema200          = p.ewm(span=200, adjust=False).mean()
        ema200_stijgend = ema200.diff(5) > 0  # stijgend over laatste 5 dagen

        # RSI(2)
        delta = p.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(window=2).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(window=2).mean()
        rsi2  = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # --- Backtest over volledige beschikbare periode ---
        # Begin pas na de opwarmperiode van 200 dagen
        WARMUP    = 200
        STOP_LOSS = 0.08  # 8% stop-loss
        kosten    = 15.0 + (inzet * 0.0035)

        profit = 0.0
        pos    = False
        instap = 0.0

        for i in range(WARMUP, len(p)):
            cp = p.iloc[i]

            if not pos:
                # --- ENTRY ---
                entry_conditie = (
                    rsi2.iloc[i]            < 5   and
                    cp                      < ma5.iloc[i] * 0.95 and
                    ema200_stijgend.iloc[i]
                )
                if entry_conditie:
                    instap  = cp
                    pos     = True
                    profit -= kosten

            else:
                # --- EXIT ---
                stop_geraakt  = cp < instap * (1 - STOP_LOSS)
                bounce_klaar  = cp > ma2.iloc[i]
                upper_geraakt = cp > upper_band.iloc[i]

                if stop_geraakt or bounce_klaar or upper_geraakt:
                    trade_pnl = (inzet * (cp / instap)) - inzet - kosten
                    profit   += trade_pnl
                    pos       = False

        # --- Open positie aan einde: mark-to-market verrekenen ---
        if pos:
            laatste_prijs = p.iloc[-1]
            trade_pnl     = (inzet * (laatste_prijs / instap)) - inzet - kosten
            profit       += trade_pnl
            logger.info(f"{ticker}: open positie mark-to-market verrekend @ {laatste_prijs:.2f}")

        # --- Huidig signaal (zelfde drempel als backtest: RSI2 < 5) ---
        signaal = None
        huidig_signaal = (
            rsi2.iloc[-1] < 5 and
            p.iloc[-1]    < ma5.iloc[-1] * 0.95 and
            ema200_stijgend.iloc[-1]
        )
        if huidig_signaal:
            signaal = (
                f"💥 ALPHA BOUNCE | €{p.iloc[-1]:.2f} | "
                f"RSI2: {rsi2.iloc[-1]:.0f} | "
                f"Stop: €{p.iloc[-1] * (1 - STOP_LOSS):.2f}"
            )

        return round(profit, 2), signaal

    except Exception as e:
        logger.error(f"{ticker}: fout tijdens berekening – {e}", exc_info=True)
        return 0, None
