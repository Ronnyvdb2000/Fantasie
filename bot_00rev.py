# ... (Config & Imports blijven gelijk) ...

# ---------------------------------------------------------------------------
# DYNAMISCHE PARAMETERS & FISCALITEIT
# ---------------------------------------------------------------------------
INZET          = 2500.0
BEURSTAKS_PCT  = 0.0035  # 0.35%
TOB_MAX        = 1600.0  # Plafond voor 0.35% tarief
MEERWAARDE     = 0.10    # 10% conservatieve schatting
BROKER_PCT     = 0.0035  # 0.35%
BROKER_VAST    = 15.0    # €15 vaste broker kost

def bereken_kosten_totaal(bedrag: float) -> float:
    tob = min(bedrag * BEURSTAKS_PCT, TOB_MAX)
    broker = BROKER_VAST + (bedrag * BROKER_PCT)
    return tob + broker

# ---------------------------------------------------------------------------
# PORTFOLIO UPDATE MET DYNAMISCHE STOP LOSS
# ---------------------------------------------------------------------------
def update_portfolio_en_rapport() -> str:
    portfolio = laad_portfolio()
    if not portfolio:
        return "📂 *PORTFOLIO* — Geen open posities."

    tickers = sorted(portfolio.keys())
    nu = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    try:
        # Haal data op voor indicatoren (ATR/MA)
        raw = yf.download(tickers, period="60d", progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"Portfolio download fout: {e}")
        return "⚠️ Portfolio update mislukt."

    regels, verkoop_tips = [], []
    t_kost, t_waarde = 0.0, 0.0

    for ticker in tickers:
        pos = portfolio[ticker]
        try:
            if len(tickers) > 1:
                df = raw.xs(ticker, axis=1, level=1).dropna(how='all')
            else:
                df = raw.dropna(how='all')

            cp = df['Close'].iloc[-1]
            # Bereken ATR voor dynamische SL (consistent met backtest)
            h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
            tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
            atr_nu = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
            
            ma5 = df['Close'].rolling(5).mean().iloc[-1]
            ma10 = df['Close'].rolling(10).mean().iloc[-1]

            # Hoogste koers sinds aankoop bijhouden voor Trailing Stop
            hi_since = max(pos.get('hi_since', cp), cp)
            portfolio[ticker]['hi_since'] = hi_since
            
            # SL Berekening: 2x ATR onder de hoogste koers (Trailing)
            sl_dynamisch = hi_since - (2 * atr_nu)
            
            p_koop = pos['prijs_koop']
            strat = pos['strategie']
            dagen = pos.get('aantal_dagen', 0) + 1
            portfolio[ticker]['aantal_dagen'] = dagen

            waarde_nu = pos['inzet'] * (cp / p_koop)
            pnl_pct = ((cp / p_koop) - 1) * 100
            
            t_kost += pos['inzet']
            t_waarde += waarde_nu

            # --- VERKOOPLOGICA (SYMBIOSE MET BACKTEST) ---
            verkoop, reden = False, ""
            
            if strat == "MRAS":
                if cp > ma5: verkoop, reden = True, "Boven MA5"
                elif cp > p_koop * MRA_SNEL_WINST: verkoop, reden = True, "Target +12%"
            elif strat == "MRAT":
                if dagen >= MRA_TRAAG_HOLD and cp > ma10: verkoop, reden = True, "Boven MA10"
                elif cp > p_koop * MRA_TRAAG_WINST: verkoop, reden = True, "Target +25%"
            else:
                # Trend strategieën: Gebruik de 2x ATR Trailing Stop
                if cp < sl_dynamisch: verkoop, reden = True, f"Trailing SL (€{sl_dynamisch:.2f})"

            pijl = "🟢" if pnl_pct >= 0 else "🔴"
            regels.append(
                f"• `{ticker}` [{strat}] | Koop: €{p_koop:.2f} | Nu: €{cp:.2f} | "
                f"{pijl} {pnl_pct:+.1f}% | Dagen: {dagen} | 🛡️ SL: €{sl_dynamisch:.2f}"
            )

            if verkoop:
                verkoop_tips.append(f"🔔 *VERKOOP* `{ticker}`: {reden} | Exit: ~€{cp:.2f}")

        except Exception as e:
            regels.append(f"• `{ticker}` — Fout: {e}")

    sla_portfolio_op(portfolio)
    
    # Rapport opbouw (gelijk aan je vorige versie, maar met dynamic t_pnl)
    t_pnl = t_waarde - t_kost
    summary = f"💰 *Totaal:* €{t_waarde:.0f} ({t_pnl:+.0f})"
    return "\n".join([f"📂 *PORTFOLIO* - {nu}", "---", *regels, "", summary, "", *verkoop_tips])

# ... (Rest van de Core Engine met bereken_winst inclusief TOB-check) ...
