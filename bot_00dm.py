@dataclass
class DMSignaal:
    ticker:          str
    price:           float
    score:           int           # 0-5
    > 0 else 0.0

        lb   = min(DM_CFG[" {rv:.1f}% (niet top {DM_CFG['top_pct']:.0f}%)" if not math.isnan(rv) else "geen omzetdata")

        # 2. Top 20% RS
        chk(in_rs20,
            f"RS={rs:.0f} — top {DM_CFG['top_pct']:.0f}% universe ({rs_pct:.0f}e percentiel)",
            f"RS={rs:.0f} — niet top {DM_CFG['top_pct']:.0f}% ({rs_pct:.0f}e percentiel)")

        # 3. Nabij of boven 52-weekse high
        chk(pct_from_high <= DM_CFG["high_proximity_pct"],
            f"{pct_from_high:.1f}% van 52w high ({h52w:.2f})",
            f"{pct_from_high:.1f}% onder 52w high (>{DM_CFG['high_proximity_pct']:.0f}%)")

        # 4. Uitbraak boven 52w high op volume
        chk(breakout and vol_ratio >= DM_CFG["breakout_vol_mult"],
            f"uitbraak boven {h52w:.2f} op {vol_ratio:.1f}× volume",
            f"geen uitbraak (vol={vol_ratio:.1f}×, high={h52w:.2f})" if not breakout
            else f"uitbraak maar volume zwak ({vol_ratio:.1f}×)")

        # 5. Boven MA50 (exit filter)
        chk(above_ma50,
            f"boven MA{DM_CFG['ma_exit']} ({ma50:.2f}) — geen exit",
            f"onder MA{DM_CFG['ma_exit']} ({ma50:.2f}) — EXIT SIGNAAL")

        if score < DM_CFG["min_score"]:
            return None

        # Ranking: pipeline kwaliteit + breakout wegen zwaarder
        total_score = (
            score * 10
            + (rs * 0.3 if in_rs20 else 0)
            + (rv * 0.1 if in_rev20 and not math.isnan(rv) else 0)
            + (15 if breakout else 0)
            + (5 if vol_ratio >= DM_CFG["breakout_vol_mult"] else 0)
        )

        return DMSignaal(
            ticker=ticker,
            price=round(current_price, 2),
            score=score,
            score_labels=labels,
            rev_growth=round(rv, 1) if not math.isnan(rv) else 0.0,
            rev_rank=round(rv_pct, 1),
            rs=round(rs, 1),
            rs_rank=round(rs_pct, 1),
            high52w=round(h52w, 2),
            pct_from_high=round(pct_from_high, 1),
            breakout=breakout,
            breakout_vol=round(vol_ratio, 2),
            ma50=round(ma50, 2) if not math.isnan(ma50) else 0.0,
            above_ma50=above_ma50,
            stop=round(stop, 2),
            total_score=round(total_score, 1),
        )

    except Exception as e:
        print(f"[WARN] {ticker}: fout — {e}")
        return None


# ============================================================
# EXIT SIGNALEN
# ============================================================

def check_exits(
    positions: Dict[str, Dict],
    df_ex:     pd.DataFrame,
) -> List[Dict]:
    """
    Controleert open posities op exit: slot onder MA50.
    """
    exits = []
    for ticker, pos in positions.items():
        g = df_ex[df_ex["Ticker"] == ticker].sort_values("Date")
        if g.empty:
            continue
        close = g["Close"]
        ma50  = close.rolling(DM_CFG["ma_exit"]).mean()
        c_now = safe_float(close.iloc[-1])
        m_now = safe_float(ma50.iloc[-1])
        if math.isnan(c_now) or math.isnan(m_now):
            continue
        if c_now < m_now:
            exits.append({
                "ticker": ticker,
                "price":  round(c_now, 2),
                "ma50":   round(m_now, 2),
                "reason": f"Slot {c_now:.2f} < MA{DM_CFG['ma_exit']} {m_now:.2f}",
                **pos,
            })
    return exits


# ============================================================
# TELEGRAM OUTPUT
# ============================================================

def _score_bar(score: int, max_score: int = 5) -> str:
    filled = "█" * score
    empty  = "░" * (max_score - score)
    return f"{filled}{empty} {score}/{max_score}"


def format_dm_per_exchange(
    exchange_name:    str,
    signalen:         List[DMSignaal],
    exits:            List[Dict],
    portfolio_waarde: float,
) -> Tuple[str, str]:
    nu = today_str()

    def koop_blok(sigs: List[DMSignaal]) -> str:
        if not sigs:
            return "_Geen kandidaten_"
        lines = []
        for s in sigs:
            lines.append(
                f"• `{s.ticker}` | Score: {_score_bar(s.score)} | EUR{s.price:.2f} | {_yahoo_link(s.ticker)}\n"
                f"  Omzet: +{s.rev_growth:.1f}% ({s.rev_rank:.0f}e pct) | "
                f"RS:{s.rs:.0f} ({s.rs_rank:.0f}e pct)\n"
                f"  52w high: EUR{s.high52w:.2f} ({s.pct_from_high:.1f}% eronder) | "
                f"Vol: {s.breakout_vol:.1f}×\n"
                + "\n".join(f"  {lbl}" for lbl in s.score_labels) + "\n"
                + sizing_tekst(s.ticker, s.price, s.stop, s.ma50, portfolio_waarde)
            )
        return "\n\n".join(lines)

    def exit_blok(ex: List[Dict]) -> str:
        if not ex:
            return "_Geen exit signalen_"
        lines = []
        for e in ex:
            lines.append(
                f"• `{e['ticker']}` 🔴 *VERKOOP*\n"
                f"  {e['reason']}\n"
                f"  Entry was: EUR{e.get('entry_price', 0):.2f} | "
                f"Commando: `/sell {e['ticker']} {e['price']:.2f}`"
            )
        return "\n\n".join(lines)

    top2   = signalen[:2]
    score5 = [s for s in signalen if s.score == 5]
    score4 = [s for s in signalen if s.score == 4]
    score3 = [s for s in signalen if s.score == 3]

    deel1 = "\n\n".join([
        f"🚀 *DUAL MOMENTUM — {exchange_name}*",
        f"_{nu} | Top 20% omzet × top 20% RS → uitbraak 52w high_",
        f"_Exit: slot onder MA{DM_CFG['ma_exit']}_",
        "─────────────────────────────",
        f"🏆 *TOP 2 HOOGSTE POTENTIEEL:*",
        koop_blok(top2) if top2 else "_Geen kandidaten vandaag_",
        "─────────────────────────────",
        f"🔴 *EXIT SIGNALEN (slot < MA{DM_CFG['ma_exit']}):*",
        exit_blok(exits),
        "─────────────────────────────",
        f"⭐ *PERFECTE SCORE (5/5):*",
        koop_blok(score5) if score5 else "_Geen_",
    ])

    deel2_parts = [
        f"🚀 *DUAL MOMENTUM — {exchange_name} (2/2)*",
        "",
        f"⚡ *STERK (4/5):*",
        koop_blok(score4) if score4 else "_Geen_",
        "",
        f"📊 *WATCHLIST (3/5):*",
        koop_blok(score3) if score3 else "_Geen_",
        "",
        "─────────────────────────────",
        f"📊 *SAMENVATTING:*",
        f"  Kandidaten (score≥{DM_CFG['min_score']}) : {len(signalen)}",
        f"  Uitbraken                 : {sum(s.breakout for s in signalen)}",
        f"  Exit signalen             : {len(exits)}",
        f"  Score 5/5                 : {len(score5)}",
        f"  Score 4/5                 : {len(score4)}",
        f"  Score 3/5                 : {len(score3)}",
        "",
        "⚙️ *DUAL MOMENTUM PARAMETERS:*",
        f"_Stap 1: top {DM_CFG['top_pct']:.0f}% omzetgroei YoY_",
        f"_Stap 2: top {DM_CFG['top_pct']:.0f}% relatieve sterkte (RS)_",
        f"_Stap 3: uitbraak boven 52w high op ≥{DM_CFG['breakout_vol_mult']}× volume_",
        f"_Stap 4: verkoop bij slot onder MA{DM_CFG['ma_exit']}_",
        f"_Hard stop: {DM_CFG['stop_pct']:.0f}% | Risico: 5% portfolio_",
    ]

    return deel1, "\n".join(deel2_parts)


# ============================================================
# LIVE ENGINE
# ============================================================

def run_live_engine():
    print(f"{'='*60}")
    print(f"DUAL MOMENTUM — LIVE  {today_str()}")
    print(f"{'='*60}")

    exchange_tickers: Dict[str, List[str]] = {}
    all_tickers: List[str] = []

    for ex_name, path in EXCHANGES.items():
        tlist = load_tickers_from_file(path)
        if tlist:
            exchange_tickers[ex_name] = tlist
            all_tickers.extend(tlist)
            print(f"  {ex_name}: {len(tlist)} tickers")

    all_tickers = sorted(set(all_tickers))
    if not all_tickers:
        print("[ERROR] Geen ticker bestanden gevonden.")
        return

    print(f"\nTotaal: {len(all_tickers)} unieke tickers")
    print("Koersdata downloaden (2 jaar)...")
    df = download_history(all_tickers, period="2y")
    if df.empty:
        print("[ERROR] Geen data.")
        return
    print(f"Data geladen: {df['Ticker'].nunique()} tickers")

    # ── Stap 1: Omzetgroei ophalen ───────────────────────────────
    print(f"\nStap 1: Omzetgroei ophalen ({len(all_tickers)} tickers)...")
    rev_growth = compute_revenue_growth(all_tickers)
    print(f"  Omzetgroei beschikbaar: {len(rev_growth)} tickers")

    # ── Stap 2: RS berekenen ─────────────────────────────────────
    print("Stap 2: RS ratings berekenen...")
    rs_ratings = compute_rs_ratings(df)
    print(f"  RS ratings: {len(rs_ratings)} tickers")

    # ── Top 20% filters ──────────────────────────────────────────
    rev_top20 = top_pct_filter(rev_growth, DM_CFG["top_pct"])
    rs_top20  = top_pct_filter(rs_ratings, DM_CFG["top_pct"])
    print(f"  Top {DM_CFG['top_pct']:.0f}% omzetgroei : {len(rev_top20)} tickers")
    print(f"  Top {DM_CFG['top_pct']:.0f}% RS         : {len(rs_top20)} tickers")

    # Overlap = kandidaten voor technische analyse
    kandidaten = sorted(set(rev_top20) & set(rs_top20))
    print(f"  Overlap (beide top 20%) : {len(kandidaten)} tickers")

    portfolio_waarde = START_CAPITAL

    for ex_name, tlist in exchange_tickers.items():
        print(f"\nAnalyseren: {ex_name}...")
        tlist_set = set(tlist)
        df_ex     = df[df["Ticker"].isin(tlist_set)].copy()

        # Kandidaten die ook in deze exchange zitten
        ex_kandidaten = [t for t in kandidaten if t in tlist_set]
        print(f"  Pipeline kandidaten in exchange: {len(ex_kandidaten)}")

        signalen: List[DMSignaal] = []
        for ticker, group in df_ex.groupby("Ticker", sort=False):
            sig = analyse_ticker(
                ticker, group, rs_ratings, rev_growth, rs_top20, rev_top20
            )
            if sig:
                signalen.append(sig)
                print(
                    f"  ✓ {ticker}: {sig.score}/5 | "
                    f"omzet+{sig.rev_growth:.1f}% | RS={sig.rs:.0f} | "
                    f"{'BREAKOUT' if sig.breakout else 'setup'}"
                )

        signalen.sort(key=lambda s: s.total_score, reverse=True)

        # Exit check op lege posities (geen live portfolio in deze versie)
        exits: List[Dict] = []

        print(f"  → {len(signalen)} Dual Momentum kandidaten")

        deel1, deel2 = format_dm_per_exchange(ex_name, signalen, exits, portfolio_waarde)
        send_telegram_message(deel1)
        time.sleep(1)
        send_telegram_message(deel2)

        if signalen:
            _log_csv(signalen, ex_name)

    print(f"\n{'='*60}")
    print("Klaar.")


# ============================================================
# CSV LOGGING
# ============================================================

def _log_csv(signalen: List[DMSignaal], exchange: str):
    fname  = f"dm_signalen_{exchange.split()[0]}_{today_str()}.csv"
    header = ["datum","exchange","ticker","score","price","rev_growth","rev_rank",
              "rs","rs_rank","high52w","pct_from_high","breakout","breakout_vol",
              "ma50","above_ma50","stop","total_score"]
    ensure_csv_header(fname, header)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in signalen:
            w.writerow([
                today_str(), exchange, s.ticker, s.score, s.price,
                s.rev_growth, s.rev_rank, s.rs, s.rs_rank,
                s.high52w, s.pct_from_high, s.breakout, s.breakout_vol,
                s.ma50, s.above_ma50, s.stop, s.total_score,
            ])
    print(f"  CSV: {fname}")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    print(f"{'='*60}")
    print(f"DUAL MOMENTUM BACKTEST  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"NB: omzetgroei backtest gebruikt technische proxy (12m prijsprestatie)")
    print(f"{'='*60}")

    all_tickers: List[str] = []
    for path in EXCHANGES.values():
        all_tickers.extend(load_tickers_from_file(path))
    all_tickers = sorted(set(all_tickers))

    if not all_tickers:
        print("[ERROR] Geen tickers gevonden.")
        return

    print(f"Tickers: {len(all_tickers)} | Data downloaden (5y)...")
    df = download_history(all_tickers, period="5y")
    if df.empty:
        print("[ERROR] Geen data.")
        return

    all_dates = sorted(df["Date"].dt.date.unique())
    rs_ratings = compute_rs_ratings(df)
    rs_top20   = top_pct_filter(rs_ratings, DM_CFG["top_pct"])

    # Proxy voor omzetgroei in backtest: 12m prijsprestatie top 20%
    perf_proxy: Dict[str, float] = {}
    for ticker, group in df.groupby("Ticker", sort=False):
        g = group.sort_values("Date")
        if len(g) < 252:
            continue
        c_now  = safe_float(g["Close"].iloc[-1])
        c_12m  = safe_float(g["Close"].iloc[-252])
        if c_12m > 0:
            perf_proxy[ticker] = (c_now - c_12m) / c_12m * 100
    rev_top20_proxy = top_pct_filter(perf_proxy, DM_CFG["top_pct"])

    cash      = START_CAPITAL
    positions: Dict[str, Dict] = {}
    trades:    List[Dict]      = []

    scan_dates = [d for d in all_dates if d.weekday() == 0]
    print(f"Scanmomenten: {len(scan_dates)} (wekelijks maandag)")

    for scan_date in scan_dates:
        df_hist = df[df["Date"] <= pd.Timestamp(scan_date)].copy()
        day_df  = df[df["Date"] == pd.Timestamp(scan_date)].copy()

        price_map: Dict[str, float] = {}
        for _, row in day_df.iterrows():
            t = row.get("Ticker")
            c = safe_float(row.get("Close"))
            if t and not math.isnan(c):
                price_map[t] = c

        # Exits: slot onder MA50
        for ticker, pos in list(positions.items()):
            pos["days"] += 1
            if ticker not in price_map:
                continue
            g     = df_hist[df_hist["Ticker"] == ticker].sort_values("Date")
            close = g["Close"]
            ma50  = safe_float(close.rolling(DM_CFG["ma_exit"]).mean().iloc[-1])
            c_now = price_map[ticker]
            reason = None

            if not math.isnan(ma50) and c_now < ma50:
                reason = f"Slot<MA{DM_CFG['ma_exit']} ({c_now:.2f}<{ma50:.2f})"
            elif c_now <= pos["stop"]:
                reason = f"Hard SL ({pos['stop']:.2f})"
            elif pos["days"] >= MAX_HOLD_DAYS:
                reason = f"Time ({pos['days']}d)"

            if reason:
                exit_slip = c_now * (1 - SLIPPAGE_PCT)
                gross     = exit_slip * pos["size"]
                cost      = trade_cost(gross)
                pnl       = gross - cost - (pos["entry_price"] * pos["size"] + pos["cost"])
                tax       = pnl * TAX_RATE if pnl > 0 else 0.0
                cash     += gross - cost - tax
                trades.append({
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   scan_date.isoformat(),
                    "ticker":      ticker,
                    "score":       pos["score"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  round(exit_slip, 4),
                    "size":        pos["size"],
                    "pnl":         round(pnl, 2),
                    "tax":         round(tax, 2),
                    "net":         round(pnl - tax, 2),
                    "reason":      reason,
                    "days":        pos["days"],
                })
                del positions[ticker]

        # Entries: pipeline filter + uitbraak
        for ticker, group in df_hist.groupby("Ticker", sort=False):
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            if ticker not in rs_top20 or ticker not in rev_top20_proxy:
                continue
            sig = analyse_ticker(
                ticker, group, rs_ratings, perf_proxy, rs_top20, rev_top20_proxy
            )
            if not sig or not sig.breakout:
                continue

            entry      = sig.price * (1 + SLIPPAGE_PCT)
            aandelen, _ = bereken_positie(cash, entry, sig.stop)
            if aandelen <= 0:
                continue
            investering = entry * aandelen + trade_cost(entry * aandelen)
            if investering > cash:
                continue

            cash -= investering
            positions[ticker] = {
                "entry_date":  scan_date,
                "entry_price": round(entry, 4),
                "size":        aandelen,
                "stop":        sig.stop,
                "score":       sig.score,
                "days":        0,
                "cost":        trade_cost(investering),
            }

    if trades:
        tdf = pd.DataFrame(trades)
        tdf.to_csv("dm_backtest_trades.csv", index=False)
        n    = len(tdf)
        nwin = (tdf["net"] > 0).sum()
        pf   = abs(tdf.loc[tdf["net"] > 0, "net"].sum()) / max(
               abs(tdf.loc[tdf["net"] <= 0, "net"].sum()), 1e-9)
        final_val = cash + sum(
            price_map.get(t, p["entry_price"]) * p["size"]
            for t, p in positions.items()
        )
        print(f"\n{'='*60}")
        print(f"Startkapitaal    : EUR{START_CAPITAL:>12,.2f}")
        print(f"Eindkapitaal     : EUR{final_val:>12,.2f}")
        print(f"Totaal rendement : {(final_val-START_CAPITAL)/START_CAPITAL*100:>+.1f}%")
        print(f"Trades           : {n} | Winnaars: {nwin} ({nwin/n*100:.1f}%)")
        print(f"Profit Factor    : {pf:.2f}")
        print(f"Belasting betaald: EUR{tdf['tax'].sum():,.2f}")
        print(f"Gem. houdduur    : {tdf['days'].mean():.1f} dagen")
        print(f"Opgeslagen       : dm_backtest_trades.csv")
        print(f"{'='*60}")
    else:
        print("Geen trades gegenereerd.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        run_backtest()
    else:
        run_live_engine()
