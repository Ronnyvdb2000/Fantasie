# --- STRATEGIE 5: POWER REVERSION ALPHA V2 (DE "RUBBER BAND" EDIT) ---
def bereken_mean_reversion_alpha(ticker, inzet):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df.empty or len(df) < 200: return 0, None
        
        p = df['Close'][ticker].dropna().astype(float) if isinstance(df.columns, pd.MultiIndex) else df['Close']
        
        # Indicatoren
        ma20 = p.rolling(window=20).mean()
        ma5 = p.rolling(window=5).mean()
        ma2 = p.rolling(window=2).mean() # Voor de snelle exit
        std20 = p.rolling(window=20).std()
        upper_band = ma20 + (2.0 * std20)
        lower_band = ma20 - (2.0 * std20) 
        ema200 = p.ewm(span=200, adjust=False).mean()
        
        # Trend filter: EMA200 moet stijgend zijn
        ema200_stijgend = ema200.diff(5) > 0

        # RSI 2
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
        rsi2 = 100 - (100 / (1 + (gain / (loss + 1e-10))))

        p_bt = p.iloc[-252:]
        profit, pos, instap = 0, False, 0
        kosten = 15.0 + (inzet * 0.0035)

        for i in range(20, len(p_bt)):
            cp = p_bt.iloc[i]
            idx = i + (len(p) - 252)
            
            if not pos:
                # ENTRY: 
                # 1. RSI2 < 5 (Extreem, bijna dood-ervaring voor het aandeel)
                # 2. Prijs minstens 5% onder het 5-daags gemiddelde (De "Stretch")
                # 3. Boven stijgende EMA200
                if rsi2.iloc[idx] < 5 and cp < (ma5.iloc[idx] * 0.95) and ema200_stijgend.iloc[idx]:
                    instap, pos = cp, True
                    profit -= kosten
            else:
                # EXIT:
                # Verkoop zodra de koers boven het 2-daags gemiddelde sluit (Einde van de bounce)
                # OF als we de Upper Band raken
                if cp > ma2.iloc[idx] or cp > upper_band.iloc[idx]:
                    profit += (inzet * (cp / instap) - inzet) - kosten
                    pos = False
        
        signaal = None
        if rsi2.iloc[-1] < 10 and p.iloc[-1] < (ma5.iloc[-1] * 0.95) and ema200_stijgend.iloc[-1]:
            signaal = f"💥 ALPHA BOUNCE | €{p.iloc[-1]:.2f} | RSI2: {rsi2.iloc[-1]:.0f}"

        return profit, signaal
    except:
        return 0, None
