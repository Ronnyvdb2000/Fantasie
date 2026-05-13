import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- CONFIGURATIE ---
# Voeg hier de tickers toe die je wilt volgen
TICKERS = ["ASML.AS", "ADYEN.AS", "INGA.AS", "HEIA.AS", "KPN.AS", "UNA.AS"]

def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Berekening voor Wilder's Smoothing (RSI/ATR/ADX standaard)."""
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    
    alpha = 1 / period
    res = series.copy()
    # Eerste waarde is het simpel voortschrijdend gemiddelde
    res.iloc[period-1] = series.iloc[:period].mean()
    # Volgende waarden via de Wilder formule
    for i in range(period, len(series)):
        res.iloc[i] = res.iloc[i-1] * (1 - alpha) + series.iloc[i] * alpha
    
    res.iloc[:period-1] = np.nan
    return res

def download_history(tickers, days=365):
    """Downloadt koersdata en zorgt voor een schone DataFrame."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    
    all_data = []
    for t in tickers:
        try:
            # Download data
            df = yf.download(t, start=start_dt, end=end_dt, interval="1d", progress=False)
            
            if df.empty:
                print(f"Waarschuwing: Geen data voor {t}")
                continue
            
            # Reset index om 'Date' als kolom te krijgen
            df = df.reset_index()
            
            # Fix voor yfinance MultiIndex kolommen
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df["Ticker"] = t
            # Sorteer op datum om berekeningen correct te laten verlopen
            df = df.sort_values("Date")
            
            # Optioneel: Next_Open voor backtest-simulatie
            df["Next_Open"] = df["Open"].shift(-1)
            
            all_data.append(df)
            print(f"Data opgehaald voor: {t}")
        except Exception as e:
            print(f"Fout bij downloaden {t}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Voegt indicatoren toe zonder de Ticker kolom te verliezen."""
    
    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        # Werk op een kopie van de groep
        g = group.copy().reset_index(drop=True)
        
        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]

        # 1. Moving Averages
        g["MA20"]  = close.rolling(20).mean()
        g["MA50"]  = close.rolling(50).mean()
        g["MA200"] = close.rolling(200).mean()

        # 2. RSI 14
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = _wilder_smooth(gain, 14)
        avg_loss = _wilder_smooth(loss, 14)
        rs       = avg_gain / (avg_loss + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        # 3. ATR 14
        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"] = _wilder_smooth(tr, 14)

        # 4. IBS (Internal Bar Strength)
        g["IBS"] = (close - low) / (high - low + 1e-9)

        # 5. ADX 14
        up_move   = high.diff()
        down_move = (-low.diff())
        plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        s_plus_dm  = _wilder_smooth(pd.Series(plus_dm,  index=g.index), 14)
        s_minus_dm = _wilder_smooth(pd.Series(minus_dm, index=g.index), 14)
        s_tr       = _wilder_smooth(tr, 14)
        
        plus_di    = 100 * (s_plus_dm  / (s_tr + 1e-9))
        minus_di   = 100 * (s_minus_dm / (s_tr + 1e-9))
        dx         = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        g["ADX14"] = _wilder_smooth(dx, 14)

        return g

    # De FIX: include_groups=True zorgt dat 'Ticker' in de resultaten blijft
    return df.groupby("Ticker", group_keys=False).apply(_calc, include_groups=True)

def run_live_engine():
    """Start de engine en print de resultaten."""
    print("--- STARTING TRADING ENGINE ---")
    
    # 1. Haal data op
    df_raw = download_history(TICKERS)
    if df_raw.empty:
        print("Geen data beschikbaar om te analyseren.")
        return

    # 2. Bereken indicatoren
    df_results = add_indicators(df_raw)
    
    # 3. Filter op de meest recente rij per aandeel
    # We sorteren eerst op datum zodat de laatste dag onderaan staat
    latest_data = df_results.sort_values("Date").groupby("Ticker").last().reset_index()
    
    print(f"\nResultaten voor {datetime.now().strftime('%Y-%m-%d')}:")
    print("-" * 80)
    
    for _, row in latest_data.iterrows():
        ticker = row["Ticker"]
        price  = row["Close"]
        rsi    = row["RSI14"]
        ma200  = row["MA200"]
        
        # Simpele voorbeeld-logica voor een signaal
        signal = "GEEN"
        if rsi < 30 and price > ma200:
            signal = "BUY (Dip in uptrend)"
        elif rsi > 70:
            signal = "SELL (Overbought)"
            
        print(f"Aandeel: {ticker:<10} | Prijs: {price:>8.2f} | RSI: {rsi:>5.2f} | Signaal: {signal}")

if __name__ == "__main__":
    run_live_engine()
