"""
Debug-script om te zien wat yfinance 1.7.0 teruggeeft.
Run lokaal of via een tijdelijke workflow-stap.
"""
import yfinance as yf
import pandas as pd

print("=" * 60)
print(f"pandas versie : {pd.__version__}")
print(f"yfinance versie: {yf.__version__}")
print("=" * 60)

ticker = "AAPL"

# --- Test 1: enkele ticker, defaults ---
print("\n[TEST 1] yf.download('AAPL') — defaults")
df1 = yf.download(ticker, period="1mo", progress=False)
print(f"  type(df)        : {type(df)}")
print(f"  df.empty        : {df1.empty}")
print(f"  type(df.columns): {type(df1.columns)}")
print(f"  df.columns      : {list(df1.columns)}")
print(f"  'Close' in cols : {'Close' in df1.columns}")
if not df1.empty:
    print(f"  shape           : {df1.shape}")
    print(f"  head:\n{df1.head(2)}")

# --- Test 2: auto_adjust=False ---
print("\n[TEST 2] yf.download('AAPL', auto_adjust=False)")
df2 = yf.download(ticker, period="1mo", auto_adjust=False, progress=False)
print(f"  type(df.columns): {type(df2.columns)}")
print(f"  df.columns      : {list(df2.columns)}")
print(f"  'Close' in cols : {'Close' in df2.columns}")

# --- Test 3: via Ticker.history() ---
print("\n[TEST 3] yf.Ticker('AAPL').history(period='1mo')")
df3 = yf.Ticker(ticker).history(period="1mo")
print(f"  type(df.columns): {type(df3.columns)}")
print(f"  df.columns      : {list(df3.columns)}")
print(f"  'Close' in cols : {'Close' in df3.columns}")

# --- Test 4: multi-ticker (als je dat ergens doet) ---
print("\n[TEST 4] yf.download(['AAPL','MSFT']) — multi-ticker")
df4 = yf.download(["AAPL", "MSFT"], period="1mo", progress=False)
print(f"  type(df.columns): {type(df4.columns)}")
print(f"  df.columns      : {list(df4.columns)[:6]} ...")
print(f"  is MultiIndex   : {isinstance(df4.columns, pd.MultiIndex)}")
if isinstance(df4.columns, pd.MultiIndex):
    print(f"  level 0         : {df4.columns.get_level_values(0).unique().tolist()}")
    print(f"  level 1         : {df4.columns.get_level_values(1).unique().tolist()}")

print("\n" + "=" * 60)
print("KLAAR — vergelijk de kolomnamen hierboven met wat jouw")
print("code verwacht (waarschijnlijk df['Close']).")
print("=" * 60)
