import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import requests
import io
import datetime

# -----------------------------------------
#  SAFE DATA DOWNLOADER (NO CRASHES)
# -----------------------------------------

def download_ticker(ticker):
    """
    Probeert Yahoo CSV → als dat faalt → Stooq → als dat faalt → skip.
    Retourneert een DataFrame of None.
    """

    # --- 1) Yahoo CSV fallback (geen JSON, dus geen JSONDecodeError)
    yahoo_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=9999999999&interval=1d&events=history"

    try:
        r = requests.get(yahoo_url, timeout=10)
        if r.status_code == 200 and len(r.text) > 50:
            df = pd.read_csv(io.StringIO(r.text))
            df["Ticker"] = ticker
            return df
    except Exception:
        pass

    # --- 2) Stooq fallback
    stooq_url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

    try:
        r = requests.get(stooq_url, timeout=10)
        if r.status_code == 200 and len(r.text) > 50:
            df = pd.read_csv(io.StringIO(r.text))
            df["Ticker"] = ticker
            return df
    except Exception:
        pass

    print(f"⚠ Ticker overgeslagen: {ticker}")
    return None


# -----------------------------------------
#  LOAD ALL TICKERS
# -----------------------------------------

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "META"]

def load_all_data():
    all_data = []

    print("\nDownload data:\n")
    for t in tqdm(TICKERS):
        df = download_ticker(t)
        if df is not None:
            all_data.append(df)

    if len(all_data) == 0:
        raise Exception("Geen enkele ticker kon worden gedownload.")

    return pd.concat(all_data, ignore_index=True)


# -----------------------------------------
#  FEATURE ENGINEERING
# -----------------------------------------

def add_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df = df.dropna()
    return df


# -----------------------------------------
#  TRAIN MODEL
# -----------------------------------------

def train_model(df):
    features = ["Close", "MA10", "MA50", "Volatility"]
    X = df[features]
    y = df["Target"]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31
    )

    model.fit(X, y)
    return model


# -----------------------------------------
#  MAIN
# -----------------------------------------

def main():
    print("\nStart xLightGBM run")
    run_id = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("RUN_ID:", run_id)

    df = load_all_data()
    df = add_features(df)

    model = train_model(df)

    # Save output
    out_file = f"results/{run_id}_summary.txt"
    with open(out_file, "w") as f:
        f.write("Model training completed.\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Tickers used: {df['Ticker'].nunique()}\n")

    print("\n✔ Run voltooid zonder blokkades.")
    print("✔ Resultaten opgeslagen in:", out_file)


if __name__ == "__main__":
    main()
