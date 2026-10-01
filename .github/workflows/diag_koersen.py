#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_koersen.py — vergelijkt yfinance-koersen met en zonder auto_adjust.

Gebruik: python diag_koersen.py MOL.WA AAPL
Toont per ticker de laatste 40 handelsdagen: aangepaste koers, ruwe koers,
dividend, en telt NaN/negatieve waarden.
"""
import sys

import pandas as pd
import yfinance as yf

START = "2026-08-01"
tickers = sys.argv[1:] or ["MOL.WA", "AAPL"]

for t in tickers:
    print("=" * 78)
    print(t)
    tk = yf.Ticker(t)
    h_adj = tk.history(start=START, auto_adjust=True)
    h_raw = tk.history(start=START, auto_adjust=False)

    df = pd.DataFrame({
        "close_adjust": h_adj["Close"],
        "close_raw": h_raw["Close"],
        "dividend": h_raw["Dividends"],
    })
    df.index = pd.to_datetime(df.index).tz_localize(None)

    print(df.tail(40).to_string())
    print()
    print("rijen:", len(df))
    print("NaN  close_adjust:", int(df["close_adjust"].isna().sum()),
          "| close_raw:", int(df["close_raw"].isna().sum()))
    print("<= 0 close_adjust:", int((df["close_adjust"] <= 0).sum()),
          "| close_raw:", int((df["close_raw"] <= 0).sum()))
    print("dividenden in venster:")
    print(df[df["dividend"] > 0][["close_raw", "dividend"]].to_string())
