import yfinance as yf
import pandas as pd

# 1. Haal data op (bijv. de S&P 500 ETF 'SPY' of Apple 'AAPL')
ticker = "SPY"
data = yf.download(ticker, start="2023-01-01", end="2026-01-01")

# 2. Bereken het 'Momentum' (bijv. over de laatste 20 dagen)
data['Returns'] = data['Close'].pct_change()
data['Momentum'] = data['Close'].pct_change(periods=20)

# 3. Genereer een signaal
# Koop (1) als momentum positief is, verkoop (-1) als het negatief is
data['Signal'] = 0
data.loc[data['Momentum'] > 0, 'Signal'] = 1
data.loc[data['Momentum'] < 0, 'Signal'] = -1

# 4. Bereken het resultaat
data['Strategy_Return'] = data['Signal'].shift(1) * data['Returns']
cumulative_return = (1 + data['Strategy_Return']).cumprod()

print(f"Totaal rendement van de strategie: {cumulative_return.iloc[-1]:.2%}")
