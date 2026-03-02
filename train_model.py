import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Stocks to train on
stocks = {
    "INFY.NS": 0,
    "TCS.NS": 1,
    "RELIANCE.NS": 2,
    "HDFCBANK.NS": 3
}

all_data = []

print("Downloading stock data...")

for ticker, stock_id in stocks.items():
    print(f"Processing {ticker}...")

    data = yf.download(ticker, start="2015-01-01")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Feature Engineering
    data['Return'] = data['Close'].pct_change()
    data['MA_10'] = data['Close'].rolling(10).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['Volatility'] = data['Close'].rolling(10).std()
    data['High_Low'] = data['High'] - data['Low']
    data['Open_Close'] = data['Open'] - data['Close']

    # Predict 7 days ahead
    data['Target'] = data['Close'].shift(-7)

    # Add Stock ID
    data['Stock_ID'] = stock_id

    data = data.dropna()

    all_data.append(data)

# Combine all stocks
combined_data = pd.concat(all_data, ignore_index=True)

print("Total combined data shape:", combined_data.shape)

# Features
X = combined_data[[
    'Return',
    'MA_10',
    'MA_50',
    'Volatility',
    'High_Low',
    'Open_Close',
    'Volume',
    'Stock_ID'
]]

y = combined_data['Target']

# Train model
print("Training model...")

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")