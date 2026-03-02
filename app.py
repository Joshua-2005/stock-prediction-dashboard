import streamlit as st
import pickle
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="7-Day Stock Prediction Dashboard", layout="wide")

# ---------------- HEADER ----------------
st.title("📈 7-Day Stock Price Prediction Dashboard")

st.markdown("""
### 📊 About This Dashboard
This application predicts **7-day future stock prices** using a Random Forest machine learning model 
trained on historical market data with technical indicators.

**Features included:**
- Multi-stock selection
- RSI (Relative Strength Index)
- Buy/Sell signal
- Model confidence estimation
- Feature importance visualization
""")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

stock_options = {
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

stock_id_map = {
    "INFY.NS": 0,
    "TCS.NS": 1,
    "RELIANCE.NS": 2,
    "HDFCBANK.NS": 3
}

selected_stock_name = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
ticker = stock_options[selected_stock_name]

period_option = st.sidebar.selectbox(
    "Select Date Range",
    ["6mo", "1y", "2y", "5y"]
)

# ---------------- MAIN BUTTON ----------------
if st.button("Predict Stock"):

    data = yf.download(ticker, period=period_option)

    if data.empty:
        st.error("Failed to fetch stock data.")
        st.stop()

    # Fix MultiIndex issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # ---------------- RSI Calculation ----------------
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ---------------- Feature Engineering ----------------
    data['Return'] = data['Close'].pct_change()
    data['MA_10'] = data['Close'].rolling(10).mean()
    data['MA_50'] = data['Close'].rolling(50).mean()
    data['Volatility'] = data['Close'].rolling(10).std()
    data['High_Low'] = data['High'] - data['Low']
    data['Open_Close'] = data['Open'] - data['Close']

    data = data.dropna()

    if len(data) == 0:
        st.error("Not enough data after processing.")
        st.stop()

    latest = data.iloc[-1]

    # ---------------- Prepare Features ----------------
    features = np.array([
        float(latest['Return']),
        float(latest['MA_10']),
        float(latest['MA_50']),
        float(latest['Volatility']),
        float(latest['High_Low']),
        float(latest['Open_Close']),
        float(latest['Volume']),
        stock_id_map[ticker]
    ]).reshape(1, -1)

    # ---------------- Prediction ----------------
    prediction = model.predict(features)[0]
    current_price = float(latest['Close'])

    # ---------------- Model Confidence ----------------
    tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
    confidence = 100 - (np.std(tree_preds) / current_price * 100)

    # ---------------- Buy/Sell Signal ----------------
    if prediction > current_price:
        signal = "📈 BUY"
        signal_color = "green"
    else:
        signal = "📉 SELL"
        signal_color = "red"

    # ---------------- Metrics Layout ----------------
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("Predicted (7 Days)", f"₹{prediction:.2f}")
    col3.metric("Model Confidence", f"{confidence:.2f}%")

    st.markdown(f"<h2 style='color:{signal_color};'>{signal}</h2>", unsafe_allow_html=True)

    # ---------------- Price Chart ----------------
    st.subheader("📊 Price Chart")
    st.line_chart(data.set_index('Date')['Close'])

    # ---------------- RSI Chart ----------------
    st.subheader("📉 RSI Indicator")
    st.line_chart(data.set_index('Date')['RSI'])

    latest_rsi = latest['RSI']
    if latest_rsi > 70:
        st.warning("RSI indicates Overbought condition")
    elif latest_rsi < 30:
        st.warning("RSI indicates Oversold condition")
    else:
        st.info("RSI in Neutral Zone")

    # ---------------- Feature Importance ----------------
    st.subheader("🔎 Feature Importance")

    importances = model.feature_importances_
    feature_names = [
        'Return', 'MA_10', 'MA_50',
        'Volatility', 'High_Low',
        'Open_Close', 'Volume', 'Stock_ID'
    ]

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("⚠️ This project is for educational purposes only. It does not provide financial advice.")