# 📈 Multi-Stock 7-Day Price Prediction Dashboard

🔗 **Live Application:** https://stock-prediction-dashboard-jxjvdw5a6yzfprnhceobrp.streamlit.app/

📂 **GitHub Repository:** https://github.com/Joshua-2005/stock-prediction-dashboard  

---

## 📊 Project Overview

This project is a **machine learning-powered stock prediction dashboard** that forecasts **7-day future closing prices** for multiple stocks using a Random Forest regression model.

The application integrates real-time financial data, technical indicators, and interactive visualizations into a fully deployed web dashboard.

---

## 🚀 Features

- ✅ Multi-stock selection (Infosys, TCS, Reliance, HDFC Bank)
- ✅ 7-day price prediction using Random Forest
- ✅ Real-time data fetching via Yahoo Finance API
- ✅ RSI (Relative Strength Index) indicator
- ✅ Buy/Sell signal generation
- ✅ Model confidence estimation
- ✅ Interactive price and RSI charts
- ✅ Feature importance visualization
- ✅ Public deployment using Streamlit Cloud

---

## 🧠 Machine Learning Approach

### Model Used:
- **Random Forest Regressor**

### Target:
- Predict closing price 7 days into the future

### Feature Engineering:
- Daily Return
- 10-day Moving Average (MA_10)
- 50-day Moving Average (MA_50)
- 10-day Volatility
- High-Low difference
- Open-Close difference
- Volume
- Stock ID encoding (for multi-stock training)

### Technical Indicator:
- 14-period RSI (Relative Strength Index)

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Yahoo Finance API (yfinance)

---

## 📂 Project Structure

stock-prediction-dashboard/

├── app.py # Streamlit web application

├── train_model.py # Model training script

├── model.pkl # Trained machine learning model

├── requirements.txt # Dependencies for deployment

└── README.md # Project documentation


---

## 🌍 Deployment

The application is deployed using **Streamlit Community Cloud** and is publicly accessible.

Live link:  
👉 https://stock-prediction-dashboard-jxjvdw5a6yzfprnhceobrp.streamlit.app/ 

---

## ⚠ Disclaimer

This project is developed for **educational purposes only**.  
It does not provide financial advice or guarantee investment outcomes.  
Stock markets are influenced by many unpredictable factors.

---

## 💡 Future Improvements
Want to:
- Implement TimeSeriesSplit for more robust validation
- Add LSTM-based deep learning model
- Add Bollinger Bands and MACD indicators
- Improve model performance using hyperparameter tuning
- Add portfolio analysis and risk metrics

---

## 👨‍💻 Author

Joshua  
Electronics & Communication Engineering Student  
Passionate about Machine Learning, AI, and Financial Modeling
