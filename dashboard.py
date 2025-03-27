import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from time_series_forecasting import arima_forecast
from lstm_forecasting import lstm_forecast
from sklearn.metrics import mean_squared_error

# Function to fetch real-time cryptocurrency prices
def fetch_live_price(crypto='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd'
    response = requests.get(url).json()
    return response[crypto]['usd']

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df['SMA_20'] = df['price'].rolling(window=20).mean()  # Simple Moving Average
    df['SMA_50'] = df['price'].rolling(window=50).mean()
    df['RSI'] = 100 - (100 / (1 + df['price'].pct_change().rolling(14).mean()))  # RSI
    return df

# Streamlit App Layout
st.set_page_config(layout="wide")
st.title("📊 Cryptocurrency Price Analysis Dashboard")

# User selects the cryptocurrency
crypto = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "dogecoin"])

# Fetch & display live price
live_price = fetch_live_price(crypto)
st.sidebar.metric(label=f"Live {crypto.capitalize()} Price (USD)", value=f"${live_price}")

# Load data
df = pd.read_csv('crypto_data.csv', index_col=0, parse_dates=True)
df = calculate_technical_indicators(df)

# Plot historical price trends
st.subheader("📈 Historical Price Trend")
fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=df.index, y=df['price'], label="Price", ax=ax)
sns.lineplot(x=df.index, y=df['SMA_20'], label="SMA 20", ax=ax)
sns.lineplot(x=df.index, y=df['SMA_50'], label="SMA 50", ax=ax)
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# Forecasting Options
forecast_method = st.sidebar.radio("Select Forecasting Model", ["ARIMA", "LSTM"])

st.subheader("🔮 Price Forecasting")
if forecast_method == "ARIMA":
    forecast = arima_forecast(df)
    st.line_chart(forecast)
elif forecast_method == "LSTM":
    forecast = lstm_forecast(df)
    st.line_chart(forecast)

# Performance Metrics
st.subheader("📊 Model Performance")
actual_prices = df['price'].values[-len(forecast):]  # Get actual values
rmse = np.sqrt(mean_squared_error(actual_prices, forecast))
st.write(f"🔹 **Root Mean Squared Error (RMSE):** {rmse:.2f}")

# Show RSI values
st.subheader("📊 RSI (Relative Strength Index)")
st.line_chart(df['RSI'])

st.sidebar.write("📌 *Technical indicators help traders make better decisions.*")

st.success("✅ Dashboard Updated with LSTM & Advanced Features!")
