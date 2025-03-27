import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    return scaled_data, scaler

def arima_forecast(df):
    model = ARIMA(df['price'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def lstm_forecast(df):
    seq_length = 60  # Using past 60 days of data
    data, scaler = preprocess_data(df)
    
    # Train-Test Split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Prepare sequences
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)
    
    # Reshape for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM Model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Forecast future prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

if __name__ == "__main__":
    df = pd.read_csv('crypto_data.csv', index_col=0, parse_dates=True)
    forecast = lstm_forecast(df)
    print(forecast)
