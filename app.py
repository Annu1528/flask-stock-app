import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Stock Price Prediction")

# Get stock data
stock_data = yf.download('AAPL', start='2010-01-01', end='2025-04-23')
stock_data.ffill(inplace=True)

# Display stock data
st.subheader("Stock Data (First 5 rows)")
st.write(stock_data.head())

# Scaling the 'Close' price for the model
scaler = MinMaxScaler()
stock_data['Close'] = scaler.fit_transform(stock_data[['Close']])

# Feature engineering for training the model
X = stock_data[['Open', 'High', 'Low', 'Volume']].values
y = stock_data['Close'].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_test_actual = scaler.inverse_transform(y_test)
y_pred_actual = scaler.inverse_transform(y_pred)

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

# Display metrics
st.subheader("Model Metrics")
st.write(f"Mean Squared Error: {round(mse, 6)}")
st.write(f"R2 Score: {round(r2, 4)}")

# Predict next hour's stock price
latest_data = yf.download('AAPL', period='1d', interval='1h')
latest_scaled = scaler.transform(latest_data[['Close']])
future_price = model.predict(latest_scaled)
future_price = scaler.inverse_transform(future_price)

# Display next hour prediction
st.subheader("Next Hour Price Prediction")
st.write(f"Predicted Next Hour Stock Price: {round(float(future_price[-1][0]), 2)}")
