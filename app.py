import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Stock Price Prediction App")

# Download stock data
stock_data = yf.download('AAPL', start='2010-01-01', end='2025-04-23')
stock_data.ffill(inplace=True)

# Separate scalers for features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale features
features = stock_data[['Open', 'High', 'Low', 'Volume']]
features_scaled = feature_scaler.fit_transform(features)

# Scale target
target = stock_data[['Close']]
target_scaled = target_scaler.fit_transform(target)

# Define X and y
X = features_scaled
y = target_scaled

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Inverse transform target and predictions
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

# Show metrics
st.subheader("Model Performance:")
st.write(f"Mean Squared Error: {round(mse, 6)}")
st.write(f"R-squared Score: {round(r2, 4)}")

# Get latest data for prediction (features only)
latest_data = yf.download('AAPL', period='1d', interval='1h')
latest_features = latest_data[['Open', 'High', 'Low', 'Volume']]

# Scale latest features with the same feature scaler
latest_scaled = feature_scaler.transform(latest_features)

# Predict future price (scaled)
future_price_scaled = model.predict(latest_scaled)

# Inverse transform predicted price to original scale
future_price = target_scaler.inverse_transform(future_price_scaled)

# Display prediction
st.subheader("Next Hour's Stock Price Prediction:")
st.write(f"Predicted Price: ${round(float(future_price[-1][0]), 2)}")
