import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Sidebar for inputs and instructions
st.sidebar.title("📋 Instructions")
st.sidebar.info(
    """
    - Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA).
    - Select the date range for training data.
    - Choose prediction interval: Next Hour or Next Day.
    - View model performance and historical prices.
    - Get the predicted stock price for the selected interval.
    """
)

stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

prediction_interval = st.sidebar.selectbox(
    "Prediction Interval",
    options=["Next Hour", "Next Day"]
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

st.title("📈 Stock Price Prediction App")

# Download stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

if stock_data.empty:
    st.error(f"No data found for ticker symbol '{stock_symbol}'. Please try another.")
    st.stop()

stock_data.ffill(inplace=True)

# Show historical close price chart
st.subheader(f"📊 Historical Closing Prices for {stock_symbol}")
st.line_chart(stock_data['Close'])

# Scale data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features = stock_data[['Open', 'High', 'Low', 'Volume']]
features_scaled = feature_scaler.fit_transform(features)

target = stock_data[['Close']]
target_scaled = target_scaler.fit_transform(target)

X = features_scaled
y = target_scaled

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred_scaled = model.predict(X_test)
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)

# Metrics
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

# Display model performance nicely
st.subheader(f"Model Performance for {stock_symbol}")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", round(mse, 6))
col2.metric("R² Score", round(r2, 4))

# Latest data download based on prediction interval
if prediction_interval == "Next Hour":
    latest_data = yf.download(stock_symbol, period='1d', interval='1h')
elif prediction_interval == "Next Day":
    latest_data = yf.download(stock_symbol, period='2d', interval='1d')

if latest_data.empty:
    st.error("No recent data to make prediction.")
    st.stop()

latest_features = latest_data[['Open', 'High', 'Low', 'Volume']]
latest_scaled = feature_scaler.transform(latest_features)

future_price_scaled = model.predict(latest_scaled)
future_price = target_scaler.inverse_transform(future_price_scaled)

predicted_price_val = round(float(future_price[-1][0]), 2)

st.subheader(f"{prediction_interval} Stock Price Prediction for {stock_symbol}")
st.metric(label="Predicted Price", value=f"${predicted_price_val}")
