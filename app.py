import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

# -------------------- UTILITY FUNCTIONS -------------------- #
def load_data(symbol, start, end):
    try:
        with st.spinner("Downloading stock data..."):
            data = yf.download(symbol, start=start, end=end)
            data.ffill(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def preprocess_data(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data[['Close']]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(features)
    y_scaled = target_scaler.fit_transform(target)

    return X_scaled, y_scaled, feature_scaler, target_scaler

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

def predict_next_price(model, latest_data, feature_scaler, target_scaler):
    try:
        latest_features = latest_data[['Open', 'High', 'Low', 'Volume']]
        latest_scaled = feature_scaler.transform(latest_features)
        latest_point = latest_scaled[-1].reshape(1, -1)
        future_scaled = model.predict(latest_point)
        future_price = target_scaler.inverse_transform(future_scaled)
        return round(float(future_price[0][0]), 2)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def display_forecast_plot(stock_data, predicted_price):
    fig, ax = plt.subplots()
    ax.plot(stock_data.index, stock_data['Close'], label='Historical Close')
    ax.axhline(predicted_price, color='red', linestyle='--', label='Predicted Price')
    ax.set_title("Historical vs Predicted Price")
    ax.legend()
    st.pyplot(fig)

def show_prediction_history(symbol):
    st.subheader("🕒 Prediction History")
    periods = {
        "Hourly (Past 24h)": ('1d', '1h'),
        "Daily (Past 7d)": ('7d', '1d'),
        "Weekly (Past 1mo)": ('1mo', '1wk')
    }
    for label, (period, interval) in periods.items():
        hist_data = yf.download(symbol, period=period, interval=interval)
        if not hist_data.empty:
            st.write(f"**{label} Close Prices**")
            st.line_chart(hist_data['Close'])


# -------------------- SIDEBAR INPUT -------------------- #
st.sidebar.title("📋 Instructions")
st.sidebar.info("""
- Select a company.
- Pick a date range.
- Choose prediction interval.
- View historical prices and predictions.
""")

stock_options = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA"
}
selected_company = st.sidebar.selectbox("Select Company", options=list(stock_options.keys()))
stock_symbol = stock_options[selected_company]

start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

prediction_interval = st.sidebar.selectbox(
    "Prediction Interval",
    options=["Next Hour", "Next Day"]
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# -------------------- MAIN SECTION -------------------- #
st.title("📈 Stock Price Prediction App")

stock_data = load_data(stock_symbol, start_date, end_date)
if stock_data.empty:
    st.stop()

st.subheader(f"📊 Historical Closing Prices for {stock_symbol}")
st.line_chart(stock_data['Close'])

X_scaled, y_scaled, feature_scaler, target_scaler = preprocess_data(stock_data)
model, X_test, y_test, y_pred_scaled = train_model(X_scaled, y_scaled)

# Metrics
y_test_actual = target_scaler.inverse_transform(y_test)
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled)
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

st.subheader(f"📌 Model Performance for {stock_symbol}")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", round(mse, 6))
col2.metric("R² Score", round(r2, 4))

# Prediction
if prediction_interval == "Next Hour":
    recent_data = yf.download(stock_symbol, period='1d', interval='1h')
else:
    recent_data = yf.download(stock_symbol, period='2d', interval='1d')

if recent_data.empty:
    st.error("No recent data found for prediction.")
    st.stop()

predicted_price = predict_next_price(model, recent_data, feature_scaler, target_scaler)

if predicted_price:
    st.subheader(f"🔮 Predicted {prediction_interval} Price for {stock_symbol}")
    st.metric(label="Predicted Price", value=f"${predicted_price}")
    display_forecast_plot(stock_data, predicted_price)
    st.success("Prediction complete!")

# Show prediction history
show_prediction_history(stock_symbol)
