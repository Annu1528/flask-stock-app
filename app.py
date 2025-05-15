import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import matplotlib.pyplot as plt

# Sidebar instructions
st.sidebar.title("📋 Instructions")
st.sidebar.info(
    """
    - Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA).
    - Select the date range for training data.
    - Choose prediction interval: Next Hour, Next Day, or Next Month.
    - View model performance and historical prices.
    - Get the predicted stock price for the selected interval.
    """
)

# Inputs
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

prediction_interval = st.sidebar.selectbox(
    "Prediction Interval",
    options=["Next Hour", "Next Day", "Next Month"]
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

st.title("📈 Stock Price Prediction App")

# Download historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

if stock_data.empty:
    st.error(f"No data found for ticker symbol '{stock_symbol}'. Please try another.")
    st.stop()

stock_data.ffill(inplace=True)

# Plot bar chart with peaks and lows
def plot_bar_with_extremes(stock_data, stock_symbol):
    dates = stock_data.index
    close_prices = stock_data['Close']

    # Calculate daily difference and clean
    diff = close_prices.diff().fillna(0).astype(float)

    # Bar colors: green if price went up or same, red if down
    colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in diff]

    plt.figure(figsize=(14, 6))
    plt.bar(dates, close_prices, color=colors, width=0.8)

    max_price = close_prices.max()
    max_date = close_prices.idxmax()

    min_price = close_prices.min()
    min_date = close_prices.idxmin()

    plt.scatter(max_date, max_price, color='gold', s=180, label='Max Price')
    plt.scatter(min_date, min_price, color='blue', s=180, label='Min Price')

    plt.title(f"Closing Prices Bar Chart with Price Changes for {stock_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.legend()

    total_change = close_prices.iloc[-1] - close_prices.iloc[0]
    change_str = f"+${total_change:.2f}" if total_change >= 0 else f"-${abs(total_change):.2f}"
    plt.figtext(0.15, 0.85, f"Total Change: {change_str}", fontsize=14, 
                color='green' if total_change >= 0 else 'red')

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(plt)

plot_bar_with_extremes(stock_data, stock_symbol)

# Data preprocessing for model
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features = stock_data[['Open', 'High', 'Low', 'Volume']]
features_scaled = feature_scaler.fit_transform(features)

target = stock_data[['Close']]
target_scaled = target_scaler.fit_transform(target)

X = features_scaled
y = target_scaled

# Train/test split
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

st.subheader(f"Model Performance for {stock_symbol}")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", round(mse, 6))
col2.metric("R² Score", round(r2, 4))

# Latest data for prediction based on interval
if prediction_interval == "Next Hour":
    latest_data = yf.download(stock_symbol, period='1d', interval='1h')
elif prediction_interval == "Next Day":
    latest_data = yf.download(stock_symbol, period='2d', interval='1d')
else:  # Next Month
    latest_data = yf.download(stock_symbol, period='1mo', interval='1d')

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

# Show last history of predictions (hourly, daily, monthly)

def show_price_changes(history_data, interval_name):
    st.subheader(f"Price Changes - Last {interval_name}")
    close_prices = history_data['Close']
    changes = close_prices.diff().fillna(0)

    # Colors for change values
    colors = ['green' if x >= 0 else 'red' for x in changes]

    for i in range(len(close_prices)):
        price = close_prices.iloc[i]
        change = changes.iloc[i]
        change_sign = "+" if change >= 0 else ""
        st.markdown(f"**{close_prices.index[i].date()}**: ${price:.2f} ({change_sign}{change:.2f})", unsafe_allow_html=True)

# Show hourly/daily/monthly price changes based on user interval choice
if prediction_interval == "Next Hour":
    hist_data = yf.download(stock_symbol, period='5d', interval='1h')
    show_price_changes(hist_data, "Hourly")

elif prediction_interval == "Next Day":
    hist_data = yf.download(stock_symbol, period='1mo', interval='1d')
    show_price_changes(hist_data, "Daily")

else:  # Monthly
    hist_data = yf.download(stock_symbol, period='12mo', interval='1mo')
    show_price_changes(hist_data, "Monthly")

