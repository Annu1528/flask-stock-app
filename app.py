import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import matplotlib.pyplot as plt

# --- Streamlit page config ---
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: Instructions & Inputs
st.sidebar.title("📋 Instructions")
st.sidebar.info(
    """
    - Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA).
    - Select the date range for historical data.
    - Choose prediction interval: Next Hour, Next Day, or Next Month.
    - View model performance and historical prices.
    - See prediction changes with positive/negative differences.
    """
)

# Search bar for ticker symbol with placeholder & uppercase
stock_symbol = st.sidebar.text_input("🔎 Enter Stock Ticker Symbol", value="AAPL").upper()

# Date selectors
start_date = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Prediction interval including monthly
prediction_interval = st.sidebar.selectbox(
    "Prediction Interval",
    options=["Next Hour", "Next Day", "Next Month"]
)

# Validate dates
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Title
st.title("📈 Stock Price Prediction App — Vibrant & Insightful")

# Download historical data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
if stock_data.empty:
    st.error(f"No data found for ticker '{stock_symbol}'. Please try another symbol.")
    st.stop()

stock_data.ffill(inplace=True)  # Fill missing data

# --- Plotting function ---
def plot_bar_with_extremes(stock_data, stock_symbol):
    close = stock_data['Close']
    dates = stock_data.index

    fig, ax = plt.subplots(figsize=(12, 5))

    diff = close.diff()
    colors = ['#2ECC71' if float(x) >= 0 else '#E74C3C' for x in diff.fillna(0)]  # Green if up, Red if down

    ax.bar(dates, close, color=colors, edgecolor='black')

    # Peak and lowest points
    peak_idx = close.idxmax()
    low_idx = close.idxmin()
    peak_val = close.max()
    low_val = close.min()

    ax.scatter(peak_idx, peak_val, color='blue', s=120, label='Peak (High)')
    ax.scatter(low_idx, low_val, color='orange', s=120, label='Lowest (Low)')

    ax.annotate(f'Peak: {peak_val:.2f}', (peak_idx, peak_val),
                textcoords="offset points", xytext=(0,10), ha='center', color='blue', weight='bold')
    ax.annotate(f'Lowest: {low_val:.2f}', (low_idx, low_val),
                textcoords="offset points", xytext=(0,-15), ha='center', color='orange', weight='bold')

    ax.set_title(f"📊 Closing Prices for {stock_symbol}", fontsize=16, weight='bold')
    ax.set_ylabel("Price ($)", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

# Show the bar chart with extremes
plot_bar_with_extremes(stock_data, stock_symbol)

# --- Data Scaling & Model Training ---

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features = stock_data[['Open', 'High', 'Low', 'Volume']]
features_scaled = feature_scaler.fit_transform(features)

target = stock_data[['Close']]
target_scaled = target_scaler.fit_transform(target)

X = features_scaled
y = target_scaled

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

# Show model performance with vibrant colors
st.markdown(
    f"""
    <div style="display:flex; gap:30px; font-size:18px; font-weight:bold;">
        <div style="color:#FF5733;">🔴 Mean Squared Error: {mse:.6f}</div>
        <div style="color:#2980B9;">🔵 R² Score: {r2:.4f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Prepare latest data for prediction ---
def get_latest_data(symbol, interval):
    if interval == "Next Hour":
        return yf.download(symbol, period='1d', interval='1h')
    elif interval == "Next Day":
        return yf.download(symbol, period='2d', interval='1d')
    elif interval == "Next Month":
        # Monthly data for last 3 months, take latest month for prediction
        return yf.download(symbol, period='3mo', interval='1mo')
    else:
        return pd.DataFrame()

latest_data = get_latest_data(stock_symbol, prediction_interval)

if latest_data.empty:
    st.error("No recent data available for prediction.")
    st.stop()

# Scale latest features and predict
latest_features = latest_data[['Open', 'High', 'Low', 'Volume']]
latest_scaled = feature_scaler.transform(latest_features)
future_price_scaled = model.predict(latest_scaled)
future_price = target_scaler.inverse_transform(future_price_scaled)

# Take last predicted value as forecast
predicted_price = float(future_price[-1][0])
predicted_price_rounded = round(predicted_price, 2)

# Calculate difference compared to last known close price
last_close_price = stock_data['Close'][-1]
price_diff = predicted_price - last_close_price
price_diff_str = f"+${abs(price_diff):.2f}" if price_diff >= 0 else f"-${abs(price_diff):.2f}"

# Display prediction with difference in vibrant style
st.markdown(
    f"""
    <h3 style="color:#8E44AD;">{prediction_interval} Stock Price Prediction for {stock_symbol}</h3>
    <h2 style="color:#27AE60;">Predicted Price: ${predicted_price_rounded}</h2>
    <h3 style="color:{'#27AE60' if price_diff >= 0 else '#C0392B'};">
        Change from last close: {price_diff_str}
    </h3>
    """,
    unsafe_allow_html=True,
)
