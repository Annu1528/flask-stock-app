import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Price Prediction App", layout="wide")

# Sidebar inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())
interval = st.sidebar.selectbox("Prediction Interval", ["Hourly", "Daily", "Monthly"], index=1)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# Fetch stock data
data_load_state = st.text('Loading data...')
yf_interval = '1d'
if interval == "Hourly":
    yf_interval = '60m'
elif interval == "Monthly":
    yf_interval = '1mo'

try:
    df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=yf_interval, progress=False)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

data_load_state.text('')

if df is None or df.empty:
    st.error("No data found for the given ticker and date range.")
    st.stop()

# Ensure 'Close' column exists and is numeric
if 'Close' not in df.columns:
    st.error("Downloaded data has no 'Close' column.")
    st.stop()

df = df[['Close']].copy()
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Sort by Date just in case
df.sort_values('Date', inplace=True)

# Bar chart of closing prices with green/red bars
df['Color'] = np.where(df['Close'].diff().fillna(0) >= 0, 'green', 'red')
df.loc[df.index[0], 'Color'] = 'green'  # first entry default green

max_close = df['Close'].max()
min_close = df['Close'].min()
max_date = df.loc[df['Close'].idxmax(), 'Date']
min_date = df.loc[df['Close'].idxmin(), 'Date']

# Prepare data for peak/trough annotation
peak_df = pd.DataFrame({'Date': [max_date], 'Close': [max_close], 'Label': [f"Max: {max_close:.2f}"]})
trough_df = pd.DataFrame({'Date': [min_date], 'Close': [min_close], 'Label': [f"Min: {min_close:.2f}"]})

bars = alt.Chart(df).mark_bar().encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Close:Q', title='Close Price'),
    color=alt.Color('Color:N', scale=alt.Scale(domain=['green', 'red'], range=['green', 'red']), legend=None)
)

text_peak = alt.Chart(peak_df).mark_text(
    align='center',
    dy=-10,
    color='black'
).encode(
    x='Date:T',
    y='Close:Q',
    text='Label:N'
)

text_trough = alt.Chart(trough_df).mark_text(
    align='center',
    dy=15,
    color='black'
).encode(
    x='Date:T',
    y='Close:Q',
    text='Label:N'
)

price_chart = (bars + text_peak + text_trough).properties(
    width=700,
    height=400,
    title=f"{ticker} Closing Prices"
)
st.altair_chart(price_chart, use_container_width=True)

# Prepare data for model
df_model = df.copy()
df_model['X'] = np.arange(len(df_model))
X = df_model[['X']].values
y = df_model['Close'].values

# Train/test split (80% train, 20% test)
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) if len(X_test) > 0 else []

mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else 0
r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 1.0

# Model performance
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.2f}")

# Predict next price
next_X = np.array([[len(df_model)]])
pred_price = model.predict(next_X)[0]

# Determine next date for display
last_date = df_model['Date'].iloc[-1]
if interval == "Hourly":
    next_date = last_date + pd.DateOffset(hours=1)
elif interval == "Monthly":
    next_date = last_date + pd.DateOffset(months=1)
else:  # Daily
    next_date = last_date + pd.DateOffset(days=1)

st.subheader("Next Price Prediction")
st.write(f"Predicted next closing price for {ticker} on {next_date.date()} is **${pred_price:.2f}**.")

# Recent history table
st.subheader("Recent Price Changes")
table_df = df[['Date', 'Close']].copy()
table_df['Change'] = table_df['Close'].diff().round(2)
table_df['Pct Change'] = (table_df['Close'].pct_change() * 100).round(2)
table_df = table_df.tail(10).reset_index(drop=True)
st.dataframe(table_df.style.format({
    'Date': lambda t: t.strftime("%Y-%m-%d %H:%M") if hasattr(t, "strftime") else t,
    'Close': "{:.2f}",
    'Change': "{:.2f}",
    'Pct Change': "{:.2f}"
}))
