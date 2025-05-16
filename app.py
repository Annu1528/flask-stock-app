import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from scipy.signal import find_peaks
from fbprophet import Prophet
import datetime

# Set page configuration
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# Title and description
st.title("📈 Stock Market Prediction with Peak-Trough Analysis")
st.markdown("""
    This application allows you to analyze stock price trends by identifying significant peaks and troughs,
    and forecast future prices using the Facebook Prophet model.
""")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch stock data
@st.cache
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = load_data(ticker, start_date, end_date)

# Display stock data
st.subheader(f"Stock Data for {ticker} ({start_date} to {end_date})")
st.write(data.tail())

# Plot stock price with peaks and troughs
st.subheader("Stock Price with Peaks and Troughs")
prices = data['Close'].values
peaks, _ = find_peaks(prices)
troughs, _ = find_peaks(-prices)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=prices, mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=data.index[peaks], y=prices[peaks], mode='markers', name='Peaks', marker=dict(color='red', size=10)))
fig.add_trace(go.Scatter(x=data.index[troughs], y=prices[troughs], mode='markers', name='Troughs', marker=dict(color='green', size=10)))
fig.update_layout(title=f"{ticker} Stock Price with Peaks and Troughs", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

# Forecasting with Facebook Prophet
st.subheader("Price Forecasting with Facebook Prophet")
df_prophet = data[['Close']].reset_index()
df_prophet.columns = ['ds', 'y']

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(df_prophet, periods=365)
forecast = model.predict(future)

fig2 = model.plot(forecast)
st.write(fig2)

# Show forecast components
st.subheader("Forecast Components")
fig3 = model.plot_components(forecast)
st.write(fig3)
