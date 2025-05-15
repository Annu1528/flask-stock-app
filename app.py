import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta

st.title("Stock Price Analysis with Peak/Trough Bars and Forecast")

# Function to plot bars with peak/trough annotations
def plot_bar_with_extremes(close_series):
    # Ensure the series is sorted by date
    close_series = close_series.sort_index()

    # Compute 1-day differences; ensure numeric (1D) and fill NaNs
    diff = close_series.diff()
    diff = pd.to_numeric(diff, errors='coerce').fillna(0)

    # Assign colors: green for positive diff, red for zero/negative
    colors = ['green' if val > 0 else 'red' for val in diff]

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(range(len(close_series)), close_series, color=colors)

    # Find index and values of peak and trough
    values = close_series.values
    if len(values) == 0:
        return fig  # nothing to plot
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    max_val = values[max_idx]
    min_val = values[min_idx]
    offset = (max_val - min_val) * 0.05 if max_val != min_val else 0.1

    # Annotate Peak (highest value)
    ax.annotate(f"Peak: {max_val:.2f}",
                xy=(max_idx, max_val), xytext=(max_idx, max_val + offset),
                ha='center', color='green',
                arrowprops=dict(facecolor='green', arrowstyle='->'))

    # Annotate Trough (lowest value)
    ax.annotate(f"Trough: {min_val:.2f}",
                xy=(min_idx, min_val), xytext=(min_idx, min_val + offset),
                ha='center', color='red',
                arrowprops=dict(facecolor='red', arrowstyle='->'))

    # Labeling
    ax.set_xlabel("Date Index")
    ax.set_ylabel("Closing Price")
    ax.set_title("Closing Prices with Peak and Trough Highlighted")
    plt.xticks([])  # Hide x-axis labels for readability
    plt.tight_layout()
    return fig

# User inputs for ticker and date range
symbol = st.text_input("Ticker Symbol", "AAPL")
today = date.today()
default_start = today - timedelta(days=365)
start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", today)

if st.button("Get Data"):
    if symbol:
        # Fetch stock data
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

        if df is None or df.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            # Ensure closing prices are numeric
            df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
            df = df.dropna(subset=["Close"])
            df = df.sort_index()

            st.subheader(f"Closing Prices for {symbol}")
            st.write(df["Close"].tail())  # show last few values

            # Plot bar chart with extremes
            fig = plot_bar_with_extremes(df["Close"])
            st.pyplot(fig)
            plt.close(fig)

            # Simple forecast: linear trend (next 30 business days by default)
            st.subheader("Future Price Forecast")
            forecast_days = st.slider("Forecast Days", 1, 90, 30)
            prices = df["Close"].values
            if len(prices) < 2:
                st.write("Not enough data for forecasting.")
            else:
                x = np.arange(len(prices))
                # Fit a linear trend (degree 1 polynomial)
                coeffs = np.polyfit(x, prices, 1)
                future_x = np.arange(len(prices), len(prices) + forecast_days)
                preds = np.polyval(coeffs, future_x)

                # Generate future business dates starting after the end date
                future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_days+1)[1:]
                forecast_series = pd.Series(preds, index=future_dates)
                forecast_series = forecast_series.rename("Predicted Close")

                st.line_chart(forecast_series)
                st.write(forecast_series.head())
