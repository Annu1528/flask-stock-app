import os
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request
from flask_caching import Cache
from flask_cors import CORS
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf

# Use recommended TensorFlow import patterns
load_model = tf.keras.models.load_model
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Load pre-trained LSTM model (ensure this path is valid)
lstm_model = load_model('lstm_model.h5')

# -------------------- ARIMA Forecasting --------------------
@app.route('/forecast/arima/<ticker>', methods=['GET'])
def arima_forecast(ticker):
    try:
        df = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        df = df[['Close']].dropna()

        model = ARIMA(df, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=30)
        forecast_dates = pd.date_range(df.index[-1], periods=31, freq='B')[1:]

        return jsonify({
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- LSTM Forecasting --------------------
@app.route('/forecast/lstm/<ticker>', methods=['GET'])
def lstm_forecast(ticker):
    try:
        df = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        df = df[['Close']].dropna()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        forecast = lstm_model.predict(X[-30:])
        forecast = scaler.inverse_transform(forecast)

        forecast_dates = pd.date_range(df.index[-1], periods=31, freq='B')[1:]

        return jsonify({
            'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast.flatten().tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Peaks and Troughs --------------------
@app.route('/peaks_troughs/<ticker>', methods=['GET'])
def peaks_troughs(ticker):
    try:
        df = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        df = df[['Close']].dropna()
        df['Return'] = df['Close'].pct_change()

        peaks = (df['Return'] > 0) & (df['Return'].shift(-1) < 0)
        troughs = (df['Return'] < 0) & (df['Return'].shift(-1) > 0)

        peak_dates = df[peaks].index
        trough_dates = df[troughs].index

        return jsonify({
            'peak_dates': peak_dates.strftime('%Y-%m-%d').tolist(),
            'trough_dates': trough_dates.strftime('%Y-%m-%d').tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Main Entry Point --------------------

if __name__ == "__main__":
    app.run(debug=True)

