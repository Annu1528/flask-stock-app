import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
from waitress import serve

# Load model and scaler
from keras.saving.legacy import load_model
lstm_model = load_model('real_stock_model.keras')

scaler = joblib.load('scaler.save')

app = Flask(__name__)
CORS(app)

# Route: Home page, displays table and plot for AAPL
@app.route('/')
def home():
    csv_path = os.path.join('data', 'AAPL_data.csv')
    df = pd.read_csv(csv_path, header=None)
    df = df.drop(index=[0, 1]).reset_index(drop=True)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce').dropna()
    table_html = df.head(5).to_html(classes='table table-striped', index=False)

    # Placeholder for chart
    graph_html = "<p>Chart goes here</p>"

    return render_template('index.html', stock_data=table_html, graph_html=graph_html)

# Download CSV endpoint
@app.route('/download')
def download():
    csv_path = os.path.join('data', 'AAPL_data.csv')
    return send_file(csv_path, as_attachment=True)

# ARIMA forecast endpoint (example)
@app.route('/forecast/arima/<ticker>', methods=['GET'])
def arima_forecast(ticker):
    # Your ARIMA code here
    pass

# LSTM forecast endpoint (example)
@app.route('/forecast/lstm/<ticker>', methods=['GET'])
def lstm_forecast(ticker):
    # Your LSTM forecast code here
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
