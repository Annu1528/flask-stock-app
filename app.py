from flask import Flask
import pandas as pd
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def index():
    stock_data = yf.download('AAPL', start='2010-01-01', end='2025-04-23')
    return stock_data.head().to_html()

if __name__ == '__main__':
    app.run(debug=True)
