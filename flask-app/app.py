from flask import Flask, render_template, send_file
import yfinance as yf
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch stock data for AAPL
    stock_data = yf.download('AAPL', start='2010-01-01', end='2025-04-23')

    # Save the stock data as a CSV file
    stock_data.to_csv('aapl_stock_data.csv')

    return render_template('index.html', stock_data=stock_data.head())

@app.route('/download')
def download():
    # Provide a download link for the CSV file
    return send_file('aapl_stock_data.csv', as_attachment=True)

# Ensure the app runs on the correct port and host when deployed
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the PORT from the environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)  # Make sure app runs with the correct host and port
