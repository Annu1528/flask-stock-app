from flask import Flask, render_template, send_file
import yfinance as yf

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

if __name__ == '__main__':
    app.run(debug=True)

