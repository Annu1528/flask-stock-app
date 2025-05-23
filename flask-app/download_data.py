import yfinance as yf

ticker = 'AAPL'  # You can change this to any stock symbol
start_date = '2010-01-01'
end_date = '2025-05-22'  # Or today's date

df = yf.download(ticker, start=start_date, end=end_date)
df.to_csv('AAPL_data.csv')

print("Stock data downloaded and saved as AAPL_data.csv")
