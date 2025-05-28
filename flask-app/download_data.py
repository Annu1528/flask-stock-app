import yfinance as yf

# Download 1 year of daily Apple stock data
data = yf.download('AAPL', period='1y', interval='1d')

# Save the cleaned data CSV file
data.to_csv('AAPL_data_clean.csv')

print(data.head())
