import pandas as pd

# Load your CSV file
df = pd.read_csv('AAPL_data.csv')

# Print the column names and first few rows
print(df.columns)
print(df.head())
