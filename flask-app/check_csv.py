import pandas as pd

df = pd.read_csv('AAPL_data.csv', header=None)
print("First 10 rows of raw CSV:")
print(df.head(10))
print("Columns:", df.columns)
print("Number of rows:", len(df))
