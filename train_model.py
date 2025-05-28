import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the data from the 'data' folder and skip extra header rows
df = pd.read_csv('data/AAPL_data.csv', skiprows=[1, 2])

# Print columns and shape for debugging
print("Columns in data:", df.columns)
print("Number of rows:", len(df))

# The date column is 'Price' after skipping rows, convert it to datetime
df['Price'] = pd.to_datetime(df['Price'])

# Use the 'Close' column for prediction; make sure it's numeric
close_prices = pd.to_numeric(df['Close'], errors='coerce')

print("Any NaNs in Close?", close_prices.isna().any())

# Drop NaNs for training
close_prices = close_prices.dropna()

# Check for infinite values - use numpy array from pandas Series with float dtype
close_array = close_prices.to_numpy(dtype=np.float64)
print("Any infinite values in Close?", np.isinf(close_array).any())

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_close = scaler.fit_transform(close_array.reshape(-1,1))

# Prepare dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(scaled_close, time_step)

# Reshape input to [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=10, batch_size=1, verbose=1)

# Save model and scaler
model.save('real_stock_model.keras')
import joblib
joblib.dump(scaler, 'scaler.save')

print("Model and scaler saved successfully.")
