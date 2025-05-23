import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import joblib

# Load CSV without header
raw_df = pd.read_csv('AAPL_data.csv', header=None)

# Drop first two rows (ticker and label rows)
raw_df = raw_df.drop(index=[0, 1]).reset_index(drop=True)

# Set proper column names
raw_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Select Close column and copy to avoid SettingWithCopyWarning
df = raw_df[['Close']].copy()

# Convert Close to float, drop NaNs if any
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()
df = df[np.isfinite(df['Close'])].copy()


print("Any NaNs?", df.isnull().any().any())
print("Any infinite values?", np.isfinite(df['Close']).all())

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df)

# Save scaler for later use
joblib.dump(scaler, 'scaler.save')

scaled_close = scaled_close.astype(np.float32)

# Prepare sequences for LSTM
X, y = [], []
for i in range(60, len(scaled_close)):
    X.append(scaled_close[i - 60:i, 0])
    y.append(scaled_close[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build model
model = Sequential()
model.add(Input(shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=32)

model.save('real_stock_model.keras')  # saving in new Keras format
