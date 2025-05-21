import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib

# Load data (adjust 'sep' if needed)
try:
    df = pd.read_csv("aapl_stock_data.csv", sep=",")
except Exception as e:
    print("Error reading CSV:", e)
    exit()

print("Columns:", df.columns)

if "Close" not in df.columns:
    raise ValueError("Expected 'Close' column in dataset")

data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'scaler.save')

X = []
y = []
time_step = 60

for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i - time_step:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(tf.keras.layers.LSTM(units=50))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

model.save("lstm_stock_model.h5")
print("Model and scaler saved successfully.")
