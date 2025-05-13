from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is running successfully!"

@app.route('/predict')
def predict():
    # Download stock data for training
    stock_data = yf.download('AAPL', start='2010-01-01', end='2025-04-23')
    stock_data.ffill(inplace=True)

    # Scale the 'Close' price
    scaler = MinMaxScaler()
    stock_data['Close'] = scaler.fit_transform(stock_data[['Close']])
    
    # Define features and target variable
    X = stock_data[['Open', 'High', 'Low', 'Volume']].values
    y = stock_data['Close'].values.reshape(-1, 1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred)

    # Calculate performance metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    # Fetch the most recent stock data for prediction
    latest_data = yf.download('AAPL', period='1d', interval='1h')

    # Select the features used during training
    latest_features = latest_data[['Open', 'High', 'Low', 'Volume']].values

    # Scale the features using the same scaler as during training
    latest_scaled = scaler.transform(latest_features)

    # Make prediction using the trained model
    future_price = model.predict(latest_scaled)

    # Reverse the scaling of the predicted price
    future_price = scaler.inverse_transform(future_price)

    # Return the results as a JSON response
    return jsonify({
        "mean_squared_error": round(mse, 6),
        "r2_score": round(r2, 4),
        "next_hour_price_prediction": round(float(future_price[-1][0]), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)


