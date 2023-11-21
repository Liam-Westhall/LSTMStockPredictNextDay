import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
import yfinance as yf

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    # Fetch stock price data for given dates
    print("Fetching stock data...")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to normalize stock data
def normalize_data(data):
    # Scale stock prices to a range between 0 and 1 for better neural network performance
    print("Normalizing data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    return normalized_data, scaler

# Function to create training sequences from stock data
def create_training_sequences(data, look_back):
    # Prepare sequences and corresponding labels for LSTM training
    print("Creating training sequences...")
    sequences = []
    labels = []
    for i in range(look_back, len(data)):
        # Each sequence consists of 'look_back' days of stock prices
        sequences.append(data[i - look_back:i, 0])
        # Label is the price following the sequence
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)

# Function to build LSTM model
def build_lstm_model(input_shape):
    # Define the LSTM model structure
    print("Building LSTM model...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),  # Helps prevent overfitting
        LSTM(50),
        Dropout(0.2),
        Dense(1)  # Output layer that predicts the stock price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and save LSTM model
def train_and_save_lstm_model(ticker, start_date, end_date, look_back=60, model_file='stock_model.h5'):
    # Train the LSTM model and save it for later use
    print("Training model for:", ticker)
    stock_data = get_stock_data(ticker, start_date, end_date)
    normalized_data, data_scaler = normalize_data(stock_data)
    sequences, labels = create_training_sequences(normalized_data, look_back)

    lstm_model = build_lstm_model((look_back, 1))
    lstm_model.fit(sequences, labels, epochs=5, batch_size=32)

    print("Saving model...")
    lstm_model.save(model_file)
    return lstm_model, data_scaler

# Function to prepare data for making a prediction
def prepare_prediction_data(scaler, data, look_back):
    # Scale and prepare recent stock data for prediction
    print("Preparing data for prediction...")
    normalized_data = scaler.transform(data['Close'].values.reshape(-1,1))
    prediction_sequences, _ = create_training_sequences(normalized_data, look_back)
    return prediction_sequences

#stock price prediction
def predict_stock_price(model, scaler, ticker, end_date, look_back=60, extra_days=30):
    # Predict the stock price for the day after the given end date
    print(f"Making prediction for {ticker}...")
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    next_day = end_date + datetime.timedelta(days=1)
    
    # Prepare recent data
    start_date = end_date - datetime.timedelta(days=look_back + extra_days)
    recent_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Enough data check 
    if len(recent_data) < look_back:
        raise ValueError(f"Insufficient data for prediction. Required: {look_back}, available: {len(recent_data)}")

    prediction_sequences = prepare_prediction_data(scaler, recent_data, look_back)

    # Make the prediction and convert it back to original scale
    predicted_price = model.predict(prediction_sequences)
    actual_predicted_price = scaler.inverse_transform(predicted_price)
    print(f"Predicted Price for {ticker} on {next_day.strftime('%m/%d/%Y')}: {actual_predicted_price[0, 0]}")

# Main function to process command-line arguments
def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Stock Ticker Symbol')
    parser.add_argument('--start_date', type=str, required=True, help='Start Date for Training Data MM/DD/YYYY')
    parser.add_argument('--end_date', type=str, required=True, help='End Date for Prediction MM/DD/YYYY')
    args = parser.parse_args()

    # Convert date for yfinance
    start_date_formatted = datetime.datetime.strptime(args.start_date, '%m/%d/%Y').strftime('%Y-%m-%d')
    end_date_formatted = datetime.datetime.strptime(args.end_date, '%m/%d/%Y').strftime('%Y-%m-%d')
    # Train and predict
    model, scaler = train_and_save_lstm_model(args.ticker, start_date_formatted, end_date_formatted)
    predict_stock_price(model, scaler, args.ticker, end_date_formatted)

if __name__ == "__main__":
    main()
