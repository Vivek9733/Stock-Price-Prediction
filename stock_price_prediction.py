import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Function to download stock data
def get_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Function to prepare data for training
def prepare_stock_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

# Function to split data into training and testing sets
def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    return data[:train_size], data[train_size:]

# Function to create sequences for LSTM
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Simple LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to train the model
def train_model(model, device, X_train, y_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

# Function to make predictions
def make_predictions(model, device, X_test):
    model.eval()
    with torch.no_grad():
        return model(X_test)

# Main function
def main():
    ticker = input("Enter stock ticker: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    
    # Get stock data
    data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    scaled_data, scaler = prepare_stock_data(data)
    
    # Split data
    train_data, test_data = split_data(scaled_data)
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_len=30)
    X_test, y_test = create_sequences(test_data, seq_len=30)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = LSTMModel().to(device)
    X_train, y_train = X_train.view(-1, 30, 1).to(device), y_train.to(device)
    X_test = X_test.view(-1, 30, 1).to(device)
    
    # Train model
    train_model(model, device, X_train, y_train)
    
    # Make predictions
    predictions = make_predictions(model, device, X_test)
    predictions = scaler.inverse_transform(predictions.cpu().numpy())
    
    # Display final predicted price
    print(f"\nPredicted Closing Price: {predictions[-1][0]:.2f}")
    
    # Plot actual vs predicted
    y_test_inv = scaler.inverse_transform(y_test.cpu().numpy())
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label='Historical Prices')
    plt.plot(predictions, label='Predicted Prices', linestyle='--')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
