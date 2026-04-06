import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def get_prices(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError("No data")
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    return data[price_col].values

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def main():
    ticker = input("Ticker: ").strip()
    start = input("Start date (YYYY-MM-DD): ").strip()
    end = input("End date (YYYY-MM-DD): ").strip()
    
    prices = get_prices(ticker, start, end)
    if len(prices) < 200:
        print(f"Warning: Only {len(prices)} days. Need at least 200 for good R².")
    
    # Convert to log returns (stationary)
    log_returns = np.diff(np.log(prices))
    # Remove any potential NaN from first diff
    log_returns = log_returns[~np.isnan(log_returns)]
    
    if len(log_returns) < 100:
        print("Not enough log returns. Using raw prices instead.")
        # Fallback to raw prices scaled
        data = prices
        use_returns = False
    else:
        data = log_returns
        use_returns = True
    
    # Standardize (zero mean, unit variance) – better for returns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    seq_len = min(90, len(scaled) // 5)  # adaptive
    if seq_len < 10:
        print("Too little data. Need longer date range.")
        return
    
    X, y = create_sequences(scaled, seq_len)
    total = len(X)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Reshape
    X_train = X_train.reshape(-1, seq_len, 1)
    X_val = X_val.reshape(-1, seq_len, 1)
    X_test = X_test.reshape(-1, seq_len, 1)
    
    # Build model
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), input_shape=(seq_len, 1)),
        LSTM(64, dropout=0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=32, shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Predict on test set
    pred_scaled = model.predict(X_test).flatten()
    pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    if use_returns:
        # Convert returns back to prices
        # We need the last price before test sequence
        last_idx = train_end + val_end + seq_len  # approximate
        last_price = prices[last_idx]
        actual_prices = [last_price]
        pred_prices = [last_price]
        for r in actual_original:
            actual_prices.append(actual_prices[-1] * np.exp(r))
        for r in pred_original:
            pred_prices.append(pred_prices[-1] * np.exp(r))
        actual_prices = np.array(actual_prices[1:])
        pred_prices = np.array(pred_prices[1:])
    else:
        # Raw prices – already in original scale
        actual_prices = actual_original
        pred_prices = pred_original
    
    # Metrics
    mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    r2 = r2_score(actual_prices, pred_prices)
    print(f"\nMAPE: {mape:.2f}%  (Accuracy: {100-mape:.2f}%)")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Next day prediction
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    next_scaled = model.predict(last_seq).flatten()[0]
    next_return = scaler.inverse_transform([[next_scaled]])[0,0]
    if use_returns:
        next_price = prices[-1] * np.exp(next_return)
    else:
        next_price = next_return
    print(f"Predicted next closing price: {next_price:.2f}")
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices, label='Actual')
    plt.plot(pred_prices, 'r--', label='Predicted')
    plt.title(f'{ticker} - R²: {r2:.3f}, MAPE: {mape:.2f}%')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
