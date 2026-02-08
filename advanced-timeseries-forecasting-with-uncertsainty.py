# -----------------------------
# Advanced Time Series Forecasting with Uncertainty
# -----------------------------

# Task 1: Generate Synthetic Multivariate Time Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

np.random.seed(42)

def generate_multivariate_series(n_steps=5000):
    time = np.arange(n_steps)
    
    # Long-term trend
    trend = 0.001 * time
    
    # Seasonality
    seasonal_1 = 10 * np.sin(2 * np.pi * time / 50)
    seasonal_2 = 5 * np.sin(2 * np.pi * time / 200)
    
    # Noise
    noise = np.random.normal(0, 1.0, (n_steps, 5))
    
    # Feature 1: Trend + seasonality + noise
    x1 = trend + seasonal_1 + seasonal_2 + noise[:,0]
    
    # Feature 2: Lagged + correlated
    x2 = 0.5 * np.roll(x1, 3) + noise[:,1]
    
    # Feature 3: Random seasonality
    x3 = 0.7 * seasonal_1 + 0.3 * seasonal_2 + noise[:,2]
    
    # Feature 4: Sawtooth signal + trend
    x4 = 8 * sawtooth(2 * np.pi * time / 100) + trend + noise[:,3]
    
    # Feature 5: Combination of others
    x5 = 0.3*x1 + 0.3*x2 + 0.4*x3 + noise[:,4]
    
    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": x3,
        "feature_4": x4,
        "feature_5": x5
    })
    
    return df

# Generate data
df = generate_multivariate_series()
print(df.head())

# Plot features
df.plot(subplots=True, figsize=(12,10), title="Synthetic Multivariate Time Series")
plt.show()


# -----------------------------
# Task 2: Prepare Data for LSTM
# -----------------------------
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(data_scaled, seq_length)

# Train/test split
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -----------------------------
# Task 2: Build LSTM with MC Dropout
# -----------------------------
def build_lstm_mc(seq_length, n_features, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_mc(seq_length, X_train.shape[2], dropout_rate=0.2)
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)

# -----------------------------
# Task 3: Monte Carlo Dropout for Prediction Intervals
# -----------------------------
def mc_dropout_predictions(model, X_input, n_simulations=50):
    predictions = []
    for _ in range(n_simulations):
        # dropout active during inference
        preds = model(X_input, training=True).numpy()
        predictions.append(preds)
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    lower_pred = np.percentile(predictions, 5, axis=0)   # 90% CI lower
    upper_pred = np.percentile(predictions, 95, axis=0)  # 90% CI upper
    return mean_pred, lower_pred, upper_pred

y_pred_mean, y_pred_lower, y_pred_upper = mc_dropout_predictions(model, X_test, n_simulations=50)

# Inverse scale
y_test_inv = scaler.inverse_transform(y_test)
y_pred_mean_inv = scaler.inverse_transform(y_pred_mean)
y_pred_lower_inv = scaler.inverse_transform(y_pred_lower)
y_pred_upper_inv = scaler.inverse_transform(y_pred_upper)

# -----------------------------
# Task 4: Evaluation
# -----------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_test_inv, y_pred_mean_inv, squared=False)
mae = mean_absolute_error(y_test_inv, y_pred_mean_inv)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Coverage probability for 90% prediction interval
coverage = np.mean((y_test_inv >= y_pred_lower_inv) & (y_test_inv <= y_pred_upper_inv))
print(f"Coverage Probability (90% CI): {coverage*100:.2f}%")

# -----------------------------
# Task 4: Plot Forecast + Uncertainty (first feature)
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:,0], label='True feature_1')
plt.plot(y_pred_mean_inv[:,0], label='Predicted feature_1', color='orange')
plt.fill_between(np.arange(len(y_test_inv)), 
                 y_pred_lower_inv[:,0], y_pred_upper_inv[:,0], 
                 color='orange', alpha=0.2, label='90% Prediction Interval')
plt.legend()
plt.title("LSTM Forecast with Monte Carlo Dropout - Feature 1")
plt.show()
