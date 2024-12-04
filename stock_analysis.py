import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM

# yf.pdr_override() # deprecated

# Single Stock Prediction with LSTM 

# Define the stock symbol and date range
stock_symbol = 'NVDA'
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

# Fetch data for the stock
df = yf.download(stock_symbol, start=start, end=end)

# Flatten the column MultiIndex
df.columns = [col[0] for col in df.columns]  # Keep only the first level of the header

# makes Close accessible by removed the other header level
data = df.filter(['Close'])

# Convert to numpy array
dataset = data.values

def preprocess_data(stock_symbol, epochs=100, batch_size=32, window_size=60):
    # Fetch data
    end = datetime.now()
    start = datetime(end.year - 5, end.month, end.day)

    # Fetch data for multiple stocks for the last 5 years
    df = yf.download(stock_symbol, start=start, end=end)

    # Flatten the column MultiIndex
    df.columns = [col[0] for col in df.columns]  # Keep only the first level of the header

    # Now, 'Close' will be accessible directly
    data = df.filter(['Close'])

    # Convert to numpy array
    dataset = data.values

    # split training data
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    # Make predictions
    test_data = scaled_data[training_data_len - window_size:, :]
    test_set = []
    for i in range(window_size, len(test_data)):
        test_set.append(test_data[i-window_size:i, 0])

    test_set = np.array(test_set)
    test_set = np.reshape(test_set, (test_set.shape[0], test_set.shape[1], 1))

    prediction = model.predict(test_set)
    scaled_pred = scaler.inverse_transform(prediction)

    # Prepare results
    valid = data[training_data_len:]
    valid['Predictions'] = scaled_pred

    return df, valid, scaled_pred

df_nvda_5, valid_nvda, pred_nvda = preprocess_data('NVDA')

r2 = r2_score(valid_nvda['Close'], valid_nvda['Predictions'])
print(f'R2 Score: {r2}')

current_price = df_nvda_5['Close'].iloc[-1]
threshold = 0.03

# predicted_price = float(pred_nvda[-1])
predicted_price = valid_nvda['Predictions'].iloc[-1]

# Decision logic
price_change = (predicted_price - current_price) / current_price  # Calculate % change
if price_change > threshold:
    action = "Buy"
elif price_change < -threshold:
    action = "Sell"
else:
    action = "Hold"

# Print the decision
print(f"Predicted Price: ${predicted_price:.2f}")
print(f"Current Price: ${current_price:.2f}")
print(f"Price Change: {price_change:.2%}")
print(f"Decision: {action}")

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Actual vs Predicted', fontsize=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(df_nvda_5['Close'])
plt.plot(valid_nvda[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Stock Prediction using influence of other Stocks in LSTM

# stock prediction focuses semiconductor stocks

# semiconductor stocks
# nvidia, taiwan semiconductor manufacturing, broadcom, qualcomm, AMD

# 5 years
stock_list = ['NVDA','TSM', 'AVGO', 'QCOM', 'AMD']

target_symbol = 'NVDA'

# Fetch data for multiple stocks for 5 years
end = datetime.now()
start = datetime(end.year - 5, end.month, end.day)

# Fetch data for multiple stocks for the last 5 years
data = yf.download(stock_list, start=start, end=end)

# data.head()

stock_data = data['Close']

# Separate the target stock (NVDA) and other stocks
target_data = stock_data[target_symbol]  # The target stock data (NVDA)
other_stocks_data = stock_data.drop(columns=[target_symbol])  # All other stocks

# Combine the data: We want the target stock's history based on all other stocks
dataset = pd.concat([other_stocks_data, target_data], axis=1).values

# Calculate the length for training data
training_data_len = int(np.ceil(len(dataset) * .95))

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

window_size = 60 # number of days to consider

# Prepare the feature (X) and target (y) sequences
X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size, :-1])  # All stock data for the window, excluding target stock
    y.append(scaled_data[i+window_size, -1])  # Target stock's next day close price, the last column


# Convert the lists into numpy arrays
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

epochs = 100
batch_size = 32

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

# Make predictions
test_data = scaled_data[training_data_len - window_size:, :]
test_set = []
for i in range(window_size, len(test_data)):
    test_set.append(test_data[i - window_size:i, :-1])  # Use the history of other stocks

test_set = np.array(test_set)
test_set = np.reshape(test_set, (test_set.shape[0], test_set.shape[1], test_set.shape[2]))

# Prediction
prediction = model.predict(test_set)

# Reshape the prediction to (num_samples, 1) for inverse transformation
prediction = prediction.reshape(-1, 1)  # Flatten the output for inverse transformation

# Inverse transform the predictions for the target stock (the last column)
scaled_pred = scaler.inverse_transform(np.hstack([np.zeros((prediction.shape[0], scaled_data.shape[1] - 1)), prediction]))

# Prepare results for the target stock
valid = target_data[training_data_len:]
valid = valid.to_frame()  # Convert to DataFrame, in the case the data is a series
valid['Predictions'] = scaled_pred[:, -1]  # Only take the last column for the target stock

r2 = r2_score(valid['NVDA'], valid['Predictions'])
print(f'R2 Score: {r2}')

# Extract the last prediction as a scalar value
predicted_price = valid['Predictions'].iloc[-1]

# Get the current price (last adjusted close price)
current_price = data['Close']['NVDA'].iloc[-1]

# Define the threshold
threshold = 0.03 

# Decision logic
price_change = (predicted_price - current_price) / current_price  # Calculate % change
if price_change > threshold:
    action = "Buy"
elif price_change < -threshold:
    action = "Sell"
else:
    action = "Hold"

# Print the decision
print(f"Predicted Price: ${predicted_price:.2f}")
print(f"Current Price: ${current_price:.2f}")
print(f"Price Change: {price_change:.2%}")
print(f"Decision: {action}")

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Actual vs Predicted', fontsize=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)

plt.plot(data[:training_data_len]['Close'], color='blue')
plt.plot(data['Close']['NVDA'], label='Validation NVDA (Actual)', color='green')
plt.plot(valid['Predictions'], label='Predictions NVDA', color='red')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.show()