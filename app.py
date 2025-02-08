import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler

# Disable TensorFlow optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define model path using absolute path for compatibility
model_path = os.path.abspath("C:/Users/hp/Desktop/Omnitricks/Stock Price/Stock Prediction model.keras")

# Load model only if the file exists
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("✅ Model successfully loaded!")
else:
    st.error("❌ Model file NOT found! Please check the path.")
    st.stop()

st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2015-01-01'
end = '2025-01-01'

# Download stock data
data = yf.download(stock, start, end)

# Ensure data is retrieved
if data.empty:
    st.error("No data found for the given stock symbol!")
    st.stop()

st.subheader('Raw Data')
st.write(data)

# Split data into training (80%) and testing (20%)
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Ensure scaler is properly fitted
if not hasattr(scaler, 'scale_'):
    st.error("Scaler has not been properly fitted. Please check your data preprocessing.")
    st.stop()

# Prepare test data (adding last 100 days from training)
past_100_days = data_train.tail(100)
past_100_days_scaled = scaler.transform(past_100_days)  # Scale before merging
data_test_scaled = np.vstack((past_100_days_scaled, scaler.transform(data_test)))  # Merge properly

# Plot Price vs Moving Averages
ma_50_days = data.Close.rolling(50).mean().dropna()
ma_100_days = data.Close.rolling(100).mean().dropna()

fig1 = plt.figure(figsize=(10, 6))
plt.plot(ma_50_days, color='r', label='MA50')
plt.plot(data.Close, color='g', label='Close Price')
plt.title('Stock Price vs MA50')
plt.legend()
st.pyplot(fig1)

fig2 = plt.figure(figsize=(10, 6))
plt.plot(ma_50_days, color='r', label='MA50')
plt.plot(ma_100_days, color='b', label='MA100')
plt.plot(data.Close, color='g', label='Close Price')
plt.title('Stock Price vs MA50 vs MA100')
plt.legend()
st.pyplot(fig2)

# Ensure sufficient data for LSTM
if data_test_scaled.shape[0] <= 100:
    st.error("Not enough data for making predictions.")
    st.stop()

# Prepare data for LSTM model (100 days sequence)
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Make predictions only if model is loaded
predicted_price = model.predict(x)

# Rescale predictions back to original price range
predicted_price = predicted_price * (1 / scaler.scale_[0])
y = y * (1 / scaler.scale_[0])

# Plot predicted vs actual prices
fig3 = plt.figure(figsize=(10, 6))
plt.plot(y, color='blue', label='Actual Price')
plt.plot(predicted_price, color='red', label='Predicted Price')
plt.title(f'{stock} Price Prediction vs Actual')
plt.legend()
st.pyplot(fig3)
