import os
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Disable TensorFlow optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ FIXED: Correct model path (Use raw string `r""`)
model_path = r"C:\Users\hp\Desktop\Omnitricks\Stock Price\Stock Prediction model.keras"

# ✅ Check if model file exists before loading
if os.path.exists(model_path):
    st.success("✅ Model Excute Sucsessfully")
    model = load_model(model_path)
else:
    st.error("❌ Model fail")
    st.stop()

# 🌟 Streamlit UI
st.title('📈 Stock Market Predictor')

# 📌 Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol (e.g., GOOG, AAPL, TSLA)', 'GOOG')
start = '2015-01-01'
end = '2025-01-01'

# Fetch stock data
st.subheader('Fetching Stock Data...')
data = yf.download(stock, start, end)

# ✅ Check if data is available
if data.empty:
    st.error("❌ No stock data found! Please enter a valid stock symbol.")
    st.stop()

# Display raw data
st.subheader('📊 Raw Data')
st.write(data)

# 🔹 Splitting Data: 80% Training | 20% Testing
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# 🔹 Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# 🔹 Prepare test data (Last 100 days from training)
past_100_days = data_train.tail(100)
past_100_days_scaled = scaler.transform(past_100_days)  # Scale before merging
data_test_scaled = np.vstack((past_100_days_scaled, scaler.transform(data_test)))  # Merge properly

# 📈 Plot Moving Averages
ma_50_days = data.Close.rolling(50).mean().dropna()
ma_100_days = data.Close.rolling(100).mean().dropna()

# 📌 Moving Average 50 Plot
fig1 = plt.figure(figsize=(10, 6))
plt.plot(ma_50_days, color='r', label='MA50')
plt.plot(data.Close, color='g', label='Close Price')
plt.title('Stock Price vs MA50')
plt.legend()
st.pyplot(fig1)

# 📌 Moving Average 50 vs 100 Plot
fig2 = plt.figure(figsize=(10, 6))
plt.plot(ma_50_days, color='r', label='MA50')
plt.plot(ma_100_days, color='b', label='MA100')
plt.plot(data.Close, color='g', label='Close Price')
plt.title('Stock Price vs MA50 vs MA100')
plt.legend()
st.pyplot(fig2)

# ✅ Ensure enough data for model predictions
if data_test_scaled.shape[0] <= 100:
    st.error("⚠️ Not enough test data for making predictions.")
    st.stop()

# 🔹 Prepare Data for Model (100 Days Sequence)
x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# ✅ Make Predictions
if model:
    predicted_price = model.predict(x)

    # 🔹 Rescale predictions back to original price range
    predicted_price = predicted_price * (1 / scaler.scale_[0])
    y = y * (1 / scaler.scale_[0])

    # 📈 Plot Predicted vs Actual Prices
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(y, color='blue', label='Actual Price')
    plt.plot(predicted_price, color='red', label='Predicted Price')
    plt.title(f'📉 {stock} Price Prediction vs Actual')
    plt.legend()
    st.pyplot(fig3)

    st.success("✅ Prediction Completed Successfully!")
else:
    st.error("❌ Prediction model is not loaded. Please check the model file.")
