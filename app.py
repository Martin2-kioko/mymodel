import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Set page config first
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")

# --- Load data function ---
def load_data():
    data = pd.read_csv("MVS.csv")

    # Detect datetime column
    for col in data.columns:
        try:
            data[col] = pd.to_datetime(data[col])
            data.set_index(col, inplace=True)
            break
        except:
            continue

    # Add moving averages and volatility
    data['MA10_V'] = data['Close_V'].rolling(window=10).mean()
    data['MA20_V'] = data['Close_V'].rolling(window=20).mean()
    data['Volatility_V'] = data['Close_V'].rolling(window=10).std()

    data.dropna(inplace=True)
    return data

# --- Load data and models ---
data = load_data()
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# --- Predict future stock prices ---
def make_future_prediction(user_date):
    latest_data_M = data[['Close_M']].tail(1).values
    latest_data_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].tail(1).values

    scaled_M = scaler_M.transform(latest_data_M)
    scaled_V = scaler_V.transform(latest_data_V)

    scaled_M = scaled_M.reshape(1, 1, -1)
    scaled_V = scaled_V.reshape(1, 1, -1)

    pred_M_scaled = model_M.predict(scaled_M)
    pred_V_scaled = model_V.predict(scaled_V)

    pred_M = scaler_M.inverse_transform(pred_M_scaled)[0][0]

    full_V = np.zeros((1, 4))
    full_V[:, 0] = pred_V_scaled[:, 0]  # fill in only the predicted Close_V
    pred_V = scaler_V.inverse_transform(full_V)[0][0]

    # Ensure predictions are not below last historical price
    pred_M = max(pred_M, data['Close_M'].iloc[-1])
    pred_V = max(pred_V, data['Close_V'].iloc[-1])

    return pred_M, pred_V

# --- Plotting functions ---
def plot_historical_prices():
    st.subheader("Stock Prices: 2008–2024")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='MasterCard', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa', line=dict(color='blue')))
    fig.update_layout(title='Stock Prices of MasterCard and Visa', xaxis_title='Date', yaxis_title='Price (USD)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    st.subheader("Yearly Trading Volume: 2008–2024")
    yearly_volume = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    yearly_volume.index = yearly_volume.index.astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_volume.index, y=yearly_volume['Volume_M'], name='Mastercard', marker=dict(color='blue')))
    fig.add_trace(go.Bar(x=yearly_volume.index, y=yearly_volume['Volume_V'], name='Visa', marker=dict(color='orange')))
    fig.update_layout(title='Yearly Volume', barmode='group', xaxis_title='Year', yaxis_title='Volume (USD)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# --- Sidebar and Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

if page == "🏠 Home":
    st.title("🏦 Visa & Mastercard - Stock Market Overview")
    col1, col2 = st.columns(2)
    with col1: plot_historical_prices()
    with col2: plot_volumes()
    st.success("Use the sidebar to predict future prices or learn about the companies.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    user_date = st.date_input("Select a future date", min_value=datetime(2024, 6, 29), value=datetime(2025, 6, 1))
    if st.button("Predict Stock Prices"):
        pred_M, pred_V = make_future_prediction(user_date)
        st.success(f"📅 Predicted Price on {user_date.strftime('%Y-%m-%d')}")
        st.write(f"💳 **Visa**: ${pred_V:.2f}")
        st.write(f"💰 **Mastercard**: ${pred_M:.2f}")

        st.subheader("💡 Investment Advice")
        advice_M = "Buy" if pred_M > data['Close_M'].iloc[-1] else "Sell"
        advice_V = "Buy" if pred_V > data['Close_V'].iloc[-1] else "Sell"
        st.write(f"Mastercard: {advice_M}")
        st.write(f"Visa: {advice_V}")

elif page == "ℹ️ Company Info":
    st.title("📢 Visa & Mastercard Info")
    st.write("Visa and Mastercard are two of the largest companies in the global payments industry.")
    st.write("Visa Inc. is an American multinational financial services corporation headquartered in Foster City, California.")
    st.write("Mastercard is an American multinational financial services corporation headquartered in Purchase, New York.")
    st.write("Both companies provide payment solutions to businesses, governments, and consumers worldwide.")
