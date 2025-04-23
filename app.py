import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("MVS.csv")
    date_column = next((col for col in data.columns if pd.to_datetime(data[col], errors='coerce').notna().all()), None)
    if date_column is None:
        raise ValueError("No datetime column found in the CSV file.")
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    data['MA10_V'] = data['Close_V'].rolling(window=10).mean()
    data['MA20_V'] = data['Close_V'].rolling(window=20).mean()
    data['Volatility_V'] = data['Close_V'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

data = load_data()

# Load models and scalers
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# Constants
seq_len = 60

# Prediction function
def make_future_prediction(user_date):
    today = data.index[-1].date()
    days_to_predict = (user_date - today).days

    if days_to_predict <= 0:
        st.error("Please select a future date beyond the last historical data.")
        return None, None

    temp_data_M = data[['Close_M']].values.tolist()
    temp_data_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].values.tolist()

    for _ in range(days_to_predict):
        last_seq_M = scaler_M.transform(np.array(temp_data_M[-seq_len:]).reshape(-1, 1)).reshape(1, seq_len, 1)
        last_seq_V = scaler_V.transform(np.array(temp_data_V[-seq_len:])).reshape(1, seq_len, 4)

        pred_M_scaled = model_M.predict(last_seq_M, verbose=0)
        pred_V_scaled = model_V.predict(last_seq_V, verbose=0)

        pred_M = scaler_M.inverse_transform(pred_M_scaled)[0][0]
        pred_V = scaler_V.inverse_transform(np.hstack([pred_V_scaled, np.zeros((1, 3))]))[0][0]

        temp_data_M.append([pred_M])
        temp_data_V.append([pred_V]*4)  # Repeat prediction for features

    last_price_M = data['Close_M'].iloc[-1]
    last_price_V = data['Close_V'].iloc[-1]
    pred_M = max(pred_M, last_price_M)
    pred_V = max(pred_V, last_price_V)

    return pred_M, pred_V

# UI setup
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

# Visualizations
def plot_historical_prices():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='MasterCard', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa', line=dict(color='blue')))
    fig.update_layout(title='Stock Prices (2008–2024)', xaxis_title='Date', yaxis_title='Price (USD)', hovermode='x')
    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    yearly_volume = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_volume.index.astype(str), y=yearly_volume['Volume_M'], name='MasterCard', marker_color='blue'))
    fig.add_trace(go.Bar(x=yearly_volume.index.astype(str), y=yearly_volume['Volume_V'], name='Visa', marker_color='orange'))
    fig.update_layout(title='Yearly Trading Volume', xaxis_title='Year', yaxis_title='Volume', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# Page routing
if page == "🏠 Home":
    st.title("🏦 Visa & Mastercard - Stock Market Overview")
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_prices()
    with col2:
        plot_volumes()
    st.success("Navigate to prediction or company info via sidebar.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    user_date = st.date_input("Select a future date (max Dec 2025)", min_value=data.index[-1].date() + timedelta(days=1), max_value=datetime(2025, 12, 31).date())

    if st.button("Predict Stock Prices"):
        pred_M, pred_V = make_future_prediction(user_date)
        if pred_M and pred_V:
            st.success(f"📅 Predicted Price on {user_date.strftime('%Y-%m-%d')}")
            st.write(f"💳 **Visa**: ${pred_V:.2f}")
            st.write(f"💰 **Mastercard**: ${pred_M:.2f}")

            advice_M = "Buy" if pred_M > data['Close_M'].iloc[-1] else "Sell"
            advice_V = "Buy" if pred_V > data['Close_V'].iloc[-1] else "Sell"

            st.subheader("💡 Investment Advice")
            st.write(f"Mastercard: {advice_M}")
            st.write(f"Visa: {advice_V}")

elif page == "ℹ️ Company Info":
    st.title("📢 Company Info")
    st.write("Visa and Mastercard are global leaders in digital payments.")
    st.write("Visa Inc. is based in Foster City, California.")
    st.write("Mastercard is headquartered in Purchase, New York.")
