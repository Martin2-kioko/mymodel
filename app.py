import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")

# --- Load and process data ---
def load_data():
    data = pd.read_csv("MVS.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)

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

# Prediction function
def make_future_prediction(user_date):
    today = datetime.now()
    days_to_predict = (user_date - today.date()).days
    seq_len = 60

    temp_data_M = data['Close_M'].values[-seq_len:].tolist()
    temp_data_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].values[-seq_len:].tolist()

    for _ in range(days_to_predict):
        scaled_M = scaler_M.transform(np.array(temp_data_M[-seq_len:]).reshape(-1, 1))
        scaled_M_input = scaled_M.reshape(1, seq_len, 1)
        pred_M = model_M.predict(scaled_M_input, verbose=0)
        inv_pred_M = scaler_M.inverse_transform(pred_M)[0][0]
        temp_data_M.append(inv_pred_M)

        scaled_V = scaler_V.transform(np.array(temp_data_V[-seq_len:]))
        scaled_V_input = scaled_V.reshape(1, seq_len, 4)
        pred_V = model_V.predict(scaled_V_input, verbose=0)
        padding = np.zeros((1, 3))
        inv_pred_V = scaler_V.inverse_transform(np.hstack((pred_V, padding)))[0][0]
        last_v_row = temp_data_V[-1][:]
        temp_data_V.append([inv_pred_V] + last_v_row[1:])

    return inv_pred_M, inv_pred_V

# --- Plotting functions ---
def plot_historical_prices():
    st.subheader("Stock Prices: 2008–2024")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='MasterCard Close', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa Close', line=dict(color='blue')))
    fig.update_layout(title='Stock Prices of MasterCard and Visa', xaxis_title='Date', yaxis_title='Stock Price (USD)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    st.subheader("Yearly Trading Volume: 2008–2024")
    yearly_volume = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    yearly_volume.index = yearly_volume.index.astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_volume.index, y=yearly_volume['Volume_M'], name='Mastercard', marker=dict(color='blue'), opacity=0.7))
    fig.add_trace(go.Bar(x=yearly_volume.index, y=yearly_volume['Volume_V'], name='Visa', marker=dict(color='orange'), opacity=0.7))
    fig.update_layout(title='Yearly Trading Volume for Mastercard and Visa (2008–2024)', xaxis_title='Year', yaxis_title='Trading Volume (USD)', barmode='group', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

if page == "🏠 Home":
    st.title("🏦 Visa & Mastercard - Stock Market Overview")
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_prices()
    with col2:
        plot_volumes()
    st.success("Use the sidebar to navigate to the prediction page or company info.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    max_date = datetime(2025, 12, 31).date()
    user_date = st.date_input("Select a future date (after June 2024)", min_value=datetime(2024, 6, 29).date(), max_value=max_date, value=datetime(2025, 6, 1).date())

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
