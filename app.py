import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Set page config as the first command
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")

# --- Utility functions ---
def load_data():
    data = pd.read_csv("MVS.csv")
    
    date_column = None
    for col in data.columns:
        try:
            pd.to_datetime(data[col])
            date_column = col
            break
        except:
            continue

    if date_column is None:
        raise ValueError("No datetime column found in the CSV file.")

    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)

    data['MA10_V'] = data['Close_V'].rolling(window=10).mean()
    data['MA20_V'] = data['Close_V'].rolling(window=20).mean()
    data['Volatility_V'] = data['Close_V'].rolling(window=10).std()

    data.dropna(inplace=True)
    
    return data

# --- Prediction function ---
def make_future_prediction(user_date):
    today = data.index[-1].date()
    if isinstance(user_date, datetime):
        user_date = user_date.date()

    days_to_predict = (user_date - today).days

    if days_to_predict <= 0:
        st.error("Please select a date after the last available date in the dataset.")
        return None, None

    seq_len = 60

    temp_data_M = data['Close_M'].values[-seq_len:].reshape(-1, 1)
    temp_data_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].values[-seq_len:]

    for _ in range(days_to_predict):
        scaled_M = scaler_M.transform(temp_data_M[-seq_len:].reshape(-1, 1))
        scaled_M = scaled_M.reshape(1, seq_len, 1)
        pred_scaled_M = model_M.predict(scaled_M)
        pred_M = scaler_M.inverse_transform(pred_scaled_M)
        temp_data_M = np.append(temp_data_M, pred_M).reshape(-1, 1)

        scaled_V = scaler_V.transform(temp_data_V[-seq_len:])
        scaled_V = scaled_V.reshape(1, seq_len, 4)
        pred_scaled_V = model_V.predict(scaled_V)
        padded_input = np.hstack((pred_scaled_V, np.zeros((pred_scaled_V.shape[0], 3))))
        pred_V = scaler_V.inverse_transform(padded_input)[:, 0]
        new_row_V = np.array([pred_V[0]] * 4).reshape(1, 4)
        temp_data_V = np.append(temp_data_V, new_row_V, axis=0)

    return temp_data_M[-1][0], temp_data_V[-1][0]

# Load data
data = load_data()

# Load models and scalers
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# --- UI setup ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

# --- Plotting and visualization ---
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

# --- Page logic ---
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
    user_date = st.date_input("Select a future date (after June 2024)", min_value=date(2024, 6, 29), value=date(2025, 6, 1))
    if st.button("Predict Stock Prices"):
        pred_M, pred_V = make_future_prediction(user_date)
        if pred_M is None or pred_V is None:
            st.error("Prediction failed. Please try again with a valid date.")
        else:
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
