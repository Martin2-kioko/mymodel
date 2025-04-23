import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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

# --- Multi-step prediction function ---
def make_future_prediction(user_date):
    steps = (user_date - data.index[-1].date()).days

    if steps <= 0:
        st.warning("Please select a date after the last available date in the dataset.")
        return None, None

    current_input_M = data[['Close_M']].tail(1).values
    current_input_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].tail(1).values

    for _ in range(steps):
        scaled_M = scaler_M.transform(current_input_M.reshape(1, -1)).reshape(1, 1, -1)
        scaled_V = scaler_V.transform(current_input_V.reshape(1, -1)).reshape(1, 1, -1)

        pred_M_scaled = model_M.predict(scaled_M, verbose=0)
        pred_V_scaled = model_V.predict(scaled_V, verbose=0)

        pred_M = scaler_M.inverse_transform(pred_M_scaled)

        pred_V_full = np.zeros((1, 4))
        pred_V_full[0][0] = pred_V_scaled[0][0]
        pred_V = scaler_V.inverse_transform(pred_V_full)[0][0]

        current_input_M = np.array([[pred_M[0][0]]])
        last_row = current_input_V[0]
        current_input_V = np.array([[pred_V, last_row[1], last_row[2], last_row[3]]])

    return pred_M[0][0], pred_V

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
    user_date = st.date_input("Select a future date (after June 2024)", min_value=datetime(2024, 6, 29), value=datetime(2025, 6, 1))

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
