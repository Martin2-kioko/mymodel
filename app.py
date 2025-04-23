import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Set Streamlit page config
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")

# --- Load & preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("MVS.csv")
    
    # Auto-detect and parse date column
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break
        except:
            continue

    df['MA10_V'] = df['Close_V'].rolling(window=10).mean()
    df['MA20_V'] = df['Close_V'].rolling(window=20).mean()
    df['Volatility_V'] = df['Close_V'].rolling(window=10).std()

    df.dropna(inplace=True)
    return df

data = load_data()

# Load models and scalers
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# --- Prediction ---
def make_future_prediction(user_date):
    # Ensure that the future date is after the last historical date
    future_days = (user_date - data.index[-1].date()).days
    if future_days <= 0:
        st.error("Select a date in the future after the latest available date.")
        return None, None

    # Use the last 60 days of data to make predictions
    # Mastercard: last 60 days, reshape to (1, 60, 1)
    past_M = data['Close_M'].tail(60).values.reshape(-1, 1)
    scaled_M = scaler_M.transform(past_M).reshape(1, 60, 1)
    
    # Visa: last 60 days with 4 features
    past_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].tail(60).values
    scaled_V = scaler_V.transform(past_V).reshape(1, 60, 4)

    # Predict the next step for both Visa and Mastercard
    pred_M_scaled = model_M.predict(scaled_M)
    pred_V_scaled = model_V.predict(scaled_V)

    # Inverse scale the predictions
    pred_M = scaler_M.inverse_transform(pred_M_scaled)[0][0]
    pred_V = scaler_V.inverse_transform(
        np.hstack((pred_V_scaled, np.zeros((1, 3))))
    )[0][0]

    return pred_M, pred_V

# --- Plotting ---
def plot_historical_prices():
    st.subheader("📉 Stock Prices (2008–2024)")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='Mastercard', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa', line=dict(color='blue')))

    fig.update_layout(
        title="Mastercard & Visa Stock Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    st.subheader("📊 Yearly Trading Volume")
    yearly = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    yearly.index = yearly.index.astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly.index, y=yearly['Volume_M'], name="Mastercard", marker_color='blue'))
    fig.add_trace(go.Bar(x=yearly.index, y=yearly['Volume_V'], name="Visa", marker_color='orange'))

    fig.update_layout(
        title="Yearly Trading Volume",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Volume",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- UI ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Info"])

if page == "🏠 Home":
    st.title("🏦 Mastercard & Visa - Market Overview")
    col1, col2 = st.columns(2)
    with col1:
        plot_historical_prices()
    with col2:
        plot_volumes()
    st.success("Use the sidebar to explore predictions or company info.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    user_date = st.date_input("Select a future date:", min_value=datetime(2024, 6, 29).date(), value=datetime(2025, 6, 1).date())
    
    if st.button("Predict"):
        pred_M, pred_V = make_future_prediction(user_date)
        if pred_M and pred_V:
            st.success(f"📆 Prediction for {user_date}")
            st.write(f"💰 **Mastercard**: ${pred_M:.2f}")
            st.write(f"💳 **Visa**: ${pred_V:.2f}")
            
            st.subheader("💡 Investment Suggestion")
            last_M, last_V = data['Close_M'].iloc[-1], data['Close_V'].iloc[-1]
            st.write(f"Mastercard: {'Buy' if pred_M > last_M else 'Sell'}")
            st.write(f"Visa: {'Buy' if pred_V > last_V else 'Sell'}")

elif page == "ℹ️ Info":
    st.title("ℹ️ Company Info")
    st.write("""
    Visa and Mastercard are two of the leading global payment technology companies.
    
    - **Visa Inc.**: Headquartered in Foster City, CA, Visa facilitates electronic funds transfers globally.
    - **Mastercard Inc.**: Based in Purchase, NY, Mastercard connects consumers, financial institutions, and businesses across the world.
    """)

