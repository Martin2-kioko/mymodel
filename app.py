import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model

# Function to load data dynamically by detecting the date column
def load_data():
    # Read CSV file
    data = pd.read_csv("MVS.csv")
    
    # Automatically detect the date column
    date_column = None
    for col in data.columns:
        try:
            # Try to convert to datetime
            pd.to_datetime(data[col])
            date_column = col
            break
        except:
            continue
    
    if date_column is None:
        raise ValueError("No datetime column found in the CSV file.")

    # Set the date column as index and parse it as dates
    data[date_column] = pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    
    # Feature Engineering: Ensure the columns for moving averages and volatility are being calculated correctly
    data['MA10_V'] = data['Close_V'].rolling(window=10).mean()
    data['MA20_V'] = data['Close_V'].rolling(window=20).mean()
    data['Volatility_V'] = data['Close_V'].rolling(window=10).std()

    # Drop NaNs after rolling
    data.dropna(inplace=True)
    
    return data

# Load data
data = load_data()

# Display column names to help debug
st.write("Columns in the dataset:", data.columns)

# Load models and scalers
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# --- UI setup ---
st.set_page_config(page_title="Visa & Mastercard Stocks", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

# --- Utility functions ---
def plot_historical_prices():
    st.subheader("Stock Prices: 2008–2024")
    
    # Plot the closing prices for Mastercard and Visa with interactivity using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close_M'], 
        mode='lines', 
        name='MasterCard Close', 
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close_V'], 
        mode='lines', 
        name='Visa Close', 
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Stock Prices of MasterCard and Visa',
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_volumes():
    st.subheader("Yearly Trading Volume: 2008–2024")
    
    # Grouping yearly volume data
    yearly_volume = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    yearly_volume.index = yearly_volume.index.astype(str)
    
    # Plotting bar chart for yearly volume of Mastercard and Visa with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=yearly_volume.index, 
        y=yearly_volume['Volume_M'],
        name='Mastercard', 
        marker=dict(color='blue'),
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        x=yearly_volume.index, 
        y=yearly_volume['Volume_V'],
        name='Visa', 
        marker=dict(color='orange'),
        opacity=0.7
    ))

    fig.update_layout(
        title='Yearly Trading Volume for Mastercard and Visa (2008–2024)',
        xaxis_title='Year',
        yaxis_title='Trading Volume (USD)',
        barmode='group',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Page logic ---
if page == "🏠 Home":
    st.title("🏦 Visa & Mastercard - Stock Market Overview")
    
    # Display stock prices and volumes in a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        plot_historical_prices()  # Stock Prices chart
        
    with col2:
        plot_volumes()  # Volume Traded chart
        
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

            # Buy/Sell Advice
            st.subheader("💡 Investment Advice")
            advice_M = "Buy" if pred_M < data['Close_M'].iloc[-1] else "Sell"
            advice_V = "Buy" if pred_V < data['Close_V'].iloc[-1] else "Sell"
            st.write(f"➡️ **Mastercard Advice**: {advice_M}")
            st.write(f"➡️ **Visa Advice**: {advice_V}")

            # Plot with future point
            future_date = pd.to_datetime(user_date)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], name='Mastercard Historical', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=[future_date], y=[pred_M], name='Mastercard Prediction', mode='markers+lines', line=dict(color='darkgreen', dash='dot')))

            fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], name='Visa Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[future_date], y=[pred_V], name='Visa Prediction', mode='markers+lines', line=dict(color='navy', dash='dot')))

            fig.update_layout(title="Stock Prices with Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ Company Info":
    st.title("ℹ️ About Visa and Mastercard")
    st.subheader("Visa Inc.")
    st.markdown("""
    Visa Inc. is a world leader in digital payments, facilitating transactions between consumers, merchants, and financial institutions across more than 200 countries.
    
    - **Ticker**: V
    - **Market Cap**: $500B+
    - **Founded**: 1958
    - **Headquarters**: Foster City, California
    """)

    st.subheader("Mastercard Inc.")
    st.markdown("""
    Mastercard is a global technology company in the payments industry. Their mission is to connect and power an inclusive digital economy.

    - **Ticker**: MA
    - **Market Cap**: $400B+
    - **Founded**: 1966
    - **Headquarters**: Purchase, New York
    """)

# --- Prediction Function ---
def make_future_prediction(user_date):
    # Check for the columns in data to avoid KeyError
    if 'Close_V' not in data.columns or 'MA10_V' not in data.columns or 'MA20_V' not in data.columns or 'Volatility_V' not in data.columns:
        st.error("Required columns for prediction are missing.")
        return None, None

    # Get the most recent data point for both Visa and Mastercard
    latest_data_M = data[['Close_M']].tail(1)  # Only the closing price for Mastercard
    latest_data_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].tail(1)  # Multiple features for Visa

    # Rescale the data
    latest_data_M_scaled = scaler_M.transform(latest_data_M.values.reshape(1, -1))
    latest_data_V_scaled = scaler_V.transform(latest_data_V.values.reshape(1, -1))

    # Make predictions for the selected future date
    pred_M_scaled = model_M.predict(latest_data_M_scaled)
    pred_V_scaled = model_V.predict(latest_data_V_scaled)

    # Inverse scaling
    pred_M = scaler_M.inverse_transform(pred_M_scaled)
    pred_V = scaler_V.inverse_transform(np.hstack((pred_V_scaled, np.zeros((pred_V_scaled.shape[0], 3)))) )[:, 0]

    return pred_M[0][0], pred_V[0]
