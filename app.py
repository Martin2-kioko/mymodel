import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    
    return data

# Load data
data = load_data()

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
    
    # Plot the closing prices for Mastercard and Visa
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close_M'], label='MasterCard Close', color='green')
    plt.plot(data.index, data['Close_V'], label='Visa Close', color='blue')
    plt.title('Stock Prices of MasterCard and Visa')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_volumes():
    st.subheader("Yearly Trading Volume: 2008–2024")
    
    # Function to format Y-axis labels as USD with billions
    def billions(x, pos):
        return f'${x * 1e-9:.1f}B'
    
    # Grouping yearly volume data
    yearly_volume = data.groupby(data.index.year)[['Volume_M', 'Volume_V']].sum()
    yearly_volume.index = yearly_volume.index.astype(str)
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.4
    years = yearly_volume.index
    x = range(len(years))
    
    # Plotting bar chart for yearly volume of Mastercard and Visa
    plt.bar([i - bar_width/2 for i in x], yearly_volume['Volume_M'],
            width=bar_width, label='Mastercard', color='blue')
    plt.bar([i + bar_width/2 for i in x], yearly_volume['Volume_V'],
            width=bar_width, label='Visa', color='orange')

    plt.xlabel('Year')
    plt.ylabel('Trading Volume (USD)')
    plt.title('Yearly Trading Volume for Mastercard and Visa (2008–2024)')
    plt.xticks(ticks=x, labels=years, rotation=45)
    plt.legend()
    
    # Format Y-axis as billions of USD
    plt.gca().yaxis.set_major_formatter(FuncFormatter(billions))
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

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
