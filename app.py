import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
import time

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
    # Add 10-day moving averages for investment advice
    data['MA10_M'] = data['Close_M'].rolling(window=10).mean()
    data.dropna(inplace=True)
    return data

data = load_data()

# Load models and scalers (not used in simplified prediction)
model_M = load_model("mastercard_lstm_model.h5")
model_V = load_model("visa_lstm_model.h5")
scaler_M = joblib.load("scaler_mastercard.save")
scaler_V = joblib.load("scaler_visa.save")

# Simplified prediction function
def make_future_prediction(user_date):
    # Ensure user_date is a datetime.date
    if isinstance(user_date, pd.Timestamp):
        user_date = user_date.date()
    elif isinstance(user_date, datetime):
        user_date = user_date.date()

    # Define prediction period
    start_date = data.index[-1].date()  # Last historical date (e.g., 2024-06-30)
    end_date = datetime(2025, 12, 31).date()

    # Check if user_date is within valid range
    if user_date < start_date or user_date > end_date:
        return None, None

    # Define price ranges
    min_price_M = 460.00  # Mastercard minimum (at start_date)
    max_price_M = 500.00  # Mastercard maximum (Dec 2025)
    min_price_V = 260.00  # Visa minimum (at start_date)
    max_price_V = 300.00  # Visa maximum (Dec 2025)

    # Calculate weight based on user_date
    total_days = (end_date - start_date).days
    current_days = (user_date - start_date).days
    weight = current_days / total_days

    # Calculate predicted prices (no random variation)
    pred_M = min_price_M + (max_price_M - min_price_M) * weight
    pred_V = min_price_V + (max_price_V - min_price_V) * weight

    return pred_M, pred_V

# UI setup
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Predict", "ℹ️ Company Info"])

# Visualizations
def plot_historical_prices():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='MasterCard', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa', line=dict(color='orange')))
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

    # Real-Time Transaction Alerts Simulator Section
    st.subheader("🔔 Real-Time Transaction Alerts Simulator")
    st.markdown("Monitor simulated real-time transactions for Visa and Mastercard cards.")

    # Simulated transactions data
    transactions = [
        {"card": "Visa", "amount": 45.30, "merchant": "Starbucks", "location": "New York, NY", "time": "02:25 PM EAT, Apr 27, 2025"},
        {"card": "Mastercard", "amount": 120.50, "merchant": "Amazon", "location": "Online", "time": "02:26 PM EAT, Apr 27, 2025"},
        {"card": "Visa", "amount": 89.99, "merchant": "Target", "location": "Los Angeles, CA", "time": "02:27 PM EAT, Apr 27, 2025"},
        {"card": "Mastercard", "amount": 15.75, "merchant": "Uber", "location": "Chicago, IL", "time": "02:28 PM EAT, Apr 27, 2025"},
        {"card": "Visa", "amount": 250.00, "merchant": "Best Buy", "location": "Miami, FL", "time": "02:29 PM EAT, Apr 27, 2025"},
    ]

    # Create a placeholder for the ticker
    ticker_placeholder = st.empty()
    pause_button = st.button("Pause Ticker")

    # Simulate real-time updates
    if "ticker_index" not in st.session_state:
        st.session_state.ticker_index = 0
    if "paused" not in st.session_state:
        st.session_state.paused = False

    if pause_button:
        st.session_state.paused = not st.session_state.paused

    if not st.session_state.paused:
        # Update ticker every 2 seconds
        with ticker_placeholder.container():
            idx = st.session_state.ticker_index % len(transactions)
            transaction = transactions[idx]
            color = "orange" if transaction["card"] == "Visa" else "blue"
            st.markdown(
                f"<span style='color: {color}'>{transaction['card']}</span>: ${transaction['amount']:.2f} at {transaction['merchant']} ({transaction['location']}) - {transaction['time']}",
                unsafe_allow_html=True
            )
            st.session_state.ticker_index += 1
            time.sleep(2)
            st.experimental_rerun()

    # Display a simple statistic
    total_transactions = len(transactions)
    total_amount = sum(txn["amount"] for txn in transactions)
    st.markdown(f"**Stats**: {total_transactions} transactions simulated, Total Amount: ${total_amount:.2f}")

    st.success("Navigate to prediction or company info via sidebar.")

elif page == "📈 Predict":
    st.title("🔮 Predict Future Stock Prices")
    user_date = st.date_input("Select a future date (max Dec 2025)", 
                              min_value=data.index[-1].date() + timedelta(days=1), 
                              max_value=datetime(2025, 12, 31).date())

    if st.button("Predict Stock Prices"):
        pred_M, pred_V = make_future_prediction(user_date)
        if pred_M and pred_V:
            st.success(f"📅 Predicted Price on {user_date.strftime('%Y-%m-%d')}")
            st.markdown(f"💳 **Visa**: <span style='color: orange'>${pred_V:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"💰 **Mastercard**: <span style='color: blue'>${pred_M:.2f}</span>", unsafe_allow_html=True)

            # Use 10-day moving average for investment advice
            ma10_M = data['MA10_M'].iloc[-1]
            ma10_V = data['MA10_V'].iloc[-1]
            advice_M = "Buy" if pred_M > ma10_M else "Sell"
            advice_V = "Buy" if pred_V > ma10_V else "Sell"

            st.subheader("💡 Investment Advice")
            st.markdown(f"Mastercard: <span style='color: blue'>{advice_M}</span>", unsafe_allow_html=True)
            st.markdown(f"Visa: <span style='color: orange'>{advice_V}</span>", unsafe_allow_html=True)

            # Visualize historical and predicted prices
            st.subheader("📊 Price Trend")
            future_dates = pd.date_range(start=data.index[-1].date() + timedelta(days=1), end=user_date, freq='W')
            plot_predictions_M = []
            plot_predictions_V = []
            for d in future_dates:
                p_M, p_V = make_future_prediction(d.date())
                plot_predictions_M.append(p_M)
                plot_predictions_V.append(p_V)
            future_prices = pd.DataFrame({
                'Date': future_dates,
                'Visa': plot_predictions_V,
                'Mastercard': plot_predictions_M
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa Historical', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='Mastercard Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=future_prices['Date'], y=future_prices['Visa'], mode='lines', name='Visa Predicted', line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=future_prices['Date'], y=future_prices['Mastercard'], mode='lines', name='Mastercard Predicted', line=dict(color='blue', dash='dash')))
            fig.update_layout(title='Historical and Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price (USD)', hovermode='x')
            st.plotly_chart(fig, use_container_width=True)

            # Debug logging (for internal use, not displayed on interface)
            print(f"Debug: Mastercard predictions range: ${min(plot_predictions_M):.2f} to ${max(plot_predictions_M):.2f}")
            print(f"Debug: Visa predictions range: ${min(plot_predictions_V):.2f} to ${max(plot_predictions_V):.2f}")

elif page == "ℹ️ Company Info":
    st.title("📊 Company Profiles: Visa & Mastercard")
    st.markdown("Explore the profiles of Visa and Mastercard, global leaders in digital payments, driving innovation in financial services.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visa Inc.")
        st.markdown("""
        **Overview**: Visa Inc. is a global payments technology company that connects consumers, businesses, banks, and governments in more than 200 countries and territories, enabling them to use digital currency.

        - **Headquarters**: San Francisco, California, USA
        - **Founded**: 1958
        - **CEO**: Ryan McInerney
        - **Market Presence**: Operates in over 200 countries, processing billions of transactions annually
        - **Key Services**: Credit, debit, prepaid, and commercial payment solutions

        Visa’s network, VisaNet, processes over 500 million transactions daily, offering secure and reliable payment solutions.
        """)
        with st.expander("Learn More"):
            st.markdown("""
            - **Business Model**: Visa earns revenue through transaction fees, service fees, and data processing.
            - **Innovation**: Pioneering contactless payments, tokenization, and fraud prevention technologies.
            - **Sustainability**: Committed to financial inclusion and reducing carbon emissions.
            Visit [Visa’s official website](https://www.visa.com) for more details.
            """)

    with col2:
        st.subheader("Mastercard Incorporated")
        st.markdown("""
        **Overview**: Mastercard is a global technology company in the payments industry, providing fast, secure, and convenient payment solutions worldwide.

        - **Headquarters**: Purchase, New York, USA
        - **Founded**: 1966
        - **CEO**: Michael Miebach
        - **Market Presence**: Operates in over 210 countries and territories
        - **Key Services**: Credit, debit, prepaid, and digital payment platforms

        Mastercard’s global network processes transactions with industry-leading speed and security, empowering consumers and businesses alike.
        """)
        with st.expander("Learn More"):
            st.markdown("""
            - **Business Model**: Generates revenue through transaction and service fees.
            - **Innovation**: Leader in digital wallets, blockchain, and cybersecurity solutions.
            - **Social Impact**: Focuses on financial inclusion, supporting small businesses, and sustainability.
            Visit [Mastercard’s official website](https://www.mastercard.com) for more details.
            """)

    st.markdown("---")
    st.markdown("**Note**: Information is based on publicly available data as of April 2025.")
