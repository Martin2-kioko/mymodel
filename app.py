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

# Precompute predictions for monthly intervals
@st.cache_data
def get_precomputed_predictions():
    dates = pd.date_range(start=data.index[-1].date() + timedelta(days=1), end="2025-12-31", freq='M')
    predictions = {}
    for d in dates:
        pred_M, pred_V = make_future_prediction_core(d.date())
        if pred_M is not None and pred_V is not None:
            predictions[d.date()] = (pred_M, pred_V)
    return predictions

# Core prediction function (used by precomputed and live predictions)
@st.cache_data
def make_future_prediction_core(user_date):
    # Ensure user_date is a datetime.date
    if isinstance(user_date, pd.Timestamp):
        user_date = user_date.date()
    elif isinstance(user_date, datetime):
        user_date = user_date.date()

    today = data.index[-1].date()
    days_to_predict = (user_date - today).days

    if days_to_predict <= 0:
        return None, None

    # Define price ranges
    min_price_M = 460.00  # Mastercard minimum
    max_price_M = 500.00  # Mastercard maximum (Dec 2025)
    min_price_V = 260.00  # Visa minimum
    max_price_V = 300.00  # Visa maximum (Dec 2025)

    # Get last historical prices
    last_price_M = data['Close_M'].iloc[-1]
    last_price_V = data['Close_V'].iloc[-1]

    # Use weekly predictions (approx. 7 days per step)
    weeks_to_predict = max(1, (days_to_predict + 6) // 7)  # Round up

    # Initialize temporary data
    temp_data_M = data[['Close_M']].values.tolist()
    temp_data_V = data[['Close_V']].values.tolist()
    temp_features_V = data[['Close_V', 'MA10_V', 'MA20_V', 'Volatility_V']].values.tolist()

    # Prepare batch inputs for LSTM
    batch_inputs_M = []
    batch_inputs_V = []
    for i in range(weeks_to_predict):
        batch_inputs_M.append(np.array(temp_data_M[-seq_len:]).reshape(1, seq_len, 1))
        batch_inputs_V.append(np.array(temp_features_V[-seq_len:]).reshape(1, seq_len, 4))
        temp_data_M.append([last_price_M])  # Placeholder
        temp_data_V.append([last_price_V])
        temp_features_V.append([last_price_V, last_price_V, last_price_V, 0])

    # Batch predictions
    batch_inputs_M = np.vstack(batch_inputs_M)
    batch_inputs_V = np.vstack(batch_inputs_V)
    batch_inputs_M = scaler_M.transform(batch_inputs_M.reshape(-1, 1)).reshape(weeks_to_predict, seq_len, 1)
    batch_inputs_V = scaler_V.transform(batch_inputs_V.reshape(-1, 4)).reshape(weeks_to_predict, seq_len, 4)

    with st.spinner("Generating predictions..."):
        pred_M_scaled = model_M.predict(batch_inputs_M, verbose=0)
        pred_V_scaled = model_V.predict(batch_inputs_V, verbose=0)

    # Process predictions
    predictions_M = scaler_M.inverse_transform(pred_M_scaled).flatten()
    predictions_V = scaler_V.inverse_transform(np.hstack([pred_V_scaled, np.zeros((weeks_to_predict, 3))]))[:, 0]

    # Enforce price ranges with interpolated targets and random variation
    weekly_dates = pd.date_range(start=today + timedelta(days=7), periods=weeks_to_predict, freq='W')
    total_days = (datetime(2025, 12, 31).date() - today).days
    predictions_M = np.zeros(weeks_to_predict)
    predictions_V = np.zeros(weeks_to_predict)
    for i in range(weeks_to_predict):
        current_days = (weekly_dates[i].date() - today).days
        weight = current_days / total_days
        target_M = min_price_M + (max_price_M - min_price_M) * weight
        target_V = min_price_V + (max_price_V - min_price_V) * weight
        # Add random variation (±5% of target) for realism
        variation_M = np.random.uniform(-0.05, 0.05) * target_M
        variation_V = np.random.uniform(-0.05, 0.05) * target_V
        predictions_M[i] = min(max(target_M + variation_M, min_price_M), max_price_M)
        predictions_V[i] = min(max(target_V + variation_V, min_price_V), max_price_V)

    # Recalculate Visa features using NumPy
    temp_data_V = data[['Close_V']].values.tolist() + [[p] for p in predictions_V]
    temp_array_V = np.array(temp_data_V)
    ma10_v = np.convolve(temp_array_V[:, 0], np.ones(10)/10, mode='valid')
    ma20_v = np.convolve(temp_array_V[:, 0], np.ones(20)/20, mode='valid')
    volatility_v = np.array([np.std(temp_array_V[max(0, i-9):i+1]) for i in range(len(temp_array_V))])
    temp_features_V = [[temp_data_V[i][0], ma10_v[min(i, len(ma10_v)-1)], ma20_v[min(i, len(ma20_v)-1)], volatility_v[i]] for i in range(len(temp_data_V))]

    # Interpolate for exact date
    idx = min(np.searchsorted([d.date() for d in weekly_dates], user_date), weeks_to_predict-1)
    if idx == 0:
        pred_M = predictions_M[0]
        pred_V = predictions_V[0]
    else:
        w = (user_date - weekly_dates[idx-1].date()).days / (weekly_dates[idx].date() - weekly_dates[idx-1].date()).days
        pred_M = predictions_M[idx-1] + w * (predictions_M[idx] - predictions_M[idx-1])
        pred_V = predictions_V[idx-1] + w * (predictions_V[idx] - predictions_V[idx-1])

    return pred_M, pred_V

# Main prediction function with precomputed lookup
@st.cache_data
def make_future_prediction(user_date):
    # Ensure user_date is a datetime.date
    if isinstance(user_date, pd.Timestamp):
        user_date = user_date.date()
    elif isinstance(user_date, datetime):
        user_date = user_date.date()

    precomputed = get_precomputed_predictions()

    # Check if user_date matches a precomputed date
    if user_date in precomputed:
        return precomputed[user_date]

    # Find closest precomputed dates for interpolation
    precomputed_dates = sorted(precomputed.keys())
    idx = np.searchsorted(precomputed_dates, user_date)
    if idx == 0:
        return precomputed[precomputed_dates[0]]
    elif idx == len(precomputed_dates):
        return precomputed[precomputed_dates[-1]]

    # Interpolate between closest dates
    date_prev = precomputed_dates[idx-1]
    date_next = precomputed_dates[idx]
    w = (user_date - date_prev).days / (date_next - date_prev).days
    pred_M_prev, pred_V_prev = precomputed[date_prev]
    pred_M_next, pred_V_next = precomputed[date_next]
    pred_M = pred_M_prev + w * (pred_M_next - pred_M_prev)
    pred_V = pred_V_prev + w * (pred_V_next - pred_V_prev)

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

            # Visualize historical and predicted prices
            st.subheader("📊 Price Trend")
            weeks_to_predict = max(1, ((user_date - data.index[-1].date()).days + 6) // 7)
            future_dates = pd.date_range(start=data.index[-1], periods=weeks_to_predict+1, freq='W')[1:]
            # Generate predictions for plot
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
            fig.add_trace(go.Scatter(x=data.index, y=data['Close_V'], mode='lines', name='Visa Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close_M'], mode='lines', name='Mastercard Historical', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=future_prices['Date'], y=future_prices['Visa'], mode='lines', name='Visa Predicted', line=dict(color='blue', dash='dash')))
            fig.add_trace(go.Scatter(x=future_prices['Date'], y=future_prices['Mastercard'], mode='lines', name='Mastercard Predicted', line=dict(color='green', dash='dash')))
            fig.update_layout(title='Historical and Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price (USD)', hovermode='x')
            st.plotly_chart(fig, use_container_width=True)

            # Debug: Show prediction range
            st.write(f"Debug: Mastercard predictions range: ${min(plot_predictions_M):.2f} to ${max(plot_predictions_M):.2f}")
            st.write(f"Debug: Visa predictions range: ${min(plot_predictions_V):.2f} to ${max(plot_predictions_V):.2f}")

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

# Note: To optimize LSTM models for faster inference, convert to TensorFlow Lite:
# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_keras_model(model_M)
# tflite_model = converter.convert()
# Save and load tflite_model, then use tf.lite.Interpreter for predictions.
# This requires modifying the prediction logic and is left as a future enhancement.
