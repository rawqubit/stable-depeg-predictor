import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predictor import fetch_data, process_data, train_anomaly_detector, get_latest_risk_summary

st.set_page_config(page_title="Stablecoin De-Peg Predictor", layout="wide")

st.title("Stablecoin De-Peg Risk Dashboard")
st.markdown("""
This dashboard monitors the top stablecoins for de-pegging risks. It uses on-chain price data, calculates
rolling volatility and peg deviations, and employs an Isolation Forest Machine Learning model to detect anomalous market behaviors.
""")

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_predict():
    raw_df = fetch_data()
    features_dict = process_data(raw_df)
    predictions_dict = train_anomaly_detector(features_dict)
    summary_df = get_latest_risk_summary(predictions_dict)
    return predictions_dict, summary_df

with st.spinner('Fetching on-chain data and running predictive models...'):
    predictions, summary = load_and_predict()

# --- Display Current Metrics ---
st.subheader("Current Market Status")

cols = st.columns(len(summary))
for i, col in enumerate(cols):
    coin_data = summary.iloc[i]
    coin_name = coin_data['Coin']
    price = coin_data['Current Price']
    status = coin_data['Risk Status']

    # Color code the status
    if status == 'Normal':
        status_color = 'green'
    else:
        status_color = 'red'

    col.metric(
        label=f"{coin_name} Price",
        value=f"${price:.4f}",
        delta=f"Peg Dev: {coin_data['Peg Deviation']:.4f}",
        delta_color="inverse" # higher deviation is bad (red)
    )
    col.markdown(f"**Risk:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)


# --- Display Summary Table ---
st.subheader("Risk Summary Data")
st.dataframe(summary, use_container_width=True)

# --- Historical Charts ---
st.subheader("Historical Price and Anomalies")
selected_coin = st.selectbox("Select a stablecoin to view historical trends:", [col.replace('-USD', '') for col in predictions.keys()])
selected_key = f"{selected_coin}-USD"

if selected_key in predictions:
    coin_df = predictions[selected_key]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(coin_df.index, coin_df['Price'], label='Price', color='blue', alpha=0.6)

    # Highlight anomalies
    anomalies = coin_df[coin_df['Risk_Flag']]
    if not anomalies.empty:
        ax.scatter(anomalies.index, anomalies['Price'], color='red', label='De-Peg Risk Detected', zorder=5)

    ax.axhline(y=1.0, color='green', linestyle='--', label='$1 Peg')
    ax.set_title(f"{selected_coin} Historical Price and Detected Anomalies")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)
