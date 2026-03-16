import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from predictor import fetch_data, process_data, train_anomaly_detector, get_latest_risk_summary

st.set_page_config(page_title="Stablecoin De-Peg Predictor 10x", layout="wide", page_icon="📈")

# --- Custom CSS for aesthetic improvements ---
st.markdown("""
<style>
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-normal { color: #00cc66; font-weight: bold; }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Stablecoin De-Peg Risk Dashboard")
st.markdown("""
This advanced dashboard provides real-time monitoring of major stablecoins using **Machine Learning (Isolation Forest)**
and **Technical Analysis (Bollinger Bands, Volatility, Volume Flow)**.
""")

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_and_predict():
    raw_dfs = fetch_data()
    features_dict = process_data(raw_dfs)
    predictions_dict = train_anomaly_detector(features_dict)
    summary_df = get_latest_risk_summary(predictions_dict)
    return predictions_dict, summary_df

with st.spinner('Fetching 1 year of market data & executing ML models...'):
    predictions, summary = load_and_predict()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🔍 Detailed Coin Analysis", "⚠️ Alert History"])

# ==========================================
# TAB 1: Market Overview
# ==========================================
with tab1:
    st.subheader("Live Market Risk Matrix")

    # Render Metrics in a Grid
    cols = st.columns(len(summary))
    for i, col in enumerate(cols):
        coin_data = summary.iloc[i]
        coin_name = coin_data['Coin']

        status_class = "risk-high" if coin_data['Risk Status'] == 'High Risk' else "risk-normal"

        with col:
            st.markdown(f"<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                label=f"{coin_name} Price",
                value=f"${coin_data['Current Price']:.4f}",
                delta=f"Risk Score: {coin_data['Risk Score']:.0f}/100",
                delta_color="inverse"
            )
            st.markdown(f"Status: <span class='{status_class}'>{coin_data['Risk Status']}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Risk Summary Table")
    # Style the dataframe
    styled_summary = summary.style.applymap(
        lambda x: 'color: red; font-weight: bold' if x == 'High Risk' else 'color: green',
        subset=['Risk Status']
    ).format({
        'Current Price': '${:.4f}',
        'Peg Deviation': '{:.5f}',
        'Volume Ratio': '{:.2f}x',
        'Risk Score': '{:.1f}'
    })
    st.dataframe(styled_summary, use_container_width=True)


# ==========================================
# TAB 2: Detailed Coin Analysis
# ==========================================
with tab2:
    selected_coin = st.selectbox("Select a stablecoin for deep dive:", [col.replace('-USD', '') for col in predictions.keys()])
    selected_key = f"{selected_coin}-USD"

    if selected_key in predictions:
        df = predictions[selected_key]

        # Create dual-axis Plotly chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f"{selected_coin} Price & Bollinger Bands", "Trading Volume"),
                            row_heights=[0.7, 0.3])

        # Price Candlesticks
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='Price'), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Upper BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Lower BB', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[1.0]*len(df), line=dict(color='green', width=2), name='$1 Peg'), row=1, col=1)

        # Anomalies
        anomalies = df[df['Risk_Flag']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'],
                                     mode='markers', marker=dict(color='red', size=10, symbol='x'),
                                     name='Detected Anomaly'), row=1, col=1)

        # Volume Bar Chart
        colors = ['red' if df.iloc[i]['Close'] < df.iloc[i-1]['Close'] else 'green' for i in range(1, len(df))]
        colors.insert(0, 'green') # first element

        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA_30'], line=dict(color='orange', width=2), name='30d Vol MA'), row=2, col=1)

        # Layout adjustments
        fig.update_layout(height=700, template='plotly_white', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Risk Score gauge
        latest_score = df.iloc[-1]['Risk_Score']
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_score,
            title = {'text': "Current ML Risk Score"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "darkblue"},
                     'steps' : [
                         {'range': [0, 30], 'color': "lightgreen"},
                         {'range': [30, 70], 'color': "yellow"},
                         {'range': [70, 100], 'color': "red"}]}
        ))
        gauge.update_layout(height=300)
        st.plotly_chart(gauge, use_container_width=True)

# ==========================================
# TAB 3: Alert History
# ==========================================
with tab3:
    st.subheader("Recent De-Peg Risk Alerts (Last 30 Days)")

    all_anomalies = []
    for coin, df in predictions.items():
        recent_df = df.tail(30)
        anomalies = recent_df[recent_df['Risk_Flag']]
        for date, row in anomalies.iterrows():
            all_anomalies.append({
                'Date': date.strftime("%Y-%m-%d"),
                'Coin': coin.replace('-USD', ''),
                'Price at Alert': f"${row['Close']:.4f}",
                'Risk Score': f"{row['Risk_Score']:.1f}",
                'Volatility': f"{row['Volatility_7d']:.5f}"
            })

    if all_anomalies:
        alerts_df = pd.DataFrame(all_anomalies).sort_values(by='Date', ascending=False)
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.success("No high-risk anomalies detected in the last 30 days. Market is stable! ✅")
