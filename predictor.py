import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore') # Suppress yfinance and TA warnings for cleaner output

def fetch_data(tickers=['USDT-USD', 'USDC-USD', 'DAI-USD', 'FDUSD-USD', 'USDD-USD'], period='1y'):
    """Fetches historical OHLCV data from Yahoo Finance."""
    # Download with actions=False to avoid dividend/split columns
    data = yf.download(tickers, period=period, actions=False)

    # Restructure the MultiIndex dataframe into a dictionary of DataFrames per coin
    coin_dfs = {}
    for ticker in tickers:
        try:
            # Extract OHLCV for specific ticker
            df = data.xs(ticker, axis=1, level=1).copy()
            # Forward fill then backward fill missing values
            df = df.ffill().bfill()
            coin_dfs[ticker] = df
        except KeyError:
            print(f"Warning: Could not fetch data for {ticker}")

    return coin_dfs

def process_data(coin_dfs):
    """Calculates advanced features: Peg Deviation, Volatility, Volume MA, Bollinger Bands."""
    features = {}
    for coin, df in coin_dfs.items():
        if df.empty or 'Close' not in df.columns:
            continue

        # 1. Percentage deviation from $1.00 peg (Absolute)
        df['Peg_Deviation'] = abs(df['Close'] - 1.0)

        # 2. Rolling 7-day price volatility (Standard Deviation)
        df['Volatility_7d'] = df['Close'].rolling(window=7).std().fillna(0)

        # 3. Volume Spike Indicator (Ratio of current volume to 30-day average)
        df['Volume_MA_30'] = df['Volume'].rolling(window=30).mean().replace(0, 1) # avoid div by zero
        df['Volume_Spike_Ratio'] = df['Volume'] / df['Volume_MA_30']
        df['Volume_Spike_Ratio'] = df['Volume_Spike_Ratio'].fillna(1.0)

        # 4. Bollinger Bands (20-day, 2 std dev) Width
        # Wide bands indicate high volatility / potential de-peg pressure
        indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Width'] = indicator_bb.bollinger_wband().fillna(0)
        df['BB_Lower'] = indicator_bb.bollinger_lband()
        df['BB_Upper'] = indicator_bb.bollinger_hband()

        features[coin] = df.dropna()

    return features

def train_anomaly_detector(features_dict):
    """Trains an advanced Isolation Forest model using multiple scaled features."""
    results = {}
    for coin, df in features_dict.items():
        if df.empty:
            continue

        # Define the feature set for anomaly detection
        feature_cols = ['Peg_Deviation', 'Volatility_7d', 'Volume_Spike_Ratio', 'BB_Width']
        X = df[feature_cols]

        # Scale features so that large volume spikes don't overwhelm small price deviations
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a robust Isolation Forest
        model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.03, # Assume 3% of days might be anomalous/high-risk
            random_state=42
        )

        # Predict anomalies (-1 for anomaly, 1 for normal)
        preds = model.fit_predict(X_scaled)

        # Also get anomaly scores (lower is more anomalous)
        scores = model.decision_function(X_scaled)

        df['Risk_Flag'] = preds == -1
        # Normalize risk score to 0-100 (100 being highest risk)
        # decision_function typically returns values between -0.5 and 0.5
        # We invert it so higher score = higher risk
        df['Risk_Score'] = ((0.5 - scores) * 100)
        df['Risk_Score'] = df['Risk_Score'].clip(lower=0, upper=100)

        results[coin] = df

    return results

def get_latest_risk_summary(results_dict):
    """Generates a comprehensive summary of the latest day's risk status."""
    summary = []
    for coin, df in results_dict.items():
        if df.empty:
            continue

        latest = df.iloc[-1]
        summary.append({
            'Coin': coin.replace('-USD', ''),
            'Current Price': float(latest['Close']),
            'Peg Deviation': float(latest['Peg_Deviation']),
            'Volume Ratio': float(latest['Volume_Spike_Ratio']),
            'Risk Score': float(latest['Risk_Score']),
            'Risk Status': 'High Risk' if latest['Risk_Flag'] else 'Normal'
        })
    return pd.DataFrame(summary)

if __name__ == "__main__":
    print("Fetching advanced market data (1 year OHLCV)...")
    raw_dfs = fetch_data()
    print("Calculating technical indicators and features...")
    features = process_data(raw_dfs)
    print("Training scaled Isolation Forest models...")
    predictions = train_anomaly_detector(features)
    print("Latest Advanced Risk Summary:")
    summary_df = get_latest_risk_summary(predictions)
    print(summary_df)
