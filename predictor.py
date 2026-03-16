import yfinance as yf
import pandas as pd
from sklearn.ensemble import IsolationForest

def fetch_data(tickers=['USDT-USD', 'USDC-USD', 'DAI-USD'], period='60d'):
    """Fetches historical price data from Yahoo Finance."""
    data = yf.download(tickers, period=period)
    # yfinance returns a MultiIndex column DataFrame if multiple tickers are provided.
    # We want the 'Close' prices.
    df = data['Close']

    # Fill any missing values forward then backward
    df = df.ffill().bfill()
    return df

def process_data(df):
    """Calculates features like deviation from peg and rolling volatility."""
    features = {}
    for coin in df.columns:
        coin_df = pd.DataFrame(df[coin])
        coin_df.columns = ['Price']

        # Percentage deviation from $1.00 peg
        coin_df['Peg_Deviation'] = abs(coin_df['Price'] - 1.0)

        # Rolling 7-day volatility (standard deviation of price)
        coin_df['Volatility_7d'] = coin_df['Price'].rolling(window=7).std().fillna(0)

        features[coin] = coin_df

    return features

def train_anomaly_detector(features_dict):
    """Trains an Isolation Forest model to detect anomalies (de-peg risks)."""
    results = {}
    for coin, df in features_dict.items():
        # Using Peg_Deviation and Volatility_7d as features
        X = df[['Peg_Deviation', 'Volatility_7d']]

        # Train Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)

        # Fit and predict. -1 means anomaly, 1 means normal.
        preds = model.fit_predict(X)

        # Convert predictions to boolean Risk Flags (True if anomalous)
        df['Risk_Flag'] = preds == -1

        results[coin] = df

    return results

def get_latest_risk_summary(results_dict):
    """Generates a summary of the latest day's risk status."""
    summary = []
    for coin, df in results_dict.items():
        latest = df.iloc[-1]
        summary.append({
            'Coin': coin.replace('-USD', ''),
            'Current Price': round(latest['Price'], 4),
            'Peg Deviation': round(latest['Peg_Deviation'], 4),
            '7d Volatility': round(latest['Volatility_7d'], 6),
            'Risk Status': 'High Risk' if latest['Risk_Flag'] else 'Normal'
        })
    return pd.DataFrame(summary)

if __name__ == "__main__":
    print("Fetching data...")
    raw_data = fetch_data()
    print("Processing features...")
    features = process_data(raw_data)
    print("Training models and predicting risks...")
    predictions = train_anomaly_detector(features)
    print("Latest Risk Summary:")
    summary_df = get_latest_risk_summary(predictions)
    print(summary_df)
