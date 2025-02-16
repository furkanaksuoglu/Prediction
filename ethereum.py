import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# Function to fetch historical data from Binance
def fetch_binance_data(symbol, interval, start_date, end_date):
    base_url = "https://api.binance.com/api/v3/klines"
    start_timestamp = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    end_timestamp = int(end_date.timestamp() * 1000)  # Convert to milliseconds

    data = []
    while start_timestamp < end_timestamp:
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_timestamp,
                "limit": 1000,  # Maximum data points per request
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an error for failed requests
            candlesticks = response.json()

            if not candlesticks:
                break  # Exit loop if no more data

            data.extend(candlesticks)
            start_timestamp = candlesticks[-1][0] + 1  # Increment start time to avoid duplicates
            
            # Delay to avoid hitting API rate limits
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(5)  # Wait before retrying
            continue

    return data

# Convert data to a DataFrame
def binance_data_to_dataframe(data):
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Define the parameters
symbol = "ETHUSDT"  # Cryptocurrency pair
interval = "5m"  # 5-minute interval
start_date = datetime(2020, 8, 18)  # Start date
end_date = datetime(2025, 1, 27)  # End date

# Fetch the data
try:
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    raw_data = fetch_binance_data(symbol, interval, start_date, end_date)
    
    if raw_data:
        df = binance_data_to_dataframe(raw_data)

        # Define the output path
        output_path = r"C:\Users\User\OneDrive - metu.edu.tr\Desktop\polkadot"
        os.makedirs(output_path, exist_ok=True)
        csv_file_path = os.path.join(output_path, f"{symbol}_historical_data.csv")

        # Save the data to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"{symbol} historical data saved to {csv_file_path}")
    else:
        print("No data retrieved. Please check the API or date range.")

except Exception as e:
    print(f"An error occurred: {e}")
