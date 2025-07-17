# ==============================================================================
#
#                            CRYPTO DATA DOWNLOADER
#
# ==============================================================================
#
# FILE: download_crypto_data.py
#
# PURPOSE:
#   This script is responsible for fetching historical cryptocurrency data from
#   the Binance exchange and saving it locally. It is designed to be the first
#   step in the trading robot's data pipeline, ensuring that the backtester
#   and optimizer have access to up-to-date market information.
#
# METHODOLOGY:
#   The script uses the `binance.client` library to connect to the Binance API.
#   It fetches historical K-line (candlestick) data for a specified symbol and
#   time interval. The data is then processed into a pandas DataFrame, with
#   columns for Open, High, Low, Close, and Volume.
#
#   To ensure data integrity and efficient storage, the final DataFrame is saved
#   in the Parquet file format. This format is highly compressed and optimized
#   for the type of columnar data used in financial analysis.
#
# KEY FEATURES:
#   - Exchange Integration: Connects directly to the Binance API, one of the
#     world's largest cryptocurrency exchanges.
#   - Flexible Configuration: Allows the user to easily specify the trading pair
#     (e.g., 'BTCUSDT'), the time interval (e.g., '5m'), and the start date for
#     the data download.
#   - Robust Data Handling: Processes the raw API response into a clean,
#     well-structured pandas DataFrame.
#   - Efficient Storage: Saves the data in the Parquet format, which is ideal
#     for large-scale data analysis and is used by the backtesting engine.
#
# ==============================================================================

# download_crypto_data.py (Public Data Version)
# This script downloads historical data without the need for API keys.

import asyncio
import datetime
import os
import sys
import traceback
import time
import logging

import pandas as pd
import ccxt.pro as ccxt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# SECTION 1: API KEY CONFIGURATION
# ==============================================================================
# --- API KEYS ARE NOT REQUIRED FOR DOWNLOADING PUBLIC DATA ---
# You can leave this section empty.

API_KEY = ""
SECRET_KEY = ""

# -------------------------------------------


# ==============================================================================
# SECTION 2: DOWNLOAD SETTINGS
# ==============================================================================
# -- Exchange and Symbol --
# Change 'cryptocom' to 'binance' or another exchange if needed.


# ==============================================================================
# SECTION 3: DOWNLOAD LOGIC (DO NOT MODIFY)
# ==============================================================================
async def download_ohlcv_to_file(symbol, timeframe, start_date_str, filename):
    """
    Downloads historical OHLCV data from an exchange and saves it to a Parquet file.
    """
    print(f"Attempting to download {timeframe} candles for {symbol} from {EXCHANGE_NAME} to {filename}...")
    print(f"Starting historical data fetch from: {start_date_str}")

    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Existing file '{filename}' deleted to ensure a fresh download.")
        except Exception as e:
            print(f"Warning: Could not delete '{filename}': {e}. Please check file permissions.")

    exchange_class = getattr(ccxt, EXCHANGE_NAME)
    # Initialize without API keys for public data access, and increase the timeout
    exchange = exchange_class({
        'enableRateLimit': True,
        'timeout': 30000,  # 30 seconds
    })
    
    try:
        initial_since_timestamp = exchange.parse8601(start_date_str)
        current_fetch_timestamp = initial_since_timestamp
        current_time = exchange.milliseconds()
        
        all_ohlcv = []
        collected_timestamps = set()
        max_fetch_limit = 1000

        while current_fetch_timestamp < current_time:
            print(f"Fetching {max_fetch_limit} candles from {pd.to_datetime(current_fetch_timestamp, unit='ms')}...")
            
            new_ohlcv_chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_fetch_timestamp, limit=max_fetch_limit)

            if not new_ohlcv_chunk:
                print("No more historical data available for this period. Download complete.")
                break

            unique_new_candles = 0
            for candle in new_ohlcv_chunk:
                timestamp = int(candle[0])
                if timestamp not in collected_timestamps:
                    all_ohlcv.append(candle)
                    collected_timestamps.add(timestamp)
                    unique_new_candles += 1
            
            if unique_new_candles == 0:
                print("Fetched chunk contained no new unique candles. Assuming we've caught up.")
                break

            current_fetch_timestamp = new_ohlcv_chunk[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            print(f"Fetched {unique_new_candles} new unique candles. Total collected: {len(all_ohlcv)}")
            await asyncio.sleep(exchange.rateLimit / 1000)

        print(f"\nTotal fetched {len(all_ohlcv)} unique candles for {symbol}.")

        if not all_ohlcv:
            print("No OHLCV data was collected.")
            return False

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- NEW DATA CLEANING STEP ---
        initial_rows = len(df)
        # Remove rows where volume is zero, as this indicates no trading activity
        # and is the source of the data corruption errors.
        df = df[df['volume'] > 0]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"\nData Cleaning: Removed {initial_rows - cleaned_rows} rows with zero volume.")
        
        # --- NEW DATA CORRUPTION FIX ---
        # Identify and remove candles with significant price wicks, which are likely data errors.
        # A simple heuristic is to check if the low price is less than 50% of the high price.
        initial_rows = len(df)
        df = df[df['low'] > (df['high'] * 0.5)]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"Data Cleaning: Removed {initial_rows - cleaned_rows} rows with significant price wicks (potential data corruption).")
        # --- END OF FIX ---

        df.sort_values('timestamp', inplace=True)
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        # Save as Parquet file
        df.to_parquet(filename)
        print(f"Data successfully saved to {filename} with {len(df)} unique entries.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        if exchange:
            await exchange.close()


# ==============================================================================
# SECTION 4: SCRIPT EXECUTION (DO NOT MODIFY)
# ==============================================================================
if __name__ == '__main__':
    # Calculate start date
    # Load the main configuration
    with open("optimization_config.json", 'r') as f:
        import json5
        config = json5.load(f)
    
    bot_settings = config.get('bot_settings', {})
    data_settings = config.get('data_settings', {})

    EXCHANGE_NAME = bot_settings.get('exchange_name', 'binance')
    SYMBOL = bot_settings.get('symbol', 'BTC/USDT')
    TIMEFRAME = bot_settings.get('timeframe', '5m')
    YEARS_OF_DATA = bot_settings.get('years_of_data_to_download', 4)
    FILENAME = data_settings.get('file_path', 'crypto_data.parquet')

    # Calculate start date
    start_date = datetime.datetime.now() - datetime.timedelta(days=YEARS_OF_DATA * 365.25)
    start_date_str_arg = start_date.strftime('%Y-%m-%d %H:%M:%S')

    try:
        download_success = asyncio.run(download_ohlcv_to_file(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date_str=start_date_str_arg,
            filename=FILENAME
        ))
        if not download_success:
            sys.exit(1)
    except Exception as e:
        print(f"\nAn unhandled error occurred during script execution: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)