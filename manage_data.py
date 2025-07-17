# ==============================================================================
#
#                               DATA MANAGEMENT
#
# ==============================================================================
#
# FILE: manage_data.py
#
# PURPOSE:
#   This script serves as a centralized utility for all data-related tasks
#   required by the trading robot pipeline. It handles the downloading of
#   historical market data from the exchange and performs critical integrity
#   checks to ensure the data is clean, complete, and reliable for backtesting.
#
# METHODOLOGY:
#   - Data Source: The script uses the `ccxt.pro` library, an asynchronous version
#     of the popular CCXT library, to connect to the Binance exchange and fetch
#     historical OHLCV (Open, High, Low, Close, Volume) data.
#   - Data Format: All downloaded data is saved to the highly efficient and
#     compressed Parquet file format (`crypto_data.parquet`), which is optimized
#     for use with the pandas library.
#   - Data Cleaning: During the download process, the script automatically
#     filters out candles with zero volume and removes data points that appear
#     corrupted (e.g., those with extreme price wicks where `low` is less than
#     50% of `high`), which could otherwise distort backtest results.
#
# KEY FEATURES:
#   - Asynchronous Downloading: Leverages Python's `asyncio` library and
#     `ccxt.pro` to fetch large amounts of historical data without blocking
#     the program.
#   - Comprehensive Integrity Checks: The `check_data_integrity` function
#     verifies the loaded data for NaNs, zero/negative values, incorrect OHLC
#     logic (e.g., high < low), duplicate timestamps, and stuck data (where
#     high == low).
#   - Modularity: The core functionalities are encapsulated in `download_data()`
#     and `check_data()` functions, allowing them to be easily imported and
#     called by the main `watcher.py` script.
#
# ==============================================================================

# manage_data.py
# This script centralizes data management tasks including downloading, cleaning, and integrity checks.

import asyncio
import datetime
import os
import sys
import traceback
import time
import logging
import argparse

import pandas as pd
import numpy as np
import ccxt.pro as ccxt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
EXCHANGE_NAME = 'binance'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
YEARS_OF_DATA = 4
DATA_FILE = 'crypto_data.parquet'

# ==============================================================================
# SECTION 2: DOWNLOAD LOGIC
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
    exchange = exchange_class({'enableRateLimit': True})
    
    try:
        initial_since_timestamp = exchange.parse8601(start_date_str)
        current_fetch_timestamp = initial_since_timestamp
        current_time = exchange.milliseconds()
        
        all_ohlcv = []
        collected_timestamps = set()
        max_fetch_limit = 1000

        while current_fetch_timestamp < current_time:
            retries = 5
            for i in range(retries):
                try:
                    print(f"Fetching {max_fetch_limit} candles from {pd.to_datetime(current_fetch_timestamp, unit='ms')}...")
                    new_ohlcv_chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_fetch_timestamp, limit=max_fetch_limit)
                    if new_ohlcv_chunk:
                        break  # Success
                except ccxt.NetworkError as e:
                    if i < retries - 1:
                        wait = 2 ** i
                        print(f"Network error: {e}. Retrying in {wait} seconds...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Network error after {retries} retries. Aborting.")
                        raise  # Re-raise the exception if all retries fail
            
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
        
        # --- Data Cleaning: Zero Volume ---
        initial_rows = len(df)
        df = df[df['volume'] > 0]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"\nData Cleaning: Removed {initial_rows - cleaned_rows} rows with zero volume.")
        
        # --- Data Cleaning: Price Wicks (Corruption Fix) ---
        initial_rows = len(df)
        df = df[df['low'] > (df['high'] * 0.5)]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"Data Cleaning: Removed {initial_rows - cleaned_rows} rows with significant price wicks (potential data corruption).")

        df.sort_values('timestamp', inplace=True)
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        df.to_parquet(filename)
        print(f"Data successfully saved to {filename} with {len(df)} unique entries.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during download: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        if exchange:
            await exchange.close()

# ==============================================================================
# SECTION 3: DATA INTEGRITY CHECK
# ==============================================================================
def check_data_integrity(file_path):
    """
    Loads and performs a series of checks on the financial data file.
    """
    print(f"--- Starting Data Integrity Check for: {file_path} ---")

    try:
        df = pd.read_parquet(file_path)
        print(f"[SUCCESS] File loaded successfully.")
        print(f"          - Found {len(df)} rows of data.")
        print(f"          - Data spans from {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"[FATAL] Could not read the Parquet file: {e}")
        return

    # --- Check 1: Missing Values (NaNs) ---
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("\n[WARNING] Missing values (NaNs) found:")
        print(nan_counts[nan_counts > 0])
    else:
        print("\n[OK] No missing values (NaNs) found.")

    # --- Check 2: Zero or Negative Prices/Volume ---
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    invalid_values = (df[numeric_cols] <= 0).sum()
    if invalid_values.sum() > 0:
        print("\n[ERROR] Zero or negative values found in critical columns:")
        print(invalid_values[invalid_values > 0])
    else:
        print("\n[OK] No zero or negative values in price/volume columns.")

    # --- Check 3: OHLC Logic (High >= Low, etc.) ---
    ohlc_errors = df[(df['high'] < df['low']) | (df['high'] < df['close']) | (df['high'] < df['open']) | \
                     (df['low'] > df['close']) | (df['low'] > df['open'])].shape[0]
    if ohlc_errors > 0:
        print(f"\n[ERROR] Found {ohlc_errors} rows with incorrect OHLC logic (e.g., high < low).")
    else:
        print("\n[OK] OHLC data logic is consistent.")

    # --- Check 4: Duplicate Timestamps ---
    if df.index.duplicated().any():
        num_duplicates = df.index.duplicated().sum()
        print(f"\n[ERROR] Found {num_duplicates} duplicate timestamps in the index.")
    else:
        print("\n[OK] No duplicate timestamps found.")
        
    # --- Check 5: Stuck Data (Zero Volatility) ---
    stuck_data_rows = df[df['high'] == df['low']].shape[0]
    if stuck_data_rows > 0:
        print(f"\n[WARNING] Found {stuck_data_rows} rows where high == low (zero intra-candle volatility).")
        print("           This can lead to an ATR of zero and cause calculation errors.")
    else:
        print("\n[OK] No stuck data (where high == low) detected.")

    print("\n--- Data Integrity Check Finished ---")

def download_data(args=None):
    """Wrapper function to download data."""
    start_date = datetime.datetime.now() - datetime.timedelta(days=YEARS_OF_DATA * 365.25)
    start_date_str_arg = start_date.strftime('%Y-%m-%d %H:%M:%S')
    try:
        download_success = asyncio.run(download_ohlcv_to_file(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_date_str=start_date_str_arg,
            filename=DATA_FILE
        ))
        if not download_success:
            print("Data download failed.")
            return False
        return True
    except Exception as e:
        print(f"\nAn unhandled error occurred during data download: {e}")
        traceback.print_exc(file=sys.stderr)
        return False

def check_data(args=None):
    """Wrapper function to check data integrity."""
    check_data_integrity(DATA_FILE)
    # For now, we assume the check is informational and doesn't block the pipeline.
    # A more robust implementation might return False on critical errors.
    return True

# ==============================================================================
# SECTION 4: SCRIPT EXECUTION
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Management Script for the Trading Robot")
    parser.add_argument('--download', action='store_true', help="Download fresh data from the exchange.")
    parser.add_argument('--check', action='store_true', help="Run an integrity check on the existing data file.")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)

    if args.download:
        if not download_data():
            sys.exit(1)

    if args.check:
        if not check_data():
            sys.exit(1)
