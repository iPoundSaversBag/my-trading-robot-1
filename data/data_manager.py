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
EXCHANGE_NAME = "binance"

# Default data file path and settings (added from manage_data.py)
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
YEARS_OF_DATA = 4
DATA_FILE = 'data/crypto_data.parquet'


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
        'timeout': 60000,  # 60 seconds (increased from 30)
        'rateLimit': 600,  # More conservative rate limiting
        'options': {
            'adjustForTimeDifference': True,
            'recvWindow': 10000,
        }
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
            
            # Add retry logic for network issues
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    new_ohlcv_chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_fetch_timestamp, limit=max_fetch_limit)
                    break  # Success, exit retry loop
                except ccxt.NetworkError as e:
                    print(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("Max retries reached for network error")
                        raise
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    raise

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
            
            # Use minimal delay for maximum speed
            rate_limit_delay = max(exchange.rateLimit / 1000, 0.1)  # Minimum 100ms only
            await asyncio.sleep(rate_limit_delay)

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
async def download_multiple_timeframes():
    """Download multiple timeframes for multi-timeframe analysis."""
    print("MULTI-TIMEFRAME DATA DOWNLOAD")
    print("=" * 50)
    
    # Load configuration - handle both project root and data directory execution
    import os
    current_dir = os.getcwd()
    if current_dir.endswith('data'):
        config_path = "../core/optimization_config.json"
    else:
        config_path = "core/optimization_config.json"
    
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    
    bot_settings = config.get('bot_settings', {})
    
    EXCHANGE_NAME = bot_settings.get('exchange_name', 'binance')
    SYMBOL = bot_settings.get('symbol', 'BTC/USDT')
    YEARS_OF_DATA = bot_settings.get('years_of_data_to_download', 4)
    
    # Define timeframes to download
    timeframes = {
        '5m': 'data/crypto_data_5m.parquet',
        '15m': 'data/crypto_data_15m.parquet',
        '1h': 'data/crypto_data_1h.parquet',
        '4h': 'data/crypto_data_4h.parquet'
    }
    
    start_date = datetime.datetime.now() - datetime.timedelta(days=YEARS_OF_DATA * 365.25)
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Exchange: {EXCHANGE_NAME}")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframes: {list(timeframes.keys())}")
    print(f"Start Date: {start_date_str}")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Check for existing files and skip if they exist
    results = {}
    for timeframe, filename in timeframes.items():
        print(f"\n[CHECK] Checking for existing {timeframe} data...")
        
        if os.path.exists(filename):
            # Check file age and size
            file_stat = os.stat(filename)
            file_size = file_stat.st_size
            file_age_hours = (time.time() - file_stat.st_mtime) / 3600
            
            if file_size > 1000000 and file_age_hours < 24:  # File > 1MB and less than 24 hours old
                print(f"[SKIP] {timeframe} data already exists and is recent ({file_size:,} bytes, {file_age_hours:.1f}h old)")
                results[timeframe] = True
                continue
            else:
                print(f"[REFRESH] {timeframe} data exists but is old or small, will refresh")
        
        print(f"[DOWNLOAD] Downloading {timeframe} data...")
        success = await download_ohlcv_to_file(
            symbol=SYMBOL,
            timeframe=timeframe,
            start_date_str=start_date_str,
            filename=filename
        )
        results[timeframe] = success
        
        if success:
            print(f"[OK] {timeframe} download completed")
        else:
            print(f"[ERROR] {timeframe} download failed")
    
    # Update config with multi-timeframe settings
    successful_files = {tf: filename for tf, filename in timeframes.items() if results[tf]}
    
    if len(successful_files) >= 2:
        print(f"\n[CONFIG] Updating configuration...")
        data_settings = config.setdefault('data_settings', {})
        data_settings.update({
            'multi_timeframe_enabled': True,
            'timeframe_files': successful_files,
            'primary_timeframe': '5m',
            'signal_timeframes': [tf for tf in successful_files.keys() if tf != '5m']
        })
        
        # Update main data file to primary timeframe
        primary_file = successful_files.get('5m', 'data/crypto_data_5m.parquet')
        # Remove 'data/' prefix if it exists since we don't want to double it
        if primary_file.startswith('data/'):
            primary_file = primary_file[5:]  # Remove 'data/' prefix
        data_settings['file_path'] = f"data/{primary_file}"
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[SUCCESS] Configuration updated with {len(successful_files)} timeframes")
    
    successful_count = sum(results.values())
    print(f"\n[SUMMARY] Downloaded {successful_count}/{len(timeframes)} timeframes successfully")
    return successful_count == len(timeframes)

if __name__ == '__main__':
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--multi-timeframe':
        # Download multiple timeframes
        try:
            success = asyncio.run(download_multiple_timeframes())
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"\nMulti-timeframe download failed: {e}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        # Original single timeframe download
        # Load configuration - handle both project root and data directory execution
        import os
        current_dir = os.getcwd()
        if current_dir.endswith('data'):
            config_path = "../core/optimization_config.json"
        else:
            config_path = "core/optimization_config.json"
        
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        
        bot_settings = config.get('bot_settings', {})
        data_settings = config.get('data_settings', {})

        EXCHANGE_NAME = bot_settings.get('exchange_name', 'binance')
        SYMBOL = bot_settings.get('symbol', 'BTC/USDT')
        TIMEFRAME = bot_settings.get('timeframe', '5m')
        YEARS_OF_DATA = bot_settings.get('years_of_data_to_download', 4)
        # Use timeframe-specific filename to match multi-timeframe expectations
        default_filename = f'crypto_data_{TIMEFRAME}.parquet' if TIMEFRAME != '5m' else 'crypto_data_5m.parquet'
        FILENAME = data_settings.get('file_path', default_filename)

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


# ==============================================================================
# MERGED FUNCTIONALITY FROM manage_data.py
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


# ==============================================================================
# ENHANCED DATA MANAGEMENT FUNCTIONS (FROM BACKUP CONSOLIDATION)
# ==============================================================================

async def async_download_with_integrity(symbol, timeframe, start_date, max_retries=3):
    """
    Enhanced async data download with comprehensive integrity checks.
    Consolidated from manage_data.py.backup with pipeline integration.
    """
    from utilities.utils import central_logger
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            central_logger.log_message(f"Starting enhanced data download (attempt {retry_count + 1}/{max_retries})", "DataManager")
            
            # Perform download
            success = await download_ohlcv_to_file(symbol, timeframe, start_date, DATA_FILE)
            
            if success:
                # Immediate integrity check
                integrity_result = advanced_data_validation(DATA_FILE)
                
                if integrity_result['passed']:
                    central_logger.log_message("Enhanced data download completed successfully", "DataManager")
                    return True
                else:
                    central_logger.log_error(f"Data integrity check failed: {integrity_result['issues']}", "DataManager")
                    retry_count += 1
                    continue
            else:
                central_logger.log_error("Data download failed", "DataManager")
                retry_count += 1
                continue
                
        except Exception as e:
            central_logger.log_error(f"Error in enhanced download: {e}", "DataManager")
            retry_count += 1
            
        if retry_count < max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            central_logger.log_message(f"Waiting {wait_time} seconds before retry", "DataManager")
            await asyncio.sleep(wait_time)
    
    central_logger.log_error("Enhanced data download failed after all retries", "DataManager")
    return False


def advanced_data_validation(filename):
    """
    Comprehensive data validation and corruption detection.
    Enhanced version of check_data_integrity with detailed reporting.
    """
    validation_result = {
        'passed': True,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        if not os.path.exists(filename):
            validation_result['passed'] = False
            validation_result['issues'].append(f"Data file {filename} does not exist")
            return validation_result
        
        # Load data
        df = pd.read_parquet(filename)
        validation_result['stats']['total_rows'] = len(df)
        validation_result['stats']['date_range'] = {
            'start': df.index.min().isoformat() if not df.empty else None,
            'end': df.index.max().isoformat() if not df.empty else None
        }
        
        # Check 1: Basic structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check 2: Data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                validation_result['passed'] = False
                validation_result['issues'].append(f"Column {col} is not numeric")
        
        # Check 3: NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            validation_result['warnings'].append(f"Found {total_nans} NaN values: {nan_counts.to_dict()}")
            # Only fail if critical columns have NaNs
            critical_nans = nan_counts[['open', 'high', 'low', 'close']].sum()
            if critical_nans > 0:
                validation_result['passed'] = False
                validation_result['issues'].append(f"Critical NaN values in OHLC data: {critical_nans}")
        
        # Check 4: OHLC logic
        ohlc_errors = 0
        if len(df) > 0:
            high_low_errors = (df['high'] < df['low']).sum()
            open_range_errors = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
            close_range_errors = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            
            ohlc_errors = high_low_errors + open_range_errors + close_range_errors
            
            if ohlc_errors > 0:
                validation_result['issues'].append(f"OHLC logic errors: {ohlc_errors} rows")
                validation_result['stats']['ohlc_errors'] = {
                    'high_low_errors': high_low_errors,
                    'open_range_errors': open_range_errors,
                    'close_range_errors': close_range_errors
                }
                
                # Fail if more than 1% of data has OHLC errors
                error_percentage = (ohlc_errors / len(df)) * 100
                if error_percentage > 1.0:
                    validation_result['passed'] = False
        
        # Check 5: Zero/negative values
        negative_volume = (df['volume'] < 0).sum()
        zero_prices = ((df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)).sum()
        
        if negative_volume > 0:
            validation_result['warnings'].append(f"Negative volume values: {negative_volume}")
        
        if zero_prices > 0:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Zero or negative price values: {zero_prices}")
        
        # Check 6: Duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Duplicate timestamps: {duplicates}")
        
        # Check 7: Data gaps
        if len(df) > 1:
            expected_freq = pd.infer_freq(df.index[:100])  # Check first 100 rows
            if expected_freq:
                expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
                missing_periods = len(expected_range) - len(df)
                if missing_periods > 0:
                    gap_percentage = (missing_periods / len(expected_range)) * 100
                    if gap_percentage > 5.0:  # More than 5% gaps
                        validation_result['warnings'].append(f"Significant data gaps: {missing_periods} missing periods ({gap_percentage:.1f}%)")
                    validation_result['stats']['data_gaps'] = {
                        'missing_periods': missing_periods,
                        'gap_percentage': gap_percentage
                    }
        
        # Check 8: Extreme volatility
        if len(df) > 0:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()  # >50% price changes
            if extreme_changes > 0:
                validation_result['warnings'].append(f"Extreme price changes (>50%): {extreme_changes}")
        
        # Check 9: Stuck data (high == low)
        stuck_periods = (df['high'] == df['low']).sum()
        if stuck_periods > 0:
            stuck_percentage = (stuck_periods / len(df)) * 100
            validation_result['warnings'].append(f"Stuck data periods (high==low): {stuck_periods} ({stuck_percentage:.1f}%)")
            
            # Fail if more than 10% of data is stuck
            if stuck_percentage > 10.0:
                validation_result['passed'] = False
                validation_result['issues'].append("Excessive stuck data periods")
        
        validation_result['stats']['stuck_periods'] = stuck_periods
        
        # Summary statistics
        validation_result['stats']['price_stats'] = {
            'min_price': float(df[['open', 'high', 'low', 'close']].min().min()) if len(df) > 0 else None,
            'max_price': float(df[['open', 'high', 'low', 'close']].max().max()) if len(df) > 0 else None,
            'avg_volume': float(df['volume'].mean()) if len(df) > 0 else None
        }
        
        return validation_result
        
    except Exception as e:
        validation_result['passed'] = False
        validation_result['issues'].append(f"Validation error: {str(e)}")
        return validation_result


def data_corruption_detection(filename, corruption_threshold=0.01):
    """
    Advanced corruption detection for market data.
    Returns detailed analysis of potential data corruption issues.
    """
    try:
        df = pd.read_parquet(filename)
        
        corruption_report = {
            'corruption_detected': False,
            'corruption_score': 0.0,
            'issues': [],
            'affected_rows': 0,
            'total_rows': len(df)
        }
        
        if len(df) == 0:
            return corruption_report
        
        corruption_indicators = []
        
        # 1. Price spike detection
        price_changes = df['close'].pct_change().abs()
        extreme_spikes = price_changes > 0.2  # >20% price changes
        spike_count = extreme_spikes.sum()
        
        if spike_count > 0:
            spike_ratio = spike_count / len(df)
            corruption_indicators.append(('price_spikes', spike_ratio, spike_count))
        
        # 2. Volume anomalies
        volume_median = df['volume'].median()
        volume_anomalies = df['volume'] > (volume_median * 100)  # Volume 100x median
        volume_anomaly_count = volume_anomalies.sum()
        
        if volume_anomaly_count > 0:
            volume_ratio = volume_anomaly_count / len(df)
            corruption_indicators.append(('volume_anomalies', volume_ratio, volume_anomaly_count))
        
        # 3. Impossible OHLC relationships
        impossible_ohlc = (df['high'] < df['low']) | (df['open'] > df['high']) | (df['open'] < df['low']) | \
                         (df['close'] > df['high']) | (df['close'] < df['low'])
        impossible_count = impossible_ohlc.sum()
        
        if impossible_count > 0:
            impossible_ratio = impossible_count / len(df)
            corruption_indicators.append(('impossible_ohlc', impossible_ratio, impossible_count))
        
        # 4. Repeated identical rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_ratio = duplicate_rows / len(df)
            corruption_indicators.append(('duplicate_rows', duplicate_ratio, duplicate_rows))
        
        # 5. Extreme intraday ranges
        daily_range = (df['high'] - df['low']) / df['close']
        extreme_ranges = daily_range > 0.5  # >50% intraday range
        extreme_range_count = extreme_ranges.sum()
        
        if extreme_range_count > 0:
            range_ratio = extreme_range_count / len(df)
            corruption_indicators.append(('extreme_ranges', range_ratio, extreme_range_count))
        
        # Calculate overall corruption score
        total_affected = 0
        for indicator_type, ratio, count in corruption_indicators:
            corruption_report['issues'].append({
                'type': indicator_type,
                'affected_rows': count,
                'percentage': ratio * 100,
                'severity': 'high' if ratio > 0.05 else 'medium' if ratio > 0.01 else 'low'
            })
            total_affected += count
            corruption_report['corruption_score'] += ratio
        
        corruption_report['affected_rows'] = total_affected
        corruption_report['corruption_detected'] = corruption_report['corruption_score'] > corruption_threshold
        
        return corruption_report
        
    except Exception as e:
        return {
            'corruption_detected': True,
            'corruption_score': 1.0,
            'issues': [{'type': 'analysis_error', 'message': str(e)}],
            'affected_rows': 0,
            'total_rows': 0
        }


def automated_data_cleanup(filename, backup_original=True):
    """
    Automated data cleaning with corruption removal.
    Creates backup and cleans common data issues.
    """
    from utilities.utils import central_logger
    
    try:
        if not os.path.exists(filename):
            central_logger.log_error(f"Data file {filename} not found for cleanup", "DataManager")
            return False
        
        # Create backup if requested
        if backup_original:
            backup_filename = f"{filename}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(filename, backup_filename)
            central_logger.log_message(f"Created backup: {backup_filename}", "DataManager")
        
        # Load data
        df = pd.read_parquet(filename)
        original_rows = len(df)
        
        cleanup_report = {
            'original_rows': original_rows,
            'cleaned_rows': 0,
            'removed_rows': 0,
            'cleanup_actions': []
        }
        
        # Cleanup actions
        # 1. Remove rows with invalid OHLC relationships
        before_ohlc = len(df)
        df = df[
            (df['high'] >= df['low']) &
            (df['open'] >= df['low']) & (df['open'] <= df['high']) &
            (df['close'] >= df['low']) & (df['close'] <= df['high'])
        ]
        ohlc_removed = before_ohlc - len(df)
        if ohlc_removed > 0:
            cleanup_report['cleanup_actions'].append(f"Removed {ohlc_removed} rows with invalid OHLC relationships")
        
        # 2. Remove rows with zero or negative prices
        before_prices = len(df)
        df = df[
            (df['open'] > 0) & (df['high'] > 0) & 
            (df['low'] > 0) & (df['close'] > 0)
        ]
        price_removed = before_prices - len(df)
        if price_removed > 0:
            cleanup_report['cleanup_actions'].append(f"Removed {price_removed} rows with invalid prices")
        
        # 3. Remove rows with negative volume
        before_volume = len(df)
        df = df[df['volume'] >= 0]
        volume_removed = before_volume - len(df)
        if volume_removed > 0:
            cleanup_report['cleanup_actions'].append(f"Removed {volume_removed} rows with negative volume")
        
        # 4. Remove extreme outliers (price changes > 50%)
        if len(df) > 1:
            before_outliers = len(df)
            price_changes = df['close'].pct_change().abs()
            df = df[price_changes <= 0.5]  # Remove >50% price changes
            outliers_removed = before_outliers - len(df)
            if outliers_removed > 0:
                cleanup_report['cleanup_actions'].append(f"Removed {outliers_removed} extreme price outliers")
        
        # 5. Remove duplicate timestamps
        before_duplicates = len(df)
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = before_duplicates - len(df)
        if duplicates_removed > 0:
            cleanup_report['cleanup_actions'].append(f"Removed {duplicates_removed} duplicate timestamps")
        
        # 6. Sort by timestamp
        df = df.sort_index()
        
        cleanup_report['cleaned_rows'] = len(df)
        cleanup_report['removed_rows'] = original_rows - len(df)
        
        # Save cleaned data
        df.to_parquet(filename)
        
        central_logger.log_message(f"Data cleanup completed: {cleanup_report['removed_rows']} rows removed", "DataManager")
        central_logger.log_message(f"Cleanup actions: {cleanup_report['cleanup_actions']}", "DataManager")
        
        return cleanup_report
        
    except Exception as e:
        central_logger.log_error(f"Error during data cleanup: {e}", "DataManager")
        return False


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