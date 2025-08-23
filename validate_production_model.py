import os
import argparse
import pandas as pd
import numpy as np
from core.production_regime_detector import ProductionRegimeDetector, MarketRegime
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_data(file_path="data/crypto_data_5m.parquet"):
    """Loads the 5-minute crypto data for validation."""
    try:
        df = pd.read_parquet(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df.sort_index() # Ensure data is sorted chronologically
        # Ensure we have the necessary columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Dataframe is missing one of the required columns: {required_cols}")
            return None
        logging.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return None

def generate_regime_labels(df: pd.DataFrame, detector: ProductionRegimeDetector, stride_env: int, max_steps_env: int):
    """Generates regime labels for the entire dataset using a rolling window."""
    if df is None:
        logging.error("DataFrame is None.")
        return None

    # Define a window size large enough for multi-timeframe analysis
    # We need enough 5m bars to create a meaningful number of 4h bars.
    # 4h = 240 minutes = 48 * 5-minute bars.
    # For a feature calculation period of 30 on the 4h timeframe, we need 48 * 30 = 1440 bars.
    # Let's use a larger window for safety.
    window_size = 2000
    
    if len(df) < window_size:
        logging.error(f"DataFrame is too small for rolling window validation. Need {window_size}, got {len(df)}.")
        return None

    logging.info(f"Generating regime labels using a rolling window of size {window_size}. This may take some time...")
    
    regimes = []
    confidences = []
    indices = []

    stride = stride_env
    max_steps = max_steps_env  # 0 = no cap
    warm_up = max(window_size, detector.min_data_points_production)
    if len(df) < warm_up:
        logging.error(f"DataFrame smaller than warm-up requirement ({warm_up}).")
        return None

    step_count = 0
    for i in range(warm_up, len(df), stride):
        if (i - warm_up) % (500 * stride) == 0:
            progress = (i - warm_up) / (len(df) - warm_up) * 100
            logging.info(f"Processing... {(i-warm_up)//stride} steps ({progress:.2f}%)")

        rolling_df = df.iloc[i-window_size:i]
        regime, confidence = detector.detect_regime_ml(rolling_df)
        short_adx = None
        try:
            if hasattr(detector, 'last_features') and detector.last_features:
                f5 = detector.last_features.get('5m', {})
                short_adx = f5.get('adx')
        except Exception:
            pass
        regimes.append(regime)
        confidences.append(confidence)
        indices.append(df.index[i])
        # Store interim diagnostic (optionally could append to arrays; keep lightweight)
        step_count += 1
        if max_steps and step_count >= max_steps:
            logging.info(f"Reached REGIME_VALIDATE_MAX_STEPS={max_steps}, stopping early.")
            break

    labeled_df = pd.DataFrame({
        'predicted_regime': regimes,
        'confidence': confidences,
    }, index=pd.DatetimeIndex(indices, name='timestamp'))

    # Merge back OHLCV columns required for effectiveness analysis
    try:
        ohlc_cols = ['open','high','low','close','volume']
        labeled_df = labeled_df.join(df[ohlc_cols], how='left')
    except Exception as e:
        logging.warning(f"Failed to merge OHLCV columns into labeled_df: {e}")
    
    logging.info("Finished generating regime labels.")
    return labeled_df

def analyze_effectiveness(labeled_df: pd.DataFrame):
    """Analyzes the effectiveness of the regime predictions."""
    if labeled_df is None:
        logging.error("Cannot analyze effectiveness on a None DataFrame.")
        return

    logging.info("\n" + "="*50)
    logging.info("      Production Model Effectiveness Report")
    logging.info("="*50)

    # --- 1. Regime Distribution ---
    regime_counts = labeled_df['predicted_regime'].value_counts(normalize=True) * 100
    logging.info("\n[1] Predicted Regime Distribution:")
    for regime, pct in regime_counts.items():
        logging.info(f"  - {regime:<20}: {pct:.2f}%")

    # --- 2. Sanity Checks (Logical Consistency) ---
    logging.info("\n[2] Logical Consistency Checks:")
    
    # Ensure required price columns exist after merge
    required_price_cols = {'open','high','low','close'}
    if not required_price_cols.issubset(labeled_df.columns):
        logging.error(f"Missing price columns in labeled_df: {required_price_cols - set(labeled_df.columns)}")
        return

    # Check for high volatility during low ATR periods (using std as proxy)
    rolling_close = labeled_df['close'].rolling(50)
    low_atr_threshold = rolling_close.std().quantile(0.25)
    atr = labeled_df['close'].rolling(14).std()
    
    high_vol_in_low_atr = labeled_df[(labeled_df['predicted_regime'] == MarketRegime.HIGH_VOLATILITY) & (atr < low_atr_threshold)]
    
    # Check for trending regimes during flat periods
    adx_period = 14
    adx, _, _ = ProductionRegimeDetector()._calculate_adx(labeled_df['high'], labeled_df['low'], labeled_df['close'], adx_period)
    low_adx_threshold = 20 # Generally accepted threshold for non-trending
    
    trending_in_flat = labeled_df[
        (labeled_df['predicted_regime'].isin([MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR])) &
        (adx < low_adx_threshold)
    ]

    total_points = len(labeled_df)
    if total_points == 0:
        logging.error("No data points to analyze after labeling. Cannot calculate effectiveness.")
        return

    logical_errors = len(high_vol_in_low_atr) + len(trending_in_flat)
    
    effectiveness_score = ((total_points - logical_errors) / total_points) * 100
    
    logging.info(f"  - High Volatility in Low ATR periods: {len(high_vol_in_low_atr)} instances")
    logging.info(f"  - Trending in Flat (ADX < 20) periods: {len(trending_in_flat)} instances")
    logging.info(f"  - Total Logical Errors: {logical_errors} / {total_points} data points")
    
    # --- 3. Final Score ---
    logging.info("\n[3] Final Effectiveness Score:")
    logging.info(f"  - Logical Consistency Score: {effectiveness_score:.2f}%")
    
    if effectiveness_score >= 98.0:
        logging.info("\n✅ SUCCESS: Model effectiveness is above the 98% target.")
    else:
        logging.info("\n❌ FAILURE: Model effectiveness is below the 98% target.")
        
    logging.info("="*50)


def main():
    parser = argparse.ArgumentParser(description="Validate production regime detector")
    parser.add_argument('--stride', type=int, default=int(os.getenv('REGIME_VALIDATE_STRIDE', '5')), help='Step stride between evaluations')
    parser.add_argument('--max-steps', type=int, default=int(os.getenv('REGIME_VALIDATE_MAX_STEPS', '0')), help='Maximum evaluation steps (0 = unlimited)')
    args = parser.parse_args()

    detector = ProductionRegimeDetector()
    df = load_test_data()
    if df is None:
        return
    labeled_df = generate_regime_labels(df, detector, args.stride, args.max_steps)
    if labeled_df is not None:
        analyze_effectiveness(labeled_df)

if __name__ == "__main__":
    main()
