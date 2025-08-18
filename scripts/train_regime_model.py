import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
import os
import glob
from pathlib import Path
import sys
from enum import Enum
import ta

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Define a local, isolated Enum for labeling to avoid any namespace conflicts.
class MarketRegime(Enum):
    TRENDING_BULL = 1
    TRENDING_BEAR = -1
    RANGING = 0
    HIGH_VOLATILITY = 2
    LOW_VOLATILITY = -2
    BREAKOUT_BULLISH = 3
    BREAKOUT_BEARISH = -3
    ACCUMULATION = 4
    DISTRIBUTION = -4

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

def label_regime(df: pd.DataFrame, window=20, trend_window=50):
    """
    Labels the market regime based on a sophisticated set of hierarchical rules
    to identify all 9 nuanced market states.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # 1. Volatility Indicators
    bb = ta.volatility.BollingerBands(close, window=window, window_dev=2)
    bbw = bb.bollinger_wband()
    atr = ta.volatility.average_true_range(high, low, close, window=window)
    atr_norm = atr / close

    # 2. Trend and Momentum Indicators
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=window)
    adx = adx_indicator.adx()
    plus_di = adx_indicator.adx_pos()
    minus_di = adx_indicator.adx_neg()
    ma_slope = close.rolling(window=trend_window).mean().pct_change(periods=5)
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    obv_slope = obv.rolling(window=window).mean().pct_change(periods=5)

    # 3. Define Dynamic Thresholds
    low_vol_threshold = bbw.quantile(0.25)
    high_vol_threshold = bbw.quantile(0.75)
    low_adx_threshold = 20
    high_adx_threshold = 25

    # 4. Define Regime Conditions (in order of precedence)
    
    # Condition: Breakouts (must come from low vol)
    was_low_vol = (bbw.shift(1) < low_vol_threshold)
    strong_move = (close.pct_change().abs() > atr_norm.shift(1) * 1.5)
    breakout_bull = was_low_vol & strong_move & (close > close.shift())
    breakout_bear = was_low_vol & strong_move & (close < close.shift())

    # Condition: Accumulation/Distribution (sideways market with volume pressure)
    is_ranging_for_obv = (adx < high_adx_threshold)
    obv_trending_up = (obv_slope > obv_slope.quantile(0.75))
    obv_trending_down = (obv_slope < obv_slope.quantile(0.25))
    accumulation = is_ranging_for_obv & obv_trending_up
    distribution = is_ranging_for_obv & obv_trending_down

    # Condition: Core Trends
    is_trending = (adx > high_adx_threshold)
    trending_bull = is_trending & (plus_di > minus_di)
    trending_bear = is_trending & (minus_di > plus_di)

    # Condition: Volatility States (if not a breakout or trend)
    high_volatility = (bbw > high_vol_threshold)
    low_volatility = (bbw < low_vol_threshold)

    # 5. Assign Labels based on Hierarchy
    conditions = [
        breakout_bull,
        breakout_bear,
        accumulation,
        distribution,
        trending_bull,
        trending_bear,
        high_volatility,
        low_volatility
    ]
    choices = [
        MarketRegime.BREAKOUT_BULLISH.value,
        MarketRegime.BREAKOUT_BEARISH.value,
        MarketRegime.ACCUMULATION.value,
        MarketRegime.DISTRIBUTION.value,
        MarketRegime.TRENDING_BULL.value,
        MarketRegime.TRENDING_BEAR.value,
        MarketRegime.HIGH_VOLATILITY.value,
        MarketRegime.LOW_VOLATILITY.value
    ]
    
    df['regime'] = np.select(conditions, choices, default=MarketRegime.RANGING.value)
    return df

def create_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the entire dataframe using efficient, vectorized operations.
    This produces the same features as the live strategy but is orders of magnitude faster.
    """
    print("Generating features using vectorized operations...")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(1, index=df.index)) # Default volume to 1 if not present

    features = pd.DataFrame(index=df.index)

    # Volatility features
    returns = close.pct_change()
    features['volatility_1d'] = returns.rolling(window=20, min_periods=20).std()
    features['volatility_5d'] = returns.rolling(window=100, min_periods=100).std()
    features['volatility_ratio'] = features['volatility_1d'] / features['volatility_5d']

    # Trend features (SMAs)
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()
    features['sma_trend_short'] = (close - sma_20) / sma_20
    features['sma_trend_medium'] = (sma_20 - sma_50) / sma_50
    features['sma_trend_long'] = (sma_50 - sma_200) / sma_200

    # Momentum features
    features['momentum'] = close.pct_change(periods=5)
    features['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    features['stochastic_oscillator'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
    features['macd_signal_diff'] = ta.trend.MACD(close).macd_diff()

    # Other indicators
    features['atr_ratio'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
    features['volume_ratio'] = volume.rolling(window=20).mean() / volume.rolling(window=100).mean()
    features['bb_bandwidth'] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_wband()
    features['adx'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    # Ichimoku Cloud features
    ichi = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    features['ichimoku_a'] = ichi.ichimoku_a()
    features['ichimoku_b'] = ichi.ichimoku_b()
    features['ichimoku_span_a_trend'] = features['ichimoku_a'].pct_change(periods=5)
    features['ichimoku_span_b_trend'] = features['ichimoku_b'].pct_change(periods=5)
    features['price_vs_ichimoku_a'] = (close - features['ichimoku_a']) / features['ichimoku_a']
    features['price_vs_ichimoku_b'] = (close - features['ichimoku_b']) / features['ichimoku_b']

    print("Feature generation complete.")
    # Replace infinite values with NaN and then drop rows with any NaN values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features


def train_model():
    """
    Loads all data, engineers features, trains a model on aggregated data,
    and saves the model and metadata.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    ml_models_dir = project_root / "ml_models"
    ml_models_dir.mkdir(exist_ok=True)

    all_data_files = glob.glob(str(data_dir / "crypto_data_*h.parquet")) + \
                     glob.glob(str(data_dir / "crypto_data_*m.parquet"))

    if not all_data_files:
        print("Error: No data files found in data/. Please generate data first.")
        return

    print(f"Found {len(all_data_files)} data files to process.")
    
    all_labeled_data = []
    
    for file_path in all_data_files:
        try:
            print(f"\nProcessing {os.path.basename(file_path)}...")
            df = pd.read_parquet(file_path)
            
            # 1. Feature Engineering (VECTORIZED)
            features_df = create_features_vectorized(df)
            
            # 2. Labeling (using our rule-based ground truth)
            print("Labeling regimes...")
            labeled_df = label_regime(df.copy()) # Use a copy to avoid side effects
            
            # 3. Combine features and labels
            print("Combining features and labels...")
            combined = features_df.join(labeled_df['regime'])
            combined.dropna(inplace=True)
            
            all_labeled_data.append(combined)
            print(f"Finished processing {os.path.basename(file_path)}. Found {len(combined)} valid data points.")
        except Exception as e:
            print(f"Could not process {file_path}: {e}")

    if not all_labeled_data:
        print("Error: No data could be processed. Aborting training.")
        return

    master_df = pd.concat(all_labeled_data, ignore_index=True)
    print(f"\nTotal data points after aggregation: {len(master_df)}")

    # Analyze regime distribution
    regime_counts = master_df['regime'].value_counts()
    # Create a map from the integer value to the string name for printing
    regime_map = {r.value: r.name for r in MarketRegime}
    print("\nRegime Distribution in Training Data:")
    print(regime_counts.rename(index=regime_map))

    # Check for sufficient data
    if regime_counts.min() < 50:
        print("\nWarning: One or more regimes have very few samples (< 50).")
        print("The model may not perform well. Consider generating more diverse data.")

    if len(regime_counts) < 3:
        print("\nError: The training data does not contain all three regime types.")
        print("Aborting training. Ensure your data covers bullish, bearish, and sideways markets.")
        return

    # Prepare data for training
    X = master_df.drop('regime', axis=1)
    y = master_df['regime']
    
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining model on {len(X_train)} samples...")
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("\nModel Evaluation on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model and metadata
    model_path = ml_models_dir / "regime_model.pkl"
    metadata_path = ml_models_dir / "model_metadata.json"

    joblib.dump(model, model_path)
    print(f"\nSuccessfully saved trained model to: {model_path}")

    metadata = {
        "features": feature_names,
        "regime_distribution": regime_counts.rename(index=regime_map).to_dict(),
        "model_class": str(model.__class__.__name__),
        "training_date": pd.Timestamp.now().isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Successfully saved model metadata to: {metadata_path}")

if __name__ == "__main__":
    train_model()
