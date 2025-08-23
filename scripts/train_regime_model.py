#!/usr/bin/env python3
"""
Standalone regime classifier training script.
Creates a pickle file for the 9-regime market classifier.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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

# Import the MarketRegime enum from your core system
try:
    from core.enums import MarketRegime
except ImportError:
    # Fallback: Define locally if import fails
    class MarketRegime(Enum):
        TRENDING_BULL = "trending_bull"
        TRENDING_BEAR = "trending_bear"
        RANGING = "ranging"
        HIGH_VOLATILITY = "high_volatility"
        LOW_VOLATILITY = "low_volatility"
        BREAKOUT_BULLISH = "breakout_bullish"
        BREAKOUT_BEARISH = "breakout_bearish"
        ACCUMULATION = "accumulation"
        DISTRIBUTION = "distribution"

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent

def label_regime(df: pd.DataFrame, window=20, trend_window=50):
    """
    Labels the market regime based on sophisticated rules to identify all 9 regimes.
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
    
    # RSI for overbought/oversold conditions
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # Moving averages for trend detection
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
    accumulation = is_ranging_for_obv & obv_trending_up & (rsi < 40)  # Oversold accumulation
    distribution = is_ranging_for_obv & obv_trending_down & (rsi > 60)  # Overbought distribution

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
    Create comprehensive features for regime classification.
    """
    features_df = df[['close', 'high', 'low', 'volume']].copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price-based features
    features_df['price_change'] = close.pct_change()
    features_df['price_change_5d'] = close.pct_change(periods=5)
    features_df['price_change_10d'] = close.pct_change(periods=10)
    
    # Volatility features
    features_df['volatility_5d'] = close.rolling(5).std()
    features_df['volatility_20d'] = close.rolling(20).std()
    features_df['volatility_ratio'] = features_df['volatility_5d'] / features_df['volatility_20d']
    
    # Trend features
    features_df['sma_5'] = close.rolling(5).mean()
    features_df['sma_20'] = close.rolling(20).mean()
    features_df['sma_50'] = close.rolling(50).mean()
    features_df['trend_5_20'] = (features_df['sma_5'] - features_df['sma_20']) / features_df['sma_20']
    features_df['trend_20_50'] = (features_df['sma_20'] - features_df['sma_50']) / features_df['sma_50']
    
    # Technical indicators
    features_df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # ADX and Directional Movement
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    features_df['adx'] = adx_indicator.adx()
    features_df['plus_di'] = adx_indicator.adx_pos()
    features_df['minus_di'] = adx_indicator.adx_neg()
    
    # MACD
    macd = ta.trend.MACD(close)
    features_df['macd'] = macd.macd()
    features_df['macd_signal'] = macd.macd_signal()
    features_df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20)
    features_df['bb_high'] = bb.bollinger_hband()
    features_df['bb_low'] = bb.bollinger_lband()
    features_df['bb_width'] = bb.bollinger_wband()
    features_df['bb_position'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    
    # Volume features
    features_df['volume_sma'] = volume.rolling(20).mean()
    features_df['volume_ratio'] = volume / features_df['volume_sma']
    features_df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    features_df['obv_sma'] = features_df['obv'].rolling(10).mean()
    
    # ATR
    features_df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
    features_df['atr_ratio'] = features_df['atr'] / close
    
    # Price position in recent range
    features_df['high_20d'] = high.rolling(20).max()
    features_df['low_20d'] = low.rolling(20).min()
    features_df['price_position'] = (close - features_df['low_20d']) / (features_df['high_20d'] - features_df['low_20d'])
    
    # Drop original OHLCV columns as they're not features
    features_df = features_df.drop(['close', 'high', 'low', 'volume'], axis=1)
    
    return features_df

def train_model():
    """
    Loads data, engineers features, trains a model, and saves the model.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    ml_models_dir = project_root / "ml_models"
    ml_models_dir.mkdir(exist_ok=True)

    # Look for data files
    all_data_files = glob.glob(str(data_dir / "**/*.csv"), recursive=True)
    all_data_files += glob.glob(str(data_dir / "**/*.parquet"), recursive=True)

    if not all_data_files:
        print("Error: No data files found. Please ensure you have data files in the data/ directory.")
        print("Expected formats: *.csv or *.parquet with OHLCV columns")
        return

    print(f"Found {len(all_data_files)} data files to process.")
    
    all_labeled_data = []
    
    for file_path in all_data_files[:5]:  # Limit to first 5 files for speed
        try:
            print(f"\nProcessing {os.path.basename(file_path)}...")
            
            # Load data
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Skipping {file_path}: missing columns {missing_columns}")
                continue
            
            # Ensure minimum data length
            if len(df) < 100:
                print(f"Skipping {file_path}: insufficient data ({len(df)} rows)")
                continue
            
            # Convert index to datetime if needed
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    # Create a simple datetime index
                    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
            
            # 1. Feature Engineering
            print("Creating features...")
            features_df = create_features_vectorized(df)
            
            # 2. Regime Labeling
            print("Labeling regimes...")
            labeled_df = label_regime(df.copy())
            
            # 3. Combine features and labels
            print("Combining features and labels...")
            combined = features_df.join(labeled_df['regime'])
            combined.dropna(inplace=True)
            
            if len(combined) == 0:
                print(f"No valid data after processing {file_path}")
                continue
            
            all_labeled_data.append(combined)
            print(f"Processed {os.path.basename(file_path)}: {len(combined)} valid samples")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if not all_labeled_data:
        print("Error: No data could be processed. Creating synthetic data for demonstration...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 10000
        synthetic_df = pd.DataFrame({
            'open': 100 + np.random.randn(n_samples).cumsum() * 0.1,
            'high': 0,
            'low': 0,
            'close': 0,
            'volume': np.random.lognormal(mean=10, sigma=0.5, size=n_samples)
        })
        
        # Adjust high, low based on open
        synthetic_df['close'] = synthetic_df['open'] + np.random.randn(n_samples) * 0.5
        synthetic_df['high'] = np.maximum(synthetic_df['open'], synthetic_df['close']) + np.abs(np.random.randn(n_samples)) * 0.2
        synthetic_df['low'] = np.minimum(synthetic_df['open'], synthetic_df['close']) - np.abs(np.random.randn(n_samples)) * 0.2
        
        synthetic_df.index = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
        
        print("Created synthetic data for training...")
        features_df = create_features_vectorized(synthetic_df)
        labeled_df = label_regime(synthetic_df.copy())
        combined = features_df.join(labeled_df['regime'])
        combined.dropna(inplace=True)
        all_labeled_data.append(combined)

    # Combine all data
    master_df = pd.concat(all_labeled_data, ignore_index=True)
    print(f"\nTotal data points after aggregation: {len(master_df)}")

    # Analyze regime distribution
    regime_counts = master_df['regime'].value_counts()
    regime_map = {r.value: r.name for r in MarketRegime}
    print("\nRegime Distribution in Training Data:")
    print(regime_counts.rename(index=regime_map))

    # Check for sufficient data
    if regime_counts.min() < 10:
        print("\nWarning: One or more regimes have very few samples (< 10).")
        print("The model may not perform well. Consider generating more diverse data.")

    if len(regime_counts) < 3:
        print("\nError: The training data does not contain enough regime variety.")
        print("Continuing with available data...")

    # Prepare data for training
    X = master_df.drop('regime', axis=1)
    y = master_df['regime']
    
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(regime_counts) > 1 else None
    )

    print(f"\nTraining model on {len(X_train)} samples...")
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    print("\nModel Evaluation on Test Set:")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model and metadata
    model_path = ml_models_dir / "regime_classifier.pkl"
    metadata_path = ml_models_dir / "regime_model_metadata.json"

    joblib.dump(model, model_path)
    print(f"\n✅ Successfully saved trained model to: {model_path}")

    # Save metadata
    metadata = {
        "features": feature_names,
        "regime_distribution": regime_counts.rename(index=regime_map).to_dict(),
        "model_class": str(model.__class__.__name__),
        "training_date": pd.Timestamp.now().isoformat(),
        "n_estimators": model.n_estimators,
        "accuracy": accuracy_score(y_test, y_pred),
        "total_samples": len(master_df),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Successfully saved model metadata to: {metadata_path}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    print("🚀 REGIME CLASSIFIER TRAINING SCRIPT")
    print("=" * 50)
    train_model()
