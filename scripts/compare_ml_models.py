#!/usr/bin/env python3
"""
ML Model Comparison Script
=========================

Compare different ML approaches for regime classification:
1. Original model (baseline)
2. Enhanced model (new)
3. Performance comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import ta
from core.enums import MarketRegime

def load_test_data():
    """Load test data for comparison"""
    test_file = "data/crypto_data_15m.parquet"
    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found")
        return None, None
    
    df = pd.read_parquet(test_file)
    
    # Take a sample for testing
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=123).sort_index()
    
    print(f"📊 Loaded test data: {len(df)} samples")
    return df

def generate_features_original(df):
    """Generate original 31 features"""
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['price_sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
    features['high_low_ratio'] = df['high'] / df['low']
    
    # Technical indicators
    features['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    features['bb_width'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
    features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Volume features
    features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['price_volume_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
    
    # More indicators
    features['macd'] = ta.trend.MACD(df['close']).macd()
    features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    features['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    
    return features.bfill().ffill()

def generate_test_labels(df):
    """Generate test labels using rule-based approach"""
    regimes = []
    
    # Calculate required indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    
    for i in range(len(df)):
        if i < 50:
            regimes.append(MarketRegime.RANGING.value)
            continue
        
        rsi = df['rsi'].iloc[i]
        adx = df['adx'].iloc[i]
        volume_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i]
        
        price_change_5 = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
        price_change_20 = (df['close'].iloc[i] - df['close'].iloc[i-20]) / df['close'].iloc[i-20]
        
        recent_vol = df['close'].iloc[i-20:i].std()
        historical_vol = df['close'].iloc[i-100:i-20].std() if i >= 100 else recent_vol
        
        if pd.isna(rsi) or pd.isna(adx):
            regime = MarketRegime.RANGING
        elif adx > 25 and abs(price_change_20) > 0.1:
            if price_change_20 > 0:
                regime = MarketRegime.TRENDING_BULL
            else:
                regime = MarketRegime.TRENDING_BEAR
        elif recent_vol > historical_vol * 2:
            regime = MarketRegime.HIGH_VOLATILITY
        elif recent_vol < historical_vol * 0.5:
            regime = MarketRegime.LOW_VOLATILITY
        elif volume_ratio > 2 and abs(price_change_5) > 0.03:
            if price_change_5 > 0:
                regime = MarketRegime.BREAKOUT_BULLISH
            else:
                regime = MarketRegime.BREAKOUT_BEARISH
        elif rsi > 70:
            regime = MarketRegime.DISTRIBUTION
        elif rsi < 30:
            regime = MarketRegime.ACCUMULATION
        else:
            regime = MarketRegime.RANGING
            
        regimes.append(regime.value)
        
    return regimes

def test_original_model():
    """Test original model performance"""
    print("🔍 Testing Original Model...")
    
    # Load original model
    original_model_path = "ml_models/regime_classifier.pkl"
    if not Path(original_model_path).exists():
        print(f"❌ Original model not found at {original_model_path}")
        return None
    
    model = joblib.load(original_model_path)
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return None
    
    # Generate features and labels
    X_test = generate_features_original(test_df)
    y_test = generate_test_labels(test_df)
    
    # Ensure same length
    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test[:min_len]
    
    # Drop NaN rows
    mask = ~X_test.isnull().any(axis=1)
    X_test = X_test[mask]
    y_test = np.array(y_test)[mask]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'model_name': 'Original Model',
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'predictions': y_pred,
        'true_labels': y_test
    }

def test_enhanced_model():
    """Test enhanced model performance"""
    print("🔍 Testing Enhanced Model...")
    
    # Load enhanced model
    enhanced_model_path = "ml_models/enhanced_regime_classifier.pkl"
    scaler_path = "ml_models/feature_scaler.pkl"
    
    if not Path(enhanced_model_path).exists():
        print(f"❌ Enhanced model not found at {enhanced_model_path}")
        return None
    
    model = joblib.load(enhanced_model_path)
    scaler = joblib.load(scaler_path)
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return None
    
    # Generate enhanced features
    from scripts.simplified_ml_improvements import AdvancedRegimeFeatureEngineer
    engineer = AdvancedRegimeFeatureEngineer()
    X_test = engineer.engineer_advanced_features(test_df)
    y_test = generate_test_labels(test_df)
    
    # Ensure same length
    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test[:min_len]
    
    # Drop NaN rows
    mask = ~X_test.isnull().any(axis=1)
    X_test = X_test[mask]
    y_test = np.array(y_test)[mask]
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'model_name': 'Enhanced Model',
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'predictions': y_pred,
        'true_labels': y_test
    }

def test_ensemble_model():
    """Test ensemble model performance"""
    print("🔍 Testing Ensemble Model...")
    
    # Load ensemble model
    ensemble_model_path = "ml_models/ensemble_regime_classifier.pkl"
    scaler_path = "ml_models/feature_scaler.pkl"
    
    if not Path(ensemble_model_path).exists():
        print(f"❌ Ensemble model not found at {ensemble_model_path}")
        return None
    
    model = joblib.load(ensemble_model_path)
    scaler = joblib.load(scaler_path)
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return None
    
    # Generate enhanced features
    from scripts.simplified_ml_improvements import AdvancedRegimeFeatureEngineer
    engineer = AdvancedRegimeFeatureEngineer()
    X_test = engineer.engineer_advanced_features(test_df)
    y_test = generate_test_labels(test_df)
    
    # Ensure same length
    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test[:min_len]
    
    # Drop NaN rows
    mask = ~X_test.isnull().any(axis=1)
    X_test = X_test[mask]
    y_test = np.array(y_test)[mask]
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'model_name': 'Ensemble Model',
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'predictions': y_pred,
        'true_labels': y_test
    }

def analyze_breakout_performance(results):
    """Analyze breakout detection performance specifically"""
    print(f"\n🎯 Breakout Detection Analysis - {results['model_name']}")
    print("=" * 50)
    
    report = results['report']
    
    # Breakout regime performance
    breakout_bull = str(MarketRegime.BREAKOUT_BULLISH.value)
    breakout_bear = str(MarketRegime.BREAKOUT_BEARISH.value)
    
    if breakout_bull in report:
        bull_metrics = report[breakout_bull]
        print(f"📈 Bullish Breakout:")
        print(f"   Precision: {bull_metrics['precision']:.3f}")
        print(f"   Recall: {bull_metrics['recall']:.3f}")
        print(f"   F1-score: {bull_metrics['f1-score']:.3f}")
        print(f"   Support: {bull_metrics['support']}")
    
    if breakout_bear in report:
        bear_metrics = report[breakout_bear]
        print(f"📉 Bearish Breakout:")
        print(f"   Precision: {bear_metrics['precision']:.3f}")
        print(f"   Recall: {bear_metrics['recall']:.3f}")
        print(f"   F1-score: {bear_metrics['f1-score']:.3f}")
        print(f"   Support: {bear_metrics['support']}")

def compare_models(results_list):
    """Compare all model results"""
    print("\n🏆 MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Sort by macro F1 score
    results_list.sort(key=lambda x: x['macro_f1'], reverse=True)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
    print("-" * 60)
    
    for result in results_list:
        print(f"{result['model_name']:<20} {result['accuracy']:<10.3f} {result['macro_f1']:<10.3f} {result['weighted_f1']:<12.3f}")
    
    # Detailed analysis for best model
    best_model = results_list[0]
    print(f"\n🥇 BEST PERFORMING MODEL: {best_model['model_name']}")
    print(f"   Overall Accuracy: {best_model['accuracy']:.3f}")
    print(f"   Macro F1-Score: {best_model['macro_f1']:.3f}")
    
    # Analyze breakout performance for all models
    for result in results_list:
        analyze_breakout_performance(result)

def main():
    """Main comparison function"""
    print("🚀 ML Model Performance Comparison")
    print("=" * 50)
    
    results = []
    
    # Test original model
    original_result = test_original_model()
    if original_result:
        results.append(original_result)
    
    # Test enhanced model
    enhanced_result = test_enhanced_model()
    if enhanced_result:
        results.append(enhanced_result)
    
    # Test ensemble model
    ensemble_result = test_ensemble_model()
    if ensemble_result:
        results.append(ensemble_result)
    
    if results:
        compare_models(results)
        
        # Save comparison results
        comparison_data = {
            'comparison_timestamp': pd.Timestamp.now().isoformat(),
            'models_tested': len(results),
            'results': {r['model_name']: {
                'accuracy': r['accuracy'],
                'macro_f1': r['macro_f1'],
                'weighted_f1': r['weighted_f1']
            } for r in results}
        }
        
        with open('ml_models/model_comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n💾 Comparison results saved to ml_models/model_comparison_results.json")
    else:
        print("❌ No models could be tested")

if __name__ == "__main__":
    main()
