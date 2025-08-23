#!/usr/bin/env python3
"""
Quick ML System Diagnosis

This script diagnoses the specific issues with the ML system
to understand why regime detection is still not working correctly.
"""

import numpy as np
import pandas as pd
from fixed_9_regime_ml_integrator import Fixed9RegimeMLIntegrator

def diagnose_ml_system():
    """Diagnose ML system issues"""
    print("🔍 ML System Quick Diagnosis")
    print("=" * 40)
    
    integrator = Fixed9RegimeMLIntegrator()
    
    if not integrator.load_fixed_system():
        print("❌ Cannot load system")
        return
    
    # Create very obvious test cases
    test_cases = {
        'Obvious Ranging': create_obvious_ranging_data(),
        'Obvious High Vol': create_obvious_high_vol_data(),
        'Obvious Trend': create_obvious_trend_data()
    }
    
    for case_name, data in test_cases.items():
        print(f"\n📊 Testing {case_name}:")
        
        # Get raw ML predictions
        predictions = integrator.get_ml_predictions(data)
        print(f"   Raw ML Predictions:")
        for model, pred in predictions.items():
            print(f"     {model}: {pred:.3f}")
        
        # Test individual detection functions
        print(f"   Individual Detection Scores:")
        for regime_name, detection_func in integrator.regime_detection_strategy.items():
            score = detection_func(predictions, data)
            print(f"     {regime_name}: {score:.3f}")
        
        # Final detection
        detected = integrator.detect_market_regime_fixed(data)
        print(f"   Final Detection: {detected.value}")

def create_obvious_ranging_data():
    """Create obviously ranging data"""
    length = 100
    # Perfect oscillation
    oscillation = np.sin(np.arange(length) * 2 * np.pi / 20)
    prices = 100 + 5 * oscillation  # 5% range around 100
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1000000 * np.ones(length)
    })
    return df

def create_obvious_high_vol_data():
    """Create obviously high volatility data"""
    length = 100
    # Very volatile returns
    returns = np.random.normal(0, 0.08, length)  # 8% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
        'open': prices,
        'high': prices * 1.05,
        'low': prices * 0.95,
        'close': prices,
        'volume': 2000000 * np.ones(length)
    })
    return df

def create_obvious_trend_data():
    """Create obviously trending data"""
    length = 100
    # Strong consistent trend
    returns = np.full(length, 0.005)  # 0.5% per period
    noise = np.random.normal(0, 0.002, length)  # Small noise
    returns += noise
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': 1500000 * np.ones(length)
    })
    return df

if __name__ == "__main__":
    diagnose_ml_system()
