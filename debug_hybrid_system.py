#!/usr/bin/env python3
"""
Debug the hybrid regime detection system
"""

import numpy as np
import pandas as pd
from hybrid_9_regime_system import Hybrid9RegimeSystem

def debug_regime_detection():
    """Debug the regime detection logic"""
    print("🔍 DEBUGGING Hybrid System Detection Logic")
    print("=" * 55)
    
    system = Hybrid9RegimeSystem()
    system.initialize_system()
    
    # Create debug test data
    print("\n1. Testing CLEAR RANGING scenario:")
    ranging_data = system._create_test_data('clear_ranging')
    
    # Analyze the data
    print("   Data analysis:")
    returns = ranging_data['close'].pct_change().tail(20).dropna()
    volatility = returns.std() * np.sqrt(288)
    print(f"   Returns std: {returns.std():.6f}")
    print(f"   Annualized volatility: {volatility:.6f}")
    print(f"   Returns range: {returns.min():.6f} to {returns.max():.6f}")
    
    # Test indicators
    indicators = system._calculate_reliable_indicators(ranging_data)
    print(f"   Calculated volatility: {indicators.get('volatility', 'ERROR'):.6f}")
    print(f"   Trend strength: {indicators.get('trend_strength', 'ERROR'):.6f}")
    print(f"   Trend direction: {indicators.get('trend_direction', 'ERROR'):.6f}")
    print(f"   Range size: {indicators.get('range_size', 'ERROR'):.6f}")
    print(f"   In range middle: {indicators.get('in_range_middle', 'ERROR')}")
    
    # Test detection
    regime = system.detect_market_regime(ranging_data)
    print(f"   Detected regime: {regime.value}")
    
    print("\n2. Testing LOW VOLATILITY scenario:")
    low_vol_data = system._create_test_data('low_vol')
    
    returns = low_vol_data['close'].pct_change().tail(20).dropna()
    volatility = returns.std() * np.sqrt(288)
    print(f"   Returns std: {returns.std():.6f}")
    print(f"   Annualized volatility: {volatility:.6f}")
    
    indicators = system._calculate_reliable_indicators(low_vol_data)
    print(f"   Calculated volatility: {indicators.get('volatility', 'ERROR'):.6f}")
    
    regime = system.detect_market_regime(low_vol_data)
    print(f"   Detected regime: {regime.value}")
    
    print("\n3. Testing volatility thresholds:")
    print(f"   High vol threshold: >0.06")
    print(f"   Low vol threshold: <0.015")
    
    # Test different volatility levels
    test_vols = [0.005, 0.010, 0.020, 0.040, 0.080]
    for vol in test_vols:
        print(f"   Vol {vol:.3f}: {'HIGH' if vol > 0.06 else 'LOW' if vol < 0.015 else 'NORMAL'}")

def debug_data_generation():
    """Debug the test data generation"""
    print("\n4. Debugging test data generation:")
    
    system = Hybrid9RegimeSystem()
    
    scenarios = ['clear_ranging', 'low_vol', 'strong_bull']
    
    for scenario in scenarios:
        print(f"\n   Scenario: {scenario}")
        data = system._create_test_data(scenario)
        
        returns = data['close'].pct_change().dropna()
        vol = returns.std()
        
        print(f"   Raw returns std: {vol:.6f}")
        print(f"   Annualized (×√288): {vol * np.sqrt(288):.6f}")
        print(f"   Mean return: {returns.mean():.6f}")
        print(f"   Min/Max returns: {returns.min():.6f} / {returns.max():.6f}")

if __name__ == "__main__":
    debug_regime_detection()
    debug_data_generation()
