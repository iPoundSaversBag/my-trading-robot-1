#!/usr/bin/env python3
"""
Deep Diagnostic Analysis for Regime Detection

This script will analyze exactly WHY the regime detection is failing
and help us build a much more accurate system.
"""

import numpy as np
import pandas as pd
from production_hybrid_regime_system import ProductionHybridRegimeSystem
import matplotlib.pyplot as plt

def deep_diagnostic_analysis():
    """Perform comprehensive diagnostic analysis"""
    print("🔍 DEEP DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print("Finding out exactly why regime detection is failing...")
    
    system = ProductionHybridRegimeSystem()
    system.initialize_system()
    
    # Test scenarios with detailed analysis
    scenarios = [
        ('strong_bull', 'trending_bull'),
        ('strong_bear', 'trending_bear'),
        ('clear_ranging', 'ranging'),
        ('high_vol', 'high_volatility'),
        ('low_vol', 'low_volatility'),
        ('bull_breakout', 'breakout_bullish'),
        ('bear_breakout', 'breakout_bearish'),
        ('accumulation', 'accumulation'),
        ('distribution', 'distribution')
    ]
    
    results = []
    
    for scenario_type, expected in scenarios:
        print(f"\n{'='*50}")
        print(f"📊 ANALYZING: {scenario_type.upper()} (expect: {expected})")
        print(f"{'='*50}")
        
        # Create test data
        data = system._create_enhanced_test_data(scenario_type)
        
        # Analyze the raw data
        print(f"\n1. RAW DATA ANALYSIS:")
        returns = data['close'].pct_change().dropna()
        
        print(f"   Data points: {len(data)}")
        print(f"   Returns std: {returns.std():.6f}")
        print(f"   Returns mean: {returns.mean():.6f}")
        print(f"   Returns min/max: {returns.min():.6f} / {returns.max():.6f}")
        
        # Price trend analysis
        first_price = data['close'].iloc[0]
        last_price = data['close'].iloc[-1]
        total_return = (last_price / first_price - 1)
        print(f"   Total return: {total_return:.3%}")
        
        # Volume analysis
        volume_mean = data['volume'].mean()
        volume_std = data['volume'].std()
        print(f"   Volume mean: {volume_mean:,.0f}")
        print(f"   Volume std: {volume_std:,.0f}")
        
        # 2. Calculate indicators
        print(f"\n2. INDICATOR ANALYSIS:")
        indicators = system._calculate_production_indicators(data)
        
        key_indicators = [
            'volatility_short', 'volatility_medium', 'volatility_ratio',
            'trend_strength', 'trend_direction', 'trend_aligned', 'trend_bullish', 'trend_bearish',
            'range_size_medium', 'range_position', 'in_range_middle', 'tight_consolidation',
            'momentum_short', 'momentum_medium', 'momentum_long', 'momentum_accelerating',
            'breakout_bullish', 'breakout_bearish',
            'volume_increasing', 'volume_spike', 'volume_trend_up'
        ]
        
        for indicator in key_indicators:
            value = indicators.get(indicator, 'MISSING')
            print(f"   {indicator}: {value}")
        
        # 3. Test detection logic step by step
        print(f"\n3. DETECTION LOGIC ANALYSIS:")
        detected = system.detect_market_regime(data)
        print(f"   Final detection: {detected.value}")
        print(f"   Expected: {expected}")
        print(f"   CORRECT: {'✅' if detected.value == expected else '❌'}")
        
        # 4. Analyze WHY it was detected as it was
        print(f"\n4. DETECTION REASONING:")
        vol_short = indicators.get('volatility_short', 0)
        vol_ratio = indicators.get('volatility_ratio', 1)
        trend_strength = indicators.get('trend_strength', 0)
        
        print(f"   Volatility check:")
        print(f"     Vol short: {vol_short:.6f} (>0.030 = HIGH, <0.006 = LOW)")
        if vol_short > 0.030:
            print(f"     → HIGH VOLATILITY triggered")
        elif vol_short < 0.006:
            print(f"     → LOW VOLATILITY triggered")
        else:
            print(f"     → Normal volatility, checking trends...")
        
        print(f"   Breakout check:")
        print(f"     Bullish breakout: {indicators.get('breakout_bullish', False)}")
        print(f"     Bearish breakout: {indicators.get('breakout_bearish', False)}")
        
        print(f"   Trend check:")
        print(f"     Trend strength: {trend_strength:.6f} (>0.006 needed)")
        print(f"     Trend aligned: {indicators.get('trend_aligned', False)}")
        print(f"     Trend bullish: {indicators.get('trend_bullish', False)}")
        print(f"     Trend bearish: {indicators.get('trend_bearish', False)}")
        
        # Store results for summary
        results.append({
            'scenario': scenario_type,
            'expected': expected,
            'detected': detected.value,
            'correct': detected.value == expected,
            'vol_short': vol_short,
            'trend_strength': trend_strength,
            'trend_direction': indicators.get('trend_direction', 0),
            'indicators': indicators
        })
    
    # Summary analysis
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count
    
    print(f"Overall Accuracy: {accuracy:.1%} ({correct_count}/{total_count})")
    
    print(f"\n🔍 FAILURE ANALYSIS:")
    failures = [r for r in results if not r['correct']]
    
    for failure in failures:
        print(f"\n❌ {failure['scenario']} → {failure['detected']} (expected {failure['expected']})")
        print(f"   Vol: {failure['vol_short']:.6f}, Trend: {failure['trend_strength']:.6f}")
        
        # Suggest fixes
        if failure['expected'] == 'trending_bull' and failure['detected'] != 'trending_bull':
            print(f"   💡 FIX: Lower trend detection thresholds or improve bull trend logic")
        elif failure['expected'] == 'high_volatility' and failure['detected'] != 'high_volatility':
            print(f"   💡 FIX: Lower volatility threshold (currently >0.030)")
        elif 'breakout' in failure['expected'] and 'breakout' not in failure['detected']:
            print(f"   💡 FIX: Improve breakout detection logic and thresholds")
        elif failure['expected'] in ['accumulation', 'distribution']:
            print(f"   💡 FIX: Add specific accumulation/distribution detection logic")
    
    return results

def suggest_improvements(results):
    """Suggest specific improvements based on analysis"""
    print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS")
    print(f"=" * 60)
    
    # Analyze common failure patterns
    vol_issues = []
    trend_issues = []
    breakout_issues = []
    
    for r in results:
        if not r['correct']:
            if 'volatility' in r['expected'] and 'volatility' not in r['detected']:
                vol_issues.append(r)
            elif 'trending' in r['expected'] and 'trending' not in r['detected']:
                trend_issues.append(r)
            elif 'breakout' in r['expected'] and 'breakout' not in r['detected']:
                breakout_issues.append(r)
    
    print(f"1. VOLATILITY DETECTION ISSUES ({len(vol_issues)} cases):")
    if vol_issues:
        vol_values = [r['vol_short'] for r in vol_issues]
        print(f"   Current threshold: >0.030 for high, <0.006 for low")
        print(f"   Failed volatilities: {[f'{v:.6f}' for v in vol_values]}")
        if vol_issues[0]['expected'] == 'high_volatility':
            suggested_high = max(vol_values) * 0.8
            print(f"   💡 SUGGESTION: Lower high vol threshold to {suggested_high:.6f}")
    
    print(f"\n2. TREND DETECTION ISSUES ({len(trend_issues)} cases):")
    if trend_issues:
        trend_values = [r['trend_strength'] for r in trend_issues]
        print(f"   Current threshold: >0.006 for trends")
        print(f"   Failed trend strengths: {[f'{v:.6f}' for v in trend_values]}")
        if trend_values:
            suggested_trend = max(trend_values) * 0.7
            print(f"   💡 SUGGESTION: Lower trend threshold to {suggested_trend:.6f}")
    
    print(f"\n3. BREAKOUT DETECTION ISSUES ({len(breakout_issues)} cases):")
    if breakout_issues:
        print(f"   💡 SUGGESTION: Completely redesign breakout detection")
        print(f"   - Use price movement + volume + momentum combined")
        print(f"   - Look for consolidation followed by sharp move")
        print(f"   - Require volume confirmation")
    
    print(f"\n4. GENERAL RECOMMENDATIONS:")
    print(f"   📊 Create calibration data from real market scenarios")
    print(f"   🎯 Use actual market data to set thresholds")
    print(f"   🔄 Implement adaptive thresholds based on market conditions")
    print(f"   ⚡ Add regime confidence scoring")
    print(f"   🛡️ Implement regime change detection and smoothing")

def main():
    """Run comprehensive diagnostic analysis"""
    print("🚀 COMPREHENSIVE REGIME DETECTION DIAGNOSTIC")
    print("=" * 60)
    print("This will analyze exactly why regime detection is failing")
    print("and provide specific recommendations for improvement.")
    print()
    
    results = deep_diagnostic_analysis()
    suggest_improvements(results)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Fix the most critical threshold issues")
    print(f"2. Redesign breakout detection completely")
    print(f"3. Add accumulation/distribution specific logic")
    print(f"4. Test with real market data")
    print(f"5. Aim for >70% accuracy before ML enhancement")

if __name__ == "__main__":
    main()
