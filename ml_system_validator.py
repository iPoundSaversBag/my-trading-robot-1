#!/usr/bin/env python3
"""
ML System Validation - Comprehensive Testing

This script validates that the ML system is working correctly by:
1. Testing each individual ML model
2. Checking regime detection accuracy
3. Validating parameter enhancements
4. Identifying any issues with the integration
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from core.enums import MarketRegime
from complete_9_regime_ml_integrator import Complete9RegimeMLIntegrator
import warnings
warnings.filterwarnings('ignore')

class MLSystemValidator:
    """Comprehensive validation of the ML system"""
    
    def __init__(self):
        self.integrator = Complete9RegimeMLIntegrator()
        self.validation_results = {}
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🔍 ML System Validation Suite")
        print("=" * 60)
        
        # Test 1: Model Loading
        print("\n1️⃣ Testing Model Loading...")
        load_success = self.test_model_loading()
        
        # Test 2: Individual Model Predictions
        print("\n2️⃣ Testing Individual Model Predictions...")
        model_success = self.test_individual_models()
        
        # Test 3: Feature Extraction
        print("\n3️⃣ Testing Feature Extraction...")
        feature_success = self.test_feature_extraction()
        
        # Test 4: Regime Detection Logic
        print("\n4️⃣ Testing Regime Detection Logic...")
        regime_success = self.test_regime_detection_logic()
        
        # Test 5: Parameter Enhancement
        print("\n5️⃣ Testing Parameter Enhancement...")
        param_success = self.test_parameter_enhancement()
        
        # Test 6: Edge Cases
        print("\n6️⃣ Testing Edge Cases...")
        edge_success = self.test_edge_cases()
        
        # Summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"{'Model Loading:':<25} {'✅ PASS' if load_success else '❌ FAIL'}")
        print(f"{'Individual Models:':<25} {'✅ PASS' if model_success else '❌ FAIL'}")
        print(f"{'Feature Extraction:':<25} {'✅ PASS' if feature_success else '❌ FAIL'}")
        print(f"{'Regime Detection:':<25} {'✅ PASS' if regime_success else '❌ FAIL'}")
        print(f"{'Parameter Enhancement:':<25} {'✅ PASS' if param_success else '❌ FAIL'}")
        print(f"{'Edge Cases:':<25} {'✅ PASS' if edge_success else '❌ FAIL'}")
        
        overall_success = all([load_success, model_success, feature_success, 
                              regime_success, param_success, edge_success])
        
        if overall_success:
            print(f"\n🎯 OVERALL RESULT: ✅ ML SYSTEM VALIDATED")
            print(f"🚀 Ready for backtesting integration")
        else:
            print(f"\n⚠️ OVERALL RESULT: ❌ ISSUES FOUND")
            print(f"🔧 Requires fixes before deployment")
            self.print_detailed_issues()
        
        return overall_success
    
    def test_model_loading(self):
        """Test if all models load correctly"""
        try:
            success = self.integrator.load_complete_system()
            
            if success:
                print("✅ All 6 ML models loaded successfully")
                
                # Check individual models
                expected_models = [
                    'volatility_predictor',
                    'trend_strength_assessor', 
                    'momentum_analyzer',
                    'accumulation_detector',
                    'range_analyzer',
                    'risk_assessor'
                ]
                
                for model_name in expected_models:
                    if model_name in self.integrator.models:
                        print(f"   ✅ {model_name}")
                    else:
                        print(f"   ❌ {model_name} missing")
                        return False
                
                return True
            else:
                print("❌ Model loading failed")
                return False
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
    
    def test_individual_models(self):
        """Test each ML model individually"""
        try:
            # Create test data
            test_data = self._create_controlled_test_data()
            
            # Test each scenario
            scenarios = {
                'low_vol_data': test_data['low_volatility'],
                'high_vol_data': test_data['high_volatility'],
                'trending_data': test_data['trending'],
                'ranging_data': test_data['ranging']
            }
            
            all_passed = True
            
            for scenario_name, data in scenarios.items():
                print(f"\n   Testing with {scenario_name}:")
                
                predictions = self.integrator.get_ml_predictions(data)
                
                # Validate predictions are reasonable
                for model_name, prediction in predictions.items():
                    if isinstance(prediction, (int, float)) and not np.isnan(prediction):
                        print(f"     {model_name}: {prediction:.3f} ✅")
                    else:
                        print(f"     {model_name}: {prediction} ❌")
                        all_passed = False
                
                # Validate scenario-specific expectations
                validation_result = self._validate_scenario_predictions(scenario_name, predictions)
                if not validation_result:
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Individual model testing error: {e}")
            return False
    
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        try:
            # Test with different data sizes
            test_sizes = [50, 100, 200]
            
            for size in test_sizes:
                test_data = self._create_simple_test_data(size)
                features = self.integrator._extract_comprehensive_features(test_data)
                
                if features is not None:
                    if len(features) == 23:  # Expected feature count
                        print(f"   ✅ Feature extraction for {size} samples: {len(features)} features")
                    else:
                        print(f"   ❌ Wrong feature count for {size} samples: {len(features)} (expected 23)")
                        return False
                else:
                    print(f"   ❌ Feature extraction failed for {size} samples")
                    return False
            
            # Test edge cases
            small_data = self._create_simple_test_data(20)  # Too small
            features_small = self.integrator._extract_comprehensive_features(small_data)
            
            if features_small is None:
                print(f"   ✅ Correctly rejected data with insufficient samples")
            else:
                print(f"   ⚠️ Should have rejected small data but didn't")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature extraction testing error: {e}")
            return False
    
    def test_regime_detection_logic(self):
        """Test regime detection logic with controlled data"""
        try:
            # Create specific test cases for each regime
            regime_test_data = {
                'TRENDING_BULL': self._create_specific_regime_data('trending_bull'),
                'TRENDING_BEAR': self._create_specific_regime_data('trending_bear'),
                'RANGING': self._create_specific_regime_data('ranging'),
                'HIGH_VOLATILITY': self._create_specific_regime_data('high_volatility'),
                'LOW_VOLATILITY': self._create_specific_regime_data('low_volatility'),
                'BREAKOUT_BULLISH': self._create_specific_regime_data('breakout_bullish'),
                'BREAKOUT_BEARISH': self._create_specific_regime_data('breakout_bearish'),
                'ACCUMULATION': self._create_specific_regime_data('accumulation'),
                'DISTRIBUTION': self._create_specific_regime_data('distribution')
            }
            
            correct_detections = 0
            total_tests = len(regime_test_data)
            
            print(f"\n   Testing regime detection accuracy:")
            
            for expected_regime, test_data in regime_test_data.items():
                detected_regime = self.integrator.detect_market_regime(test_data)
                
                # Check if detection is correct or at least reasonable
                is_correct = (detected_regime.value == expected_regime.lower())
                is_reasonable = self._is_detection_reasonable(expected_regime, detected_regime.value)
                
                if is_correct:
                    correct_detections += 1
                    print(f"     {expected_regime}: {detected_regime.value} ✅")
                elif is_reasonable:
                    correct_detections += 0.5  # Partial credit for reasonable detection
                    print(f"     {expected_regime}: {detected_regime.value} 🟡 (reasonable)")
                else:
                    print(f"     {expected_regime}: {detected_regime.value} ❌")
            
            accuracy = correct_detections / total_tests
            print(f"\n   Detection Accuracy: {accuracy:.1%}")
            
            # Consider >70% accuracy as passing
            return accuracy >= 0.7
            
        except Exception as e:
            print(f"❌ Regime detection testing error: {e}")
            return False
    
    def test_parameter_enhancement(self):
        """Test parameter enhancement functionality"""
        try:
            base_params = {
                'position_size': 1000,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'entry_threshold': 0.6
            }
            
            # Test each regime
            all_passed = True
            
            for regime in MarketRegime:
                enhanced_params = self.integrator.get_enhanced_parameters(regime, base_params)
                
                # Check that parameters were enhanced
                if 'ml_enhanced' in enhanced_params and enhanced_params['ml_enhanced']:
                    print(f"   ✅ {regime.value}: Enhanced successfully")
                    
                    # Validate reasonable parameter ranges
                    if enhanced_params['position_size'] <= 0:
                        print(f"     ❌ Invalid position size: {enhanced_params['position_size']}")
                        all_passed = False
                    
                    if enhanced_params['stop_loss'] <= 0 or enhanced_params['stop_loss'] > 1:
                        print(f"     ❌ Invalid stop loss: {enhanced_params['stop_loss']}")
                        all_passed = False
                        
                else:
                    print(f"   ❌ {regime.value}: Enhancement failed")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Parameter enhancement testing error: {e}")
            return False
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        try:
            # Test 1: Empty data
            try:
                empty_data = pd.DataFrame()
                regime = self.integrator.detect_market_regime(empty_data)
                print(f"   ✅ Empty data handled: {regime.value}")
            except Exception as e:
                print(f"   ❌ Empty data failed: {e}")
                return False
            
            # Test 2: Insufficient data
            try:
                small_data = self._create_simple_test_data(10)
                regime = self.integrator.detect_market_regime(small_data)
                print(f"   ✅ Small data handled: {regime.value}")
            except Exception as e:
                print(f"   ❌ Small data failed: {e}")
                return False
            
            # Test 3: Invalid data
            try:
                invalid_data = pd.DataFrame({
                    'close': [np.nan, np.inf, -np.inf, 100],
                    'volume': [1000, 2000, 3000, 4000],
                    'high': [101, 102, 103, 104],
                    'low': [99, 98, 97, 96],
                    'open': [100, 101, 102, 103]
                })
                regime = self.integrator.detect_market_regime(invalid_data)
                print(f"   ✅ Invalid data handled: {regime.value}")
            except Exception as e:
                print(f"   ❌ Invalid data failed: {e}")
                return False
            
            # Test 4: Extreme values
            try:
                extreme_data = self._create_simple_test_data(100)
                extreme_data['close'] *= 1e6  # Extreme prices
                regime = self.integrator.detect_market_regime(extreme_data)
                print(f"   ✅ Extreme values handled: {regime.value}")
            except Exception as e:
                print(f"   ❌ Extreme values failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Edge case testing error: {e}")
            return False
    
    def _create_controlled_test_data(self):
        """Create controlled test data for specific scenarios"""
        length = 100
        
        # Low volatility data
        low_vol_returns = np.random.normal(0, 0.005, length)
        low_vol_prices = 100 * np.exp(np.cumsum(low_vol_returns))
        
        # High volatility data
        high_vol_returns = np.random.normal(0, 0.05, length)
        high_vol_prices = 100 * np.exp(np.cumsum(high_vol_returns))
        
        # Trending data
        trend_returns = np.random.normal(0.002, 0.02, length)
        trend_prices = 100 * np.exp(np.cumsum(trend_returns))
        
        # Ranging data
        ranging_prices = 100 + 5 * np.sin(np.arange(length) * 2 * np.pi / 20)
        
        return {
            'low_volatility': self._create_ohlc_dataframe(low_vol_prices),
            'high_volatility': self._create_ohlc_dataframe(high_vol_prices),
            'trending': self._create_ohlc_dataframe(trend_prices),
            'ranging': self._create_ohlc_dataframe(ranging_prices)
        }
    
    def _create_specific_regime_data(self, regime_type):
        """Create data specifically for testing a regime"""
        length = 100
        
        if regime_type == 'trending_bull':
            returns = np.random.normal(0.003, 0.015, length)
            # Add persistence
            for i in range(1, len(returns)):
                returns[i] += 0.3 * max(0, returns[i-1])
        
        elif regime_type == 'trending_bear':
            returns = np.random.normal(-0.003, 0.015, length)
            # Add persistence
            for i in range(1, len(returns)):
                returns[i] += 0.3 * min(0, returns[i-1])
        
        elif regime_type == 'ranging':
            returns = np.random.normal(0, 0.010, length)
            # Add mean reversion
            cumulative = np.cumsum(returns)
            returns -= 0.05 * cumulative  # Pull back to mean
        
        elif regime_type == 'high_volatility':
            returns = np.random.normal(0, 0.06, length)
            # Add volatility clustering
            for i in range(1, len(returns)):
                if abs(returns[i-1]) > 0.03:
                    returns[i] *= 2
        
        elif regime_type == 'low_volatility':
            returns = np.random.normal(0.001, 0.006, length)
        
        elif regime_type == 'breakout_bullish':
            # Consolidation then breakout
            consolidation = np.random.normal(0, 0.008, length//2)
            breakout = np.random.normal(0.008, 0.025, length//2)
            returns = np.concatenate([consolidation, breakout])
        
        elif regime_type == 'breakout_bearish':
            # Consolidation then breakdown
            consolidation = np.random.normal(0, 0.008, length//2)
            breakdown = np.random.normal(-0.008, 0.025, length//2)
            returns = np.concatenate([consolidation, breakdown])
        
        elif regime_type == 'accumulation':
            returns = np.random.normal(0.0005, 0.012, length)
            # Gradual accumulation
            returns += np.linspace(0, 0.002, length)
        
        elif regime_type == 'distribution':
            returns = np.random.normal(-0.0005, 0.015, length)
            # Gradual distribution
            returns += np.linspace(0, -0.002, length)
        
        else:
            returns = np.random.normal(0, 0.02, length)
        
        prices = 100 * np.exp(np.cumsum(returns))
        return self._create_ohlc_dataframe(prices)
    
    def _create_simple_test_data(self, length):
        """Create simple test data"""
        returns = np.random.normal(0, 0.02, length)
        prices = 100 * np.exp(np.cumsum(returns))
        return self._create_ohlc_dataframe(prices)
    
    def _create_ohlc_dataframe(self, prices):
        """Create OHLC dataframe from prices"""
        length = len(prices)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
            'open': prices * (1 + 0.001 * np.random.randn(length)),
            'high': prices * (1 + np.maximum(0, 0.002 * np.random.randn(length))),
            'low': prices * (1 - np.maximum(0, 0.002 * np.random.randn(length))),
            'close': prices,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(length))
        })
        
        return df
    
    def _validate_scenario_predictions(self, scenario_name, predictions):
        """Validate that predictions make sense for the scenario"""
        if scenario_name == 'low_vol_data':
            return predictions['volatility_predictor'] < 0.7
        elif scenario_name == 'high_vol_data':
            return predictions['volatility_predictor'] > 0.3
        elif scenario_name == 'trending_data':
            return predictions['trend_strength_assessor'] > 0.3
        elif scenario_name == 'ranging_data':
            return predictions['range_analyzer'] > predictions['trend_strength_assessor'] * 0.5
        
        return True
    
    def _is_detection_reasonable(self, expected, detected):
        """Check if detection is reasonable even if not exact"""
        reasonable_mappings = {
            'TRENDING_BULL': ['trending_bull', 'breakout_bullish', 'high_volatility'],
            'TRENDING_BEAR': ['trending_bear', 'breakout_bearish', 'high_volatility'],
            'RANGING': ['ranging', 'low_volatility', 'accumulation'],
            'HIGH_VOLATILITY': ['high_volatility', 'breakout_bullish', 'breakout_bearish'],
            'LOW_VOLATILITY': ['low_volatility', 'ranging', 'accumulation'],
            'BREAKOUT_BULLISH': ['breakout_bullish', 'trending_bull', 'high_volatility'],
            'BREAKOUT_BEARISH': ['breakout_bearish', 'trending_bear', 'high_volatility'],
            'ACCUMULATION': ['accumulation', 'ranging', 'low_volatility'],
            'DISTRIBUTION': ['distribution', 'ranging', 'high_volatility']
        }
        
        return detected in reasonable_mappings.get(expected, [])
    
    def print_detailed_issues(self):
        """Print detailed information about issues found"""
        print(f"\n🔍 DETAILED ISSUE ANALYSIS:")
        print(f"If validation failed, check:")
        print(f"1. Model files exist in ml_models/nine_regime/")
        print(f"2. Feature extraction produces 23 features")
        print(f"3. Regime detection logic is working correctly")
        print(f"4. Parameter enhancement multipliers are reasonable")
        print(f"5. Edge cases are handled gracefully")

def main():
    """Run ML system validation"""
    validator = MLSystemValidator()
    success = validator.run_full_validation()
    
    if success:
        print(f"\n🚀 ML system is ready for production use!")
    else:
        print(f"\n🔧 ML system needs fixes before deployment")

if __name__ == "__main__":
    main()
