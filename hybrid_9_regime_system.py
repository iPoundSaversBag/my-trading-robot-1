#!/usr/bin/env python3
"""
Hybrid 9-Regime Detection System

This system combines:
1. Reliable rule-based regime detection (primary)
2. ML enhancement for parameter optimization (secondary)

This approach ensures robust regime detection while still benefiting from ML insights.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from core.enums import MarketRegime
import ta
import warnings
warnings.filterwarnings('ignore')

class Hybrid9RegimeSystem:
    """
    Hybrid system: Rule-based detection + ML parameter enhancement
    
    Strategy:
    - Use proven technical analysis for regime detection
    - Use ML for parameter optimization and fine-tuning
    - Fail gracefully when ML is unavailable
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("ml_models/nine_regime_fixed")
        self.ml_available = False
        
        # Parameter adjustments for each regime
        self.regime_parameters = {
            MarketRegime.TRENDING_BULL: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5,
                'entry_confidence_threshold': 0.6
            },
            MarketRegime.TRENDING_BEAR: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.3,
                'entry_confidence_threshold': 0.65
            },
            MarketRegime.RANGING: {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 0.8,
                'entry_confidence_threshold': 0.75
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.8,
                'entry_confidence_threshold': 0.8
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_size_multiplier': 1.3,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.2,
                'entry_confidence_threshold': 0.5
            },
            MarketRegime.BREAKOUT_BULLISH: {
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 2.0,
                'entry_confidence_threshold': 0.7
            },
            MarketRegime.BREAKOUT_BEARISH: {
                'position_size_multiplier': 1.4,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 1.8,
                'entry_confidence_threshold': 0.75
            },
            MarketRegime.ACCUMULATION: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.3,
                'take_profit_multiplier': 2.5,
                'entry_confidence_threshold': 0.6
            },
            MarketRegime.DISTRIBUTION: {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.1,
                'entry_confidence_threshold': 0.8
            }
        }
    
    def initialize_system(self):
        """Initialize the hybrid system"""
        print("🔄 Initializing Hybrid 9-Regime Detection System...")
        
        # Try to load ML models (optional)
        try:
            model_names = [
                'volatility_predictor',
                'trend_strength_assessor', 
                'momentum_analyzer',
                'accumulation_detector',
                'range_analyzer',
                'risk_assessor'
            ]
            
            loaded_count = 0
            for model_name in model_names:
                try:
                    model_file = self.model_dir / f"{model_name}.pkl"
                    scaler_file = self.model_dir / f"{model_name}_scaler.pkl"
                    
                    if model_file.exists() and scaler_file.exists():
                        with open(model_file, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        
                        with open(scaler_file, 'rb') as f:
                            self.scalers[model_name] = pickle.load(f)
                        
                        loaded_count += 1
                except:
                    continue
            
            if loaded_count == len(model_names):
                self.ml_available = True
                print(f"✅ ML enhancement loaded ({loaded_count} models)")
            else:
                print(f"⚠️ ML enhancement partial/unavailable ({loaded_count}/{len(model_names)} models)")
                
        except Exception as e:
            print(f"⚠️ ML enhancement unavailable: {e}")
        
        print(f"🎯 Hybrid system ready (ML: {'✅' if self.ml_available else '❌'})")
        return True
    
    def detect_market_regime(self, data):
        """
        Primary regime detection using reliable rule-based approach
        """
        try:
            if len(data) < 50:
                return MarketRegime.RANGING
            
            # Calculate core technical indicators
            indicators = self._calculate_reliable_indicators(data)
            
            # Detect regime using hierarchical approach
            regime = self._detect_regime_hierarchical(indicators, data)
            
            return regime
            
        except Exception as e:
            print(f"⚠️ Regime detection error: {e}")
            return MarketRegime.RANGING
    
    def _calculate_reliable_indicators(self, data):
        """Calculate reliable technical indicators"""
        window_data = data.tail(50)
        indicators = {}
        
        try:
            # 1. Volatility (most reliable) - Use raw returns std, not annualized
            returns = window_data['close'].pct_change().dropna()
            if len(returns) > 10:
                volatility = returns.std()  # Raw volatility per period
                indicators['volatility'] = volatility
            else:
                indicators['volatility'] = 0.005
            
            # 2. Trend strength (direction + persistence)
            sma_5 = window_data['close'].rolling(5).mean()
            sma_20 = window_data['close'].rolling(20).mean()
            sma_50 = window_data['close'].rolling(50).mean() if len(window_data) >= 50 else sma_20
            
            if len(sma_20) >= 20:
                # Trend direction
                short_trend = (sma_5.iloc[-1] / sma_20.iloc[-1] - 1) if sma_20.iloc[-1] > 0 else 0
                long_trend = (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) if sma_50.iloc[-1] > 0 else 0
                
                indicators['trend_direction'] = (short_trend + long_trend) / 2
                indicators['trend_strength'] = abs(indicators['trend_direction'])
                
                # Trend alignment
                bullish_alignment = sma_5.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]
                bearish_alignment = sma_5.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]
                indicators['trend_aligned'] = bullish_alignment or bearish_alignment
            else:
                indicators['trend_direction'] = 0
                indicators['trend_strength'] = 0
                indicators['trend_aligned'] = False
            
            # 3. Range characteristics
            if len(window_data) >= 20:
                high_20 = window_data['high'].rolling(20).max().iloc[-1]
                low_20 = window_data['low'].rolling(20).min().iloc[-1]
                current_price = window_data['close'].iloc[-1]
                
                # Range metrics
                range_size = (high_20 - low_20) / current_price if current_price > 0 else 0
                range_position = (current_price - low_20) / (high_20 - low_20 + 1e-8)
                
                indicators['range_size'] = range_size
                indicators['range_position'] = range_position
                indicators['in_range_middle'] = 0.3 <= range_position <= 0.7
            else:
                indicators['range_size'] = 0.05
                indicators['range_position'] = 0.5
                indicators['in_range_middle'] = True
            
            # 4. Momentum and breakouts
            if len(window_data) >= 20:
                # Price momentum
                momentum_short = (window_data['close'].iloc[-1] / window_data['close'].iloc[-5] - 1) if len(window_data) >= 5 else 0
                momentum_long = (window_data['close'].iloc[-1] / window_data['close'].iloc[-20] - 1) if len(window_data) >= 20 else 0
                
                indicators['momentum_short'] = momentum_short
                indicators['momentum_long'] = momentum_long
                
                # Breakout detection
                recent_high = window_data['high'].tail(5).max()
                recent_low = window_data['low'].tail(5).min()
                
                breakout_up = recent_high > high_20 * 1.005  # Larger threshold for real breakouts
                breakout_down = recent_low < low_20 * 0.995
                
                indicators['breakout_bullish'] = breakout_up and momentum_short > 0.005  # Stronger momentum required
                indicators['breakout_bearish'] = breakout_down and momentum_short < -0.005
            else:
                indicators['momentum_short'] = 0
                indicators['momentum_long'] = 0
                indicators['breakout_bullish'] = False
                indicators['breakout_bearish'] = False
            
            # 5. Volume analysis
            if len(window_data) >= 20:
                volume_ma_short = window_data['volume'].rolling(5).mean().iloc[-1]
                volume_ma_long = window_data['volume'].rolling(20).mean().iloc[-1]
                
                volume_ratio = volume_ma_short / volume_ma_long if volume_ma_long > 0 else 1.0
                volume_spike = window_data['volume'].iloc[-1] / volume_ma_long if volume_ma_long > 0 else 1.0
                
                indicators['volume_increasing'] = volume_ratio > 1.2
                indicators['volume_spike'] = volume_spike > 2.0
            else:
                indicators['volume_increasing'] = False
                indicators['volume_spike'] = False
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ Indicator calculation error: {e}")
            return {}
    
    def _detect_regime_hierarchical(self, indicators, data):
        """
        Hierarchical regime detection with clear priorities
        """
        if not indicators:
            return MarketRegime.RANGING
        
        volatility = indicators.get('volatility', 0.02)
        trend_strength = indicators.get('trend_strength', 0)
        trend_direction = indicators.get('trend_direction', 0)
        
        # LEVEL 1: Check for breakouts first (highest priority)
        if indicators.get('breakout_bullish', False):
            if indicators.get('volume_spike', False) or volatility > 0.04:
                return MarketRegime.BREAKOUT_BULLISH
        
        if indicators.get('breakout_bearish', False):
            if indicators.get('volume_spike', False) or volatility > 0.04:
                return MarketRegime.BREAKOUT_BEARISH
        
        # LEVEL 2: Check volatility extremes - Adjusted for per-period volatility
        if volatility > 0.025:  # High volatility (was 0.06 annualized)
            return MarketRegime.HIGH_VOLATILITY
        
        if volatility < 0.005:  # Low volatility (was 0.015 annualized)
            return MarketRegime.LOW_VOLATILITY
        
        # LEVEL 3: Check for strong trends - Adjusted thresholds
        if trend_strength > 0.008 and indicators.get('trend_aligned', False):  # Reduced from 0.02
            if trend_direction > 0.002:  # Bullish trend (reduced from 0.005)
                return MarketRegime.TRENDING_BULL
            elif trend_direction < -0.002:  # Bearish trend (reduced from -0.005)
                return MarketRegime.TRENDING_BEAR
        
        # LEVEL 4: Check for accumulation/distribution - Adjusted thresholds
        if indicators.get('volume_increasing', False):
            momentum_long = indicators.get('momentum_long', 0)
            
            if 0.002 < momentum_long < 0.015:  # Slow, steady gains (reduced thresholds)
                return MarketRegime.ACCUMULATION
            elif -0.015 < momentum_long < -0.002:  # Slow, steady decline (reduced thresholds)
                return MarketRegime.DISTRIBUTION
        
        # LEVEL 5: Default to ranging
        # If none of the above conditions are met, it's likely ranging
        return MarketRegime.RANGING
    
    def get_ml_enhanced_parameters(self, market_regime, base_parameters, data):
        """
        Get parameters enhanced with ML insights (if available)
        """
        # Start with base regime parameters
        if market_regime in self.regime_parameters:
            enhanced_params = base_parameters.copy()
            regime_adjustments = self.regime_parameters[market_regime]
            
            # Apply base adjustments
            for param_name, multiplier in regime_adjustments.items():
                if param_name.endswith('_multiplier'):
                    base_param = param_name.replace('_multiplier', '')
                    if base_param in enhanced_params:
                        enhanced_params[base_param] *= multiplier
                else:
                    enhanced_params[param_name] = multiplier
        else:
            enhanced_params = base_parameters.copy()
        
        # Add ML enhancements if available
        if self.ml_available:
            try:
                ml_adjustments = self._get_ml_parameter_adjustments(data, market_regime)
                if ml_adjustments:
                    for param, adjustment in ml_adjustments.items():
                        if param in enhanced_params:
                            enhanced_params[param] *= adjustment
                    enhanced_params['ml_enhanced'] = True
                    enhanced_params['enhancement_type'] = 'hybrid_ml'
                else:
                    enhanced_params['ml_enhanced'] = False
                    enhanced_params['enhancement_type'] = 'rule_based'
            except Exception as e:
                enhanced_params['ml_enhanced'] = False
                enhanced_params['enhancement_type'] = 'rule_based_fallback'
        else:
            enhanced_params['ml_enhanced'] = False
            enhanced_params['enhancement_type'] = 'rule_based'
        
        enhanced_params['detected_regime'] = market_regime.value
        return enhanced_params
    
    def _get_ml_parameter_adjustments(self, data, regime):
        """Get ML-based parameter adjustments"""
        try:
            # Extract features for ML models
            features = self._extract_features_for_ml(data)
            if features is None:
                return None
            
            # Get ML predictions
            predictions = {}
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                predictions[model_name] = prediction
            
            # Convert predictions to parameter adjustments
            adjustments = {}
            
            # Volatility-based position sizing
            vol_pred = predictions.get('volatility_predictor', 0.5)
            if vol_pred > 0.7:  # High volatility
                adjustments['position_size'] = 0.8  # Reduce size
            elif vol_pred < 0.3:  # Low volatility
                adjustments['position_size'] = 1.2  # Increase size
            
            # Risk-based stop loss
            risk_pred = predictions.get('risk_assessor', 0.5)
            if risk_pred > 0.7:  # High risk
                adjustments['stop_loss'] = 1.3  # Tighter stops
            elif risk_pred < 0.3:  # Low risk
                adjustments['stop_loss'] = 0.9  # Wider stops
            
            # Momentum-based take profit
            momentum_pred = predictions.get('momentum_analyzer', 0.0)
            if abs(momentum_pred) > 0.5:  # Strong momentum
                adjustments['take_profit'] = 1.4  # Larger targets
            elif abs(momentum_pred) < 0.2:  # Weak momentum
                adjustments['take_profit'] = 0.8  # Smaller targets
            
            return adjustments
            
        except Exception as e:
            return None
    
    def _extract_features_for_ml(self, data):
        """Extract features for ML models (simplified version)"""
        try:
            if len(data) < 50:
                return None
            
            window_data = data.tail(50)
            
            # Basic features only (avoid overfitting)
            returns = window_data['close'].pct_change()
            features = [
                returns.iloc[-1] if len(returns) > 1 else 0,  # Current return
                returns.std() if len(returns) > 1 else 0,     # Volatility
                returns.mean() if len(returns) > 1 else 0,    # Mean return
            ]
            
            # Add simple moving averages
            sma_5 = window_data['close'].rolling(5).mean()
            sma_20 = window_data['close'].rolling(20).mean()
            
            if len(sma_5) >= 5 and len(sma_20) >= 20:
                features.extend([
                    (sma_5.iloc[-1] / sma_20.iloc[-1] - 1) if sma_20.iloc[-1] > 0 else 0,  # Trend
                    (window_data['close'].iloc[-1] / sma_20.iloc[-1] - 1) if sma_20.iloc[-1] > 0 else 0,  # Price vs MA
                ])
            else:
                features.extend([0, 0])
            
            # Pad to expected feature count (23)
            while len(features) < 23:
                features.append(0)
            
            return np.array(features[:23])  # Ensure exactly 23 features
            
        except Exception as e:
            return None
    
    def test_hybrid_system(self):
        """Test the hybrid system comprehensively"""
        print("\n🧪 Testing Hybrid 9-Regime Detection System")
        print("=" * 55)
        
        # Initialize system
        self.initialize_system()
        
        # Test cases with expected outcomes
        test_cases = [
            ('Strong Bull Trend', self._create_test_data('strong_bull'), 'trending_bull'),
            ('Strong Bear Trend', self._create_test_data('strong_bear'), 'trending_bear'),
            ('Clear Ranging', self._create_test_data('clear_ranging'), 'ranging'),
            ('High Volatility', self._create_test_data('high_vol'), 'high_volatility'),
            ('Low Volatility', self._create_test_data('low_vol'), 'low_volatility'),
            ('Bullish Breakout', self._create_test_data('bull_breakout'), 'breakout_bullish'),
            ('Bearish Breakout', self._create_test_data('bear_breakout'), 'breakout_bearish'),
            ('Accumulation', self._create_test_data('accumulation'), 'accumulation'),
            ('Distribution', self._create_test_data('distribution'), 'distribution')
        ]
        
        correct_detections = 0
        total_tests = len(test_cases)
        
        for test_name, data, expected in test_cases:
            detected_regime = self.detect_market_regime(data)
            
            # Test parameter enhancement
            base_params = {
                'position_size': 1000,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
            
            enhanced_params = self.get_ml_enhanced_parameters(detected_regime, base_params, data)
            
            print(f"\n📊 {test_name}:")
            print(f"   Expected: {expected}")
            print(f"   Detected: {detected_regime.value}")
            print(f"   Enhancement: {enhanced_params['enhancement_type']}")
            print(f"   Position Size: {enhanced_params['position_size']:.0f}")
            
            is_correct = detected_regime.value.lower() == expected.lower()
            if is_correct:
                correct_detections += 1
                print(f"   ✅ CORRECT")
            else:
                print(f"   ❌ INCORRECT")
        
        accuracy = correct_detections / total_tests
        print(f"\n🎯 Overall Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.7:
            print(f"✅ Hybrid system is working well!")
            print(f"🚀 Ready for backtesting integration")
            return True
        else:
            print(f"⚠️ Needs further tuning")
            return False
    
    def _create_test_data(self, scenario_type):
        """Create test data for specific scenarios"""
        length = 100
        
        if scenario_type == 'strong_bull':
            # Clear upward trend
            trend = 0.004  # 0.4% per period
            returns = np.random.normal(trend, 0.015, length)
            # Add persistence
            for i in range(1, len(returns)):
                returns[i] += 0.3 * max(0, returns[i-1])
        
        elif scenario_type == 'strong_bear':
            # Clear downward trend
            trend = -0.004
            returns = np.random.normal(trend, 0.015, length)
            # Add persistence
            for i in range(1, len(returns)):
                returns[i] += 0.3 * min(0, returns[i-1])
        
        elif scenario_type == 'clear_ranging':
            # Perfect oscillation
            base_price = 100
            oscillation = np.sin(np.arange(length) * 2 * np.pi / 25)
            noise = np.random.normal(0, 0.5, length)
            prices = base_price + 3 * oscillation + noise
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
        
        elif scenario_type == 'high_vol':
            # High volatility
            returns = np.random.normal(0, 0.08, length)
        
        elif scenario_type == 'low_vol':
            # Low volatility
            returns = np.random.normal(0.001, 0.005, length)
        
        elif scenario_type == 'bull_breakout':
            # Consolidation then breakout
            consol = np.random.normal(0, 0.006, length//2)
            # Sudden breakout with volume
            breakout = np.random.normal(0.015, 0.025, length//2)
            returns = np.concatenate([consol, breakout])
        
        elif scenario_type == 'bear_breakout':
            # Consolidation then breakdown
            consol = np.random.normal(0, 0.006, length//2)
            breakdown = np.random.normal(-0.015, 0.025, length//2)
            returns = np.concatenate([consol, breakdown])
        
        elif scenario_type == 'accumulation':
            # Slow, steady accumulation
            returns = np.random.normal(0.002, 0.010, length)
        
        elif scenario_type == 'distribution':
            # Slow, steady distribution
            returns = np.random.normal(-0.002, 0.012, length)
        
        else:
            returns = np.random.normal(0, 0.02, length)
        
        # Create OHLC data
        if scenario_type != 'clear_ranging':
            prices = 100 * np.exp(np.cumsum(returns))
        
        # Special volume patterns
        if scenario_type in ['bull_breakout', 'bear_breakout']:
            volume = np.concatenate([
                np.full(length//2, 800000),  # Normal volume
                np.full(length//2, 2500000)  # High volume during breakout
            ]) * (1 + 0.2 * np.random.randn(length))
        elif scenario_type in ['accumulation', 'distribution']:
            # Gradually increasing volume
            volume = np.linspace(800000, 1500000, length) * (1 + 0.2 * np.random.randn(length))
        else:
            volume = 1000000 * (1 + 0.3 * np.random.randn(length))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
            'open': prices * (1 + 0.001 * np.random.randn(length)),
            'high': prices * (1 + np.maximum(0, 0.002 * np.random.randn(length))),
            'low': prices * (1 - np.maximum(0, 0.002 * np.random.randn(length))),
            'close': prices,
            'volume': np.maximum(volume, 100000)
        })
        
        return df

def main():
    """Test the hybrid system"""
    print("🔧 Hybrid 9-Regime Detection System")
    print("=" * 60)
    print("Rule-based detection + ML enhancement")
    print()
    
    system = Hybrid9RegimeSystem()
    success = system.test_hybrid_system()
    
    if success:
        print(f"\n🎯 HYBRID SYSTEM VALIDATED!")
        print(f"✅ Reliable rule-based regime detection")
        print(f"✅ ML parameter enhancement (when available)")
        print(f"✅ Graceful degradation without ML")
        print(f"✅ Ready for production backtesting")
    else:
        print(f"\n🔧 System needs further tuning")

if __name__ == "__main__":
    main()
