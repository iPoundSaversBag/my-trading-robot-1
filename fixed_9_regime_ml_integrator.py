#!/usr/bin/env python3
"""
Fixed 9-Regime ML Integrator

This integrator uses the fixed ML system with improved regime detection logic.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from core.enums import MarketRegime
import warnings
warnings.filterwarnings('ignore')

class Fixed9RegimeMLIntegrator:
    """
    Fixed integration layer with improved regime detection
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("ml_models/nine_regime_fixed")
        self.is_loaded = False
        
        # IMPROVED regime detection strategy with better scoring
        self.regime_detection_strategy = {
            'TRENDING_BULL': self._detect_trending_bull_fixed,
            'TRENDING_BEAR': self._detect_trending_bear_fixed,
            'RANGING': self._detect_ranging_fixed,
            'HIGH_VOLATILITY': self._detect_high_volatility_fixed,
            'LOW_VOLATILITY': self._detect_low_volatility_fixed,
            'BREAKOUT_BULLISH': self._detect_breakout_bullish_fixed,
            'BREAKOUT_BEARISH': self._detect_breakout_bearish_fixed,
            'ACCUMULATION': self._detect_accumulation_fixed,
            'DISTRIBUTION': self._detect_distribution_fixed
        }
        
        # Enhanced parameter adjustments
        self.regime_parameters = {
            MarketRegime.TRENDING_BULL: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5,
                'entry_confidence_threshold': 0.6,
                'momentum_weight': 1.3,
                'trend_following_strength': 1.4
            },
            MarketRegime.TRENDING_BEAR: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.3,
                'entry_confidence_threshold': 0.65,
                'momentum_weight': 1.2,
                'trend_following_strength': 1.3
            },
            MarketRegime.RANGING: {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 0.8,
                'entry_confidence_threshold': 0.75,
                'mean_reversion_strength': 1.5,
                'range_trading_enabled': True
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.8,
                'entry_confidence_threshold': 0.8,
                'volatility_filter_strength': 1.6,
                'risk_management_level': 'high'
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_size_multiplier': 1.3,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.2,
                'entry_confidence_threshold': 0.5,
                'volatility_filter_strength': 0.7,
                'risk_management_level': 'low'
            },
            MarketRegime.BREAKOUT_BULLISH: {
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 2.0,
                'entry_confidence_threshold': 0.7,
                'breakout_confirmation_strength': 1.8,
                'momentum_acceleration': True
            },
            MarketRegime.BREAKOUT_BEARISH: {
                'position_size_multiplier': 1.4,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 1.8,
                'entry_confidence_threshold': 0.75,
                'breakout_confirmation_strength': 1.7,
                'momentum_acceleration': True
            },
            MarketRegime.ACCUMULATION: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.3,
                'take_profit_multiplier': 2.5,
                'entry_confidence_threshold': 0.6,
                'accumulation_pattern_weight': 1.5,
                'long_term_hold_bias': True
            },
            MarketRegime.DISTRIBUTION: {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.1,
                'entry_confidence_threshold': 0.8,
                'distribution_pattern_weight': 1.4,
                'exit_bias_strength': 1.3
            }
        }
    
    def load_fixed_system(self):
        """Load the fixed ML system"""
        print("🔄 Loading Fixed 9-Regime ML System...")
        
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
                    print(f"✅ Loaded {model_name}")
                else:
                    print(f"❌ Missing files for {model_name}")
                    
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        self.is_loaded = (loaded_count == len(model_names))
        
        if self.is_loaded:
            print(f"🎯 Fixed ML System loaded successfully!")
        else:
            print(f"❌ System not fully loaded: {loaded_count}/{len(model_names)} models")
        
        return self.is_loaded
    
    def get_ml_predictions(self, data):
        """Get ML predictions with improved error handling"""
        if not self.is_loaded:
            return self._get_default_predictions()
        
        try:
            features = self._extract_comprehensive_features(data)
            if features is None:
                return self._get_default_predictions()
            
            predictions = {}
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                
                # Ensure predictions are in valid ranges
                if model_name == 'volatility_predictor':
                    prediction = max(0, min(prediction, 1))
                elif model_name == 'trend_strength_assessor':
                    prediction = max(0, min(prediction, 1))
                elif model_name == 'momentum_analyzer':
                    prediction = max(-1, min(prediction, 1))
                elif model_name == 'accumulation_detector':
                    prediction = max(0, min(prediction, 1))
                elif model_name == 'range_analyzer':
                    prediction = max(0, min(prediction, 1))
                elif model_name == 'risk_assessor':
                    prediction = max(0, min(prediction, 1))
                
                predictions[model_name] = prediction
            
            return predictions
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return self._get_default_predictions()
    
    def detect_market_regime_fixed(self, data):
        """
        Fixed regime detection with improved logic
        """
        if not self.is_loaded:
            return MarketRegime.RANGING
        
        predictions = self.get_ml_predictions(data)
        
        # Calculate confidence scores with improved weights
        regime_scores = {}
        for regime_name, detection_func in self.regime_detection_strategy.items():
            score = detection_func(predictions, data)
            regime_scores[regime_name] = score
        
        # Find best regime with debugging
        best_regime_name = max(regime_scores, key=regime_scores.get)
        best_score = regime_scores[best_regime_name]
        
        # Debug output (can be removed in production)
        # print(f"Debug - Regime scores: {[(k, f'{v:.3f}') for k, v in sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)]}")
        # print(f"Debug - Best: {best_regime_name} ({best_score:.3f})")
        
        # Lower confidence threshold for better detection
        if best_score < 0.3:
            return MarketRegime.RANGING
        
        regime_enum = getattr(MarketRegime, best_regime_name.upper())
        return regime_enum
    
    def get_enhanced_parameters(self, market_regime, base_parameters):
        """Get enhanced parameters (same as before)"""
        if market_regime not in self.regime_parameters:
            return base_parameters
        
        regime_adjustments = self.regime_parameters[market_regime]
        enhanced_params = base_parameters.copy()
        
        for param_name, multiplier in regime_adjustments.items():
            if param_name.endswith('_multiplier'):
                base_param = param_name.replace('_multiplier', '')
                if base_param in enhanced_params:
                    enhanced_params[base_param] *= multiplier
            else:
                enhanced_params[param_name] = multiplier
        
        enhanced_params['detected_regime'] = market_regime.value
        enhanced_params['ml_enhanced'] = True
        
        return enhanced_params
    
    # FIXED DETECTION FUNCTIONS with improved logic
    
    def _detect_trending_bull_fixed(self, predictions, data):
        """Fixed bull trend detection"""
        trend_strength = predictions.get('trend_strength_assessor', 0.5)
        momentum = predictions.get('momentum_analyzer', 0.0)
        volatility = predictions.get('volatility_predictor', 0.5)
        range_score = predictions.get('range_analyzer', 0.5)
        
        # Strong trend + positive momentum + low range score
        score = (
            trend_strength * 0.5 +
            max(0, momentum) * 0.3 +
            (1 - range_score) * 0.2
        )
        
        # Require positive momentum for bull trend
        if momentum > 0.1:
            score *= 1.3
        elif momentum < -0.1:
            score *= 0.4  # Penalize negative momentum
        
        return min(score, 1.0)
    
    def _detect_trending_bear_fixed(self, predictions, data):
        """Fixed bear trend detection"""
        trend_strength = predictions.get('trend_strength_assessor', 0.5)
        momentum = predictions.get('momentum_analyzer', 0.0)
        volatility = predictions.get('volatility_predictor', 0.5)
        range_score = predictions.get('range_analyzer', 0.5)
        
        # Strong trend + negative momentum + low range score
        score = (
            trend_strength * 0.5 +
            max(0, -momentum) * 0.3 +
            (1 - range_score) * 0.2
        )
        
        # Require negative momentum for bear trend
        if momentum < -0.1:
            score *= 1.3
        elif momentum > 0.1:
            score *= 0.4  # Penalize positive momentum
        
        return min(score, 1.0)
    
    def _detect_ranging_fixed(self, predictions, data):
        """Fixed ranging detection - MUCH IMPROVED"""
        range_score = predictions.get('range_analyzer', 0.0)
        trend_strength = predictions.get('trend_strength_assessor', 0.5)
        momentum = predictions.get('momentum_analyzer', 0.0)
        volatility = predictions.get('volatility_predictor', 0.5)
        
        # PRIMARY: High range score is the main indicator
        base_score = range_score * 0.6
        
        # SECONDARY: Low trend strength supports ranging
        anti_trend_score = (1 - trend_strength) * 0.25
        
        # TERTIARY: Neutral momentum supports ranging
        neutral_momentum_score = (1 - abs(momentum)) * 0.15
        
        score = base_score + anti_trend_score + neutral_momentum_score
        
        # BOOST for strong range signals
        if range_score > 0.6:
            score *= 1.5
        
        # BOOST for weak trends (ranging characteristic)
        if trend_strength < 0.4:
            score *= 1.2
        
        # BOOST for moderate volatility (good for ranging)
        if 0.3 <= volatility <= 0.7:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _detect_high_volatility_fixed(self, predictions, data):
        """Fixed high volatility detection"""
        volatility = predictions.get('volatility_predictor', 0.5)
        risk = predictions.get('risk_assessor', 0.5)
        
        # Primary focus on volatility
        score = volatility * 0.8 + risk * 0.2
        
        # Strong boost for high volatility
        if volatility > 0.7:
            score *= 1.4
        
        return min(score, 1.0)
    
    def _detect_low_volatility_fixed(self, predictions, data):
        """Fixed low volatility detection"""
        volatility = predictions.get('volatility_predictor', 0.5)
        risk = predictions.get('risk_assessor', 0.5)
        
        # Primary focus on low volatility
        score = (1 - volatility) * 0.8 + (1 - risk) * 0.2
        
        # Strong boost for very low volatility
        if volatility < 0.3:
            score *= 1.4
        
        return min(score, 1.0)
    
    def _detect_breakout_bullish_fixed(self, predictions, data):
        """Fixed bullish breakout detection"""
        momentum = predictions.get('momentum_analyzer', 0.0)
        volatility = predictions.get('volatility_predictor', 0.5)
        trend_strength = predictions.get('trend_strength_assessor', 0.5)
        range_score = predictions.get('range_analyzer', 0.5)
        
        # Requires STRONG positive momentum + high volatility
        score = (
            max(0, momentum) * 0.6 +
            volatility * 0.25 +
            (1 - range_score) * 0.15  # Anti-ranging
        )
        
        # STRICT requirement for bullish momentum
        if momentum > 0.3:
            score *= 2.0
        elif momentum <= 0:
            score *= 0.1  # Nearly eliminate if no positive momentum
        
        return min(score, 1.0)
    
    def _detect_breakout_bearish_fixed(self, predictions, data):
        """Fixed bearish breakout detection"""
        momentum = predictions.get('momentum_analyzer', 0.0)
        volatility = predictions.get('volatility_predictor', 0.5)
        trend_strength = predictions.get('trend_strength_assessor', 0.5)
        range_score = predictions.get('range_analyzer', 0.5)
        
        # Requires STRONG negative momentum + high volatility
        score = (
            max(0, -momentum) * 0.6 +
            volatility * 0.25 +
            (1 - range_score) * 0.15  # Anti-ranging
        )
        
        # STRICT requirement for bearish momentum
        if momentum < -0.3:
            score *= 2.0
        elif momentum >= 0:
            score *= 0.1  # Nearly eliminate if no negative momentum
        
        return min(score, 1.0)
    
    def _detect_accumulation_fixed(self, predictions, data):
        """Fixed accumulation detection"""
        accumulation_score = predictions.get('accumulation_detector', 0.5)
        volatility = predictions.get('volatility_predictor', 0.5)
        momentum = predictions.get('momentum_analyzer', 0.0)
        range_score = predictions.get('range_analyzer', 0.5)
        
        # Focus on accumulation score with supporting factors
        score = (
            accumulation_score * 0.7 +
            range_score * 0.2 +  # Often appears ranging
            max(0, momentum) * 0.1  # Slight positive bias
        )
        
        # Boost for strong accumulation signal
        if accumulation_score > 0.6:
            score *= 1.3
        
        # Prefer lower volatility for accumulation
        if volatility < 0.5:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _detect_distribution_fixed(self, predictions, data):
        """Fixed distribution detection"""
        accumulation_score = predictions.get('accumulation_detector', 0.5)
        volatility = predictions.get('volatility_predictor', 0.5)
        momentum = predictions.get('momentum_analyzer', 0.0)
        risk = predictions.get('risk_assessor', 0.5)
        
        # Focus on low accumulation (distribution) with rising risk
        score = (
            (1 - accumulation_score) * 0.6 +
            risk * 0.3 +
            max(0, -momentum) * 0.1  # Slight negative bias
        )
        
        # Boost for clear distribution pattern
        if accumulation_score < 0.3:
            score *= 1.4
        
        # Higher volatility often accompanies distribution
        if volatility > 0.5:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _extract_comprehensive_features(self, data):
        """Extract features (same as before)"""
        try:
            import ta
            
            if len(data) < 50:
                return None
            
            window_data = data.tail(50)
            features = {}
            
            # Basic price features
            returns = window_data['close'].pct_change()
            features['current_return'] = returns.iloc[-1] if len(returns) > 1 else 0
            features['return_volatility'] = returns.std() if len(returns) > 1 else 0
            features['return_skew'] = returns.skew() if len(returns) > 2 else 0
            features['return_kurtosis'] = returns.kurtosis() if len(returns) > 3 else 0
            
            # Trend features
            sma_5 = window_data['close'].rolling(5).mean()
            sma_10 = window_data['close'].rolling(10).mean()
            sma_20 = window_data['close'].rolling(20).mean()
            
            features['price_vs_sma5'] = (window_data['close'].iloc[-1] - sma_5.iloc[-1]) / sma_5.iloc[-1] if sma_5.iloc[-1] != 0 else 0
            features['price_vs_sma10'] = (window_data['close'].iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1] if sma_10.iloc[-1] != 0 else 0
            features['price_vs_sma20'] = (window_data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if sma_20.iloc[-1] != 0 else 0
            
            features['sma_alignment_5_10'] = 1 if sma_5.iloc[-1] > sma_10.iloc[-1] else -1
            features['sma_alignment_10_20'] = 1 if sma_10.iloc[-1] > sma_20.iloc[-1] else -1
            
            # Volatility features
            atr = ta.volatility.AverageTrueRange(window_data['high'], window_data['low'], window_data['close']).average_true_range()
            features['atr_normalized'] = atr.iloc[-1] / window_data['close'].iloc[-1] if atr.iloc[-1] != 0 else 0
            
            bb = ta.volatility.BollingerBands(window_data['close'])
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = (bb_upper - bb_lower) / window_data['close']
            features['bb_width'] = bb_width.iloc[-1] if len(bb_width) > 0 else 0
            features['bb_position'] = ((window_data['close'].iloc[-1] - bb_lower.iloc[-1]) / 
                                     (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0 else 0.5
            
            # Momentum features
            rsi = ta.momentum.RSIIndicator(window_data['close']).rsi()
            features['rsi'] = rsi.iloc[-1] / 100 if len(rsi) > 0 else 0.5
            
            macd = ta.trend.MACD(window_data['close'])
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            features['macd_position'] = 1 if macd_line.iloc[-1] > macd_signal.iloc[-1] else -1
            features['macd_histogram'] = (macd_line.iloc[-1] - macd_signal.iloc[-1]) / window_data['close'].iloc[-1] if window_data['close'].iloc[-1] != 0 else 0
            
            # Volume features
            features['volume_ratio'] = window_data['volume'].iloc[-1] / window_data['volume'].mean()
            features['volume_trend'] = window_data['volume'].rolling(5).mean().iloc[-1] / window_data['volume'].rolling(20).mean().iloc[-1] if len(window_data) >= 20 else 1.0
            
            obv = ta.volume.OnBalanceVolumeIndicator(window_data['close'], window_data['volume']).on_balance_volume()
            obv_sma = obv.rolling(10).mean()
            features['obv_trend'] = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
            
            # Range and position features
            high_20 = window_data['high'].rolling(20).max()
            low_20 = window_data['low'].rolling(20).min()
            features['range_position'] = ((window_data['close'].iloc[-1] - low_20.iloc[-1]) / 
                                         (high_20.iloc[-1] - low_20.iloc[-1])) if (high_20.iloc[-1] - low_20.iloc[-1]) != 0 else 0.5
            
            features['price_range_20'] = (high_20.iloc[-1] - low_20.iloc[-1]) / window_data['close'].iloc[-1] if window_data['close'].iloc[-1] != 0 else 0
            
            # Breakout features
            breakout_threshold = window_data['close'].rolling(20).std().iloc[-1] * 2
            features['breakout_signal'] = 1 if (window_data['close'].iloc[-1] - high_20.iloc[-1]) > breakout_threshold else (-1 if (low_20.iloc[-1] - window_data['close'].iloc[-1]) > breakout_threshold else 0)
            
            # Market structure features
            higher_highs = (window_data['high'].iloc[-1] > window_data['high'].iloc[-2] and 
                           window_data['high'].iloc[-2] > window_data['high'].iloc[-3]) if len(window_data) >= 3 else False
            lower_lows = (window_data['low'].iloc[-1] < window_data['low'].iloc[-2] and 
                         window_data['low'].iloc[-2] < window_data['low'].iloc[-3]) if len(window_data) >= 3 else False
            
            features['market_structure'] = 1 if higher_highs else (-1 if lower_lows else 0)
            
            # Accumulation/Distribution features
            ad_line = ta.volume.AccDistIndexIndicator(window_data['high'], window_data['low'], window_data['close'], window_data['volume']).acc_dist_index()
            ad_trend = ad_line.rolling(10).mean()
            features['ad_trend'] = 1 if ad_line.iloc[-1] > ad_trend.iloc[-1] else -1
            
            # Convert to array (23 features total)
            feature_names = [
                'current_return', 'return_volatility', 'return_skew', 'return_kurtosis',
                'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
                'sma_alignment_5_10', 'sma_alignment_10_20',
                'atr_normalized', 'bb_width', 'bb_position',
                'rsi', 'macd_position', 'macd_histogram',
                'volume_ratio', 'volume_trend', 'obv_trend',
                'range_position', 'price_range_20', 'breakout_signal',
                'market_structure', 'ad_trend'
            ]
            
            feature_array = np.array([features[name] for name in feature_names])
            
            # Validate features
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                return None
            
            return feature_array
            
        except Exception as e:
            return None
    
    def _get_default_predictions(self):
        """Default predictions when ML fails"""
        return {
            'volatility_predictor': 0.5,
            'trend_strength_assessor': 0.5,
            'momentum_analyzer': 0.0,
            'accumulation_detector': 0.5,
            'range_analyzer': 0.5,
            'risk_assessor': 0.5
        }
    
    def test_fixed_integration(self):
        """Test the fixed integration"""
        print("\n🧪 Testing Fixed 9-Regime ML Integration")
        print("=" * 50)
        
        if not self.load_fixed_system():
            print("❌ Cannot test - ML system not loaded")
            return False
        
        # Test specific scenarios
        test_scenarios = {
            'Strong Bull Trend': self._create_test_data('strong_bull'),
            'Strong Bear Trend': self._create_test_data('strong_bear'),
            'Clear Ranging': self._create_test_data('clear_ranging'),
            'High Volatility': self._create_test_data('high_vol'),
            'Low Volatility': self._create_test_data('low_vol'),
            'Bullish Breakout': self._create_test_data('bull_breakout'),
            'Bearish Breakout': self._create_test_data('bear_breakout'),
            'Accumulation Phase': self._create_test_data('accumulation'),
            'Distribution Phase': self._create_test_data('distribution')
        }
        
        correct_detections = 0
        total_tests = len(test_scenarios)
        
        for scenario_name, (data, expected_regime) in test_scenarios.items():
            detected_regime = self.detect_market_regime_fixed(data)
            
            # Get predictions for debugging
            predictions = self.get_ml_predictions(data)
            
            print(f"\n📊 {scenario_name}:")
            print(f"   Expected: {expected_regime}")
            print(f"   Detected: {detected_regime.value}")
            print(f"   ML - Vol: {predictions['volatility_predictor']:.3f}, Trend: {predictions['trend_strength_assessor']:.3f}")
            print(f"   ML - Mom: {predictions['momentum_analyzer']:.3f}, Range: {predictions['range_analyzer']:.3f}")
            
            is_correct = detected_regime.value.lower() == expected_regime.lower()
            if is_correct:
                correct_detections += 1
                print(f"   ✅ CORRECT")
            else:
                print(f"   ❌ INCORRECT")
        
        accuracy = correct_detections / total_tests
        print(f"\n🎯 Detection Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.7:
            print(f"✅ Fixed ML integration is working well!")
            return True
        else:
            print(f"⚠️ Still needs improvement")
            return False
    
    def _create_test_data(self, scenario_type):
        """Create specific test data for validation"""
        length = 100
        
        if scenario_type == 'strong_bull':
            # Strong upward trend
            returns = np.random.normal(0.003, 0.015, length)
            for i in range(1, len(returns)):
                returns[i] += 0.4 * max(0, returns[i-1])
            expected = 'trending_bull'
        
        elif scenario_type == 'strong_bear':
            # Strong downward trend
            returns = np.random.normal(-0.003, 0.015, length)
            for i in range(1, len(returns)):
                returns[i] += 0.4 * min(0, returns[i-1])
            expected = 'trending_bear'
        
        elif scenario_type == 'clear_ranging':
            # Clear ranging market
            base_price = 100
            oscillation = np.sin(np.arange(length) * 2 * np.pi / 25)
            noise = np.random.normal(0, 0.2, length)
            prices = base_price + 3 * (oscillation + noise * 0.3)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            expected = 'ranging'
        
        elif scenario_type == 'high_vol':
            # High volatility
            returns = np.random.normal(0, 0.06, length)
            expected = 'high_volatility'
        
        elif scenario_type == 'low_vol':
            # Low volatility
            returns = np.random.normal(0.001, 0.006, length)
            expected = 'low_volatility'
        
        elif scenario_type == 'bull_breakout':
            # Consolidation then bullish breakout
            consol = np.random.normal(0, 0.008, length//2)
            breakout = np.random.normal(0.008, 0.025, length//2)
            returns = np.concatenate([consol, breakout])
            expected = 'breakout_bullish'
        
        elif scenario_type == 'bear_breakout':
            # Consolidation then bearish breakdown
            consol = np.random.normal(0, 0.008, length//2)
            breakdown = np.random.normal(-0.008, 0.025, length//2)
            returns = np.concatenate([consol, breakdown])
            expected = 'breakout_bearish'
        
        elif scenario_type == 'accumulation':
            # Sideways with gradual accumulation
            returns = np.random.normal(0.0005, 0.012, length)
            expected = 'accumulation'
        
        elif scenario_type == 'distribution':
            # Sideways with gradual distribution
            returns = np.random.normal(-0.0005, 0.015, length)
            expected = 'distribution'
        
        else:
            returns = np.random.normal(0, 0.02, length)
            expected = 'ranging'
        
        # Create OHLC data
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=length, freq='5min'),
            'open': prices * (1 + 0.001 * np.random.randn(length)),
            'high': prices * (1 + np.maximum(0, 0.002 * np.random.randn(length))),
            'low': prices * (1 - np.maximum(0, 0.002 * np.random.randn(length))),
            'close': prices,
            'volume': 1000000 * (1 + 0.3 * np.random.randn(length))
        })
        
        return df, expected

def main():
    """Test the fixed integration"""
    integrator = Fixed9RegimeMLIntegrator()
    success = integrator.test_fixed_integration()
    
    if success:
        print(f"\n🎯 FIXED ML INTEGRATION VALIDATED!")
        print(f"✅ Ready for backtesting integration")
    else:
        print(f"\n🔧 Needs further improvements")

if __name__ == "__main__":
    main()
