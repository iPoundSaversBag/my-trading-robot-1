#!/usr/bin/env python3
"""
Integration adapter for the Practical ML System with existing backtesting engine.

This adapter integrates the practical ML models into the current trading system
while maintaining compatibility with existing regime detection and backtesting.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PracticalMLIntegrator:
    """
    Integrates practical ML models with existing trading system.
    
    Key Integration Points:
    1. Volatility forecasting -> position sizing
    2. Trend strength scoring -> strategy selection
    3. Momentum prediction -> entry/exit timing
    4. Risk assessment -> stop-loss/take-profit levels
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("ml_models/practical")
        self.is_loaded = False
        self.feature_calculator = PracticalFeatureCalculator()
        
    def load_models(self) -> bool:
        """Load all practical ML models"""
        try:
            model_names = ['volatility_forecaster', 'trend_strength_scorer', 
                          'momentum_predictor', 'risk_level_assessor']
            
            for model_name in model_names:
                model_file = self.model_dir / f"{model_name}.pkl"
                scaler_file = self.model_dir / f"{model_name}_scaler.pkl"
                
                if model_file.exists() and scaler_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    
                    with open(scaler_file, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    
                    print(f"✅ Loaded {model_name}")
                else:
                    print(f"❌ Missing files for {model_name}")
                    return False
            
            self.is_loaded = True
            print(f"🚀 Practical ML System loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_trading_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Get comprehensive trading signals from practical ML models.
        
        Returns:
            Dictionary with ML-enhanced trading signals
        """
        if not self.is_loaded:
            if not self.load_models():
                return self._get_default_signals()
        
        try:
            # Extract features from current market data
            features = self.feature_calculator.extract_features(data)
            if features is None:
                return self._get_default_signals()
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                scaler = self.scalers[model_name]
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                predictions[model_name] = prediction
            
            # Convert predictions to trading signals
            trading_signals = self._convert_to_trading_signals(predictions)
            
            return trading_signals
            
        except Exception as e:
            print(f"⚠️ ML prediction error: {e}")
            return self._get_default_signals()
    
    def _convert_to_trading_signals(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Convert ML predictions to actionable trading signals"""
        signals = {}
        
        # 1. Volatility-based position sizing
        volatility_forecast = predictions['volatility_forecaster']
        # Scale volatility to position size multiplier (0.5 to 1.5)
        vol_multiplier = np.clip(1.5 - (volatility_forecast * 2), 0.5, 1.5)
        signals['volatility_position_multiplier'] = vol_multiplier
        
        # 2. Trend strength for strategy selection
        trend_strength = predictions['trend_strength_scorer']
        signals['trend_strength_score'] = trend_strength
        
        # Use trend following when strength > 0.6, mean reversion when < 0.4
        if trend_strength > 0.6:
            signals['strategy_preference'] = 'trend_following'
            signals['trend_confidence'] = trend_strength
        elif trend_strength < 0.4:
            signals['strategy_preference'] = 'mean_reversion'
            signals['trend_confidence'] = 1 - trend_strength
        else:
            signals['strategy_preference'] = 'neutral'
            signals['trend_confidence'] = 0.5
        
        # 3. Momentum for entry/exit timing
        momentum_score = predictions['momentum_predictor']
        signals['momentum_score'] = momentum_score
        
        # Convert momentum to timing signals
        if momentum_score > 0.3:
            signals['entry_timing'] = 'favorable'
            signals['exit_timing'] = 'hold'
        elif momentum_score < -0.3:
            signals['entry_timing'] = 'unfavorable'
            signals['exit_timing'] = 'consider_exit'
        else:
            signals['entry_timing'] = 'neutral'
            signals['exit_timing'] = 'neutral'
        
        # 4. Risk-based stop-loss and take-profit adjustments
        risk_level = predictions['risk_level_assessor']
        signals['risk_level'] = risk_level
        
        # Adjust stop-loss based on risk (tighter stops in high risk)
        base_stop_multiplier = 1.0
        risk_stop_multiplier = np.clip(1.5 - risk_level, 0.5, 1.5)
        signals['stop_loss_multiplier'] = risk_stop_multiplier
        
        # Adjust take-profit based on risk (closer targets in high risk)
        risk_tp_multiplier = np.clip(1.2 - (risk_level * 0.5), 0.8, 1.2)
        signals['take_profit_multiplier'] = risk_tp_multiplier
        
        # 5. Overall confidence score
        # Higher confidence when all signals align
        confidence_factors = [
            1 - abs(0.5 - trend_strength),  # Strong trend or clear mean reversion
            1 - abs(momentum_score),        # Clear momentum direction
            1 - risk_level                  # Lower risk = higher confidence
        ]
        signals['overall_confidence'] = np.mean(confidence_factors)
        
        return signals
    
    def _get_default_signals(self) -> Dict[str, float]:
        """Return default signals when ML is unavailable"""
        return {
            'volatility_position_multiplier': 1.0,
            'trend_strength_score': 0.5,
            'strategy_preference': 'neutral',
            'trend_confidence': 0.5,
            'momentum_score': 0.0,
            'entry_timing': 'neutral',
            'exit_timing': 'neutral',
            'risk_level': 0.5,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'overall_confidence': 0.5
        }
    
    def enhance_regime_parameters(self, base_params: Dict, regime: str, data: pd.DataFrame) -> Dict:
        """
        Enhance regime-specific parameters with ML insights.
        
        This integrates with existing regime detection while adding ML enhancements.
        """
        enhanced_params = base_params.copy()
        
        # Get ML signals
        ml_signals = self.get_trading_signals(data)
        
        # Enhance position sizing
        base_position_size = enhanced_params.get('position_size_multiplier', 1.0)
        vol_multiplier = ml_signals['volatility_position_multiplier']
        enhanced_params['position_size_multiplier'] = base_position_size * vol_multiplier
        
        # Enhance stop-loss levels
        base_stop_loss = enhanced_params.get('stop_loss_multiplier', 1.0)
        risk_multiplier = ml_signals['stop_loss_multiplier']
        enhanced_params['stop_loss_multiplier'] = base_stop_loss * risk_multiplier
        
        # Enhance take-profit levels
        base_take_profit = enhanced_params.get('take_profit_multiplier', 1.0)
        tp_multiplier = ml_signals['take_profit_multiplier']
        enhanced_params['take_profit_multiplier'] = base_take_profit * tp_multiplier
        
        # Add ML-specific parameters
        enhanced_params['ml_trend_strength'] = ml_signals['trend_strength_score']
        enhanced_params['ml_momentum_score'] = ml_signals['momentum_score']
        enhanced_params['ml_risk_level'] = ml_signals['risk_level']
        enhanced_params['ml_confidence'] = ml_signals['overall_confidence']
        enhanced_params['ml_strategy_preference'] = ml_signals['strategy_preference']
        
        # Regime-specific ML enhancements
        if regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            # In trending markets, use trend strength to adjust aggressiveness
            trend_strength = ml_signals['trend_strength_score']
            if trend_strength > 0.7:  # Strong trend
                enhanced_params['position_size_multiplier'] *= 1.2  # More aggressive
                enhanced_params['take_profit_multiplier'] *= 1.3    # Higher targets
            elif trend_strength < 0.3:  # Weak trend
                enhanced_params['position_size_multiplier'] *= 0.8  # More conservative
                enhanced_params['stop_loss_multiplier'] *= 0.8     # Tighter stops
        
        elif regime in ['HIGH_VOLATILITY']:
            # In high volatility, use risk assessment for protection
            risk_level = ml_signals['risk_level']
            enhanced_params['position_size_multiplier'] *= (1.5 - risk_level)  # Reduce size in high risk
            enhanced_params['stop_loss_multiplier'] *= (0.5 + risk_level)      # Adjust stops
        
        elif regime in ['RANGING']:
            # In ranging markets, use momentum for mean reversion timing
            momentum = ml_signals['momentum_score']
            if abs(momentum) > 0.5:  # Strong momentum in ranging market
                enhanced_params['position_size_multiplier'] *= 1.1  # Slightly more aggressive
        
        return enhanced_params
    
    def get_ml_diagnostic_info(self, data: pd.DataFrame) -> Dict:
        """Get ML diagnostic information for monitoring and debugging"""
        if not self.is_loaded:
            return {'status': 'not_loaded'}
        
        try:
            ml_signals = self.get_trading_signals(data)
            
            return {
                'status': 'active',
                'signals': ml_signals,
                'interpretation': {
                    'volatility_outlook': 'high' if ml_signals['volatility_position_multiplier'] < 0.8 else 'low',
                    'trend_regime': ml_signals['strategy_preference'],
                    'momentum_direction': 'bullish' if ml_signals['momentum_score'] > 0.1 else ('bearish' if ml_signals['momentum_score'] < -0.1 else 'neutral'),
                    'risk_assessment': 'high' if ml_signals['risk_level'] > 0.7 else ('low' if ml_signals['risk_level'] < 0.3 else 'moderate'),
                    'overall_sentiment': 'confident' if ml_signals['overall_confidence'] > 0.7 else 'uncertain'
                }
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

class PracticalFeatureCalculator:
    """Feature calculator that matches the training data format"""
    
    def extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features that match the practical ML training format"""
        try:
            if len(data) < 50:
                return None
            
            # Use last 50 periods for feature calculation
            window_data = data.tail(50).copy()
            features = self._calculate_practical_features(window_data)
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def _calculate_practical_features(self, data):
        """Calculate the same 15 features used in training"""
        try:
            import ta
            
            features = {}
            
            # Price features
            returns = data['close'].pct_change()
            features['current_return'] = returns.iloc[-1] if len(returns) > 1 else 0
            features['return_volatility'] = returns.std()
            features['return_skew'] = returns.skew() if len(returns) > 2 else 0
            
            # Trend features
            sma_10 = data['close'].rolling(10).mean()
            sma_20 = data['close'].rolling(20).mean()
            features['price_vs_sma10'] = (data['close'].iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1] if sma_10.iloc[-1] != 0 else 0
            features['price_vs_sma20'] = (data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if sma_20.iloc[-1] != 0 else 0
            features['sma_alignment'] = 1 if sma_10.iloc[-1] > sma_20.iloc[-1] else -1
            
            # Volatility features
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
            features['atr'] = atr.iloc[-1] / data['close'].iloc[-1] if atr.iloc[-1] != 0 else 0
            
            bb = ta.volatility.BollingerBands(data['close'])
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = (bb_upper - bb_lower) / data['close']
            features['bb_width'] = bb_width.iloc[-1] if len(bb_width) > 0 else 0
            features['bb_position'] = ((data['close'].iloc[-1] - bb_lower.iloc[-1]) / 
                                     (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) != 0 else 0.5
            
            # Momentum features
            rsi = ta.momentum.RSIIndicator(data['close']).rsi()
            features['rsi'] = rsi.iloc[-1] / 100 if len(rsi) > 0 else 0.5
            
            macd = ta.trend.MACD(data['close'])
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            features['macd_position'] = 1 if macd_line.iloc[-1] > macd_signal.iloc[-1] else -1
            
            # Volume features
            features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].mean()
            
            obv = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
            obv_sma = obv.rolling(10).mean()
            features['obv_trend'] = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
            
            # Range features
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
            features['range_position'] = ((data['close'].iloc[-1] - low_20.iloc[-1]) / 
                                         (high_20.iloc[-1] - low_20.iloc[-1])) if (high_20.iloc[-1] - low_20.iloc[-1]) != 0 else 0.5
            
            # Market structure features
            higher_highs = (data['high'].iloc[-1] > data['high'].iloc[-2] and 
                           data['high'].iloc[-2] > data['high'].iloc[-3]) if len(data) >= 3 else False
            lower_lows = (data['low'].iloc[-1] < data['low'].iloc[-2] and 
                         data['low'].iloc[-2] < data['low'].iloc[-3]) if len(data) >= 3 else False
            
            features['market_structure'] = 1 if higher_highs else (-1 if lower_lows else 0)
            
            # Convert to array in same order as training
            feature_names = ['current_return', 'return_volatility', 'return_skew', 
                           'price_vs_sma10', 'price_vs_sma20', 'sma_alignment',
                           'atr', 'bb_width', 'bb_position', 'rsi', 'macd_position',
                           'volume_ratio', 'obv_trend', 'range_position', 'market_structure']
            
            return np.array([features[name] for name in feature_names])
            
        except Exception as e:
            return None

def test_integration():
    """Test the practical ML integration"""
    print("🧪 Testing Practical ML Integration")
    print("=" * 40)
    
    # Initialize integrator
    integrator = PracticalMLIntegrator()
    
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100) * 0.5,
        'high': 100 + np.random.randn(100) * 0.5 + 0.2,
        'low': 100 + np.random.randn(100) * 0.5 - 0.2,
        'close': 100 + np.random.randn(100) * 0.5,
        'volume': 1000000 + np.random.randn(100) * 100000
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = np.maximum(sample_data['high'], sample_data[['open', 'close']].max(axis=1))
    sample_data['low'] = np.minimum(sample_data['low'], sample_data[['open', 'close']].min(axis=1))
    
    # Test ML signals
    print("Testing ML signal generation...")
    signals = integrator.get_trading_signals(sample_data)
    
    print("ML Signals:")
    for key, value in signals.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test parameter enhancement
    print("\nTesting parameter enhancement...")
    base_params = {
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.0
    }
    
    enhanced_params = integrator.enhance_regime_parameters(base_params, 'TRENDING_BULL', sample_data)
    
    print("Enhanced Parameters:")
    for key, value in enhanced_params.items():
        if key.startswith('ml_'):
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.3f}")
    
    # Test diagnostics
    print("\nTesting ML diagnostics...")
    diagnostics = integrator.get_ml_diagnostic_info(sample_data)
    
    print("ML Diagnostics:")
    print(f"  Status: {diagnostics['status']}")
    if 'interpretation' in diagnostics:
        for key, value in diagnostics['interpretation'].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_integration()
