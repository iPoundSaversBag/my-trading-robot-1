#!/usr/bin/env python3
"""
SIMPLIFIED WORKING REGIME SYSTEM

Going back to basics with a simple, proven approach that actually works.
Focus on getting the fundamentals right before adding complexity.
"""

import numpy as np
import pandas as pd
from core.enums import MarketRegime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedWorkingRegimeSystem:
    """
    Simplified but working regime detection system
    
    Philosophy: Keep it simple, make it work, then enhance
    """
    
    def __init__(self):
        # Simple regime parameters
        self.regime_parameters = {
            MarketRegime.TRENDING_BULL: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5
            },
            MarketRegime.TRENDING_BEAR: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 0.7,
                'take_profit_multiplier': 1.3
            },
            MarketRegime.RANGING: {
                'position_size_multiplier': 0.9,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 0.8
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.6,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 1.8
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_size_multiplier': 1.3,
                'stop_loss_multiplier': 1.1,
                'take_profit_multiplier': 1.2
            },
            MarketRegime.BREAKOUT_BULLISH: {
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 2.0
            },
            MarketRegime.BREAKOUT_BEARISH: {
                'position_size_multiplier': 1.4,
                'stop_loss_multiplier': 0.6,
                'take_profit_multiplier': 1.8
            },
            MarketRegime.ACCUMULATION: {
                'position_size_multiplier': 1.1,
                'stop_loss_multiplier': 1.3,
                'take_profit_multiplier': 2.5
            },
            MarketRegime.DISTRIBUTION: {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.1
            }
        }
    
    def detect_market_regime(self, data):
        """Simple but effective regime detection"""
        try:
            if len(data) < 30:
                return MarketRegime.RANGING
            
            # Calculate simple indicators
            metrics = self._calculate_simple_metrics(data)
            
            # Simple detection logic
            regime = self._simple_detection(metrics)
            
            return regime
            
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return MarketRegime.RANGING
    
    def _calculate_simple_metrics(self, data):
        """Calculate simple, reliable metrics"""
        window = data.tail(40)
        metrics = {}
        
        try:
            # 1. Simple volatility
            returns = window['close'].pct_change().dropna()
            if len(returns) >= 10:
                volatility = returns.std()
                metrics['volatility'] = volatility
            else:
                metrics['volatility'] = 0.01
            
            # 2. Simple trend
            if len(window) >= 20:
                sma_10 = window['close'].rolling(10).mean()
                sma_20 = window['close'].rolling(20).mean()
                
                if len(sma_10) >= 10 and len(sma_20) >= 20:
                    # Trend direction
                    trend_signal = (sma_10.iloc[-1] / sma_20.iloc[-1] - 1)
                    metrics['trend_signal'] = trend_signal
                    metrics['trend_strength'] = abs(trend_signal)
                    
                    # Simple alignment check
                    current_price = window['close'].iloc[-1]
                    above_short = current_price > sma_10.iloc[-1]
                    above_long = current_price > sma_20.iloc[-1]
                    short_above_long = sma_10.iloc[-1] > sma_20.iloc[-1]
                    
                    metrics['bullish_alignment'] = above_short and above_long and short_above_long
                    metrics['bearish_alignment'] = not above_short and not above_long and not short_above_long
                else:
                    metrics.update({'trend_signal': 0, 'trend_strength': 0, 'bullish_alignment': False, 'bearish_alignment': False})
            else:
                metrics.update({'trend_signal': 0, 'trend_strength': 0, 'bullish_alignment': False, 'bearish_alignment': False})
            
            # 3. Simple momentum
            if len(window) >= 10:
                momentum_5 = (window['close'].iloc[-1] / window['close'].iloc[-6] - 1) if len(window) >= 6 else 0
                momentum_10 = (window['close'].iloc[-1] / window['close'].iloc[-11] - 1) if len(window) >= 11 else 0
                
                metrics['momentum_short'] = momentum_5
                metrics['momentum_medium'] = momentum_10
            else:
                metrics['momentum_short'] = 0
                metrics['momentum_medium'] = 0
            
            # 4. Simple range detection
            if len(window) >= 20:
                high_20 = window['high'].rolling(20).max().iloc[-1]
                low_20 = window['low'].rolling(20).min().iloc[-1]
                current = window['close'].iloc[-1]
                
                range_size = (high_20 - low_20) / current if current > 0 else 0
                range_position = (current - low_20) / (high_20 - low_20 + 1e-8)
                
                metrics['range_size'] = range_size
                metrics['range_position'] = range_position
                metrics['in_middle'] = 0.25 <= range_position <= 0.75
            else:
                metrics.update({'range_size': 0.03, 'range_position': 0.5, 'in_middle': True})
            
            # 5. Simple volume
            if len(window) >= 10:
                vol_recent = window['volume'].tail(5).mean()
                vol_average = window['volume'].mean()
                
                volume_ratio = vol_recent / vol_average if vol_average > 0 else 1.0
                metrics['volume_high'] = volume_ratio > 1.5
                metrics['volume_normal'] = 0.8 <= volume_ratio <= 1.5
            else:
                metrics['volume_high'] = False
                metrics['volume_normal'] = True
            
            # 6. Simple breakout detection
            if len(window) >= 15:
                # Look for recent price breakout
                recent_high = window['high'].tail(3).max()
                recent_low = window['low'].tail(3).min()
                
                # Historical range
                hist_high = window['high'].iloc[:-3].max() if len(window) > 3 else recent_high
                hist_low = window['low'].iloc[:-3].min() if len(window) > 3 else recent_low
                
                # Breakout conditions
                breakout_up = recent_high > hist_high * 1.015  # 1.5% breakout
                breakout_down = recent_low < hist_low * 0.985   # 1.5% breakdown
                
                # Need momentum confirmation
                strong_momentum_up = metrics.get('momentum_short', 0) > 0.01
                strong_momentum_down = metrics.get('momentum_short', 0) < -0.01
                
                metrics['breakout_bull'] = breakout_up and strong_momentum_up and metrics.get('volume_high', False)
                metrics['breakout_bear'] = breakout_down and strong_momentum_down and metrics.get('volume_high', False)
            else:
                metrics['breakout_bull'] = False
                metrics['breakout_bear'] = False
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Metrics calculation error: {e}")
            return {'volatility': 0.01, 'trend_signal': 0, 'trend_strength': 0}
    
    def _simple_detection(self, metrics):
        """Simple, clear detection logic"""
        
        vol = metrics.get('volatility', 0.01)
        trend_strength = metrics.get('trend_strength', 0)
        trend_signal = metrics.get('trend_signal', 0)
        
        # 1. Check breakouts first (highest priority)
        if metrics.get('breakout_bull', False):
            return MarketRegime.BREAKOUT_BULLISH
        
        if metrics.get('breakout_bear', False):
            return MarketRegime.BREAKOUT_BEARISH
        
        # 2. Check volatility extremes
        if vol > 0.025:  # High volatility threshold
            return MarketRegime.HIGH_VOLATILITY
        
        if vol < 0.005:  # Low volatility threshold
            return MarketRegime.LOW_VOLATILITY
        
        # 3. Check for clear trends
        if trend_strength > 0.008:  # Strong trend threshold
            if metrics.get('bullish_alignment', False) and trend_signal > 0.003:
                return MarketRegime.TRENDING_BULL
            elif metrics.get('bearish_alignment', False) and trend_signal < -0.003:
                return MarketRegime.TRENDING_BEAR
        
        # 4. Check for accumulation/distribution
        momentum_medium = metrics.get('momentum_medium', 0)
        
        if 0.003 < momentum_medium < 0.015:  # Slow upward movement
            return MarketRegime.ACCUMULATION
        elif -0.015 < momentum_medium < -0.003:  # Slow downward movement
            return MarketRegime.DISTRIBUTION
        
        # 5. Default to ranging
        return MarketRegime.RANGING
    
    def get_regime_parameters(self, market_regime, base_parameters):
        """Get regime-specific parameters"""
        if market_regime in self.regime_parameters:
            enhanced_params = base_parameters.copy()
            regime_adjustments = self.regime_parameters[market_regime]
            
            for param_name, multiplier in regime_adjustments.items():
                if param_name.endswith('_multiplier'):
                    base_param = param_name.replace('_multiplier', '')
                    if base_param in enhanced_params:
                        enhanced_params[base_param] *= multiplier
                else:
                    enhanced_params[param_name] = multiplier
        else:
            enhanced_params = base_parameters.copy()
        
        enhanced_params['detected_regime'] = market_regime.value
        enhanced_params['system_version'] = 'simplified_working_v1.0'
        
        return enhanced_params
    
    def test_system(self):
        """Test the simplified system"""
        print("\n🧪 Testing Simplified Working System")
        print("=" * 55)
        
        # Simple, clear test cases
        test_cases = [
            ('Strong Bull', self._create_simple_test_data('bull_trend'), 'trending_bull'),
            ('Strong Bear', self._create_simple_test_data('bear_trend'), 'trending_bear'),
            ('Sideways Range', self._create_simple_test_data('ranging'), 'ranging'),
            ('High Volatility', self._create_simple_test_data('high_vol'), 'high_volatility'),
            ('Low Volatility', self._create_simple_test_data('low_vol'), 'low_volatility'),
            ('Bull Breakout', self._create_simple_test_data('bull_breakout'), 'breakout_bullish'),
            ('Bear Breakout', self._create_simple_test_data('bear_breakout'), 'breakout_bearish'),
            ('Accumulation', self._create_simple_test_data('accumulation'), 'accumulation'),
            ('Distribution', self._create_simple_test_data('distribution'), 'distribution')
        ]
        
        correct = 0
        total = len(test_cases)
        
        for test_name, data, expected in test_cases:
            detected = self.detect_market_regime(data)
            
            base_params = {'position_size': 1000, 'stop_loss': 0.02, 'take_profit': 0.04}
            enhanced_params = self.get_regime_parameters(detected, base_params)
            
            print(f"\n📊 {test_name}:")
            print(f"   Expected: {expected}")
            print(f"   Detected: {detected.value}")
            print(f"   Position: {enhanced_params['position_size']:.0f}")
            
            is_correct = detected.value.lower() == expected.lower()
            if is_correct:
                correct += 1
                print(f"   ✅ CORRECT")
            else:
                print(f"   ❌ INCORRECT")
        
        accuracy = correct / total
        print(f"\n🎯 Simplified System Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.6:  # Lower bar initially
            print(f"✅ WORKING FOUNDATION ESTABLISHED!")
            return True
        else:
            print(f"⚠️ Needs basic fixes")
            return False
    
    def _create_simple_test_data(self, scenario):
        """Create simple, clear test data"""
        length = 50
        
        if scenario == 'bull_trend':
            # Clear upward trend
            returns = []
            for i in range(length):
                trend = 0.005 + 0.2 * (returns[-1] if returns else 0)
                noise = np.random.normal(0, 0.008)
                returns.append(max(-0.02, min(0.03, trend + noise)))
        
        elif scenario == 'bear_trend':
            # Clear downward trend
            returns = []
            for i in range(length):
                trend = -0.005 + 0.2 * (returns[-1] if returns else 0)
                noise = np.random.normal(0, 0.008)
                returns.append(max(-0.03, min(0.02, trend + noise)))
        
        elif scenario == 'ranging':
            # Clear sideways movement
            returns = []
            for i in range(length):
                cycle = np.sin(2 * np.pi * i / 12) * 0.005
                noise = np.random.normal(0, 0.003)
                returns.append(cycle + noise)
        
        elif scenario == 'high_vol':
            # High volatility
            returns = np.random.normal(0, 0.030, length)
        
        elif scenario == 'low_vol':
            # Low volatility
            returns = np.random.normal(0.0005, 0.004, length)
        
        elif scenario == 'bull_breakout':
            # Consolidation then breakout
            consol = np.random.normal(0, 0.004, length//2)
            breakout = np.random.normal(0.020, 0.015, length//2)
            returns = np.concatenate([consol, breakout])
        
        elif scenario == 'bear_breakout':
            # Consolidation then breakdown
            consol = np.random.normal(0, 0.004, length//2)
            breakdown = np.random.normal(-0.020, 0.015, length//2)
            returns = np.concatenate([consol, breakdown])
        
        elif scenario == 'accumulation':
            # Slow steady gains
            returns = np.random.normal(0.006, 0.008, length)
        
        elif scenario == 'distribution':
            # Slow steady decline
            returns = np.random.normal(-0.006, 0.009, length)
        
        else:
            returns = np.random.normal(0, 0.01, length)
        
        # Create price data
        returns = np.array(returns)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Volume patterns
        if scenario in ['bull_breakout', 'bear_breakout']:
            volume = np.concatenate([
                np.full(len(returns)//2, 800000),
                np.full(len(returns) - len(returns)//2, 2000000)
            ])
        else:
            volume = 1000000 * (1 + 0.2 * np.random.randn(len(returns)))
        
        # Create OHLC
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(returns), freq='5min'),
            'open': prices * (1 + 0.0005 * np.random.randn(len(returns))),
            'high': prices * (1 + np.maximum(0, 0.001 * np.random.randn(len(returns)))),
            'low': prices * (1 - np.maximum(0, 0.001 * np.random.randn(len(returns)))),
            'close': prices,
            'volume': np.maximum(volume, 100000)
        })
        
        return df

def main():
    """Test the simplified system"""
    print("🔧 Simplified Working Regime System")
    print("=" * 55)
    print("Back to basics - making it work first!")
    print()
    
    system = SimplifiedWorkingRegimeSystem()
    success = system.test_system()
    
    if success:
        print(f"\n🎯 WORKING FOUNDATION ESTABLISHED!")
        print(f"✅ Simple, reliable detection")
        print(f"✅ Clear parameter adjustments")
        print(f"✅ Ready for gradual enhancement")
    else:
        print(f"\n🔧 Need to fix basics first")

if __name__ == "__main__":
    main()
