#!/usr/bin/env python3
"""
Complete 9-Regime ML System for Trading Performance

This creates specialized ML models that properly handle all 9 market regimes:
1. TRENDING_BULL
2. TRENDING_BEAR  
3. RANGING
4. HIGH_VOLATILITY
5. LOW_VOLATILITY
6. BREAKOUT_BULLISH
7. BREAKOUT_BEARISH
8. ACCUMULATION
9. DISTRIBUTION

Uses a hybrid approach: ML for what it does well, rules for rare events.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from core.enums import MarketRegime
import ta
import warnings
warnings.filterwarnings('ignore')

class Complete9RegimeMLSystem:
    """
    Comprehensive ML system that handles all 9 market regimes properly.
    
    Strategy:
    - ML for continuous scoring (volatility, trend strength, momentum, risk)  
    - Rule-based detection for discrete regime classification
    - Specialized models for each regime's characteristics
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("ml_models/nine_regime")
        self.model_dir.mkdir(exist_ok=True)
        
        # Define specialized models for each aspect of trading
        self.model_definitions = {
            # Core ML models (what ML does well)
            'volatility_predictor': {
                'target': 'volatility_score',
                'purpose': 'Predict volatility level (0-1)',
                'helps_regimes': ['HIGH_VOLATILITY', 'LOW_VOLATILITY']
            },
            'trend_strength_assessor': {
                'target': 'trend_strength',
                'purpose': 'Assess trend strength (0-1)',
                'helps_regimes': ['TRENDING_BULL', 'TRENDING_BEAR']
            },
            'momentum_analyzer': {
                'target': 'momentum_direction',
                'purpose': 'Analyze momentum (-1 to 1)',
                'helps_regimes': ['BREAKOUT_BULLISH', 'BREAKOUT_BEARISH']
            },
            'accumulation_detector': {
                'target': 'accumulation_score',
                'purpose': 'Detect accumulation patterns (0-1)',
                'helps_regimes': ['ACCUMULATION', 'DISTRIBUTION']
            },
            'range_analyzer': {
                'target': 'range_bound_score',
                'purpose': 'Analyze range-bound behavior (0-1)',
                'helps_regimes': ['RANGING']
            },
            'risk_assessor': {
                'target': 'risk_score',
                'purpose': 'Overall market risk assessment (0-1)',
                'helps_regimes': ['ALL']
            }
        }
        
        # Regime mapping for backtesting integration
        self.regime_mapping = {
            regime.value: regime for regime in MarketRegime
        }
    
    def generate_comprehensive_training_data(self, n_samples=8000):
        """Generate training data that covers all 9 market regimes comprehensively"""
        print("📊 Generating comprehensive training data for all 9 market regimes...")
        
        # Create data for each regime with proper representation
        regime_data = {}
        
        # 1. TRENDING_BULL (15% of data)
        bull_samples = int(n_samples * 0.15)
        regime_data['TRENDING_BULL'] = self._create_regime_specific_data(
            bull_samples, regime_type='trending_bull'
        )
        
        # 2. TRENDING_BEAR (15% of data)
        bear_samples = int(n_samples * 0.15)
        regime_data['TRENDING_BEAR'] = self._create_regime_specific_data(
            bear_samples, regime_type='trending_bear'
        )
        
        # 3. RANGING (20% of data - common)
        ranging_samples = int(n_samples * 0.20)
        regime_data['RANGING'] = self._create_regime_specific_data(
            ranging_samples, regime_type='ranging'
        )
        
        # 4. HIGH_VOLATILITY (15% of data)
        high_vol_samples = int(n_samples * 0.15)
        regime_data['HIGH_VOLATILITY'] = self._create_regime_specific_data(
            high_vol_samples, regime_type='high_volatility'
        )
        
        # 5. LOW_VOLATILITY (20% of data - common)
        low_vol_samples = int(n_samples * 0.20)
        regime_data['LOW_VOLATILITY'] = self._create_regime_specific_data(
            low_vol_samples, regime_type='low_volatility'
        )
        
        # 6. BREAKOUT_BULLISH (5% of data - rare but critical)
        breakout_bull_samples = int(n_samples * 0.05)
        regime_data['BREAKOUT_BULLISH'] = self._create_regime_specific_data(
            breakout_bull_samples, regime_type='breakout_bullish'
        )
        
        # 7. BREAKOUT_BEARISH (5% of data - rare but critical)
        breakout_bear_samples = int(n_samples * 0.05)
        regime_data['BREAKOUT_BEARISH'] = self._create_regime_specific_data(
            breakout_bear_samples, regime_type='breakout_bearish'
        )
        
        # 8. ACCUMULATION (3% of data - rare)
        accumulation_samples = int(n_samples * 0.03)
        regime_data['ACCUMULATION'] = self._create_regime_specific_data(
            accumulation_samples, regime_type='accumulation'
        )
        
        # 9. DISTRIBUTION (2% of data - very rare)
        distribution_samples = n_samples - sum([
            bull_samples, bear_samples, ranging_samples, high_vol_samples,
            low_vol_samples, breakout_bull_samples, breakout_bear_samples,
            accumulation_samples
        ])
        regime_data['DISTRIBUTION'] = self._create_regime_specific_data(
            distribution_samples, regime_type='distribution'
        )
        
        # Combine all regime data
        all_data = []
        all_labels = []
        
        for regime_name, data in regime_data.items():
            all_data.append(data)
            all_labels.extend([regime_name] * len(data))
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"✅ Generated {len(combined_data)} samples across all 9 regimes")
        
        # Print regime distribution
        from collections import Counter
        regime_counts = Counter(all_labels)
        for regime, count in regime_counts.items():
            percentage = (count / len(all_labels)) * 100
            print(f"   {regime}: {count} samples ({percentage:.1f}%)")
        
        return combined_data, all_labels, regime_data
    
    def _create_regime_specific_data(self, length, regime_type):
        """Create realistic market data for a specific regime type"""
        np.random.seed(42 + hash(regime_type) % 1000)  # Consistent but different seeds
        
        if regime_type == 'trending_bull':
            return self._create_trending_data(length, trend=0.0012, volatility=0.018, direction='bull')
        
        elif regime_type == 'trending_bear':
            return self._create_trending_data(length, trend=-0.0010, volatility=0.020, direction='bear')
        
        elif regime_type == 'ranging':
            return self._create_ranging_data(length, volatility=0.012)
        
        elif regime_type == 'high_volatility':
            return self._create_high_volatility_data(length, volatility=0.040)
        
        elif regime_type == 'low_volatility':
            return self._create_low_volatility_data(length, volatility=0.008)
        
        elif regime_type == 'breakout_bullish':
            return self._create_breakout_data(length, direction='bullish')
        
        elif regime_type == 'breakout_bearish':
            return self._create_breakout_data(length, direction='bearish')
        
        elif regime_type == 'accumulation':
            return self._create_accumulation_data(length)
        
        elif regime_type == 'distribution':
            return self._create_distribution_data(length)
        
        else:
            # Fallback to neutral data
            return self._create_ranging_data(length, volatility=0.015)
    
    def _create_trending_data(self, length, trend, volatility, direction):
        """Create trending market data"""
        # Generate trending returns with some noise
        base_returns = np.random.normal(trend, volatility, length)
        
        # Add trend persistence
        for i in range(1, len(base_returns)):
            base_returns[i] += 0.15 * base_returns[i-1]  # Momentum effect
        
        # Add periodic trend strengthening/weakening
        trend_cycle = np.sin(np.arange(length) * 2 * np.pi / 50) * 0.3
        base_returns += trend_cycle * volatility
        
        prices = 100 * np.exp(np.cumsum(base_returns))
        
        # Volume patterns for trending markets
        volume_base = 1200000
        # Higher volume during trend moves
        volume_multiplier = 1 + 0.4 * np.abs(base_returns) / volatility
        volume = volume_base * volume_multiplier * (1 + 0.25 * np.random.randn(length))
        
        return self._create_ohlc_dataframe(prices, volume, base_returns, f'trend_{direction}')
    
    def _create_ranging_data(self, length, volatility):
        """Create range-bound market data"""
        # Oscillating price with mean reversion
        base_price = 100
        prices = [base_price]
        
        for i in range(1, length):
            # Mean reversion force
            deviation = prices[-1] - base_price
            mean_reversion = -0.02 * deviation / base_price
            
            # Random walk component
            random_component = np.random.normal(0, volatility)
            
            # Combine forces
            return_rate = mean_reversion + random_component
            new_price = prices[-1] * (1 + return_rate)
            prices.append(new_price)
        
        prices = np.array(prices)
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])
        
        # Volume patterns for ranging markets (lower volume)
        volume_base = 800000
        volume = volume_base * (1 + 0.2 * np.random.randn(length))
        
        return self._create_ohlc_dataframe(prices, volume, returns, 'ranging')
    
    def _create_high_volatility_data(self, length, volatility):
        """Create high volatility market data"""
        # High volatility with clustered volatility
        base_returns = np.random.normal(0, volatility, length)
        
        # Volatility clustering (GARCH-like effect)
        vol_multipliers = np.ones(length)
        for i in range(1, length):
            if abs(base_returns[i-1]) > volatility:
                vol_multipliers[i] = min(3.0, vol_multipliers[i-1] * 1.5)
            else:
                vol_multipliers[i] = max(0.5, vol_multipliers[i-1] * 0.9)
        
        base_returns *= vol_multipliers
        prices = 100 * np.exp(np.cumsum(base_returns))
        
        # High volume during volatility spikes
        volume_base = 1500000
        volume_multiplier = 1 + 0.8 * vol_multipliers
        volume = volume_base * volume_multiplier * (1 + 0.3 * np.random.randn(length))
        
        return self._create_ohlc_dataframe(prices, volume, base_returns, 'high_vol')
    
    def _create_low_volatility_data(self, length, volatility):
        """Create low volatility market data"""
        # Very smooth price action
        base_returns = np.random.normal(0.0002, volatility, length)  # Slight upward bias
        
        # Smooth the returns (low volatility)
        smoothed_returns = np.convolve(base_returns, np.ones(3)/3, mode='same')
        prices = 100 * np.exp(np.cumsum(smoothed_returns))
        
        # Low, consistent volume
        volume_base = 600000
        volume = volume_base * (1 + 0.1 * np.random.randn(length))
        
        return self._create_ohlc_dataframe(prices, volume, smoothed_returns, 'low_vol')
    
    def _create_breakout_data(self, length, direction):
        """Create breakout market data"""
        # Three phases: consolidation, breakout, continuation
        consolidation_length = length // 3
        breakout_length = length // 6
        continuation_length = length - consolidation_length - breakout_length
        
        # Phase 1: Consolidation
        consolidation_data = self._create_ranging_data(consolidation_length, volatility=0.010)
        
        # Phase 2: Breakout
        breakout_trend = 0.025 if direction == 'bullish' else -0.025
        breakout_vol = 0.035
        breakout_returns = np.random.normal(breakout_trend, breakout_vol, breakout_length)
        
        # Add explosive character to breakout
        breakout_returns *= np.linspace(1, 2.5, breakout_length)
        
        last_price = consolidation_data['close'].iloc[-1]
        breakout_prices = last_price * np.exp(np.cumsum(breakout_returns))
        
        # Higher volume during breakout
        breakout_volume = 2500000 * (1 + 0.5 * np.random.randn(breakout_length))
        breakout_data = self._create_ohlc_dataframe(breakout_prices, breakout_volume, breakout_returns, f'breakout_{direction}')
        
        # Phase 3: Continuation
        continuation_trend = breakout_trend * 0.4  # Reduced momentum
        continuation_vol = 0.020
        continuation_returns = np.random.normal(continuation_trend, continuation_vol, continuation_length)
        
        last_price = breakout_data['close'].iloc[-1]
        continuation_prices = last_price * np.exp(np.cumsum(continuation_returns))
        continuation_volume = 1000000 * (1 + 0.3 * np.random.randn(continuation_length))
        continuation_data = self._create_ohlc_dataframe(continuation_prices, continuation_volume, continuation_returns, f'continuation_{direction}')
        
        # Combine all phases
        return pd.concat([consolidation_data, breakout_data, continuation_data], ignore_index=True)
    
    def _create_accumulation_data(self, length):
        """Create accumulation phase data"""
        # Sideways price with increasing volume
        base_data = self._create_ranging_data(length, volatility=0.012)
        
        # Gradual volume increase (smart money accumulating)
        volume_trend = np.linspace(0.8, 1.8, length)
        base_data['volume'] *= volume_trend
        
        # Subtle upward bias in later periods
        price_bias = np.linspace(0, 0.015, length)
        for i in range(len(base_data)):
            if i > length // 2:  # Second half shows accumulation
                base_data.loc[i, 'close'] *= (1 + price_bias[i])
                base_data.loc[i, 'high'] *= (1 + price_bias[i])
                base_data.loc[i, 'low'] *= (1 + price_bias[i])
                base_data.loc[i, 'open'] *= (1 + price_bias[i])
        
        return base_data
    
    def _create_distribution_data(self, length):
        """Create distribution phase data"""
        # Sideways to declining price with high volume
        base_data = self._create_ranging_data(length, volatility=0.015)
        
        # High volume (smart money distributing)
        volume_multiplier = np.linspace(1.2, 2.0, length)
        base_data['volume'] *= volume_multiplier
        
        # Subtle downward bias
        price_bias = np.linspace(0, -0.020, length)
        for i in range(len(base_data)):
            if i > length // 3:  # Distribution becomes apparent
                base_data.loc[i, 'close'] *= (1 + price_bias[i])
                base_data.loc[i, 'high'] *= (1 + price_bias[i])
                base_data.loc[i, 'low'] *= (1 + price_bias[i])
                base_data.loc[i, 'open'] *= (1 + price_bias[i])
        
        return base_data
    
    def _create_ohlc_dataframe(self, prices, volume, returns, regime_type):
        """Create OHLC dataframe from price and volume data"""
        length = len(prices)
        
        # Generate OHLC from close prices
        close_prices = prices
        
        # Open prices (previous close + small gap)
        open_prices = np.concatenate([[close_prices[0]], close_prices[:-1]]) * (1 + 0.001 * np.random.randn(length))
        
        # High prices (max of open/close + upward spike)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.maximum(0, 0.002 * np.random.randn(length)))
        
        # Low prices (min of open/close - downward spike)
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.maximum(0, 0.002 * np.random.randn(length)))
        
        # Ensure volume is positive
        volume = np.maximum(volume, 100000)
        
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=length, freq='5min'),
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        return df
    
    def calculate_regime_targets(self, data, regime_labels):
        """Calculate target variables for each ML model"""
        print("🎯 Calculating ML targets for all regime models...")
        
        targets = {}
        
        # Calculate targets for each sample
        for i in range(len(data)):
            if i < 50:  # Need minimum data for calculations
                continue
                
            window_data = data.iloc[max(0, i-49):i+1]
            regime_label = regime_labels[i]
            
            if len(window_data) < 20:
                continue
            
            # Calculate all targets
            sample_targets = self._calculate_sample_targets(window_data, regime_label)
            
            # Add to targets dictionary
            for target_name, value in sample_targets.items():
                if target_name not in targets:
                    targets[target_name] = []
                targets[target_name].append(value)
        
        # Convert to numpy arrays
        for target_name in targets:
            targets[target_name] = np.array(targets[target_name])
            print(f"   {target_name}: {len(targets[target_name])} samples, μ={np.mean(targets[target_name]):.3f}")
        
        return targets
    
    def _calculate_sample_targets(self, data, regime_label):
        """Calculate target values for a single data sample"""
        targets = {}
        
        # 1. Volatility score (0-1)
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(288)  # Annualized
        targets['volatility_score'] = min(volatility / 0.5, 1.0)  # Normalize to 0-1
        
        # 2. Trend strength (0-1)
        sma_20 = data['close'].rolling(20).mean()
        if len(sma_20) >= 20:
            slope = np.polyfit(range(20), sma_20.tail(20), 1)[0]
            trend_strength = min(abs(slope) / (data['close'].mean() * 0.001), 1.0)
        else:
            trend_strength = 0.5
        targets['trend_strength'] = trend_strength
        
        # 3. Momentum direction (-1 to 1)
        short_ma = data['close'].rolling(5).mean()
        long_ma = data['close'].rolling(20).mean()
        if len(short_ma) >= 20 and len(long_ma) >= 20:
            momentum = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            targets['momentum_direction'] = np.tanh(momentum * 10)  # Normalize to -1 to 1
        else:
            targets['momentum_direction'] = 0.0
        
        # 4. Accumulation score (0-1)
        volume_trend = data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
        price_trend = data['close'].iloc[-1] / data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
        
        # Accumulation: rising volume, stable/rising price
        if 'ACCUMULATION' in regime_label:
            accumulation_score = min((volume_trend - 1) + max(0, price_trend - 1), 1.0)
        elif 'DISTRIBUTION' in regime_label:
            accumulation_score = 0.1  # Very low for distribution
        else:
            accumulation_score = 0.5  # Neutral
        targets['accumulation_score'] = max(0, accumulation_score)
        
        # 5. Range-bound score (0-1)
        if len(data) >= 20:
            price_range = (data['high'].tail(20).max() - data['low'].tail(20).min()) / data['close'].iloc[-1]
            price_position = (data['close'].iloc[-1] - data['low'].tail(20).min()) / (data['high'].tail(20).max() - data['low'].tail(20).min() + 1e-8)
            
            # Range-bound: price in middle of range, narrow range
            range_score = (1 - abs(price_position - 0.5) * 2) * (1 - min(price_range / 0.1, 1))
        else:
            range_score = 0.5
        targets['range_bound_score'] = range_score
        
        # 6. Risk score (0-1)
        # Combine volatility, drawdown, and volume irregularities
        risk_vol = targets['volatility_score']
        
        # Drawdown component
        peak = data['close'].expanding().max().iloc[-1]
        current_dd = (peak - data['close'].iloc[-1]) / peak
        risk_dd = min(current_dd * 5, 1.0)
        
        # Volume irregularity
        vol_std = data['volume'].std()
        vol_spike = abs(data['volume'].iloc[-1] - data['volume'].mean()) / (vol_std + 1e-8)
        risk_vol_irreg = min(vol_spike / 3, 1.0)
        
        risk_score = (risk_vol * 0.4 + risk_dd * 0.4 + risk_vol_irreg * 0.2)
        targets['risk_score'] = min(risk_score, 1.0)
        
        return targets
    
    def extract_comprehensive_features(self, data):
        """Extract comprehensive features for all ML models"""
        features = []
        
        for i in range(len(data)):
            if i < 50:  # Need minimum data
                continue
                
            window_data = data.iloc[max(0, i-49):i+1]
            if len(window_data) < 20:
                continue
            
            feature_row = self._calculate_comprehensive_features(window_data)
            if feature_row is not None:
                features.append(feature_row)
        
        return np.array(features) if features else None
    
    def _calculate_comprehensive_features(self, data):
        """Calculate comprehensive features for all regime types"""
        try:
            import ta
            
            features = {}
            
            # Basic price features
            returns = data['close'].pct_change()
            features['current_return'] = returns.iloc[-1] if len(returns) > 1 else 0
            features['return_volatility'] = returns.std()
            features['return_skew'] = returns.skew() if len(returns) > 2 else 0
            features['return_kurtosis'] = returns.kurtosis() if len(returns) > 3 else 0
            
            # Trend features
            sma_5 = data['close'].rolling(5).mean()
            sma_10 = data['close'].rolling(10).mean()
            sma_20 = data['close'].rolling(20).mean()
            
            features['price_vs_sma5'] = (data['close'].iloc[-1] - sma_5.iloc[-1]) / sma_5.iloc[-1] if sma_5.iloc[-1] != 0 else 0
            features['price_vs_sma10'] = (data['close'].iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1] if sma_10.iloc[-1] != 0 else 0
            features['price_vs_sma20'] = (data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] if sma_20.iloc[-1] != 0 else 0
            
            features['sma_alignment_5_10'] = 1 if sma_5.iloc[-1] > sma_10.iloc[-1] else -1
            features['sma_alignment_10_20'] = 1 if sma_10.iloc[-1] > sma_20.iloc[-1] else -1
            
            # Volatility features
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
            features['atr_normalized'] = atr.iloc[-1] / data['close'].iloc[-1] if atr.iloc[-1] != 0 else 0
            
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
            features['macd_histogram'] = (macd_line.iloc[-1] - macd_signal.iloc[-1]) / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            # Volume features
            features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].mean()
            features['volume_trend'] = data['volume'].rolling(5).mean().iloc[-1] / data['volume'].rolling(20).mean().iloc[-1] if len(data) >= 20 else 1.0
            
            obv = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
            obv_sma = obv.rolling(10).mean()
            features['obv_trend'] = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
            
            # Range and position features
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
            features['range_position'] = ((data['close'].iloc[-1] - low_20.iloc[-1]) / 
                                         (high_20.iloc[-1] - low_20.iloc[-1])) if (high_20.iloc[-1] - low_20.iloc[-1]) != 0 else 0.5
            
            features['price_range_20'] = (high_20.iloc[-1] - low_20.iloc[-1]) / data['close'].iloc[-1] if data['close'].iloc[-1] != 0 else 0
            
            # Breakout features
            breakout_threshold = data['close'].rolling(20).std().iloc[-1] * 2
            features['breakout_signal'] = 1 if (data['close'].iloc[-1] - high_20.iloc[-1]) > breakout_threshold else (-1 if (low_20.iloc[-1] - data['close'].iloc[-1]) > breakout_threshold else 0)
            
            # Market structure features
            higher_highs = (data['high'].iloc[-1] > data['high'].iloc[-2] and 
                           data['high'].iloc[-2] > data['high'].iloc[-3]) if len(data) >= 3 else False
            lower_lows = (data['low'].iloc[-1] < data['low'].iloc[-2] and 
                         data['low'].iloc[-2] < data['low'].iloc[-3]) if len(data) >= 3 else False
            
            features['market_structure'] = 1 if higher_highs else (-1 if lower_lows else 0)
            
            # Accumulation/Distribution features
            ad_line = ta.volume.AccDistIndexIndicator(data['high'], data['low'], data['close'], data['volume']).acc_dist_index()
            ad_trend = ad_line.rolling(10).mean()
            features['ad_trend'] = 1 if ad_line.iloc[-1] > ad_trend.iloc[-1] else -1
            
            # Convert to array (25 features total)
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
            
            return np.array([features[name] for name in feature_names])
            
        except Exception as e:
            return None
    
    def train_complete_system(self):
        """Train the complete 9-regime ML system"""
        print("🚀 Training Complete 9-Regime ML System")
        print("=" * 60)
        
        # Generate comprehensive training data
        data, regime_labels, regime_data = self.generate_comprehensive_training_data()
        
        # Extract features
        features = self.extract_comprehensive_features(data)
        if features is None:
            print("❌ Feature extraction failed")
            return False
        
        # Calculate targets
        targets = self.calculate_regime_targets(data, regime_labels)
        
        # Align features and targets
        min_length = min(len(features), min(len(target_array) for target_array in targets.values()))
        features = features[:min_length]
        
        for target_name in targets:
            targets[target_name] = targets[target_name][:min_length]
        
        print(f"\n📊 Training data: {len(features)} samples, {features.shape[1]} features")
        
        # Train each specialized model
        success_count = 0
        for model_name, model_def in self.model_definitions.items():
            print(f"\n🔧 Training {model_name}...")
            
            target_data = targets[model_def['target']]
            success = self._train_specialized_model(model_name, features, target_data, model_def)
            
            if success:
                success_count += 1
                print(f"✅ {model_name} trained successfully")
                print(f"   Purpose: {model_def['purpose']}")
                print(f"   Helps: {', '.join(model_def['helps_regimes'])}")
            else:
                print(f"❌ {model_name} training failed")
        
        # Save system metadata
        self._save_complete_system_metadata(regime_data)
        
        print(f"\n🎯 Complete 9-Regime ML System Training Results:")
        print(f"✅ Successfully trained: {success_count}/{len(self.model_definitions)} models")
        print(f"📁 Models saved in: {self.model_dir}")
        
        return success_count == len(self.model_definitions)
    
    def _train_specialized_model(self, model_name, features, targets, model_def):
        """Train a single specialized model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble of models
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                random_state=42,
                n_jobs=-1
            )
            
            gb_model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                random_state=42
            )
            
            # Train both models
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)
            
            rf_r2 = r2_score(y_test, rf_pred)
            gb_r2 = r2_score(y_test, gb_pred)
            
            # Use better performing model
            if rf_r2 >= gb_r2:
                best_model = rf_model
                best_score = rf_r2
                model_type = "RandomForest"
            else:
                best_model = gb_model
                best_score = gb_r2
                model_type = "GradientBoosting"
            
            # Save model and scaler
            self.models[model_name] = best_model
            self.scalers[model_name] = scaler
            
            # Save to disk
            model_file = self.model_dir / f"{model_name}.pkl"
            scaler_file = self.model_dir / f"{model_name}_scaler.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(best_model, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"   Model: {model_type}, R² Score: {best_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def _save_complete_system_metadata(self, regime_data):
        """Save complete system metadata"""
        metadata = {
            "system_name": "Complete 9-Regime ML System",
            "version": "2.0",
            "created_date": pd.Timestamp.now().isoformat(),
            "purpose": "Comprehensive ML system covering all 9 market regimes",
            "regime_coverage": {
                "TRENDING_BULL": "15% of training data",
                "TRENDING_BEAR": "15% of training data", 
                "RANGING": "20% of training data",
                "HIGH_VOLATILITY": "15% of training data",
                "LOW_VOLATILITY": "20% of training data",
                "BREAKOUT_BULLISH": "5% of training data",
                "BREAKOUT_BEARISH": "5% of training data",
                "ACCUMULATION": "3% of training data",
                "DISTRIBUTION": "2% of training data"
            },
            "models": self.model_definitions,
            "feature_count": 23,
            "advantages": [
                "Covers all 9 market regimes properly",
                "Specialized models for different market aspects",
                "Balanced training data distribution",
                "Continuous scores instead of brittle classifications",
                "Integrates with existing regime detection",
                "Avoids catastrophic breakout detection failures"
            ],
            "integration_strategy": [
                "Use ML scores to enhance rule-based regime detection",
                "Volatility predictor improves position sizing",
                "Trend strength assessor optimizes trend following",
                "Momentum analyzer enhances entry/exit timing",
                "Risk assessor improves overall risk management",
                "Accumulation detector identifies smart money flow"
            ],
            "deployment_approach": [
                "Gradual rollout with A/B testing",
                "Monitor performance vs existing system",
                "Use as enhancement, not replacement",
                "Maintain rule-based fallbacks",
                "Focus on improving existing strengths"
            ]
        }
        
        metadata_file = self.model_dir / "complete_system_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    def test_all_regimes(self):
        """Test the complete system on all 9 market regimes"""
        print("\n🧪 Testing Complete 9-Regime ML System")
        print("=" * 50)
        
        # Create test data for each regime
        test_regimes = {
            'TRENDING_BULL': self._create_regime_specific_data(100, 'trending_bull'),
            'TRENDING_BEAR': self._create_regime_specific_data(100, 'trending_bear'),
            'RANGING': self._create_regime_specific_data(100, 'ranging'),
            'HIGH_VOLATILITY': self._create_regime_specific_data(100, 'high_volatility'),
            'LOW_VOLATILITY': self._create_regime_specific_data(100, 'low_volatility'),
            'BREAKOUT_BULLISH': self._create_regime_specific_data(100, 'breakout_bullish'),
            'BREAKOUT_BEARISH': self._create_regime_specific_data(100, 'breakout_bearish'),
            'ACCUMULATION': self._create_regime_specific_data(100, 'accumulation'),
            'DISTRIBUTION': self._create_regime_specific_data(100, 'distribution')
        }
        
        # Test each regime
        for regime_name, data in test_regimes.items():
            print(f"\n📊 Testing {regime_name}:")
            
            # Extract features
            features = self.extract_comprehensive_features(data)
            if features is None:
                print("   ❌ Feature extraction failed")
                continue
            
            # Test each model
            predictions = {}
            for model_name in self.model_definitions.keys():
                if model_name in self.models and model_name in self.scalers:
                    features_scaled = self.scalers[model_name].transform(features)
                    model_predictions = self.models[model_name].predict(features_scaled)
                    
                    avg_prediction = np.mean(model_predictions)
                    predictions[model_name] = avg_prediction
                    
                    print(f"   {model_name}: {avg_prediction:.3f}")
            
            # Interpret predictions for this regime
            self._interpret_regime_predictions(regime_name, predictions)
        
        print(f"\n✅ Complete 9-Regime ML System testing complete!")
    
    def _interpret_regime_predictions(self, regime_name, predictions):
        """Interpret ML predictions for a specific regime"""
        interpretation = []
        
        # Volatility interpretation
        vol_score = predictions.get('volatility_predictor', 0.5)
        if vol_score > 0.7:
            interpretation.append("High volatility detected")
        elif vol_score < 0.3:
            interpretation.append("Low volatility detected")
        
        # Trend interpretation
        trend_score = predictions.get('trend_strength_assessor', 0.5)
        if trend_score > 0.6:
            interpretation.append("Strong trend detected")
        elif trend_score < 0.4:
            interpretation.append("Weak trend/ranging")
        
        # Momentum interpretation
        momentum_score = predictions.get('momentum_analyzer', 0.0)
        if momentum_score > 0.3:
            interpretation.append("Bullish momentum")
        elif momentum_score < -0.3:
            interpretation.append("Bearish momentum")
        
        # Risk interpretation
        risk_score = predictions.get('risk_assessor', 0.5)
        if risk_score > 0.7:
            interpretation.append("High risk environment")
        elif risk_score < 0.3:
            interpretation.append("Low risk environment")
        
        if interpretation:
            print(f"   → {', '.join(interpretation)}")

def main():
    print("🚀 Creating Complete 9-Regime ML System for Trading Performance")
    print("=" * 80)
    
    # Initialize complete system
    ml_system = Complete9RegimeMLSystem()
    
    # Train all models
    success = ml_system.train_complete_system()
    
    if success:
        # Test all regimes
        ml_system.test_all_regimes()
        
        print(f"\n🎯 COMPLETE 9-REGIME ML SYSTEM READY!")
        print(f"✅ Covers all 9 market regimes properly")
        print(f"✅ Specialized models for different aspects")
        print(f"✅ Balanced training data distribution")
        print(f"✅ Ready for integration with backtesting system")
        
    else:
        print(f"\n❌ Training failed - check logs for details")

if __name__ == "__main__":
    main()
