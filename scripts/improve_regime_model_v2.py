#!/usr/bin/env python3
"""
Option 2: Advanced Feature Engineering
Creates more sophisticated features specifically for breakout detection
"""

import pandas as pd
import numpy as np
import ta
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class AdvancedFeatureEngineer:
    """Advanced feature engineering for better regime classification"""
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50]
        
    def create_breakout_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically designed to detect breakouts"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        features = df.copy()
        
        # 1. Volume-Price Divergence Features
        features['volume_price_divergence'] = self._calculate_volume_price_divergence(close, volume)
        features['volume_breakout_score'] = self._calculate_volume_breakout_score(volume)
        
        # 2. Volatility Regime Change Detection
        features['volatility_regime_change'] = self._detect_volatility_regime_changes(close)
        features['volatility_acceleration'] = self._calculate_volatility_acceleration(close)
        
        # 3. Support/Resistance Breakout Features
        features['resistance_break_strength'] = self._calculate_resistance_break(high, close)
        features['support_break_strength'] = self._calculate_support_break(low, close)
        
        # 4. Momentum Convergence/Divergence
        features['momentum_convergence'] = self._calculate_momentum_convergence(close)
        features['price_momentum_ratio'] = self._calculate_price_momentum_ratio(close)
        
        # 5. Multi-timeframe Alignment
        features.update(self._create_multi_timeframe_features(close))
        
        # 6. Order Flow Proxy Features
        features['buying_pressure'] = self._calculate_buying_pressure(high, low, close, volume)
        features['selling_pressure'] = self._calculate_selling_pressure(high, low, close, volume)
        
        # 7. Fractal-based Features
        features['fractal_support'] = self._detect_fractal_support(low)
        features['fractal_resistance'] = self._detect_fractal_resistance(high)
        
        # 8. Volatility Clustering Features
        features['volatility_cluster_score'] = self._calculate_volatility_clustering(close)
        
        return features
    
    def _calculate_volume_price_divergence(self, close, volume):
        """Calculate divergence between volume and price trends"""
        price_trend = close.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        volume_trend = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Normalize trends
        price_trend_norm = (price_trend - price_trend.rolling(50).mean()) / price_trend.rolling(50).std()
        volume_trend_norm = (volume_trend - volume_trend.rolling(50).mean()) / volume_trend.rolling(50).std()
        
        return volume_trend_norm - price_trend_norm
    
    def _calculate_volume_breakout_score(self, volume):
        """Score likelihood of volume-based breakout"""
        volume_ma_short = volume.rolling(5).mean()
        volume_ma_long = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        
        # Volume spike above normal + increasing volume trend
        volume_spike = (volume - volume_ma_long) / volume_std
        volume_trend = (volume_ma_short - volume_ma_long) / volume_ma_long
        
        return volume_spike * (1 + volume_trend)
    
    def _detect_volatility_regime_changes(self, close):
        """Detect sudden changes in volatility regime"""
        volatility = close.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        vol_std = volatility.rolling(50).std()
        
        # Z-score of current volatility vs recent average
        vol_zscore = (volatility - vol_ma) / vol_std
        
        # Detect regime changes (volatility spikes)
        regime_change = np.abs(vol_zscore) > 2.0
        return regime_change.astype(int)
    
    def _calculate_volatility_acceleration(self, close):
        """Calculate rate of change in volatility"""
        volatility = close.rolling(10).std()
        vol_change = volatility.pct_change(periods=5)
        vol_acceleration = vol_change.pct_change(periods=3)
        return vol_acceleration
    
    def _calculate_resistance_break(self, high, close):
        """Calculate strength of resistance level breaks"""
        resistance_levels = high.rolling(50).max()
        distance_to_resistance = (close - resistance_levels) / resistance_levels
        
        # Strength = how far above resistance + momentum
        momentum = close.pct_change(periods=3)
        break_strength = np.where(
            distance_to_resistance > 0,
            distance_to_resistance * (1 + momentum * 10),
            0
        )
        return break_strength
    
    def _calculate_support_break(self, low, close):
        """Calculate strength of support level breaks"""
        support_levels = low.rolling(50).min()
        distance_to_support = (support_levels - close) / support_levels
        
        # Strength = how far below support + momentum
        momentum = close.pct_change(periods=3)
        break_strength = np.where(
            distance_to_support > 0,
            distance_to_support * (1 + abs(momentum) * 10),
            0
        )
        return break_strength
    
    def _calculate_momentum_convergence(self, close):
        """Calculate convergence of different momentum timeframes"""
        mom_fast = close.pct_change(periods=3)
        mom_medium = close.pct_change(periods=10)
        mom_slow = close.pct_change(periods=20)
        
        # Convergence = alignment of momentum across timeframes
        convergence = np.sign(mom_fast) * np.sign(mom_medium) * np.sign(mom_slow)
        return convergence * (abs(mom_fast) + abs(mom_medium) + abs(mom_slow)) / 3
    
    def _calculate_price_momentum_ratio(self, close):
        """Calculate ratio of short to long term momentum"""
        short_momentum = close.pct_change(periods=5)
        long_momentum = close.pct_change(periods=20)
        
        ratio = np.where(
            abs(long_momentum) > 0.001,
            short_momentum / long_momentum,
            0
        )
        return ratio
    
    def _create_multi_timeframe_features(self, close):
        """Create features across multiple timeframes"""
        features = {}
        
        # Trend alignment across timeframes
        for period in [5, 10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f'trend_{period}d'] = (close - sma) / sma
            
        # Trend strength (slope of moving averages)
        for period in [10, 20]:
            sma = close.rolling(period).mean()
            features[f'trend_strength_{period}d'] = sma.pct_change(periods=5)
            
        return features
    
    def _calculate_buying_pressure(self, high, low, close, volume):
        """Calculate buying pressure using price-volume analysis"""
        # Accumulation/Distribution Line variant
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        
        money_flow_volume = money_flow_multiplier * volume
        return money_flow_volume.rolling(10).sum()
    
    def _calculate_selling_pressure(self, high, low, close, volume):
        """Calculate selling pressure using price-volume analysis"""
        # Inverse of buying pressure
        money_flow_multiplier = ((high - close) - (close - low)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        
        money_flow_volume = money_flow_multiplier * volume
        return money_flow_volume.rolling(10).sum()
    
    def _detect_fractal_support(self, low, window=5):
        """Detect fractal support levels"""
        fractals = pd.Series(0, index=low.index)
        
        for i in range(window, len(low) - window):
            if low.iloc[i] == low.iloc[i-window:i+window+1].min():
                fractals.iloc[i] = 1
                
        return fractals
    
    def _detect_fractal_resistance(self, high, window=5):
        """Detect fractal resistance levels"""
        fractals = pd.Series(0, index=high.index)
        
        for i in range(window, len(high) - window):
            if high.iloc[i] == high.iloc[i-window:i+window+1].max():
                fractals.iloc[i] = 1
                
        return fractals
    
    def _calculate_volatility_clustering(self, close):
        """Calculate volatility clustering score"""
        returns = close.pct_change()
        squared_returns = returns ** 2
        
        # GARCH-like volatility clustering
        volatility_cluster = squared_returns.rolling(10).mean()
        vol_cluster_ma = volatility_cluster.rolling(50).mean()
        
        cluster_score = volatility_cluster / vol_cluster_ma
        return cluster_score

def demonstrate_advanced_features():
    """Demonstrate the advanced feature engineering approach"""
    
    print("🧠 Advanced Feature Engineering for Regime Classification")
    print("=" * 60)
    
    print("\n✨ New Feature Categories:")
    print("1. Volume-Price Divergence Analysis")
    print("2. Volatility Regime Change Detection") 
    print("3. Support/Resistance Breakout Strength")
    print("4. Multi-timeframe Momentum Convergence")
    print("5. Order Flow Proxy Indicators")
    print("6. Fractal-based Support/Resistance")
    print("7. Volatility Clustering Scores")
    
    print("\n🎯 Expected Improvements:")
    print("- Better breakout detection through volume-price analysis")
    print("- Improved precision via fractal support/resistance")
    print("- Enhanced volatility regime classification")
    print("- Multi-timeframe alignment features")
    
    print("\n📈 Implementation Benefits:")
    print("- Domain expertise encoded in features")
    print("- Addresses specific weaknesses in breakout detection")
    print("- Maintains interpretability")
    print("- Can be combined with existing features")

if __name__ == "__main__":
    demonstrate_advanced_features()
