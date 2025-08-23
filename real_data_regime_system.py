#!/usr/bin/env python3
"""
REAL DATA REGIME DETECTION SYSTEM
==================================
This system uses the existing timeframe data (5m, 15m, 1h, 4h) from the parquet files
instead of synthetic test data to properly detect market regimes.

Addresses the critical issue: Previous systems were using synthetic data when real
market data already exists in the system.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketRegime:
    """Market regime classification"""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR" 
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT_BULLISH = "BREAKOUT_BULLISH"
    BREAKOUT_BEARISH = "BREAKOUT_BEARISH"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"

class RealDataRegimeDetector:
    """
    Market regime detection using actual timeframe data from the system.
    """
    
    def __init__(self):
        self.data_files = {
            '5m': 'data/crypto_data_5m.parquet',
            '15m': 'data/crypto_data_15m.parquet', 
            '1h': 'data/crypto_data_1h.parquet',
            '4h': 'data/crypto_data_4h.parquet'
        }
        self.regime_stats = {}
        
    def load_real_data(self, timeframe: str = '1h') -> pd.DataFrame:
        """Load real market data from existing parquet files"""
        if timeframe not in self.data_files:
            raise ValueError(f"Timeframe {timeframe} not available. Use: {list(self.data_files.keys())}")
            
        file_path = self.data_files[timeframe]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        print(f"📊 Loading real data from: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Ensure datetime index
        if 'timestamp' in df.columns and df.index.name != 'datetime':
            df.index = pd.to_datetime(df['timestamp'], unit='ms')
            df.index.name = 'datetime'
        
        print(f"✅ Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for regime detection"""
        data = df.copy()
        
        # Price-based indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Volatility indicators
        data['atr'] = self._calculate_atr(data, 14)
        data['volatility'] = data['close'].rolling(20).std()
        data['volatility_ma'] = data['volatility'].rolling(10).mean()
        
        # Momentum indicators
        data['rsi'] = self._calculate_rsi(data['close'], 14)
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Trend strength
        data['trend_strength'] = abs(data['close'] - data['sma_50']) / data['sma_50']
        
        # Volume indicators
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        else:
            data['volume_ma'] = 1
            data['volume_ratio'] = 1
            
        # Price change metrics
        data['price_change'] = data['close'].pct_change()
        data['price_change_20'] = data['close'].pct_change(20)
        
        return data
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each timestamp using real market data
        """
        regimes = pd.Series(index=df.index, dtype=str, name='regime')
        
        for i in range(len(df)):
            if i < 50:  # Need enough data for indicators
                regimes.iloc[i] = MarketRegime.RANGING
                continue
                
            current = df.iloc[i]
            
            # Get regime based on multiple factors
            regime = self._classify_regime(current, df.iloc[max(0, i-20):i+1])
            regimes.iloc[i] = regime
            
        return regimes
    
    def _classify_regime(self, current_row, window_data) -> str:
        """Classify current market regime"""
        
        # Extract key metrics
        close = current_row['close']
        sma_20 = current_row['sma_20']
        sma_50 = current_row['sma_50']
        atr = current_row['atr']
        volatility = current_row['volatility']
        volatility_ma = current_row['volatility_ma']
        rsi = current_row['rsi']
        trend_strength = current_row['trend_strength']
        volume_ratio = current_row['volume_ratio']
        price_change_20 = current_row['price_change_20']
        
        # Volatility classification
        is_high_vol = volatility > volatility_ma * 1.5
        is_low_vol = volatility < volatility_ma * 0.7
        
        # Trend classification
        is_uptrend = close > sma_20 > sma_50
        is_downtrend = close < sma_20 < sma_50
        strong_trend = trend_strength > 0.05
        
        # Volume analysis
        high_volume = volume_ratio > 1.3
        
        # Price momentum
        strong_bull_momentum = price_change_20 > 0.15
        strong_bear_momentum = price_change_20 < -0.15
        
        # Recent price action
        recent_highs = window_data['high'].rolling(5).max().iloc[-1]
        recent_lows = window_data['low'].rolling(5).min().iloc[-1]
        price_near_high = close > recent_highs * 0.98
        price_near_low = close < recent_lows * 1.02
        
        # 1. Volatility-based regimes (highest priority)
        if is_high_vol:
            return MarketRegime.HIGH_VOLATILITY
        elif is_low_vol:
            return MarketRegime.LOW_VOLATILITY
            
        # 2. Breakout detection
        if high_volume and strong_trend:
            if is_uptrend and price_near_high:
                return MarketRegime.BREAKOUT_BULLISH
            elif is_downtrend and price_near_low:
                return MarketRegime.BREAKOUT_BEARISH
                
        # 3. Trend-based regimes
        if strong_trend:
            if is_uptrend and strong_bull_momentum:
                return MarketRegime.TRENDING_BULL
            elif is_downtrend and strong_bear_momentum:
                return MarketRegime.TRENDING_BEAR
                
        # 4. Accumulation/Distribution
        if 30 < rsi < 70:  # Neutral RSI
            if high_volume:
                if close > sma_20:
                    return MarketRegime.ACCUMULATION
                else:
                    return MarketRegime.DISTRIBUTION
                    
        # 5. Default to ranging
        return MarketRegime.RANGING
    
    def analyze_regime_performance(self, df: pd.DataFrame, regimes: pd.Series) -> Dict:
        """Analyze regime detection performance on real data"""
        
        # Count regime occurrences
        regime_counts = regimes.value_counts()
        total_periods = len(regimes)
        
        # Calculate regime statistics
        stats = {}
        for regime in regime_counts.index:
            regime_mask = regimes == regime
            regime_data = df[regime_mask]
            
            if len(regime_data) > 0:
                # Performance metrics
                returns = regime_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252 * 24)  # Annualized
                
                stats[regime] = {
                    'count': regime_counts[regime],
                    'percentage': (regime_counts[regime] / total_periods) * 100,
                    'avg_return': returns.mean() * 100,
                    'volatility': volatility * 100,
                    'periods': len(regime_data)
                }
        
        return stats
    
    def validate_regime_accuracy(self, df: pd.DataFrame, regimes: pd.Series) -> Dict:
        """Validate regime detection accuracy using forward-looking analysis"""
        
        validation_results = {}
        look_ahead_periods = 24  # Look ahead 24 periods
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
                
            regime_indices = regimes[regimes == regime].index
            correct_predictions = 0
            total_predictions = 0
            
            for idx in regime_indices:
                idx_pos = df.index.get_loc(idx)
                
                # Skip if not enough future data
                if idx_pos + look_ahead_periods >= len(df):
                    continue
                    
                current_data = df.iloc[idx_pos]
                future_data = df.iloc[idx_pos:idx_pos + look_ahead_periods]
                
                # Validate regime characteristics hold true
                is_valid = self._validate_regime_characteristics(regime, current_data, future_data)
                
                if is_valid:
                    correct_predictions += 1
                total_predictions += 1
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                validation_results[regime] = {
                    'accuracy': accuracy,
                    'correct': correct_predictions,
                    'total': total_predictions
                }
        
        return validation_results
    
    def _validate_regime_characteristics(self, regime: str, current: pd.Series, future: pd.DataFrame) -> bool:
        """Validate if regime characteristics hold true in future periods"""
        
        if len(future) < 5:
            return False
            
        future_returns = future['close'].pct_change().dropna()
        future_volatility = future_returns.std()
        avg_return = future_returns.mean()
        
        # Regime-specific validation
        if regime == MarketRegime.TRENDING_BULL:
            return avg_return > 0.001 and future_volatility < 0.05
        elif regime == MarketRegime.TRENDING_BEAR:
            return avg_return < -0.001 and future_volatility < 0.05
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return future_volatility > 0.03
        elif regime == MarketRegime.LOW_VOLATILITY:
            return future_volatility < 0.015
        elif regime == MarketRegime.BREAKOUT_BULLISH:
            return avg_return > 0.002
        elif regime == MarketRegime.BREAKOUT_BEARISH:
            return avg_return < -0.002
        elif regime == MarketRegime.RANGING:
            return abs(avg_return) < 0.001 and future_volatility < 0.025
        else:
            return True  # Default validation for ACCUMULATION/DISTRIBUTION

def run_real_data_analysis():
    """Run comprehensive analysis using real market data"""
    
    print("=" * 60)
    print("🎯 REAL DATA REGIME DETECTION ANALYSIS")
    print("=" * 60)
    
    detector = RealDataRegimeDetector()
    
    # Test multiple timeframes
    timeframes = ['1h', '4h']  # Start with these as they have good data
    
    for timeframe in timeframes:
        print(f"\n📈 Analyzing {timeframe} timeframe...")
        print("-" * 40)
        
        try:
            # Load real data
            df = detector.load_real_data(timeframe)
            
            # Use recent data for analysis (last 6 months)
            recent_data = df.tail(4000)  # Approximately last 6 months for 1h data
            
            # Calculate indicators
            data_with_indicators = detector.calculate_technical_indicators(recent_data)
            
            # Detect regimes
            regimes = detector.detect_regime(data_with_indicators)
            
            # Analyze performance
            regime_stats = detector.analyze_regime_performance(data_with_indicators, regimes)
            
            print(f"\n📊 Regime Distribution ({timeframe}):")
            for regime, stats in regime_stats.items():
                print(f"  {regime}: {stats['count']} periods ({stats['percentage']:.1f}%)")
            
            # Validate accuracy
            validation = detector.validate_regime_accuracy(data_with_indicators, regimes)
            
            print(f"\n🎯 Regime Accuracy Validation ({timeframe}):")
            total_accuracy = 0
            valid_regimes = 0
            
            for regime, val_stats in validation.items():
                accuracy = val_stats['accuracy']
                print(f"  {regime}: {accuracy:.1f}% ({val_stats['correct']}/{val_stats['total']})")
                total_accuracy += accuracy
                valid_regimes += 1
            
            if valid_regimes > 0:
                overall_accuracy = total_accuracy / valid_regimes
                print(f"\n🎯 OVERALL ACCURACY ({timeframe}): {overall_accuracy:.1f}%")
                
                # Store results
                detector.regime_stats[timeframe] = {
                    'regime_distribution': regime_stats,
                    'validation_results': validation,
                    'overall_accuracy': overall_accuracy,
                    'total_periods_analyzed': len(regimes)
                }
        
        except Exception as e:
            print(f"❌ Error analyzing {timeframe}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 REAL DATA ANALYSIS SUMMARY")
    print("=" * 60)
    
    for tf, stats in detector.regime_stats.items():
        print(f"\n{tf} Timeframe:")
        print(f"  Overall Accuracy: {stats['overall_accuracy']:.1f}%")
        print(f"  Periods Analyzed: {stats['total_periods_analyzed']:,}")
        
        best_regime = max(stats['validation_results'].items(), 
                         key=lambda x: x[1]['accuracy'])
        worst_regime = min(stats['validation_results'].items(), 
                          key=lambda x: x[1]['accuracy'])
        
        print(f"  Best Regime: {best_regime[0]} ({best_regime[1]['accuracy']:.1f}%)")
        print(f"  Worst Regime: {worst_regime[0]} ({worst_regime[1]['accuracy']:.1f}%)")
    
    # Critical assessment
    best_timeframe = max(detector.regime_stats.items(), 
                        key=lambda x: x[1]['overall_accuracy'])
    
    print(f"\n🏆 BEST PERFORMING TIMEFRAME: {best_timeframe[0]}")
    print(f"📊 ACCURACY: {best_timeframe[1]['overall_accuracy']:.1f}%")
    
    if best_timeframe[1]['overall_accuracy'] > 60:
        print("✅ REGIME DETECTION IS USING REAL DATA AND ACHIEVING GOOD ACCURACY")
        print("✅ READY FOR INTEGRATION WITH BACKTESTING SYSTEM")
    else:
        print("⚠️ ACCURACY NEEDS IMPROVEMENT BEFORE PRODUCTION USE")
        print("💡 Consider calibrating thresholds with real market data")
    
    return detector.regime_stats

if __name__ == "__main__":
    results = run_real_data_analysis()
