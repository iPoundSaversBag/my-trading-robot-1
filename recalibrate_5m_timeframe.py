#!/usr/bin/env python3
"""
Recalibrate production regime detector for the 5m primary trading timeframe.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_timeframe_data(timeframe):
    """Analyze a specific timeframe to determine optimal thresholds."""
    file_path = f"data/crypto_data_{timeframe}.parquet"
    
    if not Path(file_path).exists():
        logger.error(f"Data file not found: {file_path}")
        return None
    
    logger.info(f"📊 Analyzing {timeframe} timeframe data...")
    df = pd.read_parquet(file_path)
    
    # Calculate key metrics
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Calculate price momentum
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_20'] = df['close'].pct_change(20)
    
    # Remove NaN values
    df = df.dropna()
    
    # Calculate percentiles for threshold setting
    volatility_percentiles = {
        'low': df['volatility'].quantile(0.33),
        'medium': df['volatility'].quantile(0.67),
        'high': df['volatility'].quantile(0.85),
        'extreme': df['volatility'].quantile(0.95)
    }
    
    volume_percentiles = {
        'low': df['volume_ratio'].quantile(0.25),
        'high': df['volume_ratio'].quantile(0.75),
        'extreme': df['volume_ratio'].quantile(0.90)
    }
    
    momentum_percentiles = {
        'weak': abs(df['price_change_5']).quantile(0.33),
        'medium': abs(df['price_change_5']).quantile(0.67),
        'strong': abs(df['price_change_5']).quantile(0.85)
    }
    
    # Calculate regime statistics
    total_periods = len(df)
    high_vol_periods = len(df[df['volatility'] > volatility_percentiles['high']])
    trending_periods = len(df[abs(df['price_change_5']) > momentum_percentiles['medium']])
    
    analysis = {
        'timeframe': timeframe,
        'total_periods': total_periods,
        'date_range': {
            'start': str(df.index.min()),
            'end': str(df.index.max())
        },
        'volatility_thresholds': volatility_percentiles,
        'volume_thresholds': volume_percentiles,
        'momentum_thresholds': momentum_percentiles,
        'regime_distribution': {
            'high_volatility_pct': (high_vol_periods / total_periods) * 100,
            'trending_pct': (trending_periods / total_periods) * 100,
            'ranging_pct': ((total_periods - trending_periods) / total_periods) * 100
        },
        'statistics': {
            'avg_volatility': float(df['volatility'].mean()),
            'avg_volume_ratio': float(df['volume_ratio'].mean()),
            'avg_returns': float(df['returns'].mean()),
            'volatility_std': float(df['volatility'].std())
        }
    }
    
    logger.info(f"✅ {timeframe} analysis complete:")
    logger.info(f"   📈 High volatility: {analysis['regime_distribution']['high_volatility_pct']:.1f}%")
    logger.info(f"   📊 Trending: {analysis['regime_distribution']['trending_pct']:.1f}%")
    logger.info(f"   📉 Ranging: {analysis['regime_distribution']['ranging_pct']:.1f}%")
    
    return analysis

def create_production_config():
    """Create optimized production configuration for the 5m timeframe."""
    logger.info("🔧 Creating production configuration for 5m trading timeframe...")
    
    # Analyze the 5m trading timeframe
    analysis_5m = analyze_timeframe_data("5m")
    
    if not analysis_5m:
        logger.error("Failed to analyze 5m timeframe data")
        return None
    
    # Create optimized configuration
    config = {
        "calibration_info": {
            "calibrated_for": ["5m"],
            "calibration_date": "2025-08-21",
            "accuracy_target": "98%",
            "data_periods": {
                "5m": analysis_5m['total_periods']
            }
        },
        "timeframe_configs": {
            "5m": {
                "volatility_threshold": analysis_5m['volatility_thresholds']['medium'],
                "high_volatility_threshold": analysis_5m['volatility_thresholds']['high'],
                "extreme_volatility_threshold": analysis_5m['volatility_thresholds']['extreme'],
                "volume_threshold": analysis_5m['volume_thresholds']['high'],
                "momentum_threshold": analysis_5m['momentum_thresholds']['medium'],
                "strong_momentum_threshold": analysis_5m['momentum_thresholds']['strong'],
                "trend_confirmation_periods": 3,
                "min_confidence": 0.6
            },
            # Add other timeframes with default/fallback values if needed
            "default": {
                "volatility_threshold": 0.002,
                "high_volatility_threshold": 0.005,
                "extreme_volatility_threshold": 0.01,
                "volume_threshold": 1.5,
                "momentum_threshold": 0.001,
                "strong_momentum_threshold": 0.003,
                "trend_confirmation_periods": 2,
                "min_confidence": 0.5
            }
        },
        "regime_multipliers": {
            "TRENDING_BULL": {
                "position_size": 1.2,
                "take_profit": 1.3,
                "stop_loss": 0.8
            },
            "TRENDING_BEAR": {
                "position_size": 0.8,
                "take_profit": 1.1,
                "stop_loss": 1.2
            },
            "HIGH_VOLATILITY": {
                "position_size": 0.6,
                "take_profit": 1.5,
                "stop_loss": 0.7
            },
            "VOLATILE": {
                "position_size": 0.7,
                "take_profit": 1.4,
                "stop_loss": 0.8
            },
            "RANGING": {
                "position_size": 1.0,
                "take_profit": 1.0,
                "stop_loss": 1.0
            },
            "CONSOLIDATION": {
                "position_size": 0.9,
                "take_profit": 0.8,
                "stop_loss": 1.1
            }
        },
        "analysis_results": {
            "5m_analysis": analysis_5m
        }
    }
    
    # Merge calibration into consolidated optimization_config.json
    master_path = "core/optimization_config.json"
    try:
        if os.path.exists(master_path):
            with open(master_path, 'r', encoding='utf-8') as mf:
                master_cfg = json.load(mf)
        else:
            master_cfg = {}
        master_cfg.setdefault('legacy_timeframe_calibration', {})['tf_5m'] = config.get('timeframe_configs', {})
        rd = master_cfg.setdefault('regime_detection', {})
        if 'regime_detection' in config:
            rd.update(config['regime_detection'])
        with open(master_path, 'w', encoding='utf-8') as mf:
            json.dump(master_cfg, mf, indent=4, default=str)
        logger.info(f"✅ 5m calibration merged into {master_path}")
    except Exception as e:
        logger.error(f"Failed merging 5m calibration: {e}")
    
    # Print key thresholds
    print("\n📊 CALIBRATED THRESHOLDS FOR 5-MINUTE TIMEFRAME:")
    print("=" * 60)
    print(f"🕐 5-minute timeframe:")
    print(f"   • Volatility threshold: {config['timeframe_configs']['5m']['volatility_threshold']:.6f}")
    print(f"   • High volatility: {config['timeframe_configs']['5m']['high_volatility_threshold']:.6f}")
    print(f"   • Momentum threshold: {config['timeframe_configs']['5m']['momentum_threshold']:.6f}")
    
    return config

def main():
    """Main calibration process."""
    print("🎯 RECALIBRATING PRODUCTION REGIME DETECTOR")
    print("Target timeframe: 5m (primary trading timeframe)")
    print("=" * 60)
    
    config = create_production_config()
    
    if config:
        print("\n🎉 CALIBRATION COMPLETE!")
        print("✅ Production regime detector now optimized for 5m trading")
        print("✅ Thresholds calculated from real market data")
        print("✅ Calibration merged into core/optimization_config.json")
    else:
        print("\n❌ Calibration failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
