#!/usr/bin/env python3
"""Unified timeframe calibration script.

Replaces: recalibrate_trading_timeframes.py, recalibrate_5m_timeframe.py
Generates / updates consolidated thresholds inside core/optimization_config.json
via atomic_update_master_config metadata.
"""
import json, os, datetime
import pandas as pd
from pathlib import Path
from utilities.utils import load_master_config, atomic_update_master_config

TIMEFRAMES = ["5m", "15m"]  # extendable

def analyze(tf: str):
    path = f"data/crypto_data_{tf}.parquet"
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['price_change_5'] = df['close'].pct_change(5)
    df = df.dropna()
    if df.empty:
        return None
    return {
        'timeframe': tf,
        'periods': len(df),
        'date_range': {'start': str(df.index.min()), 'end': str(df.index.max())},
        'volatility': {
            'low': df['volatility'].quantile(0.33),
            'medium': df['volatility'].quantile(0.67),
            'high': df['volatility'].quantile(0.85),
            'extreme': df['volatility'].quantile(0.95)
        },
        'volume_ratio': {
            'low': df['volume_ratio'].quantile(0.25),
            'high': df['volume_ratio'].quantile(0.75),
            'extreme': df['volume_ratio'].quantile(0.90)
        },
        'momentum_abs_5': {
            'weak': df['price_change_5'].abs().quantile(0.33),
            'medium': df['price_change_5'].abs().quantile(0.67),
            'strong': df['price_change_5'].abs().quantile(0.85)
        }
    }

def build_thresholds(analyses):
    th = {}
    for a in analyses:
        if not a: continue
        tf = a['timeframe']
        th[tf] = {
            'volatility_threshold': a['volatility']['medium'],
            'high_volatility_threshold': a['volatility']['high'],
            'extreme_volatility_threshold': a['volatility']['extreme'],
            'volume_threshold': a['volume_ratio']['high'],
            'momentum_threshold': a['momentum_abs_5']['medium'],
            'strong_momentum_threshold': a['momentum_abs_5']['strong'],
            'trend_confirmation_periods': 3 if tf == '5m' else 2,
            'min_confidence': 0.6 if tf == '5m' else 0.65
        }
    return th

def main():
    analyses = [analyze(tf) for tf in TIMEFRAMES]
    valid = [a for a in analyses if a]
    if not valid:
        print("No timeframe data available; aborting calibration.")
        return False
    thresholds = build_thresholds(valid)
    master = load_master_config()
    master.setdefault('regime_detection', {})['calibrated_timeframes'] = list(thresholds.keys())
    master.setdefault('legacy_timeframe_calibration', {})['unified'] = thresholds
    master['regime_detection']['last_calibrated'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
    atomic_update_master_config(master, 'core/optimization_config.json', reason='timeframe_recalibration')
    print("Calibration complete for:", ', '.join(thresholds.keys()))
    for tf, vals in thresholds.items():
        print(f" {tf}: vol={vals['volatility_threshold']:.6f} high={vals['high_volatility_threshold']:.6f} mom={vals['momentum_threshold']:.6f}")
    return True

if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
