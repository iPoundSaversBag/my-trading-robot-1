#!/usr/bin/env python3

import json
import os

# Load the latest optimized parameters
with open('plots_output/20250817_133240/optimized_params_per_window.json', 'r') as f:
    all_params = json.load(f)

# Get the latest window (highest number)
latest_window = max(all_params.keys(), key=lambda x: int(x.split('_')[1]))
latest_params = all_params[latest_window]

print(f'Latest window: {latest_window}')
print(f'TENKAN_SEN_PERIOD: {latest_params.get("TENKAN_SEN_PERIOD", "N/A")}')
print(f'KIJUN_SEN_PERIOD: {latest_params.get("KIJUN_SEN_PERIOD", "N/A")}')
print(f'RSI_PERIOD: {latest_params.get("RSI_PERIOD", "N/A")}')
print(f'RSI_OVERBOUGHT: {latest_params.get("RSI_OVERBOUGHT", "N/A")}')
print(f'RSI_OVERSOLD: {latest_params.get("RSI_OVERSOLD", "N/A")}')

# Create the live trading config in the format the live bot expects
live_config = {
    "SYMBOL": "BTCUSDT",
    "TIMEFRAME": "5m",
    "INITIAL_CAPITAL": 10000,
    "POSITION_SIZE": 0.02,
    "COMMISSION_RATE": 0.001,
    "SLIPPAGE_RATE": 0.0001,
    "MAX_PORTFOLIO_RISK": 0.15,
    "MAX_CONSECUTIVE_LOSSES": 5,
    "RISK_REDUCTION_FACTOR": 0.8,
    "MIN_PROFIT_FACTOR_FOR_RISK_INCREASE": 1.2,
    "min_confidence_for_trade": latest_params.get("min_confidence", 0.04),
    "USE_ML_REGIME_DETECTION": latest_params.get("USE_ML_REGIME_DETECTION", True),
    "BLOCK_LOW_CONFIDENCE_SIGNALS": True,
    "VOLUME_CONFIRMATION": True,
    # Map optimized parameters to live bot format
    "RSI_PERIOD": latest_params.get("RSI_PERIOD", 14),
    "RSI_OVERBOUGHT": latest_params.get("RSI_OVERBOUGHT", 70),
    "RSI_OVERSOLD": latest_params.get("RSI_OVERSOLD", 30),
    "MA_FAST": latest_params.get("TENKAN_SEN_PERIOD", 12),
    "MA_SLOW": latest_params.get("KIJUN_SEN_PERIOD", 26),
    "MA_SIGNAL": 9,
    "ADX_PERIOD": latest_params.get("ADX_PERIOD", 14),
    "ATR_PERIOD": latest_params.get("ATR_PERIOD", 14),
    "STOP_LOSS_MULTIPLIER": latest_params.get("STOP_LOSS_MULTIPLIER", 2.0),
    "TAKE_PROFIT_MULTIPLIER": latest_params.get("TAKE_PROFIT_MULTIPLIER", 3.0),
    "TRAILING_STOP_MULTIPLIER": latest_params.get("TRAILING_STOP_MULTIPLIER", 1.5),
    # Optimization metadata
    "source_window": latest_window,
    "optimization_timestamp": "20250817_133240",
    "last_updated": "2025-08-18"
}

# Save to api directory
os.makedirs('api', exist_ok=True)
with open('api/live_trading_config.json', 'w') as f:
    json.dump(live_config, f, indent=2)

print('\nCreated api/live_trading_config.json with optimized parameters!')
print(f'RSI_PERIOD: {live_config["RSI_PERIOD"]}')
print(f'MA_FAST: {live_config["MA_FAST"]}')
print(f'MA_SLOW: {live_config["MA_SLOW"]}')
