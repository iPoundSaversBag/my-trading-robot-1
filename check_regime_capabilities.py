#!/usr/bin/env python3

import json

# Check live bot regime capabilities
with open('api/live_trading_config.json', 'r') as f:
    config = json.load(f)

# Count regime-specific parameters
regime_params = {}
for key, value in config.items():
    if '_FILTER_' in key:
        regime = key.split('_FILTER_')[1]
        if regime not in regime_params:
            regime_params[regime] = []
        regime_params[regime].append(key.split('_FILTER_')[0])

print('=== LIVE BOT REGIME CAPABILITIES ===')
print(f'Total regime-specific parameters: {len([k for k in config.keys() if "_FILTER_" in k])}')
print(f'Regimes with specific filters: {len(regime_params)}')
print()

for regime, filters in sorted(regime_params.items()):
    print(f'{regime}:')
    for filter_type in sorted(set(filters)):
        enabled = config.get(f'{filter_type}_FILTER_{regime}', False)
        status = '✅' if enabled else '❌'
        print(f'  {status} {filter_type}')
    print()

# Check what regime detection the live bot can do
print('=== LIVE BOT REGIME DETECTION ===')
print('✅ trending_bull - ADX > 25 + MA_FAST > MA_SLOW + price > MA_SLOW')
print('✅ trending_bear - ADX > 25 + MA_FAST < MA_SLOW + price < MA_SLOW') 
print('✅ ranging - ADX <= 25')
print('✅ high_volatility - price_volatility > volatility_threshold')
print('✅ low_volatility - price_volatility <= volatility_threshold')
print('✅ breakout_bullish - volume_ratio > threshold + price_change > 1%')
print('✅ breakout_bearish - volume_ratio > threshold + price_change < -1%')
print()

print('=== REMOTE ACCESS CAPABILITIES ===')
print('❌ Live bot CANNOT directly access local backtest results')
print('❌ Live bot CANNOT remotely update local machine parameters')
print('✅ Live bot CAN read uploaded optimization parameters from git')
print('✅ Local backtest CAN sync parameters to Vercel via git commit/push')
print()

print('=== CURRENT DATA FLOW ===')
print('1. 🏠 Local: Backtest runs → generates optimized_params_per_window.json')
print('2. 🏠 Local: backtest.py writes → api/live_trading_config.json')
print('3. 🏠 Local: git commit + push → uploads to GitHub')
print('4. ☁️  Vercel: auto-deploys → live bot gets updated parameters')
print('5. ☁️  Vercel: live bot reads → regime-specific optimized parameters')
print()

print('=== MISSING CAPABILITIES FOR FULL REMOTE ACCESS ===')
print('❓ Remote backtest trigger: Live bot → trigger local backtest')
print('❓ Remote parameter sync: Live bot → update local files')
print('❓ Bidirectional sync: Live results → update local optimization')
