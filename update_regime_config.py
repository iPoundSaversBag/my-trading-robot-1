#!/usr/bin/env python3

import json

# Load existing config
with open('api/live_trading_config.json', 'r') as f:
    config = json.load(f)

# Load regime-specific parameters from backtest
with open('plots_output/20250817_133240/optimized_params_per_window.json', 'r') as f:
    all_params = json.load(f)

# Get the latest window parameters
latest_window = max(all_params.keys(), key=lambda x: int(x.split('_')[1]))
latest_params = all_params[latest_window]

# Add regime-specific filter settings
regime_filters = {}
for key, value in latest_params.items():
    if key.startswith('USE_') and '_FILTER_' in key:
        regime_filters[key] = value

# Add volatility and regime thresholds
config.update({
    'volatility_threshold': latest_params.get('volatility_threshold', 0.03),
    'volume_threshold_multiplier': latest_params.get('volume_threshold_multiplier', 2.0),
    'ADX_THRESHOLD': 25,
    **regime_filters
})

# Save updated config
with open('api/live_trading_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Updated live trading config with regime-specific parameters!')
print(f'Added {len(regime_filters)} regime filter settings')
print(f'Volatility threshold: {config.get("volatility_threshold", "N/A")}')
print(f'Volume threshold: {config.get("volume_threshold_multiplier", "N/A")}')
