import json, os, inspect, sys
from datetime import datetime

# Adjust path to import core modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.backtest import IchimokuBacktester, compute_walk_forward_windows
import pandas as pd

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'core', 'optimization_config.json')

with open(CONFIG_PATH,'r') as f:
    config = json.load(f)

# Force minimal trials for speed
config['optimization_settings']['calls_per_window'][str(config['optimization_settings']['intensity'])] = 5

# Load sample data
parquet_path = os.path.join(PROJECT_ROOT, 'data', 'crypto_data_15m.parquet')
df = pd.read_parquet(parquet_path)

windows = compute_walk_forward_windows(config, df)
if not windows:
    print('No windows generated.')
    sys.exit(0)
train_start, train_end, test_end = windows[0]
train_df = df.loc[train_start:train_end]

# Instantiate backtester in debug but do NOT run full pipeline
bt = IchimokuBacktester(CONFIG_PATH, debug_mode=True, no_cache=True)

# Monkey patch: disable regime optimization to compare
config_opt = config['optimization_settings']
regime_enabled = config_opt.get('regime_specific_optimization', False)
print(f"Regime optimization originally: {regime_enabled}")

print('Testing regime-aware objective directly (if enabled)...')

# Build a fake Optuna trial substitute with minimal interface for param suggestions
class DummyTrial:
    def __init__(self):
        self.number = 0
    def suggest_int(self, name, low, high):
        return low
    def suggest_float(self, name, low, high):
        return low
    def suggest_categorical(self, name, categories):
        return categories[0]

trial = DummyTrial()

try:
    score = bt.objective_optuna(trial, train_df)
    print(f"Objective returned value: {score}")
except Exception as e:
    print(f"Objective raised: {e}")

print('Done.')
