"""Objective Diagnostic Tool

Purpose:
  Lightweight script to quickly exercise the optimization objective
  (regime-aware + global fallback) for a single training window.
Usage:
  python tests/objective_diagnostic_tool.py

Notes:
  - Keeps trials minimal for speed.
  - Prints whether regime fallback engaged and resulting value.
"""
import json, os, sys
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.backtest import IchimokuBacktester, compute_walk_forward_windows

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'core', 'optimization_config.json')
with open(CONFIG_PATH,'r') as f:
    config = json.load(f)

# Force minimal calls for speed
intensity = str(config['optimization_settings']['intensity'])
config['optimization_settings']['calls_per_window'][intensity] = 3

parquet_path = os.path.join(PROJECT_ROOT, 'data', 'crypto_data_15m.parquet')
df = pd.read_parquet(parquet_path)
windows = compute_walk_forward_windows(config, df)
if not windows:
    print('No windows generated.')
    sys.exit(0)
train_start, train_end, test_end = windows[0]
train_df = df.loc[train_start:train_end]

bt = IchimokuBacktester(CONFIG_PATH, debug_mode=True, no_cache=True)

class DummyTrial:
    def __init__(self, number=0):
        self.number = number
    def suggest_int(self, name, low, high):
        return low
    def suggest_float(self, name, low, high):
        return low
    def suggest_categorical(self, name, categories):
        return categories[0]

trial = DummyTrial()
try:
    value = bt.objective_optuna(trial, train_df)
    print(f"Objective value: {value}")
except Exception as e:
    print(f"Objective raised: {e}")
print('Done.')
