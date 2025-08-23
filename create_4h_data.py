#!/usr/bin/env python3
"""Create missing 4h data file by resampling from 1h data"""

import pandas as pd
import numpy as np

try:
    print("Loading 1h data...")
    df_1h = pd.read_parquet('data/crypto_data_1h.parquet')
    print(f"1h data shape: {df_1h.shape}")
    print(f"1h data columns: {df_1h.columns.tolist()}")
    print(f"1h data date range: {df_1h.index.min()} to {df_1h.index.max()}")
    
    # Resample to 4h
    print("Resampling to 4h...")
    df_4h = df_1h.resample('4H').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"4h data shape: {df_4h.shape}")
    print(f"4h data date range: {df_4h.index.min()} to {df_4h.index.max()}")
    
    # Save 4h data
    print("Saving 4h data...")
    df_4h.to_parquet('data/crypto_data_4h.parquet')
    print("✅ Created data/crypto_data_4h.parquet")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
