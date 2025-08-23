#!/usr/bin/env python3
"""
Quick diagnostic to check walk-forward window generation issues
"""

import pandas as pd
import json
from datetime import timedelta
import sys
import os

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def compute_walk_forward_windows_debug(config: dict, df: pd.DataFrame):
    """
    Debug version of walk-forward window computation with extensive logging
    """
    print(f"🔍 DEBUG: Starting walk-forward window computation")
    print(f"📊 DataFrame info:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Index type: {type(df.index)}")
    print(f"  - Index name: {df.index.name}")
    print(f"  - Index min: {df.index.min()}")
    print(f"  - Index max: {df.index.max()}")
    print(f"  - Index dtype: {df.index.dtype}")
    print(f"  - First 3 index values: {df.index[:3].tolist()}")
    print(f"  - Last 3 index values: {df.index[-3:].tolist()}")
    
    windows = []
    try:
        if df is None or df.empty:
            print("❌ DataFrame is None or empty")
            return windows

        wfo_settings = (config or {}).get('walk_forward_settings', {})
        training_days = wfo_settings.get('training_days', 365)
        testing_days = wfo_settings.get('testing_days', 90)
        
        print(f"⚙️ Walk-forward settings:")
        print(f"  - Training days: {training_days}")
        print(f"  - Testing days: {testing_days}")

        start_date = df.index.min()
        end_date = df.index.max()
        
        print(f"📅 Date range:")
        print(f"  - Start: {start_date}")
        print(f"  - End: {end_date}")
        print(f"  - Total span: {end_date - start_date}")

        current_train_start = start_date
        window_count = 0
        while True:
            train_end = current_train_start + timedelta(days=training_days)
            test_end = train_end + timedelta(days=testing_days)
            
            print(f"🪟 Window {window_count + 1}:")
            print(f"  - Train start: {current_train_start}")
            print(f"  - Train end: {train_end}")
            print(f"  - Test end: {test_end}")
            print(f"  - Test end > end_date? {test_end > end_date}")
            
            if test_end > end_date:
                print(f"🛑 Breaking loop: test_end ({test_end}) > end_date ({end_date})")
                break
                
            windows.append((current_train_start, train_end, test_end))
            current_train_start += timedelta(days=testing_days)
            window_count += 1
            
            if window_count > 10:  # Safety limit for debugging
                print("🛑 Breaking loop: safety limit reached (10 windows)")
                break
        
        print(f"✅ Generated {len(windows)} windows")
        return windows
    except Exception as e:
        print(f"❌ Exception in walk-forward computation: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    print("🚀 Walk-Forward Window Generation Diagnostic")
    print("=" * 60)
    
    # Load configuration
    try:
        with open('core/optimization_config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Load data file (using 5m data as specified in config)
    data_file = config.get('data_settings', {}).get('file_path', 'data/crypto_data_5m.parquet')
    print(f"📂 Loading data from: {data_file}")
    
    try:
        df = pd.read_parquet(data_file)
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Ensure datetime index
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print("✅ Index is already datetime type")
    elif 'timestamp' in df.columns:
        print("🔄 Converting timestamp column to datetime index")
        # Check if timestamp is Unix timestamp in milliseconds
        if df['timestamp'].dtype in ['int64', 'float64'] and df['timestamp'].iloc[0] > 1e12:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        print("🔄 Converting index to datetime")
        df.index = pd.to_datetime(df.index)
    
    # Run walk-forward computation
    windows = compute_walk_forward_windows_debug(config, df)
    
    print("\n" + "=" * 60)
    print(f"📋 SUMMARY:")
    print(f"  - Total windows generated: {len(windows)}")
    
    if windows:
        print(f"  - First window: {windows[0]}")
        print(f"  - Last window: {windows[-1]}")
        print("✅ Walk-forward window generation successful!")
    else:
        print("❌ No windows generated - this explains the backtest failure!")

if __name__ == "__main__":
    main()
