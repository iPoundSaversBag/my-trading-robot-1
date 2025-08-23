#!/usr/bin/env python3

import time
import json
from core.backtest import IchimokuBacktester

try:
    print("Initializing backtester...")
    start_time = time.time()
    backtester = IchimokuBacktester('core/optimization_config.json', debug_mode=True)
    print(f"Initialization took: {time.time() - start_time:.2f}s")
    
    # Test parameter generation
    test_params = {
        'TENKAN_SEN_PERIOD': 9, 
        'KIJUN_SEN_PERIOD': 26, 
        'SENKOU_SPAN_B_PERIOD': 52,
        'RSI_PERIOD': 14, 
        'RSI_OVERBOUGHT': 70.0, 
        'RSI_OVERSOLD': 30.0, 
        'ADX_PERIOD': 14, 
        'ADX_THRESHOLD': 25.0, 
        'BBANDS_PERIOD': 20, 
        'BBANDS_STD': 2.0,
        'STOP_LOSS_MULTIPLIER': 2.0, 
        'TAKE_PROFIT_MULTIPLIER': 3.0, 
        'POSITION_SIZE_PCT': 10.0
    }
    
    print("Testing parameter validation...")
    start_time = time.time()
    is_valid = backtester.are_params_valid(test_params)
    print(f"Parameter validation took: {time.time() - start_time:.2f}s, result: {is_valid}")
    
    # Test signal generation on small sample
    print("Testing signal generation...")
    start_time = time.time()
    
    # Get sample data for testing
    df = backtester.data_processor.get_data()
    print(f"Full data shape: {df.shape}")
    
    # Get a small sample of data
    windows = backtester.get_walk_forward_windows(df)
    if windows:
        train_start, train_end, test_start, test_end = windows[0]
        print(f"Using window: {train_start} to {train_end}")
        
        # Get training data
        train_df = backtester.data_processor.get_data_for_period(train_start, train_end)
        print(f"Training data shape: {train_df.shape}")
        print(f"Data loading took: {time.time() - start_time:.2f}s")
        
        # Test signal generation on first 1000 rows
        sample_df = train_df.head(1000).copy()
        signal_start = time.time()
        
        try:
            processed_df = backtester.cached_generate_signals(
                backtester.persistent_strategy, 
                sample_df, 
                backtester.realism_settings
            )
            print(f"Signal generation on 1000 rows took: {time.time() - signal_start:.2f}s")
            print(f"Processed data shape: {processed_df.shape}")
        except Exception as signal_error:
            print(f"Signal generation error: {signal_error}")
            import traceback
            traceback.print_exc()
    else:
        print("No windows generated")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
