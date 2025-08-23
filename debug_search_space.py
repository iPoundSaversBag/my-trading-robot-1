#!/usr/bin/env python3

import json
from core.backtest import IchimokuBacktester

try:
    backtester = IchimokuBacktester('core/optimization_config.json', debug_mode=True)
    print('Search space parameters:')
    for dim in backtester.search_space:
        low = getattr(dim, 'low', 'N/A')
        high = getattr(dim, 'high', 'N/A')
        print(f'  {dim.name}: {low} - {high} ({type(dim).__name__})')
    print(f'Total parameters: {len(backtester.search_space)}')
    
    # Test a simple objective function call
    print("\nTesting parameter generation...")
    from skopt.space import Integer, Real, Categorical
    test_params = {}
    for dim in backtester.search_space:
        if isinstance(dim, Integer):
            test_params[dim.name] = dim.low
        elif isinstance(dim, Real):
            test_params[dim.name] = dim.low
        elif isinstance(dim, Categorical):
            test_params[dim.name] = dim.categories[0]
    
    print("Test parameters:", test_params)
    print("Parameter validation:", backtester.are_params_valid(test_params))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
