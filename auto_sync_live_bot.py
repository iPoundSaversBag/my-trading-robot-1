#!/usr/bin/env python3
"""
Auto-sync script to keep live bot parameters synchronized with latest backtest results
"""

import json
import os
from datetime import datetime

def auto_sync_parameters():
    """Automatically sync parameters from backtest to live bot"""
    print(f"Auto-sync started at {datetime.now()}")
    
    try:
        # Load latest backtest results
        with open("core/optimization_config.json", 'r') as f:
            backtest_config = json.load(f)
        
        # Load current live config
        with open("api/live_trading_config.json", 'r') as f:
            live_config = json.load(f)
        
        # Update with latest optimized parameters
        if "best_parameters_so_far" in backtest_config:
            best_params = backtest_config["best_parameters_so_far"]
            
            # Core parameters
            updates = {
                "RSI_PERIOD": best_params.get("RSI_PERIOD"),
                "RSI_OVERBOUGHT": best_params.get("RSI_OVERBOUGHT"),
                "RSI_OVERSOLD": best_params.get("RSI_OVERSOLD"),
                "MA_FAST": best_params.get("TENKAN_SEN_PERIOD"),
                "MA_SLOW": best_params.get("KIJUN_SEN_PERIOD"),
                "ATR_PERIOD": best_params.get("ATR_PERIOD"),
                "ADX_THRESHOLD": best_params.get("ADX_THRESHOLD"),
                "volatility_threshold": best_params.get("volatility_threshold"),
                "last_sync": datetime.now().isoformat()
            }
            
            # Only update non-None values
            for key, value in updates.items():
                if value is not None:
                    live_config[key] = value
            
            # Save updated config
            with open("api/live_trading_config.json", 'w') as f:
                json.dump(live_config, f, indent=2)
            
            print("Parameters synchronized successfully")
            return True
            
    except Exception as e:
        print(f"Sync failed: {e}")
        return False

if __name__ == "__main__":
    auto_sync_parameters()
