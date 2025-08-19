#!/usr/bin/env python3
"""
Dynamic Parameter Loader - Syncs live bot with latest backtest optimization
"""
import json
import os
from datetime import datetime

def load_latest_backtest_params():
    """Load optimized parameters from latest backtest run"""
    try:
        # Read latest run directory
        latest_run_file = "plots_output/latest_run_dir.txt"
        if not os.path.exists(latest_run_file):
            print("❌ No latest run directory found")
            return None
            
        with open(latest_run_file, 'r') as f:
            latest_dir = f.read().strip()
        
        # Load final config from latest run
        config_path = f"{latest_dir}/final_config.json"
        if not os.path.exists(config_path):
            print(f"❌ No final_config.json found in {latest_dir}")
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract optimized parameters
        optimized_params = {
            "INITIAL_CAPITAL": config["fixed_parameters"]["INITIAL_CAPITAL"],
            "POSITION_SIZE": config["fixed_parameters"]["POSITION_SIZE"],
            "COMMISSION_RATE": config["fixed_parameters"]["COMMISSION_RATE"],
            "SLIPPAGE_RATE": config["fixed_parameters"]["SLIPPAGE_RATE"],
            "MAX_PORTFOLIO_RISK": config["fixed_parameters"]["MAX_PORTFOLIO_RISK"],
            "min_confidence_for_trade": config["fixed_parameters"]["min_confidence_for_trade"],
            "USE_ML_REGIME_DETECTION": config["fixed_parameters"]["USE_ML_REGIME_DETECTION"],
            "BLOCK_LOW_CONFIDENCE_SIGNALS": config["fixed_parameters"]["BLOCK_LOW_CONFIDENCE_SIGNALS"],
            "VOLUME_CONFIRMATION": config.get("VOLUME_CONFIRMATION", True),
            "SYMBOL": "BTCUSDT",
            "TIMEFRAME": "5m"
        }
        
        # Add optimized trading parameters from best_parameters_so_far
        if "best_parameters_so_far" in config:
            best_params = config["best_parameters_so_far"]
            optimized_params.update({
                "RSI_PERIOD": best_params.get("RSI_PERIOD", 14),
                "RSI_OVERBOUGHT": best_params.get("RSI_OVERBOUGHT", 70),
                "RSI_OVERSOLD": best_params.get("RSI_OVERSOLD", 30),
                "TENKAN_SEN_PERIOD": best_params.get("TENKAN_SEN_PERIOD", 9),
                "KIJUN_SEN_PERIOD": best_params.get("KIJUN_SEN_PERIOD", 26),
                "SENKOU_SPAN_B_PERIOD": best_params.get("SENKOU_SPAN_B_PERIOD", 52),
                "ADX_THRESHOLD": best_params.get("ADX_THRESHOLD", 25),
                "ATR_PERIOD": best_params.get("ATR_PERIOD", 14),
                "volatility_threshold": best_params.get("volatility_threshold", 0.02),
                "volatility_window": best_params.get("volatility_window", 20),
                "trend_window": best_params.get("trend_window", 50),
                "momentum_window": best_params.get("momentum_window", 14)
            })
        
        # Add derived MA parameters (commonly used alternatives)
        optimized_params.update({
            "MA_FAST": optimized_params.get("TENKAN_SEN_PERIOD", 12),
            "MA_SLOW": optimized_params.get("KIJUN_SEN_PERIOD", 26),
            "MA_SIGNAL": 9  # Standard MACD signal
        })
        
        print(f"✅ Loaded optimized parameters from: {latest_dir}")
        return optimized_params
        
    except Exception as e:
        print(f"❌ Error loading backtest parameters: {e}")
        return None

def get_default_params():
    """Fallback default parameters"""
    return {
        "INITIAL_CAPITAL": 10000,
        "POSITION_SIZE": 0.02,
        "COMMISSION_RATE": 0.001,
        "SLIPPAGE_RATE": 0.0001,
        "MAX_PORTFOLIO_RISK": 0.15,
        "min_confidence_for_trade": 0.04,
        "USE_ML_REGIME_DETECTION": True,
        "BLOCK_LOW_CONFIDENCE_SIGNALS": True,
        "VOLUME_CONFIRMATION": True,
        "SYMBOL": "BTCUSDT",
        "TIMEFRAME": "5m",
        "RSI_PERIOD": 14,
        "RSI_OVERBOUGHT": 70,
        "RSI_OVERSOLD": 30,
        "MA_FAST": 12,
        "MA_SLOW": 26,
        "MA_SIGNAL": 9
    }

def get_trading_config():
    """Get optimized trading configuration with fallback"""
    optimized = load_latest_backtest_params()
    if optimized:
        return optimized
    else:
        print("⚠️ Using default parameters - no optimized params found")
        return get_default_params()

def save_live_bot_config():
    """Save optimized config for live bot to use"""
    config = get_trading_config()
    
    # Save to a file that live bot can read
    config_file = "api/live_trading_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved live trading config to: {config_file}")
    return config

if __name__ == "__main__":
    print("🔧 Loading Latest Backtest Parameters...")
    print("="*50)
    
    config = save_live_bot_config()
    
    print(f"\n📊 Trading Configuration:")
    print(f"   Position Size: {config['POSITION_SIZE']*100}%")
    print(f"   Min Confidence: {config['min_confidence_for_trade']}")
    print(f"   RSI Period: {config.get('RSI_PERIOD', 'N/A')}")
    print(f"   RSI Levels: {config.get('RSI_OVERSOLD', 'N/A')}/{config.get('RSI_OVERBOUGHT', 'N/A')}")
    print(f"   MA Fast/Slow: {config.get('MA_FAST', 'N/A')}/{config.get('MA_SLOW', 'N/A')}")
    print(f"   Symbol: {config['SYMBOL']}")
    print(f"   Timeframe: {config['TIMEFRAME']}")
    print(f"   ATR Period: {config.get('ATR_PERIOD', 'N/A')}")
    print(f"   Volatility Threshold: {config.get('volatility_threshold', 'N/A')}")
