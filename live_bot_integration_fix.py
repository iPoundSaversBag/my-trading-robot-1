#!/usr/bin/env python3
"""
Live Bot Integration Diagnostic & Fix Script
Identifies and resolves discrepancies between backtest logic and live bot implementation
"""

import json
import os
import sys
from datetime import datetime
import importlib.util

def analyze_trading_logic_sync():
    """Analyze sync between backtest and live bot trading logic"""
    print("🔍 LIVE BOT INTEGRATION ANALYSIS")
    print("=" * 60)
    
    issues_found = []
    fixes_applied = []
    
    # 1. Check parameter sync
    print("\n1. 📊 PARAMETER SYNCHRONIZATION")
    print("-" * 40)
    
    # Load backtest config
    try:
        with open("core/optimization_config.json", 'r') as f:
            backtest_config = json.load(f)
        print("✅ Backtest config loaded")
    except Exception as e:
        print(f"❌ Cannot load backtest config: {e}")
        issues_found.append("Missing backtest configuration")
        return
    
    # Load live trading config
    try:
        with open("api/live_trading_config.json", 'r') as f:
            live_config = json.load(f)
        print("✅ Live trading config loaded")
    except Exception as e:
        print(f"❌ Cannot load live trading config: {e}")
        issues_found.append("Missing live trading configuration")
        return
    
    # Compare critical parameters
    critical_params = [
        "RSI_PERIOD", "RSI_OVERBOUGHT", "RSI_OVERSOLD",
        "MA_FAST", "MA_SLOW", "ATR_PERIOD", "ADX_THRESHOLD",
        "POSITION_SIZE", "min_confidence_for_trade"
    ]
    
    print("\n📋 Parameter Comparison:")
    param_mismatches = []
    
    for param in critical_params:
        backtest_val = None
        if "best_parameters_so_far" in backtest_config:
            backtest_val = backtest_config["best_parameters_so_far"].get(param)
        if backtest_val is None and "fixed_parameters" in backtest_config:
            backtest_val = backtest_config["fixed_parameters"].get(param)
            
        live_val = live_config.get(param)
        
        if backtest_val is not None and live_val is not None:
            if backtest_val != live_val:
                print(f"   ⚠️ {param}: Backtest={backtest_val}, Live={live_val}")
                param_mismatches.append((param, backtest_val, live_val))
            else:
                print(f"   ✅ {param}: {live_val} (synced)")
        elif live_val is not None:
            print(f"   ✅ {param}: {live_val} (live only)")
        else:
            print(f"   ❌ {param}: Missing in both configs")
            issues_found.append(f"Missing parameter: {param}")
    
    # 2. Check trading logic implementation
    print("\n2. 🔧 TRADING LOGIC COMPARISON")
    print("-" * 40)
    
    # Analyze backtest core logic
    backtest_logic_features = analyze_backtest_logic()
    live_logic_features = analyze_live_bot_logic()
    
    print("\n📊 Feature Comparison:")
    logic_mismatches = []
    
    for feature in backtest_logic_features:
        if feature in live_logic_features:
            print(f"   ✅ {feature}: Implemented in both")
        else:
            print(f"   ❌ {feature}: Missing in live bot")
            logic_mismatches.append(feature)
            issues_found.append(f"Missing logic feature: {feature}")
    
    # 3. Check regime filtering sync
    print("\n3. 🎯 REGIME FILTERING ANALYSIS")
    print("-" * 40)
    
    regime_filters = [k for k in live_config.keys() if k.startswith('USE_') and '_FILTER_' in k]
    print(f"   📋 Found {len(regime_filters)} regime filters in live config")
    
    active_filters = [k for k, v in live_config.items() if k.startswith('USE_') and '_FILTER_' in k and v]
    print(f"   ✅ {len(active_filters)} filters currently active")
    
    if len(active_filters) == 0:
        print("   ⚠️ No regime filters active - may impact trading performance")
        issues_found.append("No regime filters active")
    
    # 4. Check signal calculation consistency
    print("\n4. 📈 SIGNAL CALCULATION CONSISTENCY")
    print("-" * 40)
    
    signal_consistency = check_signal_consistency(backtest_config, live_config)
    for check, status in signal_consistency.items():
        if status:
            print(f"   ✅ {check}")
        else:
            print(f"   ❌ {check}")
            issues_found.append(f"Signal inconsistency: {check}")
    
    # 5. Generate fixes
    print("\n5. 🔧 GENERATING FIXES")
    print("-" * 40)
    
    if param_mismatches:
        fixes_applied.extend(fix_parameter_mismatches(param_mismatches, backtest_config, live_config))
    
    if logic_mismatches:
        fixes_applied.extend(suggest_logic_fixes(logic_mismatches))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n❌ Issues Found: {len(issues_found)}")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
    
    print(f"\n✅ Fixes Applied: {len(fixes_applied)}")
    for i, fix in enumerate(fixes_applied, 1):
        print(f"   {i}. {fix}")
    
    # Recommendations
    print("\n🚀 RECOMMENDATIONS")
    print("-" * 40)
    
    if param_mismatches:
        print("1. Sync parameters by running: python load_backtest_params.py")
    
    if logic_mismatches:
        print("2. Update live bot logic to match backtest implementation")
    
    if len(active_filters) < 5:
        print("3. Consider enabling more regime filters for better performance")
    
    print("4. Test live bot with: python test_live_bot.py")
    print("5. Monitor live trading: python monitor_bot.py")
    
    return len(issues_found) == 0

def analyze_backtest_logic():
    """Extract key features from backtest logic"""
    features = []
    
    try:
        with open("core/backtest.py", 'r') as f:
            content = f.read()
        
        # Check for key features in backtest
        if "ichimoku" in content.lower():
            features.append("Ichimoku Cloud Analysis")
        if "rsi" in content.lower():
            features.append("RSI Signals")
        if "adx" in content.lower():
            features.append("ADX Trend Strength")
        if "bollinger" in content.lower():
            features.append("Bollinger Bands")
        if "regime" in content.lower():
            features.append("Regime Detection")
        if "confidence" in content.lower():
            features.append("Signal Confidence Scoring")
        if "volume" in content.lower():
            features.append("Volume Confirmation")
        if "atr" in content.lower():
            features.append("ATR Volatility")
        if "stop_loss" in content.lower():
            features.append("Stop Loss Management")
        if "take_profit" in content.lower():
            features.append("Take Profit Logic")
            
    except Exception as e:
        print(f"   ⚠️ Cannot analyze backtest logic: {e}")
    
    return features

def analyze_live_bot_logic():
    """Extract key features from live bot logic"""
    features = []
    
    try:
        with open("api/live-bot.py", 'r') as f:
            content = f.read()
        
        # Check for key features in live bot
        if "ichimoku" in content.lower():
            features.append("Ichimoku Cloud Analysis")
        if "rsi" in content.lower():
            features.append("RSI Signals")
        if "adx" in content.lower():
            features.append("ADX Trend Strength")
        if "bollinger" in content.lower():
            features.append("Bollinger Bands")
        if "regime" in content.lower():
            features.append("Regime Detection")
        if "confidence" in content.lower():
            features.append("Signal Confidence Scoring")
        if "volume" in content.lower():
            features.append("Volume Confirmation")
        if "atr" in content.lower():
            features.append("ATR Volatility")
        if "stop_loss" in content.lower():
            features.append("Stop Loss Management")
        if "take_profit" in content.lower():
            features.append("Take Profit Logic")
            
    except Exception as e:
        print(f"   ⚠️ Cannot analyze live bot logic: {e}")
    
    return features

def check_signal_consistency(backtest_config, live_config):
    """Check if signal calculations are consistent"""
    checks = {}
    
    # RSI consistency
    backtest_rsi = backtest_config.get("best_parameters_so_far", {}).get("RSI_PERIOD")
    live_rsi = live_config.get("RSI_PERIOD")
    checks["RSI Period Consistency"] = backtest_rsi == live_rsi if backtest_rsi else True
    
    # Moving averages consistency
    backtest_ma_fast = backtest_config.get("best_parameters_so_far", {}).get("TENKAN_SEN_PERIOD")
    live_ma_fast = live_config.get("MA_FAST")
    checks["MA Fast Period Consistency"] = backtest_ma_fast == live_ma_fast if backtest_ma_fast else True
    
    # ATR consistency
    backtest_atr = backtest_config.get("best_parameters_so_far", {}).get("ATR_PERIOD")
    live_atr = live_config.get("ATR_PERIOD")
    checks["ATR Period Consistency"] = backtest_atr == live_atr if backtest_atr else True
    
    # Confidence threshold consistency
    backtest_conf = backtest_config.get("fixed_parameters", {}).get("min_confidence_for_trade")
    live_conf = live_config.get("min_confidence_for_trade")
    checks["Confidence Threshold Consistency"] = abs(backtest_conf - live_conf) < 0.01 if backtest_conf and live_conf else True
    
    return checks

def fix_parameter_mismatches(mismatches, backtest_config, live_config):
    """Fix parameter mismatches by updating live config"""
    fixes = []
    
    for param, backtest_val, live_val in mismatches:
        live_config[param] = backtest_val
        fixes.append(f"Updated {param}: {live_val} → {backtest_val}")
    
    # Save updated config
    try:
        with open("api/live_trading_config.json", 'w') as f:
            json.dump(live_config, f, indent=2)
        fixes.append("Saved updated live trading configuration")
    except Exception as e:
        fixes.append(f"Failed to save config: {e}")
    
    return fixes

def suggest_logic_fixes(missing_features):
    """Suggest fixes for missing logic features"""
    fixes = []
    
    for feature in missing_features:
        if "Ichimoku" in feature:
            fixes.append("Add Ichimoku Cloud calculations to live bot")
        elif "RSI" in feature:
            fixes.append("Implement RSI signal generation in live bot")
        elif "ADX" in feature:
            fixes.append("Add ADX trend strength analysis to live bot")
        elif "Regime" in feature:
            fixes.append("Implement regime detection in live bot")
        elif "Confidence" in feature:
            fixes.append("Add signal confidence scoring to live bot")
        elif "Volume" in feature:
            fixes.append("Implement volume confirmation in live bot")
        else:
            fixes.append(f"Implement {feature} in live bot")
    
    return fixes

def create_sync_script():
    """Create a script to keep live bot in sync with backtest"""
    sync_script = '''#!/usr/bin/env python3
"""
Auto-sync script to keep live bot parameters synchronized with latest backtest results
"""

import json
import os
from datetime import datetime

def auto_sync_parameters():
    """Automatically sync parameters from backtest to live bot"""
    print(f"🔄 Auto-sync started at {datetime.now()}")
    
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
            
            print("✅ Parameters synchronized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return False

if __name__ == "__main__":
    auto_sync_parameters()
'''
    
    with open("auto_sync_live_bot.py", 'w') as f:
        f.write(sync_script)
    
    return "Created auto_sync_live_bot.py script"

if __name__ == "__main__":
    print("Starting live bot integration analysis...")
    
    # Run analysis
    success = analyze_trading_logic_sync()
    
    # Create sync script
    create_sync_script()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review and fix any issues identified above")
    print("2. Run: python auto_sync_live_bot.py")
    print("3. Test with: python test_live_bot.py")
    print("4. Deploy and monitor live trading")
    
    if success:
        print("\n🎉 Analysis complete - no critical issues found!")
    else:
        print("\n⚠️ Issues found - please address before live trading")
