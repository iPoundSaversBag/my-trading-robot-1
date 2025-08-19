#!/usr/bin/env python3
"""
Check specific regime access in live bot
"""
import sys
import os
import json

# Set up the path to import from api directory
api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api')
sys.path.insert(0, api_path)

def analyze_regime_parameters():
    """Analyze what regime parameters are actually available"""
    print("🔍 DETAILED REGIME PARAMETER ANALYSIS")
    print("=" * 60)
    
    try:
        # Load the config directly
        config_path = os.path.join(api_path, 'live_trading_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded: {len(config)} total parameters")
        
        # Define the 9 regimes
        regimes = [
            'trending_bull', 'trending_bear', 'ranging',
            'high_volatility', 'low_volatility', 
            'breakout_bullish', 'breakout_bearish',
            'accumulation', 'distribution'
        ]
        
        # Define filter types
        filters = ['ICHIMOKU_CLOUD', 'RSI', 'ADX', 'BBANDS', 'MACD', 'VOLUME_BREAKOUT']
        
        print(f"\n📊 REGIME-SPECIFIC FILTER ACCESS:")
        print("=" * 60)
        
        total_expected = len(regimes) * len(filters)
        total_found = 0
        active_filters = 0
        
        for regime in regimes:
            print(f"\n🎯 {regime.upper()}:")
            regime_found = 0
            regime_active = 0
            
            for filter_type in filters:
                param_key = f"USE_{filter_type}_FILTER_{regime}"
                if param_key in config:
                    value = config[param_key]
                    if value:
                        print(f"  🟢 {filter_type}: ENABLED")
                        active_filters += 1
                        regime_active += 1
                    else:
                        print(f"  🔴 {filter_type}: DISABLED")
                    total_found += 1
                    regime_found += 1
                else:
                    print(f"  ❌ {filter_type}: MISSING")
            
            print(f"     └─ Found: {regime_found}/{len(filters)} | Active: {regime_active}")
        
        print(f"\n📈 SUMMARY:")
        print("=" * 60)
        print(f"🎯 Expected Parameters: {total_expected}")
        print(f"🎯 Found Parameters: {total_found}")
        print(f"🎯 Active Filters: {active_filters}")
        print(f"🎯 Coverage: {(total_found/total_expected)*100:.1f}%")
        
        return total_found, active_filters, total_expected
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, 0, 0

def test_live_regime_usage():
    """Test if the live bot can actually use regime-specific parameters"""
    print(f"\n🧠 TESTING LIVE REGIME USAGE")
    print("=" * 60)
    
    try:
        # Import live bot
        import importlib.util
        live_bot_path = os.path.join(api_path, 'live-bot.py')
        spec = importlib.util.spec_from_file_location("live_bot", live_bot_path)
        live_bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live_bot_module)
        
        VercelLiveBot = live_bot_module.VercelLiveBot
        bot = VercelLiveBot()
        
        print("✅ Live bot loaded successfully")
        
        # Test regime detection capability
        result = bot.execute_live_trading_cycle()
        
        if isinstance(result, dict) and 'status' in result:
            print(f"✅ Trading cycle executed: {result['status']}")
            
            if 'market_regime' in result:
                regime = result['market_regime']
                print(f"✅ Regime detected: {regime.get('primary', 'unknown')}")
                
                # Check if regime-specific logic was applied
                if 'signal' in result:
                    signal = result['signal']
                    if 'filters_used' in signal:
                        filters = signal['filters_used']
                        print(f"✅ Filters applied: {len(filters)} filters")
                        if filters:
                            for f in filters:
                                print(f"  - {f}")
                        else:
                            print(f"  - No filters active for current regime")
                    
                    return True
            else:
                print("❌ No regime information found")
                return False
        else:
            print(f"❌ Unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🤖 LIVE BOT REGIME PARAMETER ACCESS TEST")
    print("=" * 70)
    
    # Test parameter access
    found, active, expected = analyze_regime_parameters()
    
    # Test live usage
    live_test = test_live_regime_usage()
    
    print(f"\n🏁 FINAL ASSESSMENT:")
    print("=" * 70)
    
    coverage = (found / expected) * 100 if expected > 0 else 0
    
    print(f"📊 Parameter Coverage: {coverage:.1f}% ({found}/{expected})")
    print(f"🔥 Active Filters: {active}")
    print(f"🧠 Live Usage: {'✅ Working' if live_test else '❌ Issues'}")
    
    if coverage >= 95 and active >= 5 and live_test:
        print(f"\n🎉 EXCELLENT! Live bot has comprehensive regime access!")
    elif coverage >= 80 and live_test:
        print(f"\n✅ GOOD! Live bot has solid regime capabilities!")
    elif live_test:
        print(f"\n👍 WORKING! Basic regime functionality confirmed!")
    else:
        print(f"\n⚠️  ISSUES! Some regime functionality may be missing!")
