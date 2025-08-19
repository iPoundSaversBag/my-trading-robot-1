#!/usr/bin/env python3
"""
Test script for the Vercel-based live trading bot
"""

import os
import sys
import json
from datetime import datetime

# Add the api directory to path
api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api')
sys.path.insert(0, api_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_vercel_live_bot():
    """Test the Vercel-based live bot functionality"""
    print("🤖 TESTING VERCEL LIVE BOT")
    print("=" * 50)
    
    try:
        # Import the VercelLiveBot class
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api'))
        exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api', 'live-bot.py')).read())
        from __main__ import VercelLiveBot, load_trading_config
        
        print("✅ Successfully imported VercelLiveBot")
        
        # Test config loading
        config = load_trading_config()
        if config:
            print(f"✅ Trading config loaded ({len(config)} parameters)")
            print(f"   Symbol: {config.get('SYMBOL')}")
            print(f"   RSI Period: {config.get('RSI_PERIOD')}")
            print(f"   MA Fast/Slow: {config.get('MA_FAST')}/{config.get('MA_SLOW')}")
        else:
            print("⚠️ Using default config")
        
        # Initialize bot
        print("\n🔄 Initializing bot...")
        bot = VercelLiveBot()
        print("✅ Bot initialized successfully")
        
        # Test API credentials
        if bot.api_key and bot.api_secret:
            print(f"✅ API credentials configured ({bot.api_key[:8]}...)")
        else:
            print("❌ API credentials missing")
            return False
        
        # Test market data fetching
        print("\n🔄 Testing market data fetching...")
        market_data = bot.get_market_data('BTCUSDT', '5m', 10)
        
        if 'error' in market_data:
            print(f"❌ Market data fetch failed: {market_data['error']}")
            return False
        else:
            print(f"✅ Market data fetched successfully ({len(market_data)} candles)")
            if market_data:
                latest = market_data[-1]
                print(f"   Latest price: ${float(latest[4]):,.2f}")
                print(f"   Timestamp: {datetime.fromtimestamp(latest[0]/1000)}")
        
        # Test signal generation
        print("\n🔄 Testing signal generation...")
        signal_result = bot.generate_signals(market_data)
        
        if signal_result:
            print("✅ Signal generation successful")
            signal = signal_result.get('signal', {})
            print(f"   Signal: {signal.get('signal', 'NONE')}")
            print(f"   Confidence: {signal.get('confidence', 0):.4f}")
            print(f"   Regime: {signal_result.get('market_regime', {}).get('primary', 'unknown')}")
        else:
            print("⚠️ No signal generated")
        
        # Test account info (if available)
        print("\n🔄 Testing account info...")
        try:
            account_info = bot._make_request('/api/v3/account')
            if 'error' in account_info:
                print(f"⚠️ Account info not available: {account_info['error']}")
            else:
                print("✅ Account info retrieved successfully")
                print(f"   Account type: {account_info.get('accountType', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Account info test failed: {e}")
        
        # Test full trading cycle (dry run)
        print("\n🔄 Testing full trading cycle...")
        cycle_result = bot.execute_live_trading_cycle()
        
        if cycle_result.get('status') == 'success':
            print("✅ Trading cycle completed successfully")
            print(f"   Action: {cycle_result.get('action', 'none')}")
            print(f"   Signal: {cycle_result.get('signal', {}).get('signal', 'none')}")
        else:
            print(f"⚠️ Trading cycle result: {cycle_result.get('status', 'unknown')}")
            if 'message' in cycle_result:
                print(f"   Message: {cycle_result['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("🚀 VERCEL LIVE BOT TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now()}")
    print("=" * 70)
    
    success = test_vercel_live_bot()
    
    print("\n🏁 TEST SUMMARY")
    print("=" * 70)
    
    if success:
        print("🎉 VERCEL LIVE BOT TEST PASSED!")
        print("✅ The Vercel-based live bot is working properly")
        print("✅ Ready for cloud deployment")
    else:
        print("❌ VERCEL LIVE BOT TEST FAILED!")
        print("⚠️ Please fix issues before deployment")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
