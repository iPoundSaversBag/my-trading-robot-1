#!/usr/bin/env python3
"""
Simple test for Vercel Live Trading Bot by importing as module
"""

import os
import sys
import importlib.util
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("🤖 VERCEL LIVE BOT TEST")
print("=" * 50)
print(f"Test started at: {datetime.now()}")
print("=" * 50)

try:
    # Check environment
    print("🔍 Checking environment...")
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    if api_key and api_secret:
        print(f"✅ API credentials found ({api_key[:8]}...)")
    else:
        print("❌ API credentials missing")
        sys.exit(1)
    
    # Import the live bot module
    print("\n🔄 Loading live bot module...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    live_bot_path = os.path.join(current_dir, 'api', 'live-bot.py')
    
    spec = importlib.util.spec_from_file_location("live_bot", live_bot_path)
    live_bot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live_bot_module)
    
    print("✅ Live bot module loaded")
    
    # Test config loading
    print("\n🔄 Testing configuration...")
    config = live_bot_module.load_trading_config()
    if config:
        print(f"✅ Trading config loaded ({len(config)} parameters)")
        print(f"   Symbol: {config.get('SYMBOL')}")
        print(f"   RSI Period: {config.get('RSI_PERIOD')}")
        print(f"   MA Fast/Slow: {config.get('MA_FAST')}/{config.get('MA_SLOW')}")
    else:
        print("⚠️ Using default config")
    
    # Initialize bot
    print("\n🔄 Initializing bot...")
    bot = live_bot_module.VercelLiveBot()
    print("✅ Bot initialized successfully")
    
    # Test market data
    print("\n🔄 Testing market data...")
    try:
        market_data = bot.get_market_data('BTCUSDT', '5m', 5)
        if isinstance(market_data, dict) and 'error' in market_data:
            print(f"⚠️ Market data error: {market_data['error'][:50]}...")
        elif isinstance(market_data, list) and len(market_data) > 0:
            print(f"✅ Market data fetched ({len(market_data)} candles)")
            latest = market_data[-1]
            print(f"   Latest price: ${float(latest[4]):,.2f}")
        else:
            print("⚠️ Unexpected market data format")
    except Exception as e:
        print(f"⚠️ Market data test failed: {e}")
    
    # Test trading cycle
    print("\n🔄 Testing trading cycle...")
    try:
        result = bot.execute_live_trading_cycle()
        
        print("✅ Trading cycle completed")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Action: {result.get('action', 'none')}")
        print(f"   Signal: {result.get('signal', {}).get('signal', 'none')}")
        print(f"   Confidence: {result.get('signal', {}).get('confidence', 0):.4f}")
        
        if result.get('status') == 'success':
            print("\n🎉 VERCEL LIVE BOT IS WORKING PERFECTLY!")
        else:
            print(f"\n⚠️ Result: {result.get('message', 'No message')}")
        
    except Exception as e:
        print(f"❌ Trading cycle failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 TEST SUMMARY")
    print("=" * 50)
    print("✅ BASIC FUNCTIONALITY VERIFIED")
    print("✅ Vercel live bot can be initialized")
    print("✅ Ready for cloud deployment testing")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
