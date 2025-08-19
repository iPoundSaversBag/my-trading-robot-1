#!/usr/bin/env python3
"""
Simple test to run the live bot trading cycle locally
"""
import sys
import os
import json

# Set up the path to import from api directory
api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api')
sys.path.insert(0, api_path)

def test_live_bot_execution():
    """Test the actual live bot execution"""
    print("🤖 Testing Live Bot Execution")
    print("=" * 50)
    
    try:
        # Import the live bot (note the file name uses dash, so we import the module differently)
        import importlib.util
        live_bot_path = os.path.join(api_path, 'live-bot.py')
        spec = importlib.util.spec_from_file_location("live_bot", live_bot_path)
        live_bot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live_bot_module)
        
        VercelLiveBot = live_bot_module.VercelLiveBot
        
        print("✅ Live bot imported successfully")
        
        # Create bot instance
        bot = VercelLiveBot()
        print("✅ Bot instance created")
        
        # Check configuration
        if bot.config:
            print(f"✅ Configuration loaded:")
            print(f"   Symbol: {bot.config.get('SYMBOL', 'Not set')}")
            print(f"   Timeframe: {bot.config.get('TIMEFRAME', 'Not set')}")
            print(f"   RSI Period: {bot.config.get('RSI_PERIOD', 'Not set')}")
            print(f"   MA Fast/Slow: {bot.config.get('MA_FAST', 'Not set')}/{bot.config.get('MA_SLOW', 'Not set')}")
        else:
            print("❌ No configuration loaded")
            return
        
        print("\n🔄 Executing trading cycle...")
        
        # Execute the trading cycle
        result = bot.execute_live_trading_cycle()
        
        print("\n📊 RESULTS:")
        print("=" * 30)
        
        if isinstance(result, dict):
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                if 'details' in result:
                    print(f"📝 Details: {result['details']}")
                if 'status' in result:
                    print(f"🏷️  Status: {result['status']}")
            else:
                print(f"✅ Status: {result.get('status', 'unknown')}")
                
                if result.get('status') == 'success':
                    print(f"📊 Signal: {result.get('signal', 'unknown')}")
                    print(f"🎯 Confidence: {result.get('confidence', 'unknown')}")
                    print(f"💡 Reason: {result.get('reason', 'unknown')}")
                    
                    # Market data
                    if 'market_data' in result:
                        market = result['market_data']
                        print(f"💰 Current Price: ${market.get('current_price', 'N/A')}")
                        print(f"📊 RSI: {market.get('rsi', 'N/A')}")
                        print(f"📈 Price Change: {market.get('price_change_pct', 'N/A')}%")
                    
                    # Account info
                    if 'account_info' in result:
                        account = result['account_info']
                        print(f"💼 Account Status: {account.get('status', 'unknown')}")
                        balance_count = len(account.get('balances', {}))
                        print(f"💰 Active Balances: {balance_count}")
                
            print("\n📄 Full Response:")
            print(json.dumps(result, indent=2)[:1000] + "..." if len(json.dumps(result, indent=2)) > 1000 else json.dumps(result, indent=2))
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            print(f"Result: {result}")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure the live_bot.py file exists in the api directory")
    except Exception as e:
        print(f"❌ Exception: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_bot_execution()
