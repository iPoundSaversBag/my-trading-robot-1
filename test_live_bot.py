#!/usr/bin/env python3
"""
Test script to verify live bot functionality without running continuously
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import asyncio
from datetime import datetime
import pandas as pd

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_environment_setup():
    """Test if environment is properly configured"""
    print("🔍 TESTING ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Check for .env file
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print("✅ .env file found")
    else:
        print("❌ .env file missing")
        return False
    
    # Check for API credentials
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if api_key and api_secret:
        print(f"✅ Binance API credentials configured ({api_key[:8]}...)")
    else:
        print("❌ Binance API credentials missing")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 TESTING CONFIGURATION LOADING")
    print("=" * 50)
    
    # Test live trading config
    config_path = os.path.join(project_root, 'api', 'live_trading_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Live trading config loaded ({len(config)} parameters)")
        print(f"   Symbol: {config.get('SYMBOL')}")
        print(f"   RSI Period: {config.get('RSI_PERIOD')}")
        print(f"   MA Fast/Slow: {config.get('MA_FAST')}/{config.get('MA_SLOW')}")
        return True
    else:
        print("❌ Live trading config missing")
        return False

def test_binance_connection():
    """Test Binance connection in testnet mode"""
    print("\n🔍 TESTING BINANCE CONNECTION")
    print("=" * 50)
    
    try:
        from binance.client import Client
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Create client in testnet mode
        client = Client(api_key, api_secret, testnet=True)
        
        # Test connection
        server_time = client.get_server_time()
        print(f"✅ Binance testnet connection successful")
        print(f"   Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        
        # Test account info
        account = client.get_account()
        print(f"✅ Account info retrieved")
        print(f"   Account type: {account.get('accountType', 'N/A')}")
        
        # Test market data
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        print(f"✅ Market data retrieved")
        print(f"   BTC/USDT price: ${float(ticker['price']):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        return False

def test_strategy_components():
    """Test strategy components"""
    print("\n🔍 TESTING STRATEGY COMPONENTS")
    print("=" * 50)
    
    try:
        # Test imports
        from core.strategy import MultiTimeframeStrategy
        from core.portfolio import Portfolio
        from core.position_manager import PositionManager
        print("✅ Core strategy modules imported successfully")
        
        # Load config for testing
        config_path = os.path.join(project_root, 'api', 'live_trading_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test strategy initialization
        strategy = MultiTimeframeStrategy(config)
        print("✅ Strategy initialized successfully")
        
        # Test portfolio
        portfolio = Portfolio(initial_capital=config.get('INITIAL_CAPITAL', 10000))
        print("✅ Portfolio initialized successfully")
        
        # Test position manager
        position_manager = PositionManager(config)
        print("✅ Position manager initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy components test failed: {e}")
        return False

async def test_data_fetching():
    """Test data fetching functionality"""
    print("\n🔍 TESTING DATA FETCHING")
    print("=" * 50)
    
    try:
        import ccxt
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Initialize exchange
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'sandbox': True,  # Use testnet
            'enableRateLimit': True,
        })
        
        # Test data fetching
        print("🔄 Fetching market data...")
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=100)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ Market data fetched successfully")
        print(f"   Data points: {len(df)}")
        print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
        print(f"   Latest time: {df['timestamp'].iloc[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetching test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 LIVE BOT FUNCTIONALITY TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now()}")
    print("=" * 70)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Configuration Loading", test_config_loading),
        ("Binance Connection", test_binance_connection),
        ("Strategy Components", test_strategy_components),
        ("Data Fetching", lambda: asyncio.run(test_data_fetching())),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            print()
    
    print("🏁 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Live bot is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
