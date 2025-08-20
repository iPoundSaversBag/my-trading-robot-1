#!/usr/bin/env python3
"""
Test Vercel Testnet Bot with Proper Authentication
"""

import requests
import json
import os
from dotenv import load_dotenv

def test_authenticated_bot():
    """Test the bot with proper authentication"""
    
    load_dotenv()
    
    print("🔐 TESTING AUTHENTICATED TESTNET BOT")
    print("=" * 60)
    
    # Get authentication secret
    bot_secret = os.environ.get('BOT_SECRET', '')
    
    if not bot_secret:
        print("❌ BOT_SECRET not found in environment")
        return False
    
    # Test URL
    base_url = "https://my-trading-robot-1-jst2322mk-aidan-lanes-projects.vercel.app"
    api_url = f"{base_url}/api/live-bot"
    
    # Test with authentication
    headers = {
        'Authorization': f'Bearer {bot_secret}',
        'Content-Type': 'application/json'
    }
    
    print(f"🚀 Testing: {api_url}")
    print(f"🔑 Using BOT_SECRET: {bot_secret[:8]}...{bot_secret[-8:]}")
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ TESTNET BOT SUCCESS!")
            
            # Display bot status
            if 'signal' in data:
                signal = data['signal']
                print(f"\n📊 TRADING SIGNAL:")
                print(f"   Signal: {signal.get('signal', 'N/A')}")
                print(f"   Confidence: {signal.get('confidence', 0):.3f}")
            
            if 'market_regime' in data:
                print(f"   Market Regime: {data['market_regime']}")
            
            # Check trade execution
            if 'trade_executed' in data and data['trade_executed']:
                trade = data['trade_executed']
                print(f"\n💰 TRADE EXECUTION:")
                
                if trade.get('simulated'):
                    print(f"   ⚠️ MODE: SIMULATED")
                else:
                    print(f"   ✅ MODE: REAL TESTNET")
                
                print(f"   Symbol: {trade.get('symbol', 'N/A')}")
                print(f"   Side: {trade.get('side', 'N/A')}")
                print(f"   Quantity: {trade.get('quantity', 'N/A')}")
                print(f"   Price: ${trade.get('price', 0):,.2f}")
                print(f"   Value: ${trade.get('value', 0):.2f}")
            
            # Check if using testnet endpoint
            print(f"\n⚙️ CONFIGURATION:")
            print(f"   Config Source: {data.get('config_source', 'N/A')}")
            if 'parameters_used' in data:
                params = data['parameters_used']
                print(f"   RSI Period: {params.get('RSI_PERIOD', 'N/A')}")
                print(f"   MA Fast/Slow: {params.get('MA_FAST', 'N/A')}/{params.get('MA_SLOW', 'N/A')}")
            
            return True
            
        elif response.status_code == 401:
            print(f"❌ AUTHENTICATION FAILED")
            print(f"   Check BOT_SECRET in Vercel environment variables")
            return False
        else:
            print(f"❌ UNEXPECTED ERROR: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ REQUEST ERROR: {e}")
        return False

def test_public_endpoints():
    """Test public endpoints that don't need authentication"""
    
    print(f"\n🌐 TESTING PUBLIC ENDPOINTS")
    print("=" * 60)
    
    base_url = "https://my-trading-robot-1-jst2322mk-aidan-lanes-projects.vercel.app"
    
    # Test main dashboard
    try:
        response = requests.get(base_url, timeout=15)
        if response.status_code == 200:
            print(f"✅ Main Dashboard: {base_url}")
        else:
            print(f"❌ Main Dashboard: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard Error: {e}")

def main():
    """Run complete testnet verification"""
    
    print("🎯 COMPLETE TESTNET VERIFICATION")
    print("=" * 60)
    
    # Test authenticated bot
    bot_success = test_authenticated_bot()
    
    # Test public endpoints
    test_public_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS:")
    
    if bot_success:
        print("✅ TESTNET BOT: OPERATIONAL")
        print("✅ Your bot is live and ready for testnet trading!")
        print("✅ Dashboard will show real testnet trades")
        print("\n🎯 Next: Wait for trading signals to see live trades")
    else:
        print("❌ TESTNET BOT: NEEDS ATTENTION")
        print("⚠️ Check Vercel environment variables")
    
    print(f"\n🌐 Live Dashboard: https://my-trading-robot-1-jst2322mk-aidan-lanes-projects.vercel.app")
    
    return bot_success

if __name__ == "__main__":
    main()
