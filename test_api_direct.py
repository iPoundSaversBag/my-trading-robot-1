#!/usr/bin/env python3
"""
Direct API Test
Test if our live-bot API is working correctly
"""

import requests
import json

def test_live_bot_api():
    """Test the live bot API directly"""
    
    print("🤖 TESTING LIVE BOT API DIRECTLY")
    print("=" * 50)
    
    api_url = "https://my-trading-robot-1.vercel.app/api/live-bot"
    headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
    
    try:
        print(f"📡 Testing API: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=15)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Response Success!")
            
            # Show key data that the dashboard should display
            if 'signal' in data:
                signal = data['signal']
                print(f"   📊 Signal: {signal.get('signal')} ({signal.get('confidence', 0):.3f} confidence)")
                print(f"   💰 Current Price: ${signal.get('current_price', 0):,.2f}")
            
            if 'account_balance' in data:
                balances = data['account_balance']
                print(f"   💳 USDT Balance: ${balances.get('USDT', 0):,.2f}")
                print(f"   🪙 BTC Balance: {balances.get('BTC', 0):.6f}")
            
            if 'trade_executed' in data:
                trade = data['trade_executed']
                mode = 'Simulated' if trade.get('simulated') else 'Real Testnet'
                print(f"   🔄 Trade Mode: {mode}")
                print(f"   📈 Last Trade: {trade.get('side', 'N/A')}")
            
            return True
            
        else:
            print(f"   ❌ API Error: {response.status_code}")
            if response.text:
                print(f"   Error Details: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        return False

def test_dashboard_access():
    """Test basic dashboard access"""
    
    print(f"\n🌐 TESTING DASHBOARD ACCESS")
    print("=" * 50)
    
    dashboard_url = "https://my-trading-robot-1.vercel.app"
    
    try:
        response = requests.get(dashboard_url, timeout=10)
        
        print(f"   Dashboard Status: {response.status_code}")
        
        if response.status_code == 200:
            html_content = response.text
            print(f"   📄 Content Length: {len(html_content):,} characters")
            
            # Check for basic elements
            has_title = "Tearsheet" in html_content
            has_live_data = "Live Data" in html_content
            
            print(f"   📋 Title Present: {'✅' if has_title else '❌'}")
            print(f"   📡 Live Data Tab: {'✅' if has_live_data else '❌'}")
            
            return True
        else:
            print(f"   ❌ Dashboard Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Dashboard Error: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    
    api_ok = test_live_bot_api()
    dashboard_ok = test_dashboard_access()
    
    print(f"\n" + "=" * 50)
    print(f"📋 TEST SUMMARY:")
    print(f"   Live Bot API: {'✅ WORKING' if api_ok else '❌ FAILED'}")
    print(f"   Dashboard Access: {'✅ WORKING' if dashboard_ok else '❌ FAILED'}")
    
    if api_ok and dashboard_ok:
        print(f"\n💡 ISSUE DIAGNOSIS:")
        print(f"   • API is working correctly")
        print(f"   • Dashboard is accessible")
        print(f"   • Problem is likely in dashboard JavaScript integration")
        print(f"   • Vercel deployment may be cached/delayed")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        print(f"   1. Wait 5-10 more minutes for Vercel cache to clear")
        print(f"   2. Try hard refresh in browser (Ctrl+F5)")
        print(f"   3. Check browser developer console for errors")
        print(f"   4. Verify the live data panel is calling the API correctly")
    else:
        print(f"\n⚠️  Infrastructure issues detected")
    
    return api_ok and dashboard_ok

if __name__ == "__main__":
    main()
