#!/usr/bin/env python3
"""
Test Live Testnet Bot - Direct API Test
"""

import requests
import json
import os
from dotenv import load_dotenv

def test_live_bot_direct():
    """Test the live bot API directly"""
    
    load_dotenv()
    
    print("🤖 DIRECT TESTNET BOT API TEST")
    print("=" * 60)
    
    bot_secret = os.environ.get('BOT_SECRET', '')
    
    # Try the exact URL from the deployment
    url = "https://my-trading-robot-1.vercel.app/api/live-bot"
    
    # Test different request methods
    test_methods = [
        ("GET with auth", "GET", {"Authorization": f"Bearer {bot_secret}"}),
        ("POST with auth", "POST", {"Authorization": f"Bearer {bot_secret}"}),
        ("GET no auth", "GET", {}),
        ("POST no auth", "POST", {})
    ]
    
    for test_name, method, headers in test_methods:
        print(f"\n🔍 Testing: {test_name}")
        print(f"   Method: {method}")
        print(f"   Headers: {list(headers.keys())}")
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=15)
            else:
                response = requests.post(url, headers=headers, timeout=15)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"   ✅ SUCCESS!")
                    
                    if 'signal' in data:
                        print(f"   Signal: {data['signal'].get('signal', 'N/A')}")
                        print(f"   Confidence: {data['signal'].get('confidence', 0):.3f}")
                    
                    if 'trade_executed' in data and data['trade_executed']:
                        trade = data['trade_executed']
                        print(f"   Trade: {trade.get('side', 'N/A')} {trade.get('quantity', 'N/A')} @ ${trade.get('price', 0):.2f}")
                    
                    return True
                    
                except json.JSONDecodeError:
                    print(f"   ⚠️ Invalid JSON response")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def test_simple_endpoint():
    """Test if any endpoint works"""
    
    print(f"\n🌐 TESTING SIMPLE ENDPOINT ACCESS")
    print("=" * 60)
    
    # Test the main domain
    try:
        response = requests.get("https://my-trading-robot-1.vercel.app", timeout=15)
        print(f"Main site: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Main site accessible")
            return True
        else:
            print(f"❌ Main site: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Main site error: {e}")
    
    return False

def main():
    """Run comprehensive test"""
    
    print("🚀 LIVE TESTNET BOT VERIFICATION")
    print("=" * 60)
    
    # Test simple access first
    simple_ok = test_simple_endpoint()
    
    # Test bot API
    bot_ok = test_live_bot_direct()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"   Site Access: {'✅ OK' if simple_ok else '❌ FAILED'}")
    print(f"   Bot API: {'✅ OK' if bot_ok else '❌ FAILED'}")
    
    if bot_ok:
        print(f"\n🎉 TESTNET BOT IS LIVE!")
        print(f"   Your bot is executing testnet trades")
        print(f"   Dashboard will show live trade data")
    elif simple_ok:
        print(f"\n⚠️ Site accessible but API issues")
        print(f"   May need to wait for deployment to complete")
    else:
        print(f"\n❌ Deployment issues detected")
        print(f"   Try redeploying with: vercel --prod")

if __name__ == "__main__":
    main()
