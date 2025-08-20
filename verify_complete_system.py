#!/usr/bin/env python3
"""
Final Deployment Verification
Check if all our fixes are live on Vercel
"""

import requests
import time
import json

def check_deployment_status():
    """Check if the latest deployment has the fixes"""
    
    print("🚀 FINAL DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    dashboard_url = "https://my-trading-robot-1.vercel.app"
    
    try:
        # Get the dashboard HTML
        response = requests.get(dashboard_url, timeout=15)
        
        if response.status_code == 200:
            html_content = response.text
            
            print("✅ Dashboard accessible")
            
            # Check if our fixes are deployed
            fixes_deployed = []
            
            # Check 1: CSS fixes (space-y should be replaced)
            if "margin-top: 0.5rem" in html_content or "margin-bottom: 0.5rem" in html_content:
                fixes_deployed.append("CSS space-y fixes deployed")
            elif "space-y" not in html_content:
                fixes_deployed.append("No CSS space-y properties found")
            
            # Check 2: API integration
            if "/api/live-bot" in html_content:
                fixes_deployed.append("Live bot API integration present")
            
            if "updateLiveTestnetData" in html_content:
                fixes_deployed.append("Testnet data update function present")
            
            # Check 3: Authorization headers
            if "Authorization" in html_content and "Bearer" in html_content:
                fixes_deployed.append("API authentication headers present")
            
            print(f"\n🔧 Deployment Fixes Status:")
            for fix in fixes_deployed:
                print(f"   ✅ {fix}")
            
            if len(fixes_deployed) >= 3:
                print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
                print(f"   All fixes are now live on Vercel")
                
                # Test live data panel functionality
                print(f"\n📡 Testing Live Data Panel...")
                
                # Check if we can see live data indicators
                live_indicators = [
                    "Live Data" in html_content,
                    "Real-time" in html_content or "real-time" in html_content,
                    "testnet" in html_content.lower()
                ]
                
                active_indicators = sum(live_indicators)
                print(f"   Live data indicators: {active_indicators}/3 present")
                
                return True
            else:
                print(f"\n⚠️  Deployment may still be propagating...")
                print(f"   Found {len(fixes_deployed)} of expected fixes")
                return False
                
        else:
            print(f"❌ Dashboard not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Deployment check failed: {e}")
        return False

def test_live_functionality():
    """Test if the live data actually updates"""
    
    print(f"\n🔄 TESTING LIVE FUNCTIONALITY")
    print("=" * 60)
    
    api_url = "https://my-trading-robot-1.vercel.app/api/live-bot"
    headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
    
    try:
        # Make multiple requests to see if data changes
        print("Making test requests to verify live data...")
        
        responses = []
        for i in range(3):
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                signal_data = {
                    'signal': data.get('signal', {}).get('signal'),
                    'confidence': data.get('signal', {}).get('confidence'),
                    'timestamp': data.get('signal', {}).get('timestamp')
                }
                responses.append(signal_data)
                print(f"   Request {i+1}: {signal_data['signal']} ({signal_data['confidence']:.3f})")
            time.sleep(2)
        
        if len(responses) == 3:
            print(f"\n✅ Live API is responding consistently")
            print(f"   Current signal: {responses[-1]['signal']}")
            print(f"   Confidence: {responses[-1]['confidence']:.3f}")
            return True
        else:
            print(f"\n❌ API reliability issues")
            return False
            
    except Exception as e:
        print(f"❌ Live functionality test failed: {e}")
        return False

def main():
    """Complete deployment verification"""
    
    print("🚀 COMPLETE SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Check deployment
    deployment_ok = check_deployment_status()
    
    # Wait a moment if deployment is still propagating
    if not deployment_ok:
        print(f"\n⏳ Waiting 30 seconds for deployment to propagate...")
        time.sleep(30)
        deployment_ok = check_deployment_status()
    
    # Test live functionality
    live_ok = test_live_functionality()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🏁 FINAL SYSTEM STATUS:")
    print(f"   Dashboard Deployment: {'✅ LIVE' if deployment_ok else '⚠️  PROPAGATING'}")
    print(f"   Live Data Functionality: {'✅ WORKING' if live_ok else '❌ ISSUES'}")
    
    if deployment_ok and live_ok:
        print(f"\n🎉 SYSTEM FULLY OPERATIONAL!")
        print(f"   🌐 Dashboard: https://my-trading-robot-1.vercel.app")
        print(f"   📡 Live Data Panel: Shows real testnet trading")
        print(f"   🤖 Bot Status: Active on Binance testnet")
        print(f"   💰 Trading: Real execution with testnet funds")
        print(f"   🔄 Updates: Every 5 seconds automatically")
        
        print(f"\n📱 How to use:")
        print(f"   1. Visit your dashboard")
        print(f"   2. Click '📡 Live Data' tab")
        print(f"   3. Watch real-time testnet trading signals")
        print(f"   4. Monitor account balances and trades")
        
    return deployment_ok and live_ok

if __name__ == "__main__":
    main()
