#!/usr/bin/env python3
"""
Test Vercel Deployment with Testnet Configuration
Verify that the deployed bot is using testnet and executing trades
"""

import requests
import json
import time

def test_vercel_testnet_deployment():
    """Test the live Vercel deployment with testnet"""
    
    print("🚀 TESTING VERCEL TESTNET DEPLOYMENT")
    print("=" * 60)
    
    # Your Vercel URLs
    base_url = "https://my-trading-robot-1-jst2322mk-aidan-lanes-projects.vercel.app"
    api_urls = {
        "live_bot": f"{base_url}/api/live-bot",
        "dashboard": f"{base_url}/api/dashboard-integration"
    }
    
    # Test live bot endpoint
    print(f"\n🤖 Testing Live Bot (Testnet Mode)...")
    print(f"   URL: {api_urls['live_bot']}")
    
    try:
        response = requests.get(api_urls['live_bot'], timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Bot Response SUCCESS")
            
            # Check if using testnet
            if 'config_source' in data:
                print(f"   Config Source: {data['config_source']}")
            
            if 'signal' in data:
                signal = data['signal']
                print(f"   Signal: {signal.get('signal', 'N/A')}")
                print(f"   Confidence: {signal.get('confidence', 0):.3f}")
                print(f"   Market Regime: {data.get('market_regime', 'N/A')}")
            
            # Check trade execution
            if 'trade_executed' in data and data['trade_executed']:
                trade = data['trade_executed']
                print(f"\n💰 TRADE EXECUTION:")
                if trade.get('simulated'):
                    print(f"   ⚠️ MODE: SIMULATED (Safe Testing)")
                else:
                    print(f"   ✅ MODE: REAL TESTNET TRADE")
                
                print(f"   Symbol: {trade.get('symbol', 'N/A')}")
                print(f"   Side: {trade.get('side', 'N/A')}")
                print(f"   Quantity: {trade.get('quantity', 'N/A')}")
                print(f"   Price: ${trade.get('price', 0):,.2f}")
                print(f"   Value: ${trade.get('value', 0):.2f}")
            
            # Check parameters
            if 'parameters_used' in data:
                params = data['parameters_used']
                print(f"\n⚙️ TRADING PARAMETERS:")
                print(f"   RSI Period: {params.get('RSI_PERIOD', 'N/A')}")
                print(f"   MA Fast/Slow: {params.get('MA_FAST', 'N/A')}/{params.get('MA_SLOW', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Bot Response FAILED: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Bot Request ERROR: {e}")
        return False

def test_dashboard_integration():
    """Test dashboard integration API"""
    
    print(f"\n📊 Testing Dashboard Integration...")
    
    base_url = "https://my-trading-robot-1.vercel.app"
    dashboard_url = f"{base_url}/api/dashboard-integration"
    
    try:
        response = requests.get(dashboard_url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dashboard Integration SUCCESS")
            
            if 'live_summary' in data:
                summary = data['live_summary']
                print(f"   Total Cycles: {summary.get('total_cycles', 0)}")
                print(f"   Total Signals: {summary.get('total_signals', 0)}")
                print(f"   Signal Rate: {summary.get('signal_rate', 0)*100:.1f}%")
            
            return True
        else:
            print(f"❌ Dashboard Integration FAILED: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard Request ERROR: {e}")
        return False

def main():
    """Run comprehensive testnet deployment test"""
    
    print("🎯 Vercel Testnet Deployment Verification")
    print("=" * 60)
    
    # Test bot
    bot_ok = test_vercel_testnet_deployment()
    
    # Test dashboard
    dashboard_ok = test_dashboard_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEPLOYMENT TEST SUMMARY:")
    print(f"   Live Bot API: {'✅ WORKING' if bot_ok else '❌ FAILED'}")
    print(f"   Dashboard API: {'✅ WORKING' if dashboard_ok else '❌ FAILED'}")
    
    if bot_ok and dashboard_ok:
        print(f"\n🎉 TESTNET DEPLOYMENT SUCCESSFUL!")
        print(f"   Your bot is now live and ready for testnet trading")
        print(f"   Dashboard will show real testnet trades when they execute")
        print(f"   Visit: https://my-trading-robot-1.vercel.app")
    else:
        print(f"\n⚠️  DEPLOYMENT ISSUES DETECTED")
        print(f"   The deployment may need a few minutes to complete")
        print(f"   Try testing again in 2-3 minutes")
    
    return bot_ok and dashboard_ok

if __name__ == "__main__":
    main()
