#!/usr/bin/env python3
"""
Test Automated Pipeline
Verify that the automated trading pipeline is working correctly
"""

import subprocess
import sys
import time
import json
import requests
from pathlib import Path

def test_parameter_sync():
    """Test parameter synchronization"""
    print("🧪 TESTING PARAMETER SYNC")
    print("=" * 50)
    
    try:
        # Run parameter sync test
        result = subprocess.run([
            sys.executable, "automated_pipeline.py", "--sync-once"
        ], capture_output=True, text=True, timeout=60)
        
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_live_bot_api():
    """Test live bot API connectivity"""
    print("\n🧪 TESTING LIVE BOT API")
    print("=" * 50)
    
    try:
        url = "https://my-trading-robot-1.vercel.app/api/live-bot"
        headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
        
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Signal: {data.get('signal', {}).get('signal', 'N/A')}")
            print(f"Confidence: {data.get('signal', {}).get('confidence', 0):.3f}")
            print(f"USDT Balance: ${data.get('account_balance', {}).get('USDT', 0):,.2f}")
            return True
        else:
            print(f"❌ API failed: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_parameter_sync_api():
    """Test parameter sync API endpoint"""
    print("\n🧪 TESTING PARAMETER SYNC API")
    print("=" * 50)
    
    try:
        # Test GET (check current config)
        url = "https://my-trading-robot-1.vercel.app/api/parameter-sync"
        
        response = requests.get(url, timeout=15)
        print(f"GET Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Current config status: {data.get('status')}")
            print(f"Last updated: {data.get('last_updated', 'Never')}")
        
        # Test POST (update config)
        test_config = {
            "RSI_PERIOD": 14,
            "MA_FAST": 12,
            "MA_SLOW": 26,
            "test_update": True
        }
        
        response = requests.post(url, json={"config": test_config}, timeout=15)
        print(f"POST Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Update result: {data.get('status')}")
            return True
        else:
            print(f"❌ POST failed: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Parameter sync API test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring functionality"""
    print("\n🧪 TESTING MONITORING")
    print("=" * 50)
    
    try:
        # Run monitoring test
        result = subprocess.run([
            sys.executable, "automated_pipeline.py", "--monitor-once"
        ], capture_output=True, text=True, timeout=30)
        
        print(f"Exit code: {result.returncode}")
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AUTOMATED PIPELINE TESTING")
    print("=" * 70)
    
    tests = [
        ("Parameter Sync", test_parameter_sync),
        ("Live Bot API", test_live_bot_api),
        ("Parameter Sync API", test_parameter_sync_api),
        ("Monitoring", test_monitoring)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏁 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Automated pipeline is ready!")
        print("\n📝 Next Steps:")
        print("   1. Run 'python watcher.py --mode backtest' to generate optimized parameters")
        print("   2. Parameters will automatically sync to live bot")
        print("   3. Run 'python live_monitor.py' to monitor live performance")
        print("   4. Check live dashboard: https://my-trading-robot-1.vercel.app")
    else:
        print("⚠️  Some tests failed - check configuration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
