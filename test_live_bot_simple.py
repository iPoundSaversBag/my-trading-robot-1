#!/usr/bin/env python3
"""
Simple test to debug live bot endpoint issues
"""
import requests
import json

def test_vercel_endpoints():
    """Test different Vercel endpoints to isolate the issue"""
    base_url = "https://my-trading-robot-1.vercel.app"
    bot_secret = "93699b3917045092715b8e16c01f2e1d"
    
    print("🧪 Testing Vercel Endpoints")
    print("=" * 50)
    
    # Test 1: Basic API endpoint
    print("1. Testing basic API endpoint...")
    try:
        response = requests.get(f"{base_url}/api/test-python", timeout=10)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Response: {data.get('status', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Live bot endpoint (GET without auth)
    print("\n2. Testing live bot endpoint (no auth)...")
    try:
        response = requests.get(f"{base_url}/api/live-bot", timeout=10)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code != 200:
            print(f"   📄 Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Live bot endpoint (GET with auth)
    print("\n3. Testing live bot endpoint (with auth)...")
    try:
        headers = {"Authorization": f"Bearer {bot_secret}"}
        response = requests.get(f"{base_url}/api/live-bot", headers=headers, timeout=30)
        print(f"   ✅ Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Response: {data.get('status', 'unknown')}")
            if 'error' in data:
                print(f"   ❌ Error in response: {data['error']}")
        else:
            print(f"   📄 Response: {response.text[:300]}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Check if environment variables are visible
    print("\n4. Testing environment check...")
    try:
        # This would be a separate endpoint to check env vars if we had one
        print("   ℹ️  Would need separate endpoint to check environment variables")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_vercel_endpoints()
