#!/usr/bin/env python3
"""
Quick test of the live-bot endpoint
"""
import requests

# Test without authentication first
url = "https://my-trading-robot-1-i836s84l6-aidan-lanes-projects.vercel.app/api/live-bot"

print("🧪 Testing live-bot endpoint...")
print(f"URL: {url}")

try:
    # Test 1: Basic GET request
    print("\n1️⃣ Testing basic access...")
    response = requests.get(url, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
    # Test 2: With authentication
    print("\n2️⃣ Testing with authentication...")
    headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n✅ Test complete!")
