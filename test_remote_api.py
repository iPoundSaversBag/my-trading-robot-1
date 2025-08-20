#!/usr/bin/env python3
"""
Test remote Vercel API functionality
"""
import requests
import os
from dotenv import load_dotenv
import json

def test_remote_api():
    load_dotenv()
    # Use the main Vercel URL instead of deployment-specific URL
    vercel_url = "https://my-trading-robot-1.vercel.app"
    bot_secret = os.environ.get('BOT_SECRET')

    print(f'🌐 Testing Vercel Remote API...')
    print(f'URL: {vercel_url}')
    print(f'Secret: {bot_secret[:8] if bot_secret else None}...')

    # Test the API endpoint
    headers = {'Authorization': f'Bearer {bot_secret}'}
    try:
        response = requests.get(f'{vercel_url}/api/live-bot', headers=headers, timeout=15)
        print(f'📊 Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'✅ SUCCESS! Remote API working!')
            print(f'✅ Signal: {data.get("signal", {}).get("signal", "N/A")}')
            print(f'✅ Confidence: {data.get("signal", {}).get("confidence", "N/A")}')
            print(f'✅ Regime: {data.get("market_regime", {}).get("primary", "N/A")}')
            return True
        else:
            print(f'❌ Response: {response.text[:200]}...')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = test_remote_api()
    print(f'\n🎯 Remote API Test: {"✅ PASSED" if success else "❌ FAILED"}')
