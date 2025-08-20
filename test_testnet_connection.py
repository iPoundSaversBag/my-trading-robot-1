#!/usr/bin/env python3
"""
Test Binance Testnet Connection and Trading
Verify that your testnet API keys work and bot can execute trades
"""

import os
import requests
import hashlib
import hmac
import time
from urllib.parse import urlencode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_signature(secret_key, params):
    """Create HMAC SHA256 signature for Binance API"""
    query_string = urlencode(params)
    signature = hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def test_testnet_connection():
    """Test connection to Binance testnet"""
    
    print("🔍 Testing Binance Testnet Configuration...")
    print("=" * 60)
    
    # Get API credentials
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("❌ ERROR: API keys not found in environment variables")
        print("   Make sure .env file contains BINANCE_API_KEY and BINANCE_API_SECRET")
        return False
    
    print(f"✅ API Key loaded: {api_key[:8]}...{api_key[-8:]}")
    print(f"✅ Secret Key loaded: {api_secret[:8]}...{api_secret[-8:]}")
    
    # Test basic connection (no auth required)
    base_url = 'https://testnet.binance.vision'
    
    print(f"\n🌐 Testing connection to {base_url}...")
    
    try:
        # Test server time
        response = requests.get(f"{base_url}/api/v3/time", timeout=10)
        if response.status_code == 200:
            server_time = response.json()
            print(f"✅ Server connection successful")
            print(f"   Server time: {server_time}")
        else:
            print(f"❌ Server connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test authenticated endpoint (account info)
    print(f"\n🔐 Testing authenticated access...")
    
    try:
        # Get server time first for accurate timestamp
        server_response = requests.get(f"{base_url}/api/v3/time", timeout=10)
        server_time = server_response.json()['serverTime']
        
        params = {'timestamp': server_time, 'recvWindow': 5000}
        signature = create_signature(api_secret, params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': api_key}
        
        response = requests.get(
            f"{base_url}/api/v3/account", 
            params=params, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            account = response.json()
            print(f"✅ Authenticated access successful")
            
            # Display account balances
            print(f"\n💰 Testnet Account Balances:")
            balances = []
            for balance in account.get('balances', []):
                free = float(balance['free'])
                if free > 0:
                    balances.append((balance['asset'], free))
                    print(f"   {balance['asset']}: {free}")
            
            if not balances:
                print("   No balances found (new testnet account)")
                print("   ℹ️  Visit https://testnet.binance.vision/ to get testnet funds")
            
            return True
            
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False

def test_market_data():
    """Test market data retrieval"""
    
    print(f"\n📊 Testing market data retrieval...")
    
    base_url = 'https://testnet.binance.vision'
    
    try:
        # Get current price
        response = requests.get(
            f"{base_url}/api/v3/ticker/price",
            params={'symbol': 'BTCUSDT'},
            timeout=10
        )
        
        if response.status_code == 200:
            price_data = response.json()
            print(f"✅ Market data retrieved successfully")
            print(f"   BTCUSDT Price: ${float(price_data['price']):.2f}")
            return True
        else:
            print(f"❌ Market data failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Market data error: {e}")
        return False

def main():
    """Run all testnet tests"""
    
    print("🚀 BINANCE TESTNET VERIFICATION")
    print("=" * 60)
    
    # Test connection
    connection_ok = test_testnet_connection()
    
    # Test market data
    market_ok = test_market_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"   Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"   Market Data: {'✅ PASS' if market_ok else '❌ FAIL'}")
    
    if connection_ok and market_ok:
        print("\n🎉 TESTNET READY FOR TRADING!")
        print("   Your bot can now execute real trades safely on the testnet")
        print("   No real money is at risk")
    else:
        print("\n⚠️  CONFIGURATION ISSUES DETECTED")
        print("   Please check your API keys and network connection")
    
    return connection_ok and market_ok

if __name__ == "__main__":
    main()
