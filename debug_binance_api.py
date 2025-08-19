#!/usr/bin/env python3
"""
Debug Binance API response to understand the data format
"""
import requests
import json

def test_binance_api():
    """Test what Binance API actually returns"""
    print("🔍 Testing Binance API Response Format")
    print("=" * 50)
    
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 5},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data type: {type(data)}")
            print(f"Length: {len(data) if isinstance(data, list) else 'N/A'}")
            
            if isinstance(data, list) and len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item: {data[0]}")
                print(f"First item length: {len(data[0]) if isinstance(data[0], (list, tuple)) else 'N/A'}")
                
                # Try to extract close price (should be index 4)
                if len(data[0]) > 4:
                    close_price = float(data[0][4])
                    print(f"Close price extracted: {close_price}")
                else:
                    print("❌ Item doesn't have enough elements for close price (index 4)")
            else:
                print("❌ Data is not a list or is empty")
                print(f"Data content: {data}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_binance_api()
