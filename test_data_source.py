#!/usr/bin/env python3
"""
Test to determine if remote API is using real market data or simulated data
"""
import requests
import os
from dotenv import load_dotenv
import json

def test_data_source():
    load_dotenv()
    vercel_url = "https://my-trading-robot-1.vercel.app"
    bot_secret = os.environ.get('BOT_SECRET')

    print(f'🔍 TESTING DATA SOURCE ANALYSIS')
    print(f'===============================================')

    # Make multiple requests to see if data changes realistically
    responses = []
    headers = {'Authorization': f'Bearer {bot_secret}'}
    
    for i in range(3):
        try:
            response = requests.get(f'{vercel_url}/api/live-bot', headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                responses.append(data)
                print(f'\n📊 Request {i+1}:')
                print(f'   Signal: {data.get("signal", {}).get("signal", "N/A")}')
                print(f'   Confidence: {data.get("signal", {}).get("confidence", "N/A")}')
                print(f'   Regime: {data.get("market_regime", {}).get("primary", "N/A")}')
                print(f'   Timestamp: {data.get("timestamp", "N/A")}')
                
                # Check for simulation indicators
                account_balance = data.get("account_balance", {})
                trade_executed = data.get("trade_executed", {})
                
                if "simulated" in str(trade_executed).lower():
                    print(f'   ⚠️ SIMULATION detected in trade execution')
                
                # Check if using Binance testnet vs mainnet
                if "testnet" in str(data).lower():
                    print(f'   ⚠️ TESTNET detected')
                elif "simulate" in str(data).lower():
                    print(f'   ⚠️ SIMULATION detected')
                else:
                    print(f'   ✅ Appears to be real data')
                    
            else:
                print(f'❌ Request {i+1} failed: {response.status_code}')
                
        except Exception as e:
            print(f'❌ Request {i+1} error: {e}')
    
    # Analysis
    print(f'\n🔍 DATA SOURCE ANALYSIS:')
    print(f'===============================================')
    
    if len(responses) >= 2:
        # Check if signals/regimes are changing (indicates real data)
        signals = [r.get("signal", {}).get("signal") for r in responses]
        regimes = [r.get("market_regime", {}).get("primary") for r in responses]
        timestamps = [r.get("timestamp") for r in responses]
        
        signal_changes = len(set(signals)) > 1
        regime_changes = len(set(regimes)) > 1
        timestamp_changes = len(set(timestamps)) > 1
        
        print(f'   Signals changing: {signal_changes} {signals}')
        print(f'   Regimes changing: {regime_changes} {regimes}')
        print(f'   Timestamps changing: {timestamp_changes} {timestamps}')
        
        # Check if using real Binance API endpoint
        print(f'\n🌐 API ENDPOINT ANALYSIS:')
        sample_response = responses[0]
        
        # Look for Binance API status codes or real market characteristics
        if "error" in str(sample_response) and "451" in str(sample_response):
            print(f'   ⚠️ Binance API blocked (451) - Using simulated data')
            return "SIMULATED"
        elif timestamp_changes and any("api.binance.com" in str(r) for r in responses):
            print(f'   ✅ Real Binance API detected')
            return "REAL"
        elif not signal_changes and not regime_changes:
            print(f'   ⚠️ Data not changing - Likely simulated')
            return "SIMULATED"
        else:
            print(f'   ✅ Data appears to be real market data')
            return "REAL"
    else:
        print(f'   ❌ Insufficient data to analyze')
        return "UNKNOWN"

if __name__ == "__main__":
    result = test_data_source()
    print(f'\n🎯 CONCLUSION: Using {result} data')
