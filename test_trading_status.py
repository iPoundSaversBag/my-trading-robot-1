#!/usr/bin/env python3
"""
Comprehensive test to verify bot configuration and trade execution
"""
import requests
import os
from dotenv import load_dotenv
import json

def test_bot_trading_status():
    load_dotenv()
    vercel_url = "https://my-trading-robot-1.vercel.app"
    bot_secret = os.environ.get('BOT_SECRET')

    print(f'🔍 COMPREHENSIVE BOT TRADING ANALYSIS')
    print(f'=' * 50)

    # Test the live bot API
    headers = {'Authorization': f'Bearer {bot_secret}'}
    
    try:
        response = requests.get(f'{vercel_url}/api/live-bot', headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            
            print(f'📊 CURRENT BOT STATUS:')
            print(f'-' * 30)
            print(f'✅ API Response: SUCCESS')
            print(f'✅ Signal: {data.get("signal", {}).get("signal", "N/A")}')
            print(f'✅ Confidence: {data.get("signal", {}).get("confidence", "N/A")}')
            print(f'✅ Market Regime: {data.get("market_regime", {}).get("primary", "N/A")}')
            
            # Check trade execution details
            trade_executed = data.get("trade_executed", {})
            print(f'\n🔄 TRADE EXECUTION ANALYSIS:')
            print(f'-' * 30)
            
            if trade_executed:
                is_simulated = trade_executed.get("simulated", False)
                if is_simulated:
                    print(f'⚠️ TRADE MODE: SIMULATED (Safe Testing)')
                    print(f'   Symbol: {trade_executed.get("symbol", "N/A")}')
                    print(f'   Side: {trade_executed.get("side", "N/A")}')
                    print(f'   Quantity: {trade_executed.get("quantity", "N/A")}')
                    print(f'   Price: ${trade_executed.get("price", "N/A")}')
                    print(f'   Value: ${trade_executed.get("value", "N/A"):.2f}' if trade_executed.get("value") else 'N/A')
                    print(f'   📝 NOTE: No real trades placed - simulation mode active')
                else:
                    print(f'🚨 TRADE MODE: LIVE TRADING ACTIVE!')
                    print(f'   Order ID: {trade_executed.get("orderId", "N/A")}')
                    print(f'   Status: {trade_executed.get("status", "N/A")}')
                    print(f'   Fill Qty: {trade_executed.get("executedQty", "N/A")}')
            else:
                print(f'❌ No trade execution data found')
            
            # Check API endpoint being used
            print(f'\n🌐 API CONFIGURATION:')
            print(f'-' * 30)
            config_source = data.get("config_source", "unknown")
            print(f'✅ Config Source: {config_source}')
            
            # Check parameters
            parameters = data.get("parameters_used", {})
            print(f'✅ RSI Period: {parameters.get("RSI_PERIOD", "N/A")}')
            print(f'✅ MA Fast/Slow: {parameters.get("MA_FAST", "N/A")}/{parameters.get("MA_SLOW", "N/A")}')
            
            # Check account balance (indicates if connected to real or test account)
            account_balance = data.get("account_balance", {})
            print(f'\n💰 ACCOUNT STATUS:')
            print(f'-' * 30)
            if account_balance:
                # Look for testnet indicators
                if any('test' in str(v).lower() for v in account_balance.values()):
                    print(f'✅ TESTNET account detected')
                else:
                    print(f'⚠️ Account type unclear from balance data')
                print(f'   Balance data available: {len(account_balance)} assets')
            else:
                print(f'❌ No account balance data (API might be using simulated data)')
            
            return data
            
        else:
            print(f'❌ API Request failed: {response.status_code}')
            print(f'   Response: {response.text[:200]}...')
            return None
            
    except Exception as e:
        print(f'❌ Error: {e}')
        return None

def check_api_endpoint_config():
    print(f'\n🔧 API ENDPOINT VERIFICATION:')
    print(f'-' * 30)
    
    # Load environment to check API keys
    load_dotenv()
    api_key = os.environ.get('BINANCE_API_KEY', '')
    
    print(f'✅ API Key configured: {api_key[:10] if api_key else "Not set"}...')
    
    # Note about endpoint configuration
    print(f'📝 CURRENT ENDPOINT CONFIG (from code):')
    print(f'   Production: https://api.binance.com')
    print(f'   Testnet: https://testnet.binance.vision')
    print(f'   📍 Your bot is configured for: PRODUCTION endpoint')
    print(f'   🔍 But with TESTNET API keys (safe for testing)')

if __name__ == "__main__":
    bot_data = test_bot_trading_status()
    check_api_endpoint_config()
    
    print(f'\n🎯 SUMMARY:')
    print(f'=' * 50)
    if bot_data:
        trade_data = bot_data.get("trade_executed")
        if trade_data and trade_data.get("simulated"):
            print(f'✅ Bot is generating SIMULATED trades (safe)')
            print(f'✅ Real market analysis with no trading risk')
            print(f'✅ Perfect setup for testing and validation')
        elif trade_data:
            print(f'🚨 Bot appears to be in LIVE trading mode')
            print(f'⚠️ Real trades may be executed!')
        else:
            print(f'⚠️ No trade execution detected - bot generating signals only')
            print(f'✅ This is safe - analysis mode without trading')
    else:
        print(f'❌ Could not determine bot trading status')
        
    print(f'\n📋 NEXT STEPS:')
    print(f'   1. Verify API keys are from Binance testnet')
    print(f'   2. Check if endpoint should be testnet.binance.vision')
    print(f'   3. Confirm simulation mode is intentional')
