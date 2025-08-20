#!/usr/bin/env python3
import os
from dotenv import load_dotenv
load_dotenv()

print('🔍 CHECKING ENVIRONMENT CONFIGURATION:')
print('==========================================')

api_key = os.environ.get('BINANCE_API_KEY', '')
print(f'Binance API Key: {api_key[:10] if api_key else "Not set"}...')

# Check if testnet keys
if 'test' in api_key.lower():
    print('⚠️ TESTNET API key detected')
else:
    print('✅ Production API key format')

print()
print('🌐 API ENDPOINTS:')
print('Main Binance: https://api.binance.com')
print('Testnet: https://testnet.binance.vision')

# Check the base URL being used
print()
print('📊 CURRENT BOT CONFIGURATION:')
print('From live-bot.py: base_url = "https://api.binance.com"')
print('This is the PRODUCTION Binance API endpoint')
