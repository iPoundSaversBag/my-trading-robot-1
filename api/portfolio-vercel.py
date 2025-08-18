"""
Lightweight Portfolio API for Vercel - No heavy dependencies
Real-time portfolio tracking using Binance API
"""

import json
import os
import requests
import hashlib
import hmac
import time
from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler

class VercelPortfolio:
    def __init__(self):
        self.api_key = os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_API_SECRET', '')
        self.base_url = 'https://api.binance.com'
    
    def _create_signature(self, params):
        """Create HMAC SHA256 signature for Binance API"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint, params=None):
        """Make authenticated request to Binance API"""
        if not params:
            params = {}
        
        params['timestamp'] = int(time.time() * 1000)
        signature = self._create_signature(params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_account_info(self):
        """Get account balances and information"""
        return self._make_request('/api/v3/account')
    
    def get_current_prices(self, symbols):
        """Get current prices for symbols"""
        try:
            if isinstance(symbols, list):
                symbols_str = '["' + '","'.join(symbols) + '"]'
            else:
                symbols_str = f'["{symbols}"]'
            
            response = requests.get(
                f"{self.base_url}/api/v3/ticker/price",
                params={'symbols': symbols_str},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_portfolio(self):
        """Calculate portfolio value and positions"""
        account = self.get_account_info()
        
        if "error" in account:
            return {"error": "Failed to get account info", "details": account["error"]}
        
        # Get non-zero balances
        balances = []
        symbols_for_pricing = []
        
        for balance in account.get('balances', []):
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0.001:  # Only include significant balances
                balances.append({
                    'asset': balance['asset'],
                    'free': free,
                    'locked': locked,
                    'total': total
                })
                
                if balance['asset'] != 'USDT':
                    symbols_for_pricing.append(f"{balance['asset']}USDT")
        
        # Get current prices
        prices_data = self.get_current_prices(symbols_for_pricing)
        if "error" in prices_data:
            return {"error": "Failed to get prices", "details": prices_data["error"]}
        
        # Create price lookup
        prices = {}
        if isinstance(prices_data, list):
            for price_info in prices_data:
                symbol = price_info['symbol']
                asset = symbol.replace('USDT', '')
                prices[asset] = float(price_info['price'])
        
        # Calculate portfolio
        total_value_usdt = 0
        positions = []
        
        for balance in balances:
            asset = balance['asset']
            total_amount = balance['total']
            
            if asset == 'USDT':
                value_usdt = total_amount
                current_price = 1.0
            else:
                current_price = prices.get(asset, 0)
                value_usdt = total_amount * current_price
            
            if value_usdt > 1.0:  # Only include positions worth more than $1
                positions.append({
                    'asset': asset,
                    'amount': total_amount,
                    'current_price': current_price,
                    'value_usdt': value_usdt,
                    'percentage': 0  # Will calculate after total
                })
                total_value_usdt += value_usdt
        
        # Calculate percentages
        for position in positions:
            position['percentage'] = (position['value_usdt'] / total_value_usdt) * 100 if total_value_usdt > 0 else 0
        
        return {
            'total_value_usdt': total_value_usdt,
            'positions': positions,
            'account_type': account.get('accountType', 'SPOT'),
            'can_trade': account.get('canTrade', False),
            'update_time': account.get('updateTime', 0)
        }

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            portfolio = VercelPortfolio()
            portfolio_data = portfolio.calculate_portfolio()
            
            if "error" in portfolio_data:
                # Return error response
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(portfolio_data).encode())
                return
            
            # Success response
            response = {
                "status": "success",
                "portfolio": portfolio_data,
                "timestamp": int(time.time()),
                "source": "vercel_serverless"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            # Error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Portfolio calculation failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())
