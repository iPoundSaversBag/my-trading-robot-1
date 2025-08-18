"""
Lightweight Trading Engine for Vercel - No heavy dependencies
Trading signals and execution using pure Python math
"""

import json
import os
import requests
import hashlib
import hmac
import time
import math
from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler

class VercelTrader:
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
    
    def _make_request(self, endpoint, params=None, method='GET'):
        """Make authenticated request to Binance API"""
        if not params:
            params = {}
        
        params['timestamp'] = int(time.time() * 1000)
        signature = self._create_signature(params)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            if method == 'POST':
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers,
                    timeout=10
                )
            else:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers,
                    timeout=10
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_klines(self, symbol='BTCUSDT', interval='5m', limit=100):
        """Get candlestick data"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params={
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                },
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI using simple math"""
        if len(prices) < period + 1:
            return 50  # Default RSI
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [max(0, delta) for delta in deltas]
        losses = [abs(min(0, delta)) for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def get_trading_signals(self, symbol='BTCUSDT'):
        """Calculate trading signals"""
        klines = self.get_klines(symbol, '5m', 100)
        
        if "error" in klines:
            return {"error": "Failed to get market data", "details": klines["error"]}
        
        # Extract closing prices
        closes = [float(kline[4]) for kline in klines]
        current_price = closes[-1]
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes, 14)
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        
        # Calculate MACD
        macd_line = ema_12 - ema_26
        
        # Generate signals
        signals = {
            'rsi_signal': 'NEUTRAL',
            'ma_signal': 'NEUTRAL',
            'price_signal': 'NEUTRAL',
            'overall_signal': 'HOLD'
        }
        
        # RSI signals
        if rsi < 30:
            signals['rsi_signal'] = 'BUY'
        elif rsi > 70:
            signals['rsi_signal'] = 'SELL'
        
        # Moving average signals
        if current_price > sma_20 and sma_20 > sma_50:
            signals['ma_signal'] = 'BUY'
        elif current_price < sma_20 and sma_20 < sma_50:
            signals['ma_signal'] = 'SELL'
        
        # Price action signals
        if current_price > closes[-2] > closes[-3]:  # Rising trend
            signals['price_signal'] = 'BUY'
        elif current_price < closes[-2] < closes[-3]:  # Falling trend
            signals['price_signal'] = 'SELL'
        
        # Overall signal logic
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        
        if buy_signals >= 2:
            signals['overall_signal'] = 'BUY'
        elif sell_signals >= 2:
            signals['overall_signal'] = 'SELL'
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'indicators': {
                'rsi': round(rsi, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'macd': round(macd_line, 4)
            },
            'signals': signals,
            'timestamp': int(time.time())
        }
    
    def place_order(self, symbol, side, quantity, order_type='MARKET'):
        """Place a trading order"""
        params = {
            'symbol': symbol,
            'side': side,  # BUY or SELL
            'type': order_type,
            'quantity': quantity
        }
        
        return self._make_request('/api/v3/order', params, 'POST')
    
    def get_account_balance(self):
        """Get account balance"""
        account = self._make_request('/api/v3/account')
        if "error" in account:
            return {"error": "Failed to get account info"}
        
        balances = {}
        for balance in account.get('balances', []):
            free = float(balance['free'])
            if free > 0:
                balances[balance['asset']] = free
        
        return balances

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse query parameters
            if '?' in self.path:
                path, query = self.path.split('?', 1)
                params = dict(param.split('=') for param in query.split('&') if '=' in param)
            else:
                params = {}
            
            trader = VercelTrader()
            
            # Get trading signals
            symbol = params.get('symbol', 'BTCUSDT')
            signals_data = trader.get_trading_signals(symbol)
            
            if "error" in signals_data:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(signals_data).encode())
                return
            
            # Get account balance
            balance_data = trader.get_account_balance()
            
            response = {
                "status": "success",
                "trading_signals": signals_data,
                "account_balance": balance_data,
                "timestamp": int(time.time()),
                "source": "vercel_trading_engine"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Trading engine failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        """Handle trade execution requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            trade_request = json.loads(post_data.decode('utf-8'))
            
            trader = VercelTrader()
            
            # Execute trade
            result = trader.place_order(
                symbol=trade_request.get('symbol', 'BTCUSDT'),
                side=trade_request.get('side', 'BUY'),
                quantity=trade_request.get('quantity', '0.001'),
                order_type=trade_request.get('type', 'MARKET')
            )
            
            response = {
                "status": "success" if "error" not in result else "error",
                "trade_result": result,
                "timestamp": int(time.time())
            }
            
            status_code = 200 if "error" not in result else 400
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Trade execution failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())
