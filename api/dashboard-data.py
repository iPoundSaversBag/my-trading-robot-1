"""
Public Dashboard Data API - No authentication required
Provides basic market data and trading status for the dashboard
"""

import json
import requests
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for dashboard data"""
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-cache, max-age=0')
            self.end_headers()
            
            # Get live BTC data from Binance API
            btc_data = self.get_live_btc_data()
            
            # Create simplified response for dashboard
            response_data = {
                "status": "success",
                "timestamp": int(time.time()),
                "signal": {
                    "signal": "HOLD",
                    "confidence": 0.65,
                    "current_price": btc_data["price"],
                    "rsi": 45.2,
                    "trend": "NEUTRAL"
                },
                "market_regime": {
                    "regime": "NORMAL",
                    "volatility": "MEDIUM",
                    "trend_strength": 0.6
                },
                "account_balance": {
                    "USDT": 10000.0,
                    "BTC": 0.1
                },
                "trade_executed": {
                    "simulated": True,
                    "side": "HOLD",
                    "quantity": 0,
                    "price": btc_data["price"],
                    "value": 0
                },
                "execution_mode": "dashboard_display"
            }
            
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            error_response = {
                "error": f"Dashboard data failed: {str(e)}",
                "status": "error"
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def get_live_btc_data(self):
        """Get current BTC data from Binance API"""
        try:
            # Get 24hr ticker statistics
            ticker_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
            ticker_response = requests.get(ticker_url, timeout=10)
            ticker_data = ticker_response.json()
            
            # Get current order book for bid/ask
            depth_url = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5"
            depth_response = requests.get(depth_url, timeout=10)
            depth_data = depth_response.json()
            
            return {
                "price": float(ticker_data["lastPrice"]),
                "volume": float(ticker_data["volume"]),
                "high": float(ticker_data["highPrice"]),
                "low": float(ticker_data["lowPrice"]),
                "change_percent": float(ticker_data["priceChangePercent"]),
                "bid": float(depth_data["bids"][0][0]) if depth_data.get("bids") else float(ticker_data["lastPrice"]),
                "ask": float(depth_data["asks"][0][0]) if depth_data.get("asks") else float(ticker_data["lastPrice"])
            }
            
        except Exception as e:
            # Return fallback data if API fails
            return {
                "price": 45000.0,
                "volume": 25000.0,
                "high": 46000.0,
                "low": 44000.0,
                "change_percent": 2.5,
                "bid": 44995.0,
                "ask": 45005.0
            }
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
