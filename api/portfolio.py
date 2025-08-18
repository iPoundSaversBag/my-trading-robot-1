"""
Portfolio Status API - Vercel Serverless Function
Get real-time portfolio status, positions, and performance metrics
"""

import json
import os
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for portfolio status"""
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # Initialize Binance client
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise Exception("Binance API credentials not configured")
                
            client = Client(api_key, api_secret, testnet=True)
            
            # Get portfolio data
            portfolio_data = self.get_portfolio_status(client)
            
            # Return response
            response_data = {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio": portfolio_data
            }
            
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_portfolio_status(self, client):
        """Get comprehensive portfolio status"""
        
        # Get account information
        account = client.get_account()
        
        # Get current prices for major coins
        tickers = client.get_all_tickers()
        price_map = {ticker['symbol']: float(ticker['price']) for ticker in tickers}
        
        # Calculate portfolio value
        total_value_usdt = 0
        positions = []
        
        for balance in account['balances']:
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:
                if asset == 'USDT':
                    value_usdt = total
                else:
                    # Try to get price in USDT
                    symbol = f"{asset}USDT"
                    if symbol in price_map:
                        value_usdt = total * price_map[symbol]
                    else:
                        value_usdt = 0  # Can't price this asset
                
                total_value_usdt += value_usdt
                
                if value_usdt > 1:  # Only include significant positions
                    positions.append({
                        "asset": asset,
                        "free": free,
                        "locked": locked,
                        "total": total,
                        "current_price": price_map.get(f"{asset}USDT", 0),
                        "value_usdt": value_usdt,
                        "percentage": 0  # Will calculate after total
                    })
        
        # Calculate percentages
        for position in positions:
            position["percentage"] = (position["value_usdt"] / total_value_usdt) * 100 if total_value_usdt > 0 else 0
        
        # Sort positions by value
        positions.sort(key=lambda x: x["value_usdt"], reverse=True)
        
        # Get recent trades for P&L calculation
        recent_trades = self.get_recent_performance(client)
        
        return {
            "total_value_usdt": total_value_usdt,
            "positions": positions,
            "account_info": {
                "maker_commission": account.get('makerCommission', 0),
                "taker_commission": account.get('takerCommission', 0),
                "can_trade": account.get('canTrade', False),
                "can_withdraw": account.get('canWithdraw', False),
                "can_deposit": account.get('canDeposit', False)
            },
            "performance": recent_trades,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
    
    def get_recent_performance(self, client):
        """Get recent trading performance"""
        try:
            # Get recent trades for BTCUSDT (main trading pair)
            trades = client.get_my_trades(symbol='BTCUSDT', limit=10)
            
            total_commission = 0
            total_volume = 0
            trade_count = len(trades)
            
            for trade in trades:
                total_commission += float(trade['commission'])
                total_volume += float(trade['quoteQty'])
            
            return {
                "recent_trades_count": trade_count,
                "total_commission": total_commission,
                "total_volume": total_volume,
                "avg_trade_size": total_volume / trade_count if trade_count > 0 else 0
            }
            
        except Exception as e:
            return {
                "error": f"Could not fetch trade history: {e}",
                "recent_trades_count": 0,
                "total_commission": 0,
                "total_volume": 0
            }
