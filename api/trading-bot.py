"""
Live Trading Bot - Vercel Serverless Function
Main trading engine that executes trades based on strategy signals
"""

import json
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from http.server import BaseHTTPRequestHandler

# Import your core trading components
import sys
sys.path.append('/var/task')  # Vercel function path

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests for trade execution"""
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # Get request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Initialize Binance client (use environment variables for API keys)
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise Exception("Binance API credentials not configured")
                
            client = Client(api_key, api_secret, testnet=True)  # Start with testnet
            
            # Execute trading logic
            result = self.execute_trading_cycle(client, request_data)
            
            # Return response
            response_data = {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result
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
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def execute_trading_cycle(self, client, request_data):
        """
        Execute a complete trading cycle:
        1. Fetch market data
        2. Calculate signals
        3. Execute trades if needed
        4. Update positions
        """
        
        # Get current market data
        symbol = request_data.get('symbol', 'BTCUSDT')
        timeframe = request_data.get('timeframe', '5m')
        
        # Fetch latest price data
        klines = client.get_historical_klines(symbol, timeframe, "100")
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate trading signals (simplified version)
        signal = self.calculate_trading_signal(df)
        
        # Get current positions
        account = client.get_account()
        current_position = self.get_current_position(account, symbol)
        
        # Execute trades based on signals
        trade_result = None
        if signal['action'] != 'HOLD':
            trade_result = self.execute_trade(client, symbol, signal, current_position)
        
        return {
            "signal": signal,
            "current_position": current_position,
            "trade_executed": trade_result,
            "account_balance": float(account['totalWalletBalance']) if 'totalWalletBalance' in account else None,
            "market_price": float(df['close'].iloc[-1])
        }
    
    def calculate_trading_signal(self, df):
        """
        Calculate trading signals based on technical indicators
        Simplified version of your strategy logic
        """
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Calculate moving averages
        df['ma_short'] = df['close'].rolling(window=10).mean()
        df['ma_long'] = df['close'].rolling(window=20).mean()
        
        # Get latest values
        current_rsi = df['rsi'].iloc[-1]
        current_price = df['close'].iloc[-1]
        ma_short = df['ma_short'].iloc[-1]
        ma_long = df['ma_long'].iloc[-1]
        
        # Generate signals
        signal = {
            "action": "HOLD",
            "confidence": 0.0,
            "price": float(current_price),
            "indicators": {
                "rsi": float(current_rsi),
                "ma_short": float(ma_short),
                "ma_long": float(ma_long)
            }
        }
        
        # Buy signal: RSI oversold and MA crossover
        if current_rsi < 30 and ma_short > ma_long:
            signal["action"] = "BUY"
            signal["confidence"] = 0.8
        
        # Sell signal: RSI overbought and MA crossover down
        elif current_rsi > 70 and ma_short < ma_long:
            signal["action"] = "SELL"
            signal["confidence"] = 0.8
        
        return signal
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_current_position(self, account, symbol):
        """Get current position for the symbol"""
        base_asset = symbol.replace('USDT', '')
        
        for balance in account['balances']:
            if balance['asset'] == base_asset:
                return {
                    "asset": base_asset,
                    "free": float(balance['free']),
                    "locked": float(balance['locked']),
                    "total": float(balance['free']) + float(balance['locked'])
                }
        
        return {"asset": base_asset, "free": 0.0, "locked": 0.0, "total": 0.0}
    
    def execute_trade(self, client, symbol, signal, current_position):
        """
        Execute trade based on signal
        THIS IS A SIMPLIFIED VERSION - Add your position sizing and risk management
        """
        
        try:
            if signal['action'] == 'BUY' and current_position['total'] == 0:
                # Calculate position size (simplified - 1% of account)
                account = client.get_account()
                usdt_balance = 0
                
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                        break
                
                if usdt_balance > 10:  # Minimum $10 trade
                    quantity = (usdt_balance * 0.01) / signal['price']  # 1% position
                    
                    # Round quantity to valid precision
                    quantity = round(quantity, 6)
                    
                    # Place market buy order
                    order = client.order_market_buy(
                        symbol=symbol,
                        quoteOrderQty=usdt_balance * 0.01
                    )
                    
                    return {
                        "action": "BUY",
                        "order_id": order['orderId'],
                        "status": order['status'],
                        "quantity": quantity
                    }
            
            elif signal['action'] == 'SELL' and current_position['total'] > 0:
                # Sell all position
                quantity = current_position['free']
                
                if quantity > 0:
                    order = client.order_market_sell(
                        symbol=symbol,
                        quantity=quantity
                    )
                    
                    return {
                        "action": "SELL", 
                        "order_id": order['orderId'],
                        "status": order['status'],
                        "quantity": quantity
                    }
            
            return None
            
        except BinanceAPIException as e:
            return {
                "error": f"Binance API Error: {e}",
                "action": signal['action']
            }
        except Exception as e:
            return {
                "error": f"Trade execution error: {e}",
                "action": signal['action']
            }
