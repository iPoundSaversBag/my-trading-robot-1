"""
Lightweight Automated Trading Scheduler for Vercel
Runs trading signals and executes trades automatically
"""

import json
import os
import requests
import time
from http.server import BaseHTTPRequestHandler

class VercelScheduler:
    def __init__(self):
        self.cron_secret = os.environ.get('CRON_SECRET', '')
        self.base_url = os.environ.get('VERCEL_URL', 'https://my-trading-robot-1-89y7ohkh2-aidan-lanes-projects.vercel.app')
    
    def get_trading_signals(self, symbol='BTCUSDT'):
        """Get signals from trading engine"""
        try:
            response = requests.get(
                f"{self.base_url}/api/trading-engine?symbol={symbol}",
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def execute_trade(self, trade_data):
        """Execute a trade via trading engine"""
        try:
            response = requests.post(
                f"{self.base_url}/api/trading-engine",
                json=trade_data,
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_position_size(self, balance_usdt, risk_percent=1.0):
        """Calculate position size based on account balance"""
        max_risk_amount = balance_usdt * (risk_percent / 100)
        return max(0.001, min(max_risk_amount / 100, 0.01))  # Min 0.001, max based on risk
    
    def run_trading_cycle(self):
        """Execute one trading cycle"""
        results = {
            'timestamp': int(time.time()),
            'cycles_executed': 0,
            'trades_placed': 0,
            'errors': []
        }
        
        # Trading pairs to monitor
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in symbols:
            try:
                # Get trading signals
                signals_response = self.get_trading_signals(symbol)
                
                if "error" in signals_response:
                    results['errors'].append(f"{symbol}: {signals_response['error']}")
                    continue
                
                signals = signals_response.get('trading_signals', {})
                overall_signal = signals.get('signals', {}).get('overall_signal', 'HOLD')
                current_price = signals.get('current_price', 0)
                
                # Get account balance
                account_balance = signals_response.get('account_balance', {})
                usdt_balance = account_balance.get('USDT', 0)
                
                # Check if we should trade
                if overall_signal in ['BUY', 'SELL'] and usdt_balance > 10:  # Min $10 balance
                    
                    # Calculate position size
                    quantity = self.calculate_position_size(usdt_balance, 1.0)  # 1% risk
                    
                    # Prepare trade data
                    trade_data = {
                        'symbol': symbol,
                        'side': overall_signal,
                        'quantity': str(quantity),
                        'type': 'MARKET'
                    }
                    
                    # Execute trade (commented out for safety - uncomment when ready)
                    # trade_result = self.execute_trade(trade_data)
                    # if "error" not in trade_result:
                    #     results['trades_placed'] += 1
                    
                    # For now, just log the signal
                    results[f'{symbol}_signal'] = {
                        'action': overall_signal,
                        'price': current_price,
                        'quantity': quantity,
                        'status': 'SIMULATED'  # Change to 'EXECUTED' when live
                    }
                
                results['cycles_executed'] += 1
                
            except Exception as e:
                results['errors'].append(f"{symbol}: {str(e)}")
        
        return results

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Verify cron secret for security
            cron_secret = os.environ.get('CRON_SECRET', '')
            provided_secret = self.headers.get('Authorization', '').replace('Bearer ', '')
            
            if cron_secret and provided_secret != cron_secret:
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                error_response = {
                    "status": "error",
                    "message": "Unauthorized - Invalid cron secret"
                }
                self.wfile.write(json.dumps(error_response).encode())
                return
            
            # Run trading cycle
            scheduler = VercelScheduler()
            cycle_results = scheduler.run_trading_cycle()
            
            response = {
                "status": "success",
                "scheduler_results": cycle_results,
                "message": "Automated trading cycle completed",
                "timestamp": int(time.time()),
                "source": "vercel_scheduler"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Scheduler failed: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())
