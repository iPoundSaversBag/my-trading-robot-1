"""
Scheduled Trading Function - Vercel Cron Job
Runs every 5 minutes to check for trading signals and execute trades
"""

import json
import os
from datetime import datetime, timezone
import requests
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle cron job execution"""
        try:
            # Verify this is a legitimate cron request
            cron_secret = os.environ.get('CRON_SECRET')
            provided_secret = self.headers.get('Authorization', '').replace('Bearer ', '')
            
            if cron_secret and provided_secret != cron_secret:
                self.send_response(401)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unauthorized"}).encode())
                return
            
            # Execute trading cycle
            result = self.execute_scheduled_trading()
            
            # Return success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response_data = {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_result": result
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
    
    def execute_scheduled_trading(self):
        """Execute the main trading logic"""
        
        # Get the base URL of this deployment
        base_url = os.environ.get('VERCEL_URL', 'localhost:3000')
        if not base_url.startswith('http'):
            base_url = f"https://{base_url}"
        
        results = []
        
        # List of trading pairs to monitor
        trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in trading_pairs:
            try:
                # Call the trading bot function for each pair
                trading_data = {
                    "symbol": symbol,
                    "timeframe": "5m",
                    "automated": True
                }
                
                # Make internal API call to trading bot
                response = requests.post(
                    f"{base_url}/api/trading-bot",
                    json=trading_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    trade_result = response.json()
                    results.append({
                        "symbol": symbol,
                        "status": "success",
                        "result": trade_result
                    })
                else:
                    results.append({
                        "symbol": symbol,
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error", 
                    "error": str(e)
                })
        
        # Log execution summary
        successful_executions = sum(1 for r in results if r["status"] == "success")
        
        return {
            "pairs_processed": len(trading_pairs),
            "successful_executions": successful_executions,
            "failed_executions": len(trading_pairs) - successful_executions,
            "results": results,
            "next_execution": self.calculate_next_execution()
        }
    
    def calculate_next_execution(self):
        """Calculate next scheduled execution time"""
        from datetime import timedelta
        
        now = datetime.now(timezone.utc)
        next_execution = now + timedelta(minutes=5)
        
        return next_execution.isoformat()
