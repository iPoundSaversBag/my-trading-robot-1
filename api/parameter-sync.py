"""
Parameter Sync API for Vercel
Receives optimized parameters from local backtest and updates live bot config
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from datetime import datetime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Get current live trading configuration"""
        try:
            # Load current config if exists
            config_path = 'live_trading_config.json'
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Return default config
                config = {
                    "SYMBOL": "BTCUSDT",
                    "TIMEFRAME": "5m",
                    "RSI_PERIOD": 14,
                    "MA_FAST": 12,
                    "MA_SLOW": 26,
                    "updated_at": None,
                    "source": "default"
                }
            
            response_data = {
                "status": "success",
                "config": config,
                "last_updated": config.get("updated_at"),
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        """Update live trading configuration with optimized parameters"""
        try:
            # Get the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            
            # Validate required fields
            if 'config' not in data:
                raise ValueError("Missing 'config' field in request")
            
            # Add metadata
            config = data['config']
            config.update({
                "updated_at": datetime.now().isoformat(),
                "source": "automated_backtest_sync",
                "sync_method": "api_post"
            })
            
            # Save the updated config
            with open('live_trading_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            response_data = {
                "status": "success",
                "message": "Configuration updated successfully",
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            error_response = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
