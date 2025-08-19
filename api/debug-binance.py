"""
Debug endpoint to check Binance API response on Vercel
"""

import json
import requests
import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Test Binance API from Vercel
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 5},
                timeout=10
            )
            
            result = {
                "status_code": response.status_code,
                "response_type": str(type(response.json())),
                "response_length": len(response.json()) if isinstance(response.json(), list) else "N/A",
                "sample_data": response.json()[:2] if isinstance(response.json(), list) else response.json(),
                "full_response": response.text[:500]  # First 500 chars
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "error": str(e),
                "error_type": type(e).__name__
            }
            self.wfile.write(json.dumps(error_response).encode())
