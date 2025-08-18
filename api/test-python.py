"""
Simple API test to verify Python execution on Vercel
"""

import json
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Simple test response
            response_data = {
                "status": "success",
                "message": "Python is working on Vercel!",
                "timestamp": "2025-08-18",
                "python_version": "3.12",
                "test_data": {
                    "btc_price": 116142.79,
                    "system": "vercel_serverless",
                    "api_status": "operational"
                }
            }
            
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # Send response
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            # Error response
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": f"Python error: {str(e)}"
            }
            self.wfile.write(json.dumps(error_response).encode())
