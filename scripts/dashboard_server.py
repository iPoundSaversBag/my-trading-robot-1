#!/usr/bin/env python3
"""
Trading Dashboard HTTP Server
Serves the trading dashboard with CORS support for local development.
Allows JavaScript to properly fetch live trading data files.
"""

import http.server
import socketserver
import os
import sys
import json
import logging
from urllib.parse import urlparse
from datetime import datetime

class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS support and logging"""
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests with enhanced logging"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        path = self.path.split('?')[0]  # Remove query parameters for logging
        
        # Log the request
        if path.endswith('.json'):
            print(f"[{timestamp}] 📊 DATA REQUEST: {path}")
        elif path.endswith('.html'):
            print(f"[{timestamp}] 🌐 PAGE REQUEST: {path}")
        elif path == '/':
            print(f"[{timestamp}] 🏠 ROOT REQUEST")
        else:
            print(f"[{timestamp}] 📁 FILE REQUEST: {path}")
        
        # Serve the file
        super().do_GET()
    
    def log_message(self, format, *args):
        """Override default logging - keep it simple"""
        # Simplified logging - just skip noisy request logs
        # Errors will still be shown by the exception handler
        pass

class DashboardServer:
    """Trading Dashboard HTTP Server Manager"""
    
    def __init__(self, port=8080):
        self.port = port
        self.server = None
        self.server_dir = None
    
    def find_project_root(self):
        """Find the project root directory"""
        current = os.getcwd()
        
        # Look for key files that indicate project root
        indicators = ['data', 'live_trading', 'plots_output', 'core']
        
        while current != os.path.dirname(current):  # Not at filesystem root
            if all(os.path.exists(os.path.join(current, indicator)) for indicator in indicators):
                return current
            current = os.path.dirname(current)
        
        # Fallback to current directory
        return os.getcwd()
    
    def start(self):
        """Start the HTTP server"""
        # Change to project root directory
        self.server_dir = self.find_project_root()
        original_dir = os.getcwd()
        os.chdir(self.server_dir)
        
        print("\n" + "="*60)
        print("🚀 TRADING DASHBOARD SERVER")
        print("="*60)
        print(f"📁 Serving from: {self.server_dir}")
        print(f"🌐 Server URL: http://localhost:{self.port}")
        print(f"📊 Dashboard URL: http://localhost:{self.port}/plots_output/20250817_133240/performance_report.html")
        print(f"📡 Live Data URL: http://localhost:{self.port}/data/live_bot_state.json")
        print("="*60)
        
        try:
            # Create server
            with socketserver.TCPServer(("", self.port), DashboardHTTPRequestHandler) as httpd:
                self.server = httpd
                print(f"✅ Server started successfully on port {self.port}")
                print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n💡 USAGE:")
                print("   • Open browser to: http://localhost:8080")
                print("   • Navigate to dashboard via the plots_output folder")
                print("   • Live data will update automatically every 5 seconds")
                print("   • Press Ctrl+C to stop server")
                print("\n🔄 Server is running... (watching for requests)")
                print("-" * 60)
                
                # Start serving
                httpd.serve_forever()
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Server stopped at: {datetime.now().strftime('%H:%M:%S')}")
            print("👋 Dashboard server shutdown complete")
        except OSError as e:
            if e.errno == 10048:  # Port already in use
                print(f"❌ ERROR: Port {self.port} is already in use")
                print(f"💡 Try a different port or stop the other server")
                print(f"   Example: python dashboard_server.py --port 8081")
            else:
                print(f"❌ ERROR: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")
        finally:
            # Restore original directory
            os.chdir(original_dir)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Dashboard HTTP Server')
    parser.add_argument('--port', type=int, default=8080, 
                       help='Port to serve on (default: 8080)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce logging output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔧 Initializing Trading Dashboard Server...")
    
    server = DashboardServer(port=args.port)
    server.start()

if __name__ == '__main__':
    main()
