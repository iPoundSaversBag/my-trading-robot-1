#!/usr/bin/env python3
"""
API Utilities Module - Consolidated API Support Functions
Combines utility functionality for Vercel API endpoints.

Consolidated from:
- debug-binance.py (Binance API debugging)
- parameter-sync.py (Parameter synchronization)
- test-python.py (Python test endpoint)
- dashboard-integration.py (Dashboard integration utilities)

Purpose: Unified API utilities for debugging, testing, configuration,
and dashboard integration in serverless environment.
"""

import json
import requests
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CORE API UTILITIES
# ============================================================================

class APIUtilities:
    """Core API utility functions for serverless endpoints."""
    
    @staticmethod
    def send_json_response(handler: BaseHTTPRequestHandler, 
                          data: Dict[str, Any], 
                          status_code: int = 200,
                          cors_enabled: bool = True) -> None:
        """Send standardized JSON response with CORS headers."""
        handler.send_response(status_code)
        handler.send_header('Content-Type', 'application/json')
        
        if cors_enabled:
            handler.send_header('Access-Control-Allow-Origin', '*')
            handler.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            handler.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        handler.end_headers()
        handler.wfile.write(json.dumps(data, indent=2).encode())
    
    @staticmethod
    def send_error_response(handler: BaseHTTPRequestHandler, 
                           error: Exception, 
                           status_code: int = 500) -> None:
        """Send standardized error response."""
        error_data = {
            "status": "error",
            "message": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat()
        }
        APIUtilities.send_json_response(handler, error_data, status_code)
    
    @staticmethod
    def load_config_file(file_path: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration file with fallback to defaults."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return default_config
    
    @staticmethod
    def save_config_file(file_path: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration file safely."""
        try:
            config_data["updated_at"] = datetime.now().isoformat()
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception:
            return False

# ============================================================================
# BINANCE API DEBUGGING
# ============================================================================

class BinanceDebugger:
    """Binance API debugging utilities for serverless environment."""
    
    @staticmethod
    def test_binance_api(symbol: str = 'BTCUSDT', 
                        interval: str = '5m', 
                        limit: int = 5) -> Dict[str, Any]:
        """Test Binance API connectivity and data retrieval."""
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=10
            )
            
            data = response.json()
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "response_type": str(type(data)),
                "response_length": len(data) if isinstance(data, list) else "N/A",
                "sample_data": data[:2] if isinstance(data, list) else data,
                "full_response": response.text[:500],  # First 500 chars
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_server_time() -> Dict[str, Any]:
        """Get Binance server time for clock sync verification."""
        try:
            response = requests.get(
                "https://api.binance.com/api/v3/time",
                timeout=5
            )
            
            server_data = response.json()
            server_time = datetime.fromtimestamp(server_data['serverTime'] / 1000, timezone.utc)
            local_time = datetime.now(timezone.utc)
            
            return {
                "status": "success",
                "server_time": server_time.isoformat(),
                "local_time": local_time.isoformat(),
                "time_diff_ms": abs((server_time - local_time).total_seconds() * 1000),
                "sync_status": "good" if abs((server_time - local_time).total_seconds()) < 5 else "warning"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# PARAMETER SYNCHRONIZATION
# ============================================================================

class ParameterSync:
    """Parameter synchronization utilities for live trading configuration."""
    
    CONFIG_FILE = 'live_trading_config.json'
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default trading configuration."""
        return {
            "SYMBOL": "BTCUSDT",
            "TIMEFRAME": "5m",
            "RSI_PERIOD": 14,
            "MA_FAST": 12,
            "MA_SLOW": 26,
            "updated_at": None,
            "source": "default"
        }
    
    @staticmethod
    def get_current_config() -> Dict[str, Any]:
        """Get current live trading configuration."""
        config = APIUtilities.load_config_file(
            ParameterSync.CONFIG_FILE, 
            ParameterSync.get_default_config()
        )
        
        return {
            "status": "success",
            "config": config,
            "last_updated": config.get("updated_at"),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def update_config(new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading configuration with new parameters."""
        try:
            # Load current config
            current_config = APIUtilities.load_config_file(
                ParameterSync.CONFIG_FILE, 
                ParameterSync.get_default_config()
            )
            
            # Update with new parameters
            current_config.update(new_params)
            current_config["source"] = "parameter_sync"
            
            # Save updated config
            success = APIUtilities.save_config_file(ParameterSync.CONFIG_FILE, current_config)
            
            if success:
                return {
                    "status": "success",
                    "message": "Configuration updated successfully",
                    "updated_config": current_config,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save configuration",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ============================================================================
# DASHBOARD INTEGRATION
# ============================================================================

class DashboardIntegration:
    """Dashboard integration utilities for live data feeds."""
    
    @staticmethod
    def get_live_trading_summary() -> Dict[str, Any]:
        """Get comprehensive trading summary for dashboard integration."""
        try:
            # Load live results if available
            results_file = "live_trading/live_results.json"
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                cycles = data.get("trading_cycles", [])
                metadata = data.get("metadata", {})
                
                # Calculate recent activity
                recent_signals = [c for c in cycles[-20:] 
                                if c.get("signal", {}).get("signal") in ["BUY", "SELL"]]
                
                return {
                    "status": "success",
                    "data": {
                        "total_cycles": metadata.get("total_cycles", 0),
                        "total_signals": metadata.get("total_signals", 0),
                        "total_trades": metadata.get("total_trades", 0),
                        "signal_rate": (metadata.get("total_signals", 0) / 
                                      max(metadata.get("total_cycles", 1), 1)),
                        "recent_signals": recent_signals[-5:],  # Last 5 signals
                        "last_updated": metadata.get("last_updated", "Never")
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "success",
                    "data": {
                        "total_cycles": 0,
                        "total_signals": 0,
                        "total_trades": 0,
                        "signal_rate": 0.0,
                        "recent_signals": [],
                        "last_updated": "Never"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "error": f"Failed to load live data: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_optimized_parameters() -> Dict[str, Any]:
        """Get current optimized parameters from backtest results."""
        try:
            params_file = "optimization_results/best_params.json"
            
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                
                return {
                    "status": "success",
                    "parameters": params,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "warning",
                    "message": "No optimization results found",
                    "parameters": ParameterSync.get_default_config(),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load optimized parameters: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_checks = {
            "binance_api": BinanceDebugger.test_binance_api(),
            "server_time": BinanceDebugger.get_server_time(),
            "live_data": DashboardIntegration.get_live_trading_summary(),
            "config": ParameterSync.get_current_config()
        }
        
        # Determine overall health
        failed_checks = sum(1 for check in health_checks.values() 
                           if check.get("status") == "error")
        
        overall_status = "healthy" if failed_checks == 0 else "degraded" if failed_checks < 2 else "critical"
        
        return {
            "status": "success",
            "overall_health": overall_status,
            "checks": health_checks,
            "failed_checks": failed_checks,
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# PYTHON TEST UTILITIES
# ============================================================================

class PythonTester:
    """Python environment testing utilities for serverless verification."""
    
    @staticmethod
    def get_test_response() -> Dict[str, Any]:
        """Generate comprehensive test response for Python environment verification."""
        return {
            "status": "success",
            "message": "Python is working on Vercel!",
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": sys.platform,
                "executable": sys.executable
            },
            "test_data": {
                "btc_price": 116142.79,  # Mock price for testing
                "system": "vercel_serverless",
                "api_status": "operational",
                "features": ["json", "requests", "datetime", "os"]
            },
            "capabilities": {
                "file_system": os.path.exists('.'),
                "json_processing": True,
                "http_requests": True,
                "datetime_operations": True
            }
        }

# ============================================================================
# HANDLER CLASSES FOR VERCEL ENDPOINTS
# ============================================================================

class DebugHandler(BaseHTTPRequestHandler):
    """Debug endpoint handler for Binance API testing."""
    
    def do_GET(self):
        try:
            result = BinanceDebugger.test_binance_api()
            APIUtilities.send_json_response(self, result)
        except Exception as e:
            APIUtilities.send_error_response(self, e)

class TestHandler(BaseHTTPRequestHandler):
    """Test endpoint handler for Python environment verification."""
    
    def do_GET(self):
        try:
            result = PythonTester.get_test_response()
            APIUtilities.send_json_response(self, result)
        except Exception as e:
            APIUtilities.send_error_response(self, e)

class ParameterSyncHandler(BaseHTTPRequestHandler):
    """Parameter synchronization endpoint handler."""
    
    def do_GET(self):
        try:
            result = ParameterSync.get_current_config()
            APIUtilities.send_json_response(self, result)
        except Exception as e:
            APIUtilities.send_error_response(self, e)
    
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_params = json.loads(post_data.decode('utf-8'))
            
            result = ParameterSync.update_config(new_params)
            APIUtilities.send_json_response(self, result)
        except Exception as e:
            APIUtilities.send_error_response(self, e)

class DashboardHandler(BaseHTTPRequestHandler):
    """Dashboard integration endpoint handler."""
    
    def do_GET(self):
        try:
            # Parse query parameters to determine endpoint
            path = self.path.lower()
            
            if 'health' in path:
                result = DashboardIntegration.get_system_health()
            elif 'params' in path or 'parameters' in path:
                result = DashboardIntegration.get_optimized_parameters()
            else:
                result = DashboardIntegration.get_live_trading_summary()
            
            APIUtilities.send_json_response(self, result)
        except Exception as e:
            APIUtilities.send_error_response(self, e)

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test-binance":
            print("🔍 Testing Binance API...")
            result = BinanceDebugger.test_binance_api()
            print(f"📊 Result: {result['status']}")
            
        elif command == "test-python":
            print("🐍 Testing Python environment...")
            result = PythonTester.get_test_response()
            print(f"✅ Python {result['environment']['python_version']} working")
            
        elif command == "health":
            print("💓 Checking system health...")
            result = DashboardIntegration.get_system_health()
            print(f"🎯 Overall health: {result['overall_health']}")
            
        elif command == "config":
            print("⚙️ Getting current configuration...")
            result = ParameterSync.get_current_config()
            print(f"📋 Config source: {result['config']['source']}")
            
        else:
            print("❌ Unknown command. Use: test-binance, test-python, health, or config")
    else:
        print("🔧 API Utilities Module")
        print("Usage: python api/utilities.py [test-binance|test-python|health|config]")
        print("  test-binance - Test Binance API connectivity")
        print("  test-python  - Test Python environment")
        print("  health       - Check comprehensive system health")
        print("  config       - Get current trading configuration")
