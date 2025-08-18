"""
Remote Parameter Sync API - Allows live bot to access local backtest data
"""

import json
import os
import requests
from http.server import BaseHTTPRequestHandler

def get_latest_backtest_results():
    """Get the latest local backtest optimization results"""
    try:
        # Read latest run directory
        with open('plots_output/latest_run_dir.txt', 'r') as f:
            latest_dir = f.read().strip()
        
        # Load optimized parameters
        params_path = f"{latest_dir}/optimized_params_per_window.json"
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                all_params = json.load(f)
            
            # Get latest window
            latest_window = max(all_params.keys(), key=lambda x: int(x.split('_')[1]))
            latest_params = all_params[latest_window]
            
            return {
                'status': 'success',
                'latest_window': latest_window,
                'optimization_timestamp': latest_dir.split('/')[-1],
                'parameters': latest_params,
                'total_windows': len(all_params)
            }
        else:
            return {'status': 'error', 'message': 'No optimization results found'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def sync_parameters_to_vercel():
    """Sync latest parameters to Vercel live bot config"""
    try:
        results = get_latest_backtest_results()
        if results['status'] != 'success':
            return results
        
        latest_params = results['parameters']
        
        # Create live bot config
        live_config = {
            "SYMBOL": "BTCUSDT",
            "TIMEFRAME": "5m",
            "RSI_PERIOD": latest_params.get("RSI_PERIOD", 14),
            "RSI_OVERBOUGHT": latest_params.get("RSI_OVERBOUGHT", 70),
            "RSI_OVERSOLD": latest_params.get("RSI_OVERSOLD", 30),
            "MA_FAST": latest_params.get("TENKAN_SEN_PERIOD", 12),
            "MA_SLOW": latest_params.get("KIJUN_SEN_PERIOD", 26),
            "ADX_PERIOD": latest_params.get("ADX_PERIOD", 14),
            "ATR_PERIOD": latest_params.get("ATR_PERIOD", 14),
            "STOP_LOSS_MULTIPLIER": latest_params.get("STOP_LOSS_MULTIPLIER", 2.0),
            "TAKE_PROFIT_MULTIPLIER": latest_params.get("TAKE_PROFIT_MULTIPLIER", 3.0),
            "volatility_threshold": latest_params.get("volatility_threshold", 0.03),
            "volume_threshold_multiplier": latest_params.get("volume_threshold_multiplier", 2.0),
            "min_confidence_for_trade": latest_params.get("min_confidence", 0.04),
            "source_window": results['latest_window'],
            "optimization_timestamp": results['optimization_timestamp'],
            "last_sync": "auto-sync"
        }
        
        # Add regime filters
        for key, value in latest_params.items():
            if key.startswith('USE_') and '_FILTER_' in key:
                live_config[key] = value
        
        # Save to live bot config
        with open('api/live_trading_config.json', 'w') as f:
            json.dump(live_config, f, indent=2)
        
        return {
            'status': 'success',
            'message': 'Parameters synced successfully',
            'synced_parameters': len(live_config),
            'regime_filters': len([k for k in live_config.keys() if '_FILTER_' in k])
        }
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            path = self.path
            
            if path == '/api/backtest-sync':
                # Get latest backtest results
                result = get_latest_backtest_results()
            elif path == '/api/backtest-sync/update':
                # Sync parameters to live bot
                result = sync_parameters_to_vercel()
            else:
                result = {
                    'status': 'error', 
                    'message': 'Available endpoints: /api/backtest-sync, /api/backtest-sync/update'
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps(result, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"error": f"Backtest sync failed: {str(e)}"}
            self.wfile.write(json.dumps(error_response).encode())

    def do_POST(self):
        self.do_GET()  # Handle POST same as GET for now
