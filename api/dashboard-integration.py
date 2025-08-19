"""
Dashboard Integration API - Provides live data for existing Trading System Analysis Dashboard
"""
import json
import os
from datetime import datetime, timezone
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_live_trading_summary():
    """Get summary for existing dashboard integration"""
    try:
        # Load live results if available
        results_file = "live_trading/live_results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            cycles = data.get("trading_cycles", [])
            metadata = data.get("metadata", {})
            
            # Calculate recent activity
            recent_signals = [c for c in cycles[-20:] if c["signal"].get("signal") in ["BUY", "SELL"]]
            
            return {
                "total_cycles": metadata.get("total_cycles", 0),
                "total_signals": metadata.get("total_signals", 0),
                "total_trades": metadata.get("total_trades", 0),
                "signal_rate": (metadata.get("total_signals", 0) / max(metadata.get("total_cycles", 1), 1)),
                "recent_signals": recent_signals[-5:],  # Last 5 signals
                "last_updated": metadata.get("last_updated", "Never")
            }
        else:
            return {
                "total_cycles": 0,
                "total_signals": 0,
                "total_trades": 0,
                "signal_rate": 0.0,
                "recent_signals": [],
                "last_updated": "Never"
            }
            
    except Exception as e:
        return {"error": f"Failed to load live data: {str(e)}"}

def get_optimized_parameters():
    """Get current optimized parameters from backtest"""
    try:
        config_file = "api/live_trading_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            return {"error": "No optimization parameters found"}
    except Exception as e:
        return {"error": f"Failed to load parameters: {str(e)}"}

def handler(request):
    """Vercel serverless function handler for dashboard integration"""
    try:
        # Get query parameter for what data to return
        query_params = request.get('queryStringParameters', {}) or {}
        data_type = query_params.get('type', 'summary')
        
        if data_type == 'parameters':
            data = get_optimized_parameters()
        elif data_type == 'summary':
            data = get_live_trading_summary()
        elif data_type == 'status':
            data = {
                "status": "active",
                "github_actions": "running_every_5_minutes",
                "vercel_deployment": "active",
                "cost": "$0/month",
                "last_checked": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Return everything for dashboard
            data = {
                "live_summary": get_live_trading_summary(),
                "parameters": get_optimized_parameters(),
                "system_status": {
                    "status": "active",
                    "github_actions": "running_every_5_minutes",
                    "vercel_deployment": "active",
                    "cost": "$0/month"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(data, indent=2)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Dashboard integration error',
                'message': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }

# For local testing
if __name__ == "__main__":
    print("📊 Dashboard Integration API Test")
    print("="*40)
    
    # Test all endpoints
    test_requests = [
        {"queryStringParameters": {"type": "summary"}},
        {"queryStringParameters": {"type": "parameters"}},
        {"queryStringParameters": {"type": "status"}},
        {"queryStringParameters": None}  # Full data
    ]
    
    for i, req in enumerate(test_requests, 1):
        print(f"\n{i}. Testing endpoint...")
        response = handler(req)
        data = json.loads(response['body'])
        
        if response['statusCode'] == 200:
            print(f"   ✅ Status: {response['statusCode']}")
            if 'live_summary' in data:
                print(f"   📊 Full Dashboard Data Available")
            elif 'total_cycles' in data:
                print(f"   📈 Live Summary: {data['total_cycles']} cycles")
            elif 'RSI_PERIOD' in data:
                print(f"   🎯 Parameters: RSI {data['RSI_PERIOD']}")
            elif 'status' in data:
                print(f"   🔧 Status: {data['status']}")
        else:
            print(f"   ❌ Error: {data.get('error', 'Unknown')}")
    
    print(f"\n🌐 Deploy this as: /api/dashboard-integration")
    print(f"📊 Your existing dashboard can now fetch live data from this endpoint!")
