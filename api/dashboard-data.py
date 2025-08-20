"""
Dashboard Data API - Serves comprehensive dashboard data including enhanced tearsheet
"""
import json
import os
import sys
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.tearsheet import generate_tearsheet_html
    TEARSHEET_AVAILABLE = True
except ImportError:
    TEARSHEET_AVAILABLE = False

def get_live_results():
    """Get live trading results - only real data"""
    try:
        # Try multiple possible locations for live results
        possible_files = [
            "live_results/live_bot_history.json",
            "live_trading/live_results.json", 
            "live_results.json"
        ]
        
        for results_file in possible_files:
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                return data
        
        # No mock data - return error if no real data found
        return {"error": "No live trading results found - real data only"}
        
    except Exception as e:
        return {"error": f"Failed to load live results: {str(e)}"}

def get_backtest_summary():
    """Get latest backtest summary - only real data"""
    try:
        latest_run_file = "plots_output/latest_run_dir.txt"
        if not os.path.exists(latest_run_file):
            return {"error": "No backtest results found - run backtest locally first"}
        
        with open(latest_run_file, 'r') as f:
            latest_dir = f.read().strip()
        
        final_config_file = f"{latest_dir}/final_config.json"
        if not os.path.exists(final_config_file):
            return {"error": f"Backtest config file not found: {final_config_file}"}
        
        with open(final_config_file, 'r') as f:
            config = json.load(f)
        
        return {
            "optimization_trials": config.get("optimization_trial_count", 0),
            "parameters": config.get("best_parameters_so_far", {}),
            "metrics": config.get("best_metrics_so_far", {}),
            "created_at": config.get("created_at", "Unknown"),
            "run_directory": latest_dir
        }
        
    except Exception as e:
        return {"error": f"Failed to load backtest data: {str(e)}"}

def get_system_status():
    """Get overall system status"""
    live_data = get_live_results()
    backtest_data = get_backtest_summary()
    
    # Determine system health
    status = "healthy"
    issues = []
    
    if "error" in live_data:
        status = "warning"
        issues.append("Live trading data unavailable")
    
    if "error" in backtest_data:
        status = "warning"
        issues.append("Backtest data unavailable")
    
    return {
        "status": status,
        "issues": issues,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tearsheet_available": TEARSHEET_AVAILABLE
    }

def handler(request):
    """Vercel serverless function handler"""
    try:
        # Check for tearsheet request
        path = getattr(request, 'path', request.get('path', '/'))
        query_params = getattr(request, 'args', request.get('queryStringParameters', {})) or {}
        
        # Handle tearsheet endpoint
        if 'tearsheet' in path or query_params.get('format') == 'tearsheet':
            if TEARSHEET_AVAILABLE:
                try:
                    html = generate_tearsheet_html()
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'text/html',
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0'
                        },
                        'body': html
                    }
                except Exception as e:
                    return {
                        'statusCode': 500,
                        'headers': {'Content-Type': 'application/json'},
                        'body': json.dumps({
                            'error': 'Tearsheet generation failed',
                            'message': str(e),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                    }
            else:
                return {
                    'statusCode': 503,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({
                        'error': 'Tearsheet functionality unavailable',
                        'message': 'Tearsheet module not accessible',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                }
        
        # Default: return dashboard data as JSON
        live_data = get_live_results()
        backtest_data = get_backtest_summary()
        system_status = get_system_status()
        
        dashboard_data = {
            "live_trading": live_data,
            "backtest": backtest_data,
            "system": system_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_version": "2.0"
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
            'body': json.dumps(dashboard_data, indent=2)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }

# For local testing
if __name__ == "__main__":
    print("Testing Dashboard Data API...")
    
    # Test JSON response
    mock_request = {'path': '/', 'queryStringParameters': {}}
    response = handler(mock_request)
    print("JSON Response:", response['statusCode'])
    
    # Test tearsheet response
    mock_tearsheet_request = {'path': '/tearsheet', 'queryStringParameters': {'format': 'tearsheet'}}
    tearsheet_response = handler(mock_tearsheet_request)
    print("Tearsheet Response:", tearsheet_response['statusCode'])
    
    if tearsheet_response['statusCode'] == 200:
        # Save tearsheet for testing
        with open("test_tearsheet.html", "w", encoding="utf-8") as f:
            f.write(tearsheet_response['body'])
        print("Test tearsheet saved to: test_tearsheet.html")
    
    print("Dashboard Data API test complete")
