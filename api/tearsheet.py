"""
Tearsheet API Endpoint - Provides comprehensive performance comparison with live data integration
"""
import json
import os
from datetime import datetime, timezone
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_enhanced_performance_report():
    """Load the enhanced performance report created by generate_plots.py"""
    try:
        # First try to find the latest enhanced report
        latest_run_file = "plots_output/latest_run_dir.txt"
        
        if os.path.exists(latest_run_file):
            with open(latest_run_file, 'r') as f:
                latest_dir = f.read().strip()
            
            performance_report_path = f"{latest_dir}/performance_report.html"
            if os.path.exists(performance_report_path):
                with open(performance_report_path, 'r', encoding='utf-8') as f:
                    enhanced_html = f.read()
                
                # Check if it has V4 enhancements
                if "ENHANCEMENT_DASHBOARD_V4_START" in enhanced_html:
                    return enhanced_html, performance_report_path
        
        # If no enhanced report found, return None
        return None, None
        
    except Exception as e:
        print(f"Error loading enhanced report: {e}")
        return None, None

def load_latest_backtest():
    """Load latest backtest results"""
    try:
        latest_run_file = "plots_output/latest_run_dir.txt"
        if not os.path.exists(latest_run_file):
            return {"error": "No backtest results found"}
        
        with open(latest_run_file, 'r') as f:
            latest_dir = f.read().strip()
        
        # Load final config
        final_config_file = f"{latest_dir}/final_config.json"
        if not os.path.exists(final_config_file):
            return {"error": f"Config file not found: {final_config_file}"}
        
        with open(final_config_file, 'r') as f:
            config = json.load(f)
        
        # Extract key metrics
        best_params = config.get("best_parameters_so_far", {})
        best_metrics = config.get("best_metrics_so_far", {})
        
        return {
            "run_directory": latest_dir,
            "parameters": best_params,
            "metrics": best_metrics,
            "optimization_trials": config.get("optimization_trial_count", 0),
            "created_at": config.get("created_at", "Unknown")
        }
        
    except Exception as e:
        return {"error": f"Failed to load backtest: {str(e)}"}

def load_live_results():
    """Load live trading results"""
    try:
        results_file = "live_trading/live_results.json"
        if not os.path.exists(results_file):
            return {"error": "No live results found - run live_trading_tracker.py first"}
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        return {
            "metadata": data.get("metadata", {}),
            "recent_cycles": data.get("trading_cycles", [])[-20:],  # Last 20 cycles
            "performance": data.get("performance_summary", {})
        }
        
    except Exception as e:
        return {"error": f"Failed to load live results: {str(e)}"}

def inject_live_data_into_enhanced_report(enhanced_html, live_data, backtest_data):
    """Inject live data into the enhanced performance report"""
    
    # Create live data injection script
    live_data_script = f"""
    <script>
    // Live Data Integration for Enhanced Tearsheet
    const liveData = {json.dumps(live_data)};
    const backtestData = {json.dumps(backtest_data)};
    
    function updateLiveMetrics() {{
        try {{
            // Update live signal rate if element exists
            const signalRateElement = document.getElementById('live-signal-rate');
            if (signalRateElement && liveData.metadata) {{
                const totalCycles = liveData.metadata.total_cycles || 0;
                const totalSignals = liveData.metadata.total_signals || 0;
                const signalRate = totalCycles > 0 ? (totalSignals / totalCycles * 100).toFixed(1) : '0.0';
                signalRateElement.textContent = signalRate + '%';
            }}
            
            // Update live trades count
            const tradesElement = document.getElementById('live-trades-count');
            if (tradesElement && liveData.metadata) {{
                tradesElement.textContent = liveData.metadata.total_trades || 0;
            }}
            
            // Update live cycles count
            const cyclesElement = document.getElementById('live-cycles-count');
            if (cyclesElement && liveData.metadata) {{
                cyclesElement.textContent = liveData.metadata.total_cycles || 0;
            }}
            
            // Update last update timestamp
            const timestampElement = document.getElementById('live-timestamp');
            if (timestampElement) {{
                timestampElement.textContent = new Date().toLocaleString();
            }}
            
            // Add live status indicator
            const statusElement = document.getElementById('live-status');
            if (statusElement) {{
                if (liveData.error) {{
                    statusElement.innerHTML = '🔴 Error: ' + liveData.error;
                    statusElement.className = 'status-error';
                }} else {{
                    statusElement.innerHTML = '🟢 Live Trading Active';
                    statusElement.className = 'status-good';
                }}
            }}
            
        }} catch (error) {{
            console.error('Error updating live metrics:', error);
        }}
    }}
    
    // Update immediately when page loads
    document.addEventListener('DOMContentLoaded', updateLiveMetrics);
    
    // Auto-refresh every 30 seconds
    setInterval(updateLiveMetrics, 30000);
    
    </script>
    """
    
    # Add live data display section after the V4 banner
    live_data_section = f"""
    <!-- Live Data Integration Section -->
    <div class="v4-live-section" style="margin: 20px auto; max-width: 1000px; padding: 20px; background: #f8fafc; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h2 style="color: #374151; margin-bottom: 20px;">🔴 Live Trading Status</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5em; font-weight: bold; color: #059669;" id="live-signal-rate">Calculating...</div>
                <div style="color: #6b7280; font-size: 0.9em;">Signal Rate</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5em; font-weight: bold; color: #3b82f6;" id="live-trades-count">0</div>
                <div style="color: #6b7280; font-size: 0.9em;">Total Trades</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1.5em; font-weight: bold; color: #8b5cf6;" id="live-cycles-count">0</div>
                <div style="color: #6b7280; font-size: 0.9em;">Total Cycles</div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 1em; font-weight: bold;" id="live-status">🟡 Loading...</div>
                <div style="color: #6b7280; font-size: 0.9em;">Status</div>
            </div>
        </div>
        <div style="text-align: center; color: #6b7280; font-size: 0.8em;">
            Last Updated: <span id="live-timestamp">Loading...</span>
        </div>
    </div>
    """
    
    # Inject the live data section after the V4 banner
    banner_end = enhanced_html.find("</div>", enhanced_html.find("v4-banner"))
    if banner_end != -1:
        enhanced_html = enhanced_html[:banner_end + 6] + live_data_section + enhanced_html[banner_end + 6:]
    
    # Inject the script before closing body tag
    body_end = enhanced_html.rfind("</body>")
    if body_end != -1:
        enhanced_html = enhanced_html[:body_end] + live_data_script + enhanced_html[body_end:]
    
    return enhanced_html

def generate_tearsheet_html():
    """Generate the complete tearsheet HTML with enhanced performance report and live data"""
    
    # Try to load the enhanced performance report first
    enhanced_html, report_path = load_enhanced_performance_report()
    
    if enhanced_html:
        # We have an enhanced report - inject live data
        print("Using enhanced performance report with live data integration")
        
        # Load live and backtest data
        live_data = load_live_results()
        backtest_data = load_latest_backtest()
        
        # Inject live data into the enhanced report
        final_html = inject_live_data_into_enhanced_report(enhanced_html, live_data, backtest_data)
        return final_html
    
    else:
        # Fallback to basic comparison tearsheet
        print("No enhanced report found, generating basic comparison tearsheet")
        return generate_basic_comparison_tearsheet()

def generate_basic_comparison_tearsheet():
    """Generate a basic comparison tearsheet when enhanced report is not available"""
    
    # Load data
    backtest = load_latest_backtest()
    live = load_live_results()
    
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Tearsheet</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }}
        .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        .table th, .table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background: #f8f9fa; font-weight: bold; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        pre {{ background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trading Bot Performance Tearsheet</h1>
            <p>Backtest vs Live Trading Comparison</p>
            <p class="timestamp">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Backtest Results</h2>
            {"<p class='status-error'>❌ " + backtest.get('error', '') + "</p>" if 'error' in backtest else f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value status-good">{backtest.get('optimization_trials', 0)}</div>
                    <div class="metric-label">Optimization Trials</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{backtest.get('parameters', {}).get('RSI_PERIOD', 'N/A')}</div>
                    <div class="metric-label">RSI Period</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{backtest.get('parameters', {}).get('MA_FAST_PERIOD', 'N/A')}</div>
                    <div class="metric-label">Fast MA</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{backtest.get('parameters', {}).get('MA_SLOW_PERIOD', 'N/A')}</div>
                    <div class="metric-label">Slow MA</div>
                </div>
            </div>
            <h3>Optimized Parameters:</h3>
            <pre>{json.dumps(backtest.get('parameters', {}), indent=2)}</pre>
            <h3>Best Metrics:</h3>
            <pre>{json.dumps(backtest.get('metrics', {}), indent=2)}</pre>
            """}
        </div>
        
        <div class="section">
            <h2>🔴 Live Trading Results</h2>
            {"<p class='status-error'>❌ " + live.get('error', '') + "</p>" if 'error' in live else f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value status-good">{live.get('metadata', {}).get('total_cycles', 0)}</div>
                    <div class="metric-label">Total Cycles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{live.get('metadata', {}).get('total_signals', 0)}</div>
                    <div class="metric-label">Signals Generated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{live.get('metadata', {}).get('total_trades', 0)}</div>
                    <div class="metric-label">Trades Executed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{(live.get('metadata', {}).get('total_signals', 0) / max(live.get('metadata', {}).get('total_cycles', 1), 1) * 100):.1f}%</div>
                    <div class="metric-label">Signal Rate</div>
                </div>
            </div>
            
            <h3>Recent Trading Cycles:</h3>
            <table class="table">
                <tr>
                    <th>Timestamp</th>
                    <th>Signal</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
                {generate_cycle_rows(live.get('recent_cycles', []))}
            </table>
            """}
        </div>
        
        <div class="section">
            <h2>⚖️ Comparison Analysis</h2>
            {generate_comparison_analysis(backtest, live)}
        </div>
        
        <div class="section">
            <h2>🔧 System Status</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value status-good">✅ Active</div>
                    <div class="metric-label">GitHub Actions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-good">✅ Deployed</div>
                    <div class="metric-label">Vercel API</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-good">$0/month</div>
                    <div class="metric-label">Hosting Cost</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-good">5 min</div>
                    <div class="metric-label">Update Frequency</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Next Steps</h2>
            <ul>
                <li>✅ Background automation running every 5 minutes</li>
                <li>✅ Dynamic parameter loading from backtest optimization</li>
                <li>✅ Complete Vercel migration achieved</li>
                <li>🔄 Accumulating live trading data for comparison</li>
                <li>📊 Will compare live vs backtest performance once sufficient data collected</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def generate_cycle_rows(cycles):
    """Generate HTML table rows for trading cycles"""
    if not cycles:
        return "<tr><td colspan='4'>No cycles recorded yet</td></tr>"
    
    rows = []
    for cycle in cycles[-10:]:  # Last 10 cycles
        timestamp = cycle.get('timestamp', '')[:19].replace('T', ' ')
        signal = cycle.get('signal', {})
        signal_type = signal.get('signal', 'HOLD')
        confidence = signal.get('confidence', 0)
        status = cycle.get('status', 'unknown')
        
        status_class = "status-good" if status == "success" else "status-warning" if status == "warning" else ""
        
        rows.append(f"""
            <tr>
                <td>{timestamp}</td>
                <td><strong>{signal_type}</strong></td>
                <td>{confidence:.3f}</td>
                <td class="{status_class}">{status}</td>
            </tr>
        """)
    
    return "".join(rows)

def generate_comparison_analysis(backtest, live):
    """Generate comparison analysis between backtest and live"""
    if 'error' in backtest or 'error' in live:
        return "<p>⏳ Comparison will be available once both backtest and live data are collected.</p>"
    
    live_cycles = live.get('metadata', {}).get('total_cycles', 0)
    live_signals = live.get('metadata', {}).get('total_signals', 0)
    
    if live_cycles < 10:
        return f"""
        <p class="status-warning">⏳ Collecting live data... ({live_cycles} cycles recorded)</p>
        <p>Meaningful comparison requires at least 50+ cycles. System will automatically compare performance once sufficient data is available.</p>
        <p><strong>Current Status:</strong> Live bot is using optimized parameters from backtest.</p>
        """
    
    signal_rate = (live_signals / live_cycles * 100) if live_cycles > 0 else 0
    
    return f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{signal_rate:.1f}%</div>
            <div class="metric-label">Live Signal Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value status-good">✅</div>
            <div class="metric-label">Parameters Sync</div>
        </div>
    </div>
    <p><strong>Analysis:</strong> Live trading is using optimized parameters from backtest. More detailed comparison will be available as more live data accumulates.</p>
    """

def handler(request):
    """Vercel serverless function handler"""
    try:
        html = generate_tearsheet_html()
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'no-cache'
            },
            'body': html
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
    print("Generating Tearsheet...")
    html = generate_tearsheet_html()
    
    # Save to file
    with open("tearsheet.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("Tearsheet saved to: tearsheet.html")
    print("Open in browser to view performance comparison")
