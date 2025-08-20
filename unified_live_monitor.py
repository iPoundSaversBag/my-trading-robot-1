#!/usr/bin/env python3
"""
UNIFIED LIVE BOT MONITOR
Single comprehensive monitoring system for the watcher to use

This consolidates all monitoring functionality into one file:
- Real-time bot status monitoring  
- Performance tracking and analysis
- Parameter sync coordination
- Results storage and visualization
- Integration with watcher pipeline
"""

import json
import os
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

class UnifiedLiveBotMonitor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "live_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Vercel endpoints
        self.vercel_base = "https://my-trading-robot-1.vercel.app"
        self.live_bot_endpoint = f"{self.vercel_base}/api/live-bot"
        self.parameter_sync_endpoint = f"{self.vercel_base}/api/parameter-sync"
        self.auth_headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
        
        # Data files
        self.history_file = self.results_dir / "live_bot_history.json"
        self.daily_summary_file = self.results_dir / "daily_summaries.json"
        self.performance_file = self.results_dir / "performance_metrics.json"
        
        # Settings
        self.check_interval = 60  # 1 minute for continuous monitoring
        self.quick_check_interval = 30  # 30 seconds for status checks
        
        # Initialize data files
        self._initialize_data_files()
    
    def _initialize_data_files(self):
        """Initialize data files if they don't exist"""
        
        if not self.history_file.exists():
            initial_history = {
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "total_checks": 0
                },
                "records": []
            }
            with open(self.history_file, 'w') as f:
                json.dump(initial_history, f, indent=2)
        
        if not self.performance_file.exists():
            initial_performance = {
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "total_signals": 0,
                    "total_trades": 0,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "daily_stats": {},
                "overall_performance": {
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0
                }
            }
            with open(self.performance_file, 'w') as f:
                json.dump(initial_performance, f, indent=2)
    
    # ==========================================
    # CORE MONITORING FUNCTIONS
    # ==========================================
    
    def check_bot_status(self):
        """Check current bot status and return structured data"""
        try:
            response = requests.get(
                self.live_bot_endpoint, 
                headers=self.auth_headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": data,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_code": response.status_code,
                    "error_message": response.text[:200]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_message": str(e)
            }
    
    def check_dashboard_status(self):
        """Check if dashboard is accessible"""
        try:
            response = requests.get(self.vercel_base, timeout=10)
            return {
                "dashboard_online": response.status_code == 200,
                "response_time": response.elapsed.total_seconds()
            }
        except:
            return {
                "dashboard_online": False,
                "response_time": None
            }
    
    def record_bot_data(self, bot_status, dashboard_status):
        """Record bot data to history file"""
        
        # Load existing history
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        
        # Create new record
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot_status": bot_status,
            "dashboard_status": dashboard_status
        }
        
        # Add to history
        history["records"].append(record)
        history["metadata"]["last_updated"] = record["timestamp"]
        history["metadata"]["total_checks"] += 1
        
        # Keep only last 1000 records to avoid huge files
        if len(history["records"]) > 1000:
            history["records"] = history["records"][-1000:]
        
        # Save updated history
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return record
    
    # ==========================================
    # PARAMETER SYNC FUNCTIONS
    # ==========================================
    
    def sync_parameters_to_vercel(self, parameters_dict):
        """Sync optimized parameters to Vercel live bot"""
        try:
            print(f"\n🔄 SYNCING PARAMETERS TO LIVE BOT")
            print(f"=" * 50)
            
            # Prepare parameter data
            sync_data = {
                "parameters": parameters_dict,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "watcher_optimization"
            }
            
            # Send to Vercel
            response = requests.post(
                self.parameter_sync_endpoint,
                headers=self.auth_headers,
                json=sync_data,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Parameters synced successfully")
                print(f"   Updated parameters: {len(parameters_dict)} items")
                for key, value in parameters_dict.items():
                    print(f"   • {key}: {value}")
                return True
            else:
                print(f"❌ Parameter sync failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Parameter sync error: {e}")
            return False
    
    def load_latest_backtest_params(self):
        """Load the most recent optimized parameters from backtest"""
        try:
            # Check for latest backtest results
            runs_dir = self.base_dir / "runs"
            if not runs_dir.exists():
                return None
            
            # Find most recent run
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                return None
            
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            
            # Look for optimization results
            params_file = latest_run / "best_params.json"
            if params_file.exists():
                with open(params_file, 'r') as f:
                    return json.load(f)
            
            # Alternative location
            config_file = latest_run / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config.get("best_parameters", None)
            
            return None
            
        except Exception as e:
            print(f"Error loading backtest params: {e}")
            return None
    
    # ==========================================
    # ANALYSIS & REPORTING FUNCTIONS
    # ==========================================
    
    def analyze_performance(self, days=7):
        """Analyze live bot performance over specified days"""
        
        # Load history
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        
        # Filter recent records
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_records = [
            r for r in history["records"]
            if datetime.fromisoformat(r["timestamp"].replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_records:
            return {"error": "No recent data available"}
        
        # Analyze data
        analysis = {
            "period_days": days,
            "total_checks": len(recent_records),
            "success_rate": 0.0,
            "avg_response_time": 0.0,
            "signals_detected": 0,
            "trades_executed": 0,
            "uptime_percentage": 0.0
        }
        
        successful_checks = 0
        response_times = []
        signals = []
        trades = []
        
        for record in recent_records:
            bot_status = record.get("bot_status", {})
            
            if bot_status.get("status") == "success":
                successful_checks += 1
                
                if "response_time" in bot_status:
                    response_times.append(bot_status["response_time"])
                
                # Extract signal data
                data = bot_status.get("data", {})
                if "signal" in data:
                    signals.append(data["signal"])
                
                if "trade_executed" in data and data["trade_executed"]:
                    trades.append(data["trade_executed"])
        
        # Calculate metrics
        analysis["success_rate"] = (successful_checks / len(recent_records)) * 100
        analysis["uptime_percentage"] = analysis["success_rate"]
        analysis["signals_detected"] = len(signals)
        analysis["trades_executed"] = len(trades)
        
        if response_times:
            analysis["avg_response_time"] = sum(response_times) / len(response_times)
        
        return analysis
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        
        print(f"\n📊 LIVE BOT PERFORMANCE REPORT")
        print(f"=" * 50)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Current status
        current_status = self.check_bot_status()
        dashboard_status = self.check_dashboard_status()
        
        print(f"\n🔍 CURRENT STATUS:")
        if current_status["status"] == "success":
            print(f"   ✅ Bot: ONLINE")
            data = current_status["data"]
            
            if "signal" in data:
                signal = data["signal"]
                print(f"   📊 Signal: {signal.get('signal', 'N/A')} ({signal.get('confidence', 0):.3f})")
            
            if "account_balance" in data:
                balances = data["account_balance"]
                print(f"   💰 Account: {len(balances)} assets")
        else:
            print(f"   ❌ Bot: OFFLINE")
        
        dashboard_emoji = "✅" if dashboard_status["dashboard_online"] else "❌"
        print(f"   {dashboard_emoji} Dashboard: {'ONLINE' if dashboard_status['dashboard_online'] else 'OFFLINE'}")
        
        # Performance analysis
        for days in [1, 7, 30]:
            analysis = self.analyze_performance(days)
            if "error" not in analysis:
                print(f"\n📈 LAST {days} DAY{'S' if days > 1 else ''} PERFORMANCE:")
                print(f"   Uptime: {analysis['uptime_percentage']:.1f}%")
                print(f"   Signals: {analysis['signals_detected']}")
                print(f"   Trades: {analysis['trades_executed']}")
                print(f"   Avg Response: {analysis['avg_response_time']:.2f}s")
    
    # ==========================================
    # WATCHER INTEGRATION FUNCTIONS
    # ==========================================
    
    def on_backtest_complete(self, run_directory=None):
        """Called by watcher after backtest completion"""
        
        print(f"\n🎯 BACKTEST COMPLETION HOOK TRIGGERED")
        print(f"=" * 50)
        
        if run_directory:
            print(f"📁 Backtest run: {run_directory}")
        
        try:
            # Load latest optimized parameters
            latest_params = self.load_latest_backtest_params()
            
            if latest_params:
                print(f"📋 Found optimized parameters: {len(latest_params)} items")
                
                # Sync to live bot
                sync_success = self.sync_parameters_to_vercel(latest_params)
                
                if sync_success:
                    print(f"✅ Live bot updated with optimized parameters")
                    
                    # Quick status check
                    print(f"\n🔍 Verifying live bot status...")
                    time.sleep(5)  # Give it a moment
                    
                    bot_status = self.check_bot_status()
                    if bot_status["status"] == "success":
                        print(f"✅ Live bot is responding with new parameters")
                    else:
                        print(f"⚠️  Live bot verification failed")
                    
                    return True
                else:
                    print(f"❌ Failed to sync parameters to live bot")
                    return False
            else:
                print(f"⚠️  No optimized parameters found")
                return False
                
        except Exception as e:
            print(f"❌ Backtest completion hook error: {e}")
            return False
    
    def quick_status_check(self):
        """Quick status check for watcher"""
        bot_status = self.check_bot_status()
        dashboard_status = self.check_dashboard_status()
        
        return {
            "bot_online": bot_status["status"] == "success",
            "dashboard_online": dashboard_status["dashboard_online"],
            "timestamp": datetime.now().isoformat()
        }
    
    def continuous_monitoring(self, duration_minutes=None):
        """Run continuous monitoring (called by watcher or standalone)"""
        
        print(f"🔄 STARTING CONTINUOUS LIVE BOT MONITORING")
        print(f"=" * 50)
        
        if duration_minutes:
            print(f"Duration: {duration_minutes} minutes")
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
        else:
            print(f"Duration: Indefinite (Ctrl+C to stop)")
            end_time = None
        
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Check if we should stop
                if end_time and datetime.now() >= end_time:
                    print(f"\n⏰ Monitoring duration completed")
                    break
                
                # Get current status
                bot_status = self.check_bot_status()
                dashboard_status = self.check_dashboard_status()
                
                # Record data
                record = self.record_bot_data(bot_status, dashboard_status)
                
                # Print status
                timestamp = datetime.now().strftime('%H:%M:%S')
                bot_emoji = "🟢" if bot_status["status"] == "success" else "🔴"
                dashboard_emoji = "🟢" if dashboard_status["dashboard_online"] else "🔴"
                
                print(f"[{timestamp}] {bot_emoji} Bot | {dashboard_emoji} Dashboard", end="")
                
                if bot_status["status"] == "success":
                    data = bot_status["data"]
                    if "signal" in data:
                        signal = data["signal"]
                        print(f" | 📊 {signal.get('signal', 'N/A')} ({signal.get('confidence', 0):.2f})")
                    else:
                        print(f" | ✅ Online")
                else:
                    print(f" | ❌ Offline")
                
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")

# ==========================================
# SIMPLIFIED INTERFACE FOR WATCHER
# ==========================================

# Global monitor instance
_monitor = None

def get_monitor():
    """Get global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = UnifiedLiveBotMonitor()
    return _monitor

def on_backtest_complete(run_directory=None):
    """Simple function for watcher to call"""
    monitor = get_monitor()
    return monitor.on_backtest_complete(run_directory)

def quick_status():
    """Quick status check for watcher"""
    monitor = get_monitor()
    return monitor.quick_status_check()

def sync_parameters(parameters_dict):
    """Sync parameters to live bot"""
    monitor = get_monitor()
    return monitor.sync_parameters_to_vercel(parameters_dict)

def start_monitoring(duration_minutes=None):
    """Start continuous monitoring"""
    monitor = get_monitor()
    monitor.continuous_monitoring(duration_minutes)

def performance_report():
    """Generate performance report"""
    monitor = get_monitor()
    monitor.generate_performance_report()

# ==========================================
# COMMAND LINE INTERFACE
# ==========================================

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Live Bot Monitor")
    parser.add_argument("--status", action="store_true", help="Quick status check")
    parser.add_argument("--monitor", type=int, metavar="MINUTES", help="Start monitoring for N minutes")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--sync", metavar="JSON_FILE", help="Sync parameters from JSON file")
    parser.add_argument("--test-hook", metavar="RUN_DIR", help="Test backtest completion hook")
    
    args = parser.parse_args()
    
    monitor = get_monitor()
    
    if args.status:
        status = quick_status()
        print(f"Bot Online: {status['bot_online']}")
        print(f"Dashboard Online: {status['dashboard_online']}")
        print(f"Timestamp: {status['timestamp']}")
    
    elif args.monitor is not None:
        start_monitoring(args.monitor if args.monitor > 0 else None)
    
    elif args.report:
        performance_report()
    
    elif args.sync:
        try:
            with open(args.sync, 'r') as f:
                params = json.load(f)
            success = sync_parameters(params)
            print(f"Parameter sync: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"Error syncing parameters: {e}")
    
    elif args.test_hook:
        success = on_backtest_complete(args.test_hook)
        print(f"Hook test: {'Success' if success else 'Failed'}")
    
    else:
        # Default: quick status then short monitoring session
        print(f"🚀 UNIFIED LIVE BOT MONITOR")
        print(f"=" * 40)
        
        status = quick_status()
        print(f"Current Status:")
        print(f"   Bot: {'✅ Online' if status['bot_online'] else '❌ Offline'}")
        print(f"   Dashboard: {'✅ Online' if status['dashboard_online'] else '❌ Offline'}")
        
        print(f"\nStarting 10-minute monitoring session...")
        start_monitoring(10)

if __name__ == "__main__":
    main()
