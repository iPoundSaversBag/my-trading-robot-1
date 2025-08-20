#!/usr/bin/env python3
"""
Live Bot Results Monitor
Continuously monitors live bot performance and provides local feedback
"""

import json
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class LiveBotMonitor:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "live_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Vercel endpoints
        self.vercel_base = "https://my-trading-robot-1.vercel.app"
        self.live_bot_endpoint = f"{self.vercel_base}/api/live-bot"
        self.auth_headers = {"Authorization": "Bearer 93699b3917045092715b8e16c01f2e1d"}
        
        # Monitoring settings
        self.check_interval = 60  # 1 minute
        self.history_file = self.results_dir / "live_bot_history.json"
        self.daily_summary_file = self.results_dir / "daily_summaries.json"
        
        # Load existing history
        self.history = self.load_history()
    
    def load_history(self):
        """Load existing trading history"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def save_history(self):
        """Save trading history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def fetch_current_status(self):
        """Fetch current live bot status"""
        try:
            response = requests.get(self.live_bot_endpoint, headers=self.auth_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key information
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "signal": data.get("signal", {}),
                    "account_balance": data.get("account_balance", {}),
                    "trade_executed": data.get("trade_executed", {}),
                    "performance_metrics": data.get("performance_metrics", {}),
                    "system_status": data.get("system_status", {})
                }
                
                return status
            else:
                print(f"⚠️ API returned status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching status: {e}")
            return None
    
    def update_history(self, status):
        """Add new status to history"""
        if status:
            self.history.append(status)
            
            # Keep only last 1000 entries to prevent memory issues
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            
            self.save_history()
    
    def analyze_performance(self):
        """Analyze recent performance"""
        if len(self.history) < 10:
            return None
        
        recent_history = self.history[-100:]  # Last 100 data points
        
        # Extract data for analysis
        signals = []
        confidences = []
        balances = []
        timestamps = []
        
        for entry in recent_history:
            signals.append(entry.get("signal", {}).get("signal", "HOLD"))
            confidences.append(entry.get("signal", {}).get("confidence", 0))
            
            balance = entry.get("account_balance", {})
            total_balance = balance.get("USDT", 0) + (balance.get("BTC", 0) * entry.get("signal", {}).get("current_price", 0))
            balances.append(total_balance)
            
            timestamps.append(entry.get("timestamp"))
        
        # Calculate metrics
        signal_counts = {"BUY": signals.count("BUY"), "SELL": signals.count("SELL"), "HOLD": signals.count("HOLD")}
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        balance_change = 0
        if len(balances) >= 2:
            balance_change = ((balances[-1] - balances[0]) / balances[0]) * 100 if balances[0] > 0 else 0
        
        return {
            "signal_distribution": signal_counts,
            "average_confidence": avg_confidence,
            "balance_change_percent": balance_change,
            "current_balance": balances[-1] if balances else 0,
            "data_points": len(recent_history),
            "timespan_hours": self.calculate_timespan_hours(timestamps)
        }
    
    def calculate_timespan_hours(self, timestamps):
        """Calculate timespan in hours"""
        try:
            if len(timestamps) < 2:
                return 0
            
            start_time = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            
            return (end_time - start_time).total_seconds() / 3600
        except:
            return 0
    
    def create_daily_summary(self):
        """Create summary for today"""
        today = datetime.now().strftime('%Y-%m-%d')
        today_entries = []
        
        for entry in self.history:
            try:
                entry_date = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')).strftime('%Y-%m-%d')
                if entry_date == today:
                    today_entries.append(entry)
            except:
                continue
        
        if not today_entries:
            return None
        
        # Analyze today's data
        signals = [e.get("signal", {}).get("signal", "HOLD") for e in today_entries]
        confidences = [e.get("signal", {}).get("confidence", 0) for e in today_entries]
        
        # Get start and end balances
        start_balance = today_entries[0].get("account_balance", {})
        end_balance = today_entries[-1].get("account_balance", {})
        
        summary = {
            "date": today,
            "total_checks": len(today_entries),
            "signals": {
                "BUY": signals.count("BUY"),
                "SELL": signals.count("SELL"), 
                "HOLD": signals.count("HOLD")
            },
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "start_balance": start_balance,
            "end_balance": end_balance,
            "created_at": datetime.now().isoformat()
        }
        
        return summary
    
    def save_daily_summary(self, summary):
        """Save daily summary"""
        try:
            # Load existing summaries
            summaries = []
            if self.daily_summary_file.exists():
                with open(self.daily_summary_file, 'r') as f:
                    summaries = json.load(f)
            
            # Update or add today's summary
            today = summary["date"]
            summaries = [s for s in summaries if s.get("date") != today]  # Remove existing
            summaries.append(summary)
            
            # Save updated summaries
            with open(self.daily_summary_file, 'w') as f:
                json.dump(summaries, f, indent=2)
                
        except Exception as e:
            print(f"Error saving daily summary: {e}")
    
    def display_status(self, status, performance):
        """Display current status and performance"""
        print(f"\n📊 LIVE BOT STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        if status:
            signal = status.get("signal", {})
            balance = status.get("account_balance", {})
            
            print(f"📡 Current Signal: {signal.get('signal', 'N/A')} ({signal.get('confidence', 0):.3f} confidence)")
            print(f"💰 USDT Balance: ${balance.get('USDT', 0):,.2f}")
            print(f"🪙 BTC Balance: {balance.get('BTC', 0):.6f}")
            print(f"📈 Current BTC Price: ${signal.get('current_price', 0):,.2f}")
        
        if performance:
            print(f"\n📈 RECENT PERFORMANCE:")
            print(f"   Signal Distribution: BUY={performance['signal_distribution']['BUY']}, SELL={performance['signal_distribution']['SELL']}, HOLD={performance['signal_distribution']['HOLD']}")
            print(f"   Average Confidence: {performance['average_confidence']:.3f}")
            print(f"   Balance Change: {performance['balance_change_percent']:+.2f}%")
            print(f"   Data Points: {performance['data_points']} over {performance['timespan_hours']:.1f} hours")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring"""
        print("🔄 STARTING LIVE BOT MONITORING")
        print("=" * 60)
        print(f"🌐 Endpoint: {self.live_bot_endpoint}")
        print(f"⏰ Check Interval: {self.check_interval} seconds")
        print(f"📁 Results Directory: {self.results_dir}")
        print("=" * 60)
        
        cycle_count = 0
        last_daily_summary = None
        
        try:
            while True:
                cycle_count += 1
                
                # Fetch current status
                status = self.fetch_current_status()
                self.update_history(status)
                
                # Analyze performance
                performance = self.analyze_performance()
                
                # Display status every 10 cycles or on first run
                if cycle_count % 10 == 1:
                    self.display_status(status, performance)
                
                # Create daily summary once per day
                current_date = datetime.now().strftime('%Y-%m-%d')
                if last_daily_summary != current_date:
                    daily_summary = self.create_daily_summary()
                    if daily_summary:
                        self.save_daily_summary(daily_summary)
                        print(f"\n📋 Daily Summary Created: {daily_summary['total_checks']} checks today")
                    last_daily_summary = current_date
                
                # Brief status update for other cycles
                if cycle_count % 10 != 1 and status:
                    signal = status.get("signal", {})
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - {signal.get('signal', 'N/A')} ({signal.get('confidence', 0):.3f})")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user after {cycle_count} cycles")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        
        # Final summary
        final_summary = self.create_daily_summary()
        if final_summary:
            self.save_daily_summary(final_summary)
            print(f"\n📋 Final Daily Summary: {final_summary['total_checks']} total checks")

def main():
    """Main entry point"""
    monitor = LiveBotMonitor()
    
    import sys
    if "--status" in sys.argv:
        # One-time status check
        status = monitor.fetch_current_status()
        performance = monitor.analyze_performance()
        monitor.display_status(status, performance)
    else:
        # Continuous monitoring
        monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()
