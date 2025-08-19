#!/usr/bin/env python3
"""
Live Trading Results Tracker - Records live bot performance for tearsheet integration
"""
import json
import os
from datetime import datetime, timezone
import requests

class LiveTradingTracker:
    def __init__(self):
        self.results_file = "live_trading/live_results.json"
        self.ensure_results_file()
    
    def ensure_results_file(self):
        """Create results file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        if not os.path.exists(self.results_file):
            initial_data = {
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "total_cycles": 0,
                    "total_signals": 0,
                    "total_trades": 0,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "trading_cycles": [],
                "performance_summary": {
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0
                }
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def record_trading_cycle(self, bot_response):
        """Record a trading cycle from the live bot"""
        try:
            # Load existing data
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            # Extract cycle information
            cycle_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle_id": data["metadata"]["total_cycles"] + 1,
                "signal": bot_response.get("signal", {}),
                "account_balance": bot_response.get("account_balance", {}),
                "trade_executed": bot_response.get("trade_executed"),
                "config_used": bot_response.get("config_used", "BTCUSDT"),
                "status": bot_response.get("status", "unknown")
            }
            
            # Add to cycles
            data["trading_cycles"].append(cycle_data)
            
            # Update metadata
            data["metadata"]["total_cycles"] += 1
            data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Count signals and trades
            if cycle_data["signal"].get("signal") in ["BUY", "SELL"]:
                data["metadata"]["total_signals"] += 1
            
            if cycle_data["trade_executed"] and not cycle_data["trade_executed"].get("simulated"):
                data["metadata"]["total_trades"] += 1
            
            # Keep only last 1000 cycles to prevent file from getting too large
            if len(data["trading_cycles"]) > 1000:
                data["trading_cycles"] = data["trading_cycles"][-1000:]
            
            # Save updated data
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error recording trading cycle: {e}")
            return False
    
    def get_performance_summary(self):
        """Get performance summary for tearsheet"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            cycles = data["trading_cycles"]
            if not cycles:
                return data["performance_summary"]
            
            # Calculate basic statistics
            signals = [c for c in cycles if c["signal"].get("signal") in ["BUY", "SELL"]]
            trades = [c for c in cycles if c["trade_executed"] and not c["trade_executed"].get("simulated")]
            
            summary = {
                "total_cycles": len(cycles),
                "total_signals": len(signals),
                "total_trades": len(trades),
                "signal_rate": len(signals) / len(cycles) if cycles else 0,
                "last_updated": data["metadata"]["last_updated"],
                "recent_signals": signals[-10:] if signals else [],
                "account_balances": [c["account_balance"] for c in cycles[-5:]] if cycles else []
            }
            
            # Update performance summary in file
            data["performance_summary"].update(summary)
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return summary
            
        except Exception as e:
            print(f"Error calculating performance: {e}")
            return {}

def fetch_live_bot_data():
    """Fetch current live bot status"""
    try:
        bot_url = "https://my-trading-robot-1-hlu5e6f29-aidan-lanes-projects.vercel.app"
        bot_secret = os.environ.get('BOT_SECRET', '93699b3917045092715b8e16c01f2e1d')
        
        headers = {"Authorization": f"Bearer {bot_secret}"}
        response = requests.get(f"{bot_url}/api/live-bot", headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status code: {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

def update_dashboard_live_data():
    """Update your existing dashboard with live trading data"""
    try:
        # Update the live_bot_state.json that your dashboard reads
        bot_data = fetch_live_bot_data()
        
        if "error" not in bot_data:
            # Update the file your dashboard reads from
            live_state_file = "live_trading/live_bot_state.json"
            os.makedirs(os.path.dirname(live_state_file), exist_ok=True)
            
            with open(live_state_file, 'w') as f:
                json.dump(bot_data, f, indent=2)
            
            print(f"✅ Updated dashboard live data: {live_state_file}")
            return True
        else:
            print(f"⚠️ Bot API error: {bot_data['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating dashboard data: {e}")
        return False

def update_tearsheet_data():
    """Update your existing dashboard system with latest data"""
    print("🔄 Updating Dashboard Integration...")
    
    # 1. Update dashboard live data
    dashboard_updated = update_dashboard_live_data()
    
    # 2. Record trading cycle for our tracking
    tracker = LiveTradingTracker()
    current_status = fetch_live_bot_data()
    
    if "error" not in current_status:
        tracker.record_trading_cycle(current_status)
    
    # 3. Get performance summary
    live_data = tracker.get_performance_summary()
    
    # 4. Create integration data for your dashboard
    integration_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dashboard_updated": dashboard_updated,
        "live_bot_status": current_status,
        "tracking_summary": live_data,
        "status": "integrated" if dashboard_updated else "partial"
    }
    
    # 5. Save integration status
    integration_file = "live_trading/dashboard_integration.json"
    with open(integration_file, 'w') as f:
        json.dump(integration_data, f, indent=2)
    
    print(f"✅ Dashboard integration {'successful' if dashboard_updated else 'partial'}")
    return integration_data

if __name__ == "__main__":
    print("📊 Dashboard Integration System")
    print("="*50)
    
    # Update your existing dashboard with live data
    data = update_tearsheet_data()
    
    print(f"\n📈 Integration Summary:")
    tracking = data["tracking_summary"]
    print(f"   Dashboard Status: {'✅ Updated' if data['dashboard_updated'] else '⚠️ Partial'}")
    print(f"   Total Cycles: {tracking.get('total_cycles', 0)}")
    print(f"   Total Signals: {tracking.get('total_signals', 0)}")
    print(f"   Signal Rate: {tracking.get('signal_rate', 0)*100:.1f}%")
    
    # Show current bot status
    current = data["live_bot_status"]
    if "error" not in current:
        print(f"\n🤖 Live Bot Status: ✅ ACTIVE")
        if "signal" in current:
            print(f"   Latest Signal: {current['signal'].get('signal', 'N/A')}")
            print(f"   Confidence: {current['signal'].get('confidence', 0):.3f}")
    else:
        print(f"\n🤖 Live Bot Status: ⚠️ {current['error']}")
    
    print(f"\n� Your Dashboard Integration:")
    print(f"   🌐 Live Dashboard: https://my-trading-robot-1-hlu5e6f29-aidan-lanes-projects.vercel.app")
    print(f"   📈 Live Data File: live_trading/live_bot_state.json")
    print(f"   � Integration Status: live_trading/dashboard_integration.json")
    print(f"\n✨ Your existing dashboard will now show live bot data in real-time!")
