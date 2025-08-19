#!/usr/bin/env python3
"""
Trading Bot Monitor
Real-time monitoring of your background trading bot
"""
import json
import requests
import time
from datetime import datetime

# Your bot configuration
BOT_URL = "https://my-trading-robot-1.vercel.app"
BOT_SECRET = "93699b3917045092715b8e16c01f2e1d"

def check_bot_status():
    """Check if the trading bot is responding"""
    try:
        headers = {
            "Authorization": f"Bearer {BOT_SECRET}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{BOT_URL}/api/live-bot", headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {"status": "success", "data": data}
        else:
            return {"status": "error", "code": response.status_code, "text": response.text}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def monitor_dashboard():
    """Check if the dashboard is accessible"""
    try:
        response = requests.get(BOT_URL, timeout=10)
        return {"dashboard": "online" if response.status_code == 200 else "offline"}
    except:
        return {"dashboard": "offline"}

def print_status(bot_result, dashboard_result):
    """Print formatted status information"""
    print(f"\n{'='*60}")
    print(f"🤖 Trading Bot Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Dashboard status
    dashboard_status = dashboard_result.get("dashboard", "unknown")
    dashboard_emoji = "🟢" if dashboard_status == "online" else "🔴"
    print(f"{dashboard_emoji} Dashboard: {dashboard_status.upper()}")
    print(f"   URL: {BOT_URL}")
    
    # Bot status
    if bot_result["status"] == "success":
        print(f"🟢 Trading Bot: ACTIVE")
        data = bot_result["data"]
        
        if "signal" in data:
            signal = data["signal"]
            print(f"   📊 Signal: {signal.get('signal', 'N/A')}")
            print(f"   🎯 Confidence: {signal.get('confidence', 0):.2f}")
            print(f"   💡 Reason: {signal.get('reason', 'N/A')}")
            
        if "account_balance" in data:
            balances = data["account_balance"]
            print(f"   💰 Account: {len(balances)} assets")
            for asset, amount in list(balances.items())[:3]:  # Show first 3
                print(f"      {asset}: {amount}")
                
        if "trade_executed" in data and data["trade_executed"]:
            trade = data["trade_executed"]
            print(f"   🔄 Last Trade: {trade.get('side', 'N/A')} {trade.get('quantity', 'N/A')}")
            
    else:
        print(f"🔴 Trading Bot: ERROR")
        if "code" in bot_result:
            print(f"   Status Code: {bot_result['code']}")
        if "message" in bot_result:
            print(f"   Error: {bot_result['message']}")

def main():
    """Main monitoring function"""
    print("🚀 Starting Trading Bot Monitor...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Check both dashboard and bot
            dashboard_result = monitor_dashboard()
            bot_result = check_bot_status()
            
            # Print status
            print_status(bot_result, dashboard_result)
            
            # Wait 30 seconds before next check
            print(f"\n⏱️  Next check in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main()
