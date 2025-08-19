#!/usr/bin/env python3
"""
GitHub Actions Testing Guide and Monitor Script
"""

import requests
import json
import time
from datetime import datetime

def test_github_workflows():
    """Test GitHub Actions workflows after upload"""
    print("🔄 GITHUB ACTIONS TESTING GUIDE")
    print("=" * 60)
    
    print("\n📋 STEP 1: WORKFLOW VALIDATION")
    print("-" * 30)
    print("After git push, check:")
    print("✅ Go to: https://github.com/YOUR_USERNAME/my-trading-robot-1/actions")
    print("✅ Verify workflows appear without syntax errors")
    print("✅ Check 'bidirectional-sync.yml' and 'trading-bot.yml' are listed")
    
    print("\n📋 STEP 2: MANUAL WORKFLOW TESTING")
    print("-" * 30)
    print("Test bidirectional sync manually:")
    print("1. Go to Actions tab → 'Bidirectional Trading Sync'")
    print("2. Click 'Run workflow' button")
    print("3. Select action type:")
    print("   - 'trigger-backtest' (test optimization trigger)")
    print("   - 'sync-live-results' (test data download)")
    print("   - 'update-parameters' (test config sync)")
    print("4. Monitor execution logs")
    
    print("\n📋 STEP 3: AUTOMATED TRADING TEST")
    print("-" * 30)
    print("The trading-bot.yml runs every 5 minutes:")
    print("✅ Wait 5 minutes after upload")
    print("✅ Check Actions tab for automatic runs")
    print("✅ Verify it calls your Vercel endpoint")
    
    print("\n📋 STEP 4: VERCEL DEPLOYMENT TEST")
    print("-" * 30)
    print("Test the live bot endpoint:")
    print("✅ URL: https://my-trading-robot-1-f22n1sboq-aidan-lanes-projects.vercel.app/api/live-bot")
    print("✅ Should return trading signals and market analysis")
    print("✅ Check Vercel dashboard for deployment logs")

def test_vercel_endpoint():
    """Test the Vercel endpoint directly"""
    print("\n🌐 VERCEL ENDPOINT TESTING")
    print("=" * 60)
    
    vercel_url = "https://my-trading-robot-1-f22n1sboq-aidan-lanes-projects.vercel.app"
    bot_secret = "93699b3917045092715b8e16c01f2e1d"
    
    endpoints_to_test = [
        "/api/live-bot",
        "/api/live-data", 
        "/api/trading-bot",
        "/api/backtest-sync"
    ]
    
    for endpoint in endpoints_to_test:
        print(f"\n🔄 Testing {endpoint}...")
        try:
            headers = {"Authorization": f"Bearer {bot_secret}"}
            response = requests.get(f"{vercel_url}{endpoint}", 
                                  headers=headers, 
                                  timeout=30)
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response received")
                if 'signal' in data:
                    print(f"   Signal: {data.get('signal', {}).get('signal', 'none')}")
                if 'status' in data:
                    print(f"   Status: {data.get('status', 'unknown')}")
            else:
                print(f"   ⚠️ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def monitor_bidirectional_sync():
    """Monitor bidirectional sync in action"""
    print("\n🔄 BIDIRECTIONAL SYNC MONITORING")
    print("=" * 60)
    
    print("After triggering workflows, monitor these files for updates:")
    print("\n📁 Local Files to Watch:")
    print("   ✅ logs/remote_backtest_requests.jsonl")
    print("   ✅ live_trading/remote_sync_results.jsonl") 
    print("   ✅ live_trading/live_bot_state.json")
    print("   ✅ api/live_trading_config.json")
    
    print("\n📊 What to Expect:")
    print("   🔄 trigger-backtest: Creates optimization request logs")
    print("   📥 sync-live-results: Downloads live bot data from Vercel")
    print("   📤 update-parameters: Uploads optimized parameters to Vercel")
    print("   🔄 Automatic commits: GitHub will commit and push changes")

def create_test_commands():
    """Create test commands for manual testing"""
    print("\n💻 MANUAL TEST COMMANDS")
    print("=" * 60)
    
    commands = [
        {
            "name": "Test Vercel Live Bot",
            "command": "curl -H 'Authorization: Bearer 93699b3917045092715b8e16c01f2e1d' https://my-trading-robot-1-f22n1sboq-aidan-lanes-projects.vercel.app/api/live-bot",
            "expected": "Trading signal with market analysis"
        },
        {
            "name": "Trigger Bidirectional Sync",
            "command": "Repository dispatch via GitHub API or manual workflow run",
            "expected": "Workflow execution in Actions tab"
        },
        {
            "name": "Check Trading Bot Runs",
            "command": "Monitor GitHub Actions every 5 minutes",
            "expected": "Automatic trading bot execution"
        }
    ]
    
    for cmd in commands:
        print(f"\n🔧 {cmd['name']}:")
        print(f"   Command: {cmd['command']}")
        print(f"   Expected: {cmd['expected']}")

if __name__ == "__main__":
    print("🧪 GITHUB & VERCEL TESTING STRATEGY")
    print("=" * 70)
    print(f"Testing guide generated at: {datetime.now()}")
    print("=" * 70)
    
    test_github_workflows()
    test_vercel_endpoint()
    monitor_bidirectional_sync() 
    create_test_commands()
    
    print("\n🎯 TESTING CHECKLIST SUMMARY")
    print("=" * 70)
    print("1. ✅ Push to GitHub and verify workflows appear")
    print("2. ✅ Manually test bidirectional sync workflows")
    print("3. ✅ Wait 5 minutes and check automated trading runs")
    print("4. ✅ Test Vercel endpoints directly")
    print("5. ✅ Monitor file changes from sync operations")
    print("6. ✅ Verify GitHub commits from bidirectional sync")
    print("=" * 70)
