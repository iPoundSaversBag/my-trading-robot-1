#!/usr/bin/env python3
"""
GitHub Actions Status Checker
Check if your workflow is running properly
"""
import requests
import json
from datetime import datetime

def check_github_actions():
    """Check GitHub Actions workflow status via API"""
    repo = "iPoundSaversBag/my-trading-robot-1"
    url = f"https://api.github.com/repos/{repo}/actions/workflows"
    
    print(f"🔍 Checking GitHub Actions for {repo}...")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            workflows = response.json()
            print(f"✅ Found {workflows['total_count']} workflows")
            
            for workflow in workflows['workflows']:
                print(f"📋 Workflow: {workflow['name']}")
                print(f"   State: {workflow['state']}")
                print(f"   Path: {workflow['path']}")
                
            # Get recent runs
            runs_url = f"https://api.github.com/repos/{repo}/actions/runs"
            runs_response = requests.get(runs_url, timeout=10)
            
            if runs_response.status_code == 200:
                runs = runs_response.json()
                print(f"\n🏃 Recent workflow runs:")
                
                for run in runs['workflow_runs'][:5]:  # Show last 5 runs
                    status_emoji = "✅" if run['status'] == 'completed' and run['conclusion'] == 'success' else "❌" if run['status'] == 'completed' else "🟡"
                    print(f"   {status_emoji} {run['created_at'][:19]} - Status: {run['status']} - Conclusion: {run['conclusion']}")
                
                return True
            else:
                print(f"❌ Could not get workflow runs: {runs_response.status_code}")
                return False
                
        else:
            print(f"❌ Could not access GitHub API: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking GitHub Actions: {e}")
        return False

def main():
    print(f"🤖 GitHub Actions Status Check")
    print(f"{'='*50}")
    
    result = check_github_actions()
    
    if result:
        print(f"\n✅ GitHub Actions are accessible!")
        print(f"🔗 Manual check: https://github.com/iPoundSaversBag/my-trading-robot-1/actions")
    else:
        print(f"\n❌ Could not verify GitHub Actions status")
        print(f"🔗 Please check manually: https://github.com/iPoundSaversBag/my-trading-robot-1/actions")
    
    print(f"\n💡 What to look for:")
    print(f"   ✅ Green checkmarks appearing every 5 minutes")
    print(f"   🟡 Yellow dots (currently running)")
    print(f"   ❌ Red X's (failures that need attention)")

if __name__ == "__main__":
    main()
