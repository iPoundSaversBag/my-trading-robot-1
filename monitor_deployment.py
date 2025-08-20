#!/usr/bin/env python3
"""
Monitor Vercel Deployment
Wait for our fixes to go live
"""

import requests
import time
import sys

def check_if_deployed():
    """Check if our latest fixes are deployed"""
    
    try:
        response = requests.get("https://my-trading-robot-1.vercel.app", timeout=10)
        
        if response.status_code == 200:
            html_content = response.text
            
            # Look for our key fixes
            has_api_integration = "/api/live-bot" in html_content
            has_testnet_function = "updateLiveTestnetData" in html_content
            has_auth_headers = "Authorization" in html_content and "Bearer" in html_content
            
            fixes_count = sum([has_api_integration, has_testnet_function, has_auth_headers])
            
            return fixes_count >= 2, fixes_count, html_content
        
        return False, 0, ""
        
    except Exception as e:
        return False, 0, ""

def monitor_deployment():
    """Monitor until deployment is complete"""
    
    print("🚀 MONITORING VERCEL DEPLOYMENT")
    print("=" * 50)
    print("Waiting for fixes to go live...")
    
    max_attempts = 20
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        deployed, fixes_count, html_content = check_if_deployed()
        
        print(f"   Attempt {attempt}: {fixes_count}/3 fixes detected", end="")
        
        if deployed:
            print(" ✅ DEPLOYED!")
            
            # Final verification
            print(f"\n🔍 Final Verification:")
            print(f"   API Integration: {'✅' if '/api/live-bot' in html_content else '❌'}")
            print(f"   Testnet Function: {'✅' if 'updateLiveTestnetData' in html_content else '❌'}")
            print(f"   Auth Headers: {'✅' if 'Authorization' in html_content and 'Bearer' in html_content else '❌'}")
            
            return True
        else:
            print(" ⏳")
            
        time.sleep(15)  # Wait 15 seconds between checks
    
    print(f"\n⚠️  Deployment taking longer than expected")
    return False

if __name__ == "__main__":
    if monitor_deployment():
        print(f"\n🎉 DEPLOYMENT COMPLETE!")
        print(f"🌐 Your dashboard is now updated: https://my-trading-robot-1.vercel.app")
        print(f"📡 Live data panel should now show testnet integration")
    else:
        print(f"\n⏰ Monitoring timed out - check manually in a few minutes")
