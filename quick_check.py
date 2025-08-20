#!/usr/bin/env python3
"""
Quick Deployment Check
Check if our forced deployment worked
"""

import requests
import time

def quick_check():
    """Quick check for deployment"""
    
    print("🔍 QUICK DEPLOYMENT CHECK")
    print("=" * 40)
    
    for attempt in range(6):  # Check 6 times, 30 seconds apart
        try:
            response = requests.get("https://my-trading-robot-1.vercel.app", timeout=10)
            
            if response.status_code == 200:
                html_content = response.text
                
                # Check for our timestamp
                has_timestamp = "Updated: 2025-08-20" in html_content
                
                # Check for our API integration
                has_api = "/api/live-bot" in html_content
                has_function = "updateLiveTestnetData" in html_content
                has_auth = "Authorization" in html_content and "Bearer" in html_content
                
                fixes_present = sum([has_api, has_function, has_auth])
                
                print(f"Attempt {attempt + 1}:")
                print(f"   Timestamp: {'✅' if has_timestamp else '❌'}")
                print(f"   API Integration: {fixes_present}/3 fixes present")
                
                if has_timestamp and fixes_present >= 2:
                    print(f"\n🎉 DEPLOYMENT SUCCESS!")
                    print(f"   Your live data panel fixes are now deployed")
                    print(f"   🌐 Dashboard: https://my-trading-robot-1.vercel.app")
                    return True
                
            else:
                print(f"Attempt {attempt + 1}: Status {response.status_code}")
                
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
        
        if attempt < 5:  # Don't wait after last attempt
            print(f"   Waiting 30 seconds...")
            time.sleep(30)
    
    print(f"\n⏰ Still deploying - check again in a few minutes")
    return False

if __name__ == "__main__":
    quick_check()
