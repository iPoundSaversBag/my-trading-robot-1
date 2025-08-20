#!/usr/bin/env python3
"""
Debug Vercel Deployment Status
Check what's actually deployed vs what we expected
"""

import requests
import re

def check_current_deployment():
    """Check what's currently on Vercel"""
    
    print("🔍 DEBUGGING VERCEL DEPLOYMENT")
    print("=" * 60)
    
    dashboard_url = "https://my-trading-robot-1.vercel.app"
    
    try:
        response = requests.get(dashboard_url, timeout=15)
        
        if response.status_code == 200:
            html_content = response.text
            
            print("✅ Dashboard accessible")
            
            # Check specific sections we modified
            print("\n🔧 Checking Our Recent Changes:")
            
            # 1. Check for space-y properties
            space_y_matches = re.findall(r'space-y-\d+', html_content)
            if space_y_matches:
                print(f"   ❌ Found {len(space_y_matches)} space-y properties still present:")
                for match in space_y_matches[:5]:  # Show first 5
                    print(f"      • {match}")
            else:
                print(f"   ✅ No space-y properties found")
            
            # 2. Check for our API integration
            if "/api/live-bot" in html_content:
                print(f"   ✅ Live bot API endpoint found")
            else:
                print(f"   ❌ Live bot API endpoint missing")
            
            if "updateLiveTestnetData" in html_content:
                print(f"   ✅ Testnet data function found")
            else:
                print(f"   ❌ Testnet data function missing")
            
            # 3. Check for authorization headers
            if "Authorization" in html_content and "Bearer" in html_content:
                print(f"   ✅ API authorization found")
            else:
                print(f"   ❌ API authorization missing")
            
            # 4. Check the live data panel section specifically
            live_data_section = ""
            if "Live Data" in html_content:
                # Extract the live data section
                start_idx = html_content.find("Live Data")
                if start_idx != -1:
                    end_idx = html_content.find("</div>", start_idx + 500)
                    if end_idx != -1:
                        live_data_section = html_content[start_idx:end_idx + 6]
                        print(f"\n📡 Live Data Section Found:")
                        # Check for key elements
                        if "realBotSignal" in live_data_section:
                            print(f"   ✅ Bot signal element present")
                        if "realBotBalance" in live_data_section:
                            print(f"   ✅ Balance element present")
                        if "realBotTrade" in live_data_section:
                            print(f"   ✅ Trade element present")
            
            if not live_data_section:
                print(f"\n❌ Live Data section not found or incomplete")
            
            # 5. Look for our specific changes
            expected_changes = [
                "margin-top: 0.5rem",
                "margin-bottom: 0.5rem", 
                "loadRealBotData",
                "setInterval.*loadRealBotData.*5000"
            ]
            
            print(f"\n🎯 Expected Changes Status:")
            for change in expected_changes:
                if re.search(change, html_content):
                    print(f"   ✅ Found: {change}")
                else:
                    print(f"   ❌ Missing: {change}")
            
            return html_content
            
        else:
            print(f"❌ Dashboard not accessible: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def check_git_vs_vercel():
    """Compare local files with what's deployed"""
    
    print(f"\n📁 CHECKING LOCAL vs DEPLOYED")
    print("=" * 60)
    
    # Read our local index.html
    try:
        with open("public/index.html", "r", encoding="utf-8") as f:
            local_content = f.read()
        
        print("✅ Local index.html read")
        
        # Check what we expect to be there
        local_checks = []
        
        if "margin-top: 0.5rem" in local_content:
            local_checks.append("CSS margin fixes present")
        
        if "/api/live-bot" in local_content:
            local_checks.append("API endpoint present")
        
        if "updateLiveTestnetData" in local_content:
            local_checks.append("Testnet function present")
        
        if "space-y" not in local_content:
            local_checks.append("No space-y properties")
        
        print(f"\n✅ Local File Status:")
        for check in local_checks:
            print(f"   • {check}")
        
        return local_content
        
    except Exception as e:
        print(f"❌ Cannot read local file: {e}")
        return None

def main():
    """Debug the deployment issue"""
    
    # Check current deployment
    deployed_content = check_current_deployment()
    
    # Check local files
    local_content = check_git_vs_vercel()
    
    if deployed_content and local_content:
        print(f"\n🔍 DIAGNOSIS:")
        
        # Compare key sections
        if len(deployed_content) != len(local_content):
            print(f"   📏 File sizes differ: Local={len(local_content)}, Deployed={len(deployed_content)}")
        
        # Check if our changes are in local but not deployed
        local_has_fixes = (
            "margin-top: 0.5rem" in local_content and
            "/api/live-bot" in local_content and
            "updateLiveTestnetData" in local_content
        )
        
        deployed_has_fixes = (
            "margin-top: 0.5rem" in deployed_content and
            "/api/live-bot" in deployed_content and
            "updateLiveTestnetData" in deployed_content
        )
        
        if local_has_fixes and not deployed_has_fixes:
            print(f"   🚨 ISSUE: Local has fixes, but Vercel doesn't!")
            print(f"   💡 SOLUTION: Need to trigger new deployment")
        elif local_has_fixes and deployed_has_fixes:
            print(f"   ✅ Both local and deployed have fixes")
        else:
            print(f"   ⚠️  Fixes missing from local files")
    
    return deployed_content is not None

if __name__ == "__main__":
    main()
