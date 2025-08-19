#!/usr/bin/env python3
"""
VS Code GitHub Actions Fix
This script helps resolve the authentication and refresh issues
"""

import subprocess
import json
import os

def fix_vscode_github():
    """Fix VS Code GitHub integration"""
    print("🔧 FIXING VS CODE GITHUB ACTIONS")
    print("=" * 50)
    
    # Step 1: Check Git configuration
    print("\n1. Checking Git Configuration...")
    try:
        user_name = subprocess.check_output(['git', 'config', '--global', 'user.name'], text=True).strip()
        user_email = subprocess.check_output(['git', 'config', '--global', 'user.email'], text=True).strip()
        print(f"   ✅ Git User: {user_name} ({user_email})")
    except:
        print("   ❌ Git not configured properly")
        return
    
    # Step 2: Test GitHub connectivity
    print("\n2. Testing GitHub API Access...")
    try:
        result = subprocess.run(['git', 'ls-remote', 'origin'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ GitHub repository access working")
        else:
            print("   ❌ Cannot access GitHub repository")
            return
    except:
        print("   ❌ GitHub connectivity issues")
        return
    
    # Step 3: Create VS Code settings for GitHub
    print("\n3. Creating VS Code GitHub Settings...")
    
    settings = {
        "github.gitAuthentication": True,
        "github.gitProtocol": "https", 
        "git.autofetch": True,
        "git.enableSmartCommit": True,
        "github-actions.workflows.pinned.refreshInterval": 60,
        "github-actions.remote.head": "main",
        "github-actions.remote.name": "origin"
    }
    
    # Get VS Code settings path
    settings_path = os.path.expanduser("~/AppData/Roaming/Code/User/settings.json")
    
    try:
        # Read existing settings
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                existing_settings = json.load(f)
        else:
            existing_settings = {}
        
        # Merge GitHub settings
        existing_settings.update(settings)
        
        # Write back to settings
        with open(settings_path, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        
        print("   ✅ VS Code settings updated")
        
    except Exception as e:
        print(f"   ⚠️ Could not update settings: {e}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Restart VS Code completely (Ctrl+Shift+P -> Developer: Reload Window)")
    print("2. Open Command Palette (Ctrl+Shift+P)")
    print("3. Type: 'GitHub: Sign In' and authenticate")
    print("4. Try the refresh button again")
    print("\n💡 Alternative: Use browser for GitHub Actions")
    print("   https://github.com/iPoundSaversBag/my-trading-robot-1/actions")

if __name__ == "__main__":
    fix_vscode_github()
