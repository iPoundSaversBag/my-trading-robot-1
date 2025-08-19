#!/usr/bin/env python3
"""
Fix GitHub Actions Panel - No Repositories Issue
This script specifically fixes VS Code's GitHub Actions panel not showing repositories
"""

import subprocess
import json
import os
import time

def fix_github_actions_panel():
    """Fix GitHub Actions panel showing 'no repositories'"""
    print("🔧 FIXING GITHUB ACTIONS PANEL - NO REPOSITORIES")
    print("=" * 60)
    
    # Step 1: Check current directory is a git repo
    print("\n1. Verifying Git Repository...")
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
        current_dir = os.getcwd().replace('\\', '/')
        print(f"   📁 Repository root: {repo_root}")
        print(f"   📁 Current directory: {current_dir}")
        
        if repo_root.replace('\\', '/') not in current_dir:
            print("   ⚠️ Not in repository root - changing directory")
            os.chdir(repo_root)
    except:
        print("   ❌ Not a Git repository")
        return False
    
    # Step 2: Verify GitHub remote
    print("\n2. Checking GitHub Remote...")
    try:
        remote_url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], text=True).strip()
        print(f"   🔗 Remote URL: {remote_url}")
        
        if 'github.com' not in remote_url:
            print("   ❌ Not a GitHub repository")
            return False
            
        # Extract owner/repo from URL
        if remote_url.startswith('https://github.com/'):
            repo_path = remote_url.replace('https://github.com/', '').replace('.git', '')
        elif remote_url.startswith('git@github.com:'):
            repo_path = remote_url.replace('git@github.com:', '').replace('.git', '')
        else:
            print("   ❌ Invalid GitHub URL format")
            return False
            
        print(f"   📋 Repository: {repo_path}")
        
    except:
        print("   ❌ Cannot get remote URL")
        return False
    
    # Step 3: Create/Update VS Code workspace settings
    print("\n3. Updating VS Code Workspace Settings...")
    
    workspace_settings = {
        "settings": {
            "github.gitAuthentication": True,
            "github.gitProtocol": "https",
            "git.autofetch": True,
            "git.fetchOnPull": True,
            "github-actions.workflows.pinned.refreshInterval": 30,
            "github-actions.workflows.pinned.repositories": [repo_path],
            "github-actions.workflows.pinned.workflows": [
                ".github/workflows/bidirectional-sync.yml",
                ".github/workflows/trading-bot.yml", 
                ".github/workflows/deploy.yml"
            ],
            "github.enableExperimentalApi": True,
            "github.copilot.enable": {
                "*": True,
                "yaml": True,
                "plaintext": False,
                "markdown": True
            }
        }
    }
    
    # Create .vscode folder if it doesn't exist
    vscode_dir = ".vscode"
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
        print(f"   📁 Created {vscode_dir} folder")
    
    # Write workspace settings
    workspace_settings_path = os.path.join(vscode_dir, "settings.json")
    try:
        with open(workspace_settings_path, 'w') as f:
            json.dump(workspace_settings["settings"], f, indent=2)
        print(f"   ✅ Updated {workspace_settings_path}")
    except Exception as e:
        print(f"   ⚠️ Could not write workspace settings: {e}")
    
    # Step 4: Update global VS Code settings
    print("\n4. Updating Global VS Code Settings...")
    
    global_settings_path = os.path.expanduser("~/AppData/Roaming/Code/User/settings.json")
    
    try:
        # Read existing settings
        if os.path.exists(global_settings_path):
            with open(global_settings_path, 'r') as f:
                existing_settings = json.load(f)
        else:
            existing_settings = {}
        
        # Add GitHub Actions specific settings
        github_settings = {
            "github.gitAuthentication": True,
            "github.gitProtocol": "https",
            "git.autofetch": True,
            "git.fetchOnPull": True,
            "github-actions.workflows.pinned.refreshInterval": 30,
            "github.enableExperimentalApi": True,
            "github-actions.remote.head": "main",
            "github-actions.remote.name": "origin"
        }
        
        # Merge settings
        existing_settings.update(github_settings)
        
        # Write back to settings
        with open(global_settings_path, 'w') as f:
            json.dump(existing_settings, f, indent=2)
        
        print("   ✅ Global VS Code settings updated")
        
    except Exception as e:
        print(f"   ⚠️ Could not update global settings: {e}")
    
    # Step 5: Force refresh git status
    print("\n5. Refreshing Git Status...")
    try:
        subprocess.run(['git', 'fetch'], check=True, capture_output=True)
        print("   ✅ Git fetch completed")
    except:
        print("   ⚠️ Git fetch failed")
    
    # Step 6: Check GitHub Actions workflows
    print("\n6. Verifying GitHub Actions Workflows...")
    workflows_dir = ".github/workflows"
    if os.path.exists(workflows_dir):
        workflow_files = [f for f in os.listdir(workflows_dir) if f.endswith('.yml') or f.endswith('.yaml')]
        print(f"   📋 Found {len(workflow_files)} workflow files:")
        for workflow in workflow_files:
            print(f"      - {workflow}")
    else:
        print("   ❌ No .github/workflows directory found")
        return False
    
    print("\n" + "=" * 60)
    print("🚀 GITHUB ACTIONS PANEL FIX COMPLETE!")
    print("=" * 60)
    
    print("\n📋 REQUIRED STEPS TO COMPLETE:")
    print("1. 🔄 RESTART VS Code completely (File → Exit → Reopen)")
    print("2. 🔐 Sign in to GitHub in VS Code:")
    print("   • Ctrl+Shift+P → 'GitHub: Sign In'")
    print("   • Complete authentication in browser")
    print("3. 🔍 Open GitHub Actions panel:")
    print("   • View → Command Palette → 'GitHub Actions: Focus on GitHub Actions View'")
    print("4. 🔄 If still no repos, try:")
    print("   • Ctrl+Shift+P → 'Developer: Reload Window'")
    print("   • Or close/reopen the workspace folder")
    
    print("\n🌐 Alternative Access:")
    print(f"   Browser: https://github.com/{repo_path}/actions")
    
    print("\n💡 Troubleshooting:")
    print("   • Ensure GitHub extension is installed and enabled")
    print("   • Check VS Code is opened in the repository root folder")
    print("   • Verify internet connection for GitHub API access")
    
    return True

if __name__ == "__main__":
    success = fix_github_actions_panel()
    if success:
        print("\n✅ Fix applied successfully!")
    else:
        print("\n❌ Fix failed - check the errors above")
