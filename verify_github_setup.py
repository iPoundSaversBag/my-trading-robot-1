#!/usr/bin/env python3
"""
Verify GitHub Actions Panel Setup
Quick verification that everything is configured correctly
"""

import os
import json
import subprocess

def verify_setup():
    """Verify GitHub Actions panel setup"""
    print("🔍 VERIFYING GITHUB ACTIONS SETUP")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Git repository
    print("\n✅ Check 1: Git Repository")
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
        print(f"   📁 Repository root: {repo_root}")
        checks_passed += 1
    except:
        print("   ❌ Not a Git repository")
    
    # Check 2: GitHub remote
    print("\n✅ Check 2: GitHub Remote")
    try:
        remote_url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], text=True).strip()
        if 'github.com' in remote_url:
            print(f"   🔗 GitHub repository: {remote_url}")
            checks_passed += 1
        else:
            print(f"   ❌ Not a GitHub repository: {remote_url}")
    except:
        print("   ❌ No remote configured")
    
    # Check 3: Workspace settings
    print("\n✅ Check 3: Workspace Settings")
    if os.path.exists(".vscode/settings.json"):
        try:
            with open(".vscode/settings.json", 'r') as f:
                settings = json.load(f)
            if "github-actions.workflows.pinned.repositories" in settings:
                print("   📋 GitHub Actions workspace settings configured")
                checks_passed += 1
            else:
                print("   ⚠️ GitHub Actions settings missing")
        except:
            print("   ❌ Cannot read workspace settings")
    else:
        print("   ❌ No workspace settings file")
    
    # Check 4: Workflows directory
    print("\n✅ Check 4: GitHub Actions Workflows")
    if os.path.exists(".github/workflows"):
        workflow_files = [f for f in os.listdir(".github/workflows") if f.endswith(('.yml', '.yaml'))]
        if workflow_files:
            print(f"   📋 Found {len(workflow_files)} workflow files:")
            for workflow in workflow_files:
                print(f"      - {workflow}")
            checks_passed += 1
        else:
            print("   ❌ No workflow files found")
    else:
        print("   ❌ No .github/workflows directory")
    
    # Check 5: VS Code extensions
    print("\n✅ Check 5: VS Code Extensions")
    try:
        result = subprocess.run(['code', '--list-extensions'], capture_output=True, text=True)
        extensions = result.stdout.lower()
        
        required_extensions = [
            'github.vscode-github-actions',
            'github.vscode-pull-request-github'
        ]
        
        missing_extensions = []
        for ext in required_extensions:
            if ext.lower() not in extensions:
                missing_extensions.append(ext)
        
        if not missing_extensions:
            print("   🔌 All required GitHub extensions installed")
            checks_passed += 1
        else:
            print(f"   ❌ Missing extensions: {missing_extensions}")
    except:
        print("   ⚠️ Cannot check VS Code extensions")
    
    # Check 6: GitHub API access
    print("\n✅ Check 6: GitHub API Access")
    try:
        result = subprocess.run(['git', 'ls-remote', 'origin'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   🌐 GitHub API access working")
            checks_passed += 1
        else:
            print("   ❌ Cannot access GitHub repository")
    except:
        print("   ❌ GitHub connectivity issues")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 SETUP STATUS: {checks_passed}/{total_checks} checks passed")
    print("=" * 50)
    
    if checks_passed == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("\n🚀 TO COMPLETE THE SETUP:")
        print("1. Restart VS Code completely")
        print("2. Open Command Palette (Ctrl+Shift+P)")
        print("3. Type: 'GitHub Actions: Focus on GitHub Actions View'")
        print("4. Your repository should now appear!")
        
    elif checks_passed >= 4:
        print("✅ MOSTLY CONFIGURED - Should work after VS Code restart")
        print("\n🔄 NEXT STEPS:")
        print("1. Restart VS Code")
        print("2. Sign in to GitHub if prompted")
        print("3. Check GitHub Actions panel")
        
    else:
        print("⚠️ SETUP INCOMPLETE - Review failed checks above")
        print("\n🔧 RECOMMENDED ACTION:")
        print("1. Run fix_github_actions_panel.py again")
        print("2. Check internet connection")
        print("3. Verify GitHub authentication")
    
    print(f"\n🌐 Direct access: https://github.com/iPoundSaversBag/my-trading-robot-1/actions")
    
    return checks_passed == total_checks

if __name__ == "__main__":
    verify_setup()
