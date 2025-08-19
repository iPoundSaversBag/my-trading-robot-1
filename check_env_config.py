#!/usr/bin/env python3
"""
Environment Variables Verification Script
Checks if all required environment variables are properly configured
"""

import os
from pathlib import Path

def load_env_file():
    """Load .env file if it exists"""
    env_file = Path('.env')
    if env_file.exists():
        print(f"✅ Found .env file: {env_file.absolute()}")
        with open(env_file, 'r') as f:
            content = f.read()
            
        # Count non-empty, non-comment lines
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        print(f"✅ Found {len(lines)} environment variables")
        return True
    else:
        print("❌ No .env file found")
        return False

def check_required_vars():
    """Check if all required environment variables are set"""
    required_vars = {
        'API_KEY': 'Binance API Key',
        'SECRET_KEY': 'Binance Secret Key', 
        'BINANCE_API_KEY': 'Binance API Key (Vercel)',
        'BINANCE_API_SECRET': 'Binance Secret Key (Vercel)',
        'VERCEL_URL': 'Vercel deployment URL',
        'BOT_SECRET': 'Bot authentication secret',
        'CRON_SECRET': 'Cron job authentication'
    }
    
    print("\n🔍 Checking Required Environment Variables:")
    print("=" * 50)
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'SECRET' in var or 'KEY' in var:
                masked_value = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET ({description})")
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def check_github_workflows():
    """Check if GitHub Actions workflows exist and are properly configured"""
    workflows_dir = Path('.github/workflows')
    if not workflows_dir.exists():
        print("\n❌ No .github/workflows directory found")
        return False
    
    required_workflows = [
        'bidirectional-sync.yml',
        'trading-bot.yml'
    ]
    
    print(f"\n🔍 Checking GitHub Actions Workflows:")
    print("=" * 50)
    
    all_exist = True
    for workflow in required_workflows:
        workflow_path = workflows_dir / workflow
        if workflow_path.exists():
            print(f"✅ {workflow}: EXISTS")
        else:
            print(f"❌ {workflow}: MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("🔐 ENVIRONMENT CONFIGURATION CHECKER")
    print("=" * 60)
    
    # Load environment variables from .env
    from dotenv import load_dotenv
    try:
        load_dotenv()
        print("✅ Loaded .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed - using system environment only")
    except:
        print("⚠️  Could not load .env file")
    
    # Check .env file
    env_file_ok = load_env_file()
    
    # Check required variables
    vars_ok, missing_vars = check_required_vars()
    
    # Check workflows
    workflows_ok = check_github_workflows()
    
    # Summary
    print("\n📊 CONFIGURATION STATUS:")
    print("=" * 60)
    
    if env_file_ok and vars_ok and workflows_ok:
        print("✅ ALL SYSTEMS READY!")
        print("✅ Environment variables configured")
        print("✅ GitHub Actions workflows present")
        print("✅ Bidirectional sync should work!")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Push changes to GitHub")
        print("2. Check GitHub Actions runs in repository")
        print("3. Monitor bidirectional sync logs")
        
    else:
        print("⚠️  CONFIGURATION ISSUES FOUND:")
        if not env_file_ok:
            print("   - .env file missing or empty")
        if not vars_ok:
            print(f"   - Missing variables: {', '.join(missing_vars)}")
        if not workflows_ok:
            print("   - GitHub Actions workflows missing")
        
        print("\n🔧 FIXES NEEDED:")
        print("1. Ensure .env file exists with all required variables")
        print("2. Add missing environment variables")
        print("3. Verify GitHub Actions workflows are present")

if __name__ == "__main__":
    main()
