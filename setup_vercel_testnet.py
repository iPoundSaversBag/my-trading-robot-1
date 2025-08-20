#!/usr/bin/env python3
"""
Vercel Environment Setup for Testnet Trading
This script shows you exactly what environment variables to set on Vercel
"""

import os
from dotenv import load_dotenv

def show_vercel_env_setup():
    """Show required Vercel environment variables for testnet"""
    
    print("🔧 VERCEL ENVIRONMENT SETUP FOR TESTNET")
    print("=" * 60)
    
    # Load local environment
    load_dotenv()
    
    # Required environment variables for Vercel
    required_vars = {
        'BINANCE_API_KEY': os.environ.get('BINANCE_API_KEY', ''),
        'BINANCE_API_SECRET': os.environ.get('BINANCE_API_SECRET', ''),
        'BOT_SECRET': os.environ.get('BOT_SECRET', ''),
        'CRON_SECRET': os.environ.get('CRON_SECRET', '')
    }
    
    print("📋 Required Environment Variables for Vercel:")
    print("-" * 60)
    
    for var_name, var_value in required_vars.items():
        if var_value:
            # Show partial value for security
            display_value = f"{var_value[:8]}...{var_value[-8:]}" if len(var_value) > 16 else var_value
            print(f"✅ {var_name} = {display_value}")
        else:
            print(f"❌ {var_name} = NOT SET")
    
    print(f"\n🚀 VERCEL DEPLOYMENT INSTRUCTIONS:")
    print(f"=" * 60)
    print(f"1. Go to: https://vercel.com/dashboard")
    print(f"2. Select your project: my-trading-robot-1")
    print(f"3. Go to Settings → Environment Variables")
    print(f"4. Add these variables:")
    print()
    
    for var_name, var_value in required_vars.items():
        if var_value:
            print(f"   Name: {var_name}")
            print(f"   Value: {var_value}")
            print(f"   Environment: Production, Preview, Development")
            print()
    
    print(f"5. After adding variables, go to Deployments tab")
    print(f"6. Click 'Redeploy' on the latest deployment")
    print(f"7. Wait 2-3 minutes for deployment to complete")
    print(f"8. Test with: python test_vercel_testnet_deployment.py")
    
    print(f"\n⚡ QUICK SETUP COMMANDS:")
    print(f"=" * 60)
    print(f"# If you have Vercel CLI installed:")
    print(f"vercel env add BINANCE_API_KEY")
    print(f"vercel env add BINANCE_API_SECRET") 
    print(f"vercel env add BOT_SECRET")
    print(f"vercel env add CRON_SECRET")
    print(f"vercel --prod")
    
    print(f"\n🎯 EXPECTED RESULT:")
    print(f"=" * 60)
    print(f"After setup, your bot will:")
    print(f"✅ Use Binance testnet endpoint (https://testnet.binance.vision)")
    print(f"✅ Execute real trades on testnet (safe)")
    print(f"✅ Show live testnet data in dashboard")
    print(f"✅ Display trades in HTML reports")
    
    return required_vars

def create_env_file_template():
    """Create a template for Vercel environment setup"""
    
    load_dotenv()
    
    template = f"""# Vercel Environment Variables for Testnet Trading
# Copy these values to your Vercel project settings

BINANCE_API_KEY={os.environ.get('BINANCE_API_KEY', 'YOUR_TESTNET_API_KEY')}
BINANCE_API_SECRET={os.environ.get('BINANCE_API_SECRET', 'YOUR_TESTNET_SECRET_KEY')}
BOT_SECRET={os.environ.get('BOT_SECRET', 'YOUR_BOT_SECRET')}
CRON_SECRET={os.environ.get('CRON_SECRET', 'YOUR_CRON_SECRET')}

# These are used by the live trading bot on Vercel
# Make sure they match your local .env file exactly
"""
    
    with open('vercel-env-template.txt', 'w') as f:
        f.write(template)
    
    print(f"✅ Created vercel-env-template.txt with your values")
    return template

if __name__ == "__main__":
    print("🔧 Vercel Environment Configuration Tool")
    print("=" * 60)
    
    # Show setup instructions
    env_vars = show_vercel_env_setup()
    
    # Create template file
    create_env_file_template()
    
    print(f"\n📁 Files Created:")
    print(f"   📄 vercel-env-template.txt - Your environment variables")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Copy variables to Vercel dashboard")
    print(f"   2. Redeploy on Vercel")
    print(f"   3. Test with: python test_vercel_testnet_deployment.py")
    
    # Check if all required vars are present
    missing = [name for name, value in env_vars.items() if not value]
    if missing:
        print(f"\n⚠️  Missing Variables: {', '.join(missing)}")
        print(f"   Check your .env file first")
    else:
        print(f"\n✅ All environment variables ready for Vercel!")
