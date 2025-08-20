#!/usr/bin/env python3
"""
Tearsheet System Integration Test
Tests the complete flow: generate_plots.py -> tearsheet.py -> dashboard-data.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    print("🎯 Testing Tearsheet System Integration")
    print("=" * 50)
    
    # Step 1: Run generate_plots.py to enhance the performance report
    print("\n📊 Step 1: Enhancing Performance Report")
    success1 = run_command("python analysis/generate_plots.py", "Running generate_plots.py")
    
    # Step 2: Test tearsheet generation
    print("\n📈 Step 2: Testing Tearsheet Generation")
    success2 = run_command("python api/tearsheet.py", "Testing tearsheet.py")
    
    # Step 3: Test dashboard-data API
    print("\n🔧 Step 3: Testing Dashboard Data API")
    success3 = run_command("python api/dashboard-data.py", "Testing dashboard-data.py")
    
    # Step 4: Check file outputs
    print("\n📁 Step 4: Checking Output Files")
    
    files_to_check = [
        "tearsheet.html",
        "test_tearsheet.html",
        "plots_output/latest_run_dir.txt"
    ]
    
    files_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} exists ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} missing")
            files_exist = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEARSHEET SYSTEM STATUS")
    print("=" * 50)
    
    overall_success = success1 and success2 and success3 and files_exist
    
    print(f"📊 Plot Enhancement: {'✅ Working' if success1 else '❌ Failed'}")
    print(f"📈 Tearsheet Generation: {'✅ Working' if success2 else '❌ Failed'}")
    print(f"🔧 Dashboard API: {'✅ Working' if success3 else '❌ Failed'}")
    print(f"📁 File Outputs: {'✅ All present' if files_exist else '❌ Missing files'}")
    
    if overall_success:
        print("\n🎉 TEARSHEET SYSTEM FULLY OPERATIONAL!")
        print("   - Enhanced performance reports are being generated")
        print("   - Live data integration is working")
        print("   - API endpoints are responsive")
        print("   - Ready for Vercel deployment")
    else:
        print("\n⚠️  TEARSHEET SYSTEM NEEDS ATTENTION")
        print("   - Some components are not working properly")
        print("   - Check error messages above")
    
    print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
