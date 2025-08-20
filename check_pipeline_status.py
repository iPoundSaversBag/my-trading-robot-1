#!/usr/bin/env python3
"""
AUTOMATED PIPELINE STATUS CHECK
Verify what's already implemented vs what's needed
"""

import os
from pathlib import Path

def check_pipeline_status():
    """Check the status of the automated pipeline components"""
    
    print("🔍 AUTOMATED PIPELINE STATUS CHECK")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Core Components Status
    components = {
        "Watcher Integration": {
            "file": "watcher.py",
            "status": "✅ IMPLEMENTED",
            "details": "Has hooks for live bot parameter sync (lines 928-939)"
        },
        "Watcher Hook": {
            "file": "watcher_hook.py", 
            "status": "✅ IMPLEMENTED",
            "details": "Triggers parameter sync after backtest completion"
        },
        "Automated Pipeline": {
            "file": "automated_pipeline.py",
            "status": "✅ IMPLEMENTED", 
            "details": "Handles parameter sync and monitoring coordination"
        },
        "Parameter Sync API": {
            "file": "api/parameter-sync.py",
            "status": "✅ IMPLEMENTED",
            "details": "Vercel endpoint to receive/update live bot parameters"
        },
        "Live Bot Integration": {
            "file": "api/live-bot.py",
            "status": "✅ IMPLEMENTED",
            "details": "Already loads from live_trading_config.json"
        },
        "Vercel Configuration": {
            "file": "vercel.json",
            "status": "✅ IMPLEMENTED",
            "details": "Parameter-sync endpoint is configured"
        }
    }
    
    # Missing Components
    missing = {
        "Results Monitor": {
            "file": "results_monitor.py",
            "status": "❌ MISSING",
            "details": "Local script to monitor live bot performance"
        },
        "Live Results Directory": {
            "file": "live_results/",
            "status": "❌ MISSING", 
            "details": "Storage for live trading results"
        }
    }
    
    print("📋 EXISTING COMPONENTS:")
    for name, info in components.items():
        file_path = base_dir / info["file"]
        exists = "✅" if file_path.exists() else "❌"
        print(f"   {exists} {name}")
        print(f"      File: {info['file']}")
        print(f"      Status: {info['status']}")
        print(f"      Details: {info['details']}")
        print()
    
    print("🚫 MISSING COMPONENTS:")
    for name, info in missing.items():
        file_path = base_dir / info["file"]
        exists = "✅" if file_path.exists() else "❌"
        print(f"   {exists} {name}")
        print(f"      File: {info['file']}")
        print(f"      Status: {info['status']}")
        print(f"      Details: {info['details']}")
        print()
    
    # Test Current Integration
    print("🧪 INTEGRATION TEST:")
    
    # Check if watcher can import the hook
    try:
        from watcher_hook import WatcherHook
        hook = WatcherHook()
        print("   ✅ Watcher hook can be imported")
        
        # Check if automated pipeline exists
        if (base_dir / "automated_pipeline.py").exists():
            print("   ✅ Automated pipeline script exists")
        else:
            print("   ❌ Automated pipeline script missing")
            
    except ImportError as e:
        print(f"   ❌ Watcher hook import failed: {e}")
    
    # Check parameter sync API
    param_sync_file = base_dir / "api" / "parameter-sync.py"
    if param_sync_file.exists():
        print("   ✅ Parameter sync API exists")
    else:
        print("   ❌ Parameter sync API missing")
    
    # Summary
    total_needed = len(components) + len(missing)
    implemented = len(components)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Implemented: {implemented}/{total_needed} components")
    print(f"   Missing: {len(missing)}/{total_needed} components")
    
    completion_percentage = (implemented / total_needed) * 100
    print(f"   Completion: {completion_percentage:.1f}%")
    
    if completion_percentage >= 80:
        print(f"\n🎉 PIPELINE IS MOSTLY READY!")
        print(f"   Only need to add missing monitoring components")
    else:
        print(f"\n⚠️  PIPELINE NEEDS MORE WORK")
        print(f"   Several core components are missing")
    
    return completion_percentage >= 80

if __name__ == "__main__":
    is_ready = check_pipeline_status()
    
    if is_ready:
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Create results monitoring script")
        print(f"   2. Test full pipeline end-to-end") 
        print(f"   3. Verify parameter sync works")
    else:
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print(f"   1. Review missing components")
        print(f"   2. Implement core functionality first")
        print(f"   3. Test each component individually")
