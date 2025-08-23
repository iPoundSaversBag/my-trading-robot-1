#!/usr/bin/env python3
"""
Final System Verification
Check that all import issues are resolved
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_imports():
    """Verify all critical imports work"""
    
    print("🔍 VERIFYING SYSTEM IMPORTS")
    print("=" * 50)
    
    # Test 1: Unified monitor
    try:
        import unified_live_monitor
        from unified_live_monitor import get_monitor, on_backtest_complete
        print("✅ unified_live_monitor imports successfully")
    except Exception as e:
        print(f"❌ unified_live_monitor import failed: {e}")
        return False
    
    # Test 2: Watcher hook
    try:
        import watcher_hook
        from watcher_hook import on_backtest_complete as watcher_hook
        print("✅ watcher_hook imports successfully")
    except Exception as e:
        print(f"❌ watcher_hook import failed: {e}")
        return False
    
    # Test 3: Test the unified monitor functions
    try:
        monitor = get_monitor()
        print("✅ unified_live_monitor instance created")
    except Exception as e:
        print(f"❌ unified_live_monitor instance failed: {e}")
        return False
    
    return True

def verify_file_structure():
    """Verify the cleaned up file structure"""
    
    print(f"\n📁 VERIFYING FILE STRUCTURE")
    print("=" * 50)
    
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    # Essential files that should exist
    essential_files = {
        "unified_live_monitor.py": "Main monitoring system",
        "watcher_hook.py": "Watcher integration",
        "watcher.py": "Master orchestration",
        "api/live-bot.py": "Vercel live bot endpoint",
        "api/parameter-sync.py": "Vercel parameter sync endpoint"
    }
    
    # Files that should be gone
    removed_files = [
        "monitor_bot.py",
        "live_monitor.py", 
        "live_trading_tracker.py",
        "automated_pipeline.py",
        "complete_pipeline_check.py"
    ]
    
    print("✅ ESSENTIAL FILES:")
    all_essential_exist = True
    for file, description in essential_files.items():
        file_path = base_dir / file
        if file_path.exists():
            print(f"   ✅ {file} - {description}")
        else:
            print(f"   ❌ {file} - MISSING!")
            all_essential_exist = False
    
    print(f"\n🗑️  REMOVED FILES (should be gone):")
    all_removed = True
    for file in removed_files:
        file_path = base_dir / file
        if not file_path.exists():
            print(f"   ✅ {file} - Successfully removed")
        else:
            print(f"   ❌ {file} - Still exists!")
            all_removed = False
    
    return all_essential_exist and all_removed

def test_basic_functionality():
    """Test basic functionality"""
    
    print(f"\n🧪 TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from unified_live_monitor import quick_status
        status = quick_status()
        print("✅ Quick status check works")
        print(f"   Bot Online: {status['bot_online']}")
        print(f"   Dashboard Online: {status['dashboard_online']}")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run complete verification"""
    
    print("🔧 FINAL SYSTEM VERIFICATION")
    print("=" * 60)
    
    imports_ok = verify_imports()
    structure_ok = verify_file_structure()
    functionality_ok = test_basic_functionality()
    
    print(f"\n📊 VERIFICATION SUMMARY:")
    print("=" * 50)
    print(f"   Imports: {'✅ WORKING' if imports_ok else '❌ FAILED'}")
    print(f"   File Structure: {'✅ CLEAN' if structure_ok else '❌ ISSUES'}")
    print(f"   Basic Functions: {'✅ WORKING' if functionality_ok else '❌ FAILED'}")
    
    if imports_ok and structure_ok and functionality_ok:
        print(f"\n🎉 SYSTEM VERIFICATION PASSED!")
        print(f"   All import issues resolved")
        print(f"   Redundant files successfully removed")
        print(f"   Unified monitoring system operational")
        print(f"\n🚀 READY FOR AUTOMATED TRADING!")
        print(f"   Run: python watcher.py")
        print(f"   Run: python unified_live_monitor.py")
    else:
        print(f"\n⚠️  SYSTEM ISSUES DETECTED")
        print(f"   Review the failed components above")

if __name__ == "__main__":
    main()
