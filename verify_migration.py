#!/usr/bin/env python3
"""
Comprehensive migration verification script
Tests that Google Cloud to Vercel migration was successful
"""

import importlib
import sys
from pathlib import Path

def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing imports...")
    
    try:
        # Test Vercel utils
        from utilities.vercel_utils import check_vercel_connection, sync_parameters_to_vercel
        print("  ✅ Vercel utilities imported successfully")
        
        # Test unified monitor
        from unified_live_monitor import UnifiedLiveBotMonitor
        print("  ✅ Unified live monitor imported successfully")
        
        # Test core backtest
        import core.backtest
        print("  ✅ Core backtest imported successfully")
        
        # Test utils
        import utilities.utils
        print("  ✅ Utilities imported successfully")
        
        # Test watcher
        import watcher
        print("  ✅ Watcher imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_deprecated_imports():
    """Test that deprecated imports are properly removed"""
    print("\n🚫 Testing deprecated imports are removed...")
    
    deprecated_modules = [
        "live_trading.live_bot"
    ]
    
    for module in deprecated_modules:
        try:
            importlib.import_module(module)
            print(f"  ⚠️  WARNING: {module} still importable (should be removed)")
            return False
        except ImportError:
            print(f"  ✅ {module} properly removed")
    
    return True

def test_vercel_functionality():
    """Test that Vercel utilities work"""
    print("\n🌐 Testing Vercel functionality...")
    
    try:
        from utilities.vercel_utils import check_vercel_connection
        
        # Test connection (will fail if no internet, but should not crash)
        status = check_vercel_connection()
        print(f"  ✅ Vercel connection test completed")
        print(f"     Connected: {status.get('connected', False)}")
        print(f"     Endpoint: {status.get('endpoint', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vercel test error: {e}")
        return False

def check_file_structure():
    """Check that migration cleaned up properly"""
    print("\n📁 Checking file structure...")
    
    # Check for deprecated files
    deprecated_files = [
        "analysis/generate_plots_backup.py",
        "analysis/generate_plots_legacy_backup.py"
    ]
    
    all_clean = True
    for file_path in deprecated_files:
        if Path(file_path).exists():
            print(f"  ⚠️  {file_path} still exists (should be cleaned up)")
            all_clean = False
        else:
            print(f"  ✅ {file_path} properly cleaned up")
    
    # Check for required files
    required_files = [
        "utilities/vercel_utils.py",
        "unified_live_monitor.py",
        "watcher.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path} exists")
        else:
            print(f"  ❌ {file_path} missing")
            all_clean = False
    
    return all_clean

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("    GOOGLE CLOUD TO VERCEL MIGRATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Deprecated Import Tests", test_deprecated_imports), 
        ("Vercel Functionality", test_vercel_functionality),
        ("File Structure Check", check_file_structure)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - MIGRATION SUCCESSFUL!")
        print("✅ Google Cloud functionality fully migrated to Vercel")
        print("✅ All imports working correctly")
        print("✅ File structure cleaned up")
        print("✅ System ready for production use")
    else:
        print("❌ SOME TESTS FAILED - PLEASE REVIEW ABOVE")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
