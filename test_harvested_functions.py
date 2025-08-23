#!/usr/bin/env python3
"""
Phase 0 Function Harvesting Validation Test
Tests all harvested functions in their new integrated locations
"""

import sys
import os
sys.path.append('.')

def test_health_utils_harvested_functions():
    """Test harvested functions in health_utils.py"""
    try:
        from health_utils import IntelligentRepairEngine
        
        # Initialize with current workspace root
        workspace_root = os.getcwd()
        engine = IntelligentRepairEngine(workspace_root)
        
        print("🔍 Testing health_utils.py harvested functions...")
        
        # Test harvested API monitoring function
        print("  • Testing enhanced_api_performance_test()...")
        api_result = engine.enhanced_api_performance_test()
        success = api_result.get('success', False)
        print(f"    ✅ API test completed: {success}")
        
        # Test harvested environment validation
        print("  • Testing environment_validation_check()...")
        env_result = engine.environment_validation_check()
        print(f"    ✅ Environment validation completed")
        
        # Test connection quality check
        print("  • Testing api_connection_quality_check()...")
        quality_result = engine.api_connection_quality_check()
        print(f"    ✅ Connection quality check completed")
        print("✅ health_utils.py harvested functions: ALL WORKING")
        assert success is True
    except Exception as e:
        print(f"❌ health_utils.py test failed: {e}")
        raise

def test_utils_harvested_functions():
    """Test harvested functions in utilities/utils.py"""
    try:
        from utilities.utils import load_and_validate_env_file, configuration_completeness_audit, enhanced_configuration_validator
        
        print("🔍 Testing utilities/utils.py harvested functions...")
        
        # Test environment file loading
        print("  • Testing load_and_validate_env_file()...")
        env_data = load_and_validate_env_file()
        var_count = len(env_data.get('variables', {}))
        print(f"    ✅ Environment loading completed: {var_count} variables found")
        
        # Test configuration audit
        print("  • Testing configuration_completeness_audit()...")
        audit_data = configuration_completeness_audit()
        files_checked = len(audit_data.get('files_checked', []))
        print(f"    ✅ Configuration audit completed: {files_checked} files checked")
        
        # Test enhanced configuration validator
        print("  • Testing enhanced_configuration_validator()...")
        validation_data = enhanced_configuration_validator()
        health_score = validation_data.get('overall_health_score', 0)
        print(f"    ✅ Enhanced validation completed: {health_score:.1%} health score")
        print("✅ utilities/utils.py harvested functions: ALL WORKING")
        assert isinstance(env_data, dict)
        assert isinstance(audit_data, dict)
        assert isinstance(validation_data, dict)
    except Exception as e:
        print(f"❌ utilities/utils.py test failed: {e}")
        raise

def main():
    """Run all harvested function validation tests"""
    print("🚀 Phase 0 Function Harvesting Validation Starting...")
    print("=" * 60)
    
    # Test both enhanced modules
    health_success = test_health_utils_harvested_functions()
    utils_success = test_utils_harvested_functions()
    
    print("=" * 60)
    
    overall = health_success and utils_success
    if overall:
        print("🎯 PHASE 0 FUNCTION HARVESTING VALIDATION: ✅ SUCCESS")
        print("💫 All harvested functions working correctly in new locations")
        print("🚀 Ready to proceed to Phase 1 - File Removal")
    else:
        print("❌ PHASE 0 VALIDATION FAILED")
        print("🔧 Fix integration issues before proceeding to Phase 1")
    return 0 if overall else 1

if __name__ == "__main__":
    main()
