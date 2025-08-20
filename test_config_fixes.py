#!/usr/bin/env python3
"""
Configuration Fix Validation Test
Tests the configuration after applying fixes
"""

import sys
sys.path.append('.')

def test_configuration_fixes():
    """Test if configuration issues have been resolved"""
    try:
        from utilities.utils import enhanced_configuration_validator
        
        print("🔧 Testing Configuration Fixes...")
        print("=" * 50)
        
        result = enhanced_configuration_validator()
        
        print(f"\n🎯 Configuration Fix Results:")
        print(f"Overall Health Score: {result.get('overall_health_score', 0):.1%}")
        print(f"System Ready: {'✅ YES' if result.get('system_ready', False) else '❌ NO'}")
        print(f"Critical Issues: {len(result.get('critical_issues', []))}")
        
        if result.get('critical_issues'):
            print(f"\n🚨 Remaining Critical Issues:")
            for issue in result.get('critical_issues', []):
                print(f"  • {issue}")
        else:
            print(f"\n✅ No critical issues remaining!")
            
        if result.get('warnings'):
            print(f"\n⚠️  Warnings:")
            for warning in result.get('warnings', []):
                print(f"  • {warning}")
                
        env_data = result.get('environment_validation', {})
        if env_data:
            print(f"\n📊 Environment Status:")
            print(f"  • Variables loaded: {len(env_data.get('variables', {}))}")
            print(f"  • Missing variables: {len(env_data.get('missing', []))}")
            print(f"  • Invalid variables: {len(env_data.get('invalid', []))}")
            print(f"  • Validation passed: {'✅' if env_data.get('validation_passed', False) else '❌'}")
            
        audit_data = result.get('configuration_audit', {})
        if audit_data:
            print(f"\n📁 Configuration Files:")
            print(f"  • Files checked: {len(audit_data.get('files_checked', []))}")
            print(f"  • Missing files: {len(audit_data.get('missing_configurations', []))}")
            print(f"  • Config health: {audit_data.get('configuration_health_score', 0):.1%}")
        
        return result.get('system_ready', False)
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_configuration_fixes()
    print("\n" + "=" * 50)
    if success:
        print("🎯 CONFIGURATION FIXES: ✅ SUCCESS")
        print("🚀 System is now ready for operation!")
    else:
        print("❌ CONFIGURATION FIXES: Still need work")
        print("🔧 Additional fixes required")
