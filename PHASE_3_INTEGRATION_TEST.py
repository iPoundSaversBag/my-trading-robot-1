#!/usr/bin/env python3
"""
PHASE 3 INTEGRATION TEST SUITE
Comprehensive testing of the Phase 3 Integration Optimization enhancements.
"""

import sys
import os
import time
import traceback
import pandas as pd  # used in data standardizer test
import numpy as np   # used in data standardizer test

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration_manager():
    """Test ConfigurationManager functionality."""
    print("🔧 Testing ConfigurationManager...")
    
    try:
        from utilities.utils import config_manager
        
        # Test default configuration loading
        strategy_config = config_manager.get_config(None, 'strategy')
        assert 'MIN_SIGNAL_CONFIDENCE' in strategy_config
        print("   ✅ Default strategy configuration loaded")
        
        risk_config = config_manager.get_config(None, 'risk_management')
        assert 'FIXED_RISK_PERCENTAGE' in risk_config
        print("   ✅ Default risk management configuration loaded")
        
        portfolio_config = config_manager.get_config(None, 'portfolio')
        assert 'INITIAL_CAPITAL' in portfolio_config
        print("   ✅ Default portfolio configuration loaded")
        
        # Test configuration validation
        test_config = {'FIXED_RISK_PERCENTAGE': 1.5}  # Invalid value > 1.0
        validated = config_manager.get_config(test_config, 'risk_management')
        assert validated['FIXED_RISK_PERCENTAGE'] <= 1.0
        print("   ✅ Configuration validation working")
        
    except Exception as e:
        print(f"   ❌ ConfigurationManager test failed: {e}")
        raise

def test_data_standardizer():
    """Test DataStandardizer functionality."""
    print("📊 Testing DataStandardizer...")
    try:
        from utilities.utils import data_standardizer
        
        # Test DataFrame standardization
        test_df = pd.DataFrame({
            'open': [100, 101, np.nan, 103],
            'close': [101, 102, 102, 104],
            'volume': [1000, 1100, 1200, np.nan]
        })
        
        standardized = data_standardizer.standardize_dataframe(
            test_df, 
            required_columns=['open', 'high', 'low', 'close'],
            fill_method='forward'
        )
        
        assert 'high' in standardized.columns  # Missing column should be added
        assert not standardized['open'].isna().any()  # NaN should be filled
        print("   ✅ DataFrame standardization working")
        
        # Test price normalization
        prices = [100, 110, 95, 105, 120]
        normalized = data_standardizer.normalize_price_data(prices, method='minmax')
        assert 0 <= min(normalized) and max(normalized) <= 1
        print("   ✅ Price normalization working")
        
        # Test timestamp standardization
        test_df_time = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'price': [100, 101]
        })
        
        standardized_time = data_standardizer.standardize_timestamps(
            test_df_time, 
            'timestamp', 
            'datetime'
        )
        assert pd.api.types.is_datetime64_any_dtype(standardized_time['timestamp'])
        print("   ✅ Timestamp standardization working")
        
    except Exception as e:
        print(f"   ❌ DataStandardizer test failed: {e}")
        raise

def test_event_system():
    """Test EventManager functionality."""
    print("📡 Testing EventManager...")
    
    try:
        from utilities.utils import event_manager
        
        # Test event subscription and publishing
        test_events = []
        
        def test_callback(event):
            test_events.append(event)
        
        # Subscribe to test events
        event_manager.subscribe('test_event', test_callback, 'test_module')
        
        # Publish test event
        event_manager.publish(
            'test_event',
            {'message': 'Hello World', 'value': 42},
            'test_publisher'
        )
        
        # Check if event was received
        assert len(test_events) == 1
        assert test_events[0]['data']['message'] == 'Hello World'
        print("   ✅ Event subscription and publishing working")
        
        # Test event statistics
        stats = event_manager.get_event_statistics()
        assert stats['total_events'] > 0
        assert 'test_event' in stats['event_types']
        print("   ✅ Event statistics working")
        
    except Exception as e:
        print(f"   ❌ EventManager test failed: {e}")
        raise

def test_module_communication():
    """Test ModuleCommunicator functionality."""
    print("🔄 Testing ModuleCommunicator...")
    
    try:
        from utilities.utils import module_communicator
        
        # Test safe module call
        def test_function(x, y):
            return x + y
        
        result = module_communicator.safe_module_call(
            test_function,
            5, 7,
            fallback_value=0,
            timeout_seconds=1
        )
        assert result == 12
        print("   ✅ Safe module call working")
        
        # Test with failing function
        def failing_function():
            raise Exception("Test error")
        
        result = module_communicator.safe_module_call(
            failing_function,
            fallback_value="fallback",
            timeout_seconds=1
        )
        assert result == "fallback"
        print("   ✅ Error handling with fallback working")
        
    except Exception as e:
        print(f"   ❌ ModuleCommunicator test failed: {e}")
        raise

def test_performance_monitoring():
    """Test PerformanceMonitor functionality."""
    print("📈 Testing PerformanceMonitor...")
    
    try:
        from utilities.utils import performance_monitor
        
        # Test performance tracking
        performance_monitor.track_module_call('test_module', 'test_method', 0.125)
        performance_monitor.track_module_call('test_module', 'test_method', 0.200)
        
        # Get performance report
        report = performance_monitor.get_performance_report()
        assert 'test_module.test_method' in report['call_statistics']
        
        stats = report['call_statistics']['test_module.test_method']
        assert stats['call_count'] == 2
        assert 0.150 < stats['avg_time'] < 0.170  # Average should be ~0.1625
        print("   ✅ Performance monitoring working")
        
    except Exception as e:
        print(f"   ❌ PerformanceMonitor test failed: {e}")
        raise

def test_error_recovery():
    """Test ErrorRecoveryManager functionality."""
    print("🚨 Testing ErrorRecoveryManager...")
    
    try:
        from utilities.utils import ErrorRecoveryManager
        
        # Register a test recovery strategy
        def test_recovery(error, context):
            return context.get('recoverable', False)
        
        ErrorRecoveryManager.register_recovery_strategy(
            'ValueError',
            test_recovery,
            'test_module'
        )
        
        # Test error handling with recovery
        test_error = ValueError("Test error")
        recovery_result = ErrorRecoveryManager.handle_error(
            test_error,
            {'recoverable': True, 'operation': 'test'},
            'test_source'
        )
        assert recovery_result == True
        print("   ✅ Error recovery working")
        
        # Test error statistics
        stats = ErrorRecoveryManager.get_error_statistics()
        assert stats['total_errors'] > 0
        assert stats['recovery_rate'] > 0
        print("   ✅ Error statistics working")
        
    except Exception as e:
        print(f"   ❌ ErrorRecoveryManager test failed: {e}")
        raise

def test_health_monitoring():
    """Test IntegrationHealthMonitor functionality."""
    print("🏥 Testing IntegrationHealthMonitor...")
    
    try:
        from utilities.utils import integration_health_monitor
        
        # Register a test health check
        def test_health_check():
            return {
                'status': 'healthy',
                'healthy': True,
                'test_metric': 42
            }
        
        integration_health_monitor.register_health_check(
            'test_check',
            test_health_check,
            'test_module',
            interval_seconds=1
        )
        
        # Run health checks
        health_report = integration_health_monitor.run_health_checks(force_all=True)
        assert health_report['overall_healthy'] == True
        assert 'test_check' in health_report['individual_checks']
        print("   ✅ Health monitoring working")
        
        # Test health trend analysis
        trend = integration_health_monitor.get_health_trend(hours=1)
        assert 'trend' in trend
        print("   ✅ Health trend analysis working")
    except Exception as e:
        print(f"   ❌ IntegrationHealthMonitor test failed: {e}")
        raise

def test_integration_with_core_modules():
    """Test integration with actual core modules."""
    print("🔗 Testing Core Module Integration...")
    
    try:
        # Test enhanced health check functions
        from utilities.utils import safe_health_check, safe_pre_backtest_gate, safe_auto_fix_config
        
        # Test health check
        result = safe_health_check('test_component', silent=True)
        assert isinstance(result, bool)
        print("   ✅ Enhanced health check working")
        
        # Test pre-backtest gate
        result = safe_pre_backtest_gate('test_component', silent=True)
        assert isinstance(result, bool)
        print("   ✅ Enhanced pre-backtest gate working")
        
        # Test auto-fix config
        result = safe_auto_fix_config('test_component')
        assert isinstance(result, bool)
        print("   ✅ Enhanced auto-fix config working")
        
    except Exception as e:
        print(f"   ❌ Core module integration test failed: {e}")
        traceback.print_exc()
        raise

def run_comprehensive_integration_test():
    """Run comprehensive integration test suite."""
    print("=" * 60)
    print("🚀 PHASE 3 INTEGRATION OPTIMIZATION - COMPREHENSIVE TEST")
    print("=" * 60)
    
    test_functions = [
        test_configuration_manager,
        test_data_standardizer,
        test_event_system,
        test_module_communication,
        test_performance_monitoring,
        test_error_recovery,
        test_health_monitoring,
        test_integration_with_core_modules
    ]
    
    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ PASSED: {passed}")
    print(f"❌ FAILED: {failed}")
    total = passed + failed if (passed + failed) > 0 else 1
    print(f"📈 SUCCESS RATE: {(passed / total) * 100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 3 Integration Optimization is complete and functional.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Review and fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)
