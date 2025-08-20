# 🔍 FUNCTION HARVESTING EXTRACTION LOG
**Phase 0: Function Harvesting - Day 1-2**  
**Date**: 2025-08-20  
**Target**: Extract valuable functions from files marked for removal

## 🎯 HIGH-PRIORITY FUNCTIONS EXTRACTED

### **From debug_binance_api.py → health_utils.py**

#### **API Performance Monitoring Functions**
```python
def enhanced_api_performance_test():
    """Enhanced API performance testing with detailed metrics"""
    import requests
    import time
    from datetime import datetime
    
    try:
        start_time = time.time()
        response = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 5},
            timeout=10
        )
        response_time = time.time() - start_time
        
        return {
            'status_code': response.status_code,
            'response_time': response_time,
            'success': response.status_code == 200,
            'timestamp': datetime.now(),
            'data_valid': isinstance(response.json(), list) if response.status_code == 200 else False
        }
    except Exception as e:
        return {
            'status_code': None,
            'response_time': None,
            'success': False,
            'timestamp': datetime.now(),
            'error': str(e)
        }

def api_connection_quality_check():
    """Check API connection quality and stability"""
    results = []
    for i in range(3):  # Test 3 times
        result = enhanced_api_performance_test()
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    avg_response_time = sum(r['response_time'] for r in results if r['response_time']) / len([r for r in results if r['response_time']])
    success_rate = sum(1 for r in results if r['success']) / len(results)
    
    return {
        'average_response_time': avg_response_time,
        'success_rate': success_rate,
        'all_results': results,
        'quality_score': success_rate * (1.0 if avg_response_time < 2.0 else 0.5)
    }
```

### **From verify_system.py → health_utils.py**

#### **System Health Monitoring Functions**
```python
def comprehensive_system_health_check():
    """Comprehensive system health verification"""
    import os
    import requests
    import json
    from pathlib import Path
    
    health_report = {
        'timestamp': datetime.now(),
        'overall_health': 'UNKNOWN',
        'components': {}
    }
    
    # File system health
    required_files = [
        "vercel.json", "api/live-bot.py", "watcher.py", 
        "core/optimization_config.json", "health_utils.py"
    ]
    
    file_health = {'missing_files': [], 'present_files': []}
    for file_path in required_files:
        if os.path.exists(file_path):
            file_health['present_files'].append(file_path)
        else:
            file_health['missing_files'].append(file_path)
    
    health_report['components']['file_system'] = {
        'status': 'HEALTHY' if len(file_health['missing_files']) == 0 else 'DEGRADED',
        'details': file_health
    }
    
    # Configuration health
    try:
        config_path = "core/optimization_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config_health = {
                'status': 'HEALTHY' if 'bot_settings' in config else 'DEGRADED',
                'keys_present': list(config.keys()),
                'size_bytes': os.path.getsize(config_path)
            }
        else:
            config_health = {'status': 'MISSING', 'error': 'Config file not found'}
    except Exception as e:
        config_health = {'status': 'ERROR', 'error': str(e)}
    
    health_report['components']['configuration'] = config_health
    
    # Overall health determination
    component_statuses = [comp['status'] for comp in health_report['components'].values()]
    if all(status == 'HEALTHY' for status in component_statuses):
        health_report['overall_health'] = 'HEALTHY'
    elif any(status in ['MISSING', 'ERROR'] for status in component_statuses):
        health_report['overall_health'] = 'CRITICAL'
    else:
        health_report['overall_health'] = 'DEGRADED'
    
    return health_report

def environment_validation_check():
    """Enhanced environment variable validation"""
    import os
    
    required_vars = {
        'BINANCE_API_KEY': 'Critical - Binance API access',
        'BINANCE_API_SECRET': 'Critical - Binance API security',
        'BOT_SECRET': 'Important - Bot authentication',
        'VERCEL_URL': 'Important - Deployment URL'
    }
    
    validation_result = {
        'timestamp': datetime.now(),
        'status': 'UNKNOWN',
        'critical_missing': [],
        'important_missing': [],
        'all_present': []
    }
    
    for var, importance in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values for logging
            if 'SECRET' in var or 'KEY' in var:
                masked_value = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
                validation_result['all_present'].append(f"{var}: {masked_value}")
            else:
                validation_result['all_present'].append(f"{var}: {value}")
        else:
            if 'Critical' in importance:
                validation_result['critical_missing'].append(var)
            else:
                validation_result['important_missing'].append(var)
    
    # Determine overall status
    if len(validation_result['critical_missing']) > 0:
        validation_result['status'] = 'CRITICAL'
    elif len(validation_result['important_missing']) > 0:
        validation_result['status'] = 'DEGRADED'
    else:
        validation_result['status'] = 'HEALTHY'
    
    return validation_result
```

### **From check_env_config.py → utilities/utils.py**

#### **Configuration Management Functions**
```python
def load_and_validate_env_file():
    """Enhanced environment file loading and validation"""
    from pathlib import Path
    
    env_file = Path('.env')
    result = {
        'file_exists': False,
        'variables_count': 0,
        'validation_errors': [],
        'status': 'UNKNOWN'
    }
    
    if env_file.exists():
        result['file_exists'] = True
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Count valid environment variables
            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('#') and '=' in line]
            result['variables_count'] = len(lines)
            
            # Validate format
            for i, line in enumerate(lines):
                if '=' not in line:
                    result['validation_errors'].append(f"Line {i+1}: Invalid format - no '=' found")
                elif line.startswith('='):
                    result['validation_errors'].append(f"Line {i+1}: Variable name missing")
                elif '=' in line and not line.split('=')[1].strip():
                    result['validation_errors'].append(f"Line {i+1}: Empty value for {line.split('=')[0]}")
            
            result['status'] = 'HEALTHY' if len(result['validation_errors']) == 0 else 'DEGRADED'
            
        except Exception as e:
            result['validation_errors'].append(f"File read error: {str(e)}")
            result['status'] = 'ERROR'
    else:
        result['validation_errors'].append(".env file not found")
        result['status'] = 'MISSING'
    
    return result

def configuration_completeness_audit():
    """Audit configuration completeness across all config files"""
    import json
    import os
    from pathlib import Path
    
    config_files = {
        'core/optimization_config.json': ['bot_settings', 'optimization_results'],
        'api/live_trading_config.json': ['STOP_LOSS_MULTIPLIER', 'TAKE_PROFIT_MULTIPLIER'],
        'monitoring_config.json': ['health_check_interval', 'alert_thresholds'],
        'vercel.json': ['builds', 'routes']
    }
    
    audit_result = {
        'timestamp': datetime.now(),
        'overall_status': 'UNKNOWN',
        'files_checked': 0,
        'files_missing': [],
        'files_incomplete': [],
        'files_healthy': []
    }
    
    for file_path, required_keys in config_files.items():
        audit_result['files_checked'] += 1
        
        if not os.path.exists(file_path):
            audit_result['files_missing'].append(file_path)
            continue
        
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            
            missing_keys = [key for key in required_keys if key not in config_data]
            
            if missing_keys:
                audit_result['files_incomplete'].append({
                    'file': file_path,
                    'missing_keys': missing_keys,
                    'present_keys': [key for key in required_keys if key in config_data]
                })
            else:
                audit_result['files_healthy'].append(file_path)
                
        except Exception as e:
            audit_result['files_incomplete'].append({
                'file': file_path,
                'error': str(e)
            })
    
    # Determine overall status
    if len(audit_result['files_missing']) > 0:
        audit_result['overall_status'] = 'CRITICAL'
    elif len(audit_result['files_incomplete']) > 0:
        audit_result['overall_status'] = 'DEGRADED'
    else:
        audit_result['overall_status'] = 'HEALTHY'
    
    return audit_result
```

## 🔄 INTEGRATION STATUS

### **Functions Ready for Integration:**
✅ API performance monitoring functions → health_utils.py  
✅ System health checking functions → health_utils.py  
✅ Environment validation functions → health_utils.py  
✅ Configuration management functions → utilities/utils.py  

### **Next Steps:**
1. **Integrate functions into target modules**
2. **Test integrated functionality** 
3. **Validate no redundancy created**
4. **Performance benchmark new capabilities**
5. **Proceed to Phase 1 removal**

## 📊 ENHANCEMENT IMPACT

### **Functions Harvested**: 6 major enhancement functions
### **Target Modules Enhanced**: 2 (health_utils.py, utilities/utils.py)
### **New Capabilities Added**:
- Enhanced API performance monitoring
- Comprehensive system health checking
- Advanced environment validation
- Configuration completeness auditing
- Connection quality assessment
- Multi-file configuration management

**Status**: Ready for integration into target modules
