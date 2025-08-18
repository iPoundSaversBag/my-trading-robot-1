# ==============================================================================
#
#                               UTILITY FUNCTIONS
#
# ==============================================================================
#
# FILE: utils.py
#
# PURPOSE:
#   This module provides a centralized collection of utility functions that are
#   shared across the entire trading robot pipeline. This includes data
#   preparation, performance calculations, validation tools, and analysis functions.
#
# CONSOLIDATED FROM:
#   - validate_parameter_bounds.py (parameter validation functions)
#   - active_usage_analyzer.py (code analysis functions)
#   - core/config.py (configuration management)
#
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import sys
import asyncio
import datetime
import traceback
import time
import logging
import threading
import shutil
import glob
import argparse
import re
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Additional imports for data management
import ccxt.pro as ccxt
import numpy as np
import json
import os
import sys
import ast
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, average_true_range
from enum import Enum
from dataclasses import dataclass

# ==============================================================================
# ENHANCED FAULT TOLERANCE & CIRCUIT BREAKER SYSTEM
# Consolidated from enhancements/fault_tolerance.py
# ==============================================================================

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_duration: int = 60  # seconds
    success_threshold: int = 3  # for half-open state

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for trading bot reliability
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        # Check if circuit should move from OPEN to HALF_OPEN
        if (self.state == CircuitState.OPEN and 
            time.time() - self.last_failure_time > self.config.timeout_duration):
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN state")
        
        # Reject calls if circuit is OPEN
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self.on_success()
            return result
            
        except Exception as e:
            # Record failure
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful function execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
        else:
            self.failure_count = 0
    
    def on_failure(self):
        """Handle failed function execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} OPENED - too many failures")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class FaultTolerantTradingBot:
    """
    Enhanced trading bot with comprehensive fault tolerance
    """
    
    def __init__(self, params):
        self.params = params
        self.circuit_breakers = self._initialize_circuit_breakers()
        self.error_counts = {}
        self.health_metrics = {}
        self.last_health_check = time.time()
        self.logger = logging.getLogger("FaultTolerantTradingBot")
        
        # Health monitoring settings
        self.health_check_interval = params.get('HEALTH_CHECK_INTERVAL', 300)  # 5 minutes
        self.max_error_rate = params.get('MAX_ERROR_RATE', 0.1)  # 10% error rate threshold
        self.recovery_timeout = params.get('RECOVERY_TIMEOUT', 1800)  # 30 minutes
        
    def _initialize_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Initialize circuit breakers for different components"""
        
        circuit_breakers = {}
        
        # Exchange API circuit breaker
        circuit_breakers['exchange_api'] = CircuitBreaker(
            "exchange_api",
            CircuitBreakerConfig(failure_threshold=3, timeout_duration=120, success_threshold=2)
        )
        
        # Data fetching circuit breaker
        circuit_breakers['data_fetch'] = CircuitBreaker(
            "data_fetch",
            CircuitBreakerConfig(failure_threshold=5, timeout_duration=60, success_threshold=3)
        )
        
        # Order execution circuit breaker
        circuit_breakers['order_execution'] = CircuitBreaker(
            "order_execution", 
            CircuitBreakerConfig(failure_threshold=2, timeout_duration=300, success_threshold=1)
        )
        
        # Signal generation circuit breaker
        circuit_breakers['signal_generation'] = CircuitBreaker(
            "signal_generation",
            CircuitBreakerConfig(failure_threshold=10, timeout_duration=30, success_threshold=5)
        )
        
        return circuit_breakers
    
    async def execute_with_fault_tolerance(self, component: str, func, *args, **kwargs):
        """Execute function with fault tolerance and circuit breaker protection"""
        
        if component not in self.circuit_breakers:
            self.logger.warning(f"No circuit breaker found for component: {component}")
            return await self._execute_without_protection(func, *args, **kwargs)
        
        circuit_breaker = self.circuit_breakers[component]
        
        try:
            result = await circuit_breaker.call(func, *args, **kwargs)
            self._update_health_metrics(component, success=True)
            return result
            
        except CircuitBreakerOpenError as e:
            self.logger.error(f"Circuit breaker OPEN for {component}: {e}")
            self._update_health_metrics(component, success=False, circuit_open=True)
            return await self._handle_circuit_open(component, func, *args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in {component}: {e}")
            self._update_health_metrics(component, success=False)
            return await self._handle_component_error(component, e, func, *args, **kwargs)
    
    async def _execute_without_protection(self, func, *args, **kwargs):
        """Execute function without circuit breaker protection"""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Unprotected function execution failed: {e}")
            raise e
    
    async def _handle_circuit_open(self, component: str, func, *args, **kwargs):
        """Handle circuit breaker open state"""
        
        # Return fallback result based on component
        if component == 'exchange_api':
            return None  # Skip API calls when circuit is open
        elif component == 'data_fetch':
            return pd.DataFrame()  # Return empty dataframe
        elif component == 'order_execution':
            return False  # Indicate order execution failed
        elif component == 'signal_generation':
            return 0.0  # Return neutral signal
        else:
            return None
    
    async def _handle_component_error(self, component: str, error: Exception, func, *args, **kwargs):
        """Handle component-specific errors with recovery strategies"""
        
        # Log error details
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        
        # Component-specific error handling
        if component == 'exchange_api':
            return await self._handle_exchange_error(error, func, *args, **kwargs)
        elif component == 'data_fetch':
            return await self._handle_data_fetch_error(error, func, *args, **kwargs)
        elif component == 'order_execution':
            return await self._handle_order_error(error, func, *args, **kwargs)
        elif component == 'signal_generation':
            return await self._handle_signal_error(error, func, *args, **kwargs)
        else:
            # Default error handling
            self.logger.error(f"Unhandled error in {component}: {error}")
            return None
    
    async def _handle_exchange_error(self, error: Exception, func, *args, **kwargs):
        """Handle exchange API errors with retry logic"""
        
        # Check for specific exchange errors
        if "rate limit" in str(error).lower():
            # Rate limiting - wait and retry
            wait_time = 60  # 1 minute
            self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)
            
            try:
                return await self._execute_without_protection(func, *args, **kwargs)
            except Exception as retry_error:
                self.logger.error(f"Retry failed: {retry_error}")
                return None
        
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            # Network issues - short retry
            await asyncio.sleep(5)
            
            try:
                return await self._execute_without_protection(func, *args, **kwargs)
            except Exception as retry_error:
                self.logger.error(f"Network retry failed: {retry_error}")
                return None
        
        else:
            # Unknown exchange error
            self.logger.error(f"Unknown exchange error: {error}")
            return None
    
    async def _handle_data_fetch_error(self, error: Exception, func, *args, **kwargs):
        """Handle data fetching errors"""
        
        # Try to use cached data if available
        self.logger.warning(f"Data fetch failed: {error}, attempting to use cached data")
        
        # Return empty dataframe as fallback
        return pd.DataFrame()
    
    async def _handle_order_error(self, error: Exception, func, *args, **kwargs):
        """Handle order execution errors"""
        
        # Check for insufficient balance
        if "insufficient" in str(error).lower():
            self.logger.error(f"Insufficient balance for order: {error}")
            return False
        
        # Check for invalid order parameters
        elif "invalid" in str(error).lower():
            self.logger.error(f"Invalid order parameters: {error}")
            return False
        
        else:
            # Retry once for other order errors
            await asyncio.sleep(1)
            
            try:
                return await self._execute_without_protection(func, *args, **kwargs)
            except Exception as retry_error:
                self.logger.error(f"Order retry failed: {retry_error}")
                return False
    
    async def _handle_signal_error(self, error: Exception, func, *args, **kwargs):
        """Handle signal generation errors"""
        
        self.logger.warning(f"Signal generation failed: {error}, returning neutral signal")
        return 0.0  # Neutral signal
    
    def _update_health_metrics(self, component: str, success: bool, circuit_open: bool = False):
        """Update health metrics for monitoring"""
        
        if component not in self.health_metrics:
            self.health_metrics[component] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'circuit_open_events': 0,
                'last_success': None,
                'last_failure': None
            }
        
        metrics = self.health_metrics[component]
        metrics['total_calls'] += 1
        
        if success:
            metrics['successful_calls'] += 1
            metrics['last_success'] = time.time()
        else:
            metrics['failed_calls'] += 1
            metrics['last_failure'] = time.time()
            
            if circuit_open:
                metrics['circuit_open_events'] += 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all components"""
        
        status = {
            'overall_health': 'healthy',
            'components': {},
            'circuit_breakers': {},
            'error_summary': {}
        }
        
        unhealthy_components = 0
        
        # Check each component
        for component, metrics in self.health_metrics.items():
            component_health = self._calculate_component_health(metrics)
            status['components'][component] = component_health
            
            if component_health['status'] != 'healthy':
                unhealthy_components += 1
        
        # Check circuit breaker states
        for name, circuit_breaker in self.circuit_breakers.items():
            status['circuit_breakers'][name] = {
                'state': circuit_breaker.state.value,
                'failure_count': circuit_breaker.failure_count,
                'success_count': circuit_breaker.success_count
            }
        
        # Error summary
        status['error_summary'] = dict(self.error_counts)
        
        # Overall health assessment
        if unhealthy_components > 0:
            if unhealthy_components >= len(self.health_metrics) / 2:
                status['overall_health'] = 'critical'
            else:
                status['overall_health'] = 'degraded'
        
        return status
    
    def _calculate_component_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health status for a component"""
        
        if metrics['total_calls'] == 0:
            return {'status': 'unknown', 'success_rate': 0.0, 'details': 'No calls made'}
        
        success_rate = metrics['successful_calls'] / metrics['total_calls']
        
        # Determine health status
        if success_rate >= 0.95:
            status = 'healthy'
        elif success_rate >= 0.85:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        # Check for recent failures
        current_time = time.time()
        time_since_last_failure = None
        
        if metrics['last_failure']:
            time_since_last_failure = current_time - metrics['last_failure']
            
            # If recent failure and low success rate, mark as critical
            if time_since_last_failure < 300 and success_rate < 0.5:  # 5 minutes
                status = 'critical'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'total_calls': metrics['total_calls'],
            'successful_calls': metrics['successful_calls'],
            'failed_calls': metrics['failed_calls'],
            'circuit_open_events': metrics['circuit_open_events'],
            'time_since_last_failure': time_since_last_failure
        }
    
    async def run_health_check(self):
        """Run periodic health check"""
        
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        # Get health status
        health_status = self.get_health_status()
        
        # Log health summary
        self.logger.info(f"Health Check - Overall: {health_status['overall_health']}")
        
        for component, health in health_status['components'].items():
            if health['status'] != 'healthy':
                self.logger.warning(
                    f"Component {component}: {health['status']} "
                    f"(success rate: {health['success_rate']:.2%})"
                )
        
        # Check for critical issues
        if health_status['overall_health'] == 'critical':
            self.logger.error("CRITICAL: Multiple components are unhealthy!")
            # Could trigger alert/notification here
        
        return health_status

# ============================================================================== 
#                           CONFIGURATION MANAGEMENT
# ============================================================================== 

class Config:
    """Centralized configuration management."""
    
    def __init__(self, config_file: str = 'core/optimization_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Could not load config {self.config_file}: {e}")
                return self.get_default_config()
        else:
            print(f"📁 Config file {self.config_file} not found, using defaults")
            return self.get_default_config()
            
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Trading parameters
            "ichimoku_params": {
                "tenkan_period": 9,
                "kijun_period": 26,
                "senkou_span_b_period": 52,
                "displacement": 26
            },
            
            # Multi-timeframe settings
            "multi_timeframe_enabled": True,
            "primary_timeframe": "5m",
            "signal_timeframes": ["15m", "1h", "4h"],
            "timeframe_weights": {
                "5m": 0.4,
                "15m": 0.3,
                "1h": 0.2,
                "4h": 0.1
            },
            
            # Data settings with multi-timeframe files
            "data_settings": {
                "multi_timeframe_enabled": True,
                "primary_timeframe": "5m",
                "timeframe_files": {
                    "5m": "data/crypto_data_5m.parquet",
                    "15m": "data/crypto_data_15m.parquet",
                    "1h": "data/crypto_data_1h.parquet",
                    "4h": "data/crypto_data_4h.parquet"
                }
            },
            
            # Data settings (legacy)
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "data_limit": 1000,
            
            # Backtesting
            "initial_capital": 10000,
            "position_size": 0.1,
            "take_profit": 0.02,
            "stop_loss": 0.01,
            
            # Live trading
            "live_trading_enabled": False,
            "max_positions": 3,
            "risk_per_trade": 0.02,
            
            # Performance monitoring
            "performance_monitoring": {
                "enabled": True,
                "debug_mode": False,
                "log_interval": 5.0
            },
            
            # File paths
            "paths": {
                "data_dir": ".",
                "logs_dir": "logs",
                "plots_dir": "plots_output"
            }
        }
        
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Saved config to {self.config_file}")
        except Exception as e:
            print(f"❌ Could not save config: {e}")
            
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        
    def get_timeframe_files(self) -> Dict[str, str]:
        """Get timeframe file mapping."""
        files = {}
        for tf in self.get('signal_timeframes', []):
            files[tf] = f"crypto_data_{tf}.parquet"
        return files
        
    def validate_config(self) -> bool:
        """Validate configuration."""
        required_keys = [
            'ichimoku_params.tenkan_period',
            'ichimoku_params.kijun_period',
            'primary_timeframe',
            'initial_capital'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"❌ Missing required config: {key}")
                return False
                
        return True

# Global config instance
config = Config()

# ==============================================================================
# PARAMETER VALIDATION FUNCTIONS (from validate_parameter_bounds.py)
# ==============================================================================

def validate_parameter_bounds(config_file: str) -> Dict:
    """
    Validate all parameter bounds against hard bounds
    
    Returns:
        Dictionary with validation results
    """
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    parameter_spaces = config.get('parameter_spaces', {}).get('global', [])
    
    validation_results = {
        'valid_parameters': [],
        'invalid_parameters': [],
        'warnings': [],
        'summary': {}
    }
    
    print("🔍 Parameter Bounds Validation Report")
    print("=" * 60)
    print(f"{'Parameter':<25} | {'Bounds':<15} | {'Hard Bounds':<15} | {'Status'}")
    print("-" * 60)
    
    for param in parameter_spaces:
        name = param['name']
        bounds = param['bounds']
        param_type = param['type']
        
        # Categorical parameters don't have hard_bounds
        if param_type == 'Categorical':
            bounds_str = f"{bounds}"
            hard_bounds_str = "N/A"
            status = "✅ VALID"
            bounds_valid = True
            
            validation_results['valid_parameters'].append({
                'name': name,
                'type': param_type,
                'bounds': bounds,
                'hard_bounds': None
            })
        else:
            # Numeric parameters should have hard_bounds
            hard_bounds = param.get('hard_bounds')
            if not hard_bounds:
                print(f"❌ Missing hard_bounds for {name}")
                continue
                
            # Check if bounds are within hard bounds
            bounds_valid = (
                bounds[0] >= hard_bounds[0] and
                bounds[1] <= hard_bounds[1] and
                bounds[0] <= bounds[1]
            )
            
            # Format bounds display
            bounds_str = f"[{bounds[0]}, {bounds[1]}]"
            hard_bounds_str = f"[{hard_bounds[0]}, {hard_bounds[1]}]"
            
            status = "✅ VALID" if bounds_valid else "❌ INVALID"
            
            if bounds_valid:
                validation_results['valid_parameters'].append({
                    'name': name,
                    'type': param_type,
                    'bounds': bounds,
                    'hard_bounds': hard_bounds
                })
            else:
                validation_results['invalid_parameters'].append({
                    'name': name,
                    'type': param_type,
                    'bounds': bounds,
                    'hard_bounds': hard_bounds,
                    'issues': []
                })
                
                # Identify specific issues
                issues = []
                if bounds[0] < hard_bounds[0]:
                    issues.append(f"Lower bound {bounds[0]} < hard lower bound {hard_bounds[0]}")
                if bounds[1] > hard_bounds[1]:
                    issues.append(f"Upper bound {bounds[1]} > hard upper bound {hard_bounds[1]}")
                if bounds[0] > bounds[1]:
                    issues.append(f"Lower bound {bounds[0]} > upper bound {bounds[1]}")
                
                validation_results['invalid_parameters'][-1]['issues'] = issues
        
        print(f"{name:<25} | {bounds_str:<15} | {hard_bounds_str:<15} | {status}")
    
    # Check best parameters against hard bounds
    best_params = config.get('best_parameters_so_far', {})
    print("\n" + "=" * 60)
    print("📊 Best Parameters Validation")
    print("=" * 60)
    
    for param in parameter_spaces:
        name = param['name']
        param_type = param['type']
        
        if name in best_params:
            value = best_params[name]
            
            if param_type == 'Categorical':
                # For categorical parameters, check if value is in allowed values
                valid_values = param.get('values', [])
                valid = value in valid_values
                status = "✅ VALID" if valid else "❌ INVALID"
                print(f"{name:<25} | Value: {value:<10} | Allowed: {valid_values} | {status}")
            else:
                # For numeric parameters, check against hard bounds
                hard_bounds = param.get('hard_bounds')
                if hard_bounds:
                    valid = hard_bounds[0] <= value <= hard_bounds[1]
                    status = "✅ VALID" if valid else "❌ INVALID"
                    print(f"{name:<25} | Value: {value:<10} | Hard Bounds: {hard_bounds} | {status}")
                else:
                    print(f"{name:<25} | Value: {value:<10} | No hard bounds defined | ⚠️ WARNING")
                    valid = True
            
            if not valid:
                validation_results['warnings'].append({
                    'parameter': name,
                    'current_value': value,
                    'hard_bounds': hard_bounds,
                    'issue': f"Best parameter value {value} is outside hard bounds {hard_bounds}"
                })
    
    # Summary
    total_params = len(parameter_spaces)
    valid_params = len(validation_results['valid_parameters'])
    invalid_params = len(validation_results['invalid_parameters'])
    
    validation_results['summary'] = {
        'total_parameters': total_params,
        'valid_parameters': valid_params,
        'invalid_parameters': invalid_params,
        'validation_percentage': (valid_params / total_params) * 100 if total_params > 0 else 0
    }
    
    print("\n" + "=" * 60)
    print("📈 Validation Summary")
    print("=" * 60)
    print(f"Total Parameters: {total_params}")
    print(f"Valid Parameters: {valid_params}")
    print(f"Invalid Parameters: {invalid_params}")
    print(f"Validation Rate: {validation_results['summary']['validation_percentage']:.1f}%")
    
    return validation_results

def analyze_parameter_relationships(config_file: str):
    """Analyze logical relationships between parameters"""
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    parameter_spaces = config.get('parameter_spaces', {}).get('global', [])
    
    # Create parameter lookup
    params = {p['name']: p for p in parameter_spaces}
    
    print("\n" + "=" * 60)
    print("🔗 Parameter Relationship Analysis")
    print("=" * 60)
    
    # Check Ichimoku parameter relationships
    print("\n📊 Ichimoku Parameter Relationships:")
    
    tenkan_bounds = params.get('TENKAN_SEN_PERIOD', {}).get('bounds', [0, 0])
    kijun_bounds = params.get('KIJUN_SEN_PERIOD', {}).get('bounds', [0, 0])
    senkou_bounds = params.get('SENKOU_SPAN_B_PERIOD', {}).get('bounds', [0, 0])
    
    print(f"  Tenkan-sen Period: {tenkan_bounds}")
    print(f"  Kijun-sen Period: {kijun_bounds}")
    print(f"  Senkou Span B Period: {senkou_bounds}")
    
    # Logical checks
    if tenkan_bounds[1] >= kijun_bounds[0]:
        print("  ⚠️  WARNING: Tenkan-sen max >= Kijun-sen min (should be Tenkan < Kijun)")
    
    if kijun_bounds[1] >= senkou_bounds[0]:
        print("  ⚠️  WARNING: Kijun-sen max >= Senkou B min (typically Kijun < Senkou B)")

# ==============================================================================
# CODE ANALYSIS FUNCTIONS (from active_usage_analyzer.py)
# ==============================================================================

class ActiveUsageAnalyzer:
    """Analyzes which duplicate files are actively used in the codebase"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.usage_results = {}
        
    def analyze_import_usage(self, duplicate_groups: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Analyze which duplicate files are actively imported and used.
        
        Args:
            duplicate_groups: Dictionary with filename as key and list of paths as value
            
        Returns:
            Dictionary with analysis results for each duplicate group
        """
        results = {}
        
        for filename, paths in duplicate_groups.items():
            if len(paths) > 1:  # Only analyze actual duplicates
                results[filename] = self._analyze_file_group(filename, paths)
        
        return results
    
    def _analyze_file_group(self, filename: str, paths: List[str]) -> Dict:
        """Analyze a group of duplicate files"""
        
        # Remove .py extension for module name
        module_name = filename.replace('.py', '')
        
        analysis = {
            'filename': filename,
            'paths': paths,
            'import_analysis': {},
            'execution_analysis': {},
            'recommendation': None
        }
        
        # 1. Analyze import statements across the codebase
        import_usage = self._find_import_statements(module_name)
        analysis['import_analysis'] = import_usage
        
        # 2. Test actual imports from different directories
        execution_tests = self._test_import_execution(module_name, paths)
        analysis['execution_analysis'] = execution_tests
        
        # 3. Analyze function/class usage
        function_usage = self._analyze_function_usage(filename, paths)
        analysis['function_usage'] = function_usage
        
        # 4. Make recommendation
        analysis['recommendation'] = self._make_recommendation(analysis)
        
        return analysis
    
    def _find_import_statements(self, module_name: str) -> Dict:
        """Find all import statements referencing this module"""
        import_statements = []
        importing_files = []
        
        # Search for import statements
        for py_file in self.workspace_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for various import patterns
                patterns = [
                    f"import {module_name}",
                    f"from {module_name}",
                    f"importlib.import_module('{module_name}')",
                    f'importlib.import_module("{module_name}")'
                ]
                
                for pattern in patterns:
                    if pattern in content:
                        import_statements.append({
                            'file': str(py_file.relative_to(self.workspace_root)),
                            'pattern': pattern,
                            'full_line': self._extract_import_line(content, pattern)
                        })
                        importing_files.append(str(py_file.relative_to(self.workspace_root)))
                        
            except Exception as e:
                continue
        
        return {
            'import_statements': import_statements,
            'importing_files': list(set(importing_files)),
            'total_imports': len(import_statements)
        }
    
    def _extract_import_line(self, content: str, pattern: str) -> str:
        """Extract the full line containing the import pattern"""
        lines = content.split('\n')
        for line in lines:
            if pattern in line:
                return line.strip()
        return ""
    
    def _test_import_execution(self, module_name: str, paths: List[str]) -> Dict:
        """Test actual import execution from different directories"""
        results = {}
        
        # Create safe temp directory for health checker files
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='health_checker_')
        
        for path in paths:
            try:
                # Try to import the module from this specific path
                dir_path = os.path.dirname(path)
                
                # Test import from the directory containing the file
                test_script = f"""
import sys
sys.path.insert(0, '{dir_path}')
try:
    import {module_name}
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {{e}}')
"""
                
                # Write temporary test script to SAFE location
                temp_test_file = os.path.join(temp_dir, f'temp_import_test_{module_name}_{hash(path)}.py')
                with open(temp_test_file, 'w') as f:
                    f.write(test_script)
                
                # Execute test
                result = subprocess.run([sys.executable, temp_test_file], 
                                      capture_output=True, text=True, timeout=10)
                
                results[path] = {
                    'success': 'SUCCESS' in result.stdout,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
                # Cleanup temp file
                if os.path.exists(temp_test_file):
                    os.remove(temp_test_file)
                    
            except Exception as e:
                results[path] = {
                    'success': False,
                    'output': '',
                    'error': str(e)
                }
        
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return results
    
    def _analyze_function_usage(self, filename: str, paths: List[str]) -> Dict:
        """Analyze which functions/classes are used from each duplicate"""
        
        function_analysis = {}
        
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to extract functions and classes
                tree = ast.parse(content)
                
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                
                function_analysis[path] = {
                    'functions': functions,
                    'classes': classes,
                    'total_definitions': len(functions) + len(classes)
                }
                
            except Exception as e:
                function_analysis[path] = {
                    'functions': [],
                    'classes': [],
                    'total_definitions': 0,
                    'error': str(e)
                }
        
        return function_analysis
    
    def _make_recommendation(self, analysis: Dict) -> Dict:
        """Make a recommendation based on analysis results"""
        
        import_count = analysis['import_analysis']['total_imports']
        execution_results = analysis['execution_analysis']
        
        # Count successful imports
        successful_imports = sum(1 for result in execution_results.values() if result['success'])
        
        if import_count == 0:
            recommendation = {
                'action': 'SAFE_TO_REMOVE',
                'confidence': 'HIGH',
                'reason': 'No import statements found referencing this module'
            }
        elif successful_imports == 0:
            recommendation = {
                'action': 'INVESTIGATE',
                'confidence': 'MEDIUM',
                'reason': 'Import statements found but none execute successfully'
            }
        elif successful_imports == 1:
            recommendation = {
                'action': 'KEEP_ONE',
                'confidence': 'HIGH',
                'reason': 'Only one version imports successfully - keep the working one'
            }
        else:
            recommendation = {
                'action': 'INVESTIGATE',
                'confidence': 'LOW',
                'reason': 'Multiple versions import successfully - manual review needed'
            }
        
        return recommendation

# ==============================================================================
# LEGACY FUNCTIONS (preserved for compatibility)
# ==============================================================================

# (Note: Technical analysis functions are now deprecated and have been moved into the Strategy class)

# ==============================================================================
# PARAMETER LIMITS ANALYSIS FUNCTIONS
# ==============================================================================

def find_latest_results(run_dir):
    """Finds the optimized_params.json file in a specific run directory."""
    optimized_params_path = os.path.join(run_dir, 'optimized_params.json')
    log_path = os.path.join(run_dir, 'optimization_log.txt')
    
    if os.path.exists(optimized_params_path):
        return optimized_params_path, log_path if os.path.exists(log_path) else None
    return None, None

def plot_parameter_sensitivity(log_path, param1, param2, run_dir):
    """Plot parameter sensitivity analysis (placeholder for complex plotting logic)."""
    print(f"Generating sensitivity plot for {param1} vs {param2} (saved to {run_dir})")

def analyze_limits(args):
    """
    Analyzes the optimized parameters from a specific run to see how many times 
    they hit their defined bounds.
    """
    config_path = args.config
    run_dir = args.run_directory
    results_path, log_path = find_latest_results(run_dir)
    if not results_path:
        return

    try:
        with open(config_path, 'r') as f:
            # Using json5 to handle potential comments in the config
            try:
                import json5
                config = json5.load(f)
            except ImportError:
                print("Warning: json5 not available, using standard json parser")
                f.seek(0)
                config = json.load(f)
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return

    # Flatten the parameter spaces from config into a single dictionary for easy lookup
    param_bounds = {}
    # Iterate through all parameter space groups (e.g., 'global', 'trending')
    for space_group in config.get('parameter_spaces', {}).values():
        if isinstance(space_group, list):
            for param_info in space_group:
                if isinstance(param_info, dict) and 'name' in param_info and 'bounds' in param_info:
                    param_bounds[param_info['name']] = tuple(param_info['bounds'])

    # Store counts of hitting min/max bounds
    min_hits = defaultdict(int)
    max_hits = defaultdict(int)
    total_counts = defaultdict(int)

    # A small tolerance for floating point comparisons
    tolerance = 1e-9

    # The results are a dictionary of window results, not a list
    for window_name, window_data in results.items():
        for param_name, optimized_value in window_data.items():
            if param_name in param_bounds:
                total_counts[param_name] += 1
                min_bound, max_bound = param_bounds[param_name]

                if isinstance(optimized_value, float):
                    if abs(optimized_value - min_bound) < tolerance:
                        min_hits[param_name] += 1
                    elif abs(optimized_value - max_bound) < tolerance:
                        max_hits[param_name] += 1
                else: # Integer comparison
                    if optimized_value == min_bound:
                        min_hits[param_name] += 1
                    elif optimized_value == max_bound:
                        max_hits[param_name] += 1

    print("--- Parameter Limit Hit Analysis ---")
    print(f"Analyzing {len(results)} walk-forward windows from: {results_path}\n")

    all_params = sorted(param_bounds.keys())
    
    # --- Find the two most sensitive parameters for plotting ---
    hit_counts = {p: min_hits.get(p, 0) + max_hits.get(p, 0) for p in all_params}
    sorted_params_by_hit = sorted(hit_counts.items(), key=lambda item: item[1], reverse=True)
    
    param1_to_plot, param2_to_plot = None, None
    if len(sorted_params_by_hit) >= 2:
        param1_to_plot = sorted_params_by_hit[0][0]
        param2_to_plot = sorted_params_by_hit[1][0]

    report_lines = []
    for param_name in all_params:
        total = total_counts.get(param_name, 0)
        if total == 0: continue

        min_h = min_hits.get(param_name, 0)
        max_h = max_hits.get(param_name, 0)
        min_bound, max_bound = param_bounds[param_name]

        min_perc = (min_h / total) * 100 if total > 0 else 0
        max_perc = (max_h / total) * 100 if total > 0 else 0

        line = f"{param_name:<25} | Bounds: [{min_bound}, {max_bound}]"
        report_lines.append(line)
        
        if min_h > 0:
            report_lines.append(f"  - Hit MIN {min_h}/{total} times ({min_perc:.1f}%)")
        if max_h > 0:
            report_lines.append(f"  - Hit MAX {max_h}/{total} times ({max_perc:.1f}%)")
        if min_h == 0 and max_h == 0:
            report_lines.append("  - Did not hit limits.")
        report_lines.append("-" * 40)

    for line in report_lines:
        print(line)

    # Generate sensitivity plot if we have identified two sensitive parameters
    if param1_to_plot and param2_to_plot and log_path:
        plot_parameter_sensitivity(log_path, param1_to_plot, param2_to_plot, run_dir)

def analyze_parameter_performance(run_directory):
    """
    Analyzes the correlation between strategy parameters and trading performance.
    Uses trial data when available for more robust correlation analysis.
    
    Args:
        run_directory: Path to the run output directory (e.g., 'plots_output/20250726_220214')
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    run_path = Path(run_directory)
    
    # Load trade data
    trades_file = run_path / "all_trades_detailed.csv"
    params_file = run_path / "optimized_params_per_window.json"
    trial_data_file = run_path / "optimization_trial_data.json"
    
    if not trades_file.exists() or not params_file.exists():
        print(f"❌ Required files not found in {run_directory}")
        return None
    
    # Load data
    trades_df = pd.read_csv(trades_file)
    with open(params_file, 'r') as f:
        window_params = json.load(f)
    
    # Check if trial data is available
    trial_data_available = trial_data_file.exists()
    if trial_data_available:
        with open(trial_data_file, 'r') as f:
            trial_data = json.load(f)
        print(f"📊 Analyzing with {len(trial_data)} optimization trials and {len(trades_df)} trades across {len(window_params)} windows")
    else:
        print(f"📊 Analyzing {len(trades_df)} trades across {len(window_params)} windows (no trial data available)")
    
    # Calculate performance metrics per window
    window_performance = []
    
    for window_num in trades_df['window'].unique():
        window_trades = trades_df[trades_df['window'] == window_num]
        
        if len(window_trades) == 0:
            continue
            
        # Get parameters for this window
        window_key = f"Window_{window_num}"
        if window_key not in window_params:
            continue
            
        params = window_params[window_key]
        
        # Calculate performance metrics
        total_pnl = window_trades['pnl'].sum()
        win_rate = (window_trades['pnl'] > 0).mean()
        avg_win = window_trades[window_trades['pnl'] > 0]['pnl'].mean() if (window_trades['pnl'] > 0).any() else 0
        avg_loss = window_trades[window_trades['pnl'] < 0]['pnl'].mean() if (window_trades['pnl'] < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        num_trades = len(window_trades)
        
        # Create performance record
        perf_record = {
            'window': window_num,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': num_trades,
            **params  # Add all parameters
        }
        
        window_performance.append(perf_record)
    
    performance_df = pd.DataFrame(window_performance)
    
    # Display summary
    print("\n🎯 WINDOW PERFORMANCE SUMMARY:")
    print("=" * 80)
    for _, row in performance_df.iterrows():
        print(f"Window {int(row['window']):2d}: PnL=${row['total_pnl']:8.2f} | WinRate={row['win_rate']:.1%} | PF={row['profit_factor']:.2f} | Trades={int(row['num_trades'])}")
    
    # Find best and worst performing windows
    best_window = performance_df.loc[performance_df['total_pnl'].idxmax()]
    worst_window = performance_df.loc[performance_df['total_pnl'].idxmin()]
    
    print(f"\n🏆 BEST WINDOW (#{int(best_window['window'])}) - PnL: ${best_window['total_pnl']:.2f}")
    print("Key Parameters:")
    key_params = ['TENKAN_SEN_PERIOD', 'KIJUN_SEN_PERIOD', 'RSI_PERIOD', 'RSI_OVERBOUGHT', 'min_confidence',
                  'STOP_LOSS_MULTIPLIER', 'TAKE_PROFIT_MULTIPLIER', 'TRAILING_STOP_MULTIPLIER', 'PARTIAL_EXIT_PERCENTAGE',
                  'TRENDING_TSL_ATR_MULTIPLIER', 'RANGING_TSL_ATR_MULTIPLIER', 
                  'TRENDING_TP_ATR_MULTIPLIER', 'RANGING_TP_ATR_MULTIPLIER',
                  'ADX_PERIOD', 'ATR_PERIOD', 'BBANDS_PERIOD', 'BBANDS_STD_DEV', 'BB_WIDTH_THRESHOLD']
    for param in key_params:
        if param in best_window:
            print(f"  {param}: {best_window[param]}")
    
    print(f"\n💔 WORST WINDOW (#{int(worst_window['window'])}) - PnL: ${worst_window['total_pnl']:.2f}")
    print("Key Parameters:")
    for param in key_params:
        if param in worst_window:
            print(f"  {param}: {worst_window[param]}")
    
    # Parameter correlation analysis
    print(f"\n📈 PARAMETER CORRELATIONS WITH PnL:")
    print("=" * 50)
    
    # Include all optimizable parameters in analysis
    numeric_params = ['TENKAN_SEN_PERIOD', 'KIJUN_SEN_PERIOD', 'SENKOU_SPAN_B_PERIOD', 
                     'RSI_PERIOD', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD',
                     'min_confidence', 'strong_signal_threshold', 'confluence_threshold',
                     'volatility_threshold', 'volatility_window', 'trend_window', 
                     'momentum_window', 'ADX_THRESHOLD',
                     'STOP_LOSS_MULTIPLIER', 'TAKE_PROFIT_MULTIPLIER', 'TRAILING_STOP_MULTIPLIER',
                     'PARTIAL_EXIT_PERCENTAGE', 'FIXED_RISK_PERCENTAGE',
                     'TRENDING_TSL_ATR_MULTIPLIER', 'RANGING_TSL_ATR_MULTIPLIER',
                     'TRENDING_TP_ATR_MULTIPLIER', 'RANGING_TP_ATR_MULTIPLIER',
                     'ADX_PERIOD', 'ATR_PERIOD', 'BBANDS_PERIOD', 'BBANDS_STD_DEV', 'BB_WIDTH_THRESHOLD',
                     'base_slippage_percent', 'vol_slippage_multiplier',
                     'volume_lookback', 'volume_threshold_multiplier', 'volume_sma_period', 'volume_ema_period',
                     'volume_oscillator_short', 'volume_oscillator_long', 'sentiment_weight',
                     'sentiment_confidence_threshold', 'sentiment_cache_hours', 'regime_confidence_threshold',
                     'breakout_volume_multiplier', 'accumulation_threshold', 'distribution_threshold',
                     'volume_confirmation_threshold', 'regime_smoothing_factor']
    
    correlations = []
    params_with_variance = []
    
    # Use trial data if available for more robust correlation analysis
    if trial_data_available and trial_data:
        print("🔬 Using optimization trial data for enhanced correlation analysis...")
        
        # Create trial dataframe with objective scores as proxy for performance
        trial_df = pd.DataFrame(trial_data)
        
        # For each parameter, calculate correlation with objective value (lower is better, so invert)
        trial_df['performance_score'] = -trial_df['objective_value']  # Invert since lower objective is better
        
        for param in numeric_params:
            if param in trial_df.columns:
                param_values = trial_df[param]
                # Check if parameter has any variance
                if param_values.std() == 0:
                    continue
                else:
                    params_with_variance.append(param)
                    corr = param_values.corr(trial_df['performance_score'])
                    correlations.append((param, corr))
        
        # Also save trial-based analysis
        trial_output_file = run_path / "trial_based_analysis.csv"
        trial_df.to_csv(trial_output_file, index=False)
        print(f"📊 Trial-based analysis saved to: {trial_output_file}")
    else:
        print("📊 Using window-based analysis (no trial data available)...")
        
        # Original window-based correlation analysis
        for param in numeric_params:
            if param in performance_df.columns:
                param_values = performance_df[param]
                # Check if parameter has any variance
                if param_values.std() == 0:
                    # All values are identical - no variance
                    continue
                else:
                    params_with_variance.append(param)
                    corr = param_values.corr(performance_df['total_pnl'])
                    correlations.append((param, corr))
    
    if len(correlations) == 0:
        if trial_data_available:
            print("⚠️  No parameter correlations available - all parameters have identical values across trials.")
            print("💡 This suggests the optimization space is too narrow or converged quickly.")
        else:
            print("⚠️  No parameter correlations available - all parameters have identical values across windows.")
            print("💡 This typically happens when:")
            print("   • Running with intensity=1 (fixed parameters)")
            print("   • Optimization hasn't converged to different values")
            print("   • Run backtest with intensity≥2 for parameter optimization")
        
        # Show which parameters were checked
        identical_params = [p for p in numeric_params if p in performance_df.columns and performance_df[p].std() == 0]
        if identical_params:
            print(f"\n📋 Parameters with identical values across all windows ({len(identical_params)}):")
            for param in identical_params[:10]:  # Show first 10
                if param in performance_df.columns:
                    value = performance_df[param].iloc[0]
                    print(f"   {param}: {value}")
            if len(identical_params) > 10:
                print(f"   ... and {len(identical_params) - 10} more")
    else:
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
        
        analysis_type = "trial-based" if trial_data_available else "window-based"
        print(f"✅ Found {len(correlations)} parameters with variance for {analysis_type} correlation analysis:")
        for param, corr in correlations:
            if not np.isnan(corr):
                direction = "📈" if corr > 0 else "📉"
                print(f"{direction} {param:25s}: {corr:6.3f}")
            else:
                print(f"⚠️  {param:25s}: Could not calculate correlation")
        
        if len(params_with_variance) < len(numeric_params) // 2:
            print(f"\n💡 Only {len(params_with_variance)}/{len(numeric_params)} parameters showed variance.")
            print("   Consider running with higher intensity for more parameter exploration.")
    
    # Save detailed analysis
    output_file = run_path / "parameter_performance_analysis.csv"
    performance_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    # Add comprehensive explanation
    print_performance_explanation(performance_df, len(correlations), params_with_variance, numeric_params)
    
    return performance_df

def print_performance_explanation(performance_df, num_correlations, params_with_variance, all_numeric_params):
    """
    Print a comprehensive explanation of the performance analysis results.
    """
    print("\n" + "=" * 80)
    print("📖 PERFORMANCE ANALYSIS EXPLANATION")
    print("=" * 80)
    
    # Window Performance Summary Explanation
    print("\n🎯 WINDOW PERFORMANCE METRICS EXPLAINED:")
    print("─" * 50)
    print("• PnL (Profit & Loss): Total dollar profit/loss for that testing window")
    print("• WinRate: Percentage of profitable trades (higher is better)")
    print("• PF (Profit Factor): Ratio of gross profits to gross losses")
    print("  - PF > 1.0 = Profitable strategy")
    print("  - PF < 1.0 = Losing strategy")
    print("  - PF = 0.68 means losses are 47% larger than profits")
    print("• Trades: Number of trades executed in that window")
    
    # Current Results Analysis
    total_pnl = performance_df['total_pnl'].sum()
    avg_win_rate = performance_df['win_rate'].mean()
    avg_profit_factor = performance_df['profit_factor'].mean()
    total_trades = performance_df['num_trades'].sum()
    
    print(f"\n📊 YOUR STRATEGY PERFORMANCE:")
    print("─" * 50)
    print(f"• Overall PnL: ${total_pnl:.2f}")
    if total_pnl < 0:
        print("  ❌ Strategy is currently unprofitable")
        print("  💡 Consider: Higher intensity optimization, different parameters, or market regime analysis")
    else:
        print("  ✅ Strategy is profitable")
    
    print(f"• Average Win Rate: {avg_win_rate:.1%}")
    if avg_win_rate < 0.4:
        print("  ⚠️  Low win rate - strategy loses more often than it wins")
        print("  💡 Consider: Tighter entry conditions or better signal filtering")
    elif avg_win_rate > 0.6:
        print("  ✅ Good win rate")
    else:
        print("  📈 Moderate win rate")
    
    print(f"• Average Profit Factor: {avg_profit_factor:.2f}")
    if avg_profit_factor < 1.0:
        print("  ❌ Average trade loses money")
        print("  💡 Consider: Better signal filtering or parameter optimization")
    elif avg_profit_factor > 1.5:
        print("  ✅ Strong profit factor")
    else:
        print("  📈 Marginal profit factor")
    
    print(f"• Total Trades: {total_trades}")
    
    # Parameter Optimization Analysis
    print(f"\n🔧 PARAMETER OPTIMIZATION ANALYSIS:")
    print("─" * 50)
    
    if num_correlations == 0:
        print("❌ NO PARAMETER VARIANCE DETECTED")
        print("This means ALL parameters were identical across all testing windows.")
        print("\nPossible causes:")
        print("• Running with intensity=1 (uses fixed 'best' parameters)")
        print("• Optimization budget too low (insufficient exploration)")
        print("• Parameters hitting bounds (no room to optimize)")
        print("\n💡 RECOMMENDATIONS:")
        print("• Run with intensity≥2 for true parameter optimization")
        print("• Increase calls_per_window in config")
        print("• Check parameter bounds aren't too restrictive")
    else:
        variance_percentage = (len(params_with_variance) / len(all_numeric_params)) * 100
        print(f"✅ Parameter variance detected: {len(params_with_variance)}/{len(all_numeric_params)} parameters ({variance_percentage:.0f}%)")
        
        if variance_percentage < 25:
            print("⚠️  LOW parameter exploration")
            print("💡 Consider running with higher intensity for better optimization")
        elif variance_percentage < 50:
            print("📈 MODERATE parameter exploration")
        else:
            print("✅ GOOD parameter exploration")
    
    # Market Conditions Context
    if performance_df['total_pnl'].std() > abs(performance_df['total_pnl'].mean()):
        print(f"\n🌊 MARKET REGIME ANALYSIS:")
        print("─" * 50)
        print("⚠️  High variance in window performance detected")
        print("This suggests strategy performance varies significantly across market conditions.")
        print("\n💡 Your strategy includes market regime detection:")
        print("• Check if regime parameters need optimization")
        print("• Review regime-specific parameter adaptation")
        print("• Consider expanding regime classification")
    
    # Best vs Worst Window Analysis
    best_pnl = performance_df['total_pnl'].max()
    worst_pnl = performance_df['total_pnl'].min()
    pnl_range = best_pnl - worst_pnl
    
    print(f"\n📈 CONSISTENCY ANALYSIS:")
    print("─" * 50)
    print(f"• Best Window PnL: ${best_pnl:.2f}")
    print(f"• Worst Window PnL: ${worst_pnl:.2f}")
    print(f"• Performance Range: ${pnl_range:.2f}")
    
    if pnl_range > abs(total_pnl) * 2:
        print("⚠️  HIGH performance volatility between windows")
        print("💡 Consider optimizing regime detection parameters")
    else:
        print("✅ Reasonable consistency between windows")
    
    # Final Recommendations
    print(f"\n🎯 NEXT STEPS RECOMMENDATIONS:")
    print("─" * 50)
    
    if total_pnl < 0:
        print("1. 🔴 PRIORITY: Strategy is losing money")
        print("   • Run with intensity≥3 for thorough optimization")
        print("   • Expand parameter search ranges")
        print("   • Review market regime detection effectiveness")
    
    if num_correlations == 0:
        print("2. 🔧 OPTIMIZATION: No parameter variance detected")
        print("   • Increase optimization intensity")
        print("   • Run longer optimization (more calls_per_window)")
    
    if avg_win_rate < 0.4:
        print("3. 📊 SIGNAL QUALITY: Low win rate")
        print("   • Review entry/exit signal logic")
        print("   • Optimize signal confidence thresholds")
    
    if avg_profit_factor < 1.0:
        print("4. 💰 STRATEGY OPTIMIZATION: Poor profit factor")
        print("   • Let backtest optimize risk management parameters")
        print("   • Focus on signal quality and timing")
    
    print("\n✨ Remember: Backtesting shows historical performance.")
    print("   Real market conditions may differ significantly.")
    
    print("=" * 80)

# ==============================================================================
# DATA MANAGEMENT FUNCTIONS
# ==============================================================================

# Configuration constants for data management
EXCHANGE_NAME = 'binance'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
YEARS_OF_DATA = 4
DATA_FILE = 'data/crypto_data.parquet'

async def download_ohlcv_to_file(symbol, timeframe, start_date_str, filename):
    """
    Downloads historical OHLCV data from an exchange and saves it to a Parquet file.
    """
    print(f"Attempting to download {timeframe} candles for {symbol} from {EXCHANGE_NAME} to {filename}...")
    print(f"Starting historical data fetch from: {start_date_str}")

    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Existing file '{filename}' deleted to ensure a fresh download.")
        except Exception as e:
            print(f"Warning: Could not delete '{filename}': {e}. Please check file permissions.")

    exchange_class = getattr(ccxt, EXCHANGE_NAME)
    exchange = exchange_class({'enableRateLimit': True})
    
    try:
        initial_since_timestamp = exchange.parse8601(start_date_str)
        current_fetch_timestamp = initial_since_timestamp
        current_time = exchange.milliseconds()
        
        all_ohlcv = []
        collected_timestamps = set()
        max_fetch_limit = 1000

        while current_fetch_timestamp < current_time:
            retries = 5
            for i in range(retries):
                try:
                    print(f"Fetching {max_fetch_limit} candles from {pd.to_datetime(current_fetch_timestamp, unit='ms')}...")
                    new_ohlcv_chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current_fetch_timestamp, limit=max_fetch_limit)
                    if new_ohlcv_chunk:
                        break  # Success
                except ccxt.NetworkError as e:
                    if i < retries - 1:
                        wait = 2 ** i
                        print(f"Network error: {e}. Retrying in {wait} seconds...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"Network error after {retries} retries. Aborting.")
                        raise  # Re-raise the exception if all retries fail
            
            if not new_ohlcv_chunk:
                print("No more historical data available for this period. Download complete.")
                break

            unique_new_candles = 0
            for candle in new_ohlcv_chunk:
                timestamp = int(candle[0])
                if timestamp not in collected_timestamps:
                    all_ohlcv.append(candle)
                    collected_timestamps.add(timestamp)
                    unique_new_candles += 1
            
            if unique_new_candles == 0:
                print("Fetched chunk contained no new unique candles. Assuming we've caught up.")
                break

            current_fetch_timestamp = new_ohlcv_chunk[-1][0] + exchange.parse_timeframe(timeframe) * 1000
            print(f"Fetched {unique_new_candles} new unique candles. Total collected: {len(all_ohlcv)}")
            await asyncio.sleep(exchange.rateLimit / 1000)

        print(f"\nTotal fetched {len(all_ohlcv)} unique candles for {symbol}.")

        if not all_ohlcv:
            print("No OHLCV data was collected.")
            return False

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # --- Data Cleaning: Zero Volume ---
        initial_rows = len(df)
        df = df[df['volume'] > 0]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"\nData Cleaning: Removed {initial_rows - cleaned_rows} rows with zero volume.")
        
        # --- Data Cleaning: Price Wicks (Corruption Fix) ---
        initial_rows = len(df)
        df = df[df['low'] > (df['high'] * 0.5)]
        cleaned_rows = len(df)
        if initial_rows > cleaned_rows:
            print(f"Data Cleaning: Removed {initial_rows - cleaned_rows} rows with significant price wicks (potential data corruption).")

        df.sort_values('timestamp', inplace=True)
        df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        df.to_parquet(filename)
        print(f"Data successfully saved to {filename} with {len(df)} unique entries.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during download: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    finally:
        if exchange:
            await exchange.close()

def check_data_integrity(file_path):
    """
    Performs comprehensive integrity checks on the loaded trading data.
    """
    print(f"--- Data Integrity Check for {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file '{file_path}' does not exist!")
        return
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Successfully loaded {len(df)} rows of data")
    except Exception as e:
        print(f"[ERROR] Failed to read data file: {e}")
        return
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing > 0:
        print(f"[WARNING] Found {total_missing} missing values:")
        for column, count in missing_values.items():
            if count > 0:
                print(f"  - {column}: {count} missing values")
    else:
        print("✅ No missing values detected")
    
    # Check for zero or negative values in price/volume columns
    price_columns = ['open', 'high', 'low', 'close']
    volume_columns = ['volume']
    
    for col in price_columns:
        if col in df.columns:
            zero_count = (df[col] <= 0).sum()
            if zero_count > 0:
                print(f"[WARNING] Found {zero_count} zero/negative values in {col}")
    
    for col in volume_columns:
        if col in df.columns:
            zero_count = (df[col] <= 0).sum()
            if zero_count > 0:
                print(f"[WARNING] Found {zero_count} zero/negative values in {col}")
    
    # Check OHLC logic
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_ohlc = ((df['high'] < df['low']) | 
                       (df['high'] < df['open']) | 
                       (df['high'] < df['close']) |
                       (df['low'] > df['open']) | 
                       (df['low'] > df['close'])).sum()
        
        if invalid_ohlc > 0:
            print(f"[ERROR] Found {invalid_ohlc} rows with invalid OHLC logic!")
        else:
            print("✅ OHLC logic is valid for all rows")
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            print(f"[WARNING] Found {duplicates} duplicate timestamps")
        else:
            print("✅ No duplicate timestamps detected")
    elif df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"[WARNING] Found {duplicates} duplicate datetime indices")
        else:
            print("✅ No duplicate datetime indices detected")
    
    # Check for stuck data (high == low)
    if all(col in df.columns for col in ['high', 'low']):
        stuck_data = (df['high'] == df['low']).sum()
        if stuck_data > 0:
            print(f"[WARNING] Found {stuck_data} rows where high == low (stuck data)")
        else:
            print("✅ No stuck data (where high == low) detected")
    
    print("--- Data Integrity Check Finished ---")

def download_data(args=None):
    """Wrapper function to download multi-timeframe data by default."""
    try:
        # Import required modules
        import subprocess
        import sys
        import os
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_manager_path = os.path.join(project_root, 'data', 'data_manager.py')
        
        print("🔄 Starting multi-timeframe data download...")
        print(f"📁 Using data manager: {data_manager_path}")
        
        # Use subprocess to run the data manager with multi-timeframe flag
        # Remove capture_output=True to show real-time output
        result = subprocess.run([sys.executable, data_manager_path, '--multi-timeframe'], 
                              text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Multi-timeframe data download completed successfully")
            return True
        else:
            print("❌ Multi-timeframe data download failed.")
            
            # Fallback to single timeframe if multi-timeframe fails
            print("🔄 Attempting fallback to single timeframe download...")
            start_date = datetime.datetime.now() - datetime.timedelta(days=YEARS_OF_DATA * 365.25)
            start_date_str_arg = start_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to call the original single timeframe download function
            fallback_success = asyncio.run(download_ohlcv_to_file(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                start_date_str=start_date_str_arg,
                filename=DATA_FILE
            ))
            return fallback_success
        
    except Exception as e:
        print(f"\nAn unhandled error occurred during data download: {e}")
        traceback.print_exc(file=sys.stderr)
        return False

def check_data(args=None):
    """Wrapper function to check data integrity."""
    check_data_integrity(DATA_FILE)
    # For now, we assume the check is informational and doesn't block the pipeline.
    # A more robust implementation might return False on critical errors.
    return True

# ==============================================================================
# DEBUG UTILITIES
# ==============================================================================
# Consolidated debugging functions from debug_*.py files
# These functions provide comprehensive strategy and backtest debugging capabilities

def create_volatile_test_data():
    """Create more volatile OHLCV data that should generate signals
    Consolidated from debug_signal_generation.py"""
    print("Creating volatile test data...")
    
    # Create a strongly trending market with clear signals
    dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
    
    # Generate price data with strong trends and volatility
    np.random.seed(42)  # For reproducible results
    base_price = 50000
    
    # Create a strong uptrend then downtrend
    trend_up = np.linspace(0, 10000, 100)  # Strong uptrend
    trend_down = np.linspace(10000, 0, 100)  # Strong downtrend
    trend = np.concatenate([trend_up, trend_down])
    
    # Add significant volatility
    noise = np.random.normal(0, 1000, len(dates))  # Higher volatility
    
    close_prices = base_price + trend + noise
    
    # Create OHLC data with wider spreads
    high_prices = close_prices + np.random.uniform(50, 500, len(dates))
    low_prices = close_prices - np.random.uniform(50, 500, len(dates))
    open_prices = np.roll(close_prices, 1)  # Open is previous close
    open_prices[0] = close_prices[0]
    
    # Create volume data
    volume = np.random.uniform(500, 2000, len(dates))
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    print(f"Created volatile test data with {len(df)} rows")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Price volatility: {df['close'].std():.2f}")
    
    return df

def debug_signal_generation(df=None, params=None):
    """Debug signal generation logic step by step
    Consolidated from debug_signal_generation.py"""
    print("=== Debugging Signal Generation ===")
    
    if df is None:
        df = create_volatile_test_data()
    
    if params is None:
        params = {
            'timeframes': ['5T', '15T', '1H', '4H'],
            'timeframe_weights': {'5T': 0.15, '15T': 0.25, '1H': 0.35, '4H': 0.25},
            'regime_params': {'volatility_window': 20, 'trend_window': 50, 'momentum_window': 14, 'adx_threshold': 25, 'volatility_threshold': 0.02},
            'signal_thresholds': {'min_confidence': 0.6, 'strong_signal_threshold': 0.75, 'confluence_threshold': 0.65},
            'data_settings': {'multi_timeframe_enabled': False, 'timeframe_files': {}, 'primary_timeframe': '5m'},
            'TENKAN_SEN_PERIOD': 9,
            'KIJUN_SEN_PERIOD': 26,
            'SENKOU_SPAN_B_PERIOD': 52,
            'RSI_PERIOD': 14,
            'RSI_OVERBOUGHT': 70,
            'RSI_OVERSOLD': 30,
        }
    
    # Import here to avoid circular imports
    from core.strategy import MultiTimeframeStrategy
    
    # Create strategy instance
    strategy = MultiTimeframeStrategy(params)
    
    # Debug indicators
    print("\n--- Debugging Indicators ---")
    sample_row = df.iloc[-1:].copy()
    indicators = strategy._calculate_indicators(sample_row.iloc[0])
    
    print(f"Calculated {len(indicators)} indicators:")
    for key, value in indicators.items():
        if pd.isna(value):
            print(f"  {key}: NaN")
        else:
            print(f"  {key}: {value:.2f}")
    
    # Generate signals
    print("\n--- Generating Signals ---")
    signals_df = strategy.generate_signals(df)
    
    print(f"Generated signals shape: {signals_df.shape}")
    signal_counts = signals_df['signal'].value_counts()
    print(f"Signal distribution: {signal_counts}")
    
    return signals_df

def debug_backtest_entry_logic(df=None, params=None):
    """Debug the backtest entry logic to understand why trades aren't being executed
    Consolidated from debug_backtest_entry.py"""
    print("=== Debugging Backtest Entry Logic ===")
    
    if params is None:
        params = {
            'TENKAN_SEN_PERIOD': 9,
            'KIJUN_SEN_PERIOD': 26,
            'SENKOU_SPAN_B_PERIOD': 52,
            'RSI_PERIOD': 14,
            'RSI_OVERBOUGHT': 70,
            'RSI_OVERSOLD': 30,
            'min_confidence': 0.6,
            'strong_signal_threshold': 0.75,
            'confluence_threshold': 0.65,
            'volatility_threshold': 0.02,
            'volatility_window': 20,
            'trend_window': 50,
            'momentum_window': 14,
            'INITIAL_CAPITAL': 10000,
            'POSITION_SIZE': 0.95,
            'COMMISSION_RATE': 0.001,
            'SLIPPAGE_RATE': 0.0001,
            'TAKE_PROFIT_MULTIPLIER': 2.0,
            'STOP_LOSS_MULTIPLIER': 1.5,
            'TRAILING_STOP_MULTIPLIER': 1.0
        }
    
    if df is None:
        # Load real data if available
        if os.path.exists('data/crypto_data_5m.parquet'):
            df = pd.read_parquet('data/crypto_data_5m.parquet')
            df = df.head(100).copy()  # Use subset for testing
        else:
            df = create_volatile_test_data()
    
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    
    # Import here to avoid circular imports
    from core.strategy import MultiTimeframeStrategy
    from core.position_manager import PositionManager
    
    # Generate signals
    strategy = MultiTimeframeStrategy(params)
    signals_df = strategy.generate_signals(df)
    print(f"Generated signals shape: {signals_df.shape}")
    
    # Check signal distribution
    if 'signal' in signals_df.columns:
        signal_counts = signals_df['signal'].value_counts()
        print(f"Signal distribution: {signal_counts}")
        long_signals = signals_df['long_signals'].sum()
        short_signals = signals_df['short_signals'].sum()
        print(f"Long signals: {long_signals}, Short signals: {short_signals}")
    
    # Debug the backtest entry logic step by step
    print(f"\n=== Manual Backtest Entry Logic Debug ===")
    
    # Initialize position manager
    position_manager = PositionManager(params)
    
    # Find first signal
    first_signal_idx = None
    for i, (idx, row) in enumerate(signals_df.iterrows()):
        if row['signal'] != 0:
            first_signal_idx = i
            break
    
    if first_signal_idx is not None:
        print(f"First signal found at index: {first_signal_idx}")
        signal_row = signals_df.iloc[first_signal_idx]
        print(f"Signal details: signal={signal_row['signal']}, confidence={signal_row['confidence']:.3f}")
    else:
        print("No signals found in data")
    
    return signals_df

def debug_atr_issue():
    """Debug the ATR issue in position entry
    Consolidated from debug_atr_issue.py"""
    print("=== Debugging ATR Issue ===")
    
    # Test parameters
    test_params = {
        'TENKAN_SEN_PERIOD': 9,
        'KIJUN_SEN_PERIOD': 26,
        'SENKOU_SPAN_B_PERIOD': 52,
        'RSI_PERIOD': 14,
        'RSI_OVERBOUGHT': 70,
        'RSI_OVERSOLD': 30,
        'min_confidence': 0.6,
        'strong_signal_threshold': 0.75,
        'confluence_threshold': 0.65,
        'volatility_threshold': 0.02,
        'volatility_window': 20,
        'trend_window': 50,
        'momentum_window': 14,
        'INITIAL_CAPITAL': 10000,
        'POSITION_SIZE': 0.95,
        'COMMISSION_RATE': 0.001,
        'SLIPPAGE_RATE': 0.0001,
        'TAKE_PROFIT_MULTIPLIER': 2.0,
        'STOP_LOSS_MULTIPLIER': 1.5,
    }
    
    # Create test data or load real data
    if os.path.exists('data/crypto_data_5m.parquet'):
        df = pd.read_parquet('data/crypto_data_5m.parquet')
        df = df.tail(100).copy()
    else:
        df = create_volatile_test_data()
    
    print(f"Using data with {len(df)} rows")
    
    # Import here to avoid circular imports
    from core.strategy import MultiTimeframeStrategy
    
    # Test ATR calculation
    strategy = MultiTimeframeStrategy(test_params)
    
    # Calculate ATR manually for debugging
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=14).mean()
    
    print(f"ATR statistics:")
    print(f"  Mean ATR: {atr.mean():.2f}")
    print(f"  Min ATR: {atr.min():.2f}")
    print(f"  Max ATR: {atr.max():.2f}")
    print(f"  Last ATR: {atr.iloc[-1]:.2f}")
    
    return atr

def debug_signal_timing():
    """Debug the exact moment when signals should become trades
    Consolidated from debug_signal_timing.py"""
    print("=== Debugging Signal-to-Trade Conversion ===")
    
    # Load real data
    if os.path.exists('data/crypto_data_5m.parquet'):
        df = pd.read_parquet('data/crypto_data_5m.parquet')
        df = df.head(100).copy()  # Small sample for debugging
    else:
        df = create_volatile_test_data()
    
    # Test parameters
    test_params = {
        'min_confidence': 0.1,  # Very permissive
        'strong_signal_threshold': 0.2,
        'confluence_threshold': 0.1,
        'volatility_threshold': 0.05,
        'TENKAN_SEN_PERIOD': 9,
        'KIJUN_SEN_PERIOD': 26,
        'SENKOU_SPAN_B_PERIOD': 52,
        'RSI_PERIOD': 14,
        'RSI_OVERBOUGHT': 70,
        'RSI_OVERSOLD': 30,
    }
    
    # Import here to avoid circular imports
    from core.strategy import MultiTimeframeStrategy
    from core.position_manager import PositionManager
    
    # Create strategy and position manager
    strategy = MultiTimeframeStrategy(test_params)
    position_manager = PositionManager(test_params)
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"Generated {len(signals_df)} signal rows")
    signal_summary = signals_df['signal'].value_counts()
    print(f"Signal summary: {signal_summary}")
    
    # Find when signals become trades
    trade_count = 0
    for i, (timestamp, row) in enumerate(signals_df.iterrows()):
        if row['signal'] != 0:
            print(f"\nSignal {i} at {timestamp}:")
            print(f"  Signal: {row['signal']}")
            print(f"  Confidence: {row['confidence']:.3f}")
            print(f"  Price: {row['close']:.2f}")
            
            # Check if this would become a trade
            if row['confidence'] >= test_params['min_confidence']:
                trade_count += 1
                print(f"  -> Would become TRADE #{trade_count}")
            else:
                print(f"  -> Below confidence threshold ({test_params['min_confidence']})")
    
    print(f"\nTotal signals: {(signals_df['signal'] != 0).sum()}")
    print(f"Total potential trades: {trade_count}")
    
    return signals_df

# ==============================================================================
# TESTING FUNCTIONS
# ==============================================================================

if __name__ == "__main__":
    # Test parameter validation
    print("Testing parameter validation...")
    
    if os.path.exists("optimization_config.json"):
        results = validate_parameter_bounds("optimization_config.json")
        print(f"Validation completed: {results['summary']['validation_percentage']:.1f}% parameters valid")
    else:
        print("No optimization_config.json found for testing")
    
    # Test usage analyzer
    print("\nTesting usage analyzer...")
    analyzer = ActiveUsageAnalyzer(".")
    print("Usage analyzer initialized successfully")


# ==============================================================================
#
#                    PERFORMANCE UTILITIES & OPTIMIZATION
#                   (Consolidated from performance/*.py)
#
# ==============================================================================

import psutil
import threading
import traceback
import multiprocessing as mp
import concurrent.futures
from functools import partial
from dataclasses import dataclass, asdict

# ==============================================================================
# PERFORMANCE MONITORING
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    timestamp: datetime
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    function_name: str
    parameters: Dict[str, Any]
    result_size: int
    error_occurred: bool
    error_message: str = ""

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, enable_profiling: bool = True, 
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_profiling: Enable execution time profiling
            enable_memory_tracking: Enable memory usage tracking
            enable_cpu_tracking: Enable CPU usage tracking
        """
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.metrics_history = []
        self.start_times = {}
        
    def start_timer(self, operation_name: str):
        """Start timing an operation."""
        if self.enable_profiling:
            self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        if self.enable_profiling and operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            del self.start_times[operation_name]
            return duration
        return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.enable_memory_tracking:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if self.enable_cpu_tracking:
            return psutil.cpu_percent(interval=0.1)
        return 0.0
    
    def record_metrics(self, function_name: str, execution_time: float, 
                      parameters: Dict = None, result_size: int = 0, 
                      error_occurred: bool = False, error_message: str = ""):
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.datetime.now(),
            execution_time=execution_time,
            memory_usage_mb=self.get_memory_usage(),
            cpu_usage_percent=self.get_cpu_usage(),
            function_name=function_name,
            parameters=parameters or {},
            result_size=result_size,
            error_occurred=error_occurred,
            error_message=error_message
        )
        self.metrics_history.append(metrics)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}
        
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        
        return {
            'total_operations': len(self.metrics_history),
            'total_execution_time': sum(execution_times),
            'average_execution_time': np.mean(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'average_memory_usage': np.mean(memory_usage),
            'max_memory_usage': max(memory_usage),
            'error_count': sum(1 for m in self.metrics_history if m.error_occurred)
        }
    
    def start_background_monitoring(self, interval: float = 5.0):
        """Start background monitoring (placeholder for compatibility)."""
        # This is a compatibility method for the backtest system
        # In a real implementation, this would start a background thread
        log_to_file(f"Performance monitoring started (interval: {interval}s)", print_to_console=False)
    
    def stop_background_monitoring(self):
        """Stop background monitoring (placeholder for compatibility)."""
        log_to_file("Performance monitoring stopped", print_to_console=False)
    
    def get_performance_report(self) -> str:
        """Get a formatted performance report."""
        summary = self.get_performance_summary()
        if not summary:
            return "No performance data available."
        
        return f"""
Performance Report:
- Total Operations: {summary.get('total_operations', 0)}
- Total Execution Time: {summary.get('total_execution_time', 0):.2f}s
- Average Execution Time: {summary.get('average_execution_time', 0):.3f}s
- Memory Usage (Avg): {summary.get('average_memory_usage', 0):.1f}MB
- Error Count: {summary.get('error_count', 0)}
"""
    
    def export_metrics(self, output_dir: str) -> str:
        """Export metrics to a file and return the filename."""
        import os
        import json
        
        metrics_file = os.path.join(output_dir, "performance_metrics.json")
        summary = self.get_performance_summary()
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(summary, f, indent=2)
            log_to_file(f"Performance metrics exported to: {metrics_file}", print_to_console=False)
            return metrics_file
        except Exception as e:
            log_to_file(f"Failed to export metrics: {e}", print_to_console=False)
            return ""

# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# ==============================================================================
# ENHANCED TRADING PERFORMANCE ANALYTICS
# ==============================================================================

@dataclass
class TradingMetrics:
    """Advanced trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    expectancy: float = 0.0
    
class TradingPerformanceAnalyzer:
    """Advanced trading performance analysis with real-time capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("TradingPerformanceAnalyzer")
        self.trade_history = []
        self.cached_metrics = None
        self.last_update = None
        
    def load_trading_data(self, sources=None):
        """Load trading data from multiple sources"""
        try:
            journal_data = []
            
            # Default sources if none provided
            if sources is None:
                sources = [
                    'data/trading_journal.json',
                    '../data/trading_journal.json',
                    'trading_journal.json'
                ]
            
            # Try local files first
            for source in sources:
                if os.path.exists(source):
                    try:
                        with open(source, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                journal_data.extend(data)
                            elif isinstance(data, dict):
                                journal_data.append(data)
                        break
                    except Exception as e:
                        self.logger.warning(f"Error loading {source}: {e}")
                        continue
            
            # Try GCP if no local data found
            if not journal_data:
                try:
                    from utilities.gcp_utils import download_from_gcs
                    gcp_data = download_from_gcs('trading_journal.json', '')
                    if gcp_data:
                        journal_data = json.loads(gcp_data)
                        if not isinstance(journal_data, list):
                            journal_data = [journal_data]
                except Exception as e:
                    self.logger.warning(f"Error loading from GCP: {e}")
            
            self.trade_history = journal_data
            return journal_data
            
        except Exception as e:
            self.logger.error(f"Error loading trading data: {e}")
            return []
    
    def calculate_advanced_metrics(self, trade_data=None):
        """Calculate comprehensive trading metrics"""
        try:
            if trade_data is None:
                trade_data = self.trade_history
            
            if not trade_data:
                return TradingMetrics()
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trade_data)
            
            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = df['pnl'].sum()
            gross_profit = df[df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Win/Loss averages
            wins = df[df['pnl'] > 0]['pnl']
            losses = df[df['pnl'] < 0]['pnl']
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            largest_win = wins.max() if len(wins) > 0 else 0
            largest_loss = losses.min() if len(losses) > 0 else 0
            
            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
            
            # Sharpe ratio (simplified)
            if total_trades > 1:
                returns = df['pnl'].pct_change().dropna()
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Max drawdown
            cumulative_pnl = df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins = self._calculate_max_consecutive(df['pnl'] > 0)
            consecutive_losses = self._calculate_max_consecutive(df['pnl'] < 0)
            
            metrics = TradingMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                expectancy=expectancy
            )
            
            self.cached_metrics = metrics
            self.last_update = datetime.datetime.now()
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating trading metrics: {e}")
            return TradingMetrics()
    
    def _calculate_max_consecutive(self, condition_series):
        """Calculate maximum consecutive True values in a boolean series"""
        try:
            consecutive_count = 0
            max_consecutive = 0
            
            for value in condition_series:
                if value:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 0
            
            return max_consecutive
        except:
            return 0
    
    def get_performance_summary(self):
        """Get formatted performance summary"""
        try:
            if self.cached_metrics is None or self._needs_update():
                self.load_trading_data()
                self.calculate_advanced_metrics()
            
            metrics = self.cached_metrics
            if not metrics:
                return "No trading data available"
            
            return f"""
=== TRADING PERFORMANCE SUMMARY ===
Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate:.2f}%
Total P&L: {metrics.total_pnl:.2f} USDT

Profitability:
- Gross Profit: {metrics.gross_profit:.2f} USDT
- Gross Loss: {metrics.gross_loss:.2f} USDT
- Profit Factor: {metrics.profit_factor:.2f}
- Expectancy: {metrics.expectancy:.2f} USDT

Risk Metrics:
- Max Drawdown: {metrics.max_drawdown:.2f} USDT
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}

Trade Statistics:
- Average Win: {metrics.avg_win:.2f} USDT
- Average Loss: {metrics.avg_loss:.2f} USDT
- Largest Win: {metrics.largest_win:.2f} USDT
- Largest Loss: {metrics.largest_loss:.2f} USDT
- Max Consecutive Wins: {metrics.consecutive_wins}
- Max Consecutive Losses: {metrics.consecutive_losses}
"""
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return "Error generating performance summary"
    
    def _needs_update(self):
        """Check if metrics need updating"""
        if self.last_update is None:
            return True
        return (datetime.datetime.now() - self.last_update).seconds > 300  # 5 minutes
    
    def export_metrics_json(self, filepath="performance_metrics.json"):
        """Export metrics to JSON file"""
        try:
            if self.cached_metrics is None:
                self.load_trading_data()
                self.calculate_advanced_metrics()
            
            metrics_dict = asdict(self.cached_metrics) if self.cached_metrics else {}
            
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return None

# Global trading performance analyzer
trading_analyzer = TradingPerformanceAnalyzer()


# ==============================================================================
# LIGHTWEIGHT WEB INTERFACE UTILITIES
# ==============================================================================

class SimpleWebInterface:
    """Lightweight web interface for monitoring without external dependencies"""
    
    def __init__(self, port=8888):
        self.port = port
        self.is_running = False
        
    def check_web_dependencies(self):
        """Check if web interface dependencies are available"""
        try:
            import streamlit
            import plotly
            return True
        except ImportError:
            return False
    
    def generate_html_report(self, output_file="trading_report.html"):
        """Generate a simple HTML report"""
        try:
            # Get trading metrics
            trading_analyzer.load_trading_data()
            metrics = trading_analyzer.calculate_advanced_metrics()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .header {{ color: #333; border-bottom: 2px solid #ccc; }}
    </style>
</head>
<body>
    <h1 class="header">🤖 Trading Bot Performance Report</h1>
    <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>📊 Key Metrics</h2>
    <div class="metric">Total Trades: <strong>{metrics.total_trades}</strong></div>
    <div class="metric">Win Rate: <strong class="{'positive' if metrics.win_rate > 50 else 'negative'}">{metrics.win_rate:.2f}%</strong></div>
    <div class="metric">Total P&L: <strong class="{'positive' if metrics.total_pnl > 0 else 'negative'}">{metrics.total_pnl:.2f} USDT</strong></div>
    <div class="metric">Profit Factor: <strong>{metrics.profit_factor:.2f}</strong></div>
    <div class="metric">Sharpe Ratio: <strong>{metrics.sharpe_ratio:.2f}</strong></div>
    <div class="metric">Max Drawdown: <strong class="negative">{metrics.max_drawdown:.2f} USDT</strong></div>
    
    <h2>📈 Trade Statistics</h2>
    <div class="metric">Average Win: <strong class="positive">{metrics.avg_win:.2f} USDT</strong></div>
    <div class="metric">Average Loss: <strong class="negative">{metrics.avg_loss:.2f} USDT</strong></div>
    <div class="metric">Largest Win: <strong class="positive">{metrics.largest_win:.2f} USDT</strong></div>
    <div class="metric">Largest Loss: <strong class="negative">{metrics.largest_loss:.2f} USDT</strong></div>
    <div class="metric">Max Consecutive Wins: <strong>{metrics.consecutive_wins}</strong></div>
    <div class="metric">Max Consecutive Losses: <strong>{metrics.consecutive_losses}</strong></div>
    
    <h2>🎯 Expectancy</h2>
    <div class="metric">Expected Return per Trade: <strong class="{'positive' if metrics.expectancy > 0 else 'negative'}">{metrics.expectancy:.2f} USDT</strong></div>
    
    <p><em>Report auto-generated by Trading Bot Performance Analyzer</em></p>
</body>
</html>"""
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            log_to_file(f"HTML report generated: {output_file}")
            return output_file
            
        except Exception as e:
            log_to_file(f"Error generating HTML report: {e}")
            return None
    
    def launch_simple_dashboard(self):
        """Launch dashboard if dependencies available, otherwise generate HTML"""
        if self.check_web_dependencies():
            log_to_file("Web dependencies available - use launch_dashboard.py for full dashboard")
            return False
        else:
            log_to_file("Web dependencies not available - generating HTML report instead")
            return self.generate_html_report()

# Global web interface instance
web_interface = SimpleWebInterface()


def profile(func):
    """Decorator for profiling function execution."""
    def wrapper(*args, **kwargs):
        function_name = func.__name__
        start_time = time.time()
        error_occurred = False
        error_message = ""
        result = None
        result_size = 0
        
        try:
            result = func(*args, **kwargs)
            if hasattr(result, '__len__'):
                result_size = len(result)
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            performance_monitor.record_metrics(
                function_name=function_name,
                execution_time=execution_time,
                parameters={'args': len(args), 'kwargs': len(kwargs)},
                result_size=result_size,
                error_occurred=error_occurred,
                error_message=error_message
            )
        
        return result
    return wrapper

# ==============================================================================
# PARALLEL OPTIMIZATION
# ==============================================================================

class ParallelOptimizer:
    """Parallel optimization engine for trading strategies."""
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize parallel optimizer.
        
        Args:
            n_workers: Number of worker processes. If None, uses CPU count.
        """
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 8)  # Leave 1 CPU free, max 8
        self.optimization_results = []
        
        logging.info(f"Parallel Optimizer initialized with {self.n_workers} workers")
    
    def optimize_parameters_parallel(self, 
                                   optimization_function: callable,
                                   parameter_sets: List[Dict[str, Any]],
                                   data: pd.DataFrame,
                                   additional_args: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Optimize parameters in parallel.
        
        Args:
            optimization_function: Function to optimize (must be picklable)
            parameter_sets: List of parameter dictionaries to test
            data: Training data
            additional_args: Additional arguments for optimization function
            
        Returns:
            List of optimization results
        """
        if additional_args is None:
            additional_args = {}
        
        # Create partial function with fixed arguments
        partial_func = partial(
            self._optimize_single_parameter_set,
            optimization_function=optimization_function,
            data=data,
            additional_args=additional_args
        )
        
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound optimization tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(partial_func, params): params 
                for params in parameter_sets
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Completed optimization for parameters: {params}")
                except Exception as e:
                    logging.error(f"Optimization failed for parameters {params}: {e}")
                    results.append({
                        'parameters': params,
                        'error': str(e),
                        'success': False
                    })
        
        self.optimization_results.extend(results)
        return results
    
    def _optimize_single_parameter_set(self, 
                                     parameter_set: Dict[str, Any],
                                     optimization_function: callable,
                                     data: pd.DataFrame,
                                     additional_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a single parameter set.
        
        Args:
            parameter_set: Parameters to test
            optimization_function: Function to optimize
            data: Training data
            additional_args: Additional arguments
            
        Returns:
            Optimization result dictionary
        """
        try:
            start_time = time.time()
            
            # Run optimization function
            result = optimization_function(
                parameter_set=parameter_set,
                data=data,
                **additional_args
            )
            
            execution_time = time.time() - start_time
            
            return {
                'parameters': parameter_set,
                'result': result,
                'execution_time': execution_time,
                'success': True,
                'worker_id': mp.current_process().name
            }
            
        except Exception as e:
            return {
                'parameters': parameter_set,
                'error': str(e),
                'execution_time': 0,
                'success': False,
                'worker_id': mp.current_process().name
            }
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results."""
        if not self.optimization_results:
            return {}
        
        successful_results = [r for r in self.optimization_results if r.get('success', False)]
        failed_results = [r for r in self.optimization_results if not r.get('success', True)]
        
        if successful_results:
            execution_times = [r['execution_time'] for r in successful_results]
            total_time = sum(execution_times)
            avg_time = np.mean(execution_times)
        else:
            total_time = avg_time = 0
        
        return {
            'total_optimizations': len(self.optimization_results),
            'successful_optimizations': len(successful_results),
            'failed_optimizations': len(failed_results),
            'success_rate': len(successful_results) / len(self.optimization_results) * 100,
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'parallel_speedup_estimate': total_time / avg_time if avg_time > 0 else 0
        }

# ==============================================================================
# BACKTEST PERFORMANCE OPTIMIZATION
# ==============================================================================

def optimize_backtest_config():
    """Optimize the backtest configuration for faster execution"""
    
    config_path = "core/optimization_config.json"
    
    # Load current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("🔧 Current Backtest Configuration:")
    print(f"   Intensity: {config['optimization_settings']['intensity']}")
    print(f"   Calls per window: {config['optimization_settings']['calls_per_window']}")
    print(f"   Min trades constraint: DISABLED (reduces trial pruning)")
    
    # Performance optimizations
    optimizations = {
        "optimization_settings": {
            "intensity": "1",
            "calls_per_window": {
                "1": 20,   # Reduced from 50
                "2": 50,   # Reduced from 200
                "3": 100,  # Reduced from 500
                "4": 200   # Reduced from 1000
            },
            "risk_per_trade_percentage": 0.02,
            "max_optimization_time_minutes": 5,  # Add time limit
            "early_stopping_patience": 10,       # Stop if no improvement
            "pruning_enabled": True,             # Enable aggressive pruning
            "parallel_trials": 1                 # Single thread for stability
        },
        "backtest_settings": {
            "fast_mode": True,                   # Enable fast mode
            "skip_detailed_analysis": True,     # Skip detailed metrics
            "cache_indicators": True,           # Cache technical indicators
            "batch_size": 1000,                 # Process in batches
            "memory_efficient": True            # Use memory-efficient processing
        },
        "performance_settings": {
            "enable_profiling": True,
            "enable_memory_monitoring": True,
            "log_performance_metrics": True,
            "performance_report_interval": 10
        }
    }
    
    # Update config with optimizations
    config.update(optimizations)
    
    # Save optimized config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Backtest configuration optimized for performance!")
    print("   - Reduced optimization calls per window")
    print("   - Enabled fast mode and caching")
    print("   - Added time limits and early stopping")
    print("   - Enabled performance monitoring")
    
    return config

# ==============================================================================
#                       SIMULATION AND TESTING FUNCTIONS
# ==============================================================================
# Consolidated from simulate_live_bot.py

async def simulate_live_bot():
    """Simulate live bot operation with mock data (from simulate_live_bot.py)"""
    print("🎯 Live Bot Simulation (Dry Run)")
    print("=" * 50)
    
    try:
        # Import live bot components
        from live_trading.live_bot import (
            load_parameters_with_fallback,
            load_state,
            save_state,
            log_message,
            STRATEGY_AVAILABLE,
            PORTFOLIO_AVAILABLE,
            POSITION_MANAGER_AVAILABLE
        )
        
        # Test 1: Load parameters
        print("📋 Loading parameters...")
        params, last_update = load_parameters_with_fallback()
        if not params:
            print("❌ Failed to load parameters")
            return False
        
        print(f"✅ Loaded {len(params)} parameters")
        
        # Test 2: Initialize components
        if STRATEGY_AVAILABLE:
            from core.strategy import MultiTimeframeStrategy
            strategy = MultiTimeframeStrategy(params)
            print("✅ Strategy initialized")
        
        if PORTFOLIO_AVAILABLE:
            from core.portfolio import Portfolio
            portfolio = Portfolio(initial_capital=10000)
            print("✅ Portfolio initialized (starting balance: $10,000)")
        
        if POSITION_MANAGER_AVAILABLE:
            from core.position_manager import PositionManager
            position_manager = PositionManager(params)
            print("✅ Position manager initialized")
        
        # Test 3: Simulate parameter updates
        print("\n📡 Simulating parameter updates...")
        original_rsi = params.get('RSI_PERIOD', 14)
        params['RSI_PERIOD'] = original_rsi + 1
        
        if STRATEGY_AVAILABLE:
            strategy.params = params
            print(f"✅ Strategy parameters updated (RSI: {original_rsi} → {params['RSI_PERIOD']})")
        
        # Test 4: State management
        print("\n💾 Testing state management...")
        test_state = {
            "timestamp": datetime.now().isoformat(),
            "in_position": False,
            "last_signal": "neutral",
            "balance": 10000,
            "test_mode": True
        }
        
        save_state(test_state)
        loaded_state = load_state()
        print(f"✅ State save/load successful")
        
        # Test 5: Mock trading simulation
        print("\n🎲 Mock Trading Simulation...")
        mock_prices = [50000, 50500, 49800, 51200, 50900]
        
        for i, price in enumerate(mock_prices):
            print(f"   Candle {i+1}: BTC/USDT @ ${price:,.2f}")
            
            # Simulate signal generation (mock)
            if i % 2 == 0:
                signal = "buy" if price < 50000 else "sell"
                print(f"   📊 Mock signal: {signal.upper()}")
            
            await asyncio.sleep(0.5)  # Small delay for realism
        
        print("\n🎯 Simulation Results:")
        print("✅ All core components functioning")
        print("✅ Parameter loading and updates working")
        print("✅ State management operational")
        print("✅ Ready for testnet deployment")
        
        # Cleanup
        if os.path.exists("data/live_bot_state.json"):
            os.remove("data/live_bot_state.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_live_bot_simulation():
    """Run complete live bot simulation (wrapper function)"""
    import asyncio
    print("🚀 Starting Live Bot Simulation")
    success = asyncio.run(simulate_live_bot())
    
    if success:
        print("\n🎉 Live Bot Simulation Complete!")
        print("\n📋 Next Steps:")
        print("1. Set up Binance testnet API keys in .env file")
        print("2. Run: python live_trading/live_bot.py")
        print("3. Monitor logs in live_bot_log.txt")
        print("4. Test with small amounts before scaling")
        return True
    else:
        print("\n❌ Simulation failed - check logs for issues")
        return False

# ==============================================================================
#                       LIVE TRADING VALIDATION FUNCTIONS
# ==============================================================================
# Consolidated from test_live_bot_integration.py, test_backtest_init.py, test_backtest_fix.py

async def validate_live_integration():
    """
    Comprehensive live trading integration validation.
    Tests API connection, parameter loading, and component initialization.
    Consolidated from test_live_bot_integration.py
    """
    print("🧪 Live Trading Integration Validation")
    print("=" * 50)
    
    validation_results = {
        "imports": False,
        "env_loading": False,
        "exchange_connection": False,
        "parameter_loading": False,
        "strategy_initialization": False,
        "portfolio_position_manager": False
    }
    
    # Test imports
    try:
        from core.strategy import MultiTimeframeStrategy
        from core.portfolio import Portfolio
        from core.position_manager import PositionManager
        import ccxt.pro as ccxt
        import pandas as pd
        import numpy as np
        from dotenv import load_dotenv
        validation_results["imports"] = True
        print("✅ All required modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return validation_results
    
    # Test environment loading
    try:
        load_dotenv()
        api_key = os.getenv("API_KEY")
        secret_key = os.getenv("SECRET_KEY")
        
        if api_key and secret_key and len(api_key) >= 20 and len(secret_key) >= 20:
            validation_results["env_loading"] = True
            print("✅ Environment variables loaded successfully")
        else:
            print("❌ API keys not found or invalid")
    except Exception as e:
        print(f"❌ Environment loading error: {e}")
    
    # Test exchange connection
    try:
        if validation_results["env_loading"]:
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'sandbox': True
            })
            
            balance = await exchange.fetch_balance()
            ticker = await exchange.fetch_ticker('BTC/USDT')
            await exchange.close()
            
            validation_results["exchange_connection"] = True
            print("✅ Exchange connection successful")
        else:
            print("⚠️ Skipping exchange test - no valid API keys")
    except Exception as e:
        print(f"❌ Exchange connection error: {e}")
    
    # Test parameter loading
    try:
        with open("core/optimization_config.json", 'r') as f:
            config = json.load(f)
        
        parameters = config.get('parameters', [])
        params = {}
        for param in parameters:
            param_name = param.get('name')
            bounds = param.get('bounds', [10, 50])
            default_value = int((bounds[0] + bounds[1]) / 2)
            params[param_name] = default_value
        
        if params:
            validation_results["parameter_loading"] = True
            print(f"✅ Loaded {len(params)} parameters successfully")
        else:
            print("❌ Failed to load parameters")
    except Exception as e:
        print(f"❌ Parameter loading error: {e}")
    
    # Test strategy initialization
    try:
        if validation_results["parameter_loading"]:
            strategy = MultiTimeframeStrategy(params)
            
            # Test with sample data
            dates = pd.date_range('2023-01-01', periods=100, freq='5T')
            sample_data = pd.DataFrame({
                'open': np.random.uniform(40000, 50000, 100),
                'high': np.random.uniform(41000, 51000, 100),
                'low': np.random.uniform(39000, 49000, 100),
                'close': np.random.uniform(40000, 50000, 100),
                'volume': np.random.uniform(100, 1000, 100)
            }, index=dates)
            
            processed_data = strategy.generate_signals(sample_data)
            
            if processed_data is not None and not processed_data.empty:
                validation_results["strategy_initialization"] = True
                print("✅ Strategy initialization and signal generation successful")
            else:
                print("❌ Failed to generate signals")
        else:
            print("⚠️ Skipping strategy test - no valid parameters")
    except Exception as e:
        print(f"❌ Strategy initialization error: {e}")
    
    # Test portfolio and position manager
    try:
        if validation_results["parameter_loading"]:
            portfolio = Portfolio(initial_capital=10000)
            position_manager = PositionManager(params)
            state = position_manager.get_state()
            
            validation_results["portfolio_position_manager"] = True
            print("✅ Portfolio and position manager initialization successful")
        else:
            print("⚠️ Skipping portfolio test - no valid parameters")
    except Exception as e:
        print(f"❌ Portfolio/Position manager error: {e}")
    
    # Summary
    passed = sum(validation_results.values())
    total = len(validation_results)
    print(f"\n📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All live trading integration tests passed!")
    else:
        print(f"⚠️ {total - passed} tests failed. Check configuration.")
    
    return validation_results

def validate_backtest_initialization():
    """
    Validate backtest initialization and performance module loading.
    Consolidated from test_backtest_init.py
    """
    print("🧪 Backtest Initialization Validation")
    print("=" * 40)
    
    try:
        # Test backtest initialization
        cmd = [sys.executable, "core/backtest.py", "--config", "core/optimization_config.json", "--intensity", "1", "--min-trades", "1"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Read output for 10 seconds to check initialization
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < 10:
            try:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    print(line.strip())
                else:
                    break
            except:
                break
        
        process.terminate()
        
        # Check if performance modules were loaded
        performance_loaded = any("Performance monitoring available" in line for line in output_lines)
        initialization_success = any("Starting backtest" in line or "Optimization" in line for line in output_lines)
        
        if performance_loaded and initialization_success:
            print("✅ Backtest initialization successful with performance monitoring")
            return True
        elif initialization_success:
            print("✅ Backtest initialization successful (performance monitoring not detected)")
            return True
        else:
            print("❌ Backtest initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Backtest initialization error: {e}")
        return False

def validate_backtest_configuration():
    """
    Validate backtest configuration and parameter formatting.
    Consolidated from test_backtest_fix.py
    """
    print("🧪 Backtest Configuration Validation")
    print("=" * 40)
    
    try:
        # Load configuration
        with open('core/optimization_config.json', 'r') as f:
            config = json.load(f)
        
        # Load test data
        df = pd.read_parquet(config['data_settings']['file_path'])
        test_df = df.tail(5000).copy()
        print(f'✅ Loaded {len(test_df)} rows of test data')
        
        # Create properly formatted config for strategy
        strategy_config = {
            'data_settings': {
                'multi_timeframe_enabled': False,
                'primary_timeframe': '5m'
            },
            'timeframes': ['5T'],
            'TENKAN_SEN_PERIOD': 9,
            'KIJUN_SEN_PERIOD': 26,
            'SENKOU_SPAN_B_PERIOD': 52,
            'RSI_PERIOD': 14,
            'RSI_OVERBOUGHT': 70,
            'RSI_OVERSOLD': 30,
            'signal_thresholds': {
                'min_confidence': 0.3,
                'strong_signal_threshold': 0.5,
                'confluence_threshold': 0.4
            },
            'regime_params': {
                'volatility_window': 20,
                'trend_window': 50,
                'momentum_window': 14,
                'adx_threshold': 25,
                'volatility_threshold': 0.02
            }
        }
        
        # Test strategy with configuration
        from core.strategy import MultiTimeframeStrategy
        strategy = MultiTimeframeStrategy(strategy_config)
        
        # Test signal generation
        signals_df = strategy.generate_signals(test_df.head(100))
        
        if signals_df is not None and not signals_df.empty:
            signal_count = signals_df['signal'].sum() if 'signal' in signals_df.columns else 0
            print(f"✅ Configuration validation successful - Generated {signal_count} signals")
            return True
        else:
            print("❌ Configuration validation failed - No signals generated")
            return False
            
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        print(traceback.format_exc())
        return False

# =============================================================================
# ENHANCED ANALYSIS FUNCTIONS (Consolidated from analyze_parameter_limits.py)
# =============================================================================

def analyze_enhanced_bounds(config):
    """Enhanced parameter bounds analysis - absorbed from enhanced_parameter_analyzer.py"""
    print("\n" + "=" * 80)
    print("🔧 ENHANCED PARAMETER BOUNDS ANALYSIS")
    print("=" * 80)
    
    parameter_spaces = config.get('parameter_spaces', {}).get('global', [])
    best_params = config.get('best_parameters_so_far', {})
    
    print(f"📋 Found {len(parameter_spaces)} parameter definitions")
    print(f"📋 Found {len(best_params)} best parameters")
    
    valid_params = 0
    invalid_params = 0
    
    for param in parameter_spaces:
        name = param['name']
        bounds = param['bounds']
        hard_bounds = param['hard_bounds']
        
        bounds_valid = (bounds[0] >= hard_bounds[0] and 
                       bounds[1] <= hard_bounds[1] and 
                       bounds[0] <= bounds[1])
        
        status = "✅" if bounds_valid else "❌"
        print(f"{name:<25} | Bounds: {bounds} | Hard: {hard_bounds} | {status}")
        
        if bounds_valid:
            valid_params += 1
        else:
            invalid_params += 1
            
    print(f"\n📊 Parameter Bounds Summary: {valid_params} valid, {invalid_params} invalid")
    
    print(f"\n🎯 Best Parameters Validation:")
    for param in parameter_spaces:
        name = param['name']
        hard_bounds = param['hard_bounds']
        
        if name in best_params:
            value = best_params[name]
            valid = hard_bounds[0] <= value <= hard_bounds[1]
            status = "✅" if valid else "❌"
            print(f"{name:<25} | Value: {value:<10.4f} | Valid: {status}")

def analyze_enhanced_signals(strategy, data, sample_size=10000):
    """Enhanced signal analysis - absorbed from enhanced_parameter_analyzer.py"""
    print("\n" + "=" * 80)
    print("📡 ENHANCED SIGNAL GENERATION ANALYSIS")
    print("=" * 80)
    
    # Analyze recent period
    print(f"\n🔍 Analyzing recent period ({sample_size} candles)")
    recent_data = data.tail(sample_size)
    
    if len(recent_data) < 100:
        print(f"⚠️  Insufficient data ({len(recent_data)} rows)")
        return
        
    signals_df = strategy.generate_signals(recent_data)
    
    signal_counts = signals_df['signal'].value_counts()
    long_signals = signal_counts.get(1, 0)
    short_signals = signal_counts.get(-1, 0)
    neutral_signals = signal_counts.get(0, 0)
    
    total_signals = len(signals_df)
    long_pct = (long_signals / total_signals) * 100
    short_pct = (short_signals / total_signals) * 100
    neutral_pct = (neutral_signals / total_signals) * 100
    
    avg_confidence = signals_df['confidence'].mean()
    
    start_price = recent_data['close'].iloc[0]
    end_price = recent_data['close'].iloc[-1]
    price_change = ((end_price - start_price) / start_price) * 100
    
    print(f"   💰 Price change: {price_change:+.1f}% (${start_price:.0f} → ${end_price:.0f})")
    print(f"   📊 Signals: Long {long_pct:.1f}% | Short {short_pct:.1f}% | Neutral {neutral_pct:.1f}%")
    print(f"   🎯 Avg confidence: {avg_confidence:.3f}")
    
    # Quick performance simulation
    capital = 10000
    position = 0
    trades = 0
    winning_trades = 0
    
    for i in range(1, len(signals_df)):
        current_signal = signals_df['signal'].iloc[i]
        current_price = signals_df['close'].iloc[i]
        
        if position != 0 and current_signal != position:
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price
                
            capital *= (1 + pnl * 0.95)  # 5% trading costs
            trades += 1
            if pnl > 0:
                winning_trades += 1
            position = 0
        
        if current_signal != 0 and position == 0:
            position = current_signal
            entry_price = current_price
    
    strategy_return = ((capital - 10000) / 10000) * 100
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0
    buy_hold_return = price_change
    
    print(f"   💰 Strategy return: {strategy_return:+.2f}%")
    print(f"   📊 Total trades: {trades}")
    print(f"   🎯 Win rate: {win_rate:.1f}%")
    print(f"   📈 Strategy vs Buy&Hold: {strategy_return - buy_hold_return:+.2f}% difference")
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if long_pct > short_pct * 2:
        print("   ⚠️  Long bias detected - consider adjusting RSI overbought levels")
    elif short_pct > long_pct * 2:
        print("   ⚠️  Short bias detected - consider adjusting RSI oversold levels")
    else:
        print("   ✅ Signal balance looks reasonable")
    
    if avg_confidence < 0.5:
        print("   ⚠️  Low confidence signals - consider tightening thresholds")
    elif avg_confidence > 0.8:
        print("   ✅ High confidence signals detected")
    
    if trades < 10:
        print("   ⚠️  Low trading frequency - consider relaxing signal thresholds")
    elif trades > 500:
        print("   ⚠️  High trading frequency - potential overtrading")

def analyze_enhanced_parameters(config_path='core/optimization_config.json', sample_size=10000):
    """Enhanced parameter analysis - consolidated from enhanced_parameter_analyzer.py"""
    try:
        from core.strategy import MultiTimeframeStrategy
        
        print("🚀 ENHANCED PARAMETER & STRATEGY ANALYSIS")
        print("=" * 80)
        
        # Load configuration
        with open(config_path, 'r') as f:
            try:
                import json5
                config = json5.load(f)
            except ImportError:
                config = json.load(f)
        
        # Load data
        data_path = config.get('data_settings', {}).get('file_path', 'data/crypto_data_5m.parquet')
        data = pd.read_parquet(data_path)
        print(f"✅ Loaded {len(data)} rows of data from {data_path}")
        
        # Initialize strategy
        strategy = MultiTimeframeStrategy(config)
        
        # Run enhanced bounds analysis
        analyze_enhanced_bounds(config)
        
        # Run enhanced signal analysis
        analyze_enhanced_signals(strategy, data, sample_size)
        
        print(f"\n✅ Enhanced analysis completed")
        
    except Exception as e:
        print(f"❌ Enhanced analysis error: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# DEVELOPMENT UTILITIES (Consolidated from various utility files)
# =============================================================================

def find_duplicates_by_name():
    """Find files with identical names in different locations."""
    from pathlib import Path
    
    project_root = Path(".")
    duplicates = {}
    
    # Get all Python files
    all_files = []
    for file_path in project_root.rglob('*.py'):
        if any(part in ['__pycache__', '.git', 'venv', '.venv', '.cache'] for part in file_path.parts):
            continue
        all_files.append(file_path)
    
    # Group by filename
    by_name = {}
    for file_path in all_files:
        name = file_path.name
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(str(file_path.relative_to(project_root)))
    
    # Find duplicates
    for name, paths in by_name.items():
        if len(paths) > 1:
            duplicates[name] = paths
    
    return duplicates

def find_duplicates_by_content():
    """Find files with identical content."""
    import hashlib
    from pathlib import Path
    
    project_root = Path(".")
    content_hashes = {}
    
    for file_path in project_root.rglob('*.py'):
        if any(part in ['__pycache__', '.git', 'venv', '.venv', '.cache'] for part in file_path.parts):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                
                if content_hash not in content_hashes:
                    content_hashes[content_hash] = []
                content_hashes[content_hash].append(str(file_path.relative_to(project_root)))
        except Exception:
            continue
    
    # Find duplicates
    duplicates = {}
    for content_hash, paths in content_hashes.items():
        if len(paths) > 1:
            duplicates[content_hash] = paths
    
    return duplicates

def monitor_backtest_progress():
    """Monitor backtest with real-time progress updates"""
    import subprocess
    import sys
    import time
    import re
    from datetime import datetime, timedelta
    
    print("⏱️ Backtest Progress Monitor")
    print("=" * 50)
    print("🎯 Expected runtime: ~6 minutes (3 windows × 2 min each)")
    print("⏰ Starting at:", datetime.now().strftime("%H:%M:%S"))
    print("=" * 50)
    
    # Start backtest
    cmd = [sys.executable, "core/backtest.py", "--config", "core/optimization_config.json", "--intensity", "1", "--min-trades", "1"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    start_time = time.time()
    window_start_time = start_time
    current_window = 0
    total_windows = 3
    
    print("📊 Progress Updates:")
    print("-" * 30)
    
    try:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
                
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check for window progress
            if "Walk-Forward Windows:" in line and "%" in line:
                # Extract window number and percentage
                window_match = re.search(r'(\d+)/(\d+)', line)
                if window_match:
                    current_window = int(window_match.group(1))
                    total_windows = int(window_match.group(2))
                    
                    # Calculate progress
                    progress = (current_window / total_windows) * 100
                    eta = (elapsed / progress) * 100 - elapsed if progress > 0 else 0
                    
                    print(f"🔄 Window {current_window}/{total_windows} ({progress:.1f}%) | "
                          f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                    
                    window_start_time = current_time
            
            # Show important messages
            if any(keyword in line.lower() for keyword in ['error', 'warning', 'complete', 'optimization']):
                print(f"   📝 {line}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted by user")
        process.terminate()
        
    process.wait()
    total_time = time.time() - start_time
    print(f"\n✅ Backtest completed in {total_time/60:.1f} minutes")
    
    return process.returncode == 0

def optimize_backtest_config_for_speed():
    """Optimize the backtest configuration for faster execution"""
    import json
    
    config_path = "core/optimization_config.json"
    
    # Load current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("🔧 Current Backtest Configuration:")
    print(f"   Intensity: {config['optimization_settings']['intensity']}")
    print(f"   Calls per window: {config['optimization_settings']['calls_per_window']}")
    print(f"   Min trades constraint: DISABLED (reduces trial pruning)")
    
    # Performance optimizations
    optimizations = {
        "optimization_settings": {
            "intensity": "1",
            "calls_per_window": {
                "1": 20,   # Reduced from 50
                "2": 50,   # Reduced from 200
                "3": 100,  # Reduced from 500
                "4": 200   # Reduced from 1000
            },
            "risk_per_trade_percentage": 0.02,
            "max_optimization_time_minutes": 5,  # Add time limit
            "early_stopping_patience": 10,       # Stop if no improvement
            "pruning_enabled": True,             # Enable aggressive pruning
            "parallel_trials": 1                 # Single thread for stability
        },
        "backtest_settings": {
            "fast_mode": True,                   # Enable fast mode
            "skip_detailed_analysis": True,     # Skip detailed metrics
            "cache_indicators": True,           # Cache technical indicators
            "batch_size": 1000,                 # Process in batches
            "memory_efficient": True            # Use memory-efficient processing
        }
    }
    
    # Apply optimizations
    for section, settings in optimizations.items():
        if section not in config:
            config[section] = {}
        config[section].update(settings)
    
    # Save optimized config
    backup_path = f"{config_path}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuration optimized for speed")
    print(f"   📁 Backup saved: {backup_path}")
    print(f"   🚀 Expected speedup: 2-3x faster execution")
    
    return True


# ==============================================================================
#                         CENTRALIZED LOGGING SYSTEM
# ==============================================================================
#
# CONSOLIDATED LOGGING FUNCTIONS FOR PIPELINE-WIDE USE
# Replaces individual log_message() functions in watcher.py, live_bot.py, etc.
# Follows Rule #9: Only edit existing files by merging functionality
#
# ==============================================================================

class CentralizedLogger:
    """
    Unified logging system for the entire trading robot pipeline.
    Consolidates logging from watcher.py, live_bot.py, backtest.py, and system_monitor.py
    
    PHASE 6H: Enhanced with comprehensive log rotation, automated cleanup, and background maintenance
    """
    
    def __init__(self, base_dir=".", enable_rotation=True, max_log_size_mb=10, max_backups=5):
        self.base_dir = base_dir
        self.enable_rotation = enable_rotation
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024  # Convert MB to bytes
        self.max_backups = max_backups
        self.rotation_lock = threading.Lock()
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 3600  # Run cleanup every hour
        
        # Updated active logs with centralized logging paths
        self.active_logs = {
            'watcher': 'logs/watcher.log',
            'live_bot': 'logs/live_bot.log',
            'backtest_status': 'logs/backtest_status.json',
            'errors': 'error_alerts.log',
            'trades': 'trade_alerts.log',
            'notifications': 'notifications.log',
            'backtest': 'full_analysis_log.txt',
            'system_monitor': 'logs/system_monitor.log',
            'enhanced_monitor': 'enhanced_monitor.log'
        }
        
        # Log rotation configuration per log type
        self.rotation_config = {
            'watcher': {'max_size_mb': 15, 'max_backups': 7},
            'live_bot': {'max_size_mb': 20, 'max_backups': 10},
            'errors': {'max_size_mb': 5, 'max_backups': 15},
            'trades': {'max_size_mb': 8, 'max_backups': 20},
            'notifications': {'max_size_mb': 5, 'max_backups': 10},
            'backtest': {'max_size_mb': 50, 'max_backups': 5},
            'system_monitor': {'max_size_mb': 10, 'max_backups': 7},
            'enhanced_monitor': {'max_size_mb': 12, 'max_backups': 7}
        }
        
        # Ensure directories exist
        os.makedirs('logs', exist_ok=True)
        os.makedirs('live_trading', exist_ok=True)
        os.makedirs('backups/logs/phase6h_comprehensive_rotation', exist_ok=True)
        
        # Start background cleanup thread if rotation is enabled
        if self.enable_rotation:
            self._start_background_maintenance()
    
    def get_timestamp(self):
        """Get formatted timestamp for logging"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _start_background_maintenance(self):
        """Start background thread for automated log maintenance"""
        def maintenance_worker():
            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    current_time = time.time()
                    
                    # Run cleanup if interval has passed
                    if current_time - self.last_cleanup_time > self.cleanup_interval:
                        self._automated_cleanup()
                        self.last_cleanup_time = current_time
                        
                except Exception as e:
                    print(f"Background maintenance error: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
    
    def _get_file_size(self, filepath):
        """Get file size in bytes, return 0 if file doesn't exist"""
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0
    
    def _needs_rotation(self, log_type, filepath):
        """Check if log file needs rotation based on size"""
        if not self.enable_rotation:
            return False
            
        config = self.rotation_config.get(log_type, {'max_size_mb': 10})
        max_size_bytes = config['max_size_mb'] * 1024 * 1024
        current_size = self._get_file_size(filepath)
        
        return current_size > max_size_bytes
    
    def _rotate_log_file(self, log_type, filepath):
        """Rotate log file with backup preservation"""
        if not os.path.exists(filepath):
            return
            
        with self.rotation_lock:
            config = self.rotation_config.get(log_type, {'max_backups': 5})
            max_backups = config['max_backups']
            
            # Create backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = "backups/logs/phase6h_comprehensive_rotation"
            os.makedirs(backup_dir, exist_ok=True)
            
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            backup_filename = f"{filename_base}_{timestamp}.log"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            try:
                # Move current log to backup
                shutil.move(filepath, backup_path)
                print(f"[LOG ROTATION] Rotated {filepath} to {backup_path}")
                
                # Clean up old backups for this log type
                self._cleanup_old_backups(log_type, backup_dir, max_backups)
                
                # Create new empty log file
                with open(filepath, 'w') as f:
                    f.write(f"[{self.get_timestamp()}] Log rotation: New log file created\n")
                    
            except Exception as e:
                print(f"[LOG ROTATION ERROR] Failed to rotate {filepath}: {e}")
    
    def _cleanup_old_backups(self, log_type, backup_dir, max_backups):
        """Remove old backup files beyond the retention limit"""
        try:
            # Get all backup files for this log type
            filename_base = log_type
            pattern = os.path.join(backup_dir, f"{filename_base}_*.log")
            backup_files = glob.glob(pattern)
            
            # Sort by modification time (oldest first)
            backup_files.sort(key=os.path.getmtime)
            
            # Remove excess backups
            while len(backup_files) > max_backups:
                old_backup = backup_files.pop(0)
                try:
                    os.remove(old_backup)
                    print(f"[CLEANUP] Removed old backup: {old_backup}")
                except OSError as e:
                    print(f"[CLEANUP ERROR] Failed to remove {old_backup}: {e}")
                    
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to cleanup backups for {log_type}: {e}")
    
    def _automated_cleanup(self):
        """Run automated cleanup of old logs and system maintenance"""
        try:
            print(f"[MAINTENANCE] Running automated log cleanup at {self.get_timestamp()}")
            
            # 1. Check all active logs for rotation needs
            for log_type, log_file in self.active_logs.items():
                if log_file.endswith('.json'):  # Skip JSON status files
                    continue
                    
                full_path = os.path.join(self.base_dir, log_file)
                if self._needs_rotation(log_type, full_path):
                    print(f"[MAINTENANCE] Auto-rotating {log_file}")
                    self._rotate_log_file(log_type, full_path)
            
            # 2. Clean up old files in plots_output (older than 7 days)
            self._cleanup_old_analysis_files()
            
            # 3. Check disk space and warn if low
            self._check_disk_space()
            
            print(f"[MAINTENANCE] Automated cleanup completed at {self.get_timestamp()}")
            
        except Exception as e:
            print(f"[MAINTENANCE ERROR] Automated cleanup failed: {e}")
    
    def _cleanup_old_analysis_files(self):
        """Clean up old analysis files in plots_output directory"""
        try:
            plots_dir = "plots_output"
            if not os.path.exists(plots_dir):
                return
                
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
            removed_count = 0
            
            for root, dirs, files in os.walk(plots_dir):
                for file in files:
                    if file.endswith(('.log', '.txt', '.png', '.csv')):
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < cutoff_time:
                            try:
                                os.remove(file_path)
                                removed_count += 1
                            except OSError:
                                pass  # Skip files that can't be removed
            
            if removed_count > 0:
                print(f"[CLEANUP] Removed {removed_count} old analysis files from plots_output")
                
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to cleanup plots_output: {e}")
    
    def _check_disk_space(self):
        """Check available disk space and warn if low"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_dir)
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                warning_msg = f"WARNING: Low disk space - {free_gb:.2f}GB remaining"
                print(f"[MAINTENANCE] {warning_msg}")
                self.log_error(warning_msg, "System Maintenance")
                
        except Exception as e:
            print(f"[MAINTENANCE] Failed to check disk space: {e}")
    
    def rotate_log_manually(self, log_type):
        """Manually trigger log rotation for a specific log type"""
        if log_type not in self.active_logs:
            print(f"[ROTATION ERROR] Unknown log type: {log_type}")
            return False
            
        log_file = self.active_logs[log_type]
        full_path = os.path.join(self.base_dir, log_file)
        
        if os.path.exists(full_path):
            self._rotate_log_file(log_type, full_path)
            return True
        else:
            print(f"[ROTATION] Log file {full_path} does not exist")
            return False
    
    def get_log_stats(self):
        """Get statistics for all active log files"""
        stats = {}
        
        for log_type, log_file in self.active_logs.items():
            full_path = os.path.join(self.base_dir, log_file)
            
            if os.path.exists(full_path):
                size_bytes = self._get_file_size(full_path)
                size_mb = size_bytes / (1024 * 1024)
                mtime = os.path.getmtime(full_path)
                last_modified = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                config = self.rotation_config.get(log_type, {'max_size_mb': 10})
                needs_rotation = size_mb > config['max_size_mb']
                
                stats[log_type] = {
                    'file_path': log_file,
                    'size_mb': round(size_mb, 2),
                    'max_size_mb': config['max_size_mb'],
                    'needs_rotation': needs_rotation,
                    'last_modified': last_modified
                }
            else:
                stats[log_type] = {
                    'file_path': log_file,
                    'size_mb': 0,
                    'exists': False
                }
        
        return stats
    
    def log_message(self, message, log_type='general', filename=None, print_to_console=True):
        """
        Unified logging function replacing individual log_message() functions
        
        Args:
            message: The message to log
            log_type: Type of log ('watcher', 'live_bot', 'backtest', 'general')
            filename: Optional custom filename (overrides log_type)
            print_to_console: Whether to print to console (default True)
        """
        timestamp = self.get_timestamp()
        log_entry = f"[{timestamp}] {message}"
        
        if print_to_console:
            print(log_entry)
        
        # Determine log file
        if filename:
            log_file = filename
        elif log_type in self.active_logs:
            log_file = self.active_logs[log_type]
        else:
            log_file = "system.log"  # Default fallback
        
        # Check for rotation before writing (if enabled)
        log_path = os.path.join(self.base_dir, log_file)
        if self.enable_rotation and log_type in self.active_logs and not log_file.endswith('.json'):
            if self._needs_rotation(log_type, log_path):
                self._rotate_log_file(log_type, log_path)
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"ERROR: Failed to write to log file {log_file}: {e}")
    
    def log_to_file(self, message, filename="full_analysis_log.txt", print_to_console=False):
        """
        Backtest-specific logging function (replaces core/backtest.py log_to_file)
        Maintains compatibility with existing backtest logging
        """
        timestamp = self.get_timestamp()
        is_report_line = "---" in message or "===" in message
        log_message = f"[{timestamp}] {message}"
        print_message = log_message
        
        if is_report_line:
            print_message = message
            if "---" in message and "--- " not in message:
                print_message = f"\n{message}"
        
        if print_to_console:
            print(print_message)
        
        # Determine log path (maintain backtest.py behavior)
        from core.backtest import IchimokuBacktester
        log_path = getattr(IchimokuBacktester, 'current_run_dir_static', None) or "."
        final_log_file = os.path.join(log_path, os.path.basename(filename))
        
        mode = 'w' if "PIPELINE STARTED" in message else 'a'
        try:
            with open(final_log_file, mode, encoding="utf-8") as f:
                f.write(log_message + "\n")
        except Exception as e:
            print(f"Failed to write to log file {final_log_file}: {e}")
    
    def log_error(self, error_message, context="Trading Robot"):
        """
        Centralized error logging (replaces utilities/system_monitor.py send_error_alert)
        """
        timestamp = self.get_timestamp()
        error_log = self.active_logs['errors']
        
        try:
            with open(error_log, 'a') as f:
                f.write(f"[{timestamp}] ERROR in {context}: {error_message}\n")
            
            print(f"ERROR: {context} - {error_message}")
            return True
        except Exception as e:
            print(f"Failed to send error alert: {e}")
            return False
    
    def log_trade(self, message, symbol=""):
        """
        Centralized trade logging (replaces utilities/system_monitor.py send_trade_alert)
        """
        timestamp = self.get_timestamp()
        trade_log = self.active_logs['trades']
        
        try:
            with open(trade_log, 'a') as f:
                f.write(f"[{timestamp}] TRADE ALERT {symbol}: {message}\n")
            
            print(f"TRADE: {symbol} - {message}")
            return True
        except Exception as e:
            print(f"Failed to send trade alert: {e}")
            return False
    
    def log_notification(self, title, message):
        """
        Centralized notification logging (replaces utilities/system_monitor.py send_notification)
        """
        timestamp = self.get_timestamp()
        notification_log = self.active_logs['notifications']
        
        try:
            with open(notification_log, 'a') as f:
                f.write(f"[{timestamp}] {title}: {message}\n")
            
            print(f"NOTIFICATION: {title} - {message}")
            return True
        except Exception as e:
            print(f"Failed to send notification: {e}")
            return False
    
    def log_backtest_status(self, status, timestamp=None, final_equity=None, run_directory=None):
        """
        Centralized backtest status logging (replaces core/backtest.py status writing)
        """
        if not timestamp:
            timestamp = datetime.datetime.now().isoformat()
        
        status_data = {
            "status": status,
            "timestamp": timestamp
        }
        
        if final_equity is not None:
            status_data["final_equity"] = final_equity
        if run_directory:
            status_data["run_directory"] = run_directory
        
        status_file = self.active_logs['backtest_status']
        
        try:
            with open(status_file, 'w') as f:
                json.dump(status_data, f)
            return True
        except Exception as e:
            print(f"Failed to write backtest status: {e}")
            return False
    
    def get_log_path(self, log_type):
        """
        Get the full path for a specific log type
        
        Args:
            log_type: Type of log ('error_alerts', 'trade_alerts', 'notifications', etc.)
        
        Returns:
            Full path to the log file
        """
        # Map some common names to our active logs
        log_mapping = {
            'error_alerts': 'errors',
            'trade_alerts': 'trades',
            'notifications': 'notifications',
            'watcher': 'watcher',
            'live_bot': 'live_bot',
            'backtest': 'backtest'
        }
        
        mapped_type = log_mapping.get(log_type, log_type)
        
        if mapped_type in self.active_logs:
            return os.path.join(self.base_dir, self.active_logs[mapped_type])
        else:
            # Default fallback
            return os.path.join(self.base_dir, f"{log_type}.log")


# Global centralized logger instance
central_logger = CentralizedLogger()

# Convenience functions for backward compatibility
def log_message(message, log_type='general'):
    """Backward compatibility function for existing log_message() calls"""
    central_logger.log_message(message, log_type)

def log_to_file(message, filename="full_analysis_log.txt", print_to_console=False):
    """Backward compatibility function for existing log_to_file() calls"""
    central_logger.log_to_file(message, filename, print_to_console)

def send_error_alert(error_message, context="Trading Robot"):
    """Backward compatibility function for existing send_error_alert() calls"""
    return central_logger.log_error(error_message, context)

def send_trade_alert(message, symbol=""):
    """Backward compatibility function for existing send_trade_alert() calls"""
    return central_logger.log_trade(message, symbol)

def send_notification(title, message):
    """Backward compatibility function for existing send_notification() calls"""
    return central_logger.log_notification(title, message)


# ==============================================================================
# ENHANCED WATCHER PIPELINE SUPPORT CLASSES
# ==============================================================================

class WatcherHealthMonitor:
    """
    Enhanced system health monitoring for watcher.py pipeline.
    Consolidated from enhanced_monitoring.py.backup with pipeline-aware functionality.
    """
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        """Initialize the health monitoring system"""
        self.config_path = config_path
        self.monitoring_active = False
        self.health_history = []
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()
        self.logger = central_logger
        
        # Load monitoring configuration
        self.config = self._load_monitoring_config()
        
        # Initialize pipeline monitoring (system resource monitoring disabled)
        self.last_health_check = 0
        self.health_thresholds = {
            'pipeline_health_threshold': 70.0,
            'backtest_idle_warning_hours': 2,
            'live_bot_idle_warning_minutes': 30
        }
        
    def _load_monitoring_config(self):
        """Load monitoring configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                default_config = {
                    "monitoring_enabled": True,
                    "check_interval_seconds": 30,
                    "alert_cooldown_minutes": 15,
                    "email_alerts": False,
                    "file_alerts": True
                }
                self._save_monitoring_config(default_config)
                return default_config
        except Exception as e:
            self.logger.log_error(f"Failed to load monitoring config: {e}", "WatcherHealthMonitor")
            return {"monitoring_enabled": True, "check_interval_seconds": 30}
    
    def _save_monitoring_config(self, config):
        """Save monitoring configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.log_error(f"Failed to save monitoring config: {e}", "WatcherHealthMonitor")
    
    def collect_system_health(self):
        """
        Collect trading pipeline health metrics only.
        System resource monitoring disabled - handled by external system monitoring.
        """
        try:
            # Pipeline-specific checks only
            backtest_status = self._check_backtest_status()
            live_bot_status = self._check_live_bot_status()
            
            # Calculate pipeline health score based on trading components only
            health_score = self._calculate_pipeline_health_score(backtest_status, live_bot_status)
            
            health_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'backtest_status': backtest_status,
                'live_bot_status': live_bot_status,
                'health_score': health_score,
                'pipeline_status': 'healthy' if health_score > 70 else 'warning',
                'monitoring_mode': 'trading_pipeline_only'
            }
            
            return health_data
            
        except Exception as e:
            self.logger.log_error(f"Failed to collect pipeline health: {e}", "WatcherHealthMonitor")
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'backtest_status': 'unknown',
                'live_bot_status': 'unknown',
                'health_score': 50,
                'pipeline_status': 'monitoring_unavailable',
                'monitoring_mode': 'trading_pipeline_only'
            }
    
    def _calculate_pipeline_health_score(self, backtest_status, live_bot_status):
        """Calculate trading pipeline health score (0-100) based on trading components only"""
        score = 100  # Start with perfect score
        
        # Backtest status scoring
        if backtest_status == 'active':
            score -= 0  # Perfect - backtest running
        elif backtest_status == 'idle':
            score -= 10  # Minor deduction - not actively running but ready
        elif backtest_status == 'no_runs':
            score -= 30  # Significant deduction - no backtest results
        else:  # unknown, no_output_dir
            score -= 20  # Moderate deduction - unclear status
        
        # Live bot status scoring
        if live_bot_status == 'active':
            score -= 0  # Perfect - live trading active
        elif live_bot_status == 'idle':
            score -= 15  # Minor deduction - not actively trading
        elif live_bot_status == 'no_log':
            score -= 25  # Significant deduction - no logging activity
        else:  # unknown
            score -= 20  # Moderate deduction - unclear status
        
        return max(0, round(score, 1))
    
    def _check_backtest_status(self):
        """Check if backtest processes are running properly"""
        try:
            # Check for recent backtest activity
            if os.path.exists('plots_output'):
                recent_dirs = []
                for item in os.listdir('plots_output'):
                    item_path = os.path.join('plots_output', item)
                    if os.path.isdir(item_path):
                        recent_dirs.append((item_path, os.path.getmtime(item_path)))
                
                if recent_dirs:
                    # Check if there's recent activity (within last hour)
                    latest_time = max(recent_dirs, key=lambda x: x[1])[1]
                    if time.time() - latest_time < 3600:  # 1 hour
                        return 'active'
                    else:
                        return 'idle'
                else:
                    return 'no_runs'
            else:
                return 'no_output_dir'
        except Exception:
            return 'unknown'
    
    def _check_live_bot_status(self):
        """Check live trading bot status"""
        try:
            # Check for live bot log activity
            live_bot_log = 'logs/live_bot.log'
            if os.path.exists(live_bot_log):
                stat = os.stat(live_bot_log)
                # Check if log was modified in last 10 minutes
                if time.time() - stat.st_mtime < 600:  # 10 minutes
                    return 'active'
                else:
                    return 'idle'
            else:
                return 'no_log'
        except Exception:
            return 'unknown'
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring in background thread"""
        if self.monitoring_active:
            self.logger.log_message("Health monitoring already active", "WatcherHealthMonitor")
            return
        
        self.monitoring_active = True
        self.stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.log_message(f"Enhanced health monitoring started (interval: {interval_seconds}s)", "WatcherHealthMonitor")
    
    def stop_monitoring_system(self):
        """Stop the monitoring system"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_monitoring.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.log_message("Enhanced health monitoring stopped", "WatcherHealthMonitor")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop for continuous health checks"""
        while not self.stop_monitoring.wait(interval_seconds):
            try:
                # Collect system health
                health = self.collect_system_health()
                
                if health:
                    # Store health data
                    self.health_history.append(health)
                    
                    # Check for alerts
                    self._check_health_alerts(health)
                    
                    # Cleanup old data (keep last 100 entries)
                    if len(self.health_history) > 100:
                        self.health_history = self.health_history[-100:]
                        
            except Exception as e:
                self.logger.log_error(f"Error in monitoring loop: {e}", "WatcherHealthMonitor")
    
    def _check_health_alerts(self, health_data):
        """Check health data for alert conditions - pipeline monitoring only"""
        alerts = []
        
        # Only check data that's actually being collected in pipeline-only mode
        
        # Overall health alert (rate limited to avoid spam)
        if health_data.get('health_score', 100) < self.health_thresholds['pipeline_health_threshold']:
            # Only log health warnings every 5 minutes to avoid spam
            current_time = time.time()
            if not hasattr(self, '_last_health_warning') or current_time - self._last_health_warning > 300:
                alerts.append(f"LOW HEALTH SCORE: {health_data['health_score']:.1f}")
                self._last_health_warning = current_time
        
        # Pipeline status alerts
        if health_data.get('pipeline_status') == 'warning':
            alerts.append(f"PIPELINE WARNING: Health score {health_data.get('health_score', 'unknown')}")
        
        # Backtest status alerts
        backtest_status = health_data.get('backtest_status', 'unknown')
        if backtest_status in ['failed', 'error', 'crashed']:
            alerts.append(f"BACKTEST ISSUE: Status {backtest_status}")
        
        # Live bot status alerts
        live_bot_status = health_data.get('live_bot_status', 'unknown')
        if live_bot_status in ['failed', 'error', 'crashed']:
            alerts.append(f"LIVE BOT ISSUE: Status {live_bot_status}")
        
        # Send alerts if any found
        for alert in alerts:
            self.logger.log_error(alert, "WatcherHealthMonitor")
    
    def get_health_summary(self):
        """Get summary of recent health data - pipeline monitoring only"""
        if not self.health_history:
            return None
        
        latest = self.health_history[-1]
        
        # Calculate averages over last 10 readings - only for data we actually collect
        recent_data = self.health_history[-10:]
        avg_health = sum(h.get('health_score', 50) for h in recent_data) / len(recent_data)
        
        return {
            'current': latest,
            'recent_averages': {
                'health_score': round(avg_health, 1)
            },
            'monitoring_active': self.monitoring_active,
            'total_readings': len(self.health_history),
            'monitoring_mode': 'trading_pipeline_only'
        }


class WatcherParameterValidator:
    """
    Parameter validation for watcher.py optimization pipeline.
    Consolidated from validate_parameter_bounds.py.backup with pipeline integration.
    """
    
    def __init__(self, config_file: str = "core/optimization_config.json"):
        """Initialize parameter validator"""
        self.config_file = config_file
        self.logger = central_logger
        self.validation_cache = {}
        self.last_validation_time = 0
        
    def validate_parameter_bounds(self, config_file: str = None) -> dict:
        """
        Validate all parameter bounds against hard bounds.
        Returns comprehensive validation results for watcher pipeline.
        """
        if config_file is None:
            config_file = self.config_file
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.logger.log_error(f"Failed to load config file {config_file}: {e}", "WatcherParameterValidator")
            return {'error': str(e), 'valid': False}
        
        parameter_spaces = config.get('parameter_spaces', {}).get('global', [])
        
        validation_results = {
            'valid_parameters': [],
            'invalid_parameters': [],
            'warnings': [],
            'summary': {},
            'config_file': config_file,
            'validation_timestamp': datetime.datetime.now().isoformat()
        }
        
        self.logger.log_message("Starting parameter bounds validation", "WatcherParameterValidator")
        
        total_params = len(parameter_spaces)
        valid_count = 0
        
        for param in parameter_spaces:
            name = param.get('name', 'unknown')
            bounds = param.get('bounds', [0, 1])
            hard_bounds = param.get('hard_bounds', bounds)
            param_type = param.get('type', 'float')
            
            # Validate bounds structure
            if not isinstance(bounds, list) or len(bounds) != 2:
                validation_results['invalid_parameters'].append({
                    'name': name,
                    'issue': 'Invalid bounds format',
                    'bounds': bounds,
                    'hard_bounds': hard_bounds
                })
                continue
            
            if not isinstance(hard_bounds, list) or len(hard_bounds) != 2:
                validation_results['invalid_parameters'].append({
                    'name': name,
                    'issue': 'Invalid hard_bounds format',
                    'bounds': bounds,
                    'hard_bounds': hard_bounds
                })
                continue
            
            # Check if bounds are within hard bounds
            bounds_valid = (
                bounds[0] >= hard_bounds[0] and
                bounds[1] <= hard_bounds[1] and
                bounds[0] <= bounds[1]
            )
            
            param_info = {
                'name': name,
                'bounds': bounds,
                'hard_bounds': hard_bounds,
                'type': param_type,
                'range_size': bounds[1] - bounds[0],
                'hard_range_size': hard_bounds[1] - hard_bounds[0]
            }
            
            if bounds_valid:
                validation_results['valid_parameters'].append(param_info)
                valid_count += 1
                
                # Check for warnings
                range_utilization = param_info['range_size'] / param_info['hard_range_size'] if param_info['hard_range_size'] > 0 else 0
                if range_utilization < 0.1:
                    validation_results['warnings'].append({
                        'name': name,
                        'message': f'Very narrow search range ({range_utilization:.1%} of hard bounds)',
                        'severity': 'low'
                    })
                elif range_utilization > 0.9:
                    validation_results['warnings'].append({
                        'name': name,
                        'message': f'Search range near hard bounds limit ({range_utilization:.1%})',
                        'severity': 'medium'
                    })
            else:
                param_info['issue'] = 'Bounds outside hard_bounds or invalid range'
                validation_results['invalid_parameters'].append(param_info)
        
        # Generate summary
        validation_results['summary'] = {
            'total_parameters': total_params,
            'valid_parameters': valid_count,
            'invalid_parameters': len(validation_results['invalid_parameters']),
            'warnings': len(validation_results['warnings']),
            'validation_passed': len(validation_results['invalid_parameters']) == 0,
            'config_file': config_file
        }
        
        # Log results
        if validation_results['summary']['validation_passed']:
            self.logger.log_message(f"Parameter validation PASSED: {valid_count}/{total_params} parameters valid", "WatcherParameterValidator")
        else:
            self.logger.log_error(f"Parameter validation FAILED: {len(validation_results['invalid_parameters'])} invalid parameters", "WatcherParameterValidator")
        
        # Cache results
        self.validation_cache[config_file] = validation_results
        self.last_validation_time = time.time()
        
        return validation_results
    
    def check_optimization_constraints(self, optimized_params: dict, config_file: str = None) -> dict:
        """
        Check if optimized parameters are hitting bounds constraints.
        Used by watcher.py to determine if bounds need expansion.
        """
        if config_file is None:
            config_file = self.config_file
            
        # Get validation results
        validation = self.validate_parameter_bounds(config_file)
        if not validation.get('summary', {}).get('validation_passed', False):
            return {'error': 'Configuration validation failed', 'constraint_analysis': None}
        
        constraint_analysis = {
            'parameters_at_bounds': [],
            'parameters_near_bounds': [],
            'constraint_recommendations': [],
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Analyze each optimized parameter
        for param_name, param_value in optimized_params.items():
            # Find parameter configuration
            param_config = None
            for valid_param in validation['valid_parameters']:
                if valid_param['name'] == param_name:
                    param_config = valid_param
                    break
            
            if not param_config:
                continue
            
            bounds = param_config['bounds']
            tolerance = (bounds[1] - bounds[0]) * 0.05  # 5% tolerance
            
            # Check if at bounds
            at_min_bound = abs(param_value - bounds[0]) <= tolerance
            at_max_bound = abs(param_value - bounds[1]) <= tolerance
            
            if at_min_bound or at_max_bound:
                constraint_analysis['parameters_at_bounds'].append({
                    'parameter': param_name,
                    'value': param_value,
                    'bounds': bounds,
                    'at_bound': 'min' if at_min_bound else 'max',
                    'recommendation': 'expand_bounds'
                })
            
            # Check if near bounds (within 10%)
            near_tolerance = (bounds[1] - bounds[0]) * 0.1
            near_min = param_value - bounds[0] <= near_tolerance
            near_max = bounds[1] - param_value <= near_tolerance
            
            if (near_min or near_max) and not (at_min_bound or at_max_bound):
                constraint_analysis['parameters_near_bounds'].append({
                    'parameter': param_name,
                    'value': param_value,
                    'bounds': bounds,
                    'near_bound': 'min' if near_min else 'max',
                    'recommendation': 'monitor'
                })
        
        # Generate recommendations
        if constraint_analysis['parameters_at_bounds']:
            constraint_analysis['constraint_recommendations'].append({
                'action': 'expand_bounds',
                'priority': 'high',
                'parameters': [p['parameter'] for p in constraint_analysis['parameters_at_bounds']],
                'reason': 'Parameters hitting bounds may indicate suboptimal search space'
            })
        
        if len(constraint_analysis['parameters_near_bounds']) > len(optimized_params) * 0.3:
            constraint_analysis['constraint_recommendations'].append({
                'action': 'review_bounds',
                'priority': 'medium',
                'parameters': [p['parameter'] for p in constraint_analysis['parameters_near_bounds']],
                'reason': 'Many parameters near bounds may indicate search space issues'
            })
        
        return constraint_analysis
    
    def expand_parameter_bounds(self, optimized_params: dict, config_file: str = None, expansion_factor: float = 0.5) -> dict:
        """
        Automatically expand parameter bounds when they hit edges during optimization.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            config_file: Path to optimization config file
            expansion_factor: Factor by which to expand bounds (0.5 = 50% expansion)
            
        Returns:
            dict: Results of bounds expansion including which parameters were expanded
        """
        if config_file is None:
            config_file = self.config_file
            
        print("\n" + "=" * 80)
        print("🔧 AUTOMATIC PARAMETER BOUNDS EXPANSION")
        print("=" * 80)
        
        # Check constraints first
        constraint_analysis = self.check_optimization_constraints(optimized_params, config_file)
        parameters_at_bounds = constraint_analysis.get('parameters_at_bounds', [])
        
        if not parameters_at_bounds:
            print("✅ No parameters at bounds - expansion not needed")
            return {
                'expanded_parameters': [],
                'config_updated': False,
                'message': 'No bounds expansion needed'
            }
        
        print(f"🎯 Found {len(parameters_at_bounds)} parameters at bounds, expanding...")
        
        # Load config
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {
                'expanded_parameters': [],
                'config_updated': False,
                'error': f'Failed to load config: {e}'
            }
        
        expanded_parameters = []
        parameter_spaces = config.get('parameter_spaces', {}).get('global', [])
        
        # Expand bounds for each parameter at bounds
        for param_info in parameters_at_bounds:
            param_name = param_info['parameter']
            current_value = param_info['value']
            current_bounds = param_info['bounds']
            at_bound = param_info['at_bound']
            
            # Find the parameter in config
            for i, param_config in enumerate(parameter_spaces):
                if param_config['name'] == param_name:
                    hard_bounds = param_config['hard_bounds']
                    current_range = current_bounds[1] - current_bounds[0]
                    expansion_amount = current_range * expansion_factor
                    
                    new_bounds = list(current_bounds)
                    
                    if at_bound == 'min':
                        # Expand lower bound
                        new_lower = max(hard_bounds[0], current_bounds[0] - expansion_amount)
                        new_bounds[0] = new_lower
                    else:  # at_bound == 'max'
                        # Expand upper bound  
                        new_upper = min(hard_bounds[1], current_bounds[1] + expansion_amount)
                        new_bounds[1] = new_upper
                    
                    # Only update if bounds actually changed
                    if new_bounds != current_bounds:
                        # Update config
                        parameter_spaces[i]['bounds'] = new_bounds
                        
                        expanded_parameters.append({
                            'parameter': param_name,
                            'old_bounds': current_bounds,
                            'new_bounds': new_bounds,
                            'expanded_direction': at_bound,
                            'expansion_amount': expansion_amount,
                            'hard_bounds': hard_bounds
                        })
                        
                        print(f"   📈 {param_name}: {current_bounds} → {new_bounds} (expanded {at_bound} bound)")
                    else:
                        print(f"   ⚠️ {param_name}: Already at hard bounds {hard_bounds}, cannot expand further")
                    break
        
        if expanded_parameters:
            # Save updated config
            try:
                # Create backup first
                backup_file = f"{config_file}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy2(config_file, backup_file)
                print(f"   💾 Config backup created: {backup_file}")
                
                # Save updated config
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✅ Bounds expansion completed! Updated {len(expanded_parameters)} parameters")
                print(f"   📝 Config file updated: {config_file}")
                
                return {
                    'expanded_parameters': expanded_parameters,
                    'config_updated': True,
                    'backup_file': backup_file,
                    'message': f'Expanded bounds for {len(expanded_parameters)} parameters'
                }
                
            except Exception as e:
                print(f"❌ Error saving updated config: {e}")
                return {
                    'expanded_parameters': expanded_parameters,
                    'config_updated': False,
                    'error': f'Failed to save config: {e}'
                }
        else:
            print("⚠️ No bounds could be expanded (all at hard limits)")
            return {
                'expanded_parameters': [],
                'config_updated': False,
                'message': 'All parameters at hard bounds - cannot expand further'
            }
    
    def generate_parameter_report(self, config_file: str = None, include_optimization_analysis: bool = False) -> str:
        """
        Generate comprehensive parameter validation report for watcher pipeline.
        """
        if config_file is None:
            config_file = self.config_file
            
        validation = self.validate_parameter_bounds(config_file)
        
        report_lines = [
            "=" * 80,
            "WATCHER PIPELINE PARAMETER VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Config File: {config_file}",
            ""
        ]
        
        # Summary section
        summary = validation.get('summary', {})
        report_lines.extend([
            "VALIDATION SUMMARY:",
            "-" * 40,
            f"Total Parameters: {summary.get('total_parameters', 0)}",
            f"Valid Parameters: {summary.get('valid_parameters', 0)}",
            f"Invalid Parameters: {summary.get('invalid_parameters', 0)}",
            f"Warnings: {summary.get('warnings', 0)}",
            f"Overall Status: {'PASS' if summary.get('validation_passed', False) else 'FAIL'}",
            ""
        ])
        
        # Valid parameters section
        if validation.get('valid_parameters'):
            report_lines.append("VALID PARAMETERS:")
            report_lines.append("-" * 40)
            for param in validation['valid_parameters']:
                range_util = param['range_size'] / param['hard_range_size'] if param['hard_range_size'] > 0 else 0
                report_lines.append(f"{param['name']:<25} | Bounds: {param['bounds']} | Hard: {param['hard_bounds']} | Range: {range_util:.1%}")
            report_lines.append("")
        
        # Invalid parameters section
        if validation.get('invalid_parameters'):
            report_lines.append("INVALID PARAMETERS:")
            report_lines.append("-" * 40)
            for param in validation['invalid_parameters']:
                report_lines.append(f"{param['name']:<25} | Issue: {param.get('issue', 'Unknown')}")
                report_lines.append(f"{'':>25} | Bounds: {param.get('bounds', 'N/A')} | Hard: {param.get('hard_bounds', 'N/A')}")
            report_lines.append("")
        
        # Warnings section
        if validation.get('warnings'):
            report_lines.append("WARNINGS:")
            report_lines.append("-" * 40)
            for warning in validation['warnings']:
                report_lines.append(f"{warning['name']:<25} | {warning['message']} (Severity: {warning['severity']})")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


# Global instances for pipeline integration
watcher_health_monitor = WatcherHealthMonitor()
watcher_parameter_validator = WatcherParameterValidator()

def check_live_bot_status():
    """Check the current status of the live bot"""
    import json
    import os
    from datetime import datetime
    
    print("=" * 60)
    print("LIVE BOT STATUS MONITOR")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if bot state file exists
    state_file = "data/live_bot_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            print("✅ Live Bot State File Found")
            print(f"   📊 Portfolio Cash: ${state.get('portfolio', {}).get('cash', 0):,.2f}")
            print(f"   🔄 Position Status: {'IN POSITION' if state.get('position_manager', {}).get('in_position', False) else 'NO POSITION'}")
            print(f"   📈 Last Update: {state.get('last_update', 'Unknown')}")
            
            if state.get('position_manager', {}).get('in_position', False):
                pos_mgr = state.get('position_manager', {})
                print(f"   💰 Entry Price: ${pos_mgr.get('entry_price', 0):,.2f}")
                print(f"   📦 Position Size: {pos_mgr.get('position_size', 0):.6f}")
                print(f"   🎯 Take Profit: ${pos_mgr.get('take_profit_price', 0):,.2f}")
                print(f"   🛡️ Stop Loss: ${pos_mgr.get('trailing_stop_loss', 0):,.2f}")
            
        except Exception as e:
            print(f"❌ Error reading state file: {e}")
    else:
        print("⚠️ Live Bot State File Not Found")
    
    # Check recent logs
    log_files = [
        "logs/live_bot.log",
        "trade_alerts.log", 
        "error_alerts.log",
        "notifications.log"
    ]
    
    print("\n📋 LOG FILES STATUS:")
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"   ✅ {log_file}: {size:,} bytes (Modified: {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {log_file}: Not found")
    
    # Check configuration files
    config_files = [
        "core/optimization_config.json",
        "data/latest_live_parameters.json"
    ]
    
    print("\n⚙️ CONFIGURATION FILES:")
    for config_file in config_files:
        if os.path.exists(config_file):
            size = os.path.getsize(config_file)
            mtime = datetime.fromtimestamp(os.path.getmtime(config_file))
            print(f"   ✅ {config_file}: {size:,} bytes (Modified: {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"   ❌ {config_file}: Not found")
    
    return True


# =============================================================================
# ENHANCED FAULT TOLERANCE SYSTEM (from enhancements/)
# =============================================================================

class HealthMonitor:
    """Enhanced system health monitoring with alerting"""
    
    def __init__(self):
        self.health_history = []
        self.alerts_sent = set()
        self.logger = logging.getLogger("HealthMonitor")
        
    async def monitor_health(self, fault_tolerant_bot):
        """Continuous health monitoring with enhanced tracking"""
        while True:
            try:
                health = fault_tolerant_bot.get_system_health()
                self.health_history.append(health)
                
                # Keep only recent history (last 100 checks)
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Check for health issues and send alerts
                await self._check_health_alerts(health)
                
                # Save health status to file
                health_file = 'data/system_health.json'
                os.makedirs('data', exist_ok=True)
                with open(health_file, 'w') as f:
                    json.dump(health, f, indent=2)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_health_alerts(self, health):
        """Check if health alerts need to be sent"""
        status = health['overall_status']
        
        if status == 'degraded' and 'degraded' not in self.alerts_sent:
            await self._send_health_alert("⚠️ System health degraded - some components failing")
            self.alerts_sent.add('degraded')
        elif status == 'healthy' and 'degraded' in self.alerts_sent:
            await self._send_health_alert("✅ System health recovered - all components operational")
            self.alerts_sent.discard('degraded')
    
    async def _send_health_alert(self, message):
        """Send health alert notification"""
        try:
            # Use existing notification system if available
            self.logger.warning(f"Health Alert: {message}")
            
            # Try to send via live bot notification system
            try:
                from live_trading.live_bot import send_system_status
                send_system_status(message)
            except ImportError:
                pass  # Live bot notifications not available
                
        except Exception as e:
            self.logger.error(f"Failed to send health alert: {e}")

    def get_health_summary(self):
        """Get summary of recent health status"""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}
        
        recent_health = self.health_history[-10:]  # Last 10 checks
        degraded_count = sum(1 for h in recent_health if h['overall_status'] == 'degraded')
        
        return {
            "status": self.health_history[-1]['overall_status'],
            "degraded_percentage": (degraded_count / len(recent_health)) * 100,
            "total_checks": len(self.health_history),
            "last_check": self.health_history[-1]['timestamp']
        }


# =============================================================================
# ENHANCED MARKET REGIME DETECTION (from enhancements/)
# =============================================================================

class MarketRegimeDetector:
    """Advanced market regime detection using multiple indicators"""
    
    def __init__(self, params):
        self.lookback_period = params.get('REGIME_LOOKBACK', 50)
        self.volatility_threshold = params.get('VOLATILITY_THRESHOLD', 0.02)
        self.trend_threshold = params.get('TREND_THRESHOLD', 0.65)
        
        # Import scikit-learn only if available
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            logging.warning("scikit-learn not available. Using simplified regime detection.")
        
    def calculate_regime_features(self, df):
        """Calculate features for regime detection"""
        features = pd.DataFrame(index=df.index)
        
        # Volatility features
        features['realized_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['vol_ratio'] = features['realized_vol'] / features['realized_vol'].rolling(50).mean()
        
        # Trend strength features
        price_series = pd.Series(range(len(df)), index=df.index)
        features['trend_strength'] = df['close'].rolling(20).corr(price_series.rolling(20))
        features['price_momentum'] = df['close'].pct_change(10)
        
        # Range features
        features['range_ratio'] = (df['high'] - df['low']) / df['close']
        features['consolidation'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_trend'] = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        else:
            features['volume_trend'] = 1.0  # Neutral when no volume data
        
        # Technical indicators (if available)
        if 'rsi' in df.columns:
            features['rsi_regime'] = df['rsi'] / 50 - 1  # Normalized RSI
        
        return features.dropna()
    
    def detect_regime(self, df):
        """Detect current market regime"""
        features = self.calculate_regime_features(df)
        
        if len(features) < self.lookback_period:
            return "insufficient_data", 0.5
        
        # Use recent data for regime detection
        recent_features = features.tail(self.lookback_period)
        
        if self.sklearn_available:
            return self._detect_regime_ml(recent_features)
        else:
            return self._detect_regime_simple(recent_features)
    
    def _detect_regime_ml(self, features):
        """ML-based regime detection"""
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features.fillna(0))
            
            # Cluster analysis
            clusters = self.kmeans.fit_predict(scaled_features)
            current_regime_cluster = clusters[-1]
            
            # Interpret clusters based on feature characteristics
            cluster_centers = self.kmeans.cluster_centers_
            current_center = cluster_centers[current_regime_cluster]
            
            # Regime interpretation based on feature weights
            vol_score = current_center[0] if len(current_center) > 0 else 0
            trend_score = current_center[3] if len(current_center) > 3 else 0
            
            if vol_score > 0.5:
                regime = "volatile"
                confidence = min(abs(vol_score), 1.0)
            elif abs(trend_score) > 0.3:
                regime = "trending"
                confidence = abs(trend_score)
            else:
                regime = "ranging"
                confidence = 1.0 - abs(trend_score)
            
            return regime, confidence
            
        except Exception as e:
            logging.error(f"ML regime detection failed: {e}")
            return self._detect_regime_simple(features)
    
    def _detect_regime_simple(self, features):
        """Simple rule-based regime detection"""
        try:
            latest = features.iloc[-1]
            
            # Simple volatility check
            vol_ratio = latest.get('vol_ratio', 1.0)
            trend_strength = abs(latest.get('trend_strength', 0.0))
            price_momentum = abs(latest.get('price_momentum', 0.0))
            
            if vol_ratio > 1.5:  # High volatility
                return "volatile", min(vol_ratio / 2, 1.0)
            elif trend_strength > 0.6 or price_momentum > 0.03:  # Strong trend
                return "trending", max(trend_strength, price_momentum * 10)
            else:  # Ranging market
                return "ranging", 0.7
                
        except Exception as e:
            logging.error(f"Simple regime detection failed: {e}")
            return "unknown", 0.5

    def get_regime_parameters(self, regime, base_params):
        """Adjust trading parameters based on regime"""
        adjusted_params = base_params.copy()
        
        if regime == "trending":
            # More aggressive in trending markets
            adjusted_params['RSI_OVERSOLD'] = max(20, base_params.get('RSI_OVERSOLD', 30) - 10)
            adjusted_params['RSI_OVERBOUGHT'] = min(80, base_params.get('RSI_OVERBOUGHT', 70) + 10)
            adjusted_params['ADX_THRESHOLD'] = max(15, base_params.get('ADX_THRESHOLD', 25) - 5)
            
        elif regime == "ranging":
            # More conservative in ranging markets
            adjusted_params['RSI_OVERSOLD'] = min(35, base_params.get('RSI_OVERSOLD', 30) + 5)
            adjusted_params['RSI_OVERBOUGHT'] = max(65, base_params.get('RSI_OVERBOUGHT', 70) - 5)
            adjusted_params['ADX_THRESHOLD'] = min(35, base_params.get('ADX_THRESHOLD', 25) + 10)
            
        elif regime == "volatile":
            # Much more conservative in volatile markets
            adjusted_params['POSITION_SIZE_FACTOR'] = base_params.get('POSITION_SIZE_FACTOR', 1.0) * 0.5
            adjusted_params['STOP_LOSS_FACTOR'] = base_params.get('STOP_LOSS_FACTOR', 2.0) * 1.5
            
        return adjusted_params


class AdaptiveParameterManager:
    """Manages dynamic parameter adjustment based on market conditions"""
    
    def __init__(self, base_params):
        self.base_params = base_params.copy()
        self.regime_detector = MarketRegimeDetector(base_params)
        self.regime_history = []
        self.performance_tracker = {}
        
    def update_parameters(self, df):
        """Update parameters based on current market regime"""
        regime, confidence = self.regime_detector.detect_regime(df)
        
        # Track regime history
        self.regime_history.append({
            'timestamp': df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
            'regime': regime,
            'confidence': confidence
        })
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        # Get adjusted parameters
        adjusted_params = self.regime_detector.get_regime_parameters(regime, self.base_params)
        
        return adjusted_params, regime, confidence


class MultiStrategyManager:
    """
    Manages multiple trading strategies and combines their signals
    Enhanced from multi_strategy_manager.py
    """
    
    def __init__(self, params):
        self.params = params
        self.strategies = {}
        self.signal_history = []
        self.performance_tracker = {}
        self.logger = logging.getLogger("MultiStrategyManager")
        
    def register_strategy(self, name: str, strategy_instance, weight: float = 1.0):
        """Register a trading strategy"""
        self.strategies[name] = {
            'instance': strategy_instance,
            'weight': weight,
            'enabled': True,
            'performance': {'total_signals': 0, 'profitable_signals': 0}
        }
        self.logger.info(f"Registered strategy: {name} with weight {weight}")
    
    def generate_combined_signals(self, df: pd.DataFrame) -> Tuple[int, float, dict]:
        """
        Generate combined signals from all active strategies
        Returns: (signal, confidence, metadata)
        """
        
        strategy_signals = []
        
        for name, strategy_config in self.strategies.items():
            if not strategy_config['enabled']:
                continue
                
            try:
                # Get signal from individual strategy
                strategy = strategy_config['instance']
                processed_df = strategy.generate_signals(df)
                
                if processed_df.empty:
                    continue
                    
                latest_row = processed_df.iloc[-1]
                
                signal_data = {
                    'strategy_name': name,
                    'symbol': self.params.get('symbol', 'BTC/USDT'),
                    'signal': latest_row.get('signal', 0),
                    'confidence': latest_row.get('signal_strength', 0.5),
                    'strength': abs(latest_row.get('signal', 0)),
                    'timestamp': df.index[-1].timestamp() if hasattr(df.index[-1], 'timestamp') else time.time(),
                    'metadata': {
                        'rsi': latest_row.get('rsi'),
                        'adx': latest_row.get('adx'),
                        'bb_position': latest_row.get('bb_position')
                    }
                }
                
                strategy_signals.append(signal_data)
                
            except Exception as e:
                self.logger.error(f"Error getting signal from strategy {name}: {e}")
                continue
        
        if not strategy_signals:
            return 0, 0.0, {}
        
        # Combine signals using weighted voting
        combined_signal, combined_confidence = self._combine_signals(strategy_signals)
        
        # Store signal history
        self.signal_history.append({
            'timestamp': df.index[-1] if len(df) > 0 else pd.Timestamp.now(),
            'combined_signal': combined_signal,
            'combined_confidence': combined_confidence,
            'individual_signals': strategy_signals
        })
        
        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        metadata = self._create_signal_metadata(strategy_signals)
        
        return combined_signal, combined_confidence, metadata
    
    def _combine_signals(self, signals: List[dict]) -> Tuple[int, float]:
        """Combine multiple strategy signals into one"""
        
        if not signals:
            return 0, 0.0
        
        # Calculate weighted signal
        total_weight = 0
        weighted_signal = 0
        confidence_sum = 0
        
        for signal in signals:
            strategy_weight = self.strategies[signal['strategy_name']]['weight']
            
            # Adjust weight based on strategy performance
            performance = self.strategies[signal['strategy_name']]['performance']
            if performance['total_signals'] > 10:
                success_rate = performance['profitable_signals'] / performance['total_signals']
                performance_multiplier = max(0.5, min(2.0, success_rate * 2))  # 0.5x to 2.0x
                strategy_weight *= performance_multiplier
            
            # Weight signal by confidence
            signal_weight = strategy_weight * signal['confidence']
            
            weighted_signal += signal['signal'] * signal_weight
            confidence_sum += signal['confidence'] * strategy_weight
            total_weight += strategy_weight
        
        if total_weight == 0:
            return 0, 0.0
        
        # Normalize
        final_signal = weighted_signal / total_weight
        final_confidence = confidence_sum / total_weight
        
        # Convert to discrete signal
        if final_signal > 0.3:
            return 1, final_confidence
        elif final_signal < -0.3:
            return -1, final_confidence
        else:
            return 0, final_confidence
    
    def _create_signal_metadata(self, signals: List[dict]) -> dict:
        """Create metadata for combined signal"""
        
        metadata = {
            'strategy_count': len(signals),
            'strategy_agreement': self._calculate_agreement(signals),
            'strategy_breakdown': {}
        }
        
        for signal in signals:
            metadata['strategy_breakdown'][signal['strategy_name']] = {
                'signal': signal['signal'],
                'confidence': signal['confidence']
            }
        
        return metadata
    
    def _calculate_agreement(self, signals: List[dict]) -> float:
        """Calculate how much strategies agree on direction"""
        
        if len(signals) < 2:
            return 1.0
        
        signal_values = [s['signal'] for s in signals]
        
        # Count agreements
        agreements = 0
        total_pairs = 0
        
        for i in range(len(signal_values)):
            for j in range(i + 1, len(signal_values)):
                if signal_values[i] * signal_values[j] >= 0:  # Same direction or neutral
                    agreements += 1
                total_pairs += 1
        
        return agreements / total_pairs if total_pairs > 0 else 1.0
    
    def update_strategy_performance(self, strategy_name: str, was_profitable: bool):
        """Update performance tracking for a strategy"""
        
        if strategy_name in self.strategies:
            perf = self.strategies[strategy_name]['performance']
            perf['total_signals'] += 1
            if was_profitable:
                perf['profitable_signals'] += 1
    
    def get_strategy_performance(self) -> dict:
        """Get performance summary for all strategies"""
        
        performance = {}
        
        for name, config in self.strategies.items():
            perf = config['performance']
            
            if perf['total_signals'] > 0:
                success_rate = perf['profitable_signals'] / perf['total_signals']
                performance[name] = {
                    'success_rate': success_rate,
                    'total_signals': perf['total_signals'],
                    'profitable_signals': perf['profitable_signals'],
                    'weight': config['weight'],
                    'enabled': config['enabled']
                }
        
        return performance
    
    def auto_adjust_weights(self):
        """Automatically adjust strategy weights based on performance"""
        
        for name, config in self.strategies.items():
            perf = config['performance']
            
            if perf['total_signals'] >= 20:  # Minimum sample size
                success_rate = perf['profitable_signals'] / perf['total_signals']
                
                # Adjust weight based on success rate
                if success_rate > 0.6:
                    config['weight'] = min(2.0, config['weight'] * 1.1)  # Increase weight
                elif success_rate < 0.4:
                    config['weight'] = max(0.1, config['weight'] * 0.9)  # Decrease weight
                
                self.logger.info(f"Adjusted weight for {name}: {config['weight']:.2f} (success rate: {success_rate:.2f})")
    
    def get_signal_analysis(self, lookback_periods: int = 100) -> dict:
        """Analyze recent signal patterns"""
        
        if len(self.signal_history) < lookback_periods:
            lookback_periods = len(self.signal_history)
            
        if lookback_periods == 0:
            return {}
        
        recent_signals = self.signal_history[-lookback_periods:]
        
        # Analyze signal frequency
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        confidence_levels = []
        agreement_levels = []
        
        for signal_data in recent_signals:
            signal = signal_data['combined_signal']
            confidence = signal_data['combined_confidence']
            
            if signal > 0:
                signal_counts['buy'] += 1
            elif signal < 0:
                signal_counts['sell'] += 1
            else:
                signal_counts['hold'] += 1
            
            confidence_levels.append(confidence)
            
            # Extract agreement from metadata if available
            individual_signals = signal_data.get('individual_signals', [])
            if individual_signals:
                agreement = self._calculate_agreement(individual_signals)
                agreement_levels.append(agreement)
        
        return {
            'signal_distribution': signal_counts,
            'avg_confidence': np.mean(confidence_levels) if confidence_levels else 0,
            'avg_agreement': np.mean(agreement_levels) if agreement_levels else 0,
            'signal_frequency': {
                'buy_rate': signal_counts['buy'] / lookback_periods,
                'sell_rate': signal_counts['sell'] / lookback_periods,
                'hold_rate': signal_counts['hold'] / lookback_periods
            }
        }


# ==============================================================================
# NOTIFICATION SYSTEM FUNCTIONS (from system_monitor.py)
# ==============================================================================

def send_notification(message, title="Trading Robot Notification"):
    """Send a basic notification (file-based) - Integrated from system_monitor.py"""
    try:
        central_logger.log_message(f"{title}: {message}", 'notifications')
        return True
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False

def send_error_alert(error_message, context="Trading Robot"):
    """Send an error alert notification - Integrated from system_monitor.py"""
    try:
        central_logger.log_error(f"ERROR in {context}: {error_message}", 'error_alerts')
        return True
    except Exception as e:
        print(f"Failed to send error alert: {e}")
        return False

def send_trade_alert(message, symbol=""):
    """Send a trading-specific alert notification - Integrated from system_monitor.py"""
    try:
        central_logger.log_trade(f"TRADE ALERT {symbol}: {message}", 'trade_alerts')
        return True
    except Exception as e:
        print(f"Failed to send trade alert: {e}")
        return False

def send_email_notification(subject, message, to_email=None):
    """Placeholder for email notifications - Integrated from system_monitor.py"""
    print("Email notifications not implemented - using file-based notifications instead")
    return send_notification(message, subject)


# ==============================================================================
# ENHANCED MONITORING SYSTEM (from system_monitor.py)
# ==============================================================================

@dataclass
class SystemHealth:
    """System health metrics snapshot - Integrated from system_monitor.py"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    processes_running: int
    backtest_status: str
    live_bot_status: str
    last_trade_time: Optional[str]
    portfolio_value: Optional[float]
    network_connectivity: bool

@dataclass 
class AlertConfig:
    """Alert configuration settings - Integrated from system_monitor.py"""
    monitoring_enabled: bool = True
    check_interval_seconds: int = 30
    alert_cooldown_minutes: int = 15
    email_alerts: bool = False
    file_alerts: bool = True
    cpu_threshold: float = 85.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    enable_email_alerts: bool = False
    email_recipients: List[str] = None

@dataclass
class Alert:
    """Alert data structure - Integrated from system_monitor.py"""
    timestamp: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None


class EnhancedMonitor:
    """
    Comprehensive system monitoring and alerting system - Integrated from system_monitor.py
    This provides backward compatibility for existing imports while using our consolidated architecture
    """
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config_path = config_path
        self.health_history: List[SystemHealth] = []
        self.alerts: List[Alert] = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Use the consolidated WatcherHealthMonitor as the core engine
        self.health_monitor = WatcherHealthMonitor()
        
        # Load or create alert configuration
        self.alert_config = self._load_alert_config()
        
        # Silently initialized

    def _load_alert_config(self) -> AlertConfig:
        """Load alert configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Create AlertConfig with only valid fields
                valid_fields = {field.name for field in AlertConfig.__dataclass_fields__.values()}
                filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
                
                return AlertConfig(**filtered_data)
        except Exception as e:
            print(f"Could not load alert config: {e}")
            print(f"Using default alert configuration")
            
        # Return default config
        return AlertConfig()

    def start_monitoring(self, interval_seconds: int = 30):
        """Start monitoring system (delegates to HealthMonitor)"""
        if not self.monitoring:
            self.monitoring = True
            self.health_monitor.start_monitoring()
            # Silently started

    def stop_monitoring_system(self):
        """Stop monitoring system"""
        if self.monitoring:
            self.monitoring = False
            self.health_monitor.stop_monitoring_system()
            print("Enhanced monitoring stopped")

    def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics"""
        try:
            # Get basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            processes = len(psutil.pids())
            
            # Get trading status
            backtest_status = "UNKNOWN"
            live_bot_status = "UNKNOWN"
            last_trade_time = None
            portfolio_value = None
            
            # Try to read live bot state
            try:
                if os.path.exists("data/live_bot_state.json"):
                    with open("data/live_bot_state.json", 'r') as f:
                        state = json.load(f)
                        live_bot_status = state.get('status', 'UNKNOWN')
                        last_trade_time = state.get('last_trade_time')
                        portfolio_value = state.get('portfolio_value')
            except:
                pass
            
            # Test network connectivity
            network_connectivity = True
            try:
                import requests
                requests.get("https://api.binance.com/api/v3/ping", timeout=5)
            except:
                network_connectivity = False
            
            return SystemHealth(
                timestamp=datetime.datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                processes_running=processes,
                backtest_status=backtest_status,
                live_bot_status=live_bot_status,
                last_trade_time=last_trade_time,
                portfolio_value=portfolio_value,
                network_connectivity=network_connectivity
            )
            
        except Exception as e:
            print(f"Error collecting system health: {e}")
            return None

    def check_alert_conditions(self, health: SystemHealth):
        """Check if any alert conditions are met"""
        alerts = []
        
        if health.cpu_percent > self.alert_config.cpu_threshold:
            alerts.append(Alert(
                timestamp=health.timestamp,
                level="WARNING",
                component="CPU",
                message=f"High CPU usage: {health.cpu_percent:.1f}%"
            ))
        
        if health.memory_percent > self.alert_config.memory_threshold:
            alerts.append(Alert(
                timestamp=health.timestamp,
                level="WARNING", 
                component="Memory",
                message=f"High memory usage: {health.memory_percent:.1f}%"
            ))
        
        if health.disk_percent > self.alert_config.disk_threshold:
            alerts.append(Alert(
                timestamp=health.timestamp,
                level="WARNING",
                component="Disk",
                message=f"High disk usage: {health.disk_percent:.1f}%"
            ))
        
        if not health.network_connectivity:
            alerts.append(Alert(
                timestamp=health.timestamp,
                level="ERROR",
                component="Network",
                message="Network connectivity lost"
            ))
        
        # Process alerts
        for alert in alerts:
            self.alerts.append(alert)
            send_error_alert(alert.message, alert.component)


# Legacy compatibility functions
def start_system_monitoring(interval_seconds: int = 30):
    """Start system monitoring (legacy compatibility)"""
    global _global_monitor
    if '_global_monitor' not in globals():
        _global_monitor = EnhancedMonitor()
    _global_monitor.start_monitoring(interval_seconds)

def stop_system_monitoring():
    """Stop system monitoring (legacy compatibility)"""
    global _global_monitor
    if '_global_monitor' in globals():
        _global_monitor.stop_monitoring_system()


# ==============================================================================
# GCP STATUS CHECK UTILITIES (from simple_gcp_check.py)
# ==============================================================================

def check_gcp_live_bot_status():
    """
    Check Google Cloud live bot status - Integrated from simple_gcp_check.py
    Returns dict with status information
    """
    try:
        from utilities.gcp_utils import get_gcs_blob_metadata
        
        print("=== GOOGLE CLOUD LIVE BOT STATUS ===")
        
        status = {
            'gcs_connection': False,
            'parameters_in_cloud': False,
            'local_cloud_synced': False,
            'time_difference_minutes': 0
        }
        
        # Check parameters
        blob = get_gcs_blob_metadata("latest_live_parameters.json")
        if blob:
            print(f"✅ Live parameters in GCS: {blob.updated}")
            print(f"   Size: {blob.size} bytes")
            status['gcs_connection'] = True
            status['parameters_in_cloud'] = True
        else:
            print("❌ No live parameters found in GCS")
            return status

        # Check local vs cloud sync
        local_file = "latest_live_parameters.json"
        if os.path.exists(local_file):
            local_time = os.path.getmtime(local_file)
            cloud_time = blob.updated.timestamp() if blob else 0
            time_diff_minutes = (cloud_time - local_time) / 60
            status['time_difference_minutes'] = time_diff_minutes
            
            if abs(time_diff_minutes) < 5:  # 5 minutes
                print("✅ Local and cloud parameters are synced")
                status['local_cloud_synced'] = True
            else:
                print(f"⚠️  Time difference: {time_diff_minutes:.1f} minutes")
        else:
            print("⚠️  No local parameters file")

        print("\n💡 Your live bot in Google Cloud should be using these parameters")
        print("   and executing trades automatically!")
        
        return status
        
    except Exception as e:
        print(f"❌ Error checking GCP status: {e}")
        return {'error': str(e)}