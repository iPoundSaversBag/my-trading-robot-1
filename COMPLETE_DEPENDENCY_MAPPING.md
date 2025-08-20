# 🔍 COMPREHENSIVE DEPENDENCY MAPPING ANALYSIS

## 📊 **COMPLETE FILE-BY-FILE DEPENDENCY MAPPING**

Based on systematic analysis of all major files in the watcher pipeline, here's the comprehensive dependency mapping:

---

## 🤖 **1. WATCHER.PY (1,298 lines) - MASTER ORCHESTRATOR**

### **Direct Imports:**
```python
# Standard Libraries
import os, sys, traceback, subprocess, time, json, json5, re, argparse
from datetime import datetime

# Core Dependencies
from utilities.utils import send_notification, send_error_alert, log_message, download_data, check_data, analyze_limits
from health_utils import run_health_check, get_repair_engine, run_comprehensive_health_check, display_health_problems_with_descriptions, show_robot_problems_with_descriptions

# Dynamic Imports (conditional)
from unified_live_monitor import start_monitoring (line 957)
from core.backtest import run_backtest_instance (line 890) 
from core.strategy import MultiTimeframeStrategy (line 977)
from watcher_hook import on_backtest_complete (line 930)
from health_utils import pre_backtest_safety_gate (line 569)
```

### **Dependency Chain:**
```
watcher.py → utilities.utils (5,500+ lines)
           → health_utils (1,700+ lines)
           → unified_live_monitor 
           → core.backtest (5,700+ lines)
           → core.strategy (6,881 lines)
           → watcher_hook
```

---

## 🏥 **2. HEALTH_UTILS.PY (1,700+ lines) - HEALTH MONITORING SYSTEM**

### **Direct Imports:**
```python
# Standard Libraries Only
import os, sys, subprocess, json, ast, hashlib, re, fnmatch
from datetime import datetime
from typing import Dict, Optional, Tuple, List

# NO INTERNAL DEPENDENCIES - Standalone health system
```

### **Key Features:**
- **IntelligentRepairEngine** - AI-driven auto-repair system
- **Comprehensive health checks** - System validation
- **Predictive issue detection** - Future problem forecasting
- **Self-contained** - No dependencies on other custom modules

---

## 📊 **3. UTILITIES/UTILS.PY (5,500+ lines) - MASSIVE UTILITY ENGINE**

### **Direct Imports:**
```python
# Standard Libraries (50+ imports)
import pandas as pd, numpy as np, json, os, sys, asyncio, datetime, traceback, time
import logging, threading, shutil, glob, argparse, re, matplotlib.pyplot as plt
import ccxt.pro as ccxt, ast, subprocess, importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator

# Internal Dependencies (scattered throughout file)
from core.strategy import MultiTimeframeStrategy (multiple locations)
from core.position_manager import PositionManager (multiple locations)
from core.portfolio import Portfolio (multiple locations)
from unified_live_monitor import get_monitor, on_backtest_complete, send_system_status
from core.backtest import IchimokuBacktester (line 3917)
```

### **Classes Contained:**
- **WatcherHealthMonitor** - Pipeline monitoring
- **HealthMonitor** - Async health monitoring  
- **EnhancedMonitor** - System monitoring wrapper
- **PerformanceMonitor** - Performance tracking
- **ParallelOptimizer** - Optimization utilities
- **Multiple other utility classes**

---

## 🎯 **4. CORE/BACKTEST.PY (5,700+ lines) - BACKTESTING ENGINE**

### **Direct Imports:**
```python
# Standard Libraries (30+ imports)
import os, datetime, json, pandas as pd, numpy as np, logging, multiprocessing
import warnings, sys, glob, random, shutil, stat, traceback, tempfile
import signal, threading, urllib3, time, argparse, matplotlib
from numba import njit, typed
from numba.core import types
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from joblib import Parallel, delayed, parallel_backend, Memory

# Internal Dependencies
from utilities.utils import (
    performance_monitor, profile, ParallelOptimizer, EnhancedMonitor,
    log_to_file, central_logger, analyze_parameter_performance,
    WatcherParameterValidator
)
from core.config_validation import validate_config
from core.strategy import (
    CachedIndicatorStrategy, SignalQualityStrategy, MultiTimeframeStrategy,
    RegimeSpecificOptimizer, MarketRegime
)
from core.portfolio import Portfolio
from core.position_manager import PositionManager
from utilities.vercel_utils import upload_to_vercel, sync_parameters_to_vercel, check_vercel_connection
from health_utils import auto_fix_on_config_load
```

### **Key Features:**
- **IchimokuBacktester** - Main backtesting class
- **Walk-forward optimization** - Parameter optimization
- **Optuna/Bayesian optimization** - Advanced optimization
- **Parameter bounds auto-adjustment** - Self-optimization

---

## 🎯 **5. CORE/STRATEGY.PY (6,881 lines) - STRATEGY ENGINE**

### **Direct Imports:**
```python
# Standard Libraries
import os, pandas as pd, numpy as np, ta, logging, time, joblib, json, hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, average_true_range

# Internal Dependencies
from core.enums import MarketRegime as CentralMarketRegime

# Dynamic Health Imports (conditional)
from health_utils import ensure_system_health (lines 2344, 2565)
```

### **Key Features:**
- **MultiTimeframeStrategy** - Main strategy class
- **Market regime detection** - Trend/range/volatile detection
- **RegimeSpecificOptimizer** - Adaptive optimization
- **CachedIndicatorStrategy** - Performance optimization
- **SignalQualityStrategy** - Signal validation

---

## 📡 **6. UNIFIED_LIVE_MONITOR.PY - LIVE MONITORING SYSTEM**

### **Direct Imports:**
```python
# Standard Libraries
import json, os, requests, time, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# NO INTERNAL DEPENDENCIES - Self-contained monitoring system
```

### **Key Features:**
- **UnifiedLiveBotMonitor** - Main monitoring class
- **Real-time bot monitoring** - Live performance tracking
- **Parameter sync coordination** - Live bot parameter updates
- **Self-contained** - No dependencies on other custom modules

---

## 🔧 **7. CORE/PORTFOLIO.PY & CORE/POSITION_MANAGER.PY**

### **Dependencies:**
```python
# Both have health_utils integration
from health_utils import ensure_system_health

# Standard libraries + basic imports
# Minimal internal dependencies
```

---

## 🚨 **CRITICAL DEPENDENCY ISSUES IDENTIFIED:**

### **1. CIRCULAR DEPENDENCY RISK:**
```
utilities.utils ←→ core.strategy ←→ core.backtest ←→ utilities.utils
```

### **2. MASSIVE UTILITIES.UTILS BLOAT:**
- **5,500+ lines** with multiple overlapping monitoring classes
- **Imports from 4+ core modules** creating tight coupling
- **Contains redundant functionality** already in health_utils

### **3. HEALTH MONITORING REDUNDANCY:**
- **health_utils.py** - Standalone, comprehensive (1,700+ lines)
- **WatcherHealthMonitor** in utilities.utils - Pipeline monitoring
- **HealthMonitor** in utilities.utils - Async monitoring
- **EnhancedMonitor** in utilities.utils - System monitoring

### **4. IMPORT SCATTER:**
- **Multiple files importing same modules** at different locations
- **Conditional imports throughout** making dependency tracking difficult
- **Dynamic imports in functions** creating runtime dependencies

---

## 📈 **CONSOLIDATION PRIORITY MATRIX:**

### **HIGH PRIORITY (Critical):**
1. **Consolidate health monitoring** - 4 overlapping systems → 1 unified system
2. **Reduce utilities.utils bloat** - 5,500 lines → 2,500 lines focused utilities
3. **Break circular dependencies** - Restructure core module imports

### **MEDIUM PRIORITY:**
1. **Streamline core module imports** - Reduce tight coupling
2. **Centralize monitoring config** - Single configuration management
3. **Optimize import patterns** - Reduce conditional/dynamic imports

### **LOW PRIORITY:**
1. **API file cleanup** - Remove redundant API endpoints
2. **Test file organization** - Consolidate testing utilities
3. **Documentation updates** - Update dependency documentation

---

## 🎯 **RECOMMENDED CONSOLIDATION APPROACH:**

### **PHASE 1: Health Monitoring Unification**
```python
# CREATE: health_engine.py (consolidated)
class WatcherHealthEngine:
    # Combines: health_utils + WatcherHealthMonitor + HealthMonitor + EnhancedMonitor
    
# REMOVE: All monitoring classes from utilities.utils
# RESULT: Single health monitoring system
```

### **PHASE 2: Utilities Streamlining**
```python
# EXTRACT: utilities/core.py (essential only)
def analyze_limits()    # Parameter analysis
def download_data()     # Data management  
def log_message()       # Centralized logging

# REMOVE: WatcherHealthMonitor, HealthMonitor, EnhancedMonitor classes
# RESULT: utilities.utils: 5,500 → 2,500 lines
```

### **PHASE 3: Import Optimization**
```python
# BEFORE: watcher.py imports
from utilities.utils import send_notification, send_error_alert, log_message, download_data, check_data, analyze_limits
from health_utils import run_health_check, get_repair_engine, run_comprehensive_health_check

# AFTER: watcher.py imports  
from health_engine import WatcherHealthEngine
from utilities.core import analyze_limits, download_data, log_message
from unified_live_monitor import start_monitoring
```

This dependency mapping reveals massive redundancy and circular dependencies that can be dramatically streamlined through systematic consolidation.
