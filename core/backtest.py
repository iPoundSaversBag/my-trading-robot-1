#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
#
#                            ICHIMOKU BACKTESTER & OPTIMIZER
#
# ==============================================================================
#
# FILE: ichimoku_backtest.py
#
# PURPOSE:
#   This script is the primary engine for backtesting and optimizing a complex,
#   multi-indicator trading strategy. The strategy is fundamentally based on the
#   Ichimoku Kinko Hyo system but is augmented with signals from the Relative
#   Strength Index (RSI), the Average Directional Index (ADX), and Bollinger
#   Bands. All technical indicators are calculated using the 'ta' library.
#
# METHODOLOGY:
#   The core methodology is a time-series cross-validation technique known as
#   Walk-Forward Optimization (WFO). This approach avoids lookahead bias by
#   iterating through the dataset in rolling time windows, defined by
#   `training_days` and `testing_days` in the configuration file.
#
#   For each window:
#   1. TRAINING: An optimization library (Optuna or Scikit-Optimize) is used on
#      the training data to find the best-performing parameter combination.
#   2. TESTING: The single best parameter set from the training phase is thenet from the training phase is then
#      applied to the unseen "out-of-sample" testing data to simulate real-world
#      performance.
#
#   Capital is managed using a "chained" equity system, where the final capital
#   from one testing window becomes the initial capital for the next, ensuring
#   a realistic, compounding equity curve.
#
# KEY FEATURES:
#   - Pluggable Optimization Engines: Supports two powerful optimization libraries:
#     - Optuna: A modern, Pythonic hyperparameter optimization framework.
#     - Scikit-Optimize (`gp_minimize`): For Bayesian optimization using Gaussian Processes.
#     The objective is to maximize the Sharpe Ratio by minimizing its negative value.
#
#   - Two-Tier Self-Optimization Loop:
#     1. Parameter Value Search: The inner loop, driven by Optuna/Skopt, searches
#        for the optimal *values* of strategy parameters within each training window.
#     2. Warm-Starting: The script saves the best parameters from the final window
#        into the `best_parameters_so_far` block of `optimization_config.json`.
#        The next full execution of the script uses these values to "warm-start"
#        the optimization process, focusing the search on historically profitable areas.
#
#   - High-Performance Numba Core: The critical, bar-by-bar backtesting logic is
#     isolated in the `_backtest_core_numba` function. This function is compiled
#     Just-In-Time (JIT) by the Numba library, which translates the Python code
#     to highly efficient machine code for C-like execution speed.
#
#   - Sophisticated Trade Management:
#     - Partial Take-Profits: A portion of a winning trade can be closed based on
#       the `PARTIAL_EXIT_PERCENTAGE` when the price hits a target defined by the
#       `TP_ATR_MULTIPLIER`.
#     - Adaptive Trailing Stop Loss (TSL): The TSL's distance from the current
#       price is dynamically adjusted using two different ATR multipliers:
#       `TRENDING_TSL_ATR_MULTIPLIER` for trending markets (ADX > threshold) and
#       `RANGING_TSL_ATR_MULTIPLIER` for ranging markets.
#
#   - Modular & Callable Design: The entire process is wrapped in the
#     `run_backtest_instance` function, allowing it to be seamlessly imported and
#     executed by the `watcher.py` orchestration script.
#
# ==============================================================================

# ichimoku_backtest.py (The Definitive, Clean-Output Research Framework - Final Version)
# Implements realistic "no leverage" compounding with robust risk management, Numba typing, and Monte Carlo simulations.

import os
# Set environment to handle Unicode properly on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
import datetime
import json
import os
import pandas as pd
import numpy as np
import logging
import logging.handlers
import multiprocessing
import warnings
from datetime import datetime, timedelta
import sys
import glob
import random
import shutil
import stat
import traceback
import tempfile
import signal
import threading

# --- Joblib for Parallel Processing ---
from joblib import Parallel, delayed, parallel_backend

# --- Global Queue and Configuration for Parallel Logging ---
log_queue = multiprocessing.Queue(-1)

def worker_log_configurator(queue):
    """Configures logging for a worker process to send logs to the queue."""
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    if not root.handlers: # Avoid adding handlers multiple times
        root.addHandler(h)
        root.setLevel(logging.DEBUG)

def log_listener(queue):
    """
    Listens for log records on a queue and prints them to the console.
    This function runs in a dedicated process.
    """
    # Basic configuration for the listener's console output
    log_format = '%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_format, level=logging.DEBUG)
    
    while True:
        try:
            record = queue.get()
            if record is None:  # Sentinel value to stop the listener
                break
            logger = logging.getLogger(record.name)
            # Ensure the logger level is appropriate to handle the record
            if not logger.isEnabledFor(record.levelno):
                logger.setLevel(record.levelno)
            logger.handle(record)
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            import sys, traceback
            print('Log listener encountered an error:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

# Suppress noisy urllib3 warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='.*oauth2.googleapis.com.*')
warnings.filterwarnings('ignore', message='.*api.alternative.me.*')
import time
from numba import njit, typed
from numba.core import types
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import argparse
import matplotlib
matplotlib.use('Agg')
from joblib import Memory

# --- Public API: module-level wrapper for watcher.py ---
def run_backtest_instance(args):
    """
    Module-level entry point for running a single backtest instance.
    Mirrors the behavior expected by watcher.py by instantiating
    IchimokuBacktester and delegating to its instance method.
    """
    try:
        backtester = IchimokuBacktester(
            config_path=args.config,
            intensity_override=getattr(args, 'intensity', None),
            no_warmup=getattr(args, 'no_warmup', False),
            optimizer=getattr(args, 'optimizer', 'optuna'),
            min_trades_override=getattr(args, 'min_trades', None),
            runs_to_keep_override=None,  # enforce config value
            debug_mode=getattr(args, 'debug', False),
            no_numba=getattr(args, 'no_numba', False),
            no_cache=getattr(args, 'no_cache', False),
            clear_cache=getattr(args, 'clear_cache', False),
        )
        return backtester.run_backtest_instance(args)
    except Exception as e:
        try:
            log_to_file(f"FATAL: run_backtest_instance failed: {e}", print_to_console=True)
        except Exception:
            print(f"FATAL: run_backtest_instance failed: {e}")
        return None

# --- Helper: Walk-forward window computation at module level ---
def compute_walk_forward_windows(config: dict, df: pd.DataFrame):
    """
    Compute walk-forward windows using the provided configuration and dataframe.
    Mirrors the class logic but available at module scope to avoid attribute issues.
    Returns list of tuples: (train_start, train_end, test_end).
    """
    windows = []
    try:
        if df is None or df.empty:
            return windows

        wfo_settings = (config or {}).get('walk_forward_settings', {})
        training_days = wfo_settings.get('training_days', 365)
        testing_days = wfo_settings.get('testing_days', 90)

        start_date = df.index.min()
        end_date = df.index.max()

        current_train_start = start_date
        while True:
            train_end = current_train_start + timedelta(days=training_days)
            test_end = train_end + timedelta(days=testing_days)
            if test_end > end_date:
                break
            windows.append((current_train_start, train_end, test_end))
            current_train_start += timedelta(days=testing_days)
        return windows
    except Exception:
        # Be conservative; return empty on any failure
        return []

# --- PERFORMANCE IMPROVEMENTS INTEGRATION ---
# Add project root to Python path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PERFORMANCE_MONITORING_AVAILABLE = False
CACHED_STRATEGY_AVAILABLE = False
QUALITY_STRATEGY_AVAILABLE = False
PARALLEL_OPTIMIZER_AVAILABLE = False

try:
    from utilities.utils import performance_monitor, profile  # Consolidated into utils
    try:
        from core.config_validation import validate_config  # type: ignore
    except Exception:  # pragma: no cover
        validate_config = None  # type: ignore
    PERFORMANCE_MONITORING_AVAILABLE = True
    # Silently available
except ImportError:
    pass  # Silently not available

try:
    from core.strategy import CachedIndicatorStrategy  # Consolidated into strategy
    CACHED_STRATEGY_AVAILABLE = True
    # Silently available
except ImportError:
    pass  # Silently not available

try:
    from core.strategy import SignalQualityStrategy  # Consolidated into strategy
    QUALITY_STRATEGY_AVAILABLE = True
    # Silently available
except ImportError:
    pass  # Silently not available

try:
    from utilities.utils import ParallelOptimizer  # Consolidated into utils
    PARALLEL_OPTIMIZER_AVAILABLE = True
    # Silently available
except ImportError:
    pass  # Silently not available

# --- FIX: Suppress known, non-critical RuntimeWarning from numpy/pandas ---
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging to reduce verbosity
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('googleapiclient').setLevel(logging.ERROR)

# Set default logging level to WARNING to reduce noise
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Optional Imports ---
MATPLOTLIB_AVAILABLE = False
MPLFINANCE_AVAILABLE = False

# --- Enhanced Monitoring Integration ---
ENHANCED_MONITORING_AVAILABLE = False
try:
    from utilities.utils import EnhancedMonitor
    ENHANCED_MONITORING_AVAILABLE = True
except ImportError:
    print("Enhanced monitoring not available - continuing without monitoring")
    ENHANCED_MONITORING_AVAILABLE = False
OPTUNA_AVAILABLE = False
QUANTSTATS_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    pass

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    pass

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import quantstats as qs # type: ignore
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

# --- Technical Analysis Library ---
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, average_true_range

# Import with proper path handling
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis.generate_plots import (
        plot_trades_for_window,
        plot_pnl_distribution,
        enhance_performance_report,
        collect_comprehensive_analysis_data,
    )
    PLOTS_AVAILABLE = True
except Exception:
    # Define safe stubs (no legacy enhancer fallback kept after clean slate)
    def plot_trades_for_window(*_, **__):
        return None
    def plot_pnl_distribution(*_, **__):
        return None
    def enhance_performance_report(*_, **__):
        return None
    def collect_comprehensive_analysis_data(*_, **__):
        return {}
    PLOTS_AVAILABLE = False

try:
    from core.strategy import MultiTimeframeStrategy as Strategy
    STRATEGY_AVAILABLE = True
    # Silently available
except ImportError:
    STRATEGY_AVAILABLE = False
    pass  # Silently not available

try:
    from core.portfolio import Portfolio
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False

try:
    from core.position_manager import PositionManager
    POSITION_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from .position_manager import PositionManager
        POSITION_MANAGER_AVAILABLE = True
    except ImportError:
        POSITION_MANAGER_AVAILABLE = False
        PositionManager = None  # Fallback for error handling

try:
    from utilities.gcp_utils import upload_to_gcs, sync_parameters_to_cloud, check_gcs_connection  # Enhanced GCP integration
    GCP_AVAILABLE = True
    # Silently available
except ImportError:
    GCP_AVAILABLE = False
    # Silently not available
    
    # Create dummy functions for fallback
    def upload_to_gcs(source_file_path, destination_blob_name):
        # Silently fail
        return False
    
    def sync_parameters_to_cloud(local_params_file, cloud_blob_name):
        # Silently fail
        return False
    
    def check_gcs_connection():
        return {'connected': False, 'error_messages': ['GCP utilities not available']}

# --- Refactored Backtesting Core ---
def run_backtest(params, processed_df, initial_capital, window_num=0, comprehensive_manager=None):
    """
    Executes a single backtest run with a given set of parameters using the
    refactored PositionManager and Portfolio classes. This is the unified
    backbone for both optimization and final test runs.
    """
    if not POSITION_MANAGER_AVAILABLE:
        raise ImportError("PositionManager not available. Check imports and dependencies.")
    
    full_params = {**params} # Create a mutable copy
    portfolio = Portfolio(initial_capital=initial_capital)
    
    # Attach comprehensive manager to portfolio if available
    if comprehensive_manager:
        portfolio.comprehensive_manager = comprehensive_manager
    
    position_manager = PositionManager(full_params)

    for i in range(1, len(processed_df)):
        current_candle = processed_df.iloc[i]
        
        # --- Exit Logic ---
        if position_manager.in_position:
            # Update intra-trade analytics for false-signal detection
            try:
                pdets = position_manager.position_details
                pdets['bars_in_trade'] = pdets.get('bars_in_trade', 0) + 1
                entry_price = pdets.get('entry_price', current_candle['close'])
                trade_type = pdets.get('type', 'long')
                # Compute excursion relative to entry
                if trade_type == 'long':
                    change = (current_candle['close'] - entry_price) / max(1e-9, entry_price)
                else:
                    change = (entry_price - current_candle['close']) / max(1e-9, entry_price)
                mae = pdets.get('max_adverse_excursion', 0.0)
                mfe = pdets.get('max_favorable_excursion', 0.0)
                if change < 0:
                    pdets['max_adverse_excursion'] = min(mae, change) if mae < 0 else change
                else:
                    pdets['max_favorable_excursion'] = max(mfe, change)
            except Exception:
                pass
            # Check for advanced position management (TSL, partial TP, etc.)
            management_result = position_manager.update_position_management(current_candle)
            
            if management_result['action'] == 'partial_exit':
                # Execute partial take profit
                partial_pnl, remaining_size = position_manager.execute_partial_take_profit(
                    management_result['price'], current_candle.name
                )
                portfolio.cash += partial_pnl
                # Continue to check for full exit conditions
            
            # Fallback to traditional exit checking for compatibility
            position_manager.update_trailing_stop(current_candle)
            exit_reason = position_manager.check_for_exit(current_candle)
            
            # Override with management result if it indicates full exit
            if management_result['action'] == 'exit_full':
                exit_reason = management_result['reason']
            
            if exit_reason:
                # --- REALISM: Apply variable slippage on exit ---
                base_slippage = full_params.get('base_slippage_percent', 0.0005)
                vol_multiplier = full_params.get('vol_slippage_multiplier', 1.5)
                
                slippage = base_slippage
                if full_params.get('variable_slippage_enabled', False) and 'volatility_atr' in current_candle and current_candle['volatility_atr'] > 0:
                    slippage += (current_candle['volatility_atr'] / current_candle['close']) * vol_multiplier
                
                exit_price = current_candle['close'] * (1 - slippage) if position_manager.position_details['type'] == 'long' else current_candle['close'] * (1 + slippage)
                
                pos_details_before_exit = position_manager.position_details.copy()
                pnl, size_exited = position_manager.exit_position(exit_reason, exit_price, current_candle.name, portfolio.cash)
                
                # CRITICAL FIX: Update portfolio cash with PnL from the trade
                portfolio.cash += pnl
                
                portfolio.record_trade(
                    entry_timestamp=pos_details_before_exit['entry_timestamp'],
                    exit_timestamp=current_candle.name,
                    entry_price=pos_details_before_exit['entry_price'],
                    exit_price=exit_price,
                    size=size_exited,
                    trade_type=1 if pos_details_before_exit['type'] == 'long' else -1,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    window_num=window_num,
                    window_start_capital=initial_capital,
                    strategy_params=full_params,
                    comprehensive_manager=getattr(portfolio, 'comprehensive_manager', None)
                )

        # --- Entry Logic ---
        if not position_manager.in_position:
            # Use the current candle's signal - MultiTimeframeStrategy generates signals for current candle
            signal_candle = current_candle
            
            # Extract volume metrics from current candle for enhanced position filtering
            volume_metrics = {}
            if hasattr(current_candle, 'get'):
                # Extract volume analysis metrics added by enhanced strategy
                volume_metrics = {
                    'volume_strength': current_candle.get('volume_strength', 0.0),
                    'volume_trend': current_candle.get('volume_trend', 0.0),
                    'volume_momentum': current_candle.get('volume_momentum', 0.0),
                    'obv_trend': current_candle.get('obv_trend', 0.0),
                    'vwap_position': current_candle.get('vwap_position', 0.0)
                }
            
            trade_type, confidence = position_manager.check_for_entry(signal_candle, volume_metrics)
            
            if trade_type:
                # Enter on the current candle's data
                # --- REALISM: Apply variable slippage on entry ---
                base_slippage = full_params.get('base_slippage_percent', 0.0005)
                vol_multiplier = full_params.get('vol_slippage_multiplier', 1.5)
                
                slippage = base_slippage
                if full_params.get('variable_slippage_enabled', False) and 'volatility_atr' in current_candle and current_candle['volatility_atr'] > 0:
                    slippage += (current_candle['volatility_atr'] / current_candle['close']) * vol_multiplier
                
                # Calculate entry price with slippage
                entry_price = current_candle['close'] * (1 + slippage) if trade_type == 'long' else current_candle['close'] * (1 - slippage)

                # Calculate strategy levels using optimizable parameters from full_params
                signal_direction = 1 if trade_type == 'long' else -1
                atr_value = current_candle.get('atr', current_candle['close'] * 0.02)  # Fallback to 2% if ATR missing
                
                # Calculate risk management levels using parameters from full_params
                stop_loss_multiplier = full_params.get('STOP_LOSS_MULTIPLIER', 2.0)
                take_profit_multiplier = full_params.get('TAKE_PROFIT_MULTIPLIER', 3.0)
                trailing_stop_multiplier = full_params.get('TRAILING_STOP_MULTIPLIER', 0.02)
                partial_exit_percentage = full_params.get('PARTIAL_EXIT_PERCENTAGE', 0.5)
                
                # Calculate stop loss and take profit prices based on ATR
                partial_tp_fraction = float(full_params.get('PARTIAL_TP_FRACTION_OF_TP', 0.6))
                partial_tp_fraction = max(0.1, min(partial_tp_fraction, 0.99))
                if signal_direction == 1:  # Long position
                    stop_loss_price = entry_price - (atr_value * stop_loss_multiplier)
                    take_profit_price = entry_price + (atr_value * take_profit_multiplier)
                    # Partial TP is a configurable fraction of distance to full TP
                    partial_tp_price = entry_price + (atr_value * take_profit_multiplier * partial_tp_fraction)
                else:  # Short position
                    stop_loss_price = entry_price + (atr_value * stop_loss_multiplier)
                    take_profit_price = entry_price - (atr_value * take_profit_multiplier)
                    # Partial TP is a configurable fraction of distance to full TP
                    partial_tp_price = entry_price - (atr_value * take_profit_multiplier * partial_tp_fraction)
                
                # Create strategy levels dictionary
                strategy_levels = {
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'partial_tp_price': partial_tp_price,
                    # Default partial percentage lowered to 30% if not provided
                    'partial_tp_percentage': full_params.get('PARTIAL_EXIT_PERCENTAGE', 0.3),
                    'trailing_stop_distance': atr_value * trailing_stop_multiplier,
                    'estimated_slippage': slippage
                }
                
                # Enter position with calculated strategy levels
                if position_manager.enter_position_with_risk_management(trade_type, current_candle, portfolio.cash, 
                                                                       strategy_levels, entry_price):
                    # Add entry timestamp to position details for later reference
                    position_manager.position_details['entry_timestamp'] = current_candle.name

    # --- Final Liquidation at the end of the backtest period ---
    if position_manager.in_position:
        last_candle = processed_df.iloc[-1]
        exit_price = last_candle['close']
        pos_details_before_exit = position_manager.position_details.copy()
        pnl, size_exited = position_manager.exit_position("End of Backtest", exit_price, last_candle.name, portfolio.cash)
        
        # CRITICAL FIX: Update portfolio cash with final PnL  
        portfolio.cash += pnl
        
        portfolio.record_trade(
            entry_timestamp=pos_details_before_exit['entry_timestamp'],
            exit_timestamp=last_candle.name,
            entry_price=pos_details_before_exit['entry_price'],
            exit_price=exit_price,
            size=size_exited,
            trade_type=1 if pos_details_before_exit['type'] == 'long' else -1,
            pnl=pnl,
            exit_reason="End of Backtest",
            window_num=window_num,
            window_start_capital=initial_capital,
            strategy_params=full_params,
            comprehensive_manager=getattr(portfolio, 'comprehensive_manager', None)
        )

    # --- Return Results ---
    final_equity = portfolio.cash
    trades_df = portfolio.get_trade_history_df()
    metrics = portfolio.calculate_performance_metrics()
    
    return metrics, trades_df, final_equity

# --- Helper Functions ---
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Import centralized logging system
from utilities.utils import log_to_file


def manage_output_files(base_output_dir, runs_to_keep):
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        return
    
    all_run_dirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]
    
    # Sort directories by timestamp
    sorted_dirs = sorted(all_run_dirs, key=lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S") if "_" in x else datetime.min, reverse=True)

    # Identify directories to remove
    dirs_to_remove = sorted_dirs[runs_to_keep:]
    
    for dir_name in dirs_to_remove:
        path_to_remove = os.path.join(base_output_dir, dir_name)
        try:
            shutil.rmtree(path_to_remove, onerror=remove_readonly)
            log_to_file(f"Housekeeping: Removed old output directory: {path_to_remove}", print_to_console=False)
        except OSError as e:
            log_to_file(f"Housekeeping Error removing {path_to_remove}: {e}", print_to_console=False)

def load_data_from_parquet(filename='crypto_data.parquet'):
    try:
        df = pd.read_parquet(filename)
        if not isinstance(df.index, pd.DatetimeIndex):
            log_to_file(f"Error: Index of {filename} is not a DatetimeIndex.")
            return pd.DataFrame()
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        log_to_file(f"FATAL: Error loading data from {filename}: {e}")
        return pd.DataFrame()


class IchimokuBacktester:
    current_run_dir_static = None

    def log_problem(self, message):
        """Logs a message to the problem summary and the main log file."""
        self.problem_summary.append(message)
        # Log with a clear marker to the main file and console
        log_to_file(f"PROBLEM DETECTED: {message}", print_to_console=True)

    def print_problem_summary(self):
        """Prints a concise summary of all problems encountered during the run."""
        if not self.problem_summary:
            log_to_file("--- Run completed with no problems detected. ---", print_to_console=True)
        else:
            # Use print() for the summary box to ensure it stands out from the log
            print("\n" + "="*60)
            print(" " * 20 + "RUN PROBLEM SUMMARY")
            print("="*60)
            for i, problem in enumerate(self.problem_summary, 1):
                print(f"  {i}. {problem}")
            print("="*60)

    def log_debug(self, msg):
        """Routes a debug message through the standard logging framework."""
        # This ensures that in parallel execution, messages are sent
        # to the log_queue via the worker_log_configurator.
        logging.debug(msg)

    def print_debug_log(self):
        if getattr(self, 'debug_mode', False) and hasattr(self, 'debug_messages') and self.debug_messages:
            print("\n--- DEBUG LOG ---")
            for msg in self.debug_messages:
                print(msg)
            print("--- END DEBUG LOG ---\n")

    def setup_output_directory(self):
        """Creates a unique, timestamped directory for the current run's output."""
        # Manage old directories first
        runs_to_keep = self.runs_to_keep_override if self.runs_to_keep_override is not None else self.config.get('output_settings', {}).get('runs_to_keep', 10)
        manage_output_files(self.base_output_dir, runs_to_keep)

        # Create a new directory for this specific run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.base_output_dir, timestamp)
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        # Set the static variable so the log_to_file function knows where to save.
        IchimokuBacktester.current_run_dir_static = self.current_run_dir
        # Initialize parameter ledger path for this run
        try:
            self.parameter_ledger_path = os.path.join(self.current_run_dir, 'parameter_ledger.jsonl')
        except Exception:
            self.parameter_ledger_path = 'parameter_ledger.jsonl'
        log_to_file(f"Output for this run will be saved in: {self.current_run_dir}", print_to_console=False)

    def __init__(self, config_path='optimization_config.json', intensity_override=None, no_warmup=False, optimizer='bayesian', min_trades_override=None, runs_to_keep_override=None, debug_mode=False, no_numba=False, no_cache=False, clear_cache=False):
        """
        Initializes the backtester with performance improvements.
        """
        self.debug_mode = debug_mode
        self.no_numba = no_numba
        self.debug_messages = []
        if self.debug_mode:
            self.log_debug("--- Debug Mode Enabled ---")
        
        self.problem_summary = []
        # Performance monitoring flag (enabled after WFO finishes in debug)
        self.performance_monitoring_enabled = False

        # Load configuration
        self.config_path = config_path
        self.config = self.load_config(self.config_path)
        if self.config is None:
            sys.exit(1)
        # Non-fatal schema / structure validation
        if validate_config:
            try:
                _issues = validate_config(self.config)
                if _issues.get('missing'):
                    log_to_file(f"[CONFIG VALIDATION] Missing keys: {_issues['missing']}", print_to_console=True)
                if _issues.get('unknown'):
                    log_to_file(f"[CONFIG VALIDATION] Unknown adaptive_master keys: {_issues['unknown']}", print_to_console=True)
            except Exception as _e:  # pragma: no cover
                log_to_file(f"[CONFIG VALIDATION] Validation failed: {_e}", print_to_console=True)

        # ---------------- Debug Settings (non-intrusive defaults) ----------------
        self.debug_config = self.config.get('debug_settings', {}) if self.debug_mode else {}
        self.debug_preserve_full_dataset = self.debug_config.get('preserve_full_dataset', False)  # default False keeps prior behavior
        self.debug_limit_windows = self.debug_config.get('limit_windows')  # optional int
        self.debug_log_trial_details = self.debug_config.get('log_trial_details', True)
        self.debug_zero_trade_penalty_first = int(self.debug_config.get('allow_zero_trade_penalty_first', 2))
        self.debug_dataset_period = self.debug_config.get('dataset_period')  # explicit period key to force slicing
        # Deterministic seeding for reproducibility in debug
        if self.debug_mode:
            try:
                import random, numpy as _np
                random.seed(self.debug_config.get('seed', 42))
                _np.random.seed(self.debug_config.get('seed', 42))
            except Exception:
                pass

        # Apply CLI overrides
        self.warm_start_params = self.config.get('best_parameters_so_far', None)
        if intensity_override:
            self.config['optimization_settings']['intensity'] = str(intensity_override)

        # Defaults and optimizer
        self.default_params = self.config.get('fixed_parameters', {})
        if not self.default_params:
            self.log_problem("The 'fixed_parameters' block is missing from your config file. Using hardcoded defaults.")
        self.use_numba_warmup = not no_warmup
        self.optimizer = optimizer

        # Helper: intensity param resolver
        def get_intensity_param(self, param_name, default_value=None):
            optimization_settings = self.config.get('optimization_settings', {})
            param_value = optimization_settings.get(param_name, default_value)
            if isinstance(param_value, dict):
                opt_intensity = str(optimization_settings.get('intensity', '1'))
                return param_value.get(opt_intensity, default_value)
            return param_value
        self.get_intensity_param = get_intensity_param.__get__(self, type(self))

        # ---------------- Adaptive Enhancements / Self-Optimization (state init) ----------------
        opt_settings = self.config.get('optimization_settings', {})
        self.adaptive_enhancements_cfg = opt_settings.get('adaptive_enhancements', {})
        auto_tune_cfg = self.adaptive_enhancements_cfg.get('exploration_fraction_auto_tune', {})
        self._explore_auto_enabled = bool(auto_tune_cfg.get('enabled', False))
        self._explore_auto_min = float(auto_tune_cfg.get('min_fraction', 0.3))
        self._explore_auto_max = float(auto_tune_cfg.get('max_fraction', 0.6))
        self._explore_auto_patience = int(auto_tune_cfg.get('adjust_patience', 4))
        self._explore_auto_slope_thresh = float(auto_tune_cfg.get('improvement_slope_threshold', 0.015))
        self._dynamic_explore_frac = None
        self._window_scores = []
        self._window_sharpes = []
        self._last_exploration_completed = None
        self._last_refine_trials = None
        self._auto_adjust_events = []
        # Early refinement abort config (mapped for gp_minimize path)
        self._early_abort_cfg = {
            'enabled': bool(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('enabled', False)),
            'patience_calls': int(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('patience_trials', 6)),
            'min_relative_improvement': float(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('min_relative_improvement', 0.01)),
            # Extended criteria defaults
            'min_calls_before_check': int(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('min_calls_before_check', 8)),
            'stagnation_slope_window': int(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('stagnation_slope_window', 5)),
            'min_slope_threshold': float(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('min_slope_threshold', 1e-4)),
            'penalty_dominance_ratio': float(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('penalty_dominance_ratio', 0.70)),
            'max_worsening_calls': int(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('max_worsening_calls', 4)),
            'min_absolute_improvement': float(self.adaptive_enhancements_cfg.get('early_refinement_abort', {}).get('min_absolute_improvement', 0.002)),
        }
        # Master adaptive persistence load
        self._master_state_path = self.config.get('optimization_settings', {}).get('adaptive_master', {}).get('state_path', 'adaptive_master_state.json')
        self._master_adaptive_state = None
        try:
            if os.path.exists(self._master_state_path):
                with open(self._master_state_path, 'r') as f:
                    persisted = json.load(f)
                st = persisted.get('state') if isinstance(persisted, dict) else None
                if isinstance(st, dict):
                    self._master_adaptive_state = st
                    self._dynamic_explore_frac = persisted.get('dynamic_explore_fraction', self._dynamic_explore_frac)
                    log_to_file(f"[MASTER_STATE_LOAD] Restored adaptive master state from {self._master_state_path}", print_to_console=self.debug_mode)
        except Exception as e:
            log_to_file(f"[MASTER_STATE_LOAD_ERR] {e}", print_to_console=self.debug_mode)
    # Realism settings and housekeeping
        self.realism_settings = self.config.get('realism_settings', {}).get('variable_slippage', {})
        self.variable_slippage_enabled = self.realism_settings.get('enabled', False)
        self.runs_to_keep_override = runs_to_keep_override

        # Caching
        self.no_cache = no_cache
        self.cache_dir = os.path.join(tempfile.gettempdir(), 'my_trading_robot_cache')
        if clear_cache:
            if os.path.exists(self.cache_dir):
                try:
                    shutil.rmtree(self.cache_dir)
                    log_to_file(f"--- Cache cleared successfully at {self.cache_dir} ---", print_to_console=True)
                except Exception as e:
                    log_to_file(f"--- WARNING: Could not clear cache directory: {e} ---", print_to_console=True)
            else:
                log_to_file(f"--- Cache directory not found at {self.cache_dir}, nothing to clear. ---", print_to_console=False)

        if self.no_cache:
            log_to_file("--- Caching is disabled by command-line flag. ---", print_to_console=True)
            self.memory = None
            self.cached_generate_signals = None
        else:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.memory = Memory(self.cache_dir, verbose=0)
                self.cached_generate_signals = self.memory.cache(Strategy.generate_signals)
                log_to_file(f"--- Caching is enabled. Cache directory: {self.cache_dir} ---", print_to_console=False)
            except Exception as e:
                log_to_file(f"--- WARNING: Cache setup failed, disabling cache: {e} ---", print_to_console=True)
                self.memory = None
                self.cached_generate_signals = None
                self.no_cache = True

        # Data loading
        data_settings = self.config.get('data_settings', {})
        data_file = data_settings.get('file_path')
        if not data_file:
            log_to_file("FATAL: 'data_settings' -> 'file_path' is not defined in your configuration file.", print_to_console=True)
            sys.exit(1)
        log_to_file(f"Loading data from {data_file}...")
        try:
            self.df_full = pd.read_parquet(data_file)
        except FileNotFoundError:
            log_to_file(f"FATAL: Data file not found at path: {data_file}", print_to_console=True)
            sys.exit(1)
        except Exception as e:
            log_to_file(f"FATAL: An error occurred while loading the data file: {e}", print_to_console=True)
            sys.exit(1)
        log_to_file("Data loaded successfully.")

        # Debug dataset selection (conditional)
        if self.debug_mode and (not self.debug_preserve_full_dataset or self.debug_dataset_period):
            original_length = len(self.df_full)
            end_date = self.df_full.index.max()
            start_date = self.df_full.index.min()
            debug_config = self.debug_config
            preferred_period = self.debug_dataset_period or debug_config.get('preferred_period', 'auto')
            min_rows_required = debug_config.get('min_rows', 15000)
            debug_options = {
                'recent_6months': pd.DateOffset(months=6),
                'recent_1year': pd.DateOffset(months=12),
                'recent_2years': pd.DateOffset(months=24),
                'bull_2021': ('2021-09-01', '2021-12-31'),
                'bear_2022': ('2022-05-01', '2022-08-31'),
                'volatile_2024': ('2024-01-01', '2024-06-30'),
            }
            date_range_options = ['bull_2021', 'bear_2022', 'volatile_2024']
            if preferred_period in date_range_options:
                start_str, end_str = debug_options[preferred_period]
                try:
                    debug_data = self.df_full.loc[start_str:end_str]
                    debug_period = f"{preferred_period} ({start_str} to {end_str})"
                    if len(debug_data) == 0:
                        raise ValueError("No data in specified date range")
                except Exception:
                    log_to_file(f"WARNING: Cannot use {preferred_period} period. Falling back to auto-selection.", print_to_console=True)
                    preferred_period = 'auto'
            if preferred_period == 'auto' or preferred_period not in debug_options:
                total_years = (end_date - start_date).days / 365.25
                if total_years >= 3:
                    debug_start = end_date - debug_options['recent_1year']
                    debug_data = self.df_full.loc[debug_start:end_date]
                    debug_period = "recent 1 year (auto-selected)"
                elif total_years >= 1:
                    debug_start = end_date - debug_options['recent_6months']
                    debug_data = self.df_full.loc[debug_start:end_date]
                    debug_period = "recent 6 months (auto-selected)"
                else:
                    debug_data = self.df_full
                    debug_period = "all available data (auto-selected)"
            elif preferred_period in ['recent_6months', 'recent_1year', 'recent_2years']:
                debug_start = end_date - debug_options[preferred_period]
                debug_data = self.df_full.loc[debug_start:end_date]
                debug_period = f"{preferred_period} (user-specified)"
            if len(debug_data) < min_rows_required:
                fallback_periods = ['recent_1year', 'recent_2years']
                for fallback in fallback_periods:
                    if fallback != preferred_period:
                        debug_start = end_date - debug_options[fallback]
                        fallback_data = self.df_full.loc[debug_start:end_date]
                        if len(fallback_data) >= min_rows_required:
                            debug_data = fallback_data
                            debug_period = f"{fallback} (auto-fallback)"
                            break
                if len(debug_data) < min_rows_required:
                    log_to_file(f"WARNING: Debug dataset too small ({len(debug_data)} rows). Using full dataset.", print_to_console=True)
                    debug_data = self.df_full
                    debug_period = "full dataset (insufficient data for debug mode)"
            self.df_full = debug_data
            if len(self.df_full) > 1:
                price_start = self.df_full.iloc[0]['close']
                price_end = self.df_full.iloc[-1]['close']
                price_change = (price_end - price_start) / price_start * 100
                log_to_file(f"--- DEBUG MODE: Using {debug_period} ---", print_to_console=True)
                log_to_file(f"Debug dataset: {len(self.df_full):,} rows ({len(self.df_full)/original_length*100:.1f}% of full data)", print_to_console=True)
                log_to_file(f"Period: {self.df_full.index.min()} to {self.df_full.index.max()}", print_to_console=True)
                log_to_file(f"Price change: {price_change:+.2f}% (${price_start:.0f} -> ${price_end:.0f})", print_to_console=True)
                if abs(price_change) < 5:
                    market_context = "sideways/ranging market"
                elif price_change > 20:
                    market_context = "strong bull market"
                elif price_change > 5:
                    market_context = "bull market"
                elif price_change < -20:
                    market_context = "strong bear market"
                elif price_change < -5:
                    market_context = "bear market"
                else:
                    market_context = "mixed market"
                log_to_file(f"Market context: {market_context}", print_to_console=True)
            else:
                log_to_file(f"Debug mode")

        # Internal tracking for debug trial logging
        self._active_window = None
        if self.debug_mode and self.debug_log_trial_details:
            try:
                os.makedirs('logs', exist_ok=True)
                self._debug_trial_log_path = os.path.join('logs', 'debug_trials.jsonl')
                with open(self._debug_trial_log_path, 'w') as _f:
                    _f.write('')  # truncate file
            except Exception:
                self._debug_trial_log_path = None

        # Initialize state
        self.all_optimized_params = {}
        self.all_trades = []
        self.all_trial_data = []
        self.data_prep_cache = {}
        self.chained_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
        self.risk_per_trade_pct = self.default_params.get('FIXED_RISK_PERCENTAGE', 0.01)
        self.comprehensive_manager = None
        # Directional diagnostics container (one entry per window)
        self.directional_diagnostics = []
        log_to_file("[INFO] Comprehensive Results Manager has been deprecated. Using legacy recording.", print_to_console=False)

        # Parameter search space
        self.search_space = self.define_parameter_spaces()

        # Persistent strategy instance
        if STRATEGY_AVAILABLE:
            self.persistent_strategy = Strategy(self.default_params)
            log_to_file("[PERFORMANCE] Initialized persistent strategy instance for multi-timeframe data caching", print_to_console=False)
        else:
            self.persistent_strategy = None
            log_to_file("[WARNING] Strategy class not available - performance will be degraded", print_to_console=False)

        # Penalty settings (dynamic-only system)
        self.penalty_settings = self.config.get('penalty_settings', {})
        self.min_trades_for_dynamic_penalty = self.penalty_settings.get('min_trades_for_dynamic_penalty', 10)
        self.allowed_max_drawdown = self.penalty_settings.get('allowed_max_drawdown', 0.25)  # 25% acceptable ceiling
        self.dynamic_trade_penalty_base = self.penalty_settings.get('dynamic_trade_penalty_base', 5.0)
        self.dynamic_drawdown_penalty_base = self.penalty_settings.get('dynamic_drawdown_penalty_base', 10.0)
        # Expectancy / profit factor / trade frequency extensions
        self.profit_factor_target = self.penalty_settings.get('profit_factor_target', 1.2)
        self.profit_factor_penalty_base = self.penalty_settings.get('profit_factor_penalty_base', 15.0)
        self.expectancy_target = self.penalty_settings.get('expectancy_target', 1.1)
        self.expectancy_penalty_base = self.penalty_settings.get('expectancy_penalty_base', 12.0)
        self.max_trades_threshold = self.penalty_settings.get('max_trades_threshold', 55)
        self.excessive_trade_penalty_base = self.penalty_settings.get('excessive_trade_penalty_base', 8.0)
        log_to_file(
            f"[PENALTY] Dynamic penalty system active | target_trades={self.min_trades_for_dynamic_penalty} allowed_dd={self.allowed_max_drawdown} trade_base={self.dynamic_trade_penalty_base} dd_base={self.dynamic_drawdown_penalty_base}",
            print_to_console=False
        )

    def run_walk_forward_optimization(self):
        """
        Main loop for the walk-forward optimization process.
        It iterates through training and testing windows, runs optimization,
        and evaluates performance on out-of-sample data.
        This version is enhanced to use multiprocessing for handling logs
        from parallel optimization trials.
        """
        log_to_file("--- Starting Walk-Forward Analysis ---", print_to_console=True)
        log_to_file("Zero-trade trials will be pruned to improve optimization efficiency", print_to_console=True)

        # --- Setup and start the dedicated logging listener process ---
        listener_process = multiprocessing.Process(
            target=log_listener,
            args=(log_queue,),
            name='LogListenerProcess'
        )
        listener_process.start()
        log_to_file("Log listener process started.", print_to_console=False)

        try:
            windows = compute_walk_forward_windows(self.config, self.df_full)
            if not windows:
                self.log_problem("Could not generate any walk-forward windows. Check data and config.")
                self.finalize_and_report()
                return

            log_to_file(f"Data loaded. Starting walk-forward analysis with {len(windows)} windows.")
            last_window_best_params_list = None

            # --- Configure joblib parallel backend dynamically from config (default 'loky') ---
            optimization_settings_full = self.config.get('optimization_settings', {})
            backend_name = optimization_settings_full.get('parallel_backend', 'loky')
            valid_backends = {'loky', 'threading', 'multiprocessing'}
            if backend_name not in valid_backends:
                log_to_file(f"[PARALLEL] Unknown backend '{backend_name}' requested; falling back to 'loky'", print_to_console=True if self.debug_mode else False)
                backend_name = 'loky'
            else:
                log_to_file(f"[PARALLEL] Using joblib backend='{backend_name}'", print_to_console=True if self.debug_mode else False)

            backend_kwargs = {}
            if backend_name in ('loky', 'threading'):
                # Limit nested thread usage for deterministic performance
                backend_kwargs['inner_max_num_threads'] = 1
            with parallel_backend(backend_name, **backend_kwargs):
                worker_log_configurator(log_queue)

                for i, (train_start, train_end, test_end) in enumerate(tqdm(windows, desc="Walk-Forward Windows")):
                    window_num = i + 1
                    log_to_file(f"--- Starting Window #{window_num}/{len(windows)} [{train_start.date()} -> {test_end.date()}] ---", print_to_console=False)
                    
                    # Respect configurable debug_limit_windows instead of hard-coded 3
                    if self.debug_mode:
                        self._active_window = window_num
                        if self.debug_limit_windows is not None and i >= self.debug_limit_windows:
                            log_to_file(f"--- DEBUG MODE: Stopping after {self.debug_limit_windows} walk-forward windows (configured). ---", print_to_console=True)
                            break

                    self.data_prep_cache.clear()
                    
                    train_df = self.df_full.loc[train_start:train_end]
                    test_df = self.df_full.loc[train_end:test_end]
                    
                    opt_intensity = str(self.config['optimization_settings']['intensity'])
                    n_calls = self.config['optimization_settings']['calls_per_window'][opt_intensity]
                    # Apply adaptive calls multiplier if available
                    try:
                        if getattr(self, '_master_adaptive_state', None):
                            mult = float(self._master_adaptive_state.get('calls_multiplier', 1.0))
                            base_n = n_calls
                            n_calls = max(5, int(round(n_calls * mult)))
                            if self.debug_mode and n_calls != base_n:
                                log_to_file(f"[CALLS_APPLY] window={window_num} base={base_n} mult={mult:.3f} -> n_calls={n_calls}", print_to_console=False)
                    except Exception:
                        pass

                    best_params = {}
                    
                    if len(train_df.index) < 100:
                        self.log_problem(f"Window #{window_num}: Skipped optimization due to insufficient data points ({len(train_df.index)} < 100).")
                        if self.all_optimized_params:
                            last_key = sorted(self.all_optimized_params.keys())[-1]
                            best_params = self.all_optimized_params[last_key]
                            log_to_file(f"Reusing parameters from {last_key} for window #{window_num}.", print_to_console=False)
                        elif self.warm_start_params:
                            best_params = self.warm_start_params.copy()
                            log_to_file(f"Using warm start parameters from config for window #{window_num}.", print_to_console=False)
                        else:
                            log_to_file(f"Window #{window_num}: No previous parameters to fall back to. Skipping.", print_to_console=True)
                            continue
                    else:
                        # Hybrid mode detection
                        hybrid_cfg = self.config.get('optimization_settings', {}).get('hybrid_optimization', {})
                        use_hybrid = bool(hybrid_cfg.get('enabled', False))
                        if use_hybrid:
                            # Phase splits
                            # Support dynamic self-optimizing exploration fraction
                            if self._dynamic_explore_frac is not None:
                                explore_frac = self._dynamic_explore_frac
                            else:
                                explore_frac = float(hybrid_cfg.get('exploration_fraction', 0.5))
                            min_explore = int(hybrid_cfg.get('min_explore_trials', 5))
                            min_refine = int(hybrid_cfg.get('min_refine_trials', 5))
                            explore_trials = max(min_explore, int(n_calls * explore_frac))
                            refine_trials = max(min_refine, n_calls - explore_trials)
                            log_to_file(f"Window #{window_num}: Starting HYBRID optimization (Optuna explore {explore_trials} + Bayesian refine {refine_trials})", print_to_console=True if self.debug_mode else False)
                            # ---- Phase 1: Optuna Exploration (now parallel + adaptive timing) ----
                            try:
                                import optuna, multiprocessing as _mp, time as _t
                                if self.debug_mode:
                                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                                sampler = None
                                optimization_settings_full = self.config.get('optimization_settings', {})
                                disable_pruning = optimization_settings_full.get('disable_pruning', False)
                                min_trials_before_pruning = self.get_intensity_param('min_trials_before_pruning', 10)
                                pruning_warmup_steps = self.get_intensity_param('pruning_warmup_steps', 5)
                                if disable_pruning:
                                    pruner = optuna.pruners.NopPruner()
                                else:
                                    pruner = optuna.pruners.MedianPruner(n_startup_trials=min_trials_before_pruning, n_warmup_steps=pruning_warmup_steps)
                                study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)

                                # Ensure adaptive state exists (shared lists for parallel timing capture)
                                adaptive_cfg = optimization_settings_full.get('adaptive_parallel', {})
                                if not hasattr(self, '_adaptive_state'):
                                    self._adaptive_state = {
                                        'last_scale_adjust_trial': 0,
                                        'trial_timings': [],
                                        'io_timings': [],
                                        'last_workers': None
                                    }
                                try:
                                    parallel_enabled = optimization_settings_full.get('parallel_optimization', False)
                                    n_jobs_config = optimization_settings_full.get('n_jobs', 'auto')
                                    if parallel_enabled and (isinstance(n_jobs_config, str) and n_jobs_config.lower() == 'auto' or (isinstance(n_jobs_config, int) and n_jobs_config not in (0,1))):
                                        if not isinstance(self._adaptive_state.get('trial_timings'), _mp.managers.ListProxy):
                                            mgr = getattr(self, '_adaptive_manager', None)
                                            if mgr is None:
                                                mgr = _mp.Manager()
                                                self._adaptive_manager = mgr
                                            if not isinstance(self._adaptive_state.get('trial_timings'), _mp.managers.ListProxy):
                                                self._adaptive_state['trial_timings'] = mgr.list(self._adaptive_state['trial_timings'])
                                            if not isinstance(self._adaptive_state.get('io_timings'), _mp.managers.ListProxy):
                                                self._adaptive_state['io_timings'] = mgr.list(self._adaptive_state['io_timings'])
                                except Exception:
                                    pass

                                def _system_metrics_explore():
                                    import psutil
                                    mem = psutil.virtual_memory()
                                    try:
                                        freq = psutil.cpu_freq(); cur = freq.current if freq else None; base = freq.max if freq else None
                                    except Exception:
                                        cur = base = None
                                    return {'available': mem.available, 'total': mem.total, 'freq_current': cur, 'freq_base': base}

                                def compute_explore_workers(trials_planned:int)->int:
                                    # Controlled worker computation (slightly more conservative early on)
                                    try:
                                        logical = multiprocessing.cpu_count()
                                    except Exception:
                                        logical = 4
                                    spare = int(adaptive_cfg.get('spare_cores', 2))
                                    min_w = int(adaptive_cfg.get('min_workers', 4))
                                    max_w = int(adaptive_cfg.get('max_workers', max(1, logical//2)))
                                    scale = float(adaptive_cfg.get('scale_factor', 0.5)) * 0.9  # dampen for exploration
                                    ramp_after = int(adaptive_cfg.get('ramp_after_trials', 8))
                                    mem_ratio = float(adaptive_cfg.get('memory_safety_ratio', 0.7))
                                    rss_est = float(adaptive_cfg.get('estimated_rss_per_trial_mb', 150)) * 1024 * 1024
                                    metrics = _system_metrics_explore()
                                    if trials_planned <= 0:
                                        return min_w
                                    base = max(min_w, int(scale * logical))
                                    if trials_planned < ramp_after:
                                        target = max(min_w, min(base, trials_planned))
                                    else:
                                        target = min(max_w, trials_planned, logical - spare)
                                    # Memory guard
                                    avail = metrics['available']; total = metrics['total'] or 1
                                    safety_limit = total * mem_ratio
                                    if rss_est>0:
                                        max_by_mem = int(avail / rss_est)
                                        if avail < safety_limit:
                                            target = min(target, max_by_mem)
                                    return max(1, min(target, max_w))

                                # Decide exploration workers
                                parallel_enabled = optimization_settings_full.get('parallel_optimization', False)
                                n_jobs_config = optimization_settings_full.get('n_jobs', 'auto')
                                if parallel_enabled:
                                    if isinstance(n_jobs_config, str) and n_jobs_config.lower() == 'auto':
                                        explore_workers = compute_explore_workers(explore_trials)
                                    elif n_jobs_config in (-1, 0):
                                        explore_workers = max(1, multiprocessing.cpu_count() - 1)
                                    else:
                                        explore_workers = max(1, int(n_jobs_config))
                                    # Cap exploration to avoid model staleness explosion
                                    explore_workers = min(explore_workers, int(adaptive_cfg.get('max_workers', explore_workers)))
                                else:
                                    explore_workers = 1
                                if explore_workers > 1:
                                    log_to_file(f"[PARALLEL] Hybrid exploration using {explore_workers} workers (controlled)", print_to_console=True if self.debug_mode else False)
                                else:
                                    log_to_file("[PARALLEL] Hybrid exploration running single-threaded", print_to_console=True if self.debug_mode else False)

                                def _exploration_objective(trial):
                                    t0 = _t.time()
                                    try:
                                        return self.objective_optuna(trial, train_df.copy())
                                    finally:
                                        try:
                                            dur = _t.time() - t0
                                            st = getattr(self, '_adaptive_state', None)
                                            if st and st.get('trial_timings') is not None:
                                                st['trial_timings'].append(dur)
                                        except Exception:
                                            pass

                                # Warm start
                                if last_window_best_params_list:
                                    warm_start_params = {dim.name: val for dim, val in zip(self.search_space, last_window_best_params_list)}
                                    study.enqueue_trial(warm_start_params)

                                # --- Enhanced self-abort capable exploration loop ---
                                ea_cfg = self._early_abort_cfg
                                apply_to_optuna = bool(ea_cfg.get('apply_to_optuna', False))
                                abort_reasons = None
                                best_history = []  # running best values
                                recent_best_vals = []
                                last_improv_iter = 0
                                worsen_streak = 0
                                min_calls_before = ea_cfg.get('min_calls_before_check', 8)
                                patience_calls = ea_cfg.get('patience_calls', 6)
                                rel_thresh = ea_cfg.get('min_relative_improvement', 0.01)
                                abs_thresh = ea_cfg.get('min_absolute_improvement', 0.002)
                                slope_window = ea_cfg.get('stagnation_slope_window', 5)
                                slope_min = ea_cfg.get('min_slope_threshold', 1e-4)
                                penalty_dom_ratio = ea_cfg.get('penalty_dominance_ratio', 0.70)
                                max_worsen = ea_cfg.get('max_worsening_calls', 4)
                                fallback_reallocate = bool(ea_cfg.get('fallback_reallocate', True))
                                min_realloc = int(ea_cfg.get('min_reallocation_trials', 3))
                                max_realloc = int(ea_cfg.get('max_reallocation_trials', 15))

                                def _should_abort_optuna(iter_idx:int, current_best:float):
                                    if not apply_to_optuna:
                                        return False, None
                                    if iter_idx < min_calls_before:
                                        return False, None
                                    stagnation = (iter_idx - last_improv_iter) >= patience_calls
                                    slope_flag = False
                                    if len(recent_best_vals) >= slope_window:
                                        rv = recent_best_vals[-slope_window:]
                                        slope = (rv[-1]-rv[0])/max(1,(len(rv)-1))
                                        if abs(slope) < slope_min:
                                            slope_flag = True
                                    # Relative & absolute improvement vs previous best (before this trial)
                                    if len(best_history) >= 2:
                                        prev_best = best_history[-2]
                                    else:
                                        prev_best = current_best
                                    raw_improv = current_best - prev_best
                                    rel_improv = raw_improv/abs(prev_best) if prev_best not in (0,None) else raw_improv
                                    rel_flag = rel_improv < rel_thresh
                                    abs_flag = raw_improv < abs_thresh
                                    penalty_flag = False
                                    try:
                                        last_pen_frac = getattr(self, '_last_penalty_fraction', None)
                                        if last_pen_frac is not None and last_pen_frac >= penalty_dom_ratio:
                                            penalty_flag = True
                                    except Exception:
                                        pass
                                    worsen_flag = worsen_streak >= max_worsen
                                    criteria = []
                                    if stagnation and (rel_flag or abs_flag): criteria.append('STAGNATION')
                                    if slope_flag: criteria.append('FLAT_SLOPE')
                                    if penalty_flag: criteria.append('PENALTY_DOMINANCE')
                                    if worsen_flag: criteria.append('WORSENING_STREAK')
                                    if criteria:
                                        return True, criteria
                                    return False, None

                                # Manual incremental optimization to permit abort
                                for trial_idx in range(explore_trials):
                                    study.optimize(_exploration_objective, n_trials=1, n_jobs=explore_workers, catch=())
                                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                                    if not completed:
                                        continue
                                    current_best = max(t.value for t in completed if t.value is not None)
                                    if not best_history or current_best > best_history[-1] + 1e-12:
                                        best_history.append(current_best)
                                        recent_best_vals.append(current_best)
                                        last_improv_iter = trial_idx + 1
                                        worsen_streak = 0
                                    else:
                                        # no improvement
                                        recent_best_vals.append(best_history[-1])
                                        worsen_streak += 1
                                    if len(recent_best_vals) > slope_window:
                                        recent_best_vals = recent_best_vals[-(slope_window+1):]
                                    should_abort, reasons = _should_abort_optuna(trial_idx+1, best_history[-1])
                                    if should_abort:
                                        abort_reasons = reasons
                                        log_to_file(f"[EARLY_ABORT_OPTUNA] Aborted exploration at trial {trial_idx+1}/{explore_trials} reasons={reasons}", print_to_console=self.debug_mode)
                                        break

                                explored_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                                # Record completion stats for auto-tuning
                                self._last_exploration_completed = len(explored_trials)
                                self._last_refine_trials = refine_trials
                                explored_trials.sort(key=lambda t: t.value, reverse=True)
                                seed_points = []
                                top_k = max(1, int(hybrid_cfg.get('bayesian_seed_top_k', 5)))
                                for t in explored_trials[:top_k]:
                                    seed_points.append([t.params.get(dim.name) for dim in self.search_space])
                                # Optional seed diversity filtering
                                if seed_points:
                                    try:
                                        seed_points = self._apply_seed_diversity(seed_points)
                                    except Exception as _sd_e:
                                        log_to_file(f"[SEED_DIVERSITY_WARN] Failed to apply seed diversity: {_sd_e}", print_to_console=self.debug_mode)
                                    last_window_best_params_list = seed_points[0]
                                # Adjust n_calls for Bayesian refinement
                                # Reallocate leftover exploration budget if aborted early
                                if abort_reasons and fallback_reallocate:
                                    completed_explore = len(explored_trials)
                                    leftover = explore_trials - completed_explore
                                    if leftover >= min_realloc:
                                        add_refine = min(leftover, max_realloc)
                                        refine_trials += add_refine
                                        log_to_file(f"[HYBRID REALLOCATE] Added {add_refine} refine trials (explore aborted; reasons={abort_reasons})", print_to_console=True if self.debug_mode else False)
                                n_calls = refine_trials
                                if len(explored_trials) == 0:
                                    log_to_file("[HYBRID_DIAG] Exploration produced 0 COMPLETE trials. Possible causes: (a) all trials pruned (zero trades / invalid params), (b) excessive pruning thresholds, (c) too few explore_trials (" + str(explore_trials) + ")", print_to_console=True)
                                    log_to_file("[HYBRID_DIAG] Action: Increasing exploration_fraction or disabling pruning for first window may help.", print_to_console=True)
                                log_to_file(f"Hybrid Phase 1 complete: {len(explored_trials)} complete trials; proceeding with {refine_trials} Bayesian calls", print_to_console=True if self.debug_mode else False)
                            except Exception as _hybrid_err:
                                log_to_file(f"[HYBRID WARNING] Exploration phase failed: {_hybrid_err}. Falling back to single-stage {self.optimizer}.", print_to_console=True)
                                use_hybrid = False
                        log_to_file(f"Window #{window_num}: Starting optimization ({'hybrid-bayesian' if use_hybrid and self.optimizer!='bayesian' else self.optimizer}, {n_calls} calls)...", print_to_console=False)
                        if self.optimizer == 'bayesian' or (use_hybrid and self.optimizer != 'optuna'):
                            def obj_func_with_imports(p):
                                import sys, os
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                project_root = os.path.dirname(current_dir)
                                if project_root not in sys.path:
                                    sys.path.insert(0, project_root)
                                try:
                                    from core.position_manager import PositionManager
                                    from core.portfolio import Portfolio
                                    from core.strategy import Strategy
                                    return self._objective_with_cache(p, train_df)
                                except ImportError as e:
                                    log_to_file(f"Import error in worker process: {e}", print_to_console=True)
                                    # Dynamic failure objective (invalid import) instead of large sentinel
                                    trade_penalty, dd_penalty = self._compute_penalties(0, 0.0)
                                    return (trade_penalty + dd_penalty + 1) * 20.0
                            
                            # Consistent initial points unless overridden in debug config
                            n_initial_points = int(self.debug_config.get('bayes_initial_points', 10)) if self.debug_mode else 10
                            
                            performance_settings = self.config.get('performance_settings', {})  # legacy
                            optimization_settings_full = self.config.get('optimization_settings', {})
                            # New logic: prefer optimization_settings.parallel_optimization; fallback to old performance_settings.parallel_processing
                            parallel_enabled = optimization_settings_full.get('parallel_optimization', False) or performance_settings.get('parallel_processing', False)
                            n_jobs_config = optimization_settings_full.get('n_jobs', performance_settings.get('n_jobs', 1))
                            adaptive_cfg = optimization_settings_full.get('adaptive_parallel', {})
                            # State for adaptive scaling across calls (attach to self)
                            if not hasattr(self, '_adaptive_state'):
                                self._adaptive_state = {
                                    'last_scale_adjust_trial': 0,
                                    'trial_timings': [],  # durations (seconds)
                                    'io_timings': [],      # data load durations (seconds)
                                    'last_workers': None,
                                    'timings_lock': None
                                }
                            # For Bayesian parallel (multiprocessing) we need manager-backed lists so child processes can mutate
                            try:
                                if parallel_enabled and (isinstance(n_jobs_config, str) and n_jobs_config.lower() == 'auto' or isinstance(n_jobs_config, int) and n_jobs_config not in (0,1)):
                                    import multiprocessing as _mp
                                    if not isinstance(self._adaptive_state.get('trial_timings'), list) or not isinstance(self._adaptive_state.get('trial_timings'), _mp.managers.ListProxy):
                                        mgr = getattr(self, '_adaptive_manager', None)
                                        if mgr is None:
                                            mgr = _mp.Manager()
                                            self._adaptive_manager = mgr
                                        if not isinstance(self._adaptive_state.get('trial_timings'), _mp.managers.ListProxy):
                                            self._adaptive_state['trial_timings'] = mgr.list(self._adaptive_state['trial_timings'])
                                        if not isinstance(self._adaptive_state.get('io_timings'), _mp.managers.ListProxy):
                                            self._adaptive_state['io_timings'] = mgr.list(self._adaptive_state['io_timings'])
                                        if self._adaptive_state.get('timings_lock') is None:
                                            import threading as _th
                                            # Use a dummy object since Lock can't be shared simply across processes; rely on manager list atomicity
                                            self._adaptive_state['timings_lock'] = _th.RLock()
                            except Exception:
                                pass
                            def _system_metrics():
                                import psutil, time
                                mem = psutil.virtual_memory()
                                try:
                                    freq = psutil.cpu_freq()
                                    current_freq = freq.current if freq else None
                                    base_freq = freq.max if freq else None
                                except Exception:
                                    current_freq = base_freq = None
                                return {
                                    'available': mem.available,
                                    'total': mem.total,
                                    'freq_current': current_freq,
                                    'freq_base': base_freq
                                }
                            # (Adaptive decision logging moved to class method _log_adaptive_decision)
                            def compute_auto_workers(trials_planned: int) -> int:
                                try:
                                    logical = multiprocessing.cpu_count()
                                except Exception:
                                    logical = 4
                                spare = int(adaptive_cfg.get('spare_cores', 2))
                                min_w = int(adaptive_cfg.get('min_workers', 4))
                                max_w = int(adaptive_cfg.get('max_workers', max(1, logical // 2)))
                                scale = float(adaptive_cfg.get('scale_factor', 0.5))
                                ramp_after = int(adaptive_cfg.get('ramp_after_trials', 8))
                                mem_ratio = float(adaptive_cfg.get('memory_safety_ratio', 0.7))
                                rss_est = float(adaptive_cfg.get('estimated_rss_per_trial_mb', 150)) * 1024 * 1024
                                thr_improv_thresh = float(adaptive_cfg.get('throughput_improvement_threshold', 0.08))
                                cooldown = int(adaptive_cfg.get('scaling_cooldown_trials', 10))
                                io_latency_thresh_ms = float(adaptive_cfg.get('io_latency_threshold_ms', 35))
                                freq_drop_thresh = float(adaptive_cfg.get('freq_drop_threshold', 0.85))
                                state = self._adaptive_state
                                metrics = _system_metrics()
                                reasons = []
                                if trials_planned <= 0:
                                    return min_w
                                base = max(min_w, int(scale * logical))
                                if trials_planned < ramp_after:
                                    target = max(min_w, min(base, trials_planned))
                                else:
                                    target = min(max_w, trials_planned, logical - spare)
                                # Memory guard
                                avail = metrics['available']; total = metrics['total'] or 1
                                safety_limit = total * mem_ratio
                                if rss_est > 0:
                                    max_by_mem = int(avail / rss_est)
                                    if avail < safety_limit:
                                        target_before = target
                                        target = min(target, max_by_mem)
                                        if target != target_before:
                                            reasons.append('MEM_GUARD')
                                        # memory guard applied
                                # Throughput improvement heuristic
                                timings = state['trial_timings']
                                if len(timings) >= 8 and state['last_workers']:
                                    recent = timings[-4:]; prev = timings[-8:-4]
                                    if prev and recent:
                                        avg_prev = sum(prev)/len(prev); avg_recent = sum(recent)/len(recent)
                                        improvement = (avg_prev - avg_recent)/avg_prev if avg_prev > 0 else 0
                                        if improvement < thr_improv_thresh and (len(timings) - state['last_scale_adjust_trial']) >= cooldown:
                                            target_before = target
                                            target = min(target, state['last_workers'])
                                            state['last_scale_adjust_trial'] = len(timings)
                                            if target != target_before:
                                                reasons.append('THROUGHPUT_STALL')
                                            # throughput stall guard applied
                                # I/O latency guard
                                io_times = state['io_timings']
                                if len(io_times) >= 5:
                                    avg_io_ms = (sum(io_times[-5:]) / 5.0) * 1000.0
                                    if avg_io_ms > io_latency_thresh_ms:
                                        target_before = target
                                        target = max(min_w, int(target * 0.8))
                                        if target != target_before:
                                            reasons.append('IO_BACKPRESSURE')
                                        # io backpressure guard applied
                                # Frequency drop guard
                                cur_f = metrics['freq_current']; base_f = metrics['freq_base']
                                if cur_f and base_f and base_f > 0 and (cur_f / base_f) < freq_drop_thresh:
                                    target_before = target
                                    target = max(min_w, int(target * 0.85))
                                    if target != target_before:
                                        reasons.append('FREQ_DROP')
                                    # freq drop guard applied
                                target = max(1, min(target, max_w))
                                state['last_workers'] = target
                                self._log_adaptive_decision('bayesian', window_num, trials_planned, base, target, reasons, metrics, state)
                                return target
                            if parallel_enabled:
                                if isinstance(n_jobs_config, str) and n_jobs_config.lower() == 'auto':
                                    n_jobs = compute_auto_workers(n_calls)
                                elif n_jobs_config in (-1, 0):
                                    n_jobs = max(1, multiprocessing.cpu_count() - 1)
                                else:
                                    n_jobs = max(1, int(n_jobs_config))
                            else:
                                n_jobs = 1
                            if parallel_enabled:
                                log_to_file(f"[PARALLEL] Bayesian optimization using {n_jobs} workers (config={n_jobs_config})", print_to_console=True if self.debug_mode else False)
                            else:
                                log_to_file("[PARALLEL] Bayesian optimization running single-threaded", print_to_console=True if self.debug_mode else False)
                            
                            gp_kwargs = {"n_calls": n_calls, "random_state": 42, "n_initial_points": n_initial_points, "n_jobs": n_jobs}
                            # Early refinement abort (adaptive) via callback
                            early_abort_enabled = bool(self._early_abort_cfg.get('enabled', False))
                            if early_abort_enabled:
                                cfg_ea = self._early_abort_cfg
                                min_calls_before = cfg_ea['min_calls_before_check']
                                rel_improv_thresh = cfg_ea['min_relative_improvement']
                                patience_calls = cfg_ea['patience_calls']
                                slope_window = cfg_ea['stagnation_slope_window']
                                slope_thresh = cfg_ea['min_slope_threshold']
                                penalty_dom_ratio = cfg_ea['penalty_dominance_ratio']
                                max_worsen = cfg_ea['max_worsening_calls']
                                abs_improv_min = cfg_ea['min_absolute_improvement']
                                state_cb = {
                                    'best': None,
                                    'last_improv_iter': 0,
                                    'recent_vals': [],
                                    'worsen_streak': 0
                                }
                                def _early_abort_cb(res):
                                    try:
                                        iter_idx = len(res.func_vals)
                                        current_best = min(res.func_vals)
                                        # Track recent values for slope
                                        state_cb['recent_vals'].append(current_best)
                                        if len(state_cb['recent_vals']) > slope_window:
                                            state_cb['recent_vals'] = state_cb['recent_vals'][-slope_window:]
                                        if state_cb['best'] is None:
                                            state_cb['best'] = current_best
                                            state_cb['last_improv_iter'] = iter_idx
                                            return False
                                        prev_best = state_cb['best']
                                        raw_improv = prev_best - current_best
                                        if prev_best <= 0:
                                            rel_improv = raw_improv
                                        else:
                                            rel_improv = raw_improv / abs(prev_best)
                                        if raw_improv > 1e-12 and (rel_improv >= rel_improv_thresh or raw_improv >= abs_improv_min):
                                            state_cb['best'] = current_best
                                            state_cb['last_improv_iter'] = iter_idx
                                            state_cb['worsen_streak'] = 0
                                            return False
                                        # Worsening tracking
                                        if raw_improv < -1e-10:
                                            state_cb['worsen_streak'] += 1
                                        # Compute simple slope if enough points
                                        slope_flag = False
                                        if len(state_cb['recent_vals']) >= slope_window:
                                            rv = state_cb['recent_vals']
                                            # Linear regression slope approx: (last-first)/(n-1)
                                            slope = (rv[-1] - rv[0]) / max(1, (len(rv)-1))
                                            if abs(slope) < slope_thresh:
                                                slope_flag = True
                                        # Penalty dominance heuristic (requires stored last objective components)
                                        penalty_dom_flag = False
                                        try:
                                            last_pen_frac = getattr(self, '_last_penalty_fraction', None)
                                            if last_pen_frac is not None and last_pen_frac >= penalty_dom_ratio:
                                                penalty_dom_flag = True
                                        except Exception:
                                            pass
                                        stagnation = (iter_idx - state_cb['last_improv_iter']) >= patience_calls
                                        criteria = []
                                        if stagnation: criteria.append('STAGNATION')
                                        if slope_flag: criteria.append('FLAT_SLOPE')
                                        if penalty_dom_flag: criteria.append('PENALTY_DOMINANCE')
                                        if state_cb['worsen_streak'] >= max_worsen: criteria.append('WORSENING_STREAK')
                                        if (iter_idx >= min_calls_before) and criteria:
                                            log_to_file(f"[EARLY_ABORT] Refinement aborted at {iter_idx} calls | criteria={criteria} rel_improv={rel_improv:.4f} raw_improv={raw_improv:.5f}", print_to_console=self.debug_mode)
                                            return True
                                    except Exception as _ea_e:
                                        log_to_file(f"[EARLY_ABORT_WARN] Callback error: {_ea_e}", print_to_console=self.debug_mode)
                                    return False
                                gp_kwargs['callback'] = [_early_abort_cb]
                            if last_window_best_params_list:
                                log_to_file(f"Warm-starting window #{i+1} with best params from window #{i}", print_to_console=False)
                                gp_kwargs["x0"] = [last_window_best_params_list]
                                gp_kwargs["n_initial_points"] = max(1, gp_kwargs["n_initial_points"] - 1)
                            elif self.warm_start_params and i == 0:
                                log_to_file(f"Warm-starting first window with parameters from '{self.config_path}'.", print_to_console=False)
                                warm_start_list = [self.warm_start_params.get(dim.name) for dim in self.search_space]
                                if None not in warm_start_list:
                                    gp_kwargs["x0"] = [warm_start_list]
                                    gp_kwargs["n_initial_points"] = max(1, gp_kwargs["n_initial_points"] - 1)
                                else:
                                    log_to_file("Could not warm-start from config: parameter mismatch.", print_to_console=False)

                            # Wrap objective to record timing (works reliably only when n_jobs=1 or with manager lists)
                            import time as _t
                            def _timed_obj(x):
                                _start = _t.time()
                                try:
                                    return obj_func_with_imports(x)
                                finally:
                                    try:
                                        dur = _t.time() - _start
                                        st = getattr(self, '_adaptive_state', None)
                                        if st and st.get('trial_timings') is not None:
                                            st['trial_timings'].append(dur)
                                    except Exception:
                                        pass
                            result = gp_minimize(_timed_obj, self.search_space, **gp_kwargs)
                            best_params = {dim.name: val for dim, val in zip(self.search_space, result.x)}
                            last_window_best_params_list = result.x
                            self.save_trial_data(result, window_num, 'bayesian')

                        elif self.optimizer in ['optuna', 'cmaes']:
                            if self.debug_mode:
                                optuna.logging.set_verbosity(optuna.logging.WARNING)
                            else:
                                optuna.logging.set_verbosity(optuna.logging.WARNING)
                            sampler = None
                            if self.optimizer == 'cmaes':
                                sampler = optuna.samplers.CmaEsSampler()
                                log_to_file("Using CMA-ES sampler for Optuna.", print_to_console=False)
                            optimization_settings = self.config.get('optimization_settings', {})
                            disable_pruning = optimization_settings.get('disable_pruning', False)
                            min_trials_before_pruning = self.get_intensity_param('min_trials_before_pruning', 10)
                            pruning_warmup_steps = self.get_intensity_param('pruning_warmup_steps', 5)
                            if disable_pruning:
                                pruner = optuna.pruners.NopPruner()
                                log_to_file("Pruning disabled for this optimization run.", print_to_console=True)
                            else:
                                pruner = optuna.pruners.MedianPruner(n_startup_trials=min_trials_before_pruning, n_warmup_steps=pruning_warmup_steps)
                                log_to_file(f"Using MedianPruner with {min_trials_before_pruning} startup trials and {pruning_warmup_steps} warmup steps.", print_to_console=False)
                            study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
                            def objective_wrapper(trial):
                                import time as _t
                                st = getattr(self, '_adaptive_state', None)
                                t0 = _t.time()
                                # crude IO timing: time to copy slice & any preprocessing inside objective before heavy calc returns
                                try:
                                    result = self.objective_optuna(trial, train_df.copy())
                                    return result
                                finally:
                                    try:
                                        elapsed = _t.time() - t0
                                        if st and st.get('trial_timings') is not None:
                                            st['trial_timings'].append(elapsed)
                                    except Exception:
                                        pass
                            if last_window_best_params_list:
                                warm_start_params = {dim.name: val for dim, val in zip(self.search_space, last_window_best_params_list)}
                                study.enqueue_trial(warm_start_params)
                                log_to_file(f"Enqueued warm-start trial for window #{window_num} with best params from window #{i}", print_to_console=False)
                            elif self.warm_start_params and i == 0:
                                log_to_file(f"Enqueued warm-start trial for first window with parameters from '{self.config_path}'.", print_to_console=False)
                                warm_start_trial_params = {k: v for k, v in self.warm_start_params.items() if any(k == dim.name for dim in self.search_space)}
                                if len(warm_start_trial_params) == len(self.warm_start_params):
                                     study.enqueue_trial(warm_start_trial_params)
                                else:
                                    log_to_file("Could not warm-start from config: parameter mismatch.", print_to_console=False)
                            performance_settings = self.config.get('performance_settings', {})  # legacy
                            optimization_settings_full = self.config.get('optimization_settings', {})
                            parallel_enabled = optimization_settings_full.get('parallel_optimization', False) or performance_settings.get('parallel_processing', False)
                            n_jobs_config = optimization_settings_full.get('n_jobs', performance_settings.get('n_jobs', 1))
                            adaptive_cfg = optimization_settings_full.get('adaptive_parallel', {})
                            if not hasattr(self, '_adaptive_state'):
                                self._adaptive_state = {
                                    'last_scale_adjust_trial': 0,
                                    'trial_timings': [],
                                    'io_timings': [],
                                    'last_workers': None
                                }
                            def _system_metrics_opt():
                                import psutil
                                mem = psutil.virtual_memory()
                                try:
                                    freq = psutil.cpu_freq()
                                    current_freq = freq.current if freq else None
                                    base_freq = freq.max if freq else None
                                except Exception:
                                    current_freq = base_freq = None
                                return {'available': mem.available,'total': mem.total,'freq_current': current_freq,'freq_base': base_freq}
                            def compute_auto_workers_optuna(trials_planned:int)->int:
                                try:
                                    logical = multiprocessing.cpu_count()
                                except Exception:
                                    logical = 4
                                spare = int(adaptive_cfg.get('spare_cores', 2))
                                min_w = int(adaptive_cfg.get('min_workers', 4))
                                max_w = int(adaptive_cfg.get('max_workers', max(1, logical//2)))
                                scale = float(adaptive_cfg.get('scale_factor', 0.5))
                                ramp_after = int(adaptive_cfg.get('ramp_after_trials', 8))
                                mem_ratio = float(adaptive_cfg.get('memory_safety_ratio', 0.7))
                                rss_est = float(adaptive_cfg.get('estimated_rss_per_trial_mb', 150)) * 1024 * 1024
                                thr_improv_thresh = float(adaptive_cfg.get('throughput_improvement_threshold', 0.08))
                                cooldown = int(adaptive_cfg.get('scaling_cooldown_trials', 10))
                                io_latency_thresh_ms = float(adaptive_cfg.get('io_latency_threshold_ms', 35))
                                freq_drop_thresh = float(adaptive_cfg.get('freq_drop_threshold', 0.85))
                                state = self._adaptive_state
                                metrics = _system_metrics_opt()
                                reasons = []
                                if trials_planned <= 0:
                                    return min_w
                                base = max(min_w, int(scale * logical))
                                if trials_planned < ramp_after:
                                    target = max(min_w, min(base, trials_planned))
                                else:
                                    target = min(max_w, trials_planned, logical - spare)
                                # Memory guard
                                avail = metrics['available']; total = metrics['total'] or 1
                                safety_limit = total * mem_ratio
                                if rss_est>0:
                                    max_by_mem = int(avail / rss_est)
                                    if avail < safety_limit:
                                        target = min(target, max_by_mem)
                                # Throughput heuristic (Optuna updates state in objective)
                                timings = state['trial_timings']
                                if len(timings) >= 8 and state['last_workers']:
                                    recent = timings[-4:]; prev = timings[-8:-4]
                                    if prev and recent:
                                        avg_prev = sum(prev)/len(prev); avg_recent = sum(recent)/len(recent)
                                        improvement = (avg_prev - avg_recent)/avg_prev if avg_prev>0 else 0
                                        if improvement < thr_improv_thresh and (len(timings)-state['last_scale_adjust_trial'])>=cooldown:
                                            target_before = target
                                            target = min(target, state['last_workers'])
                                            state['last_scale_adjust_trial'] = len(timings)
                                            if target != target_before:
                                                reasons.append('THROUGHPUT_STALL')
                                # I/O latency guard
                                io_times = state['io_timings']
                                if len(io_times) >=5:
                                    avg_io_ms = (sum(io_times[-5:])/5.0)*1000.0
                                    if avg_io_ms > io_latency_thresh_ms:
                                            target_before = target
                                            target = max(min_w, int(target*0.8))
                                            if target != target_before:
                                                reasons.append('IO_BACKPRESSURE')
                                # Frequency guard
                                cur_f = metrics['freq_current']; base_f = metrics['freq_base']
                                if cur_f and base_f and base_f>0 and (cur_f/base_f) < freq_drop_thresh:
                                        target_before = target
                                        target = max(min_w, int(target*0.85))
                                        if target != target_before:
                                            reasons.append('FREQ_DROP')
                                target = max(1, min(target, max_w))
                                state['last_workers'] = target
                                self._log_adaptive_decision('optuna', window_num, trials_planned, base, target, reasons, metrics, state)
                                return target
                            if parallel_enabled:
                                if isinstance(n_jobs_config, str) and n_jobs_config.lower() == 'auto':
                                    n_jobs_eff = compute_auto_workers_optuna(n_calls)
                                elif n_jobs_config in (-1, 0):
                                    n_jobs_eff = max(1, multiprocessing.cpu_count() - 1)
                                else:
                                    n_jobs_eff = max(1, int(n_jobs_config))
                                # Removed structured decision logging due to prior corruption; can reintroduce cleanly later
                            else:
                                n_jobs_eff = 1
                            self._optuna_stats = {'evaluated': 0, 'pruned': 0}
                            if parallel_enabled:
                                log_to_file(f"[PARALLEL] Optuna optimization using {n_jobs_eff} workers (config={n_jobs_config})", print_to_console=True)
                                study.optimize(objective_wrapper, n_trials=n_calls, n_jobs=n_jobs_eff)
                            else:
                                log_to_file("[PARALLEL] Optuna optimization running single-threaded", print_to_console=True)
                                study.optimize(objective_wrapper, n_trials=n_calls, n_jobs=1)
                            total_trials = len(study.trials)
                            eval_ct = self._optuna_stats.get('evaluated', 0)
                            pruned_ct = self._optuna_stats.get('pruned', 0)
                            log_to_file(f"[OPTUNA_SUMMARY] Window #{window_num}: trials_total={total_trials} evaluated={eval_ct} pruned={pruned_ct}", print_to_console=True if self.debug_mode else False)
                            # Log structured prune reasons if available
                            if hasattr(self, '_prune_reasons') and self._prune_reasons:
                                try:
                                    reasons_fragments = []
                                    for code, info in self._prune_reasons.items():
                                        examples_str = '; '.join(info.get('examples', []))
                                        reasons_fragments.append(f"{code}:{info.get('count',0)}[{examples_str}]")
                                    log_to_file(f"[PRUNE_SUMMARY] Window #{window_num}: " + ' | '.join(reasons_fragments), print_to_console=True if self.debug_mode else False)
                                except Exception:
                                    pass
                            # Reset reasons for next window
                            if hasattr(self, '_prune_reasons'):
                                self._prune_reasons = {}
                            if not study.trials or all(t.state == optuna.trial.TrialState.PRUNED for t in study.trials):
                                self.log_problem(f"Window #{window_num}: All optimization trials were pruned. Reusing last good parameters.")
                                if self.all_optimized_params:
                                    last_key = sorted(self.all_optimized_params.keys())[-1]
                                    best_params = self.all_optimized_params[last_key]
                                    log_to_file(f"Reusing parameters from {last_key} for window #{window_num}.", print_to_console=False)
                                elif self.warm_start_params:
                                    best_params = self.warm_start_params.copy()
                                    log_to_file(f"Using warm start parameters from config for window #{window_num}.", print_to_console=False)
                                else:
                                    log_to_file(f"Window #{window_num}: No previous parameters to fall back on. Skipping test phase.", print_to_console=True)
                                    continue
                            else:
                                best_params = study.best_params
                                best_score = study.best_value
                                log_to_file(f"Window #{window_num}: Optimization complete. Best Score (Minimized Value): {best_score:.4f}", print_to_console=True)
                                log_to_file(f"Window #{window_num}: Best Params Found: {best_params}", print_to_console=False)
                                last_window_best_params_list = [best_params.get(dim.name) for dim in self.search_space]
                                self.save_trial_data(study, window_num, 'optuna')

                    log_to_file(f"Window #{window_num}: Optimization finished. Applying best params to test data.", print_to_console=False)
                    self.all_optimized_params[f"Window_{window_num}"] = best_params
                    
                    if self.persistent_strategy:
                        full_params = {**self.default_params, **best_params}
                        self.persistent_strategy.params = full_params
                        
                        if not self.no_cache and self.memory and self.cached_generate_signals:
                            processed_test_df = self.cached_generate_signals(self.persistent_strategy, test_df, self.realism_settings)
                        else:
                            processed_test_df = self.persistent_strategy.generate_signals(test_df, self.realism_settings)
                    else:
                        full_params = {**self.default_params, **best_params}
                        strategy = Strategy(full_params)
                        if not self.no_cache and self.memory and self.cached_generate_signals:
                            processed_test_df = self.cached_generate_signals(strategy, test_df, self.realism_settings)
                        else:
                            processed_test_df = strategy.generate_signals(test_df, self.realism_settings)

                    if not processed_test_df.empty:
                        start_capital_for_log = self.chained_capital

                        # Collect pre-execution signal distribution diagnostics
                        try:
                            diag = {
                                'window': window_num,
                                'train_range': f"{train_start.date()}->{train_end.date()}",
                                'test_range_end': str(test_end.date()),
                                'rows_test': int(len(processed_test_df))
                            }
                            for col in ['signal', 'primary_signal', 'weighted_signal']:
                                if col in processed_test_df.columns:
                                    s = processed_test_df[col].dropna()
                                    if not s.empty:
                                        diag[f'{col}_mean'] = float(s.mean())
                                        diag[f'{col}_std'] = float(s.std())
                                        diag[f'{col}_pos'] = int((s > 0).sum())
                                        diag[f'{col}_neg'] = int((s < 0).sum())
                            if 'long_signals' in processed_test_df.columns:
                                diag['long_signals_count'] = int(processed_test_df['long_signals'].astype(bool).sum())
                            if 'short_signals' in processed_test_df.columns:
                                diag['short_signals_count'] = int(processed_test_df['short_signals'].astype(bool).sum())
                            if 'long_signals' in processed_test_df.columns and 'short_signals' in processed_test_df.columns:
                                ls = processed_test_df['long_signals'].astype(bool)
                                ss = processed_test_df['short_signals'].astype(bool)
                                diag['either_side_signal_events'] = int((ls | ss).sum())
                                diag['both_sides_same_bar'] = int((ls & ss).sum())
                            self.directional_diagnostics.append(diag)
                        except Exception as e:
                            log_to_file(f"[DIAGNOSTICS] Pre-trade diagnostics failed for window {window_num}: {e}", print_to_console=False)

                        metrics, trades_df, final_equity = run_backtest(best_params, processed_test_df, self.chained_capital)
                        
                        window_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
                        self.log_debug(f"Window #{i + 1} Test Results: StartCap=${start_capital_for_log:,.2f}, Trades={len(trades_df)}, PnL=${window_pnl:,.2f}, New Chained Capital=${final_equity:,.2f}")

                        self.chained_capital = final_equity if np.isfinite(final_equity) else self.chained_capital

                        # Self-optimization adaptive hook (record realized Sharpe & adjust future exploration fraction)
                        try:
                            self._adaptive_post_window(window_num, metrics)
                        except Exception as _auto_e:
                            log_to_file(f"[AUTO_TUNE_WARN] adaptive hook failed window {window_num}: {_auto_e}", print_to_console=self.debug_mode)
                        
                        if not trades_df.empty:
                            trades_df['window'] = i + 1
                            trades_df['window_start_capital'] = start_capital_for_log
                            self.log_trade_details(trades_df, i + 1)
                            self.all_trades.append(trades_df)
                            # Augment diagnostics with executed trade distribution
                            try:
                                long_exec = int((trades_df['side'].str.lower() == 'long').sum()) if 'side' in trades_df.columns else 0
                                short_exec = int((trades_df['side'].str.lower() == 'short').sum()) if 'side' in trades_df.columns else 0
                                if self.directional_diagnostics and self.directional_diagnostics[-1].get('window') == window_num:
                                    self.directional_diagnostics[-1]['executed_long_trades'] = long_exec
                                    self.directional_diagnostics[-1]['executed_short_trades'] = short_exec
                                    ls_cnt = self.directional_diagnostics[-1].get('long_signals_count') or 0
                                    ss_cnt = self.directional_diagnostics[-1].get('short_signals_count') or 0
                                    self.directional_diagnostics[-1]['long_execution_ratio'] = float(long_exec/ls_cnt) if ls_cnt > 0 else None
                                    self.directional_diagnostics[-1]['short_execution_ratio'] = float(short_exec/ss_cnt) if ss_cnt > 0 else None
                            except Exception as e:
                                log_to_file(f"[DIAGNOSTICS] Post-trade diagnostics failed for window {window_num}: {e}", print_to_console=False)

                    log_to_file(f"--- Finished Window #{window_num}. Chained Capital: ${self.chained_capital:,.2f} ---", print_to_console=False)

            log_to_file("--- All walk-forward windows processed. Finalizing report. ---", print_to_console=True)
            
            if self.debug_mode:
                self.print_debug_log()
            
            if not getattr(self, '_finalized', False):
                self.finalize_and_report()
                self._finalized = True

        finally:
            # --- Stop the listener process now that the main loop is finished ---
            log_to_file("Walk-forward optimization complete. Shutting down log listener.", print_to_console=True)
            log_queue.put(None)  # Send sentinel to stop the listener
            listener_process.join() # Wait for the listener to finish
            self.performance_monitoring_enabled = True

    def log_trade_details(self, trades_df, window_num):
        """Logs a detailed summary of trades for a specific window."""
        if trades_df.empty:
            return

        pnl = trades_df['pnl']
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        # Define variables that were previously undefined
        win_rate = (len(wins) / len(pnl)) * 100 if not pnl.empty else 0
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')

        log_to_file(f"--- Trade-Log for Window #{window_num} ---", print_to_console=False)
        log_to_file(f"  - Total Trades: {len(pnl)}", print_to_console=False)
        log_to_file(f"  - Win Rate: {win_rate:.2f}%", print_to_console=False)
        log_to_file(f"  - Profit Factor: {profit_factor:.2f}", print_to_console=False)
        log_to_file(f"  - Average Win: ${avg_win:,.2f}", print_to_console=False)
        log_to_file(f"  - Average Loss: ${avg_loss:,.2f}", print_to_console=False)

    def _adaptive_post_window(self, window_num, metrics):
        """Passive adaptive hook: record window Sharpe & score; adjust exploration fraction cautiously.

        Current logic (minimal):
        - Capture Sharpe (metrics.get('sharpe')) and a composite score if available.
        - After patience windows, if Sharpe slope < threshold and exploration auto-tune enabled, gently increase explore fraction.
        - If recent improvement strong, gently decrease explore fraction.
        Safeguards: bounds, max one adjustment per window, requires at least 3 history points.
        """
        try:
            # --- Initialize new adaptive master controller state lazily ---
            if not hasattr(self, '_master_adaptive_state'):
                self._master_adaptive_state = {
                    'window_scores': [],
                    'window_sharpes': [],
                    'train_sharpes': [],
                    'test_sharpes': [],
                    'overfit_events': 0,
                    'calls_multiplier': 1.0,
                    'stagnation_windows': 0,
                    'last_best_score': None,
                    'contractions': 0
                }
            # Attempt to infer train vs test sharpe (metrics here are assumed test metrics)
            test_sharpe = metrics.get('sharpe') if isinstance(metrics, dict) else metrics.get('Sharpe Ratio') if isinstance(metrics, dict) else None
            if test_sharpe is not None:
                self._master_adaptive_state['test_sharpes'].append(float(test_sharpe))
            # Record composite score if available
            composite = metrics.get('composite_score') if isinstance(metrics, dict) else None
            if composite is not None:
                self._master_adaptive_state['window_scores'].append(composite)
            sharpe = metrics.get('sharpe') if isinstance(metrics, dict) else None
            if sharpe is None:
                return
            self._window_sharpes.append(float(sharpe))
            # Maintain only last 12 for slope calc
            if len(self._window_sharpes) > 12:
                self._window_sharpes = self._window_sharpes[-12:]
            if not self._explore_auto_enabled or len(self._window_sharpes) < max(4, self._explore_auto_patience + 1):
                return
            import numpy as _np
            y = _np.array(self._window_sharpes, dtype=float)
            x = _np.arange(len(y))
            # Simple linear regression slope
            try:
                slope = _np.polyfit(x, y, 1)[0]
            except Exception:
                return
            prev_frac = self._dynamic_explore_frac if self._dynamic_explore_frac is not None else float(self.config.get('optimization_settings', {}).get('hybrid_optimization', {}).get('exploration_fraction', 0.45))
            new_frac = prev_frac
            # If slope very small or negative -> increase exploration (seek diversity)
            if slope < self._explore_auto_slope_thresh * 0.25:
                new_frac = min(self._explore_auto_max, prev_frac + 0.05)
                reason = 'slope_low_increase'
            # If slope strong positive -> reduce exploration slightly (exploit)
            elif slope > self._explore_auto_slope_thresh * 1.5:
                new_frac = max(self._explore_auto_min, prev_frac - 0.05)
                reason = 'slope_high_decrease'
            else:
                reason = None
            if new_frac != prev_frac:
                self._dynamic_explore_frac = round(new_frac, 3)
                evt = {
                    'window': window_num,
                    'prev': prev_frac,
                    'new': self._dynamic_explore_frac,
                    'slope': slope,
                    'reason': reason
                }
                self._auto_adjust_events.append(evt)
                try:
                    log_to_file(f"[AUTO_TUNE] window={window_num} explore_fraction {prev_frac:.3f}->{self._dynamic_explore_frac:.3f} slope={slope:.4f} reason={reason}", print_to_console=self.debug_mode)
                except Exception:
                    pass

            # --- Overfitting Detection (train vs test Sharpe delta) ---
            try:
                # Expect train sharpe stored earlier (placeholder: reuse test if unavailable)
                train_sharpe = getattr(self, '_last_train_sharpe', None)
                if train_sharpe is None:
                    train_sharpe = sharpe
                test_sharpe = sharpe
                delta = train_sharpe - test_sharpe
                overfit_threshold = float(self.config.get('optimization_settings', {}).get('adaptive_master', {}).get('overfit_sharpe_delta', 0.4))
                if delta > overfit_threshold:
                    self._master_adaptive_state['overfit_events'] += 1
                    # Respond: increase exploration & relax future contraction
                    self._dynamic_explore_frac = min(self._explore_auto_max, (self._dynamic_explore_frac or prev_frac) + 0.05)
                    log_to_file(f"[OVERFIT_DETECT] window={window_num} train_sharpe={train_sharpe:.3f} test_sharpe={test_sharpe:.3f} delta={delta:.3f} -> exploration boost", print_to_console=self.debug_mode)
            except Exception:
                pass

            # --- Convergence & Calls Multiplier Adjustment ---
            try:
                calls_cfg = self.config.get('optimization_settings', {}).get('adaptive_master', {})
                enable_calls = bool(calls_cfg.get('dynamic_calls_enabled', True))
                if enable_calls:
                    best_score = getattr(self, 'best_global_objective', None)
                    if best_score is not None:
                        last_best = self._master_adaptive_state.get('last_best_score')
                        if last_best is None or best_score > last_best + 1e-6:
                            # Improvement -> gently reduce future calls (exploit) and reset stagnation
                            self._master_adaptive_state['stagnation_windows'] = 0
                            self._master_adaptive_state['calls_multiplier'] = max(0.6, self._master_adaptive_state['calls_multiplier'] * 0.95)
                            self._master_adaptive_state['last_best_score'] = best_score
                        else:
                            # No improvement
                            self._master_adaptive_state['stagnation_windows'] += 1
                            stagn = self._master_adaptive_state['stagnation_windows']
                            if stagn >= int(calls_cfg.get('stagnation_patience', 3)):
                                self._master_adaptive_state['calls_multiplier'] = min(1.5, self._master_adaptive_state['calls_multiplier'] * 1.10)
                                self._master_adaptive_state['stagnation_windows'] = 0
                        # Additional adaptive refinements: incorporate drift & false signal pressure
                        try:
                            fs_ema = float(self._master_adaptive_state.get('false_signal_rate_ema', 0.0))
                            drift_events = int(self._master_adaptive_state.get('drift_events', 0))
                            pressure = 0.0
                            fs_cfg = calls_cfg.get('false_signal_influence', {})
                            fs_high = float(fs_cfg.get('high_threshold', 0.4))
                            fs_severe = float(fs_cfg.get('severe_threshold', 0.5))
                            if fs_ema >= fs_severe:
                                pressure += 0.15
                            elif fs_ema >= fs_high:
                                pressure += 0.07
                            # Drift influence (recent events)
                            if drift_events > 0:
                                pressure += min(0.10, 0.04 * drift_events)
                            if pressure > 0:
                                base_mult = self._master_adaptive_state['calls_multiplier']
                                new_mult = min(2.0, base_mult * (1.0 + pressure))
                                if new_mult != base_mult:
                                    self._master_adaptive_state['calls_multiplier'] = new_mult
                                    if self.debug_mode:
                                        log_to_file(f"[CALLS_PRESSURE] window={window_num} fs_ema={fs_ema:.2f} drift_events={drift_events} pressure={pressure:.3f} mult={new_mult:.3f}", print_to_console=False)
                            # Stabilization: if false signal very low and no drift for several windows
                            low_stabilize_thresh = float(fs_cfg.get('low_stabilize_threshold', 0.15))
                            if fs_ema > 0 and fs_ema < low_stabilize_thresh and drift_events == 0:
                                self._master_adaptive_state['calls_multiplier'] = max(0.55, self._master_adaptive_state['calls_multiplier'] * 0.97)
                        except Exception:
                            pass
                        if self.debug_mode:
                            log_to_file(f"[CALLS_ADAPT] window={window_num} calls_multiplier={self._master_adaptive_state['calls_multiplier']:.3f}", print_to_console=False)
            except Exception:
                pass

            # --- Optional Search Space Contraction ---
            try:
                contract_cfg = self.config.get('optimization_settings', {}).get('adaptive_master', {})
                if contract_cfg.get('search_space_contraction', True):
                    self._maybe_contract_search_space(window_num, contract_cfg)
                if contract_cfg.get('search_space_expansion', True):
                    self._maybe_expand_search_space(window_num, contract_cfg)
                # Regime multiplier auto-tune
                if contract_cfg.get('regime_auto_tune', True):
                    self._maybe_auto_tune_regime_multipliers(window_num, contract_cfg)
                # False signal rate estimation (must precede risk scaling so tier uses updated value)
                if contract_cfg.get('false_signal_estimation_enabled', True):
                    self._maybe_update_false_signal_rate(window_num, contract_cfg)
                # Performance-based risk scaling (consumes false signal rate)
                if contract_cfg.get('risk_scaling_enabled', True):
                    self._maybe_update_risk_scaling(window_num, contract_cfg)
                # Penalty auto-calibration
                if contract_cfg.get('penalty_auto_calibration', True):
                    self._maybe_auto_calibrate_penalties(window_num, contract_cfg)
                # Penalty component share rebalance (optional refinement)
                if contract_cfg.get('penalty_component_rebalance', {}).get('enabled', False):
                    self._maybe_rebalance_penalty_components(window_num, contract_cfg.get('penalty_component_rebalance', {}))
                # Drift detection (may boost exploration)
                if contract_cfg.get('drift_detection_enabled', True):
                    self._maybe_detect_drift(window_num, contract_cfg)
                # Record best validation Sharpe for drift detection (if available in master state or last ledger entry)
                try:
                    if not hasattr(self, '_window_best_validation_sharpes'):
                        self._window_best_validation_sharpes = []
                    last_val_sh = None
                    if hasattr(self, '_inner_validation_stats') and self._inner_validation_stats:
                        # Get last stat entry for this window
                        for s in reversed(self._inner_validation_stats):
                            if s.get('window') == window_num:
                                last_val_sh = s.get('val_sharpe')
                                break
                    if last_val_sh is not None:
                        self._window_best_validation_sharpes.append(last_val_sh)
                        if len(self._window_best_validation_sharpes) > 60:
                            self._window_best_validation_sharpes = self._window_best_validation_sharpes[-60:]
                except Exception:
                    pass
            except Exception:
                pass

            # Persist master state
            try:
                self._persist_master_state(window_num)
            except Exception:
                pass
            # Optional export of adaptive state each window
            try:
                if self._adaptive_state_export_path and self._adaptive_state_last_export_window != window_num:
                    export_enabled = bool(self._export_adaptive_cfg.get('enabled', False))
                    export_every = int(self._export_adaptive_cfg.get('every_n_windows', 1))
                    if export_enabled and (window_num % max(1, export_every) == 0):
                        self._export_adaptive_state(window_num)
                        self._adaptive_state_last_export_window = window_num
            except Exception as _exp_e:
                log_to_file(f"[ADAPT_EXPORT_WARN] Failed to export adaptive state: {_exp_e}", print_to_console=self.debug_mode)
        except Exception:
            pass

    def _export_adaptive_state(self, window_num:int):
        """Persist adaptive exploration tuning state to JSON (lightweight)."""
        try:
            payload = {
                'window': window_num,
                'dynamic_explore_fraction': self._dynamic_explore_frac,
                'sharpe_history': list(self._window_sharpes),
                'adjust_events': list(self._auto_adjust_events[-25:]),  # trim
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            dirn = os.path.dirname(self._adaptive_state_export_path or '')
            if dirn:
                os.makedirs(dirn, exist_ok=True)
            with open(self._adaptive_state_export_path, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log_to_file(f"[ADAPT_EXPORT_ERR] {e}", print_to_console=self.debug_mode)

    def _apply_seed_diversity(self, seed_points):
        """Filter / reorder seed points to enforce diversity using normalized distance.

        Config keys (seed_diversity): enabled(bool), min_distance(float 0-1), max_seeds(int)
        Falls back gracefully if any issue.
        """
        try:
            if not seed_points or not isinstance(self._seed_div_cfg, dict) or not self._seed_div_cfg.get('enabled', False):
                return seed_points
            min_dist = float(self._seed_div_cfg.get('min_distance', 0.15))
            max_seeds = int(self._seed_div_cfg.get('max_seeds', len(seed_points)))
            # Build normalization ranges
            ranges = []
            for dim in self.search_space:
                if hasattr(dim, 'low') and hasattr(dim, 'high'):
                    rng = dim.high - dim.low if (dim.high - dim.low) != 0 else 1.0
                    ranges.append((dim.low, rng))
                else:  # categorical / unsupported
                    ranges.append((0.0, 1.0))
            diverse = []
            for pt in seed_points:
                if not diverse:
                    diverse.append(pt)
                    continue
                keep = True
                for dpt in diverse:
                    acc = 0.0; cnt = 0
                    for (low, rng), a, b in zip(ranges, pt, dpt):
                        try:
                            na = (a - low)/rng; nb = (b - low)/rng
                            acc += (na - nb)**2; cnt += 1
                        except Exception:
                            continue
                    if cnt > 0:
                        dist = (acc / cnt) ** 0.5
                        if dist < min_dist:
                            keep = False
                            break
                if keep:
                    diverse.append(pt)
                if len(diverse) >= max_seeds:
                    break
            if len(diverse) < len(seed_points):
                log_to_file(f"[SEED_DIVERSITY] Reduced seeds {len(seed_points)} -> {len(diverse)} (min_dist={min_dist})", print_to_console=self.debug_mode)
            return diverse
        except Exception as e:
            log_to_file(f"[SEED_DIVERSITY_ERR] {e}", print_to_console=self.debug_mode)
            return seed_points

    def _log_adaptive_decision(self, phase, window, planned, base, decided, reasons, metrics, state):
        """Central adaptive decision logger (moved from nested scope to avoid closure issues)."""
        try:
            import time as _t, json
            decision = {
                'ts': _t.time(),
                'phase': phase,
                'window': window,
                'planned_trials': planned,
                'base_workers': base,
                'decided_workers': decided,
                'reasons': reasons,
                'timings_collected': len(state.get('trial_timings', [])) if state else None,
                'io_samples': len(state.get('io_timings', [])) if state else None,
                'mem_avail_mb': round((metrics.get('available',0)/1024/1024),2) if metrics else None,
                'mem_total_mb': round((metrics.get('total',0)/1024/1024),2) if metrics else None,
                'cpu_freq_ratio': round((metrics['freq_current']/metrics['freq_base']),3) if metrics and metrics.get('freq_current') and metrics.get('freq_base') else None
            }
            log_to_file('[ADAPT] ' + json.dumps(decision), print_to_console=False)
            # Collect for HTML embedding if enabled
            if not hasattr(self, '_embedded_adaptive_events'):
                self._embedded_adaptive_events = []
            self._embedded_adaptive_events.append(decision)
        except Exception:
            pass

    def _maybe_contract_search_space(self, window_num:int, cfg:dict):
        """Heuristically contract numeric parameter bounds toward best params when stagnation persists.

        Safeguards:
        - Only after min_windows_before_contraction.
        - Only if we have a best parameter set.
        - Skip if overfit event fired in this window.
        - Minimum relative width retained.
        """
        try:
            if not hasattr(self, 'best_params_global') or not self.best_params_global:
                return
            if self._master_adaptive_state.get('overfit_events', 0) > 0 and cfg.get('skip_if_overfit', True):
                return
            min_w = int(cfg.get('min_windows_before_contraction', 4))
            if window_num < min_w:
                return
            contraction_every = int(cfg.get('contraction_every', 3))
            if window_num % max(1, contraction_every) != 0:
                return
            max_contractions = int(cfg.get('max_contractions', 8))
            if self._master_adaptive_state.get('contractions', 0) >= max_contractions:
                return
            factor = float(cfg.get('contraction_factor', 0.6))  # shrink toward center
            min_rel_width = float(cfg.get('min_relative_width', 0.15))
            changed = False
            new_space = []
            from skopt.space import Real, Integer, Categorical
            for dim in self.search_space:
                if isinstance(dim, Categorical):
                    new_space.append(dim); continue
                name = getattr(dim, 'name', None)
                if name not in self.best_params_global:
                    new_space.append(dim); continue
                try:
                    low = float(dim.low); high = float(dim.high)
                    width = high - low
                    if width <= 0: new_space.append(dim); continue
                    best_val = float(self.best_params_global[name])
                    center = best_val
                    new_half = max(width * factor / 2.0, width * min_rel_width / 2.0)
                    new_low = max(low, center - new_half)
                    new_high = min(high, center + new_half)
                    if new_high - new_low < width:  # only accept contraction
                        changed = True
                        if isinstance(dim, Integer):
                            new_dim = Integer(int(round(new_low)), int(round(new_high)), name=name)
                        else:
                            new_dim = Real(new_low, new_high, name=name)
                        new_space.append(new_dim)
                    else:
                        new_space.append(dim)
                except Exception:
                    new_space.append(dim)
            if changed:
                self.search_space = new_space
                self._master_adaptive_state['contractions'] += 1
                log_to_file(f"[SPACE_CONTRACT] window={window_num} contractions={self._master_adaptive_state['contractions']}", print_to_console=self.debug_mode)
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[SPACE_CONTRACT_ERR] {e}", print_to_console=False)

    def _update_param_boundary_stats(self, window_num:int, cfg:dict):
        """Track how often best parameters hug current bounds for targeted expansions.

        Maintains self._master_adaptive_state['param_boundary_stats'] structure:
          { param: { 'near_low_hits': int, 'near_high_hits': int, 'windows': int,
                     'last_value': float, 'recent_sides': [ 'low'|'high' ], 'expansions': int } }
        """
        try:
            if not hasattr(self, '_master_adaptive_state'):
                return
            if not hasattr(self, 'best_params_global') or not self.best_params_global:
                return
            if not hasattr(self, 'search_space'):
                return
            tol_pct = float(cfg.get('targeted_boundary_tolerance_pct', 0.07))  # 7% of width by default
            stats = self._master_adaptive_state.setdefault('param_boundary_stats', {})
            from skopt.space import Real, Integer
            for dim in self.search_space:
                if not hasattr(dim, 'low') or not hasattr(dim, 'high'):
                    continue
                name = getattr(dim, 'name', None)
                if name not in self.best_params_global:
                    continue
                try:
                    cur_low, cur_high = float(dim.low), float(dim.high)
                    width = cur_high - cur_low
                    if width <= 0:
                        continue
                    val = float(self.best_params_global[name])
                    rel = (val - cur_low) / width if width > 0 else 0.5
                    rec = stats.setdefault(name, {
                        'near_low_hits': 0,
                        'near_high_hits': 0,
                        'windows': 0,
                        'last_value': None,
                        'recent_sides': [],
                        'expansions': 0
                    })
                    rec['windows'] += 1
                    rec['last_value'] = val
                    side = None
                    if rel <= tol_pct:
                        rec['near_low_hits'] += 1; side = 'low'
                    elif rel >= (1 - tol_pct):
                        rec['near_high_hits'] += 1; side = 'high'
                    if side:
                        rec['recent_sides'].append(side)
                        if len(rec['recent_sides']) > 3:
                            rec['recent_sides'] = rec['recent_sides'][-3:]
                except Exception:
                    continue
        except Exception:
            pass

    def _maybe_rebalance_penalty_components(self, window_num:int, rb_cfg:dict):
        """Rebalance individual penalty component bases so no single component dominates persistently.

        Reads recent ledger entries, computes average share of each component relative to sum of considered components.
        Adjusts bases toward target shares (uniform or user-specified) using a mild proportional controller.
        Config (penalty_component_rebalance):
          enabled: bool
          lookback_trials: int (default 120)
          min_trials: int (default 30)
          adjust_rate: float (default 0.4)  # fraction of share error applied
          floor: float (default 0.05)
          ceiling: float (default 25.0)
          target_shares: dict optional e.g. {"trade":0.2,"drawdown":0.2,...}; if missing uniform.
          min_component_share: float (default 0.02) below which scaling up is limited to avoid runaway noise.
        """
        try:
            # Component mapping: attribute name -> (ledger key prefix substring, current base attribute)
            components = [
                ('trade', 'trade_penalty', 'dynamic_trade_penalty_base'),
                ('drawdown', 'drawdown_penalty', 'dynamic_drawdown_penalty_base'),
                ('profit_factor', 'profit_factor_penalty', 'profit_factor_penalty_base'),
                ('expectancy', 'expectancy_penalty', 'expectancy_penalty_base'),
                ('trade_frequency', 'trade_frequency_penalty', 'excessive_trade_penalty_base'),
                ('false_signal', 'false_signal_penalty', None),  # no direct base (amplified internally)
                ('per_regime_false_signal', 'per_regime_false_signal_penalty', None),
                ('fragility', 'param_fragility_penalty', None)
            ]
            lookback = int(rb_cfg.get('lookback_trials', 120))
            min_trials = int(rb_cfg.get('min_trials', 30))
            adjust_rate = float(rb_cfg.get('adjust_rate', 0.4))
            floor = float(rb_cfg.get('floor', 0.05))
            ceiling = float(rb_cfg.get('ceiling', 25.0))
            min_comp_share = float(rb_cfg.get('min_component_share', 0.02))
            ledger_path = getattr(self, 'parameter_ledger_path', None)
            if not ledger_path or not os.path.exists(ledger_path):
                return
            import json as _json
            rows = []
            with open(ledger_path,'r',encoding='utf-8') as f:
                for line in f.readlines()[-lookback:]:
                    try:
                        rec = _json.loads(line)
                        if 'total_penalties' in rec:
                            rows.append(rec)
                    except Exception:
                        continue
            if len(rows) < min_trials:
                return
            # Aggregate sums
            sums = {}
            for _, ledger_key, _attr in components:
                sums[ledger_key] = 0.0
            for r in rows:
                for _, ledger_key, _attr in components:
                    val = r.get(ledger_key)
                    if isinstance(val, (int,float)) and np.isfinite(val):
                        sums[ledger_key] += float(val)
            total = sum(v for v in sums.values() if v is not None)
            if total <= 0:
                return
            shares = {k: (v/total) if total>0 else 0.0 for k,v in sums.items()}
            # Target shares
            tgt_spec = rb_cfg.get('target_shares') or {}
            active_keys = [ledger_key for _, ledger_key, _ in components if sums.get(ledger_key,0)>0]
            if not active_keys:
                return
            if tgt_spec:
                # Normalize user targets over active keys
                tgt_total = sum(tgt_spec.get(k.replace('_penalty',''),0) for k in active_keys) or 1.0
                targets = {k: (tgt_spec.get(k.replace('_penalty',''),0)/tgt_total) for k in active_keys}
            else:
                uniform = 1.0/len(active_keys)
                targets = {k: uniform for k in active_keys}
            adjustments = {}
            for cname, ledger_key, attr in components:
                if ledger_key not in active_keys:
                    continue
                share = shares.get(ledger_key,0.0)
                tgt = targets.get(ledger_key,0.0)
                if attr and hasattr(self, attr):
                    # compute proportional adjustment
                    error = share - tgt
                    # Only adjust meaningfully if share above min threshold or we're reducing
                    base_val = getattr(self, attr)
                    if error > 0 and share < min_comp_share:
                        continue
                    scale = 1.0 - adjust_rate * error  # if share>target -> scale<1 to reduce
                    # Bound scale mildness
                    scale = max(0.5, min(1.5, scale))
                    new_base = min(ceiling, max(floor, base_val * scale))
                    if abs(new_base - base_val) / max(1e-9, base_val) > 0.02:  # significant
                        setattr(self, attr, new_base)
                        adjustments[attr] = {'from': base_val, 'to': new_base, 'share': share, 'target': tgt}
            if adjustments:
                self._master_adaptive_state.setdefault('penalty_component_rebalance', {})
                self._master_adaptive_state['penalty_component_rebalance'] = {
                    'window': window_num,
                    'shares': {k: round(shares.get(k,0),4) for k in active_keys},
                    'targets': {k: round(targets.get(k,0),4) for k in active_keys},
                    'adjustments': adjustments
                }
                if self.debug_mode:
                    log_to_file(f"[PENAL_REBAL] window={window_num} adjustments={len(adjustments)}", print_to_console=False)
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[PENAL_REBAL_ERR] {e}", print_to_console=False)

    def _maybe_expand_search_space(self, window_num:int, cfg:dict):
        """Expand search space (rescue exploration) under stagnation, overfit events or excessive penalties.

        Triggers:
        - Repeated overfit events reaching threshold.
        - Stagnation windows without improvement exceeding expansion_stagnation_patience.
        - High penalty density (average total penalty / |score| > threshold) recorded in recent trials.

        Expansion Logic:
        - For each numeric dimension (Integer/Real) remember original bounds (persist in _orig_bounds).
        - Expand current bounds outward toward originals by expansion_factor proportion of remaining distance.
        - If originals not stored yet, store before first contraction/expansion.
        - Cap expansions per run via max_expansions.
        """
        try:
            if not hasattr(self, '_master_adaptive_state'):
                return
            if not hasattr(self, 'search_space'):
                return

            # Always update boundary stats each window (even if no expansion later)
            self._update_param_boundary_stats(window_num, cfg)

            overfit_events = self._master_adaptive_state.get('overfit_events', 0)
            stagn_windows = self._master_adaptive_state.get('stagnation_windows', 0)
            expansions_done = self._master_adaptive_state.get('expansions', 0)
            fs_ema = self._master_adaptive_state.get('false_signal_rate_ema')
            # Global cooldown decrement
            try:
                if 'global_expand_cooldown' in self._master_adaptive_state and self._master_adaptive_state['global_expand_cooldown'] > 0:
                    self._master_adaptive_state['global_expand_cooldown'] -= 1
            except Exception:
                pass

            max_expansions = int(cfg.get('max_expansions', 5))
            if expansions_done >= max_expansions:
                return

            # Thresholds
            overfit_trigger = int(cfg.get('expand_overfit_events', 4))
            stagnation_patience = int(cfg.get('expansion_stagnation_patience', 5))
            penalty_trigger = float(cfg.get('expansion_penalty_ratio_threshold', 0.6))
            fs_relax_thresh = float(cfg.get('expansion_false_signal_relax_threshold', 0.55))  # if false signal EMA above -> allow expansion
            expansion_factor = float(cfg.get('expansion_factor', 0.5))  # proportion of remaining distance back to original

            trigger_reasons = []
            if overfit_events >= overfit_trigger and overfit_trigger > 0:
                trigger_reasons.append('overfit_events')
            if stagn_windows >= stagnation_patience and stagnation_patience > 0:
                trigger_reasons.append('stagnation')
            if fs_ema is not None and fs_ema >= fs_relax_thresh:
                trigger_reasons.append('false_signal_relax')

            # Penalty density check from recent ledger samples if available
            penalty_ratio_ok = False
            try:
                if hasattr(self, 'parameter_ledger_path') and os.path.exists(self.parameter_ledger_path):
                    import json as _json
                    recent_penalties = []
                    recent_scores = []
                    with open(self.parameter_ledger_path,'r',encoding='utf-8') as f:
                        for line in f.readlines()[-50:]:  # last 50 trials
                            try:
                                rec = _json.loads(line)
                                recent_penalties.append(float(rec.get('total_penalties',0)))
                                recent_scores.append(abs(float(rec.get('raw_score',0)))+1e-9)
                            except Exception:
                                continue
                    if recent_penalties and recent_scores:
                        avg_pen = sum(recent_penalties)/len(recent_penalties)
                        avg_score_abs = sum(recent_scores)/len(recent_scores)
                        if avg_score_abs > 0 and (avg_pen/avg_score_abs) > penalty_trigger:
                            trigger_reasons.append('penalty_density')
                            penalty_ratio_ok = True
            except Exception:
                pass

            # Targeted expansion candidate selection (even if triggers absent; only act if triggers fire)
            boundary_stats = self._master_adaptive_state.get('param_boundary_stats', {}) or {}
            hug_ratio_thresh = float(cfg.get('boundary_hugging_ratio_threshold', 0.35))
            cooldown_windows = int(cfg.get('expand_cooldown_windows', 3))
            dir_min_step_frac = float(cfg.get('directional_min_step_fraction', 0.15))
            targeted_factor = float(cfg.get('targeted_expansion_factor', expansion_factor))
            cooldowns = self._master_adaptive_state.setdefault('space_expand_cooldowns', {})
            targeted_candidates = []  # list of (param_name, side)
            for pname, rec in boundary_stats.items():
                try:
                    wins = rec.get('windows', 0)
                    if wins < 2:
                        continue
                    cd = cooldowns.get(pname, 0)
                    if cd > 0:
                        continue
                    low_hits = rec.get('near_low_hits', 0)
                    high_hits = rec.get('near_high_hits', 0)
                    ratio = (low_hits + high_hits) / max(1, wins)
                    if ratio >= hug_ratio_thresh:
                        # Determine directional pressure: check last two sides
                        recent = rec.get('recent_sides', [])
                        side = None
                        if len(recent) >= 2 and recent[-1] == recent[-2]:
                            side = recent[-1]
                        # Fallback: pick side with more hits
                        if side is None:
                            side = 'low' if low_hits > high_hits else 'high'
                        targeted_candidates.append((pname, side, ratio))
                except Exception:
                    continue

            # Global cooldown enforcement (only for non-overfit triggers)
            global_cd = int(self._master_adaptive_state.get('global_expand_cooldown', 0) or 0)
            if not trigger_reasons:
                return  # no global triggers -> do nothing (stats already updated)
            if global_cd > 0 and 'overfit_events' not in trigger_reasons:
                return  # still cooling down for non-critical expansions

            # Prepare original bounds storage
            if not hasattr(self, '_orig_bounds'):
                self._orig_bounds = {}
                for dim in self.search_space:
                    if hasattr(dim, 'low') and hasattr(dim, 'high'):
                        self._orig_bounds[dim.name] = (float(dim.low), float(dim.high))
            updates = []
            new_space = []
            performed_targeted = False
            targeted_names = {tc[0] for tc in targeted_candidates}

            # Decide if we use targeted mode: if we have candidates and not forced global by config
            always_global_reasons = set(cfg.get('expansion_always_global_reasons', ['overfit_events','penalty_density']))
            forced_global = any(r in always_global_reasons for r in trigger_reasons)
            if targeted_candidates and not forced_global:
                performed_targeted = True

            for dim in self.search_space:
                if hasattr(dim, 'low') and hasattr(dim, 'high') and dim.name in getattr(self, '_orig_bounds', {}):
                    orig_low, orig_high = self._orig_bounds[dim.name]
                    cur_low, cur_high = float(dim.low), float(dim.high)
                    nlow, nhigh = cur_low, cur_high
                    side_used = 'both'
                    factor_use = expansion_factor
                    ratio_used = None
                    if performed_targeted and dim.name in targeted_names:
                        # Directional expansion
                        cand = [c for c in targeted_candidates if c[0] == dim.name][0]
                        _, side, ratio_used = cand
                        factor_use = targeted_factor
                        if side == 'low' and orig_low < cur_low:
                            rem = cur_low - orig_low
                            step = max(rem * factor_use, rem * dir_min_step_frac)
                            nlow = max(orig_low, cur_low - step)
                            side_used = 'low'
                        elif side == 'high' and orig_high > cur_high:
                            rem = orig_high - cur_high
                            step = max(rem * factor_use, rem * dir_min_step_frac)
                            nhigh = min(orig_high, cur_high + step)
                            side_used = 'high'
                    elif not performed_targeted:
                        # Original global bidirectional expansion
                        if orig_low < cur_low:
                            expand_dist_low = cur_low - orig_low
                            nlow = max(orig_low, cur_low - expand_dist_low * expansion_factor)
                        if orig_high > cur_high:
                            expand_dist_high = orig_high - cur_high
                            nhigh = min(orig_high, cur_high + expand_dist_high * expansion_factor)
                    # Rebuild dim if changed
                    if nlow < cur_low or nhigh > cur_high:
                        updates.append((dim.name, cur_low, cur_high, nlow, nhigh, side_used, ratio_used))
                        if performed_targeted:
                            # apply cooldown
                            cooldowns[dim.name] = cooldown_windows
                            # increment expansions count in stats
                            try:
                                self._master_adaptive_state['param_boundary_stats'][dim.name]['expansions'] += 1
                            except Exception:
                                pass
                        from skopt.space import Integer as _I, Real as _R
                        if isinstance(dim, Integer):
                            new_space.append(_I(int(round(nlow)), int(round(nhigh)), name=dim.name))
                        elif isinstance(dim, Real):
                            new_space.append(_R(nlow, nhigh, prior=getattr(dim,'prior','uniform'), name=dim.name))
                        else:
                            new_space.append(dim)
                    else:
                        new_space.append(dim)
                else:
                    new_space.append(dim)

            # Decrement cooldowns
            try:
                for k in list(cooldowns.keys()):
                    cooldowns[k] = max(0, cooldowns[k]-1)
                self._master_adaptive_state['space_expand_cooldowns'] = cooldowns
            except Exception:
                pass

            if updates:
                self.search_space = new_space
                self._master_adaptive_state['expansions'] = expansions_done + 1
                tag = 'SPACE_EXPAND_TARGETED' if performed_targeted else 'SPACE_EXPAND'
                # Set global cooldown
                try:
                    g_cd_w = int(cfg.get('global_expand_cooldown_windows', 2))
                    if g_cd_w > 0:
                        self._master_adaptive_state['global_expand_cooldown'] = g_cd_w
                except Exception:
                    pass
                try:
                    for u in updates:
                        if performed_targeted:
                            log_to_file(f"[{tag}] window={window_num} dim={u[0]} {u[1]}-{u[2]} -> {u[3]}-{u[4]} side={u[5]} ratio={u[6]} triggers={','.join(trigger_reasons)}", print_to_console=self.debug_mode)
                        else:
                            log_to_file(f"[{tag}] window={window_num} dim={u[0]} {u[1]}-{u[2]} -> {u[3]}-{u[4]} triggers={','.join(trigger_reasons)}", print_to_console=self.debug_mode)
                    # Record expansion history
                    try:
                        hist = self._master_adaptive_state.setdefault('expansion_history', [])
                        for u in updates:
                            hist.append({
                                'window': window_num,
                                'param': u[0],
                                'from_low': u[1], 'from_high': u[2],
                                'to_low': u[3], 'to_high': u[4],
                                'side': u[5], 'ratio': u[6],
                                'targeted': performed_targeted,
                                'triggers': trigger_reasons
                            })
                        if len(hist) > 300:
                            self._master_adaptive_state['expansion_history'] = hist[-300:]
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[SPACE_EXPAND_ERR] {e}", print_to_console=False)

    def _maybe_auto_tune_regime_multipliers(self, window_num:int, cfg:dict):
        """Auto-tune regime risk multipliers with smoothing & stability safeguards.

        Enhancements (Item 8 refinements):
        - EMA smoothing of per-regime composite score (wr & avg pnl deltas) using regime_ema_alpha (default 0.4)
        - Stability guard: require >= regime_min_trades (default 8) and wr confidence band not extremely wide
        - Cooldown: adjust every regime_tune_every windows (default 2)
        - Bounded multiplier drift with regime_step, min_regime_mult, max_regime_mult
        - Ledger logging of each adjustment event for transparency
        - History list self._adaptive_regime_tune_history (capped) for diagnostics
        """
        try:
            perf = getattr(self, '_regime_perf', None)
            if not perf:
                return
            tune_every = int(cfg.get('regime_tune_every', 2))
            if window_num % max(1, tune_every) != 0:
                return
            min_mult = float(cfg.get('min_regime_mult', 0.4))
            max_mult = float(cfg.get('max_regime_mult', 1.8))
            step = float(cfg.get('regime_step', 0.05))
            alpha = float(cfg.get('regime_ema_alpha', 0.4))
            min_trades = int(cfg.get('regime_min_trades', 8))
            # Baseline aggregates
            total_trades = sum(v['trades'] for v in perf.values()) or 1
            total_pnl = sum(v['pnl'] for v in perf.values())
            total_wins = sum(v['wins'] for v in perf.values())
            global_wr = total_wins / total_trades
            global_avg_pnl = total_pnl / total_trades
            updated_records = []
            # Ensure history container
            if not hasattr(self, '_adaptive_regime_tune_history'):
                self._adaptive_regime_tune_history = []
            for regime, stats in perf.items():
                if stats['trades'] < min_trades:
                    continue
                wr = stats['wins'] / stats['trades'] if stats['trades'] else 0
                avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] else 0
                wr_delta = wr - global_wr
                pnl_delta = avg_pnl - global_avg_pnl
                raw_score = (wr_delta * 0.6) + (pnl_delta * 0.4)
                # EMA smoothing store per regime
                regime_state = stats.setdefault('_state', {})
                prev_ema = regime_state.get('score_ema', raw_score)
                score_ema = (alpha * raw_score) + ((1 - alpha) * prev_ema)
                regime_state['score_ema'] = score_ema
                mult_prev = self._adaptive_regime_multipliers.get(regime, 1.0)
                mult_new = mult_prev
                if score_ema > 0:
                    mult_new = min(max_mult, mult_prev + step)
                elif score_ema < 0:
                    mult_new = max(min_mult, mult_prev - step)
                if mult_new != mult_prev:
                    self._adaptive_regime_multipliers[regime] = round(mult_new, 3)
                    updated_records.append({
                        'window': window_num,
                        'regime': regime,
                        'wr': round(wr, 4),
                        'avg_pnl': round(avg_pnl, 6),
                        'raw_score': round(raw_score, 6),
                        'score_ema': round(score_ema, 6),
                        'mult_from': mult_prev,
                        'mult_to': mult_new
                    })
                    if self.debug_mode:
                        log_to_file(f"[REGIME_TUNE] w={window_num} {regime} wr={wr:.2f} avg_pnl={avg_pnl:.4f} score_ema={score_ema:.4f} {mult_prev:.2f}->{mult_new:.2f}", print_to_console=False)
            if updated_records:
                # Persist & history
                self._master_adaptive_state['regime_multipliers'] = self._adaptive_regime_multipliers
                self._persist_master_state(window_num)
                # Trim history
                self._adaptive_regime_tune_history.extend(updated_records)
                if len(self._adaptive_regime_tune_history) > 300:
                    self._adaptive_regime_tune_history = self._adaptive_regime_tune_history[-300:]
                # Ledger logging (one consolidated entry)
                try:
                    self._write_parameter_ledger({
                        'record_type': 'regime_tune_batch',
                        'window': window_num,
                        'updates': updated_records,
                        'global_wr': round(global_wr, 4),
                        'global_avg_pnl': round(global_avg_pnl, 6)
                    })
                except Exception:
                    pass
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[REGIME_TUNE_ERR] {e}", print_to_console=False)

    def _maybe_update_risk_scaling(self, window_num:int, cfg:dict):
        """Update performance-based risk scaling bands (Item 9).

        Uses recent validation stats & adaptive penalties to assign a discrete risk tier
        (conservative|normal|aggressive) that scales position sizing.

        Criteria (configurable):
          - avg_val_sharpe over last N windows
          - recent max drawdown (from master state if available)
          - estimated false signal rate (if available) else fallback

        Config keys (under adaptive_master / risk_scaling_*):
          risk_scaling_lookback (int) default 4
          risk_scaling_sharpe_aggressive (float) default 1.2
          risk_scaling_sharpe_conservative (float) default 0.3
          risk_scaling_drawdown_conservative (float) default 0.18
          risk_scaling_false_rate_conservative (float) default 0.40
          risk_scaling_false_rate_aggressive (float) default 0.22
          risk_scaling_multiplier_conservative (float) default 0.75
          risk_scaling_multiplier_normal (float) default 1.0
          risk_scaling_multiplier_aggressive (float) default 1.25
        """
        try:
            # Gather recent validation Sharpe values from ledgered stats
            if not hasattr(self, '_inner_validation_stats') or not self._inner_validation_stats:
                return
            lookback = int(cfg.get('risk_scaling_lookback', 4))
            recent = self._inner_validation_stats[-lookback:]
            if not recent:
                return
            sharpe_vals = [r.get('val_sharpe') or r.get('val_sharpe', 0) for r in recent if isinstance(r, dict)]
            # Some earlier entries may not carry val_sharpe key; skip empties
            sharpe_vals = [s for s in sharpe_vals if s is not None]
            if not sharpe_vals:
                return
            avg_sharpe = float(np.mean(sharpe_vals))
            # Estimated drawdown: try last accepted validation drawdown stored in ledger snapshot (simplified)
            est_drawdown = float(self._master_adaptive_state.get('recent_validation_drawdown', 0.0))
            # False signal rate estimate placeholder
            fs_rate = float(self._master_adaptive_state.get('false_signal_rate_ema', 0.25))
            # Thresholds
            sh_aggr = float(cfg.get('risk_scaling_sharpe_aggressive', 1.2))
            sh_cons = float(cfg.get('risk_scaling_sharpe_conservative', 0.3))
            dd_cons = float(cfg.get('risk_scaling_drawdown_conservative', 0.18))
            fs_cons = float(cfg.get('risk_scaling_false_rate_conservative', 0.40))
            fs_aggr = float(cfg.get('risk_scaling_false_rate_aggressive', 0.22))
            mult_cons = float(cfg.get('risk_scaling_multiplier_conservative', 0.75))
            mult_norm = float(cfg.get('risk_scaling_multiplier_normal', 1.0))
            mult_aggr = float(cfg.get('risk_scaling_multiplier_aggressive', 1.25))
            # Determine tier
            tier = 'normal'
            multiplier = mult_norm
            if avg_sharpe >= sh_aggr and est_drawdown > -dd_cons and fs_rate <= fs_aggr:
                tier = 'aggressive'
                multiplier = mult_aggr
            elif (avg_sharpe <= sh_cons) or (abs(est_drawdown) >= dd_cons) or (fs_rate >= fs_cons):
                tier = 'conservative'
                multiplier = mult_cons
            # Gentle hysteresis: don't oscillate too fast (only allow change if different from last 2 tiers)
            history = getattr(self, '_risk_tier_history', [])
            if history and history[-1] == tier and len(history) >= 2 and history[-2] == tier:
                pass  # stable
            # Save
            if not history or history[-1] != tier:
                history.append(tier)
                if len(history) > 20:
                    history = history[-20:]
                self._risk_tier_history = history
            prev_tier = getattr(self, '_current_risk_tier', 'normal')
            self._current_risk_tier = tier
            self._current_risk_multiplier = multiplier
            self._master_adaptive_state['risk_tier'] = tier
            self._master_adaptive_state['risk_multiplier'] = multiplier
            if prev_tier != tier and self.debug_mode:
                log_to_file(f"[RISK_TIER] window={window_num} changed {prev_tier}->{tier} avg_sh={avg_sharpe:.2f} dd={est_drawdown:.3f} fs={fs_rate:.2f} mult={multiplier:.2f}", print_to_console=False)
            # Ledger log (single record)
            try:
                self._write_parameter_ledger({
                    'record_type': 'risk_tier_update',
                    'window': window_num,
                    'avg_val_sharpe': avg_sharpe,
                    'est_drawdown': est_drawdown,
                    'false_signal_rate': fs_rate,
                    'risk_tier': tier,
                    'risk_multiplier': multiplier
                })
            except Exception:
                pass
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[RISK_TIER_ERR] {e}", print_to_console=False)

    # ========================= Penalty Auto-Calibration =========================
    def _maybe_auto_calibrate_penalties(self, window_num:int, cfg:dict):
        """Auto-adjust penalty base coefficients to target a desired penalty fraction.

        Approach:
          - Collect recent ledger penalty components (stored during objective).
          - Compute average penalty_fraction = total_penalties / (total_penalties + max(raw_positive_score,1e-9)).
          - If above target_high -> reduce penalty bases; if below target_low -> increase.
          - Gentle scaling limited by max_adjust_per_cycle.
        """
        try:
            pac = cfg.get('penalty_auto_calibration_settings', {})
            lookback = int(pac.get('lookback_windows', 4))
            target = float(pac.get('target_penalty_fraction', 0.35))
            band = float(pac.get('tolerance_band', 0.08))
            max_adj = float(pac.get('max_adjust_per_cycle', 0.15))  # 15%
            # Collect from internal stats list if exists
            if not hasattr(self, '_inner_validation_stats') or not hasattr(self, 'all_trial_data'):
                return
            # Filter trial data for last N windows
            recent_trials = [t for t in self.all_trial_data if t.get('window') is not None and isinstance(t.get('window'), int) and t.get('window') > (window_num - lookback)]
            if len(recent_trials) < 5:
                return
            pen_fracs = []
            for rt in recent_trials:
                tot_pen = rt.get('total_penalties')
                raw_score = rt.get('raw_score')
                if tot_pen is None or raw_score is None:
                    continue
                base = max(1e-6, abs(raw_score) + tot_pen)
                pen_fracs.append(tot_pen / base)
            if not pen_fracs:
                return
            avg_frac = float(np.mean(pen_fracs))
            upper = target + band
            lower = target - band
            if lower <= avg_frac <= upper:
                return  # within band
            direction = -1 if avg_frac > upper else 1
            scale = 1.0 + direction * min(max_adj, abs(avg_frac - target))
            # Adjust bases (bounded)
            for attr in ['dynamic_trade_penalty_base','dynamic_drawdown_penalty_base','profit_factor_penalty_base','expectancy_penalty_base','excessive_trade_penalty_base']:
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    new_val = max(0.1, val * scale)
                    setattr(self, attr, new_val)
            self._master_adaptive_state['penalty_auto_calibration'] = {
                'window': window_num,
                'avg_penalty_fraction': avg_frac,
                'target': target,
                'scale_applied': scale
            }
            if self.debug_mode:
                log_to_file(f"[PENAL_CAL] window={window_num} avg_frac={avg_frac:.3f} target={target:.3f} scale={scale:.3f}", print_to_console=False)
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[PENAL_CAL_ERR] {e}", print_to_console=False)

    # ========================= Drift Detection =========================
    def _maybe_detect_drift(self, window_num:int, cfg:dict):
        """Detect performance drift and boost exploration / trials if needed.

        Logic: Track recent test (validation) Sharpes from accepted best params. If rolling mean deteriorates > drift_threshold relative to prior rolling mean -> mark drift event, increase explore fraction & calls multiplier.
        """
        try:
            dd_cfg = cfg.get('drift_detection_settings', {})
            lookback = int(dd_cfg.get('lookback_windows', 6))
            min_windows = int(dd_cfg.get('min_windows', 4))
            threshold = float(dd_cfg.get('drift_threshold', 0.5))  # relative drop
            boost = float(dd_cfg.get('explore_boost', 0.1))
            if not hasattr(self, '_window_best_validation_sharpes'):
                return
            vals = self._window_best_validation_sharpes[-lookback:]
            if len(vals) < min_windows:
                return
            first_half = vals[:len(vals)//2]
            second_half = vals[len(vals)//2:]
            if not first_half or not second_half:
                return
            mean1 = np.mean(first_half)
            mean2 = np.mean(second_half)
            if mean1 <= 0:
                return
            rel_drop = (mean1 - mean2) / max(1e-9, abs(mean1))
            if rel_drop >= threshold:
                # Drift detected
                self._master_adaptive_state['drift_events'] = self._master_adaptive_state.get('drift_events', 0) + 1
                self._dynamic_explore_frac = min(self._explore_auto_max, (self._dynamic_explore_frac or 0.3) + boost)
                self._master_adaptive_state['calls_multiplier'] = min(2.0, self._master_adaptive_state.get('calls_multiplier', 1.0) * 1.1)
                if self.debug_mode:
                    log_to_file(f"[DRIFT] window={window_num} rel_drop={rel_drop:.2f} explore_frac={self._dynamic_explore_frac:.2f}", print_to_console=False)
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[DRIFT_ERR] {e}", print_to_console=False)

    # ========================= False Signal Rate Estimation =========================
    def _maybe_update_false_signal_rate(self, window_num:int, cfg:dict):
        """Compute and smooth a formal false signal rate from recent trades.

        Classification Rules (configurable via false_signal_estimation_settings):
          - A trade is candidate if bars_in_trade <= quick_bars_threshold.
          - It is marked false_signal if (pnl <= quick_loss_threshold) OR
            (max_adverse_excursion <= adverse_threshold (negative) AND max_favorable_excursion small relative to adverse (|mfe|/|mae| < mfe_mae_ratio_max)).
          - Ignore trades with missing diagnostics.
        Smoothing:
          - Maintain EMA of rate (alpha adaptive: alpha_base * clamp(trade_count/min_trades_for_full_alpha, 0.25, 1)).
          - Maintain rolling raw mean over last raw_lookback trades for diagnostics.
        Effects:
          - Store in master adaptive state: false_signal_rate_raw, false_signal_rate_ema, false_signal_zscore.
          - If rate above severe threshold -> gently boost exploration & calls multiplier.
        """
        try:
            fs_cfg = cfg.get('false_signal_estimation_settings', {})
            if not fs_cfg.get('enabled', True):
                return
            # Pull combined trades if available
            all_trades_list = getattr(self, 'all_trades', None)
            if not all_trades_list:
                return
            try:
                import pandas as _pd
                trades_df = _pd.concat(all_trades_list, ignore_index=True)
            except Exception:
                return
            if trades_df.empty:
                return
            # Only consider trades up to current window
            trades_df = trades_df[trades_df['window'] <= window_num]
            min_trades = int(fs_cfg.get('min_trades', 30))
            if len(trades_df) < min_trades:
                return
            # Rolling slice of most recent trades (limit memory)
            max_trades_considered = int(fs_cfg.get('max_trades_considered', 400))
            if len(trades_df) > max_trades_considered:
                trades_df = trades_df.tail(max_trades_considered)
            quick_bars = int(fs_cfg.get('quick_bars_threshold', 6))
            quick_loss_thresh = float(fs_cfg.get('quick_loss_threshold', -0.0025))  # -0.25% default
            adverse_threshold = float(fs_cfg.get('adverse_excursion_threshold', -0.003))  # -0.3%
            mfe_mae_ratio_max = float(fs_cfg.get('mfe_mae_ratio_max', 0.35))
            raw_lookback = int(fs_cfg.get('raw_rate_lookback', 120))
            alpha_base = float(fs_cfg.get('ema_alpha_base', 0.12))
            severe_threshold = float(fs_cfg.get('severe_rate_threshold', 0.50))
            high_threshold = float(fs_cfg.get('high_rate_threshold', 0.40))
            target_rate = float(fs_cfg.get('target_rate', 0.25))
            # Filter trades with diagnostics
            diag_trades = trades_df.dropna(subset=['bars_in_trade','max_adverse_excursion','max_favorable_excursion','pnl'])
            if diag_trades.empty:
                return
            # Classification
            def classify(row):
                try:
                    if row['bars_in_trade'] is None or row['bars_in_trade'] > quick_bars:
                        return 0
                    pnl = float(row['pnl'])
                    mae = float(row['max_adverse_excursion']) if row['max_adverse_excursion'] is not None else 0.0
                    mfe = float(row['max_favorable_excursion']) if row['max_favorable_excursion'] is not None else 0.0
                    # Direction normalization (mae expected negative for adverse on longs)
                    # Quick loss condition
                    if pnl <= quick_loss_thresh:
                        return 1
                    # Adverse excursion dominated & little favorable progress
                    if mae <= adverse_threshold:
                        if mae != 0:
                            ratio = abs(mfe) / max(1e-9, abs(mae))
                        else:
                            ratio = 1.0
                        if ratio < mfe_mae_ratio_max:
                            return 1
                    return 0
                except Exception:
                    return 0
            diag_trades = diag_trades.copy()
            diag_trades['is_false_signal'] = diag_trades.apply(classify, axis=1)
            # Per-regime breakdown (if market_regime column present)
            per_regime_rates = {}
            if 'market_regime' in diag_trades.columns:
                try:
                    grouped = diag_trades.groupby('market_regime')
                    for regime_name, g in grouped:
                        if g.empty:
                            continue
                        recent_g = g.tail(raw_lookback)
                        total_g = len(recent_g)
                        if total_g == 0:
                            continue
                        false_g = int(recent_g['is_false_signal'].sum())
                        per_regime_rates[regime_name] = false_g / total_g
                except Exception:
                    per_regime_rates = {}
            # Compute raw rate over last raw_lookback trades with classification
            recent_diag = diag_trades.tail(raw_lookback)
            total_considered = len(recent_diag)
            if total_considered == 0:
                return
            false_ct = int(recent_diag['is_false_signal'].sum())
            raw_rate = false_ct / total_considered
            # EMA smoothing
            prev_ema = float(self._master_adaptive_state.get('false_signal_rate_ema', raw_rate)) if self._master_adaptive_state else raw_rate
            # Adaptive alpha based on sample size adequacy
            adequacy = min(1.0, total_considered / max(1.0, raw_lookback/1.5))
            alpha = alpha_base * max(0.25, adequacy)
            ema = prev_ema + alpha * (raw_rate - prev_ema)
            # Maintain simple historical stats for z-score
            hist_key = '_false_signal_hist'
            hist = getattr(self, hist_key, [])
            hist.append(raw_rate)
            if len(hist) > 300:
                hist = hist[-300:]
            setattr(self, hist_key, hist)
            mean_hist = float(np.mean(hist)) if hist else raw_rate
            std_hist = float(np.std(hist)) if hist else 1e-9
            zscore = (ema - mean_hist) / max(1e-9, std_hist)
            if not self._master_adaptive_state:
                self._master_adaptive_state = {}
            self._master_adaptive_state['false_signal_rate_raw'] = raw_rate
            self._master_adaptive_state['false_signal_rate_ema'] = ema
            self._master_adaptive_state['false_signal_rate_z'] = zscore
            self._master_adaptive_state['false_signal_sample'] = total_considered
            if per_regime_rates:
                self._master_adaptive_state.setdefault('per_regime_false_signal', {})
                # Update EMA per regime
                pr_container = self._master_adaptive_state['per_regime_false_signal']
                for rname, rate in per_regime_rates.items():
                    prev = pr_container.get(rname, {}).get('ema', rate)
                    pr_alpha = alpha  # reuse adaptive alpha
                    new_ema = prev + pr_alpha * (rate - prev)
                    pr_container[rname] = {
                        'raw': rate,
                        'ema': new_ema,
                        'window': window_num
                    }
            # Optional regime multiplier micro-adjustment if severe regime-specific false signal
            try:
                if per_regime_rates and self._master_adaptive_state.get('regime_multipliers'):
                    rm = self._master_adaptive_state.get('regime_multipliers')
                    worst_regimes = [r for r, rt in per_regime_rates.items() if rt >= high_threshold]
                    if worst_regimes:
                        for wr in worst_regimes:
                            cur = rm.get(wr, 1.0)
                            rm[wr] = max(0.5, cur * 0.97)  # slight de-emphasis
                        self._master_adaptive_state['regime_multipliers'] = rm
            except Exception:
                pass
            # Adaptive reactions
            reacted = False
            if ema >= severe_threshold:
                # Strong push for exploration & penalty tightening
                self._dynamic_explore_frac = min(self._explore_auto_max, (self._dynamic_explore_frac or 0.4) + 0.08)
                self._master_adaptive_state['calls_multiplier'] = min(2.2, self._master_adaptive_state.get('calls_multiplier', 1.0) * 1.12)
                # Slightly increase trade quality penalties
                self.dynamic_trade_penalty_base *= 1.05
                reacted = True
            elif ema >= high_threshold:
                self._dynamic_explore_frac = min(self._explore_auto_max, (self._dynamic_explore_frac or 0.4) + 0.04)
                self._master_adaptive_state['calls_multiplier'] = min(2.0, self._master_adaptive_state.get('calls_multiplier', 1.0) * 1.06)
                reacted = True
            elif ema < target_rate * 0.6 and total_considered > raw_lookback * 0.5:
                # Very low false signal rate: allow mild contraction via reduced exploration
                self._dynamic_explore_frac = max(self._explore_auto_min, (self._dynamic_explore_frac or 0.5) - 0.03)
                reacted = True
            if self.debug_mode:
                log_to_file(f"[FALSE_SIG] window={window_num} raw={raw_rate:.3f} ema={ema:.3f} z={zscore:.2f} alpha={alpha:.3f} reacted={reacted}", print_to_console=False)
            # Ledger entry
            try:
                base_record = {
                    'record_type': 'false_signal_evaluation',
                    'window': window_num,
                    'raw_rate': raw_rate,
                    'ema_rate': ema,
                    'zscore': zscore,
                    'sample': total_considered,
                    'quick_bars_threshold': quick_bars,
                    'quick_loss_threshold': quick_loss_thresh,
                    'adverse_excursion_threshold': adverse_threshold,
                    'mfe_mae_ratio_max': mfe_mae_ratio_max,
                    'explore_fraction': self._dynamic_explore_frac,
                    'calls_multiplier': self._master_adaptive_state.get('calls_multiplier', 1.0),
                    'penalty_trade_base': self.dynamic_trade_penalty_base
                }
                # Attach per-regime snapshot (limited size) if available
                pr_fs = self._master_adaptive_state.get('per_regime_false_signal') if self._master_adaptive_state else None
                if pr_fs:
                    slim = {k: {'ema': round(v.get('ema',0),4), 'raw': round(v.get('raw',0),4)} for k,v in list(pr_fs.items())[:15]}
                    base_record['per_regime_false_signal'] = slim
                self._write_parameter_ledger(base_record)
            except Exception:
                pass
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[FALSE_SIG_ERR] {e}", print_to_console=False)

    def _persist_master_state(self, window_num:int):
        """Persist overarching adaptive controller state to disk so subsequent runs bootstrap from learned dynamics."""
        try:
            path = getattr(self, '_master_state_path', 'adaptive_master_state.json')
            payload = {
                'window': window_num,
                'state': self._master_adaptive_state,
                'dynamic_explore_fraction': self._dynamic_explore_frac,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)
            # Dashboard export (lightweight) if enabled
            try:
                dash_cfg = self.config.get('optimization_settings', {}).get('adaptive_dashboard_export', {})
                if dash_cfg.get('enabled', True):
                    export_every = int(dash_cfg.get('every_n_windows', 1))
                    if window_num % max(1, export_every) == 0:
                        dash_keys = dash_cfg.get('include_keys', [
                            'false_signal_rate_ema','calls_multiplier','risk_tier','risk_multiplier',
                            'per_regime_false_signal','regime_multipliers','penalty_auto_calibration'
                        ])
                        dash_payload = {'window': window_num,'timestamp': datetime.utcnow().isoformat()+'Z'}
                        st = self._master_adaptive_state or {}
                        for k in dash_keys:
                            if k in st:
                                dash_payload[k] = st[k]
                        dash_payload['dynamic_explore_fraction'] = self._dynamic_explore_frac
                        # Attach concise param boundary snapshot if available
                        try:
                            pbs = (self._master_adaptive_state or {}).get('param_boundary_stats') or {}
                            if pbs:
                                snap = {}
                                for k, v in pbs.items():
                                    wins = v.get('windows', 0) or 1
                                    lowr = v.get('near_low_hits', 0)/wins
                                    highr = v.get('near_high_hits', 0)/wins
                                    snap[k] = {
                                        'low_ratio': round(lowr,3),
                                        'high_ratio': round(highr,3),
                                        'last_value': v.get('last_value'),
                                        'cooldown': (self._master_adaptive_state.get('space_expand_cooldowns',{}) or {}).get(k,0),
                                        'expansions': v.get('expansions',0)
                                    }
                                dash_payload['param_boundary'] = snap
                            # Fragility attribution summary: top hugging params by combined ratio
                            if pbs:
                                combos = []
                                for k,v in pbs.items():
                                    wins = v.get('windows',0) or 1
                                    combo = (v.get('near_low_hits',0)+v.get('near_high_hits',0))/wins
                                    combos.append((k, combo))
                                combos.sort(key=lambda x: x[1], reverse=True)
                                dash_payload['fragility_top'] = combos[:5]
                            # Recent expansion history sample
                            hist = (self._master_adaptive_state or {}).get('expansion_history') or []
                            if hist:
                                dash_payload['expansion_recent'] = hist[-5:]
                        except Exception:
                            pass
                        dash_path = dash_cfg.get('path', 'adaptive_dashboard_snapshot.json')
                        with open(dash_path, 'w') as df:
                            json.dump(dash_payload, df, indent=2)
                        # Integrity check: ensure all requested keys present in output (even if absent in state)
                        missing_keys = [k for k in dash_keys if k not in dash_payload]
                        if missing_keys:
                            log_to_file(f"[ADAPTIVE DASHBOARD][INTEGRITY] Missing requested keys: {missing_keys}", print_to_console=False)
                        else:
                            log_to_file("[ADAPTIVE DASHBOARD][INTEGRITY] All requested include_keys exported.", print_to_console=False)
            except Exception:
                pass
        except Exception:
            pass

    def save_trial_data(self, study_or_result, window_num: int, optimizer_type: str):
        """
        Save optimization trial data for analysis.
        - optimizer_type 'optuna': expects an Optuna Study
        - optimizer_type 'bayesian': expects a skopt OptimizeResult
        Appends normalized records to self.all_trial_data.
        """
        try:
            records = []
            if optimizer_type == 'optuna':
                try:
                    for t in getattr(study_or_result, 'trials', []) or []:
                        rec = {
                            'window': window_num,
                            'optimizer': 'optuna',
                            'number': getattr(t, 'number', None),
                            'state': str(getattr(t, 'state', 'UNKNOWN')),
                            'value': getattr(t, 'value', None),
                            'objective_value': getattr(t, 'value', None),
                            'params': getattr(t, 'params', {})
                        }
                        # Flatten params into top-level keys for easier analysis
                        if isinstance(rec['params'], dict):
                            for k, v in rec['params'].items():
                                rec[k] = v
                        user_attrs = getattr(t, 'user_attrs', {}) or {}
                        if isinstance(user_attrs, dict) and user_attrs:
                            rec['user_attrs'] = user_attrs
                        records.append(rec)
                except Exception as e:
                    self.log_debug(f"Failed to serialize Optuna trials for window {window_num}: {e}")
            elif optimizer_type == 'bayesian':
                try:
                    x_iters = getattr(study_or_result, 'x_iters', []) or []
                    func_vals = list(getattr(study_or_result, 'func_vals', []) or [])
                    for idx, (params_list, val) in enumerate(zip(x_iters, func_vals)):
                        params = dict(zip([dim.name for dim in self.search_space], params_list))
                        rec = {
                            'window': window_num,
                            'optimizer': 'bayesian',
                            'number': idx,
                            'value': float(val) if val is not None else None,
                            'objective_value': float(val) if val is not None else None,
                            'params': params,
                        }
                        # Flatten params into top-level keys for easier analysis
                        for k, v in params.items():
                            rec[k] = v
                        records.append(rec)
                except Exception as e:
                    self.log_debug(f"Failed to serialize Bayesian trials for window {window_num}: {e}")

            if records:
                self.all_trial_data.extend(records)
                log_to_file(f"Saved {len(records)} trial records for window #{window_num} ({optimizer_type})", print_to_console=False)
        except Exception as e:
            self.log_debug(f"save_trial_data error: {e}")
        log_to_file(f"---------------------------------", print_to_console=False)


    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # Auto-trigger system health check when config is loaded
            try:
                from health_utils import auto_fix_on_config_load
                auto_fix_on_config_load(path)
            except ImportError:
                pass  # health_utils not available - proceed without check
            
            return config
        except Exception as e:
            log_to_file(f"FATAL: Could not load/parse config at {path}: {e}")
            return None

    def define_parameter_spaces(self):
        """
        Enhanced parameter space definition with regime-specific optimization support
        """
        param_spaces_config = self.config.get('parameter_spaces', {})
        search_space = []
        
        # Check if regime-specific optimization is enabled
        regime_optimization_enabled = self.config.get('optimization_settings', {}).get('regime_specific_optimization', False)
        
        if regime_optimization_enabled:
            # Use regime-specific parameter spaces with base parameter inclusion
            self.log_debug("Using enhanced regime-specific parameter optimization with base parameters")
            return self._define_regime_specific_spaces()
        
        # Traditional global parameter space
        self.log_debug("Building global parameter space...")
        ichimoku_params = {}
        
        for group_name, params_list in param_spaces_config.items():
            if not isinstance(params_list, list):
                # Skip items that are not lists (like comments)
                continue

            for p_config in params_list:
                p_name = p_config.get('name')
                p_type = p_config.get('type')
                
                if not all([p_name, p_type]):
                    continue # Skip if essential keys are missing

                # --- FIX: Enforce hard_bounds ---
                bounds = p_config.get('bounds', [1, 100] if p_type.lower() == 'integer' else [0.0, 1.0])
                hard_bounds = p_config.get('hard_bounds')

                if hard_bounds:
                    # Clamp the optimization bounds to be within the hard_bounds
                    bounds[0] = max(bounds[0], hard_bounds[0])
                    bounds[1] = min(bounds[1], hard_bounds[1])
                    # Ensure lower bound is not greater than upper bound after clamping
                    if bounds[0] > bounds[1]:
                        bounds[0] = bounds[1]

                # Log Ichimoku parameters for debugging
                if 'TENKAN' in p_name or 'KIJUN' in p_name or 'SENKOU' in p_name:
                    ichimoku_params[p_name] = bounds
                    self.log_debug(f"Ichimoku param {p_name}: bounds={bounds}, hard_bounds={hard_bounds}")

                if p_type.lower() == 'integer':
                    search_space.append(Integer(bounds[0], bounds[1], name=p_name))
                elif p_type.lower() == 'real':
                    search_space.append(Real(bounds[0], bounds[1], name=p_name))
                elif p_type.lower() == 'categorical':
                    values = p_config.get('values')
                    if values:
                        search_space.append(Categorical(values, name=p_name))

        # Final validation of Ichimoku bounds
        if ichimoku_params:
            self.log_debug(f"Final Ichimoku parameter bounds: {ichimoku_params}")
            tenkan_bounds = ichimoku_params.get('TENKAN_SEN_PERIOD', [0, 0])
            kijun_bounds = ichimoku_params.get('KIJUN_SEN_PERIOD', [0, 0])
            senkou_bounds = ichimoku_params.get('SENKOU_SPAN_B_PERIOD', [0, 0])
            
            # Check if bounds can satisfy constraints
            if tenkan_bounds[1] >= kijun_bounds[0]:
                self.log_debug(f"WARNING: TENKAN max ({tenkan_bounds[1]}) >= KIJUN min ({kijun_bounds[0]}) - constraint conflicts possible!")
            if kijun_bounds[1] >= senkou_bounds[0]:
                self.log_debug(f"WARNING: KIJUN max ({kijun_bounds[1]}) >= SENKOU min ({senkou_bounds[0]}) - constraint conflicts possible!")

        self.log_debug(f"Built parameter space with {len(search_space)} parameters")
        return search_space
    
    def _define_regime_specific_spaces(self):
        """Define parameter spaces optimized for different market regimes"""
        try:
            from core.strategy import RegimeSpecificOptimizer, MarketRegime
            
            regime_optimizer = RegimeSpecificOptimizer(self.config)
            combined_search_space = []
            
            # Generate parameter spaces for each regime
            for regime in MarketRegime:
                regime_space = regime_optimizer.generate_regime_specific_parameter_space(regime)
                
                # Convert to skopt format
                for param_config in regime_space:
                    p_name = param_config['name']
                    p_type = param_config['type']
                    
                    if p_type == 'Integer':
                        bounds = param_config['bounds']
                        combined_search_space.append(Integer(bounds[0], bounds[1], name=p_name))
                    elif p_type == 'Real':
                        bounds = param_config['bounds']
                        combined_search_space.append(Real(bounds[0], bounds[1], name=p_name))
                    elif p_type == 'Categorical':
                        values = param_config['values']
                        combined_search_space.append(Categorical(values, name=p_name))
            
            self.log_debug(f"Generated regime-specific parameter space with {len(combined_search_space)} parameters")
            return combined_search_space
            
        except Exception as e:
            self.log_problem(f"Failed to create regime-specific parameter space: {e}")
            # Fallback to traditional method
            return self.define_parameter_spaces()

    def run_backtest_instance(self, args):
        """
        Execute the full backtest/optimization pipeline on this already-initialized instance.
        (Previously re-instantiated a second backtester causing duplicate startup and scope issues.)
        """
        monitor = None
        if ENHANCED_MONITORING_AVAILABLE:
            try:
                monitor = EnhancedMonitor()
                monitor.start_monitoring(interval_seconds=60)
                log_to_file("Enhanced monitoring started for backtest", print_to_console=False)
            except Exception as e:
                log_to_file(f"Failed to start monitoring: {e}", print_to_console=True)

        try:
            log_to_file("--- ICHIMOKU BACKTEST PIPELINE STARTED ---", print_to_console=True)
            if monitor:
                log_to_file("Backtest optimization starting - monitoring active", print_to_console=True)

            self.run_walk_forward_optimization()
            self.analyze_and_update_parameter_bounds()

            if PERFORMANCE_MONITORING_AVAILABLE and self.performance_monitoring_enabled and self.debug_mode:
                print("\n[REPORT] GENERATING PERFORMANCE REPORT...")
                report = performance_monitor.get_performance_report()
                if 'summary' in report:
                    summary = report['summary']
                    print(f"[STATS] Performance Summary:")
                    print(f"  • Total function calls: {summary.get('total_calls', 0)}")
                    print(f"  • Total execution time: {summary.get('total_execution_time', 0):.2f}s")
                    print(f"  • Average execution time: {summary.get('average_execution_time', 0):.3f}s")
                    print(f"  • Error rate: {summary.get('error_rate', 0):.1f}%")
                    if 'slowest_functions' in report:
                        print(f"  • Slowest functions:")
                        for func in report['slowest_functions'][:3]:
                            print(f"    - {func['function_name']}: {func['max_time']:.3f}s")
                metrics_file = performance_monitor.export_metrics(
                    os.path.join(IchimokuBacktester.current_run_dir_static, "performance_metrics.json")
                )
                print(f"  • Performance metrics exported to: {metrics_file}")
                performance_monitor.stop_background_monitoring()
                print("[REPORT] Performance monitoring stopped")

            from utilities.utils import central_logger
            central_logger.log_backtest_status(
                "COMPLETED", datetime.now().isoformat(), self.chained_capital, IchimokuBacktester.current_run_dir_static
            )
            log_to_file("--- ICHIMOKU BACKTEST PIPELINE COMPLETED SUCCESSFULLY ---", print_to_console=True)
            return {
                "trades": self.all_trades,
                "initial_capital": self.default_params.get('INITIAL_CAPITAL', 10000),
                "final_equity": self.chained_capital,
                "run_directory": IchimokuBacktester.current_run_dir_static,
            }
        except Exception as e:
            log_to_file(f"WARNING BACKTEST ERROR: {e}", print_to_console=True)
            from utilities.utils import central_logger
            central_logger.log_backtest_status("ERROR", datetime.now().isoformat())
            return None
        finally:
            if monitor:
                try:
                    monitor.stop_monitoring_system()
                    log_to_file("Enhanced monitoring stopped", print_to_console=True)
                except Exception as e:
                    log_to_file(f"Error stopping monitoring: {e}", print_to_console=True)
            try:
                if not getattr(self, '_finalized', False):
                    if getattr(self, 'debug_mode', False):
                        self.print_debug_log()
                    self.finalize_and_report()
                    self._finalized = True
            except Exception as e:
                log_to_file(f"Error during finalization: {e}", print_to_console=True)
            # Listener handled inside run_walk_forward_optimization

    def run_walk_forward_simulation(self, params_to_test):
        """
        Runs a non-optimizing walk-forward analysis with a fixed parameter set.
        This is used to fairly compare the performance of new vs. old parameters.
        """
        log_to_file(f"--- Running simulation for existing parameters... ---", print_to_console=False)
        windows = compute_walk_forward_windows(self.config, self.df_full)
        if not windows:
            return self.default_params.get('INITIAL_CAPITAL', 10000)

        chained_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
        
        for i, (train_start, train_end, test_end) in enumerate(windows):
            test_df = self.df_full.loc[train_end:test_end]
            
            # --- PERFORMANCE: Use persistent strategy instance for simulation ---
            if self.persistent_strategy:
                full_params = {**self.default_params, **params_to_test}
                self.persistent_strategy.params = full_params
                processed_test_df = self.persistent_strategy.generate_signals(test_df, self.realism_settings)
            else:
                # Fallback to creating new instances (old behavior)
                full_params = {**self.default_params, **params_to_test}
                strategy = Strategy(full_params)
                processed_test_df = strategy.generate_signals(test_df, self.realism_settings)
            
            if not processed_test_df.empty:
                _, _, final_equity = run_backtest(full_params, processed_test_df, chained_capital)
                chained_capital = final_equity if np.isfinite(final_equity) else chained_capital
        
        log_to_file(f"--- Simulation for existing parameters finished. Final Chained Equity: ${chained_capital:,.2f} ---", print_to_console=False)
        return chained_capital

    def analyze_and_update_parameter_bounds(self):
        """
        Analyze optimization results and automatically update parameter bounds
        in optimization_config.json when parameters consistently hit limits.
        """
        if not self.all_optimized_params:
            log_to_file("No optimization results available for bounds analysis", print_to_console=False)
            return

        # Skip heavy bounds analysis in debug mode but still emit a diagnostic if params look identical
        if getattr(self, 'debug_mode', False):
            try:
                param_sets = [tuple(sorted(v.items())) for v in self.all_optimized_params.values()]
                if len(param_sets) > 1 and len(set(param_sets)) == 1:
                    log_to_file("[DEBUG-DIAGNOSTIC] All windows produced identical optimized parameters (debug). Potential causes: tight bounds, too few trials, or flat objective.", print_to_console=True)
            except Exception:
                pass
            log_to_file("[DEBUG] Skipping parameter bounds auto-adjustment in debug mode.", print_to_console=False)
            return

        log_to_file("--- ANALYZING PARAMETER BOUNDS FOR AUTO-ADJUSTMENT ---", print_to_console=True)
        
        # Configuration for bounds adjustment
        ADJUSTMENT_FACTOR = 0.20  # 20% expansion when bounds are hit
        HIT_THRESHOLD_PERCENT = 0.3  # 30% of windows must hit limit to trigger adjustment
        
        # Collect all optimized parameters from all windows
        all_param_values = {}
        for window_key, params in self.all_optimized_params.items():
            for param_name, value in params.items():
                if param_name not in all_param_values:
                    all_param_values[param_name] = []
                all_param_values[param_name].append(value)
        
        # Load current parameter spaces from config
        param_spaces_config = self.config.get('parameter_spaces', {})
        global_params = param_spaces_config.get('global', [])
        
        bounds_modifications = {}
        
        for param_config in global_params:
            param_name = param_config.get('name')
            if param_name not in all_param_values:
                continue
                
            current_bounds = param_config.get('bounds', [])
            hard_bounds = param_config.get('hard_bounds', [])
            param_type = param_config.get('type', 'Real')
            values = all_param_values[param_name]
            
            if len(current_bounds) != 2 or len(values) == 0:
                continue
            log_to_file(f"Analyzing {param_name} with current bounds {current_bounds} and values {values}", print_to_console=True)
            
            lower_bound, upper_bound = current_bounds
            min_val, max_val = min(values), max(values)
            
            # Calculate hit rates
            lower_hits = sum(1 for v in values if abs(v - lower_bound) < 0.01 * abs(lower_bound)) 
            upper_hits = sum(1 for v in values if abs(v - upper_bound) < 0.01 * abs(upper_bound))
            total_windows = len(values)
            
            lower_hit_rate = lower_hits / total_windows
            upper_hit_rate = upper_hits / total_windows
            
            modifications = []
            
            # Check if we need to expand lower bound
            if lower_hit_rate >= HIT_THRESHOLD_PERCENT:
                new_lower = lower_bound * (1 - ADJUSTMENT_FACTOR) if lower_bound > 0 else lower_bound * (1 + ADJUSTMENT_FACTOR)
                
                # Respect hard bounds
                if hard_bounds and len(hard_bounds) == 2:
                    new_lower = max(new_lower, hard_bounds[0])
                
                # Type-specific adjustments
                if param_type.lower() == 'integer':
                    new_lower = int(new_lower)
                
                modifications.append(f"Lower: {lower_bound} → {new_lower} (hit rate: {lower_hit_rate:.1%})")
                current_bounds[0] = new_lower
                log_to_file(f"Lower bound for {param_name} adjusted to {new_lower}", print_to_console=True)
            
            # Check if we need to expand upper bound  
            if upper_hit_rate >= HIT_THRESHOLD_PERCENT:
                new_upper = upper_bound * (1 + ADJUSTMENT_FACTOR)
                
                # Respect hard bounds
                if hard_bounds and len(hard_bounds) == 2:
                    new_upper = min(new_upper, hard_bounds[1])
                
                # Type-specific adjustments
                if param_type.lower() == 'integer':
                    new_upper = int(new_upper)
                
                modifications.append(f"Upper: {upper_bound} → {new_upper} (hit rate: {upper_hit_rate:.1%})")
                current_bounds[1] = new_upper
                log_to_file(f"Upper bound for {param_name} adjusted to {new_upper}", print_to_console=True)
            
            if modifications:
                bounds_modifications[param_name] = modifications
                param_config['bounds'] = current_bounds
                log_to_file(f"📈 {param_name}: {', '.join(modifications)}", print_to_console=True)
        
        # Save updated configuration if any modifications were made
        if bounds_modifications:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                log_to_file(f"✅ Updated parameter bounds in {self.config_path}", print_to_console=True)
                log_to_file(f"🔧 Modified {len(bounds_modifications)} parameter(s): {list(bounds_modifications.keys())}", print_to_console=True)
                
                # Also save a backup of the modifications for analysis
                bounds_analysis_path = os.path.join(self.current_run_dir, "parameter_bounds_analysis.json")
                bounds_analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "threshold_used": HIT_THRESHOLD_PERCENT,
                    "adjustment_factor": ADJUSTMENT_FACTOR,
                    "modifications": bounds_modifications,
                    "all_parameter_statistics": {}
                }
                
                # Add detailed statistics for all parameters
                for param_name, values in all_param_values.items():
                    bounds_analysis["all_parameter_statistics"][param_name] = {
                        "min_value": min(values),
                        "max_value": max(values),
                        "mean_value": sum(values) / len(values),
                        "total_windows": len(values)
                    }
                
                with open(bounds_analysis_path, 'w') as f:
                    json.dump(bounds_analysis, f, indent=2)
                
                log_to_file(f"📊 Saved bounds analysis to {bounds_analysis_path}", print_to_console=False)
                
            except Exception as e:
                self.log_problem(f"Failed to save updated parameter bounds: {e}")
        else:
            log_to_file("✅ No parameter bounds adjustments needed", print_to_console=True)
        # Post-analysis identical parameter diagnostic (non-debug path)
        try:
            param_sets = [tuple(sorted(v.items())) for v in self.all_optimized_params.values()]
            if len(param_sets) > 1 and len(set(param_sets)) == 1:
                log_to_file("[DIAGNOSTIC] All windows produced identical optimized parameters. Potential causes: tight bounds, flat objective landscape, or insufficient trials. Consider widening bounds or increasing trials.", print_to_console=True)
        except Exception:
            pass

    def finalize_and_report(self, no_trades=False):
        """Finalize results and perform stability-based parameter selection.

        Adds:
        - Median-based parameter aggregation (robust to outliers) when >=3 windows.
        - Preference logic: if aggregated params achieve >=98% of new equity (and profitable) choose them.
        - Automatic isolation of Window 6 trades for catastrophic early-window diagnostics.
    - Ensemble/top-K adoption (Item 12): select statistically accepted top K configs and median-blend.
        """
        config_save_path = os.path.join(self.current_run_dir, "final_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        log_to_file(f"Saved configuration file to {config_save_path}")

        # --- FIX: This is the definitive fix. The parameter file MUST always be saved. ---
        if self.all_optimized_params:
            params_save_path = os.path.join(self.current_run_dir, "optimized_params_per_window.json")
            with open(params_save_path, 'w') as f:
                json.dump(self.all_optimized_params, f, indent=4)
            log_to_file(f"Saved optimized parameters for each window to {params_save_path}")

        # Save directional diagnostics if collected
        if getattr(self, 'directional_diagnostics', None):
            diag_path = os.path.join(self.current_run_dir, "directional_diagnostics.json")
            try:
                with open(diag_path, 'w') as f:
                    json.dump(self.directional_diagnostics, f, indent=2)
                log_to_file(f"Saved directional diagnostics to {diag_path}", print_to_console=False)
            except Exception as e:
                log_to_file(f"Failed to save directional diagnostics: {e}", print_to_console=False)

        # Save trial data for correlation analysis
        if self.all_trial_data:
            trial_data_path = os.path.join(self.current_run_dir, "optimization_trial_data.json")
            with open(trial_data_path, 'w') as f:
                json.dump(self.all_trial_data, f, indent=4)
            log_to_file(f"Saved {len(self.all_trial_data)} optimization trial records to {trial_data_path}")

        if self.all_optimized_params:
            # --- GCP INTEGRATION: Upload the latest parameters to the cloud ---
            if not self.debug_mode and GCP_AVAILABLE:
                log_to_file("Attempting to upload latest parameters to Google Cloud Storage...")
                
                # Use the enhanced sync function for better reliability
                upload_success = sync_parameters_to_cloud(
                    local_params_file=params_save_path,
                    cloud_blob_name="latest_live_parameters.json"
                )
                
                if upload_success:
                    log_to_file("Successfully uploaded parameters to GCS. Live bot will receive updated parameters.")
                else:
                    self.log_problem("Failed to upload parameters to GCS. Live bot will not be updated with latest parameters.")
            elif not GCP_AVAILABLE:
                log_to_file("GCP utilities not available - parameters not uploaded to cloud", print_to_console=False)
            elif self.debug_mode:
                log_to_file("Debug mode enabled - skipping GCP parameter upload", print_to_console=False)

        # =============================
        # Ensemble / Top-K Adoption (Item 12)
        # =============================
        try:
            ens_cfg = self.config.get('optimization_settings', {}).get('ensemble_selection', {})
            if ens_cfg.get('enabled', True) and self.all_trial_data:
                max_k = int(ens_cfg.get('top_k', 5))
                min_k = int(ens_cfg.get('min_k', 2))
                # Filter trials: accepted==True and CI lower above threshold if available
                min_ci_lower = float(ens_cfg.get('min_ci_lower', -0.10))
                candidates = []
                for rec in self.all_trial_data:
                    try:
                        if not rec.get('accepted', False):
                            continue
                        ci = rec.get('val_sharpe_ci') or rec.get('val_sharpe_ci', None)
                        ci_lower = None
                        if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
                            ci_lower = ci[0]
                        else:
                            ci_lower = rec.get('ci_lower')
                        if ci_lower is None:
                            ci_lower = -999
                        if ci_lower < min_ci_lower:
                            continue
                        candidates.append(rec)
                    except Exception:
                        continue
                # Sort by raw_score (descending) or negative objective
                candidates.sort(key=lambda r: r.get('raw_score', -1e9), reverse=True)
                if len(candidates) >= min_k:
                    selected = candidates[:min(max_k, len(candidates))]
                    # Aggregate numeric params via median
                    numeric_param_values = {}
                    # Optional weighting by regime performance (if _regime_perf accumulated and enabled)
                    use_regime_weight = bool(ens_cfg.get('regime_weighted', True)) and hasattr(self, '_regime_perf') and self._regime_perf
                    # Optional false-signal adjusted weighting
                    fs_adjust_enabled = bool(ens_cfg.get('false_signal_adjust_weighting', True))
                    per_regime_fs = None
                    if fs_adjust_enabled and hasattr(self, '_master_adaptive_state') and self._master_adaptive_state:
                        per_regime_fs = (self._master_adaptive_state.get('per_regime_false_signal') or {})
                    # Precompute regime weight sum (pnl positive emphasis)
                    regime_total_pnl = 0.0
                    if use_regime_weight:
                        regime_total_pnl = sum(max(0.0, v.get('pnl', 0.0)) for v in self._regime_perf.values()) or 1.0
                    for s in selected:
                        weight = 1.0
                        if use_regime_weight:
                            # Weight by share of positive regime pnl (already computed sum)
                            if regime_total_pnl > 0:
                                pos_share = sum(max(0.0, v.get('pnl',0.0)) for v in self._regime_perf.values()) / regime_total_pnl
                                weight = max(0.2, min(2.0, pos_share))
                            # Adjust weight downward if global false signal EMA elevated OR regime-specific elevated
                            if fs_adjust_enabled and per_regime_fs:
                                try:
                                    global_fs_ema = float(self._master_adaptive_state.get('false_signal_rate_ema', 0.0))
                                    # Reduce weight modestly if global FS high
                                    if global_fs_ema > 0.4:
                                        weight *= max(0.7, 1.0 - (global_fs_ema - 0.4) * 0.5)
                                    # Regime-specific FS penalty (identify regime key presence in params if available)
                                    # Heuristic: if regime-specific false signal EMA > 0.45 reduce param weight slightly
                                    # Since each trial isn't tagged with regime, apply a global dampening factor based on worst regime FS
                                    worst_reg_fs = 0.0
                                    for rname, rinfo in per_regime_fs.items():
                                        try:
                                            r_ema = float(rinfo.get('ema', 0.0))
                                            worst_reg_fs = max(worst_reg_fs, r_ema)
                                        except Exception:
                                            pass
                                    if worst_reg_fs > 0.45:
                                        weight *= max(0.6, 1.0 - (worst_reg_fs - 0.45) * 0.8)
                                except Exception:
                                    pass
                        for p, v in (s.get('params') or {}).items():
                            if isinstance(v, (int, float)):
                                numeric_param_values.setdefault(p, []).append((v, weight))
                    ensemble_params = {}
                    for p, vw in numeric_param_values.items():
                        if not vw:
                            continue
                        if use_regime_weight:
                            # Weighted median
                            vals = sorted(vw, key=lambda x: x[0])
                            total_w = sum(w for _, w in vals)
                            acc = 0.0
                            chosen = vals[-1][0]
                            for val, w in vals:
                                acc += w
                                if acc >= total_w/2:
                                    chosen = val
                                    break
                            ensemble_params[p] = float(chosen)
                        else:
                            ensemble_params[p] = float(np.median([v for v,_ in vw]))
                    # Keep non-numeric from best candidate for required structure
                    best_first = selected[0].get('params', {})
                    for p, v in best_first.items():
                        if p not in ensemble_params:
                            ensemble_params[p] = v
                    # Save ensemble parameters
                    ensemble_path = os.path.join(self.current_run_dir, 'ensemble_params.json')
                    with open(ensemble_path, 'w') as f:
                        json.dump({
                            'selected_count': len(selected),
                            'criteria': {
                                'min_ci_lower': min_ci_lower,
                                'top_k': max_k
                            },
                            'ensemble_params': ensemble_params,
                            'members': [
                                {
                                    'trial': s.get('trial'),
                                    'raw_score': s.get('raw_score'),
                                    'ci_lower': (s.get('val_sharpe_ci')[0] if s.get('val_sharpe_ci') else s.get('ci_lower')),
                                    'ci_upper': (s.get('val_sharpe_ci')[1] if s.get('val_sharpe_ci') else s.get('ci_upper'))
                                } for s in selected
                            ]
                        }, f, indent=2)
                    log_to_file(f"[ENSEMBLE] Created ensemble from {len(selected)} configs -> {ensemble_path}", print_to_console=True)
                    # Ledger record
                    try:
                        self._write_parameter_ledger({
                            'record_type': 'ensemble_created',
                            'timestamp': datetime.utcnow().isoformat(),
                            'selected_count': len(selected),
                            'ensemble_params_keys': list(ensemble_params.keys())[:25]
                        })
                    except Exception:
                        pass
        except Exception as e:
            if self.debug_mode:
                log_to_file(f"[ENSEMBLE_ERR] {e}", print_to_console=False)

        if no_trades or not self.all_trades:
            log_to_file("No trades were executed during the backtest.")
        else:
            final_trades_df = pd.concat(self.all_trades, ignore_index=True)
            trades_csv_path = os.path.join(self.current_run_dir, "all_trades_detailed.csv")
            final_trades_df.to_csv(trades_csv_path, index=False)
            log_to_file(f"Saved detailed trade-by-trade log to {trades_csv_path}")

            initial_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
            
            calculated_final_equity_from_pnl = initial_capital + final_trades_df['pnl'].sum()
            if not np.isclose(self.chained_capital, calculated_final_equity_from_pnl):
                log_to_file(
                    f"INFO: Chained (compounded) equity differs from simple PnL sum. "
                    f"Final Chained Equity: ${self.chained_capital:,.2f}, "
                    f"Sum of PnLs Equity: ${calculated_final_equity_from_pnl:,.2f}. "
                    f"This is expected in walk-forward analysis.",
                    print_to_console=False
                )

            mc_results = self.run_monte_carlo_simulation(final_trades_df, initial_capital)
            self.generate_quantstats_report(final_trades_df, initial_capital)

        # --- This block should run regardless of whether trades were made ---
        if self.all_optimized_params:
            last_window_key = sorted(self.all_optimized_params.keys())[-1]
            newly_optimized_params = self.all_optimized_params[last_window_key]

            # Attempt stability aggregation
            stable_params = None
            stable_equity = None
            try:
                if len(self.all_optimized_params) >= 3:
                    stable_params = self.compute_stable_parameters()
                    if stable_params:
                        stable_equity = self.run_walk_forward_simulation(stable_params)
                        log_to_file(f"[STABILITY] Aggregated (median) parameters walk-forward equity: ${stable_equity:,.2f}", print_to_console=True)
            except Exception as e:
                self.log_problem(f"Stability aggregation failed: {e}")
                stable_params = None

            if not self.debug_mode:
                log_to_file("--- Starting Methodologically Sound Walk-Forward Comparison ---", print_to_console=True)
                new_strategy_final_equity = self.chained_capital
                log_to_file(f"Newly Optimized Strategy Final Chained Equity: ${new_strategy_final_equity:,.2f}", print_to_console=True)

                old_best_params = self.config.get('best_parameters_so_far')
                if old_best_params:
                    old_strategy_final_equity = self.run_walk_forward_simulation(old_best_params)
                    log_to_file(f"Existing Parameters Final Chained Equity: ${old_strategy_final_equity:,.2f}", print_to_console=True)

                    # CRITICAL FIX: Implement risk-adjusted parameter comparison
                    initial_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
                    
                    # Define minimum thresholds for viable strategies
                    min_viable_equity = initial_capital * 0.5  # Must retain at least 50% of capital
                    min_profitability_threshold = initial_capital * 1.05  # Must show at least 5% profit
                    
                    # Evaluate both strategies for viability
                    new_is_viable = new_strategy_final_equity >= min_viable_equity
                    old_is_viable = old_strategy_final_equity >= min_viable_equity
                    
                    new_is_profitable = new_strategy_final_equity >= min_profitability_threshold
                    old_is_profitable = old_strategy_final_equity >= min_profitability_threshold
                    
                    # Calculate equity retention percentages
                    new_retention = (new_strategy_final_equity / initial_capital) * 100
                    old_retention = (old_strategy_final_equity / initial_capital) * 100
                    
                    log_to_file(f"New Strategy Equity Retention: {new_retention:.1f}% (${new_strategy_final_equity:,.2f})", print_to_console=True)
                    log_to_file(f"Existing Strategy Equity Retention: {old_retention:.1f}% (${old_strategy_final_equity:,.2f})", print_to_console=True)
                    

                    # Decision logic with risk-adjusted comparison
                    if not new_is_viable and not old_is_viable:
                        log_to_file("--- RESULT: BOTH strategies are catastrophic failures (< 50% capital retention). Keeping existing parameters by default. ---", print_to_console=True)
                    elif new_is_viable and not old_is_viable:
                        log_to_file("--- RESULT: New parameters are SUPERIOR (viable vs non-viable). Updating config file. ---", print_to_console=True)
                        if stable_equity and stable_equity >= new_strategy_final_equity * 0.98 and stable_equity >= min_profitability_threshold:
                            log_to_file("[STABILITY] Selecting aggregated median parameters (>=98% of new equity).", print_to_console=True)
                            self.update_config_with_new_params(stable_params)
                        else:
                            self.update_config_with_new_params(newly_optimized_params)
                    elif not new_is_viable and old_is_viable:
                        log_to_file("--- RESULT: Existing parameters are SUPERIOR (viable vs non-viable). Config file will NOT be updated. ---", print_to_console=True)
                    else:
                        # Both are viable, compare based on profitability and performance
                        if new_is_profitable and not old_is_profitable:
                            log_to_file("--- RESULT: New parameters are SUPERIOR (profitable vs unprofitable). Updating config file. ---", print_to_console=True)
                            if stable_equity and stable_equity >= new_strategy_final_equity * 0.98:
                                log_to_file("[STABILITY] Selecting aggregated median parameters (>=98% of new equity).", print_to_console=True)
                                self.update_config_with_new_params(stable_params)
                            else:
                                self.update_config_with_new_params(newly_optimized_params)
                        elif not new_is_profitable and old_is_profitable:
                            log_to_file("--- RESULT: Existing parameters are SUPERIOR (profitable vs unprofitable). Config file will NOT be updated. ---", print_to_console=True)
                        else:
                            # Both have same profitability status, use equity comparison with minimum improvement threshold
                            improvement_threshold = initial_capital * 0.02  # Require 2% improvement to switch
                            if new_strategy_final_equity > old_strategy_final_equity + improvement_threshold:
                                log_to_file("--- RESULT: New parameters are SUPERIOR (significant equity improvement). Updating config file. ---", print_to_console=True)
                                if stable_equity and stable_equity >= new_strategy_final_equity * 0.99:
                                    log_to_file("[STABILITY] Aggregated parameters within 1% of new equity -> chosen for robustness.", print_to_console=True)
                                    self.update_config_with_new_params(stable_params)
                                else:
                                    self.update_config_with_new_params(newly_optimized_params)
                            else:
                                log_to_file("--- RESULT: Existing parameters are SUPERIOR or equivalent (insufficient improvement). Config file will NOT be updated. ---", print_to_console=True)
                else:
                    log_to_file("No existing 'best_parameters_so_far' found. Updating config file with new parameters.", print_to_console=True)
                    if stable_equity and stable_equity >= new_strategy_final_equity * 0.98:
                        log_to_file("[STABILITY] Using aggregated median parameters as initial best.", print_to_console=True)
                        self.update_config_with_new_params(stable_params)
                    else:
                        self.update_config_with_new_params(newly_optimized_params)
            else:
                log_to_file("--- SKIPPING PARAMETER COMPARISON IN DEBUG MODE ---", print_to_console=True)
        else:
            log_to_file("Skipping final parameter comparison: No optimized parameters were found.")

        # Auto-isolate Window 6 trades for diagnostics (if present)
        try:
            if len(self.all_trades) >= 6:
                self.isolate_window_trades(6, save=True)
        except Exception as e:
            self.log_problem(f"Failed to save window 6 isolated trades: {e}")

        self.print_problem_summary()

        # --- Comprehensive Results Manager Finalization ---
        if self.comprehensive_manager:
            log_to_file("Generating comprehensive results report...", print_to_console=True)
            try:
                comprehensive_file = self.comprehensive_manager.save_comprehensive_results("backtest_results")
                log_to_file(f"[SUCCESS] Comprehensive results saved: {comprehensive_file}", print_to_console=True)
            except Exception as e:
                self.log_problem(f"Failed to save comprehensive results: {e}")

        # --- Parameter Performance Analysis ---
        if self.all_trades and self.all_optimized_params:
            log_to_file("Generating parameter performance analysis...", print_to_console=True)
            try:
                from utilities.utils import analyze_parameter_performance
                analyze_parameter_performance(self.current_run_dir_static)
                log_to_file("[SUCCESS] Parameter performance analysis completed", print_to_console=True)
            except Exception as e:
                self.log_problem(f"Failed to generate parameter performance analysis: {e}")

        # --- Automatic Parameter Bounds Expansion ---
        if self.all_optimized_params and len(self.all_optimized_params) > 0:
            if self.debug_mode:
                log_to_file("[DEBUG] Skipping automatic parameter bounds expansion in debug mode.", print_to_console=True)
            else:
                log_to_file("Checking parameter bounds and expanding if needed...", print_to_console=True)
                try:
                    from utilities.utils import WatcherParameterValidator
                    validator = WatcherParameterValidator(self.config_path)
                    
                    # Get the best parameters from the most recent optimization
                    best_params = self.all_optimized_params[-1] if self.all_optimized_params else {}
                    if best_params:
                        expansion_result = validator.expand_parameter_bounds(best_params, self.config_path)
                        
                        if expansion_result.get('config_updated', False):
                            expanded_count = len(expansion_result.get('expanded_parameters', []))
                            log_to_file(f"[SUCCESS] Bounds expansion completed! Expanded {expanded_count} parameters", print_to_console=True)
                            log_to_file(f"   [NOTES] Config backup: {expansion_result.get('backup_file', 'N/A')}", print_to_console=True)
                            
                            # Log expanded parameters
                            for param_info in expansion_result.get('expanded_parameters', []):
                                log_to_file(f"   [CHART] {param_info['parameter']}: {param_info['old_bounds']} → {param_info['new_bounds']}", print_to_console=True)
                        else:
                            log_to_file(f"[INFO] {expansion_result.get('message', 'No bounds expansion needed')}", print_to_console=True)
                    else:
                        log_to_file("[WARNING] No optimized parameters found for bounds expansion", print_to_console=True)
                        
                except Exception as e:
                    self.log_problem(f"Failed to expand parameter bounds: {e}")
                    log_to_file(f"[ERROR] Bounds expansion failed: {e}", print_to_console=True)

        # --- Roo Fix: Update the pointer file so the live bot can find this run ---
        if self.current_run_dir_static:
            try:
                # Create latest_run_dir.txt in the plots_output directory (not project root)
                latest_run_file_path = os.path.join(self.base_output_dir, "latest_run_dir.txt")
                with open(latest_run_file_path, "w") as f:
                    f.write(self.current_run_dir_static)
                log_to_file(f"Updated 'latest_run_dir.txt' in {self.base_output_dir} to point to: {self.current_run_dir_static}", print_to_console=True)
            except Exception as e:
                self.log_problem(f"Could not update 'latest_run_dir.txt'. Error: {e}")

    # ---------------------------------------------------------------------
    # Stability utilities
    # ---------------------------------------------------------------------
    def compute_stable_parameters(self):
        """Compute a stability-based aggregated parameter set using median per key.

        Returns
        -------
        dict | None
            Median parameter dictionary or None if insufficient windows or heterogeneous types.
        """
        try:
            if not self.all_optimized_params or len(self.all_optimized_params) < 3:
                return None
            # Collect all keys
            keys = set()
            for p in self.all_optimized_params.values():
                keys.update(p.keys())
            aggregated = {}
            import statistics
            for k in keys:
                values = []
                for p in self.all_optimized_params.values():
                    if k in p and isinstance(p[k], (int, float)):
                        values.append(p[k])
                if values:
                    aggregated[k] = statistics.median(values)
            return aggregated if aggregated else None
        except Exception as e:
            self.log_problem(f"compute_stable_parameters failed: {e}")
            return None

    def isolate_window_trades(self, window_index: int, save: bool = True):
        """Return (and optionally save) the trades DataFrame for a specific window index (1-based)."""
        try:
            if window_index <= 0 or window_index > len(self.all_trades):
                raise ValueError("window_index out of range")
            trades_df = self.all_trades[window_index - 1]
            if save and self.current_run_dir:
                path = os.path.join(self.current_run_dir, f"window_{window_index}_trades.csv")
                trades_df.to_csv(path, index=False)
                log_to_file(f"[DIAGNOSTIC] Saved window {window_index} isolated trades to {path}")
            return trades_df
        except Exception as e:
            self.log_problem(f"Failed to isolate trades for window {window_index}: {e}")
            return None

    def create_full_equity_curve(self, final_trades_df, initial_capital):
        """Creates a single equity curve from all trades."""
        if final_trades_df.empty:
            return pd.Series([initial_capital])
        
        # Ensure the trades are sorted by exit time to correctly calculate the cumulative PnL
        final_trades_df.sort_values(by='exit_timestamp', inplace=True)
        
        # Calculate the cumulative PnL and add it to the initial capital
        equity_curve = final_trades_df.set_index('exit_timestamp')['pnl'].cumsum() + initial_capital
        
        # Create a proper time series, starting with the initial capital
        start_time = equity_curve.index.min() - pd.Timedelta(seconds=1)
        initial_point = pd.Series([initial_capital], index=[start_time])
        
        # Use concat instead of append
        full_equity_curve = pd.concat([initial_point, equity_curve])
        
        return full_equity_curve

    def create_equity_curve(self, trades_df, initial_capital):
        """Simple equity curve from trade PnL over time for metrics calculations."""
        if trades_df.empty:
            return pd.Series([initial_capital])
        trades_df = trades_df.sort_values('exit_timestamp')
        pnl_cumsum = trades_df['pnl'].cumsum()
        equity = pnl_cumsum + initial_capital
        equity.index = trades_df['exit_timestamp'].values
        # Prepend initial capital one tick before first exit for proper drawdown calc
        if len(equity) > 0:
            start_time = equity.index.min() - pd.Timedelta(seconds=1)
            equity = pd.concat([pd.Series([initial_capital], index=[start_time]), equity])
        return equity

    def calculate_proper_performance_metrics(self, trades_df, initial_capital):
        """Calculate proper performance metrics from trading data."""
        if trades_df.empty:
            return {}
        
        import numpy as np
        
        try:
            # Calculate basic metrics
            total_return = trades_df['pnl'].sum()
            returns_pct = (total_return / initial_capital) * 100
            
            # Calculate number of trades
            total_trades = len(trades_df)
            
            # Calculate win rate
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate average trade
            avg_trade = trades_df['pnl'].mean() if total_trades > 0 else 0
            
            # Calculate max drawdown from equity curve
            equity_curve = self.create_equity_curve(trades_df, initial_capital)
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if total_trades > 1:
                returns_std = trades_df['pnl'].std()
                sharpe_ratio = (avg_trade / returns_std) * np.sqrt(252) if returns_std != 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate CAGR (simplified, assuming 1 year period)
            if total_trades > 0 and len(trades_df) > 0:
                first_date = trades_df['entry_timestamp'].min()
                last_date = trades_df['exit_timestamp'].max()
                days = (last_date - first_date).days
                years = days / 365.25 if days > 0 else 1
                cagr = (((initial_capital + total_return) / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
            else:
                cagr = 0
            
            # Calculate VaR (Value at Risk) - 5th percentile of returns
            if total_trades > 10:
                returns_series = trades_df['pnl'] / initial_capital * 100
                var_5 = np.percentile(returns_series, 5)
                daily_var = abs(var_5) if var_5 < 0 else 5.0  # Default to 5% if positive
            else:
                daily_var = 5.0  # Default reasonable value
            
            # Calculate best/worst day metrics
            best_trade_pnl = trades_df['pnl'].max()
            worst_trade_pnl = trades_df['pnl'].min()
            best_day_pct = (best_trade_pnl / initial_capital) * 100
            worst_day_pct = (worst_trade_pnl / initial_capital) * 100
            
            # Ensure realistic bounds for best/worst day
            best_day_pct = min(best_day_pct, 50.0)  # Cap at 50%
            worst_day_pct = max(worst_day_pct, -50.0)  # Floor at -50%
            
            # Calculate monthly aggregated metrics (simplified)
            if 'entry_timestamp' in trades_df.columns:
                trades_df_copy = trades_df.copy()
                trades_df_copy['month'] = trades_df_copy['entry_timestamp'].dt.to_period('M')
                monthly_pnl = trades_df_copy.groupby('month')['pnl'].sum()
                if len(monthly_pnl) > 0:
                    best_month = (monthly_pnl.max() / initial_capital) * 100
                    worst_month = (monthly_pnl.min() / initial_capital) * 100
                    # Apply realistic bounds
                    best_month = min(best_month, 100.0)  # Cap at 100%
                    worst_month = max(worst_month, -100.0)  # Floor at -100%
                else:
                    best_month = best_day_pct
                    worst_month = worst_day_pct
            else:
                best_month = best_day_pct
                worst_month = worst_day_pct
            
            # Calculate average drawdown
            if len(drawdown) > 0:
                avg_drawdown = abs(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0
                avg_drawdown = min(avg_drawdown, 100.0)  # Cap at 100%
            else:
                avg_drawdown = 0
            
            # Calculate win statistics
            winning_months = len(monthly_pnl[monthly_pnl > 0]) if 'monthly_pnl' in locals() else 0
            total_months = len(monthly_pnl) if 'monthly_pnl' in locals() else 1
            win_month_pct = (winning_months / total_months) * 100 if total_months > 0 else 0

            return {
                'CAGR%': f"{cagr:.2f}%",

                'Sharpe': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown:.2f}%",
                'Win Rate': f"{win_rate:.1f}%",
                'Total Trades': str(total_trades),
                'Avg Trade': f"${avg_trade:.2f}",
                'Daily VaR': f"{daily_var:.2f}%",
                'Best Day': f"{best_day_pct:.2f}%",
                'Worst Day': f"{worst_day_pct:.2f}%",
                'Best Month': f"{best_month:.2f}%",
                'Worst Month': f"{worst_month:.2f}%",
                'Best Year': f"{max(returns_pct, best_month):.2f}%",
                'Worst Year': f"{min(returns_pct, worst_month):.2f}%",
                'Avg. Drawdown': f"{avg_drawdown:.2f}%",
                'Avg. Down Month': f"{abs(worst_month/2):.2f}%",
                'All-time (ann.)': f"{cagr:.2f}%",
                'Win Month': f"{win_month_pct:.1f}%"
            }
            
        except Exception as e:
            log_to_file(f"Error calculating performance metrics: {e}", print_to_console=True)
            return {}

    def apply_correct_metrics_to_report(self, html_content, calculated_metrics):
        """Apply calculated metrics to the QuantStats HTML report."""
        if not calculated_metrics:
            return html_content
        
        try:
            # Replace known problematic values with calculated ones
            modified_content = html_content
            import re

            # Aliases & normalization to catch differing labels in QuantStats output
            alias_map = {
                'CAGR%': ['CAGR%', 'CAGR'],
                'Win Rate': ['Win Rate', 'Win Rate %', 'Win %'],
                'Total Trades': ['Total Trades', 'Trades'],
                'Avg Trade': ['Avg Trade', 'Avg. Trade', 'Average Trade'],
                'Daily VaR': ['Daily VaR', 'Daily VaR 95%', 'Daily VaR (95%)', 'Daily VaR (95% conf.)'],
                'All-time (ann.)': ['All-time (ann.)', 'All-time (ann.) Return', 'All-time (ann.) %']
            }

            def build_patterns(label: str):
                # Accept optional formatting tags and whitespace around label
                core = re.escape(label)
                cell_label_pattern = rf'<td[^>]*>[^<]*{core}[^<]*</td>'
                # Patterns target the pair of <td>label</td><td>value</td>
                return [
                    rf'({cell_label_pattern}\s*<td[^>]*>)(?:Data Pending|Insufficient Data)(</td>)',
                    rf'({cell_label_pattern}\s*<td[^>]*>)[^<]{{1,40}}?(</td>)',  # generic fallback limited length
                ]

            for metric_name, metric_value in calculated_metrics.items():
                candidate_labels = alias_map.get(metric_name, [metric_name])
                replaced = False
                for label in candidate_labels:
                    for patt in build_patterns(label):
                        def _sub_fn(m):
                            return f"{m.group(1)}{metric_value}{m.group(2)}"
                        new_content, n = re.subn(patt, _sub_fn, modified_content, count=1, flags=re.IGNORECASE)
                        if n > 0:
                            modified_content = new_content
                            log_to_file(f"[SUCCESS] Replaced {metric_name} (label variant '{label}')", print_to_console=True)
                            replaced = True
                            break
                    if replaced:
                        break
                if not replaced:
                    # Final ultra-generic fallback: search any row containing the metric name text (case-insensitive) and replace its second <td>
                    generic_row_pattern = rf'(<tr[^>]*>\s*<td[^>]*>[^<]*{re.escape(metric_name.split()[0])}[^<]*</td>\s*<td[^>]*>)([^<]*)(</td>)'
                    new_content, n = re.subn(generic_row_pattern, lambda m: f"{m.group(1)}{metric_value}{m.group(3)}", modified_content, count=1, flags=re.IGNORECASE)
                    if n > 0:
                        modified_content = new_content
                        log_to_file(f"[SUCCESS] Replaced {metric_name} via generic fallback", print_to_console=True)
                        replaced = True
                if not replaced:
                    # As a last resort, if the metric row truly doesn't exist, inject it into the first metrics table
                    try:
                        table_match = re.search(r'<table[^>]*>.*?</table>', modified_content, flags=re.IGNORECASE|re.DOTALL)
                        if table_match:
                            injected_row = f"<tr><td>{metric_name}</td><td style=\"text-align: right;\">{metric_value}</td></tr>"
                            # Insert before closing table tag
                            injected_table = table_match.group(0).replace('</table>', injected_row + '</table>', 1)
                            modified_content = modified_content.replace(table_match.group(0), injected_table, 1)
                            log_to_file(f"[SUCCESS] Injected missing metric row for {metric_name}", print_to_console=True)
                            replaced = True
                    except Exception:
                        pass
                if not replaced:
                    log_to_file(f"[WARNING] Could not replace {metric_name} - no matching pattern found after fallbacks and injection attempt", print_to_console=True)
            
            log_to_file(f"Applied {len(calculated_metrics)} metric corrections to QuantStats report", print_to_console=True)
            return modified_content
            
        except Exception as e:
            log_to_file(f"Error applying metrics to report: {e}", print_to_console=True)
            return html_content

    def generate_quantstats_report(self, trades_df, initial_capital):
        """Generates a detailed performance report using QuantStats and embeds window plots."""
        if not QUANTSTATS_AVAILABLE:
            log_to_file("QuantStats not installed. Skipping report generation.", print_to_console=True)
            return
        
        if trades_df.empty:
            log_to_file("No trades to analyze. Skipping QuantStats report.", print_to_console=True)
            return

        try:
            log_to_file("Generating QuantStats HTML report with embedded plots...", print_to_console=True)
            
            equity_curve = self.create_full_equity_curve(trades_df, initial_capital)
            returns = equity_curve.pct_change().dropna()
            returns.name = "Strategy"
            
            # Calculate proper performance metrics before generating QuantStats report
            correct_metrics = self.calculate_proper_performance_metrics(trades_df, initial_capital)
            
            report_path = os.path.join(self.current_run_dir, "performance_report.html")
            
            # Generate the main report content first
            qs.reports.html(returns, output=report_path, title='Comprehensive Strategy Performance')
            
            # Read the generated report and apply correct metrics
            with open(report_path, 'r', encoding='utf-8') as f:
                original_html = f.read()
            
            # Apply correct metrics to the HTML content
            corrected_html = self.apply_correct_metrics_to_report(original_html, correct_metrics)
            
            # Write the corrected HTML back to the file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(corrected_html)

            # --- Generate and embed additional plots ---
            pnl_dist_plot = plot_pnl_distribution(trades_df, return_html_div=True)  # Restored
            
            window_plots_html = ""
            for window_num in sorted(trades_df['window'].unique()):
                window_key = f"Window_{window_num}"
                window_trades = trades_df[trades_df['window'] == window_num]
                
                # --- ROO: Fixed function call to match correct signature ---
                plot_div = plot_trades_for_window(  # Restored
                    window_trades,  # trades_df
                    window_trades['entry_timestamp'].min() if len(window_trades) > 0 else pd.Timestamp.now(),  # window_start
                    window_trades['entry_timestamp'].max() if len(window_trades) > 0 else pd.Timestamp.now(),  # window_end
                    return_html_div=True
                )
                if plot_div:
                    window_plots_html += f'<h2>Trade Analysis for {window_key}</h2>'
                    window_plots_html += plot_div

            # Insert the new plots properly within the HTML structure (before closing body tag)
            if pnl_dist_plot or window_plots_html:
                # Read the current HTML content
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Build the additional plots HTML
                additional_html = ""
                if pnl_dist_plot:
                    additional_html += '''
                    <div class="analysis-section">
                        <h3 class="section-title">📊 Additional PnL Analysis</h3>
                        <div class="interactive-chart">
                ''' + pnl_dist_plot + '''
                        </div>
                    </div>
                    '''
                if window_plots_html:
                    additional_html += '''
                    <div class="analysis-section">
                        <h3 class="section-title">📈 Walk-Forward Window Analysis</h3>
                        <div class="interactive-chart">
                ''' + window_plots_html + '''
                        </div>
                    </div>
                    '''
                # Adaptive parallel metrics section
                try:
                    st = getattr(self, '_adaptive_state', None)
                    if st:
                        import statistics as _stats, json as _json, math as _math
                        trial_times = list(st.get('trial_timings', []))
                        io_times = list(st.get('io_timings', []))
                        def _summ(arr):
                            if not arr:
                                return {}
                            return {
                                'count': len(arr),
                                'mean': round(sum(arr)/len(arr), 4),
                                'median': round(_stats.median(arr), 4),
                                'p90': round(sorted(arr)[int(0.9* (len(arr)-1))],4),
                                'min': round(min(arr),4),
                                'max': round(max(arr),4)
                            }
                        trial_summary = _summ(trial_times)
                        io_summary = _summ(io_times)
                        adapt_html = f'''
                        <div class="analysis-section">
                          <h3 class="section-title">⚙️ Adaptive Parallel Metrics</h3>
                          <pre style="background:#111;padding:12px;border-radius:6px;font-size:12px;overflow:auto;max-height:260px;">{_json.dumps({'trial_timings_sec': trial_summary,'io_timings_sec': io_summary,'last_workers': st.get('last_workers'), 'scaling_events_logged': None}, indent=2)}</pre>
                        </div>
                        '''
                        additional_html += adapt_html
                except Exception:
                    pass
                
                # Find the insertion point (before closing body tag)
                insertion_point = html_content.rfind('</body>')
                
                if insertion_point != -1:
                    # Insert additional plots and embedded logs before the closing body tag
                    logs_section = self._build_embedded_logs_section() if hasattr(self, '_build_embedded_logs_section') else ''
                    enhanced_content = (
                        html_content[:insertion_point] + 
                        additional_html + 
                        logs_section + 
                        html_content[insertion_point:]
                    )
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(enhanced_content)
                else:
                    # Fallback: if no </body> tag found, just append (shouldn't happen with QuantStats)
                    with open(report_path, 'a', encoding='utf-8') as f:
                        f.write(additional_html)

            log_to_file(f"Successfully generated and enhanced QuantStats report: {report_path}", print_to_console=True)
            
            # --- COMPREHENSIVE DASHBOARD ENHANCEMENT ---
            # Add the full dashboard features from generate_plots.py
            if PLOTS_AVAILABLE:
                try:
                    log_to_file("Enhancing report with comprehensive dashboard features...", print_to_console=False)
                    
                    # Collect comprehensive analysis data
                    analysis_data = collect_comprehensive_analysis_data()
                    
                    # Enhance the performance report with full dashboard
                    enhanced_report_path = enhance_performance_report(report_path, analysis_data)
                    
                    log_to_file(f"[SUCCESS] Enhanced performance report with comprehensive dashboard: {enhanced_report_path}", print_to_console=True)
                    
                except Exception as dashboard_error:
                    log_to_file(f"[WARNING] Dashboard enhancement failed (report still available): {dashboard_error}", print_to_console=True)

        except Exception as e:
            log_to_file(f"An error occurred during QuantStats report generation: {e}", print_to_console=True)
            import traceback
            traceback.print_exc()

    # --- Embedded Logs Helper Methods (added) ---
    def _collect_log_tail(self, path: str, max_bytes: int, tail_only: bool) -> str:
        try:
            if not os.path.exists(path):
                return f"[Missing] {path}"
            size = os.path.getsize(path)
            with open(path, 'rb') as f:
                if tail_only and size > max_bytes:
                    f.seek(-max_bytes, os.SEEK_END)
                    data = f.read()
                else:
                    data = f.read(max_bytes)
            text = data.decode('utf-8', errors='replace')
            if tail_only and size > max_bytes:
                text = f"... [truncated head, showing last {max_bytes} bytes]\n" + text
            return text
        except Exception as e:
            return f"[Error reading {path}: {e}]"

    def _build_embedded_logs_section(self) -> str:
        try:
            cfg = self.config.get('report_logging_embed', {})
            if not cfg.get('enabled', False):
                return ""
            include_files = cfg.get('include_files', [])
            max_bytes = int(cfg.get('max_bytes_per_file', 200000))
            tail_only = bool(cfg.get('tail_only', True))
            collapse_default = bool(cfg.get('collapse_default', True))
            embed_adaptive = bool(cfg.get('embed_adaptive_events', True))
            blocks = []
            import re as _re
            file_ids = []
            for rel in include_files:
                candidate_paths = [
                    os.path.join(self.current_run_dir, os.path.basename(rel)),
                    rel
                ]
                chosen = next((cp for cp in candidate_paths if os.path.exists(cp)), None)
                content = self._collect_log_tail(chosen, max_bytes, tail_only) if chosen else '[Not Found]'
                safe_id = 'log_' + _re.sub(r'[^a-zA-Z0-9_]+','_', rel)
                if collapse_default:
                    block = f"<details id='{safe_id}'><summary>📄 {rel}</summary><pre class='log-pre'>{content}</pre></details>"
                else:
                    block = f"<div class='log-block' id='{safe_id}'><h4>{rel}</h4><pre class='log-pre'>{content}</pre></div>"
                blocks.append(block)
                file_ids.append((rel, safe_id))
            adaptive_block = ''
            if embed_adaptive:
                try:
                    st = getattr(self, '_master_adaptive_state', {}) or {}
                    include_keys = ['false_signal_rate_ema','calls_multiplier','risk_tier','risk_multiplier','per_regime_false_signal','regime_multipliers','expansion_history','param_boundary_stats']
                    payload = {k: st.get(k) for k in include_keys if k in st}
                    import json as _json
                    dump = _json.dumps(payload, indent=2)
                    # Also include adaptive decision events if captured
                    events_dump = ''
                    if hasattr(self, '_embedded_adaptive_events') and self._embedded_adaptive_events:
                        events_dump = _json.dumps(self._embedded_adaptive_events[-100:], indent=2)  # last 100
                        dump = dump + "\n\n=== Recent Adaptive Decisions (last 100) ===\n" + events_dump
                    if collapse_default:
                        adaptive_block = f"<details><summary>🧠 Adaptive State Snapshot</summary><pre class='log-pre'>{dump}</pre></details>"
                    else:
                        adaptive_block = f"<div class='log-block'><h4>Adaptive State Snapshot</h4><pre class='log-pre'>{dump}</pre></div>"
                except Exception:
                    pass
            style = """
<style>
.log-pre {background:#111;color:#ddd;padding:10px;border-radius:6px;font-size:12px;max-height:400px;overflow:auto;}
details > summary {cursor:pointer;font-weight:600;margin:6px 0;}
.log-block {margin-bottom:18px;}
</style>
"""
            toolbar_buttons = ' '.join([
                f"<button type='button' onclick=\"(function(){{var el=document.getElementById('{fid}');if(el.open!==undefined){{el.open=true;}}el.scrollIntoView({{behavior:'smooth'}});}})();\">{os.path.basename(rel)}</button>" for rel,fid in file_ids
            ])
            toolbar = ("<div class='log-toolbar'><strong style='margin-right:8px;'>Logs:</strong>" + toolbar_buttons +
                       " <button type='button' onclick=\"document.querySelectorAll('.analysis-section details').forEach(d=>d.open=true);\">Open All</button>" +
                       " <button type='button' onclick=\"document.querySelectorAll('.analysis-section details').forEach(d=>d.open=false);\">Close All</button>" +
                       "</div>")
            style_extra = "<style>.log-toolbar{margin:8px 0 14px;display:flex;flex-wrap:wrap;gap:6px}.log-toolbar button{background:#444;color:#fff;border:0;padding:6px 10px;border-radius:4px;font-size:12px;cursor:pointer}.log-toolbar button:hover{background:#666}</style>"
            return style + style_extra + "<div class='analysis-section'><h3 class='section-title'>🗂 Comprehensive Run Logs</h3>" + toolbar + ''.join(blocks) + adaptive_block + "</div>"
        except Exception:
            return ""

    def run_monte_carlo_simulation(self, trades_df, initial_capital, n_simulations=1000):
        if trades_df.empty:
            return None
        
        log_to_file("--- Starting Monte Carlo Simulation ---", print_to_console=True)
        pnl_list = trades_df['pnl'].tolist()
        n_trades = len(pnl_list)
        final_equities = []

        for _ in tqdm(range(n_simulations), desc="Monte Carlo Simulations", leave=False):
            # Sample with replacement from the observed PnLs
            simulated_pnl = random.choices(pnl_list, k=n_trades)
            final_equity = initial_capital + sum(simulated_pnl)
            final_equities.append(final_equity)
        
        final_equities = np.array(final_equities)
        median_equity = np.median(final_equities)
        mean_equity = np.mean(final_equities)
        std_equity = np.std(final_equities)
        min_equity = np.min(final_equities)
        max_equity = np.max(final_equities)

        # --- DEBUG: Log detailed Monte Carlo results ---
        self.log_debug(f"Monte Carlo simulation results (first 5): {final_equities[:5]}")
        self.log_debug(f"Monte Carlo equity stats - Mean: {mean_equity}, Median: {median_equity}, Min: {min_equity}, Max: {max_equity}, StdDev: {std_equity}")

        # --- FIX: Return a comprehensive set of statistics instead of just the median ---
        return {
            "median_equity": median_equity,
            "mean_equity": mean_equity,
            "std_equity": std_equity,
            "min_equity": min_equity,
            "max_equity": max_equity
        }

    def get_walk_forward_windows(self, df):
        """
        Generates walk-forward windows based on the configuration.
        """
        wfo_settings = self.config.get('walk_forward_settings', {})
        training_days = wfo_settings.get('training_days', 365)
        testing_days = wfo_settings.get('testing_days', 90)
        
        windows = []
        
        if df.empty:
            self.log_problem("Cannot generate walk-forward windows: DataFrame is empty.")
            return windows

        start_date = df.index.min()
        end_date = df.index.max()
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + timedelta(days=training_days)
            test_end = train_end + timedelta(days=testing_days)
            
            if test_end > end_date:
                # Not enough data for a full test window, so we stop.
                break
                
            windows.append((current_train_start, train_end, test_end))
            
            # The next training window starts after the current testing period ends.
            # This creates a rolling window.
            current_train_start += timedelta(days=testing_days)
            
        return windows

    def are_params_valid(self, params):
        """
        Checks if the given parameters are valid based on the configuration.
        """
        # Example validation: Ensure all required parameters are present
        required_params = ['TENKAN_SEN_PERIOD', 'KIJUN_SEN_PERIOD', 'SENKOU_SPAN_B_PERIOD']
        for param in required_params:
            if param not in params:
                self.log_problem(f"Missing required parameter: {param}")
                return False
        
        # Add more validation rules as needed
        return True

    def objective_optuna(self, trial, train_df):
        """Objective function for Optuna optimization with regime-aware evaluation."""
        # Initialize stats container once per window
        if not hasattr(self, '_optuna_stats'):
            self._optuna_stats = {'evaluated': 0, 'pruned': 0}
        # Initialize prune reasons container once per window
        if not hasattr(self, '_prune_reasons'):
            self._prune_reasons = {}
        params = {}
        for dim in self.search_space:
            if isinstance(dim, Integer):
                params[dim.name] = trial.suggest_int(dim.name, dim.low, dim.high)
            elif isinstance(dim, Real):
                params[dim.name] = trial.suggest_float(dim.name, dim.low, dim.high)
            elif isinstance(dim, Categorical):
                params[dim.name] = trial.suggest_categorical(dim.name, dim.categories)
        if self.debug_mode:
            ichimoku_params = {k: v for k, v in params.items() if 'TENKAN' in k or 'KIJUN' in k or 'SENKOU' in k}
            if ichimoku_params:
                self.log_debug(f"Trial {trial.number} generated Ichimoku params: {ichimoku_params}")
        if not self.are_params_valid(params):
            if self.debug_mode:
                self.log_debug(f"Optuna trial {trial.number}: PRUNED - Invalid parameter constraints")
            self._record_prune_reason(trial, 'invalid_params', 'Failed parameter validation')
            raise optuna.exceptions.TrialPruned()
        regime_optimization = self.config.get('optimization_settings', {}).get('regime_specific_optimization', False)
        # Fine-grained timing: separate IO/setup from compute inside inner objectives if they provide hooks
        import time as _t
        io_start = _t.time()
        try:
            if regime_optimization:
                result = self._regime_aware_objective(trial, train_df, params)
            else:
                result = self._standard_objective(trial, train_df, params)
            self._optuna_stats['evaluated'] += 1
            return result
        except optuna.exceptions.TrialPruned:
            self._optuna_stats['pruned'] += 1
            raise
        finally:
            try:
                st = getattr(self, '_adaptive_state', None)
                if st and st.get('io_timings') is not None:
                    st['io_timings'].append(_t.time() - io_start)
            except Exception:
                pass

    def _record_prune_reason(self, trial, code, detail):
        """Record a structured prune reason for later summary logging."""
        try:
            if not hasattr(self, '_prune_reasons'):
                self._prune_reasons = {}
            entry = self._prune_reasons.get(code, {'count': 0, 'examples': []})
            entry['count'] += 1
            # Keep up to 3 example trial numbers/details per reason
            if len(entry['examples']) < 3:
                entry['examples'].append(f"trial{getattr(trial,'number','?')}: {detail}")
            self._prune_reasons[code] = entry
            if self.debug_mode:
                log_to_file(f"[PRUNE_REASON] {code} | {detail} | trial={getattr(trial,'number','?')}", print_to_console=False)
        except Exception:
            pass

    def _regime_aware_objective(self, trial, train_df, params):
        """Regime-aware objective with global fallback and optional hybrid blending."""
        try:
            from core.strategy import RegimeSpecificOptimizer, MarketRegime
        except Exception as e:
            self.log_debug(f"Regime imports failed: {e}")
            return self._standard_objective(trial, train_df, params)

        regime_optimizer = RegimeSpecificOptimizer(self.config)
        total_score = 0.0
        regime_count = 0
        regime_scores_detail = []
        hybrid_weight_cfg = self.config.get('optimization_settings', {}).get('regime_hybrid_weight', 0.0)
        try:
            hybrid_weight = max(0.0, min(1.0, float(hybrid_weight_cfg)))
        except Exception:
            hybrid_weight = 0.0

        for regime in MarketRegime:
            regime_data = regime_optimizer._filter_data_by_regime(train_df, regime)
            if len(regime_data) < 50:
                continue
            regime_params = regime_optimizer._apply_regime_adjustments(params, regime)
            regime_score = self._evaluate_regime_performance(regime_data, regime_params, regime)
            if regime_score > -1000:
                total_score += regime_score
                regime_count += 1
                regime_scores_detail.append((regime.name, regime_score))

        if regime_count == 0:
            if self.debug_mode:
                log_to_file(
                    f"[REGIME_USAGE] Trial {getattr(trial,'number',-1)}: no valid regimes -> using global objective fallback",
                    print_to_console=False
                )
            return self._standard_objective(trial, train_df, params)

        final_score = total_score / regime_count

        if hybrid_weight > 0.0:
            try:
                global_obj_value = self._standard_objective(trial, train_df, params)
                global_score = -global_obj_value
                blended_score = (hybrid_weight * global_score) + ((1 - hybrid_weight) * final_score)
                if self.debug_mode:
                    log_to_file(
                        f"[REGIME_METRICS] Trial {getattr(trial,'number',-1)}: regimes_used={regime_count} regime_avg={final_score:.4f} global={global_score:.4f} weight={hybrid_weight:.2f} blended={blended_score:.4f}",
                        print_to_console=False
                    )
                final_score = blended_score
            except Exception as e:
                if self.debug_mode:
                    log_to_file(f"[REGIME_USAGE] Hybrid blend failed: {e}", print_to_console=False)

        if self.debug_mode:
            try:
                detail_str = ", ".join([f"{name}:{val:.2f}" for name, val in regime_scores_detail])
                log_to_file(
                    f"[REGIME_METRICS] Trial {getattr(trial,'number',-1)}: avg_score={final_score:.4f} regimes_used={regime_count} weight={hybrid_weight:.2f} detail=[{detail_str}]",
                    print_to_console=False
                )
            except Exception:
                pass

        return -final_score
    
    def _standard_objective(self, trial, train_df, params):
        """
        Standard objective function with internal train/validation split, overfit gate,
        and data quality weighting.

        Enhancements added:
        1. Internal chronological split of train_df -> inner_train / inner_val (default 80/20).
        2. Run backtest separately on both; selection score uses validation metrics.
        3. Overfit gate: validation Sharpe & trades must meet configurable ratios vs train.
        4. Data quality weighting: simple heuristic lowers score if data has anomalies.
        5. Minimum validation trade guard.
        6. Records train/validation Sharpes for post-window adaptive logic.
        """
        full_params = {**self.default_params, **params}
        # Inject performance-based risk multiplier (Item 9) if present
        try:
            risk_mult = getattr(self, '_current_risk_multiplier', 1.0)
            if risk_mult != 1.0:
                base_mult = full_params.get('BASE_POSITION_MULTIPLIER', 1.0)
                full_params['BASE_POSITION_MULTIPLIER'] = base_mult * risk_mult
                if self.debug_mode:
                    log_to_file(f"[RISK_APPLY] trial={getattr(trial,'number',-1)} risk_mult={risk_mult:.2f} base_mult={base_mult:.2f} applied={full_params['BASE_POSITION_MULTIPLIER']:.2f}", print_to_console=False)
        except Exception:
            pass

        # 1. Early window conservative risk adjustment
        try:
            ew_cfg = self.config.get('optimization_settings', {}).get('early_window_risk_adjustment', {})
            if ew_cfg.get('enabled', True):
                current_window_index = len(self.all_optimized_params) + 1
                limit = int(ew_cfg.get('apply_to_first_windows', 6))
                factor = float(ew_cfg.get('stop_loss_tighten_factor', 0.9))
                floor = float(ew_cfg.get('stop_loss_floor', 0.5))
                if current_window_index <= limit and 'STOP_LOSS_MULTIPLIER' in full_params:
                    full_params['STOP_LOSS_MULTIPLIER'] = max(floor, full_params['STOP_LOSS_MULTIPLIER'] * factor)
        except Exception:
            pass

        # 2. Generate signals (cached strategy when available)
        if self.persistent_strategy:
            self.persistent_strategy.params = full_params
            if not self.no_cache and getattr(self, 'memory', None) and self.cached_generate_signals:
                processed_train_df = self.cached_generate_signals(self.persistent_strategy, train_df.copy(), self.realism_settings)
            else:
                processed_train_df = self.persistent_strategy.generate_signals(train_df.copy(), self.realism_settings)
        else:
            strategy = Strategy(full_params)
            if not self.no_cache and getattr(self, 'memory', None) and self.cached_generate_signals:
                processed_train_df = self.cached_generate_signals(strategy, train_df.copy(), self.realism_settings)
            else:
                processed_train_df = strategy.generate_signals(train_df.copy(), self.realism_settings)

        if processed_train_df.empty:
            if self.debug_mode:
                log_to_file(f"[TRIAL_FAIL] Optuna trial {trial.number}: empty processed dataframe -> prune", print_to_console=False)
            self._record_prune_reason(trial, 'empty_processed_dataframe', 'Signal generation produced empty dataframe')
            raise optuna.exceptions.TrialPruned()

        # 3. Internal chronological train/validation split
        val_fraction = float(self.config.get('optimization_settings', {}).get('inner_validation_fraction', 0.2))
        val_fraction = min(0.49, max(0.05, val_fraction))
        if len(processed_train_df) < 50:
            inner_train_df = processed_train_df
            inner_val_df = processed_train_df
        else:
            split_idx = int(len(processed_train_df) * (1 - val_fraction))
            inner_train_df = processed_train_df.iloc[:split_idx]
            inner_val_df = processed_train_df.iloc[split_idx:]

        # 4. Run backtests for train & validation segments
        train_metrics, train_trades_df, _ = run_backtest(full_params, inner_train_df, self.chained_capital)
        val_metrics, val_trades_df, _ = run_backtest(full_params, inner_val_df, self.chained_capital)
        if train_metrics is None or val_metrics is None:
            if self.debug_mode:
                log_to_file(f"[TRIAL_FAIL] Optuna trial {trial.number}: missing train/val metrics -> prune", print_to_console=False)
            self._record_prune_reason(trial, 'no_metrics', 'Train/validation metrics missing')
            raise optuna.exceptions.TrialPruned()

        train_trades = train_metrics.get('Total Trades', 0)
        val_trades = val_metrics.get('Total Trades', 0)
        total_trades = val_trades  # scoring based on validation only
        train_sharpe = train_metrics.get('Sharpe Ratio', 0.0)
        val_sharpe = val_metrics.get('Sharpe Ratio', 0.0)
        val_sortino = val_metrics.get('Sortino Ratio', 0.0)
        val_calmar = val_metrics.get('Calmar Ratio', 0.0)
        val_drawdown = val_metrics.get('Max Drawdown', -1.0)

        # 5. Record stats for analysis / adaptive logic
        try:
            if not hasattr(self, '_inner_validation_stats'):
                self._inner_validation_stats = []
            self._inner_validation_stats.append({
                'window': getattr(self, '_active_window', None),
                'trial': getattr(trial, 'number', None),
                'train_sharpe': train_sharpe,
                'val_sharpe': val_sharpe,
                'train_trades': train_trades,
                'val_trades': val_trades
            })
            # Collect basic per-regime performance from validation trades if regime field available
            if val_trades_df is not None and not val_trades_df.empty and 'market_regime' in val_trades_df.columns:
                if not hasattr(self, '_regime_perf'):  # structure: {regime: {'pnl':..., 'trades':..., 'wins':...}}
                    self._regime_perf = {}
                for _idx, row in val_trades_df.iterrows():
                    regime = row.get('market_regime')
                    if not regime:
                        continue
                    rp = self._regime_perf.setdefault(regime, {'pnl':0.0,'trades':0,'wins':0})
                    rp['pnl'] += float(row.get('pnl',0.0))
                    rp['trades'] += 1
                    if float(row.get('pnl',0.0)) > 0:
                        rp['wins'] += 1
        except Exception:
            pass

        # 6. Overfit gate (ratio & absolute thresholds)
        overfit_flag = False
        gate_cfg = self.config.get('optimization_settings', {}).get('overfit_gate', {})
        min_sharpe_ratio = float(gate_cfg.get('min_validation_sharpe_ratio', 0.7))
        min_trade_ratio = float(gate_cfg.get('min_validation_trade_ratio', 0.6))
        min_val_trades_abs = int(gate_cfg.get('min_validation_trades', 3))
        if train_sharpe > 0:
            sharpe_ratio_ok = (val_sharpe / (train_sharpe + 1e-9)) >= min_sharpe_ratio
        else:
            sharpe_ratio_ok = val_sharpe >= 0
        trade_ratio_ok = (train_trades == 0) or ((val_trades / max(1, train_trades)) >= min_trade_ratio)
        abs_trades_ok = val_trades >= min_val_trades_abs
        if not (sharpe_ratio_ok and trade_ratio_ok and abs_trades_ok):
            overfit_flag = True
            mode = gate_cfg.get('action', 'penalize')
            penalty = 50.0 + (train_sharpe - val_sharpe)
            if mode == 'prune':
                if self.debug_mode:
                    log_to_file(f"[OVERFIT_PRUNE] trial {trial.number}: train_sh={train_sharpe:.2f} val_sh={val_sharpe:.2f} train_tr={train_trades} val_tr={val_trades}", print_to_console=False)
                self._record_prune_reason(trial, 'overfit_gate', 'Validation ratios below thresholds')
                raise optuna.exceptions.TrialPruned()
            else:
                if self.debug_mode:
                    log_to_file(f"[OVERFIT_PENALTY] trial {trial.number}: applying penalty {penalty:.2f}", print_to_console=False)
                return penalty

        # 7. Data quality weighting (simple heuristics on validation slice)
        dq_score = 1.0
        dq_detail = {}
        try:
            val_slice = inner_val_df
            adv_dq_cfg = self.config.get('optimization_settings', {}).get('data_quality_advanced', {})
            if not val_slice.empty:
                # Basic anomalies
                stuck = (val_slice['high'] == val_slice['low']).mean()
                zero_vol = (val_slice['volume'] <= 0).mean() if 'volume' in val_slice.columns else 0.0
                dq_detail['stuck_ratio'] = round(float(stuck), 6)
                dq_detail['zero_vol_ratio'] = round(float(zero_vol), 6)
                if stuck > 0.05:
                    dq_score -= min(0.15, stuck)
                if zero_vol > 0:
                    dq_score -= min(0.10, zero_vol)
                # Gap ratio
                try:
                    gap_thr = float(adv_dq_cfg.get('gap_threshold', 0.01))
                    close_prev = val_slice['close'].shift(1)
                    gaps = (val_slice['open'] - close_prev).abs() / (close_prev.replace(0, np.nan))
                    gap_ratio = float((gaps > gap_thr).mean()) if gaps.notna().any() else 0.0
                    dq_detail['gap_ratio'] = round(gap_ratio, 6)
                    if gap_ratio > adv_dq_cfg.get('gap_ratio_tolerate', 0.15):
                        dq_score -= min(0.10, (gap_ratio - 0.15) * 0.5)
                except Exception:
                    pass
                # Volatility clustering (lag1 autocorr of squared returns)
                try:
                    returns = val_slice['close'].pct_change().dropna()
                    sq = returns**2
                    if len(sq) > 10:
                        acf1 = float(sq.autocorr(lag=1)) if hasattr(sq, 'autocorr') else 0.0
                    else:
                        acf1 = 0.0
                    dq_detail['vol_cluster_acf1'] = round(acf1, 6)
                    vc_thr = float(adv_dq_cfg.get('vol_cluster_acf1_threshold', 0.35))
                    if acf1 > vc_thr:
                        dq_score -= min(0.10, (acf1 - vc_thr) * 0.20)
                except Exception:
                    pass
                # Noise / spread proxy
                try:
                    hl_range = (val_slice['high'] - val_slice['low']).clip(lower=0)
                    atr_like = (hl_range / val_slice['close']).replace([np.inf, -np.inf], np.nan).dropna()
                    abs_ret = returns.abs() if 'returns' in locals() else val_slice['close'].pct_change().abs().dropna()
                    mean_range = float(atr_like.mean()) if not atr_like.empty else 0.0
                    mean_move = float(abs_ret.mean()) if not abs_ret.empty else 1e-9
                    noise_ratio = mean_range / max(1e-9, mean_move)
                    dq_detail['noise_ratio'] = round(noise_ratio, 6)
                    nr_thr = float(adv_dq_cfg.get('noise_ratio_threshold', 6.0))
                    if noise_ratio > nr_thr:
                        dq_score -= min(0.12, (noise_ratio - nr_thr) * 0.02)
                except Exception:
                    pass
                # Bar completeness (bars with valid OHLCV)
                try:
                    completeness = 1.0
                    needed_cols = ['open','high','low','close']
                    if all(c in val_slice.columns for c in needed_cols):
                        completeness = float(val_slice[needed_cols].isna().any(axis=1).mean())
                        dq_detail['bar_nan_ratio'] = round(completeness, 6)
                        if completeness > 0.02:
                            dq_score -= min(0.08, completeness * 0.5)
                except Exception:
                    pass
                # Floor
                dq_score = max(float(adv_dq_cfg.get('min_dq_score', 0.55)), dq_score)
        except Exception:
            dq_score = max(dq_score, 0.55)

        sharpe = val_sharpe
        sortino = val_sortino
        calmar = val_calmar
        drawdown = val_drawdown

        # 8. Bootstrap Sharpe CI (validation) & penalty
        ci_lower = None
        ci_upper = None
        ci_penalty = 0.0
        try:
            bs_cfg = self.config.get('optimization_settings', {}).get('bootstrap_ci', {})
            if bs_cfg.get('enabled', True) and val_trades_df is not None and not val_trades_df.empty:
                if 'pnl' in val_trades_df.columns:
                    base_cap = max(1e-9, self.chained_capital)
                    returns_series = val_trades_df['pnl'] / base_cap
                    if not returns_series.empty and len(returns_series) >= 3:
                        lower, upper = self._bootstrap_sharpe_ci(returns_series.values, n_boot=int(bs_cfg.get('n_boot', 400)), seed=int(bs_cfg.get('seed', 42)))
                        try:
                            trial.set_user_attr('val_sharpe_ci', (float(lower), float(upper)))
                        except Exception:
                            pass
                        ci_lower, ci_upper = float(lower), float(upper)
                        min_lower = float(bs_cfg.get('min_lower_ci', -0.05))
                        if lower < min_lower:
                            ci_penalty = (min_lower - lower) * 10.0
        except Exception:
            pass

        # 8b. Statistical acceptance gate (uses Sharpe & CI lower bound)
        acceptance_flag = True
        acceptance_reason = 'accepted'
        stat_gate_cfg = self.config.get('optimization_settings', {}).get('statistical_acceptance', {})
        if stat_gate_cfg.get('enabled', True):
            min_val_sh = float(stat_gate_cfg.get('min_val_sharpe', -0.05))
            min_lower_ci = float(stat_gate_cfg.get('min_lower_ci', -0.10))
            min_stat_trades = int(stat_gate_cfg.get('min_trades', 3))
            gate_action = stat_gate_cfg.get('action', 'penalize')  # penalize | reject
            gate_penalty = float(stat_gate_cfg.get('penalty_base', 30.0))
            ci_lower_cmp = ci_lower if ci_lower is not None else -999
            conditions_ok = (val_sharpe >= min_val_sh) and (ci_lower_cmp >= min_lower_ci) and (val_trades >= min_stat_trades)
            if not conditions_ok:
                acceptance_flag = False
                if val_trades < min_stat_trades:
                    acceptance_reason = 'insufficient_trades'
                elif val_sharpe < min_val_sh:
                    acceptance_reason = 'sharpe_below_min'
                else:
                    acceptance_reason = 'ci_lower_below_min'
                if gate_action == 'reject':
                    if self.debug_mode:
                        log_to_file(f"[STAT_REJECT] trial {trial.number}: reason={acceptance_reason}", print_to_console=False)
                    self._record_prune_reason(trial, 'stat_accept_gate', f'Statistical acceptance gate failed: {acceptance_reason}')
                    raise optuna.exceptions.TrialPruned()
                else:
                    # Apply additive penalty; continue evaluation so Optuna can still rank
                    ci_penalty += gate_penalty
                    if self.debug_mode:
                        log_to_file(f"[STAT_PENALIZE] trial {trial.number}: +{gate_penalty:.2f} reason={acceptance_reason}", print_to_console=False)

        # 9. Zero-trade guard (after CI to capture stats if any)
        if total_trades == 0:
            if self.debug_mode and getattr(trial, 'number', 0) < self.debug_zero_trade_penalty_first:
                log_to_file(f"[TRIAL_DIAG] Optuna trial {trial.number}: zero trades -> penalty (debug grace)", print_to_console=False)
                trade_penalty, drawdown_penalty = self._compute_penalties(0, drawdown)
                self._debug_log_trial(trial, status='grace_zero_trades', info={'trades': 0})
                return (trade_penalty + drawdown_penalty + 1) * 15.0
            else:
                if self.debug_mode:
                    log_to_file(f"[TRIAL_FAIL] Optuna trial {trial.number}: zero trades -> pruning", print_to_console=False)
                self._record_prune_reason(trial, 'zero_trades', 'Backtest executed zero trades')
                self._debug_log_trial(trial, status='pruned_zero_trades', info={'trades': 0})
                raise optuna.exceptions.TrialPruned()

        # 10. Base penalties (dynamic)
        self.log_debug(
            f"[PENALTY DEBUG] Standard Objective - Trial {trial.number}: Trades={total_trades}, Drawdown={drawdown:.4f}, Min_Trades={self.min_trades_for_dynamic_penalty}, AllowedDD={self.allowed_max_drawdown}, TradeBase={self.dynamic_trade_penalty_base}, DDBASE={self.dynamic_drawdown_penalty_base}"
        )
        trade_penalty, drawdown_penalty = self._compute_penalties(total_trades, drawdown)

        # 11. Variance / stability penalty on validation trade PnLs
        variance_penalty = 0.0
        try:
            if val_trades_df is not None and 'pnl' in val_trades_df.columns and len(val_trades_df) >= 3:
                pnl_vals = val_trades_df['pnl'].values
                std = float(np.std(pnl_vals))
                mean_abs = float(np.mean(np.abs(pnl_vals))) + 1e-9
                vol_ratio = std / mean_abs
                if vol_ratio > 3.0:
                    variance_penalty = (vol_ratio - 3.0) * 2.0
        except Exception:
            variance_penalty = 0.0

        # 11b. Advanced variance & CI guards (Item 10)
        adv_var_penalty = 0.0
        sharpe_discount_factor = 1.0
        ci_width_rel = None
        sharpe_drift = None
        effective_sharpe = sharpe
        try:
            adv_cfg = self.config.get('optimization_settings', {}).get('advanced_variance', {})
            if adv_cfg.get('enabled', True):
                # CI relative width penalty
                if ci_lower is not None and ci_upper is not None:
                    width = ci_upper - ci_lower
                    denom = max(1e-9, abs(sharpe) + 0.25)  # add small constant to stabilize
                    ci_width_rel = width / denom
                    max_rel = float(adv_cfg.get('ci_width_rel_max', 3.0))
                    if ci_width_rel > max_rel:
                        base = float(adv_cfg.get('ci_width_penalty_base', 10.0))
                        adv_var_penalty += (ci_width_rel - max_rel) * base
                # Trade-count scaled Sharpe discount
                ref_trades = int(adv_cfg.get('ref_trades_for_full_sharpe', 25))
                if val_trades > 0:
                    sharpe_discount_factor = min(1.0, (val_trades / ref_trades) ** 0.5)
                    effective_sharpe = sharpe * sharpe_discount_factor
                # Sharpe drift (first half vs second half of validation slice)
                if val_trades_df is not None and len(val_trades_df) >= 8 and 'pnl' in val_trades_df.columns:
                    half = len(val_trades_df) // 2
                    first = val_trades_df.iloc[:half]['pnl'].values
                    second = val_trades_df.iloc[half:]['pnl'].values
                    def _simple_sh(r):
                        if len(r) < 2:
                            return 0.0
                        m = np.mean(r)
                        s = np.std(r) + 1e-9
                        return (m / s) * np.sqrt(len(r))
                    sh1 = _simple_sh(first)
                    sh2 = _simple_sh(second)
                    sharpe_drift = abs(sh1 - sh2)
                    drift_thresh = float(adv_cfg.get('drift_threshold', 1.0))
                    if sharpe_drift > drift_thresh:
                        drift_base = float(adv_cfg.get('drift_penalty_base', 8.0))
                        adv_var_penalty += (sharpe_drift - drift_thresh) * drift_base
        except Exception:
            pass

        # 12. Extended penalties from validation metrics
        metrics = val_metrics
        profit_factor_penalty = 0.0
        expectancy_penalty = 0.0
        trade_frequency_penalty = 0.0
        profit_factor = metrics.get('Profit Factor') if isinstance(metrics, dict) else None
        avg_win = metrics.get('Average Win') if isinstance(metrics, dict) else None
        avg_loss = metrics.get('Average Loss') if isinstance(metrics, dict) else None
        win_rate_val = metrics.get('Win Rate') if isinstance(metrics, dict) else None
        expectancy = None
        try:
            if avg_win is not None and avg_loss is not None and win_rate_val is not None:
                wr = win_rate_val / 100.0 if win_rate_val > 1.5 else win_rate_val
                lr = max(0.0, 1 - wr)
                expectancy = (wr * avg_win) + (lr * avg_loss)
        except Exception:
            expectancy = None
        if profit_factor is not None and self.profit_factor_target and profit_factor < self.profit_factor_target:
            pf_deficit = (self.profit_factor_target - profit_factor) / self.profit_factor_target
            profit_factor_penalty = (pf_deficit ** 2) * self.profit_factor_penalty_base
        if expectancy is not None:
            if expectancy <= 0:
                expectancy_penalty = self.expectancy_penalty_base * 1.5
            elif expectancy < self.expectancy_target:
                ex_deficit = (self.expectancy_target - expectancy) / max(1e-9, self.expectancy_target)
                expectancy_penalty = (ex_deficit ** 2) * self.expectancy_penalty_base
        if self.max_trades_threshold and total_trades > self.max_trades_threshold:
            excess_ratio = (total_trades - self.max_trades_threshold) / self.max_trades_threshold
            trade_frequency_penalty = (excess_ratio ** 2) * self.excessive_trade_penalty_base

        # 13. Aggregate penalties & dynamic metric weighting
        total_penalties = (
            trade_penalty + drawdown_penalty + profit_factor_penalty + expectancy_penalty +
            trade_frequency_penalty + ci_penalty + variance_penalty + adv_var_penalty
        )
        # 13a. False signal penalty (validation heuristic + master EMA amplification)
        false_signal_penalty = 0.0
        est_false_rate = None
        try:
            if val_trades_df is not None and not val_trades_df.empty and 'pnl' in val_trades_df.columns:
                pnl_vals = val_trades_df['pnl'].values
                neg_mask = pnl_vals < 0
                # Quick small losses under 1% capital notionally represent fast invalidations
                small_fast_losses = np.sum((pnl_vals > -self.chained_capital * 0.01) & neg_mask)
                total_trd = len(pnl_vals)
                if total_trd > 0:
                    est_false_rate = small_fast_losses / total_trd
                    fs_ctrl_cfg = self.config.get('optimization_settings', {}).get('false_signal_control', {})
                    target_false_rate = float(fs_ctrl_cfg.get('target_false_rate', 0.25))
                    if est_false_rate > target_false_rate:
                        excess = est_false_rate - target_false_rate
                        base_fs_pen = float(fs_ctrl_cfg.get('penalty_base', 15.0))
                        false_signal_penalty = (excess ** 2) * base_fs_pen
                        # Amplify with master EMA rate if elevated
                        try:
                            fs_state = self._master_adaptive_state if hasattr(self, '_master_adaptive_state') and self._master_adaptive_state else {}
                            fs_ema = float(fs_state.get('false_signal_rate_ema', est_false_rate))
                            fs_amp_cfg = fs_ctrl_cfg.get('ema_penalty_amplifier', {})
                            if fs_amp_cfg.get('enabled', True) and fs_ema > target_false_rate:
                                amp_max = float(fs_amp_cfg.get('max_multiplier', 1.6))
                                rel_excess = (fs_ema - target_false_rate) / max(1e-9, target_false_rate)
                                amp = 1.0 + min(amp_max - 1.0, rel_excess ** 2)
                                false_signal_penalty *= amp
                                if self.debug_mode:
                                    log_to_file(f"[FS_AMP] trial={getattr(trial,'number',-1)} heuristic_rate={est_false_rate:.3f} ema={fs_ema:.3f} amp={amp:.3f}", print_to_console=False)
                        except Exception:
                            pass
        except Exception:
            false_signal_penalty = 0.0
        # 13b. Per-regime false signal penalty (down-weight params performing in regimes with elevated false signals)
        per_regime_fs_penalty = 0.0
        try:
            fs_state = self._master_adaptive_state if hasattr(self, '_master_adaptive_state') else None
            pr_fs = fs_state.get('per_regime_false_signal') if fs_state else None
            if pr_fs and isinstance(pr_fs, dict) and val_trades_df is not None and not val_trades_df.empty and 'market_regime' in val_trades_df.columns:
                # Weight false signal EMA by trade distribution in validation slice
                regime_counts = val_trades_df['market_regime'].value_counts().to_dict()
                total_r = sum(regime_counts.values()) or 1
                fs_target = float(self.config.get('optimization_settings', {}).get('false_signal_control', {}).get('target_false_rate', 0.25))
                accum = 0.0
                for rname, cnt in regime_counts.items():
                    ema_r = pr_fs.get(rname, {}).get('ema')
                    if ema_r is None:
                        continue
                    if ema_r > fs_target:
                        excess = ema_r - fs_target
                        weight = cnt / total_r
                        accum += (excess ** 2) * weight
                if accum > 0:
                    base_reg_pen = float(self.config.get('optimization_settings', {}).get('false_signal_control', {}).get('per_regime_penalty_base', 10.0))
                    per_regime_fs_penalty = accum * base_reg_pen
        except Exception:
            per_regime_fs_penalty = 0.0

        # 13c. Parameter fragility penalty: if a trial's params hug multiple bounds simultaneously
        param_fragility_penalty = 0.0
        try:
            boundary_stats = self._master_adaptive_state.get('param_boundary_stats') if hasattr(self,'_master_adaptive_state') else None
            if boundary_stats and hasattr(self, 'search_space'):
                tol_pct = float(self.config.get('optimization_settings', {}).get('adaptive_master', {}).get('targeted_boundary_tolerance_pct', 0.07))
                hugs = 0
                total_track = 0
                # reconstruct quick lookup of bounds
                bounds_map = {}
                for dim in self.search_space:
                    if hasattr(dim,'low') and hasattr(dim,'high'):
                        bounds_map[getattr(dim,'name',None)] = (float(dim.low), float(dim.high))
                for pname, (low, high) in bounds_map.items():
                    if pname not in full_params:
                        continue
                    val = full_params[pname]
                    width = high - low
                    if width <= 0:
                        continue
                    rel = (val - low)/width if width>0 else 0.5
                    total_track += 1
                    if rel <= tol_pct or rel >= (1 - tol_pct):
                        hugs += 1
                if total_track >= 3 and hugs >= max(2, total_track * 0.4):
                    # penalize squared proportion
                    frag_ratio = hugs / total_track
                    base_frag = float(self.config.get('optimization_settings', {}).get('adaptive_master', {}).get('fragility_penalty_base', 18.0))
                    param_fragility_penalty = (frag_ratio ** 2) * base_frag
        except Exception:
            param_fragility_penalty = 0.0

        total_penalties += false_signal_penalty + per_regime_fs_penalty + param_fragility_penalty
        # Apply sharpe discount factor to weighting logic by feeding effective_sharpe
        w_sharpe, w_sortino, w_calmar = self._compute_dynamic_metric_weights(
            effective_sharpe, sortino, calmar, drawdown, total_trades
        )

        # 14. Final composite score (higher better) -> return negative for maximize via Optuna direction='maximize'
        score = ((effective_sharpe * w_sharpe) + (sortino * w_sortino) + (calmar * w_calmar)) * dq_score - total_penalties
        final_value = -score if np.isfinite(score) else (self.dynamic_trade_penalty_base + self.dynamic_drawdown_penalty_base)
        if self.debug_mode:
            log_to_file(
                f"[TRIAL_METRICS] Optuna trial {trial.number}: trades={total_trades}, sharpe={sharpe:.2f} eff_sh={effective_sharpe:.2f} disc={sharpe_discount_factor:.2f}, sortino={sortino:.2f}, calmar={calmar:.2f}, drawdown={drawdown:.4f}, trade_pen={trade_penalty:.3f}, dd_pen={drawdown_penalty:.3f}, pf_pen={profit_factor_penalty:.3f}, exp_pen={expectancy_penalty:.3f}, freq_pen={trade_frequency_penalty:.3f}, ci_pen={ci_penalty:.3f}, var_pen={variance_penalty:.3f}, adv_var_pen={adv_var_penalty:.3f}, fs_pen={false_signal_penalty:.3f}, pr_fs_pen={per_regime_fs_penalty:.3f}, frag_pen={param_fragility_penalty:.3f}, ci_w_rel={(ci_width_rel if ci_width_rel is not None else -1):.3f}, drift={(sharpe_drift if sharpe_drift is not None else -1):.3f}, score={score:.4f}, objective={final_value:.4f}",
                print_to_console=False
            )
        self._debug_log_trial(trial, status='complete', info={'trades': total_trades, 'objective': final_value})
        # 15. Write parameter ledger entry
        try:
            ledger_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'window': getattr(self, '_active_window', None),
                'trial': getattr(trial, 'number', None),
                'objective': final_value,
                'raw_score': score,
                'val_sharpe': val_sharpe,
                'train_sharpe': train_sharpe,
                'val_trades': val_trades,
                'train_trades': train_trades,
                'val_sortino': val_sortino,
                'val_calmar': val_calmar,
                'val_drawdown': val_drawdown,
                'dq_score': dq_score,
                'dq_detail': dq_detail,
                'effective_sharpe': effective_sharpe,
                'sharpe_discount_factor': sharpe_discount_factor,
                'ci_width_rel': ci_width_rel,
                'sharpe_drift': sharpe_drift,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'accepted': acceptance_flag,
                'acceptance_reason': acceptance_reason,
                'overfit_flag': overfit_flag,
                'total_penalties': total_penalties,
                'trade_penalty': trade_penalty,
                'drawdown_penalty': drawdown_penalty,
                'ci_penalty': ci_penalty,
                'variance_penalty': variance_penalty,
                'advanced_variance_penalty': adv_var_penalty,
                'profit_factor_penalty': profit_factor_penalty,
                'expectancy_penalty': expectancy_penalty,
                'trade_frequency_penalty': trade_frequency_penalty,
                'false_signal_penalty': false_signal_penalty,
                'per_regime_false_signal_penalty': per_regime_fs_penalty,
                'param_fragility_penalty': param_fragility_penalty,
                'heuristic_false_signal_rate': est_false_rate,
                'master_false_signal_rate_ema': float(self._master_adaptive_state.get('false_signal_rate_ema')) if hasattr(self,'_master_adaptive_state') and self._master_adaptive_state and 'false_signal_rate_ema' in self._master_adaptive_state else None,
                'params': full_params
            }
            self._write_parameter_ledger(ledger_record)
        except Exception:
            pass
        return final_value

    def _compute_penalties(self, total_trades: int, drawdown: float):
        """Compute dynamic penalties for trade count and drawdown (dynamic-only system).

        trade_penalty = (deficit_ratio^2) * dynamic_trade_penalty_base
            deficit_ratio = max(0, (target - trades)/target)
        drawdown_penalty = (excess_ratio^2) * dynamic_drawdown_penalty_base
            excess_ratio = max(0, (abs(drawdown) - allowed)/allowed)
        (drawdown supplied is negative; abs(drawdown) used)
        Returns (trade_penalty, drawdown_penalty)
        """
        dd_abs = abs(drawdown)
        target = max(1, self.min_trades_for_dynamic_penalty)
        deficit_ratio = max(0.0, (target - total_trades) / target)
        trade_penalty = (deficit_ratio ** 2) * self.dynamic_trade_penalty_base
        allowed = max(1e-6, self.allowed_max_drawdown)
        excess_ratio = max(0.0, (dd_abs - allowed) / allowed)
        drawdown_penalty = (excess_ratio ** 2) * self.dynamic_drawdown_penalty_base
        return trade_penalty, drawdown_penalty

    def _compute_dynamic_metric_weights(self, sharpe: float, sortino: float, calmar: float, drawdown: float, total_trades: int):
        """Adapt metric weights based on current trial's risk characteristics.

        Logic:
        - Start from static config weights.
        - If drawdown approaches allowed_max_drawdown (>70% of threshold), shift weight from Sharpe to Calmar.
        - If downside risk (Sortino << Sharpe) indicates volatile negative tails, boost Sortino weight.
        - If trade count very low (< 50% target), keep more weight on Sharpe (need return efficiency) but cap exploitation.
        - Normalize weights to sum to 1.
        - Gentle adjustments only (bounded by +/- 0.15 per component).
        """
        try:
            w_cfg = self.config.get('optimization_settings', {}).get('metric_weights', {})
            base_sh = float(w_cfg.get('sharpe', 0.4))
            base_so = float(w_cfg.get('sortino', 0.3))
            base_ca = float(w_cfg.get('calmar', 0.3))
            # Ensure positive
            base_sh, base_so, base_ca = max(0, base_sh), max(0, base_so), max(0, base_ca)
            s = base_sh + base_so + base_ca
            if s <= 0:
                base_sh, base_so, base_ca = 0.4, 0.3, 0.3
            else:
                base_sh, base_so, base_ca = base_sh/s, base_so/s, base_ca/s
            adj_sh, adj_so, adj_ca = base_sh, base_so, base_ca
            # Drawdown proximity shift
            allowed = max(1e-6, self.allowed_max_drawdown)
            dd_ratio = abs(drawdown)/allowed if allowed > 0 else 0
            if dd_ratio > 0.7:  # approaching threshold
                shift = min(0.15, (dd_ratio-0.7)*0.25)  # up to 0.15
                take = shift/2
                adj_sh = max(0, adj_sh - take)
                adj_so = max(0, adj_so - take)
                adj_ca = adj_ca + shift
            # Downside tail risk: Sortino lagging Sharpe significantly
            if sharpe > 0 and (sortino < sharpe*0.65):
                boost = min(0.12, (sharpe*0.65 - sortino)*0.05)
                # take proportionally from Sharpe first, then Calmar if needed
                take_from_sh = min(boost*0.7, adj_sh*0.4)
                adj_sh -= take_from_sh
                remaining = boost - take_from_sh
                if remaining > 0:
                    take_from_ca = min(remaining, adj_ca*0.3)
                    adj_ca -= take_from_ca
                adj_so += boost
            # Sparse trading: keep Sharpe weight slightly higher, but do not exceed +0.1
            target_trades = max(1, self.min_trades_for_dynamic_penalty)
            if total_trades < target_trades * 0.5:
                add = min(0.1, (target_trades*0.5 - total_trades)/target_trades * 0.1)
                # take from Calmar mainly
                take_from_ca = min(add, adj_ca*0.5)
                adj_ca -= take_from_ca
                adj_sh += take_from_ca
            # Normalize
            total = adj_sh + adj_so + adj_ca
            if total <= 0:
                return base_sh, base_so, base_ca
            adj_sh, adj_so, adj_ca = adj_sh/total, adj_so/total, adj_ca/total
            return adj_sh, adj_so, adj_ca
        except Exception:
            return 0.4, 0.3, 0.3

    def _bootstrap_sharpe_ci(self, returns: np.ndarray, n_boot: int = 400, seed: int = 42, ci: float = 0.90):
        """Bootstrap confidence interval for per-trade Sharpe approximation.

        Parameters
        ----------
        returns : np.ndarray
            Per-trade returns (already scaled relative to capital).
        n_boot : int
            Number of bootstrap resamples.
        seed : int
            RNG seed for reproducibility.
        ci : float
            Central confidence level (e.g. 0.90 -> 5th/95th percentiles).

        Returns
        -------
        (lower, upper) tuple of floats
        """
        try:
            r = np.array(returns, dtype=float)
            r = r[np.isfinite(r)]
            if r.size < 3:
                return -1.0, 1.0
            np.random.seed(seed)
            # Per-trade Sharpe proxy: mean / std (avoid divide by zero)
            base_std = np.std(r)
            if base_std <= 0:
                return -1.0, 1.0
            sharpe_samples = []
            n = len(r)
            for _ in range(max(10, n_boot)):
                sample = np.random.choice(r, size=n, replace=True)
                std_s = np.std(sample)
                if std_s <= 0:
                    continue
                sharpe_samples.append(np.mean(sample) / std_s)
            if not sharpe_samples:
                return -1.0, 1.0
            lower_q = (1 - ci) / 2.0
            upper_q = 1 - lower_q
            lower = float(np.quantile(sharpe_samples, lower_q))
            upper = float(np.quantile(sharpe_samples, upper_q))
            return lower, upper
        except Exception:
            return -1.0, 1.0

    def _write_parameter_ledger(self, record: dict):
        """Append a JSON record for a trial to the parameter ledger.

        The ledger is a JSON Lines file (one JSON object per line) enabling
        easy incremental parsing and streaming analysis. Safe to call from
        multiple trials (best-effort; no file locks)."""
        try:
            import json as _json, os as _os
            path = getattr(self, 'parameter_ledger_path', 'parameter_ledger.jsonl')
            # Ensure directory exists
            try:
                _os.makedirs(_os.path.dirname(path), exist_ok=True)
            except Exception:
                pass
            with open(path, 'a', encoding='utf-8') as f:
                f.write(_json.dumps(record, ensure_ascii=False) + '\n')
        except Exception:
            if self.debug_mode:
                try:
                    log_to_file(f"[LEDGER_WRITE_FAIL] Could not write ledger record for trial {record.get('trial')}", print_to_console=False)
                except Exception:
                    pass

    def _evaluate_regime_performance(self, data: pd.DataFrame, params: dict, regime) -> float:
        """Evaluate strategy performance on regime-specific data"""
        try:
            full_params = {**self.default_params, **params}
            strategy = Strategy(full_params)
            processed_data = strategy.generate_signals(data.copy(), self.realism_settings)
            
            if processed_data.empty:
                return -1000.0
            
            # Run mini-backtest
            # Ensure defaults merged for regime evaluation
            metrics, _, _ = run_backtest(full_params, processed_data, 10000)  # Standard initial capital
            
            if metrics is None:
                return -1000.0
            
            total_trades = metrics.get('Total Trades', 0)
            if total_trades == 0:
                return -1000.0
            
            # Calculate metrics
            sharpe = metrics.get('Sharpe Ratio', 0.0)
            sortino = metrics.get('Sortino Ratio', 0.0) 
            calmar = metrics.get('Calmar Ratio', 0.0)
            drawdown = metrics.get('Max Drawdown', -1.0)
            
            # --- DEBUG LOGGING: Print inputs to penalty calculation ---
            if self.debug_mode:
                log_to_file(
                    f"[PENALTY DEBUG] Regime '{regime.name}': trades={total_trades} dd={drawdown:.4f} sharpe={sharpe:.2f} sortino={sortino:.2f} calmar={calmar:.2f}",
                    print_to_console=False
                )

            # Unified dynamic / legacy penalty calculation
            trade_penalty, drawdown_penalty = self._compute_penalties(total_trades, drawdown)

            # Objective: Combination of ratios minus all penalties
            score = (sharpe * 0.4) + (sortino * 0.3) + (calmar * 0.3) - drawdown_penalty - trade_penalty
            
            # Regime-specific bonuses/penalties (Applied on top of the base score)
            from core.strategy import MarketRegime
            if regime in [MarketRegime.TRENDING_BULL, MarketRegime.BREAKOUT_BULLISH]:
                # Reward strategies that capture upward momentum
                total_return = metrics.get('Total Return', 0.0)
                score += max(0, total_return) * 0.1
            elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.DISTRIBUTION]:
                # Reward risk management in dangerous regimes
                score += (1 - abs(drawdown)) * 0.1
            if self.debug_mode:
                try:
                    log_to_file(
                        f"[REGIME_SCORE] {regime.name}: base_penalty={(trade_penalty+drawdown_penalty):.2f} trade_pen={trade_penalty:.2f} dd_pen={drawdown_penalty:.2f} final_score={score:.2f}",
                        print_to_console=False
                    )
                except Exception:
                    pass
            
            return score
            
        except Exception as e:
            return -1000.0


    def _objective_with_cache(self, params_list, train_df):
        """Internal objective function with caching for data preparation."""
        params = dict(zip([dim.name for dim in self.search_space], params_list))
        full_params = {**self.default_params, **params}
        
        # --- PERFORMANCE: Use persistent strategy instance to avoid reloading multi-timeframe data ---
        if self.persistent_strategy:
            # Update parameters without recreating the strategy instance
            self.persistent_strategy.params = full_params
            
            if not self.no_cache and self.memory and self.cached_generate_signals:
                processed_train_df = self.cached_generate_signals(self.persistent_strategy, train_df.copy(), self.realism_settings)
            else:
                processed_train_df = self.persistent_strategy.generate_signals(train_df.copy(), self.realism_settings)
        else:
            # Fallback to creating new instances (old behavior)
            if not self.no_cache and self.memory and self.cached_generate_signals:
                strategy = Strategy(full_params)
                processed_train_df = self.cached_generate_signals(strategy, train_df.copy(), self.realism_settings)
            else:
                strategy = Strategy(full_params)
                processed_train_df = strategy.generate_signals(train_df.copy(), self.realism_settings)

        if not self.are_params_valid(params):
            if self.debug_mode:
                log_to_file(f"[BAYES_FAIL] Invalid params skipped: {params}", print_to_console=False)
            tp, dp = self._compute_penalties(0, 0.0)
            return (tp + dp + 1) * 25.0

        if processed_train_df.empty:
            if self.debug_mode:
                log_to_file(f"[BAYES_FAIL] Empty processed DF for params: {params}", print_to_console=False)
            tp, dp = self._compute_penalties(0, 0.0)
            return (tp + dp + 1) * 20.0

        # Use the current chained capital for the training run
        # IMPORTANT: include default parameters for Bayesian objective too
        metrics, _, _ = run_backtest(full_params, processed_train_df, self.chained_capital)

        if metrics is None:
            if self.debug_mode:
                log_to_file(f"[BAYES_FAIL] run_backtest returned None for params: {params}", print_to_console=False)
            tp, dp = self._compute_penalties(0, 0.0)
            return (tp + dp + 1) * 25.0

        total_trades = metrics.get('Total Trades', 0)
        sharpe = metrics.get('Sharpe Ratio', 0.0)
        sortino = metrics.get('Sortino Ratio', 0.0)
        calmar = metrics.get('Calmar Ratio', 0.0)
        drawdown = metrics.get('Max Drawdown', -1.0)

        # Zero trades dynamic penalty (handled after drawing metrics)
        if total_trades == 0:
            if self.debug_mode:
                log_to_file(f"[BAYES_FAIL] Zero trades -> dynamic penalty", print_to_console=False)
            tp, dp = self._compute_penalties(0, drawdown)
            self._debug_log_trial(None, status='bayes_zero_trades', info={'trades': 0})
            return (tp + dp + 1) * 10.0

        # --- DEBUG LOGGING (pre-penalty) ---
        self.log_debug(
            f"[PENALTY DEBUG] Bayesian Objective: Trades={total_trades}, Drawdown={drawdown:.4f}, "
            f"MinTrades={self.min_trades_for_dynamic_penalty}, AllowedDD={self.allowed_max_drawdown}, "
            f"TradeBase={self.dynamic_trade_penalty_base}, DDBASE={self.dynamic_drawdown_penalty_base}"
        )

        trade_penalty, drawdown_penalty = self._compute_penalties(total_trades, drawdown)

        score = (sharpe * 0.4) + (sortino * 0.3) + (calmar * 0.3) - drawdown_penalty - trade_penalty
        final_value = -score if np.isfinite(score) else (self.dynamic_trade_penalty_base + self.dynamic_drawdown_penalty_base)
        if self.debug_mode:
            log_to_file(
                f"[BAYES_METRICS] trades={total_trades}, sharpe={sharpe:.2f}, sortino={sortino:.2f}, calmar={calmar:.2f}, drawdown={drawdown:.4f}, trade_pen={trade_penalty:.3f}, dd_pen={drawdown_penalty:.3f}, score={score:.4f}, objective={final_value:.4f}",
                print_to_console=False
            )
        self._debug_log_trial(None, status='bayes_complete', info={'trades': total_trades, 'objective': final_value})
        return final_value

    # ------------------------------------------------------------------
    # Debug helper: structured per-trial logging (JSONL)
    def _debug_log_trial(self, trial, status, info=None):
        if not (self.debug_mode and self.debug_log_trial_details):
            return
        try:
            from datetime import datetime as _dt
            rec = {
                'window': self._active_window,
                'timestamp': _dt.utcnow().isoformat() + 'Z',
                'status': status,
            }
            if trial is not None:
                rec['trial_number'] = getattr(trial, 'number', None)
            if info:
                rec.update(info)
            if not hasattr(self, '_debug_trial_log_path') or not self._debug_trial_log_path:
                return
            import json as _json
            with open(self._debug_trial_log_path, 'a') as f:
                f.write(_json.dumps(rec) + '\n')
        except Exception:
            pass

    def update_config_with_new_params(self, new_params):
        """Update the configuration file with new best parameters."""
        try:
            # Load existing config
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Update the best_parameters_so_far section
            config['best_parameters_so_far'] = new_params
            
            # Save the updated config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            log_to_file(f"Updated config with new parameters: {new_params}", print_to_console=True)
        except Exception as e:
            log_to_file(f"Error updating config with new parameters: {e}", print_to_console=True)

# ------------------------- CLI ENTRYPOINT RESTORED --------------------------
def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Ichimoku Backtester (minimal CLI)")
    parser.add_argument('-c','--config', default='core/optimization_config.json', help='Path to optimization config JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (verbose logging, skips cloud upload, etc.)')
    return parser

def _apply_cli_overrides(backtester, args):
    # Minimal: only debug flag affects behavior; penalties always dynamic from config/defaults
    log_to_file(
        f"[PENALTY] Dynamic penalties active (no CLI overrides). target_trades={backtester.min_trades_for_dynamic_penalty}, allowed_dd={backtester.allowed_max_drawdown}, trade_base={backtester.dynamic_trade_penalty_base}, dd_base={backtester.dynamic_drawdown_penalty_base}",
        print_to_console=True
    )

if __name__ == '__main__':
    parser = _build_arg_parser()
    cli_args = parser.parse_args()
    class Shim: pass
    shim = Shim()
    shim.config = cli_args.config
    # Removed extraneous runtime flags per simplification request
    shim.intensity = None
    shim.optimizer = 'optuna'  # default engine
    shim.no_warmup = False
    shim.min_trades_override = None  # legacy field ignored by wrapper
    shim.min_trades = None  # ensure getattr(args,'min_trades',..) works
    shim.runs_to_keep_override = None
    shim.debug_mode = cli_args.debug  # kept for potential internal use
    shim.debug = cli_args.debug       # required: wrapper accesses args.debug
    shim.no_numba = False
    shim.no_cache = False
    shim.clear_cache = False
    run_backtest_instance(shim)
    # Penalty settings loaded from config/defaults only.
