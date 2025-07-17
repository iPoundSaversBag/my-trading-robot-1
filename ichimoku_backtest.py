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
#   2. TESTING: The single best parameter set from the training phase is then
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
import datetime
import json
import pandas as pd
import numpy as np
import json5
import logging
from datetime import datetime, timedelta
import sys
import glob
import random
import shutil
import stat
import traceback
import tempfile
from numba import njit, typed
from numba.core import types
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import argparse
import matplotlib
matplotlib.use('Agg')
from joblib import Memory

# --- FIX: Suppress known, non-critical RuntimeWarning from numpy/pandas ---
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Optional Imports ---
MATPLOTLIB_AVAILABLE = False
MPLFINANCE_AVAILABLE = False
OPTUNA_AVAILABLE = False
QUANTSTATS_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
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
from generate_plots import plot_trades_for_window, plot_pnl_distribution
from strategy import Strategy
from portfolio import Portfolio
from position_manager import PositionManager
from position_manager import PositionManager
from gcp_utils import upload_to_gcs

# --- Numba-Optimized Backtesting Core ---
# The core logic is defined in a pure Python function.
# This allows it to be run without Numba for easier debugging.
def _backtest_core_logic(
    close, high, low, long_signals, short_signals, atr, is_trending,
    initial_capital, commission_pct, risk_per_trade_pct,
    TRENDING_TP_ATR_MULTIPLIER, RANGING_TP_ATR_MULTIPLIER, PARTIAL_EXIT_PERCENTAGE,
    TRENDING_TSL_ATR_MULTIPLIER, RANGING_TSL_ATR_MULTIPLIER,
    TRENDING_EXIT_STRATEGY, RANGING_EXIT_STRATEGY,
    # --- REALISM: Add params for variable slippage ---
    variable_slippage_enabled, base_slippage_percent, vol_slippage_multiplier, volatility_atr,
    FIXED_TP_PCT=0.01, FIXED_SL_PCT=0.01, TIME_IN_TRADE_BARS=20
):
    capital = initial_capital
    position_size = 0.0
    entry_price = 0.0
    in_trade = False
    is_long = False
    trailing_stop_loss = 0.0
    take_profit_price = 0.0
    partial_exit_taken = False
    entry_index = 0 # FIX: Initialize entry_index
    
    capital_over_time = np.full_like(close, initial_capital)
    trades_list = []

    bars_in_trade = 0
    for i in range(1, len(close)):
        current_price = close[i]
        current_atr = atr[i]
        
        # --- Trade Management ---
        if in_trade:
            # Determine regime and strategy
            regime = 'TRENDING' if is_trending[i] else 'RANGING'
            exit_strategy = TRENDING_EXIT_STRATEGY if regime == 'TRENDING' else RANGING_EXIT_STRATEGY
            bars_in_trade += 1
            pnl = (current_price - entry_price) * position_size if is_long else (entry_price - current_price) * position_size
            exit_reason = ""

            # 1. Check for Partial Take Profit (only if TP is enabled)
            if not partial_exit_taken and exit_strategy in ["TP_AND_TSL"]:
                if is_long and current_price >= take_profit_price:
                    exit_reason = "Partial TP"
                elif not is_long and current_price <= take_profit_price:
                    exit_reason = "Partial TP"
                if exit_reason == "Partial TP":
                    # --- REALISM: Calculate dynamic slippage for this exit ---
                    current_slippage = base_slippage_percent
                    if variable_slippage_enabled and volatility_atr[i] > 0 and current_price > 0:
                        volatility_component = (volatility_atr[i] / current_price) * vol_slippage_multiplier
                        current_slippage += volatility_component

                    exit_price = current_price * (1 - current_slippage) if is_long else current_price * (1 + current_slippage)
                    partial_size = position_size * PARTIAL_EXIT_PERCENTAGE
                    
                    # --- ROO: Simplified Capital Calculation ---
                    # Calculate PNL for logging/records first.
                    gross_partial_pnl = (exit_price - entry_price) * partial_size if is_long else (entry_price - exit_price) * partial_size
                    
                    # Apply commission on the partial exit
                    exit_commission = partial_size * exit_price * commission_pct
                    # Also account for the entry commission for this partial size
                    entry_commission = partial_size * entry_price * commission_pct
                    net_partial_pnl = gross_partial_pnl - entry_commission - exit_commission

                    # Capital update logic is correct (models cash flow)
                    capital += (partial_size * entry_price) + gross_partial_pnl - exit_commission
                    
                    remaining_size = position_size - partial_size
                    # Log the capital after the transaction as the definitive equity
                    trades_list.append((entry_index, i, entry_price, exit_price, partial_size, 1 if is_long else -1, net_partial_pnl, capital, exit_reason))
                    
                    position_size -= partial_size
                    partial_exit_taken = True # Mark as taken for this trade

                    # --- FIX: Move Trailing Stop to Break-Even after Partial Profit ---
                    if is_long:
                        trailing_stop_loss = max(trailing_stop_loss, entry_price)
                    else: # is_short
                        trailing_stop_loss = min(trailing_stop_loss, entry_price)

            # 2. Exit logic by strategy
            if exit_strategy == "TSL_ONLY":
                # Only trailing stop loss, no TP
                if is_long:
                    tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                    new_tsl = current_price - (current_atr * tsl_multiplier)
                    trailing_stop_loss = max(trailing_stop_loss, new_tsl)
                    if current_price <= trailing_stop_loss:
                        exit_reason = "TSL"
                else:
                    tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                    new_tsl = current_price + (current_atr * tsl_multiplier)
                    trailing_stop_loss = min(trailing_stop_loss, new_tsl)
                    if current_price >= trailing_stop_loss:
                        exit_reason = "TSL"
            elif exit_strategy == "TP_AND_TSL":
                # Standard logic (TP and TSL)
                if is_long:
                    tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                    new_tsl = current_price - (current_atr * tsl_multiplier)
                    if partial_exit_taken:
                        trailing_stop_loss = max(trailing_stop_loss, new_tsl, entry_price)
                    else:
                        trailing_stop_loss = max(trailing_stop_loss, new_tsl)
                    if current_price <= trailing_stop_loss:
                        exit_reason = "TSL"
                else:
                    tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                    new_tsl = current_price + (current_atr * tsl_multiplier)
                    if partial_exit_taken:
                        trailing_stop_loss = min(trailing_stop_loss, new_tsl, entry_price)
                    else:
                        trailing_stop_loss = min(trailing_stop_loss, new_tsl)
                    if current_price >= trailing_stop_loss:
                        exit_reason = "TSL"
            elif exit_strategy == "FIXED_TP_AND_SL":
                # Fixed TP and SL (percent-based)
                fixed_tp = entry_price * (1 + FIXED_TP_PCT) if is_long else entry_price * (1 - FIXED_TP_PCT)
                fixed_sl = entry_price * (1 - FIXED_SL_PCT) if is_long else entry_price * (1 + FIXED_SL_PCT)
                if is_long:
                    if current_price >= fixed_tp:
                        exit_reason = "Fixed TP"
                    elif current_price <= fixed_sl:
                        exit_reason = "Fixed SL"
                else:
                    if current_price <= fixed_tp:
                        exit_reason = "Fixed TP"
                    elif current_price >= fixed_sl:
                        exit_reason = "Fixed SL"
            elif exit_strategy == "TIME_IN_TRADE":
                # Exit after TIME_IN_TRADE_BARS if not stopped out
                if bars_in_trade >= TIME_IN_TRADE_BARS:
                    exit_reason = "Time Exit"

            # 3. Execute Full Exit if TSL is hit
            if exit_reason and exit_reason != "Partial TP":
                # --- REALISM: Calculate dynamic slippage for this exit ---
                current_slippage = base_slippage_percent
                if variable_slippage_enabled and volatility_atr[i] > 0 and current_price > 0:
                    volatility_component = (volatility_atr[i] / current_price) * vol_slippage_multiplier
                    current_slippage += volatility_component
                
                exit_price = current_price * (1 - current_slippage) if is_long else current_price * (1 + current_slippage)
                # --- ROO: Simplified Capital Calculation ---
                # Calculate PNL for logging/records first.
                gross_final_pnl = (exit_price - entry_price) * position_size if is_long else (entry_price - exit_price) * position_size
                
                # Calculate total commissions for the round trip
                entry_commission = position_size * entry_price * commission_pct
                exit_commission = position_size * exit_price * commission_pct
                net_final_pnl = gross_final_pnl - entry_commission - exit_commission

                # Capital update logic is correct (models cash flow)
                capital += (position_size * entry_price) + gross_final_pnl - exit_commission
                
                # Log the capital after the transaction as the definitive equity
                trades_list.append((entry_index, i, entry_price, exit_price, position_size, 1 if is_long else -1, net_final_pnl, capital, exit_reason))
                in_trade = False
                position_size = 0.0

        # --- Entry Logic ---
        if not in_trade:
            if long_signals[i-1]:
                is_long = True
            elif short_signals[i-1]:
                is_long = False
            else:
                continue

            # Determine regime and strategy
            regime = 'TRENDING' if is_trending[i] else 'RANGING'
            exit_strategy = TRENDING_EXIT_STRATEGY if regime == 'TRENDING' else RANGING_EXIT_STRATEGY

            # --- REALISM: Calculate dynamic slippage for this entry ---
            current_slippage = base_slippage_percent
            if variable_slippage_enabled and volatility_atr[i] > 0 and current_price > 0:
                volatility_component = (volatility_atr[i] / current_price) * vol_slippage_multiplier
                current_slippage += volatility_component

            entry_price = current_price * (1 + current_slippage) if is_long else current_price * (1 - current_slippage)
            # --- FIX: Base risk on the true total equity from the previous bar ---
            risk_amount = capital_over_time[i-1] * risk_per_trade_pct

            # --- Regime-based exit logic ---
            if exit_strategy == "TSL_ONLY":
                tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                initial_sl_price = entry_price - (current_atr * tsl_multiplier) if is_long else entry_price + (current_atr * tsl_multiplier)
                take_profit_price = 0
            elif exit_strategy == "TP_AND_TSL":
                tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                tp_multiplier = TRENDING_TP_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TP_ATR_MULTIPLIER
                initial_sl_price = entry_price - (current_atr * tsl_multiplier) if is_long else entry_price + (current_atr * tsl_multiplier)
                take_profit_price = entry_price + (current_atr * tp_multiplier) if is_long else entry_price - (current_atr * tp_multiplier)
            elif exit_strategy == "FIXED_TP_AND_SL":
                initial_sl_price = entry_price * (1 - FIXED_SL_PCT) if is_long else entry_price * (1 + FIXED_SL_PCT)
                take_profit_price = entry_price * (1 + FIXED_TP_PCT) if is_long else entry_price * (1 - FIXED_TP_PCT)
            elif exit_strategy == "TIME_IN_TRADE":
                tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                initial_sl_price = entry_price - (current_atr * tsl_multiplier) if is_long else entry_price + (current_atr * tsl_multiplier)
                take_profit_price = 0
            else:
                # Default fallback
                tsl_multiplier = TRENDING_TSL_ATR_MULTIPLIER if regime == 'TRENDING' else RANGING_TSL_ATR_MULTIPLIER
                initial_sl_price = entry_price - (current_atr * tsl_multiplier) if is_long else entry_price + (current_atr * tsl_multiplier)
                take_profit_price = 0

            if (is_long and entry_price <= initial_sl_price) or (not is_long and entry_price >= initial_sl_price):
                continue # Skip trade if entry is already beyond SL

            position_size = risk_amount / abs(entry_price - initial_sl_price)

            if position_size * entry_price > capital:
                position_size = capital / entry_price # Cannot use more capital than available

            if position_size > 0:
                in_trade = True
                entry_index = i # FIX: Store the entry index when a new trade is initiated
                partial_exit_taken = False # Reset for the new trade
                bars_in_trade = 0
                # --- REALISM: Apply commission on entry ---
                trade_value = position_size * entry_price
                commission = trade_value * commission_pct
                capital -= (trade_value + commission)
                trailing_stop_loss = initial_sl_price

        capital_over_time[i] = capital + (position_size * current_price if in_trade else 0)

    # --- FIX: Liquidate any open position at the end of the backtest ---
    if in_trade:
        # --- FIX: Use dynamic slippage for final liquidation ---
        current_price = close[-1]
        current_slippage = base_slippage_percent
        if variable_slippage_enabled and volatility_atr[-1] > 0 and current_price > 0:
            volatility_component = (volatility_atr[-1] / current_price) * vol_slippage_multiplier
            current_slippage += volatility_component
        
        exit_price = current_price * (1 - current_slippage) if is_long else current_price * (1 + current_slippage)
        # --- ROO: Simplified Capital Calculation ---
        # Calculate PNL for logging/records first.
        gross_final_pnl = (exit_price - entry_price) * position_size if is_long else (entry_price - exit_price) * position_size
        
        # Calculate total commissions for the round trip
        entry_commission = position_size * entry_price * commission_pct
        exit_commission = position_size * exit_price * commission_pct
        net_final_pnl = gross_final_pnl - entry_commission - exit_commission

        # Capital update logic is correct (models cash flow)
        capital += (position_size * entry_price) + gross_final_pnl - exit_commission
        
        # Log the capital after the transaction as the definitive equity
        trades_list.append((entry_index, len(close) - 1, entry_price, exit_price, position_size, 1 if is_long else -1, net_final_pnl, capital, "End of Backtest"))
        capital_over_time[-1] = capital # Update the very last capital value

    # --- FIX: Re-affirm final equity in capital_over_time to resolve potential Numba issue ---
    elif trades_list:
        # If trades occurred, but none are open, the final value in the array should be the equity from the last recorded trade.
        # This ensures consistency between the returned array and the trade list.
        capital_over_time[-1] = trades_list[-1][-2] # Get the final_equity from the last trade

    return capital_over_time, trades_list

# A second version of the function is created and compiled with Numba for performance.
_backtest_core_numba = njit(_backtest_core_logic)


def _backtest_core_oop(processed_df, initial_capital, params, risk_per_trade_pct):
    """
    A pure Python, object-oriented backtest implementation using PositionManager.
    This version is for debugging and clarity, not for performance. It ensures
    the backtest logic is identical to the live bot's logic when --no-numba is used.
    """
    position_manager = PositionManager(params)
    
    capital_over_time = np.full(len(processed_df), initial_capital, dtype=float)
    trades_list = []
    entry_index = 0

    for i in range(1, len(processed_df)):
        capital_over_time[i] = capital_over_time[i-1]
        
        latest_candle = processed_df.iloc[i]
        
        # --- Exit Logic ---
        if position_manager.in_position:
            position_manager.update_trailing_stop(latest_candle)
            exit_reason = position_manager.check_for_exit(latest_candle)
            if exit_reason:
                exit_price = latest_candle['close']
                pos_details_before_exit = position_manager.position_details.copy()
                
                pnl, size_exited = position_manager.exit_position(exit_reason, exit_price)
                
                capital_over_time[i] += pnl
                
                trade_type_int = 1 if pos_details_before_exit['type'] == 'long' else -1
                trades_list.append((
                    entry_index, i, pos_details_before_exit['entry_price'], exit_price,
                    size_exited, trade_type_int, pnl, capital_over_time[i], exit_reason
                ))

        # --- Entry Logic ---
        if not position_manager.in_position:
            signal_candle = processed_df.iloc[i-1]
            trade_type, _ = position_manager.check_for_entry(signal_candle)
            
            if trade_type:
                execution_candle = processed_df.iloc[i]
                balance_for_trade = capital_over_time[i-1]
                
                if position_manager.enter_position(trade_type, execution_candle, balance_for_trade):
                    entry_index = i

    # --- Final Liquidation ---
    if position_manager.in_position:
        exit_price = processed_df['close'].iloc[-1]
        pos_details_before_exit = position_manager.position_details.copy()
        pnl, size_exited = position_manager.exit_position("End of Backtest", exit_price)
        capital_over_time[-1] += pnl
        trade_type_int = 1 if pos_details_before_exit['type'] == 'long' else -1
        trades_list.append((
            entry_index, len(processed_df) - 1, pos_details_before_exit['entry_price'], exit_price,
            size_exited, trade_type_int, pnl, capital_over_time[-1], "End of Backtest"
        ))

    return capital_over_time, trades_list


# --- Helper Functions ---
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def log_to_file(message, filename="full_analysis_log.txt", print_to_console=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_report_line = "---" in message or "===" in message
    log_message = f"[{timestamp}] {message}"
    print_message = log_message
    if is_report_line:
        print_message = message
        if "---" in message and "--- " not in message:
            print_message = f"\n{message}"
    if print_to_console:
        print(print_message)
    
    # Use the current run directory if it's set
    log_path = IchimokuBacktester.current_run_dir_static or "."
    final_log_file = os.path.join(log_path, os.path.basename(filename))

    mode = 'w' if "PIPELINE STARTED" in message else 'a'
    try:
        with open(final_log_file, mode, encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Failed to write to log file {final_log_file}: {e}")


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
        if getattr(self, 'debug_mode', False):
            if not hasattr(self, 'debug_messages'):
                self.debug_messages = []
            self.debug_messages.append(str(msg))

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
        log_to_file(f"Output for this run will be saved in: {self.current_run_dir}", print_to_console=False)

    def __init__(self, config_path='optimization_config.json', intensity_override=None, no_warmup=False, optimizer='bayesian', min_trades_override=None, runs_to_keep_override=None, debug_mode=False, no_numba=False, no_cache=False, clear_cache=False):
        """
        Initializes the backtester.
        """
        self.debug_mode = debug_mode
        self.no_numba = no_numba
        self.debug_messages = []
        if self.debug_mode:
            self.log_debug("--- Debug Mode Enabled ---")
        
        self.problem_summary = []

        self.config_path = config_path
        self.config = self.load_config(self.config_path)
        if self.config is None:
            # load_config already logs the error, just exit.
            sys.exit(1) # Exit if config fails to load.

        self.warm_start_params = self.config.get('best_parameters_so_far', None)
        
        # Override settings from command line if provided
        if intensity_override:
            self.config['optimization_settings']['intensity'] = str(intensity_override)
        
        # --- FIX: Use 'fixed_parameters' from config, which is the correct key ---
        self.default_params = self.config.get('fixed_parameters', {})
        if not self.default_params:
            self.log_problem("The 'fixed_parameters' block is missing from your config file. Using hardcoded defaults.")

        self.use_numba_warmup = not no_warmup
        self.optimizer = optimizer
        
        # --- REALISM: Load variable slippage settings ---
        self.realism_settings = self.config.get('realism_settings', {}).get('variable_slippage', {})
        self.variable_slippage_enabled = self.realism_settings.get('enabled', False)
        
        # FIX: Properly initialize min_trades_per_window as a class attribute
        if min_trades_override is not None:
            self.min_trades_per_window = min_trades_override
        else:
            self.min_trades_per_window = self.config.get('optimization_settings', {}).get('min_trades_per_window', 5)

        self.runs_to_keep_override = runs_to_keep_override

        # Define base_output_dir BEFORE calling setup_output_directory
        self.base_output_dir = self.config.get('output_settings', {}).get('base_directory', 'plots_output')
        self.setup_output_directory()
        
        # --- FEATURE: Persistent Caching with joblib ---
        self.no_cache = no_cache
        # FIX: Move cache directory to the project root to avoid long path issues on Windows
        # --- ROO: Use a temporary directory for caching to avoid long path issues on Windows ---
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
            self.cached_generate_signals = self.memory.cache(Strategy.generate_signals)
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.memory = Memory(self.cache_dir, verbose=0)
            self.cached_generate_signals = self.memory.cache(Strategy.generate_signals)
            log_to_file(f"--- Caching is enabled. Cache directory: {self.cache_dir} ---", print_to_console=False)

        # --- FIX: Add robust data loading with clear error handling ---
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

        # --- FIX: Add a fast mode for debugging ---
        if self.debug_mode:
            log_to_file("--- DEBUG MODE: Reducing dataset to last 8 months for faster debugging. ---", print_to_console=True)
            # FIX: Replace deprecated .last() with modern .loc slicing
            end_date = self.df_full.index.max()
            start_date = end_date - pd.DateOffset(months=8)
            self.df_full = self.df_full.loc[start_date:end_date]
            if len(self.df_full) < 300: # Ensure there's enough data for at least one window
                log_to_file("WARNING: Not enough data in the last 8 months for a debug run. Using full dataset.", print_to_console=True)
                self.df_full = pd.read_parquet(data_file) # Reload
            else:
                log_to_file(f"Debug mode using data from {self.df_full.index.min()} to {self.df_full.index.max()}", print_to_console=False)


        self.all_optimized_params = {}
        self.all_trades = []
        self.data_prep_cache = {}
        self.chained_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
        # --- FIX: Use 'FIXED_RISK_PERCENTAGE' from config, which is the correct key ---
        self.risk_per_trade_pct = self.default_params.get('FIXED_RISK_PERCENTAGE', 0.01)

        # Initialize search space for the optimizer
        self.search_space = self.define_parameter_spaces()

        if self.use_numba_warmup and not self.no_numba:
            self._warmup_numba()

    def log_trade_details(self, trades_df, window_num):
        """Logs a detailed summary of trades for a specific window."""
        if trades_df.empty:
            return

        pnl = trades_df['pnl']
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        
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
        log_to_file(f"---------------------------------", print_to_console=False)

    def _warmup_numba(self):
        """Runs the Numba JIT compiler on the backtest function ahead of time."""
        log_to_file("Pre-compiling Numba backtest core... this may take a moment.", print_to_console=True)
        try:
            # Create small, correctly-typed dummy arrays to trigger compilation
            dummy_close = np.random.random(10)
            dummy_high = np.random.random(10)
            dummy_low = np.random.random(10)
            dummy_long = np.zeros(10, dtype=np.bool_)
            dummy_short = np.zeros(10, dtype=np.bool_)
            dummy_atr = np.random.random(10)
            dummy_trending = np.zeros(10, dtype=np.bool_)
            
            _backtest_core_numba(
                dummy_close, dummy_high, dummy_low, dummy_long, dummy_short,
                dummy_atr, dummy_trending, 10000.0, 0.001, 0.01,
                2.0, 1.5, 0.5, 3.0, 2.0, "TP_AND_TSL", "TP_AND_TSL",
                # --- REALISM: Add dummy values for new slippage params ---
                True, 0.0005, 1.5, np.random.random(10)
            )
            log_to_file("Numba core compiled successfully.", print_to_console=True)
        except Exception as e:
            log_to_file(f"Warning: Numba warm-up failed. Performance may be degraded. Error: {e}", print_to_console=True)


    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json5.load(f)
        except Exception as e:
            log_to_file(f"FATAL: Could not load/parse config at {path}: {e}")
            return None

    def define_parameter_spaces(self):
        param_spaces_config = self.config.get('parameter_spaces', {})
        search_space = []
        
        # The config can have multiple groups of parameters (e.g., "global", "ranging").
        # We need to iterate through each group.
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

                if p_type.lower() == 'integer':
                    search_space.append(Integer(bounds[0], bounds[1], name=p_name))
                elif p_type.lower() == 'real':
                    search_space.append(Real(bounds[0], bounds[1], name=p_name))
                elif p_type.lower() == 'categorical':
                    values = p_config.get('values')
                    if values:
                        search_space.append(Categorical(values, name=p_name))

        return search_space

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
        """Checks if the combination of parameters is valid."""
        # Use the correct parameter names as used in the optimization space
        try:
            return (
                params['TENKAN_SEN_PERIOD'] < params['KIJUN_SEN_PERIOD'] and
                params['KIJUN_SEN_PERIOD'] < params['SENKOU_SPAN_B_PERIOD']
            )
        except KeyError:
            # If any of the keys are missing, the params are not valid
            return False

    def run_backtest(self, params, processed_df, capital):
        """
        Executes a single backtest run with a given set of parameters using the
        high-performance Numba core.
        
        Args:
            params (dict): The parameters for the backtest.
            processed_df (pd.DataFrame): A DataFrame that has *already* been processed
                                         with all necessary indicators and signals.
            capital (float): The initial capital for this backtest run.
        """
        # 1. Combine all parameters
        full_params = {**self.default_params, **params}

        if processed_df.empty or not self.are_params_valid(full_params):
            # Return a structure indicating failure, with very poor metrics
            return {"Sharpe Ratio": -100, "Total Trades": 0, "Final Equity": capital}, pd.DataFrame(), capital

        # 2. Prepare arrays for Numba
        close_prices = processed_df['close'].to_numpy()
        high_prices = processed_df['high'].to_numpy()
        low_prices = processed_df['low'].to_numpy()
        long_signals = processed_df['long_signals'].to_numpy()
        short_signals = processed_df['short_signals'].to_numpy()
        atr = processed_df['atr'].to_numpy()
        is_trending = processed_df['is_trending'].to_numpy()
        volatility_atr = processed_df['volatility_atr'].to_numpy()

        # 3. Execute the backtest core
        if self.no_numba:
            # Use the new OOP backtest core for debugging. It's slower but uses the unified PositionManager.
            capital_over_time, trades_list = _backtest_core_oop(
                processed_df=processed_df,
                initial_capital=capital,
                params=full_params,
                risk_per_trade_pct=self.risk_per_trade_pct
            )
        else:
            # Use the high-performance Numba core for production runs
            try:
                capital_over_time, trades_list = _backtest_core_numba(
                    close=close_prices,
                    high=high_prices,
                    low=low_prices,
                    long_signals=long_signals,
                    short_signals=short_signals,
                    atr=atr,
                    is_trending=is_trending,
                    initial_capital=capital,
                    commission_pct=full_params.get('COMMISSION_PERCENTAGE', 0.001),
                    risk_per_trade_pct=self.risk_per_trade_pct,
                    TRENDING_TP_ATR_MULTIPLIER=full_params.get('TRENDING_TP_ATR_MULTIPLIER', 2.0),
                    RANGING_TP_ATR_MULTIPLIER=full_params.get('RANGING_TP_ATR_MULTIPLIER', 1.5),
                    PARTIAL_EXIT_PERCENTAGE=full_params.get('PARTIAL_EXIT_PERCENTAGE', 0.5),
                    TRENDING_TSL_ATR_MULTIPLIER=full_params.get('TRENDING_TSL_ATR_MULTIPLIER', 3.0),
                    RANGING_TSL_ATR_MULTIPLIER=full_params.get('RANGING_TSL_ATR_MULTIPLIER', 2.0),
                    TRENDING_EXIT_STRATEGY=full_params.get('TRENDING_EXIT_STRATEGY', "TP_AND_TSL"),
                    RANGING_EXIT_STRATEGY=full_params.get('RANGING_EXIT_STRATEGY', "TP_AND_TSL"),
                    variable_slippage_enabled=self.variable_slippage_enabled,
                    base_slippage_percent=self.realism_settings.get('base_slippage_percent', 0.0005),
                    vol_slippage_multiplier=self.realism_settings.get('vol_slippage_multiplier', 1.5),
                    volatility_atr=volatility_atr,
                    FIXED_TP_PCT=full_params.get('FIXED_TP_PCT', 0.05),
                    FIXED_SL_PCT=full_params.get('FIXED_SL_PCT', 0.02),
                    TIME_IN_TRADE_BARS=full_params.get('TIME_IN_TRADE_BARS', 50)
                )
            except Exception as e:
                self.log_debug(f"Numba core execution failed for params: {params}. Error: {e}")
                return {"Sharpe Ratio": -100, "Total Trades": 0, "Final Equity": capital}, pd.DataFrame(), capital

        # 5. Process the results
        final_equity = capital_over_time[-1] if len(capital_over_time) > 0 else capital
        
        if not trades_list:
            return {"Sharpe Ratio": 0, "Total Trades": 0, "Final Equity": final_equity}, pd.DataFrame(), final_equity

        trades_df = pd.DataFrame(
            trades_list,
            columns=['entry_index', 'exit_index', 'entry_price', 'exit_price', 'size', 'trade_type', 'pnl', 'final_equity', 'exit_reason']
        )
        
        # Convert indices to timestamps
        trades_df['entry_timestamp'] = processed_df.index[trades_df['entry_index']]
        trades_df['exit_timestamp'] = processed_df.index[trades_df['exit_index']]

        # 6. Calculate performance metrics
        portfolio = Portfolio(initial_capital=capital)
        metrics = portfolio.calculate_performance_metrics(trades_df, capital, final_equity)
        
        return metrics, trades_df, final_equity


    def objective_optuna(self, trial, train_df):
        """Objective function for Optuna optimization."""
        params = {}
        # Suggest parameters for the trial
        for dim in self.search_space:
            if isinstance(dim, Integer):
                params[dim.name] = trial.suggest_int(dim.name, dim.low, dim.high)
            elif isinstance(dim, Real):
                params[dim.name] = trial.suggest_float(dim.name, dim.low, dim.high)
            elif isinstance(dim, Categorical):
                params[dim.name] = trial.suggest_categorical(dim.name, dim.categories)

        if not self.are_params_valid(params):
            # Prune this trial if the basic parameter constraints are not met
            raise optuna.exceptions.TrialPruned()

        # --- PERFORMANCE: Use the data preparation cache ---
        if not self.no_cache and hasattr(self, 'memory'):
            strategy = Strategy(params)
            processed_train_df = self.cached_generate_signals(strategy, train_df.copy(), self.realism_settings)
        else:
            strategy = Strategy(params)
            processed_train_df = strategy.generate_signals(train_df.copy(), self.realism_settings)

        if processed_train_df.empty:
            return 1e9  # Return a large value to indicate failure

        # Use the current chained capital for the training run
        metrics, _, _ = self.run_backtest(params, processed_train_df, self.chained_capital)
        if metrics is None:
            return 1e9

        total_trades = metrics.get('Total Trades', 0)
        if total_trades < self.min_trades_per_window:
            # Prune if not enough trades, as it's not a representative sample
            raise optuna.exceptions.TrialPruned()

        sharpe = metrics.get('Sharpe Ratio', 0.0)
        sortino = metrics.get('Sortino Ratio', 0.0)
        calmar = metrics.get('Calmar Ratio', 0.0)
        drawdown = metrics.get('Max Drawdown', -1.0)

        # Objective: Combination of Sharpe, Sortino, Calmar, and Drawdown
        score = (sharpe * 0.4) + (sortino * 0.3) + (calmar * 0.3) + (drawdown * 0.2)
        
        final_value = -score if np.isfinite(score) else 1e9
        self.log_debug(f"Optuna trial {trial.number}: Score={score:.4f}, Value={final_value:.4f}, Trades={total_trades}, Sharpe={sharpe:.2f}")
        return final_value


    def _objective_with_cache(self, params_list, train_df):
        """Internal objective function with caching for data preparation."""
        params = dict(zip([dim.name for dim in self.search_space], params_list))
        
        if not self.no_cache and self.memory:
            strategy = Strategy(params)
            processed_train_df = self.cached_generate_signals(strategy, train_df.copy(), self.realism_settings)
        else:
            strategy = Strategy(params)
            processed_train_df = strategy.generate_signals(train_df.copy(), self.realism_settings)

        if not self.are_params_valid(params):
            self.log_debug(f"Bayesian trial: Skipping invalid params: {params}")
            return 1e9

        if processed_train_df.empty:
            self.log_debug(f"Bayesian trial: Data prep failed for params: {params}")
            return 1e9

        # Use the current chained capital for the training run
        metrics, _, _ = self.run_backtest(params, processed_train_df, self.chained_capital)
        
        if metrics is None:
            self.log_debug(f"Bayesian trial: Backtest failed for params: {params}")
            return 1e9

        total_trades = metrics.get('Total Trades', 0)
        if total_trades < self.min_trades_per_window:
            trade_shortfall_penalty = (self.min_trades_per_window - total_trades) * 1000
            self.log_debug(f"Bayesian trial: Penalized for insufficient trades ({total_trades} < {self.min_trades_per_window}). Params: {params}")
            return 1e9 + trade_shortfall_penalty

        sharpe = metrics.get('Sharpe Ratio', 0.0)
        sortino = metrics.get('Sortino Ratio', 0.0)
        calmar = metrics.get('Calmar Ratio', 0.0)
        drawdown = metrics.get('Max Drawdown', -1.0)

        # Objective: Combination of Sharpe, Sortino, Calmar, and Drawdown
        score = (sharpe * 0.4) + (sortino * 0.3) + (calmar * 0.3) + (drawdown * 0.2)
        final_value = -score if np.isfinite(score) else 1e9
        self.log_debug(f"Bayesian trial: Score={score:.4f}, Value={final_value:.4f}, Trades={total_trades}, Sharpe={sharpe:.2f}")
        return final_value

    def update_config_with_new_params(self, new_params):
        """Updates the 'best_parameters_so_far' in the main config file."""
        log_to_file(f"Updating {self.config_path} with new superior parameters.", print_to_console=True)
        try:
            # Read the existing config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json5.load(f)
            
            # Update the specific block
            config['best_parameters_so_far'] = new_params
            
            # Write the updated config back
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            
            log_to_file("Config file updated successfully.", print_to_console=False)
        except Exception as e:
            log_to_file(f"FATAL: Could not update config file at {self.config_path}. Error: {e}", print_to_console=True)

    def run_walk_forward_optimization(self):
        """
        Main loop for the walk-forward optimization process.
        It iterates through training and testing windows, runs optimization,
        and evaluates performance on out-of-sample data.
        """
        log_to_file(f"--- Starting Walk-Forward Analysis ---", print_to_console=True)
        
        # FIX: Use the class attribute for min_trades_per_window
        log_to_file(f"Using minimum trades per window setting: {self.min_trades_per_window}", print_to_console=True)

        windows = self.get_walk_forward_windows(self.df_full)
        if not windows:
            self.log_problem("Could not generate any walk-forward windows. Check data and config.")
            self.finalize_and_report()
            return

        log_to_file(f"Data loaded. Starting walk-forward analysis with {len(windows)} windows.")
        last_window_best_params_list = None

        for i, (train_start, train_end, test_end) in enumerate(tqdm(windows, desc="Walk-Forward Windows")):
            window_num = i + 1
            log_to_file(f"--- Starting Window #{window_num}/{len(windows)} [{train_start.date()} -> {test_end.date()}] ---", print_to_console=False)
            # --- FIX: Add a fast mode for debugging ---
            if self.debug_mode and i >= 5:
                log_to_file("--- DEBUG MODE: Stopping after 5 walk-forward windows. ---", print_to_console=True)
                break

            self.data_prep_cache.clear() # Clear cache for the new window
            train_df = self.df_full.loc[train_start:train_end]
            test_df = self.df_full.loc[train_end:test_end]
            
            opt_intensity = str(self.config['optimization_settings']['intensity'])
            n_calls = self.config['optimization_settings']['calls_per_window'][opt_intensity]
            
            # --- FIX: Add a fast mode for debugging ---
            if self.debug_mode:
                log_to_file("--- DEBUG MODE: Reducing optimization trials to 5 for faster debugging. ---", print_to_console=True)
                n_calls = 5

            best_params = {}
            
            # Correctly define window_num for logging purposes before it's used.
            window_num = i + 1

            # Skip optimization if the training window has too few trades
            if len(train_df.index) < self.min_trades_per_window * 2: # A rough proxy for trades
                self.log_problem(f"Window #{window_num}: Skipped optimization due to insufficient data points ({len(train_df.index)}).")
                if self.all_optimized_params:
                    # Find the last available parameters to use
                    last_key = sorted(self.all_optimized_params.keys())[-1]
                    best_params = self.all_optimized_params[last_key]
                    log_to_file(f"Reusing parameters from {last_key} for window #{window_num}.", print_to_console=False)
                else:
                    log_to_file(f"Window #{window_num}: No previous parameters to fall back to. Skipping.", print_to_console=True)
                    continue # No previous params to fall back to, skip this window
            else:
                log_to_file(f"Window #{window_num}: Starting optimization ({self.optimizer}, {n_calls} calls)...", print_to_console=False)
                # This block runs the optimization
                if self.optimizer == 'bayesian':
                    obj_func = lambda p: self._objective_with_cache(p, train_df)
                    # FIX: Adjust n_initial_points in debug mode to be less than n_calls
                    n_initial_points = 4 if self.debug_mode else 10
                    gp_kwargs = {"n_calls": n_calls, "random_state": 42, "n_initial_points": n_initial_points, "n_jobs": -1}
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

                    result = gp_minimize(obj_func, self.search_space, **gp_kwargs)
                    best_params = {dim.name: val for dim, val in zip(self.search_space, result.x)}
                    last_window_best_params_list = result.x

                elif self.optimizer == 'optuna':
                    # Suppress verbose trial-by-trial output in debug mode
                    if self.debug_mode:
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                    else:
                        # FIX: Reduce verbosity in normal mode to keep the console clean.
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
                    
                    # Optuna doesn't have a direct 'warm start' like skopt, but you can enqueue a trial.
                    if last_window_best_params_list:
                        warm_start_params = {dim.name: val for dim, val in zip(self.search_space, last_window_best_params_list)}
                        study.enqueue_trial(warm_start_params)
                        log_to_file(f"Enqueued warm-start trial for window #{window_num} with best params from window #{i}", print_to_console=False)
                    elif self.warm_start_params and i == 0:
                        log_to_file(f"Enqueued warm-start trial for first window with parameters from '{self.config_path}'.", print_to_console=False)
                        # Ensure all params for warm start are present in the search space to avoid errors
                        warm_start_trial_params = {k: v for k, v in self.warm_start_params.items() if any(k == dim.name for dim in self.search_space)}
                        if len(warm_start_trial_params) == len(self.warm_start_params):
                             study.enqueue_trial(warm_start_trial_params)
                        else:
                            log_to_file("Could not warm-start from config: parameter mismatch.", print_to_console=False)

                    # The objective function now implicitly uses self.chained_capital
                    obj_func_optuna = lambda trial: self.objective_optuna(trial, train_df.copy())
                    study.optimize(obj_func_optuna, n_trials=n_calls, n_jobs=1)
                    

                    if not study.trials or all(t.state == optuna.trial.TrialState.PRUNED for t in study.trials):
                        self.log_problem(f"Window #{window_num}: All optimization trials were pruned. Reusing last good parameters.")
                        if self.all_optimized_params:
                            last_key = sorted(self.all_optimized_params.keys())[-1]
                            best_params = self.all_optimized_params[last_key]
                            log_to_file(f"Reusing parameters from {last_key} for window #{window_num}.", print_to_console=False)
                        else:
                            log_to_file(f"Window #{window_num}: No previous parameters to fall back on. Skipping test phase.", print_to_console=True)
                            continue
                    else:
                        best_params = study.best_params
                        best_score = study.best_value
                        log_to_file(f"Window #{window_num}: Optimization complete. Best Score (Minimized Value): {best_score:.4f}", print_to_console=True)
                        log_to_file(f"Window #{window_num}: Best Params Found: {best_params}", print_to_console=False)
                        # Convert best params to a list for the next warm start
                        last_window_best_params_list = [best_params.get(dim.name) for dim in self.search_space]


            log_to_file(f"Window #{window_num}: Optimization finished. Applying best params to test data.", print_to_console=False)
            self.all_optimized_params[f"Window_{window_num}"] = best_params
            
            if not self.no_cache and self.memory:
                strategy = Strategy(best_params)
                processed_test_df = self.cached_generate_signals(strategy, test_df, self.realism_settings)
            else:
                strategy = Strategy(best_params)
                processed_test_df = strategy.generate_signals(test_df, self.realism_settings)

            if not processed_test_df.empty:
                # IMPORTANT: Log the starting capital *before* running the test and updating the chain.
                start_capital_for_log = self.chained_capital
                
                metrics, trades_df, final_equity = self.run_backtest(best_params, processed_test_df, self.chained_capital)
                
                # --- DEBUG: Log window-specific results ---
                window_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
                self.log_debug(f"Window #{i + 1} Test Results: StartCap=${start_capital_for_log:,.2f}, Trades={len(trades_df)}, PnL=${window_pnl:,.2f}, New Chained Capital=${final_equity:,.2f}")

                # Update the main chained capital for the *next* window's loop.
                self.chained_capital = final_equity if np.isfinite(final_equity) else self.chained_capital
                
                if not trades_df.empty:
                    trades_df['window'] = i + 1
                    # Log the correct starting capital that was used for this window's test.
                    trades_df['window_start_capital'] = start_capital_for_log
                    self.log_trade_details(trades_df, i + 1)
                    self.all_trades.append(trades_df)

            log_to_file(f"--- Finished Window #{window_num}. Chained Capital: ${self.chained_capital:,.2f} ---", print_to_console=False)

        log_to_file("--- All walk-forward windows processed. Finalizing report. ---", print_to_console=True)
        self.finalize_and_report()

    def run_walk_forward_simulation(self, params_to_test):
        """
        Runs a non-optimizing walk-forward analysis with a fixed parameter set.
        This is used to fairly compare the performance of new vs. old parameters.
        """
        log_to_file(f"--- Running simulation for existing parameters... ---", print_to_console=False)
        windows = self.get_walk_forward_windows(self.df_full)
        if not windows:
            return self.default_params.get('INITIAL_CAPITAL', 10000)

        chained_capital = self.default_params.get('INITIAL_CAPITAL', 10000)
        
        for i, (train_start, train_end, test_end) in enumerate(windows):
            test_df = self.df_full.loc[train_end:test_end]
            
            full_params = {**self.default_params, **params_to_test}
            strategy = Strategy(full_params)
            processed_test_df = strategy.generate_signals(test_df, self.realism_settings)
            
            if not processed_test_df.empty:
                _, _, final_equity = self.run_backtest(full_params, processed_test_df, chained_capital)
                chained_capital = final_equity if np.isfinite(final_equity) else chained_capital
        
        log_to_file(f"--- Simulation for existing parameters finished. Final Chained Equity: ${chained_capital:,.2f} ---", print_to_console=False)
        return chained_capital

    def finalize_and_report(self, no_trades=False):
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

            # --- GCP INTEGRATION: Upload the latest parameters to the cloud ---
            if not self.debug_mode:
                log_to_file("Attempting to upload latest parameters to Google Cloud Storage...")
                upload_success = upload_to_gcs(
                    source_file_path=params_save_path,
                    destination_blob_name="latest_live_parameters.json"
                )
                if upload_success:
                    log_to_file("Successfully uploaded parameters to GCS.")
                else:
                    self.log_problem("Failed to upload parameters to GCS. Live bot will not be updated.")

        if no_trades:
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
            
            if not self.debug_mode:
                log_to_file("--- Starting Methodologically Sound Walk-Forward Comparison ---", print_to_console=True)
                
                new_strategy_final_equity = self.chained_capital
                log_to_file(f"Newly Optimized Strategy Final Chained Equity: ${new_strategy_final_equity:,.2f}", print_to_console=True)

                old_best_params = self.config.get('best_parameters_so_far')
                
                if old_best_params:
                    old_strategy_final_equity = self.run_walk_forward_simulation(old_best_params)
                    log_to_file(f"Existing Parameters Final Chained Equity: ${old_strategy_final_equity:,.2f}", print_to_console=True)

                    if new_strategy_final_equity > old_strategy_final_equity:
                        log_to_file("--- RESULT: New parameters are SUPERIOR based on walk-forward equity. Updating config file. ---", print_to_console=True)
                        self.update_config_with_new_params(newly_optimized_params)
                    else:
                        log_to_file("--- RESULT: Existing parameters are SUPERIOR or equal based on walk-forward equity. Config file will NOT be updated. ---", print_to_console=True)
                else:
                    log_to_file("No existing 'best_parameters_so_far' found. Updating config file with new parameters.", print_to_console=True)
                    self.update_config_with_new_params(newly_optimized_params)
            else:
                log_to_file("--- SKIPPING PARAMETER COMPARISON IN DEBUG MODE ---", print_to_console=True)
        else:
            log_to_file("Skipping final parameter comparison: No optimized parameters were found.")

        self.print_problem_summary()

        # --- Roo Fix: Update the pointer file so the live bot can find this run ---
        if self.current_run_dir_static:
            try:
                with open("latest_run_dir.txt", "w") as f:
                    f.write(self.current_run_dir_static)
                log_to_file(f"Updated 'latest_run_dir.txt' to point to: {self.current_run_dir_static}", print_to_console=True)
            except Exception as e:
                self.log_problem(f"Could not update 'latest_run_dir.txt'. Error: {e}")

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
            
            report_path = os.path.join(self.current_run_dir, "performance_report.html")
            
            # Generate the main report content first
            qs.reports.html(returns, output=report_path, title='Comprehensive Strategy Performance')

            # --- Generate and embed additional plots ---
            pnl_dist_plot = plot_pnl_distribution(trades_df, return_html_div=True)
            
            window_plots_html = ""
            for window_num in sorted(trades_df['window'].unique()):
                window_key = f"Window_{window_num}"
                window_trades = trades_df[trades_df['window'] == window_num]
                
                # --- ROO: Removed extra self.current_run_dir argument to fix TypeError ---
                plot_div = plot_trades_for_window(
                    self.df_full,
                    window_trades,
                    window_key,
                    self.all_optimized_params,
                    self.config,
                    return_html_div=True
                )
                if plot_div:
                    window_plots_html += f'<h2>Trade Analysis for {window_key}</h2>'
                    window_plots_html += plot_div

            # Append the new plots to the end of the generated HTML file
            if pnl_dist_plot or window_plots_html:
                with open(report_path, 'a', encoding='utf-8') as f:
                    if pnl_dist_plot:
                        f.write('<h1>Additional Analysis</h1>')
                        f.write(pnl_dist_plot)
                    if window_plots_html:
                        f.write('<h1>Walk-Forward Window Plots</h1>')
                        f.write(window_plots_html)

            log_to_file(f"Successfully generated and enhanced QuantStats report: {report_path}", print_to_console=True)

        except Exception as e:
            log_to_file(f"An error occurred during QuantStats report generation: {e}", print_to_console=True)
            import traceback
            traceback.print_exc()

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

def run_backtest_instance(args):
    """
    Callable instance of the backtester, designed to be run from another script.
    """
    try:
        log_to_file("--- ICHIMOKU BACKTEST PIPELINE STARTED ---", print_to_console=True)
        
        backtester = IchimokuBacktester(
            config_path=args.config,
            intensity_override=args.intensity,
            no_warmup=args.no_warmup,
            optimizer=args.optimizer,
            min_trades_override=args.min_trades,
            runs_to_keep_override=args.runs_to_keep,
            debug_mode=args.debug,
            no_numba=args.no_numba,
            no_cache=args.no_cache,
            clear_cache=args.clear_cache
        )
        
        backtester.run_walk_forward_optimization()
        
        # Return a dictionary of results for post-execution analysis
        return {
            "trades": backtester.all_trades,
            "initial_capital": backtester.default_params.get('INITIAL_CAPITAL', 10000),
            "final_equity": backtester.chained_capital,
            "run_directory": IchimokuBacktester.current_run_dir_static
        }

    except Exception as e:
        tb_str = traceback.format_exc()
        log_to_file(f"FATAL ERROR in backtest instance: {e}\n{tb_str}", print_to_console=True)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ichimoku Cloud Backtester and Optimizer")
    parser.add_argument('--config', type=str, default='optimization_config.json', help='Path to the configuration file.')
    parser.add_argument('--intensity', type=int, help='Override the optimization intensity level (e.g., 1, 2, 3).')
    parser.add_argument('--no-warmup', action='store_true', help='Skip the Numba JIT compilation warm-up.')
    parser.add_argument('--optimizer', type=str, default='optuna', choices=['bayesian', 'optuna'], help='Specify the optimization engine.')
    parser.add_argument('--min-trades', type=int, help='Override the minimum number of trades required per window.')
    parser.add_argument('--runs-to-keep', type=int, help='Override the number of historical run output folders to keep.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for faster, smaller test runs.')
    parser.add_argument('--no-numba', action='store_true', help='Disable Numba JIT for easier debugging of the core logic.')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of data preparation steps.')
    parser.add_argument('--clear-cache', action='store_true', help='Delete the existing cache before running.')
    args = parser.parse_args()
    
    # --- Main Execution ---
    # --- Main Execution ---
    results = run_backtest_instance(args)

    # --- Post-Execution Verification Test (Now runs automatically) ---
    if results:
        from verify_backtest import run_pnl_test
        import io
        import sys
        import json

        print("\n--- Running Post-Execution Verification ---")
        
        # Capture the output of the test script to log it
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Pass the actual results from the backtest to the test function
            run_pnl_test(
                trades_df=pd.concat(results['trades'], ignore_index=True) if results['trades'] else pd.DataFrame(),
                initial_capital=results['initial_capital'],
                final_equity=results['final_equity']
            )
        finally:
            # Ensure stdout is always restored
            sys.stdout = old_stdout

        test_output = captured_output.getvalue()
        
        # Print the captured output to the console and log it to the file
        print(test_output)
        log_to_file("\n--- Post-Execution Verification Results ---\n" + test_output, print_to_console=False)

        # --- Roo Fix: Print results as JSON for the watcher ---
        output_for_watcher = {
            "run_directory": results.get("run_directory"),
            "final_equity": float(results.get("final_equity", 0))
        }
        print("\n---WATCHER_RESULTS_START---")
        print(json.dumps(output_for_watcher))
        print("---WATCHER_RESULTS_END---")