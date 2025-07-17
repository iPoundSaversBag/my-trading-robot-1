# ==============================================================================
#
#                         ORCHESTRATION & AUTO-OPTIMIZATION ENGINE
#
# ==============================================================================
#
# FILE: watcher.py
#
# PURPOSE:
#   This script is the master controller of the entire trading robot pipeline.
#   It orchestrates a sophisticated, two-level optimization process, making it
#   the "brain" of the system. It automates the cycle of running backtests,
#   analyzing the results, and then adjusting the system's parameters to
#   continuously search for better performance.
#
# METHODOLOGY:
#   The watcher operates in a continuous loop, performing the following steps:
#   1.  **Inner Loop (Parameter Value Optimization):** It calls the main
#       `ichimoku_backtest.py` script. This script runs a full walk-forward
#       optimization to find the best *values* for the strategy's parameters
#       (like RSI period, ADX threshold, etc.) within the currently defined
#       search space.
#   2.  **Outer Loop (Parameter Bounds Optimization):** After a backtest is
#       complete, the watcher calls the `analyze_parameter_limits.py` script.
#       This analysis script checks if the optimal parameters found in the inner
#       loop are frequently hitting the upper or lower boundaries of their
#       search space.
#   3.  **Self-Correction:** If the analysis shows that the optimizer is being
#       constrained by its boundaries, the watcher script will automatically
#       *modify the `optimization_config.json` file*, widening the search
#       bounds for the constrained parameters.
#   4.  **Iteration:** The watcher then repeats the cycle, running a new backtest
#       with the newly expanded search space. This allows the system to "hill climb"
#       and explore new, potentially more profitable parameter regions over time.
#
# KEY FEATURES:
#   - Two-Tier Optimization: Implements a powerful auto-optimization loop that
#     not only finds the best parameter values but also optimizes the search
#     space itself.
#   - Pipeline Orchestration: Acts as the central script that calls and coordinates
#     all other modules in the system (`ichimoku_backtest`, `analyze_parameter_limits`,
#     `generate_plots`).
#   - Automated Configuration Management: Has the ability to read and write to
#     its own configuration files, allowing for true, hands-off, continuous
#     improvement.
#   - Robust Logging: Maintains a clear and detailed log of all its actions,
#     making it easy to monitor the long-term optimization process.
#
# ==============================================================================

# watcher.py

import os
import sys
import traceback
import subprocess
import time
import json
import json5
from notifications import send_notification, send_error_alert
from datetime import datetime
import re
import argparse
# --- Global Settings ---
LOG_FILE = "watcher_log.txt"
WATCH_INTERVAL_MINUTES = 1 # Default wait time in minutes

def log_message(message):
    """Logs a message to the console and the watcher's log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# --- Import refactored functions ---
from manage_data import download_data, check_data
from analyze_parameter_limits import analyze_limits

# --- CONFIGURATION ---
CONFIG_FILE = "optimization_config.json"
# The percentage to increase/decrease a parameter bound when it gets hit
ADJUSTMENT_FACTOR = 0.20 
# The percentage of times a parameter needs to hit a limit to be adjusted
HIT_THRESHOLD_PERCENT = 0.1 
# How long to wait after starting the live bot to check its logs for errors
BOT_VALIDATION_PERIOD_SECONDS = 60

# [REMOVED] Functions for local bot management are no longer needed.

def run_script(script_name, args=None):
    """Runs a python script and waits for it to complete."""
    command = [sys.executable, script_name]
    if args:
        command.extend(args)
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Script {script_name} output: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_name}: {e.stderr}")
        return None

def parse_analysis(analysis_output):
    """
    Parses the text output of analyze_parameter_limits.py to find which parameters
    are hitting their boundaries and at what frequency.
    """
    params_hitting_limits = {}
    current_param = None
    
    # Regex to find the parameter name, e.g., "ADX_PERIOD | ..."
    param_regex = re.compile(r"^([A-Z_]+)")
    # Regex to find the hit count and percentage, e.g., "- Hit MIN 5/10 times (50.0%)"
    hit_regex = re.compile(r"Hit (MIN|MAX) \d+/\d+ times \(([0-9.]+)%\)")

    for line in analysis_output.splitlines():
        param_match = param_regex.match(line)
        if param_match:
            current_param = param_match.group(1).strip()
        
        hit_match = hit_regex.search(line)
        if hit_match and current_param:
            # Correctly parse the hit percentage
            limit_type = hit_match.group(1) # MIN or MAX
            try:
                hit_percentage_str = hit_match.group(2)
                hit_percentage = float(hit_percentage_str)
                
                # Use the HIT_THRESHOLD_PERCENT from config
                if hit_percentage >= (HIT_THRESHOLD_PERCENT * 100):
                    params_hitting_limits[current_param] = limit_type
                    print(f"FLAGGED: '{current_param}' will be adjusted for hitting {limit_type} limit {hit_percentage:.1f}% of the time.")
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse hit percentage from line: '{line}'. Error: {e}")


    return params_hitting_limits

def update_config_bounds(params_to_update):
    """
    Reads, updates, and writes the optimization_config.json file.
    It widens the bounds for the parameters that were flagged by the analysis.
    """
    if not params_to_update:
        print("\nNo parameters flagged for adjustment. Bounds are optimal for now.")
        return False

    print(f"\n--- Updating bounds in {CONFIG_FILE} ---")
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json5.load(f)
    except Exception as e:
        print(f"ERROR: Could not read or parse {CONFIG_FILE}: {e}")
        return False

    config_was_changed = False
    
    # Iterate through all parameter space groups
    for space_group in config.get('parameter_spaces', {}).values():
        if isinstance(space_group, list):
            for param_info in space_group:
                param_name = param_info.get('name')
                if param_name in params_to_update:
                    limit_hit = params_to_update[param_name]
                    current_bounds = param_info['bounds']
                    param_type = param_info.get('type', 'real')
                    hard_min, hard_max = param_info.get('hard_bounds', [float('-inf'), float('inf')])
                    
                    print(f"  - Adjusting '{param_name}' | Current: {current_bounds} | Hard: [{hard_min}, {hard_max}]")

                    original_bounds = list(current_bounds)

                    if limit_hit == 'MAX':
                        new_max = current_bounds[1] * (1 + ADJUSTMENT_FACTOR)
                        new_max = min(new_max, hard_max)
                        current_bounds[1] = int(new_max) if param_type.lower() == 'integer' else round(new_max, 4)
                    elif limit_hit == 'MIN':
                        new_min = current_bounds[0] * (1 - ADJUSTMENT_FACTOR)
                        new_min = max(1.0, new_min) if "PERIOD" in param_name.upper() else max(0.0, new_min)
                        new_min = max(new_min, hard_min)
                        current_bounds[0] = int(new_min) if param_type.lower() == 'integer' else round(new_min, 4)
                    
                    if original_bounds != current_bounds:
                        config_was_changed = True
                        print(f"    New Bounds -> {current_bounds}")
                    else:
                        print(f"    Bounds for '{param_name}' are at their hard limit.")

    if not config_was_changed:
        print("No meaningful bound updates could be made.")
        return False

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Successfully updated bounds in {CONFIG_FILE}.")
        return True
    except Exception as e:
        print(f"ERROR: Could not write updated {CONFIG_FILE}: {e}")
        return False

# [REMOVED] Functions for local bot management are no longer needed.

def main_loop(args):
    """
    The main continuous loop for the watcher.
    """
    run_count = 0
    first_run = True

    while True:
        run_count += 1
        print(f"\n{'='*40}\n{'='*10} STARTING WATCHER CYCLE #{run_count} {'='*10}\n{'='*40}")

        send_notification("Watcher: Starting new optimization cycle.")

        # --- Step 1: Data Management ---
        if first_run and not args.skip_download:
            print("\n--- (First Cycle) Downloading and verifying data ---")
            if not download_data(args):
                sys.exit("Exiting due to failure in data download.")
            if not check_data(args):
                sys.exit("Exiting due to failure in data integrity check.")
            first_run = False
        
        # --- Step 2: Run the Self-Optimizing Backtest ---
        print("\n--- Running self-optimizing backtest ---")
        # The backtest script now returns the directory of its results
        # --- Roo Fix: Run backtest as a separate, isolated process ---
        backtest_command = [
            sys.executable, "ichimoku_backtest.py",
            '--config', args.config,
            '--intensity', str(args.intensity),
            '--runs-to-keep', str(args.runs_to_keep),
            '--optimizer', args.optimizer
        ]
        if args.no_warmup: backtest_command.append('--no-warmup')
        if args.min_trades: backtest_command.extend(['--min-trades', str(args.min_trades)])
        if args.debug: backtest_command.append('--debug')
        if args.no_numba: backtest_command.append('--no-numba')

        log_message(f"Executing command: {' '.join(backtest_command)}")
        
        try:
            process = subprocess.run(backtest_command, check=True, text=True, capture_output=True)
            
            # --- Roo Fix: Parse JSON output from the backtest script ---
            output = process.stdout
            json_output_str = re.search(r"---WATCHER_RESULTS_START---(.*?)---WATCHER_RESULTS_END---", output, re.DOTALL)
            
            if not json_output_str:
                log_message("ERROR: Could not find watcher results JSON in backtest output.")
                send_error_alert("Watcher: Backtest ran but failed to return results.")
                time.sleep(60)
                continue

            backtest_results = json5.loads(json_output_str.group(1))
            latest_run_dir = backtest_results.get("run_directory")

            if not latest_run_dir:
                log_message("ERROR: Backtest results JSON is missing the 'run_directory'.")
                time.sleep(60)
                continue
                
            log_message(f"Backtest complete. Results saved in: {latest_run_dir}")

        except subprocess.CalledProcessError as e:
            log_message(f"\n--- Backtest script failed with exit code {e.returncode}. Waiting 60 seconds before retrying. ---")
            log_message(f"STDOUT: {e.stdout}")
            log_message(f"STDERR: {e.stderr}")
            send_error_alert(f"Watcher: The backtest script failed to execute. Check logs. STDERR: {e.stderr}")
            time.sleep(60)
            continue
        except Exception as e:
            log_message(f"An unexpected error occurred while running the backtest subprocess: {e}")
            send_error_alert(f"Watcher: An unexpected error occurred. Check logs. Error: {e}")
            time.sleep(60)
            continue
        send_notification(f"Watcher: Backtest complete. Results are in {latest_run_dir}.")

        # --- Step 3: Analyze Parameter Bounds ---
        print(f"\n--- Analyzing parameter bounds from: {latest_run_dir} ---")
        # We need to capture the stdout of the analysis script to parse it
        # This is a bit tricky when calling functions directly. A temporary redirect of stdout is one way.
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # FIX: Pass the run_directory to the analysis script via the args object
        args.run_directory = latest_run_dir
        analyze_limits(args)
        
        sys.stdout = old_stdout # Restore stdout
        analysis_text = captured_output.getvalue()
        print(analysis_text) # Print the captured analysis

        # --- Step 5: Update Configuration Bounds ---
        # This part remains the same as it was parsing text output
        params_to_update = parse_analysis(analysis_text)
        config_updated = update_config_bounds(params_to_update)
        
        # --- Step 6: Live Bot Validation (REMOVED) ---
        # This step is no longer needed as the live bot runs independently on the VM
        # and will pick up the new parameters automatically from Google Cloud Storage.
        log_message("Watcher cycle complete. New parameters have been uploaded to GCS for the live bot.")

        print(f"\n{'='*40}\n{'='*10} WATCHER CYCLE #{run_count} COMPLETE {'='*10}\n{'='*40}")
        
        if args.test_run:
            log_message("Test run complete. Exiting.")
            break
            
        wait_interval_minutes = args.interval if args.interval is not None else WATCH_INTERVAL_MINUTES
        send_notification(f"Watcher: Cycle complete. Next run in {wait_interval_minutes} minutes.")
        print(f"Waiting for {wait_interval_minutes} minutes before starting the next cycle...")
        time.sleep(wait_interval_minutes * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Trading Strategy Watcher and Optimizer.")
    # Add all arguments from ichimoku_backtest.py to the watcher
    parser.add_argument('-c', '--config', type=str, default='optimization_config.json', help='Path to the configuration file.')
    parser.add_argument('-i', '--intensity', type=int, default=1, help='Optimization intensity level (1-4).')
    parser.add_argument('--runs-to-keep', type=int, default=20, help='Number of old run directories to keep.')
    parser.add_argument('--interval', type=int, help="Override the watch interval in minutes.")
    parser.add_argument('--skip-download', action='store_true', help='Skip initial data download.')
    parser.add_argument('--no-warmup', action='store_true', help='Disable Numba JIT warm-up.')
    parser.add_argument('--min-trades', type=int, help='Override minimum trades per window.')
    parser.add_argument('--optimizer', type=str, default='optuna', choices=['bayesian', 'optuna'], help='Optimizer to use.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--no-numba', action='store_true', help='Disable the Numba JIT compiler for the backtesting core.')
    parser.add_argument('--test-run', action='store_true', help='Run the watcher for a single cycle and then exit.')
    
    args = parser.parse_args()

    try:
        # Pass all parsed arguments to the main loop
        main_loop(args)
    except KeyboardInterrupt:
        print("\n\nWatcher script stopped manually by user. Exiting.")
        sys.exit(0)