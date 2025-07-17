# ==============================================================================
#
#                         PARAMETER BOUNDS ANALYZER
#
# ==============================================================================
#
# FILE: analyze_parameter_limits.py
#
# PURPOSE:
#   This script is a crucial component of the outer optimization loop, which is
#   orchestrated by `watcher.py`. Its sole purpose is to analyze the results of
#   a completed backtest run and determine if the optimizer is being constrained
#   by the search space boundaries defined in `optimization_config.json`.
#
# METHODOLOGY:
#   The script loads the `optimized_params.json` file from a specific run, which
#   contains the best parameters found for each walk-forward window. It then
#   compares these optimized values against the `bounds` defined for each
#   parameter in the main configuration file.
#
#   It calculates and prints a "hit rate" for each parameter—that is, the
#   percentage of walk-forward windows in which the optimal value for that
#   parameter was exactly the minimum or maximum of its allowed range. This
#   output is then parsed by `watcher.py`.
#
# KEY FEATURES:
#   - Bound-Hitting Detection: Provides the core logic for the second level of
#     the optimization pipeline (optimizing the search space itself).
#   - Clear Reporting: Generates a simple, readable text-based report that
#     is parsed by the `watcher.py` script to make decisions about widening
#     the parameter bounds for the next optimization cycle.
#   - Modularity: Encapsulates the analysis logic in a single `analyze_limits`
#     function that can be easily called from the watcher script.
#
# ==============================================================================

import json
from collections import defaultdict
import os
import glob
import sys
import argparse

def find_latest_results(run_dir):
    """Finds the optimized_params.json file in a specific run directory."""
    if not run_dir or not os.path.isdir(run_dir):
        print(f"Error: Provided run directory '{run_dir}' does not exist.")
        return None
    
    results_path = os.path.join(run_dir, "optimized_params_per_window.json") # FIX: Correct filename
    if not os.path.isfile(results_path):
        print(f"Error: No 'optimized_params_per_window.json' file found in '{run_dir}'.")
        return None
    return results_path

def analyze_limits(args):
    """
    Analyzes the optimized parameters from a specific run to see how many times 
    they hit their defined bounds.
    """
    config_path = args.config
    run_dir = args.run_directory
    results_path = find_latest_results(run_dir)
    if not results_path:
        return

    try:
        with open(config_path, 'r') as f:
            # Using json5 to handle potential comments in the config
            import json5
            config = json5.load(f)
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return
    except ImportError:
        print("Error: json5 library is required. Please install it using 'pip install json5'")
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

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter limits from a backtest run.")
    parser.add_argument(
        "run_directory",
        type=str,
        nargs='?',
        default=None,
        help="The path to the specific run directory (e.g., 'plots_output/20250713_165846'). If omitted, the latest run will be used."
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='optimization_config.json',
        help='Path to the optimization configuration JSON file.'
    )
    args = parser.parse_args()
    
    # If run_directory is not provided, find the latest one
    if not args.run_directory:
        # A bit of a workaround to find the latest dir if not specified
        base_dir = os.path.dirname(args.config) if os.path.dirname(args.config) else '.'
        all_run_dirs = [d for d in glob.glob(os.path.join(base_dir, "plots_output", "*")) if os.path.isdir(d)]
        if not all_run_dirs:
            print("Error: Could not find any run directories in 'plots_output'.")
            return
        args.run_directory = max(all_run_dirs, key=os.path.getmtime)
        print(f"No run directory specified. Analyzing the latest run: {os.path.basename(args.run_directory)}")

    analyze_limits(args)

if __name__ == "__main__":
    main()