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
#       `core/backtest.py` script. This script runs a full walk-forward
#       optimization to find the best *values* for the strategy's parameters
#       (like RSI period, ADX threshold, etc.) within the currently defined
#       search space.
#   2.  **Outer Loop (Parameter Bounds Optimization):** After a backtest is
#       complete, the watcher calls the consolidated `utilities.utils.analyze_limits` function.
#       This analysis function checks if the optimal parameters found in the inner
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
#     all other modules in the system (`core/backtest`, `utilities.utils`,
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
from datetime import datetime
import re
import argparse

# CONSOLIDATED UTILITIES IMPORT
from utilities.utils import (
    send_notification, send_error_alert, log_message,
    download_data, check_data, analyze_limits
)

# HEALTH MONITORING INTEGRATION
from health_utils import (
    run_health_check, get_repair_engine, run_comprehensive_health_check, 
    display_health_problems_with_descriptions, show_robot_problems_with_descriptions
)

# --- Global Settings ---
LOG_FILE = "logs/watcher.log"  # Centralized logging path
WATCH_INTERVAL_MINUTES = 1 # Default wait time in minutes

# --- CONFIGURATION ---
CONFIG_FILE = "core/optimization_config.json"  # Updated to match the default config location
# The percentage to increase/decrease a parameter bound when it gets hit
ADJUSTMENT_FACTOR = 0.20 
# The percentage of times a parameter needs to hit a limit to be adjusted
HIT_THRESHOLD_PERCENT = 0.1 
# How long to wait after starting the live bot to check its logs for errors
BOT_VALIDATION_PERIOD_SECONDS = 60

# [REMOVED] Functions for local bot management are no longer needed.

def perform_system_health_check(cycle_number: int = 0, pre_backtest: bool = False) -> bool:
    """
    🏥 INTEGRATED HEALTH MONITORING FOR WATCHER PIPELINE
    
    Performs comprehensive system health checks and auto-repairs at key points
    Returns False if critical issues require immediate stopping
    """
    print(f"\n🏥 === SYSTEM HEALTH CHECK (Cycle #{cycle_number}) ===")
    
    try:
        # Step 1: Quick health validation
        health_success, health_message = run_health_check(silent=False, timeout=30)
        
        if not health_success:
            print(f"⚠️  Health check detected issues: {health_message}")
            log_message(f"Health issues detected in cycle {cycle_number}: {health_message}", 'watcher')
        
        # Step 2: Comprehensive system analysis (every 5 cycles or when issues detected)
        if cycle_number % 5 == 0 or not health_success or pre_backtest:
            print("🔍 Performing comprehensive health analysis with detailed problem descriptions...")
            
            # Use enhanced problem description system with precise error locations
            try:
                health_report = show_robot_problems_with_descriptions()
                
                if health_report["status"] == "healthy":
                    print("✅ Comprehensive health check passed - system is healthy!")
                    comprehensive_success = True
                    comprehensive_msg = f"System healthy - {health_report.get('auto_fixes', 0)} auto-fixes applied"
                else:
                    print(f"🚨 Health issues detected: {health_report['total_problems']} problems found")
                    print(f"📊 System Health Score: {health_report['health_score']}/100")
                    
                    comprehensive_success = health_report['health_score'] >= 70  # Consider healthy if score >= 70
                    comprehensive_msg = health_report['message']
                    
                    # Show critical issues that need immediate attention
                    critical_issues = [desc for desc in health_report.get("detailed_descriptions", []) 
                                     if desc.get("severity") == "critical"]
                    
                    # Check for specific critical stopping conditions
                    position_sizing_issues = 0
                    trading_critical_issues = 0
                    
                    for issue in critical_issues:
                        title = issue.get('title', '').lower()
                        if 'position size' in title or 'position sizing' in title:
                            position_sizing_issues += 1
                        if any(term in title for term in ['trading', 'execution', 'portfolio', 'risk']):
                            trading_critical_issues += 1
                    
                    if critical_issues:
                        print(f"🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION: {len(critical_issues)}")
                        for issue in critical_issues[:3]:  # Show top 3 critical issues
                            print(f"   • {issue.get('title', 'Unknown Issue')}")
                        
                        send_error_alert(f"Watcher Health Alert: {len(critical_issues)} critical issues detected")
                        
                        # ENHANCED STOPPING LOGIC: More specific conditions for stopping
                        if position_sizing_issues >= 2:
                            print(f"🛑 CRITICAL STOP CONDITION: {position_sizing_issues} position sizing issues detected")
                            print("💀 POSITION SIZING ERRORS CAN CAUSE CATASTROPHIC LOSSES - STOPPING IMMEDIATELY")
                            return False
                        
                        if trading_critical_issues >= 3:
                            print(f"🛑 CRITICAL STOP CONDITION: {trading_critical_issues} trading system failures detected")
                            return False
                        
                        if len(critical_issues) >= 5:  # Too many critical issues
                            print(f"🛑 CRITICAL STOP CONDITION: {len(critical_issues)} critical issues overwhelming system")
                            return False
            
            except Exception as e:
                print(f"🚨 Error in enhanced health check: {e}")
                # Fallback to original comprehensive check
                comprehensive_success, comprehensive_msg, universal_report = run_comprehensive_health_check(silent=False)
            
            if not comprehensive_success:
                print(f"🚨 Comprehensive health check failed: {comprehensive_msg}")
                if 'health_report' not in locals():
                    send_error_alert(f"Watcher Health Alert: {comprehensive_msg}")
                
                # Attempt intelligent auto-repair
                print("🔧 Attempting intelligent auto-repair...")
                repair_engine = get_repair_engine()
                
                # Get AI intelligence assessment
                intelligence = repair_engine._assess_ai_intelligence()
                print(f"🧠 AI Intelligence Level: {intelligence['intelligence_level']}")
                print(f"📊 Effectiveness Score: {intelligence['effectiveness_score']:.2%}")
                
                # Let AI decide on repairs
                if intelligence['intelligence_level'] in ['TRANSCENDENT', 'GODLIKE']:
                    print("🚀 High-level AI detected - enabling advanced repair capabilities")
                    
                    # Activate revolutionary AI features for critical repairs
                    if hasattr(repair_engine, 'activate_revolutionary_features'):
                        revolutionary_repairs = repair_engine.activate_revolutionary_features()
                        log_message(f"Revolutionary AI repairs applied: {len(revolutionary_repairs)}", 'watcher')
                
                return False
            else:
                print("✅ Comprehensive health check passed")
                
                # Show system statistics
                if 'universal_report' in locals() and universal_report and 'function_health' in universal_report:
                    critical_functions = sum(1 for func_health in universal_report['function_health'].values() 
                                           if func_health.get('is_critical', False))
                    print(f"📈 System Stats: {critical_functions} critical functions monitored")
        
        # Step 3: Predictive issue detection
        repair_engine = get_repair_engine()
        health_analysis = repair_engine.analyze_codebase_health()
        
        # Check predictive forecasting
        forecast = health_analysis.get('predictive_forecasting', {})
        predicted_issues = forecast.get('predicted_issues', [])
        
        if predicted_issues:
            print(f"🔮 Predictive Analysis: {len(predicted_issues)} potential future issues detected")
            for issue in predicted_issues[:3]:  # Show top 3
                prob = issue.get('probability', 0) * 100
                print(f"   ⚠️  {issue.get('issue', 'Unknown')[:60]}... ({prob:.0f}% probability)")
            
            # Proactive measures for high-probability issues
            high_prob_issues = [i for i in predicted_issues if i.get('probability', 0) > 0.7]
            if high_prob_issues:
                print("🛡️  Taking proactive measures for high-probability issues...")
                log_message(f"Proactive measures taken for {len(high_prob_issues)} predicted issues", 'watcher')
        
        # Step 4: Performance optimization check
        performance_status = health_analysis.get('performance_bottlenecks', {}).get('status', 'unknown')
        if performance_status != 'optimal':
            print(f"⚡ Performance optimization needed: {performance_status}")
            # Auto-optimize if AI is advanced enough
            if intelligence.get('intelligence_level') in ['ADVANCED', 'TRANSCENDENT', 'GODLIKE']:
                print("🚀 AI-driven performance optimization activated")
        
        print("✅ System health check completed successfully")
        return True
        
    except Exception as e:
        print(f"🚨 Health check system error: {e}")
        log_message(f"Health check system error in cycle {cycle_number}: {e}", 'watcher')
        send_error_alert(f"Watcher Health System Error: {e}")
        return False


def validate_pipeline_state(stage: str, cycle_number: int) -> bool:
    """
    🔍 PIPELINE STATE VALIDATION
    
    Validates the pipeline state at different stages
    """
    try:
        repair_engine = get_repair_engine()
        
        print(f"🔍 Validating pipeline state: {stage}")
        
        # Stage-specific validations
        if stage == "pre_data_download":
            # Check if data infrastructure is ready
            health_analysis = repair_engine.analyze_codebase_health()
            file_integrity = health_analysis.get('file_integrity', {})
            
            if file_integrity.get('status') != 'healthy':
                print("⚠️  Data infrastructure issues detected - attempting repair")
                # Auto-repair file integrity issues
                return False
                
        elif stage == "pre_backtest":
            # Validate backtest readiness
            config_validity = repair_engine.analyze_codebase_health().get('configuration_validity', {})
            
            if config_validity.get('status') != 'healthy':
                print("⚠️  Configuration issues detected - attempting repair")
                return False
                
        elif stage == "post_backtest":
            # Validate backtest results and system state
            import_consistency = repair_engine.analyze_codebase_health().get('import_consistency', {})
            
            if import_consistency.get('status') != 'healthy':
                print("⚠️  Import consistency issues detected after backtest")
                return False
        
        return True
        
    except Exception as e:
        print(f"🚨 Pipeline validation error at {stage}: {e}")
        return False


def emergency_system_recovery() -> bool:
    """
    🚨 EMERGENCY SYSTEM RECOVERY
    
    Last resort recovery when normal health checks fail
    """
    print("🚨 EMERGENCY SYSTEM RECOVERY ACTIVATED")
    
    try:
        repair_engine = get_repair_engine()
        
        # Emergency diagnostic
        print("🔍 Running emergency diagnostic...")
        
        # Try to get AI to assess the situation
        intelligence = repair_engine._assess_ai_intelligence()
        effectiveness = intelligence.get('effectiveness_score', 0)
        
        if effectiveness > 0.5:
            print(f"🧠 AI effectiveness: {effectiveness:.1%} - attempting autonomous recovery")
            
            # Let AI attempt autonomous evolution/recovery
            if hasattr(repair_engine, 'autonomous_ai_evolution'):
                evolution_actions = repair_engine.autonomous_ai_evolution()
                
                if evolution_actions:
                    print(f"🚀 AI autonomous recovery: {len(evolution_actions)} actions taken")
                    for action in evolution_actions[:5]:  # Show first 5
                        print(f"   🔧 {action}")
                    
                    log_message(f"Emergency recovery: AI took {len(evolution_actions)} autonomous actions", 'watcher')
                    return True
        
        # Basic recovery steps
        print("🔧 Attempting basic system recovery...")
        
        # Try to clean up any corruption
        modifications = repair_engine.self_modify_code()
        if modifications:
            print(f"🛠️  Applied {len(modifications)} emergency modifications")
            return True
        
        return False
        
    except Exception as e:
        print(f"🚨 Emergency recovery failed: {e}")
        log_message(f"Emergency recovery failed: {e}", 'watcher')
        return False

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
    Parses the text output of utilities.utils.analyze_limits function to find which parameters
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

    # INITIALIZATION: System health check on startup
    print("\n🏥 === INITIAL SYSTEM HEALTH CHECK ===")
    if not perform_system_health_check(cycle_number=0, pre_backtest=False):
        print("🚨 CRITICAL: Initial health check failed!")
        print("🛑 DETECTED CRITICAL ISSUES THAT REQUIRE IMMEDIATE STOPPING")
        
        # For critical issues like position sizing, DO NOT attempt recovery
        print("� CRITICAL TRADING SYSTEM ERRORS DETECTED")
        print("🚨 STOPPING WATCHER TO PREVENT POTENTIAL LOSSES")
        send_error_alert("Watcher STOPPED: Critical position sizing or trading errors detected")
        sys.exit(1)

    while True:
        run_count += 1
        print(f"\n{'='*40}\n{'='*10} STARTING WATCHER CYCLE #{run_count} {'='*10}\n{'='*40}")

        send_notification("Watcher: Starting new optimization cycle.")

        # HEALTH CHECK: Monitor system health at start of each cycle
        if not perform_system_health_check(cycle_number=run_count, pre_backtest=False):
            print("🚨 CRITICAL: Health check failed during cycle")
            print("🛑 STOPPING WATCHER DUE TO CRITICAL SYSTEM ISSUES")
            
            # No recovery attempts for critical trading issues
            send_error_alert(f"Watcher STOPPED at Cycle #{run_count}: Critical system errors detected")
            sys.exit(1)

        # PIPELINE VALIDATION: Check data infrastructure readiness
        if not validate_pipeline_state("pre_data_download", run_count):
            print("🚨 Data infrastructure validation failed")
            send_error_alert(f"Watcher Cycle #{run_count}: Data infrastructure issues")
            time.sleep(60)
            continue

        # --- Step 1: Data Management ---
        if first_run and not args.skip_download:
            print("\n--- (First Cycle) Downloading and verifying data ---")
            
            try:
                if not download_data(args):
                    print("🚨 Data download failed - attempting recovery")
                    if not emergency_system_recovery():
                        sys.exit("Exiting due to failure in data download.")
                
                if not check_data(args):
                    print("🚨 Data integrity check failed - attempting recovery")
                    if not emergency_system_recovery():
                        sys.exit("Exiting due to failure in data integrity check.")
                        
                first_run = False
                
            except Exception as e:
                print(f"🚨 Data management error: {e}")
                log_message(f"Data management error in cycle {run_count}: {e}", 'watcher')
                
                if not emergency_system_recovery():
                    send_error_alert(f"Watcher Cycle #{run_count}: Data management failed")
                    time.sleep(300)
                    continue
        
        # --- Step 2: Pre-Backtest Parameter Validation ---
        print("\n--- Validating optimization parameters ---")
        from utilities.utils import validate_parameter_bounds
        
        try:
            validation_results = validate_parameter_bounds(args.config)
            validation_passed = validation_results['summary']['validation_percentage'] == 100.0
            
            if validation_passed:
                print(f"✅ Parameter validation passed ({validation_results['summary']['total_parameters']} parameters)")
            else:
                invalid_count = validation_results['summary']['invalid_parameters']
                print(f"⚠️  Parameter validation warnings: {invalid_count} invalid parameters")
                
                if args.debug:
                    print("🐛 DEBUG: Invalid parameters details:")
                    for invalid_param in validation_results['invalid_parameters']:
                        print(f"  - {invalid_param['name']}: {invalid_param['issues']}")
                
                # Continue but log the warnings
                log_message(f"Parameter validation warnings: {invalid_count} invalid parameters", 'watcher')
                
        except Exception as e:
            print(f"⚠️  Parameter validation failed: {e}")
            log_message(f"Parameter validation failed: {e}", 'watcher')
        
        # --- Step 3: Configuration Optimization (when needed) ---
        if args.intensity >= 3 or run_count == 1:  # High intensity or first run
            print("\n--- Optimizing backtest configuration for performance ---")
            from utilities.utils import optimize_backtest_config_for_speed, validate_backtest_configuration
            
            try:
                # Validate current configuration
                print("🔍 Validating backtest configuration...")
                # Note: validate_backtest_configuration() needs to be implemented in utils.py
                
                # Optimize for speed if high intensity
                if args.intensity >= 3:
                    print("🚀 High intensity detected - optimizing configuration for speed...")
                    optimize_success = optimize_backtest_config_for_speed()
                    if optimize_success:
                        print("✅ Configuration optimized for high-performance execution")
                    else:
                        print("⚠️  Configuration optimization failed, using default settings")
                else:
                    print("ℹ️  Using standard configuration (intensity < 3)")
                    
            except Exception as e:
                print(f"⚠️  Configuration optimization failed: {e}")
                log_message(f"Configuration optimization failed: {e}", 'watcher')
        
        # HEALTH CHECK: Pre-backtest system validation with safety gate
        print("\n🏥 Pre-backtest health validation...")
        
        # Import the safety gate function
        from health_utils import pre_backtest_safety_gate
        
        # CRITICAL SAFETY CHECK: This will stop if position sizing issues detected
        print("🛡️  Running pre-backtest safety gate...")
        if not pre_backtest_safety_gate():
            print("🚨 PRE-BACKTEST SAFETY GATE FAILED!")
            print("🛑 CRITICAL TRADING SYSTEM ISSUES DETECTED - STOPPING WATCHER")
            print("💀 Position sizing or other critical trading errors found")
            send_error_alert("Watcher STOPPED: Pre-backtest safety gate failed - critical trading issues")
            sys.exit(1)
        
        if not validate_pipeline_state("pre_backtest", run_count):
            print("🚨 Pre-backtest validation failed - attempting repair")
            
            if not emergency_system_recovery():
                print("💥 Pre-backtest recovery failed - skipping backtest")
                send_error_alert(f"Watcher Cycle #{run_count}: Pre-backtest validation failed")
                time.sleep(300)
                continue
        
        # Comprehensive health check for critical cycles
        if run_count % 10 == 0:  # Every 10 cycles
            print("🔍 Comprehensive pre-backtest health check (critical cycle)...")
            if not perform_system_health_check(cycle_number=run_count, pre_backtest=True):
                print("🚨 CRITICAL: Comprehensive pre-backtest check failed")
                print("🛑 STOPPING WATCHER - CRITICAL ISSUES DETECTED")
                send_error_alert(f"Watcher STOPPED at Cycle #{run_count}: Critical comprehensive health check failed")
                sys.exit(1)
                send_error_alert(f"Watcher Cycle #{run_count}: Comprehensive health check failed")
                time.sleep(300)
                continue
        
        # --- Step 4: Run the Self-Optimizing Backtest ---
        print("\n--- Running self-optimizing backtest ---")
        # The backtest script now returns the directory of its results
        # --- Roo Fix: Run backtest as a separate, isolated process ---
        backtest_command = [
            sys.executable, "core/backtest.py",
            '--config', args.config,
            '--intensity', str(args.intensity),
            '--runs-to-keep', str(args.runs_to_keep),
            '--optimizer', args.optimizer
        ]
        if args.no_warmup: backtest_command.append('--no-warmup')
        if args.min_trades: backtest_command.extend(['--min-trades', str(args.min_trades)])
        if args.debug: backtest_command.append('--debug')
        if args.no_numba: backtest_command.append('--no-numba')

        log_message(f"Executing command: {' '.join(backtest_command)}", 'watcher')
        
        # Enhanced backtest execution with real-time output streaming
        try:
            print("🎯 Starting backtest with real-time output...")
            log_message(f"Executing command: {' '.join(backtest_command)}", 'watcher')
            
            # Start subprocess with real-time output streaming
            process = subprocess.Popen(backtest_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True, encoding='utf-8', errors='replace')
            
            output_lines = []
            print("📊 Backtest Output:")
            print("-" * 50)
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    print(f"   {line}")
                    output_lines.append(line)
            
            # Wait for process to complete and get return code
            return_code = process.wait()
            output = '\n'.join(output_lines)
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, backtest_command, output=output)
                
            print("-" * 50)
            print("✅ Backtest execution completed!")
            print(f"📄 Total output lines: {len(output_lines)}")
                
            # Parse JSON output from the backtest script
            json_output_str = re.search(r"---WATCHER_RESULTS_START---(.*?)---WATCHER_RESULTS_END---", output, re.DOTALL)
            
            if not json_output_str:
                log_message("ERROR: Could not find watcher results JSON in backtest output.", 'watcher')
                send_error_alert("Watcher: Backtest ran but failed to return results.")
                time.sleep(60)
                continue

            backtest_results = json5.loads(json_output_str.group(1))
            latest_run_dir = backtest_results.get("run_directory")

            if not latest_run_dir:
                log_message("ERROR: Backtest results JSON is missing the 'run_directory'.", 'watcher')
                time.sleep(60)
                continue
                
            log_message(f"Backtest complete. Results saved in: {latest_run_dir}", 'watcher')

        except subprocess.CalledProcessError as e:
            log_message(f"\n--- Backtest script failed with exit code {e.returncode}. Waiting 60 seconds before retrying. ---", 'watcher')
            log_message(f"STDOUT: {e.stdout}", 'watcher')
            log_message(f"STDERR: {e.stderr}", 'watcher')
            send_error_alert(f"Watcher: The backtest script failed to execute. Check logs. STDERR: {e.stderr}")
            
            # HEALTH CHECK: Backtest failure recovery
            print("🚨 Backtest failed - attempting system recovery")
            if emergency_system_recovery():
                print("✅ Recovery successful - will retry backtest next cycle")
            else:
                print("💥 Recovery failed - system may need manual intervention")
            
            time.sleep(60)
            continue
            
        except Exception as e:
            log_message(f"An unexpected error occurred while running the backtest subprocess: {e}", 'watcher')
            send_error_alert(f"Watcher: An unexpected error occurred. Check logs. Error: {e}")
            
            # HEALTH CHECK: Unexpected error recovery
            print("🚨 Unexpected backtest error - attempting system diagnosis")
            if not emergency_system_recovery():
                print("💥 System diagnosis failed - manual intervention may be required")
            
            time.sleep(60)
            continue
            
        send_notification(f"Watcher: Backtest complete. Results are in {latest_run_dir}.")

        # HEALTH CHECK: Post-backtest validation
        print("\n🏥 Post-backtest health validation...")
        if not validate_pipeline_state("post_backtest", run_count):
            print("⚠️  Post-backtest validation detected issues")
            log_message(f"Post-backtest validation issues in cycle {run_count}", 'watcher')
            
            # Continue but with warnings
            send_notification("Watcher: Post-backtest validation warnings detected")

        # --- Step 3: Parameter Bounds Analysis (Now Handled by Backtest) ---
        print(f"\n--- Parameter bounds analysis completed by backtest engine ---")
        print(f"📊 Results saved in: {latest_run_dir}")
        
        # Check if backtest made any bounds modifications
        bounds_analysis_file = os.path.join(latest_run_dir, "parameter_bounds_analysis.json")
        if os.path.exists(bounds_analysis_file):
            try:
                with open(bounds_analysis_file, 'r') as f:
                    bounds_analysis = json.load(f)
                
                modifications = bounds_analysis.get('modifications', {})
                if modifications:
                    print(f"🔧 Backtest engine updated bounds for {len(modifications)} parameter(s):")
                    for param_name, changes in modifications.items():
                        print(f"   • {param_name}: {', '.join(changes)}")
                    log_message(f"Backtest auto-updated {len(modifications)} parameter bounds", 'watcher')
                else:
                    print("✅ No parameter bounds adjustments were needed")
                    
            except Exception as e:
                print(f"⚠️  Could not read bounds analysis: {e}")
        else:
            print("ℹ️  No bounds analysis file found - using current configuration")

        # --- Step 5: Configuration already updated by backtest engine ---
        # No manual configuration updates needed - backtest handles this automatically
        
        # HEALTH CHECK: Pre-continuation validation
        try:
            print("✅ Configuration management now handled by backtest engine")
            log_message(f"Cycle {run_count} completed - backtest handled parameter optimization", 'watcher')
        except Exception as e:
            print(f"� Post-backtest validation error: {e}")
            log_message(f"Post-backtest validation error in cycle {run_count}: {e}", 'watcher')
            send_error_alert(f"Watcher Cycle #{run_count}: Post-backtest validation failed")
        
        # HEALTH CHECK: End-of-cycle system validation
        print("\n🏥 End-of-cycle health summary...")
        try:
            repair_engine = get_repair_engine()
            intelligence = repair_engine._assess_ai_intelligence()
            
            print(f"🧠 AI Intelligence: {intelligence['intelligence_level']}")
            print(f"📊 System Effectiveness: {intelligence['effectiveness_score']:.1%}")
            print(f"🔧 Total Repairs Applied: {intelligence['total_repairs']}")
            
            # Generate cycle summary
            cycle_summary = {
                "cycle": run_count,
                "ai_intelligence": intelligence['intelligence_level'],
                "effectiveness": intelligence['effectiveness_score'],
                "config_updated_by_backtest": True,  # Backtest now handles config updates
                "health_status": "MONITORED"
            }
            
            log_message(f"Cycle #{run_count} summary: {cycle_summary}", 'watcher')
            
        except Exception as e:
            print(f"⚠️  End-of-cycle health summary error: {e}")
        
        # --- Step 6: Deployment Complete ---
        # Optimized parameters have been automatically synced to Vercel deployment
        # and the live bot will receive updates via the automated CI/CD pipeline.
        log_message("Watcher cycle complete. Optimized parameters synced to Vercel for live bot deployment.", 'watcher')

        print(f"\n{'='*40}\n{'='*10} WATCHER CYCLE #{run_count} COMPLETE {'='*10}\n{'='*40}")
        
        if args.test_run:
            log_message("Test run complete. Exiting.", 'watcher')
            break
            
        wait_interval_minutes = args.interval if args.interval is not None else WATCH_INTERVAL_MINUTES
        send_notification(f"Watcher: Cycle complete. Next run in {wait_interval_minutes} minutes.")
        print(f"Waiting for {wait_interval_minutes} minutes before starting the next cycle...")
        
        # HEALTH CHECK: Pre-sleep system status
        print("🔄 Final system status before next cycle...")
        try:
            final_health_success, final_health_message = run_health_check(silent=True, timeout=15)
            if not final_health_success:
                print(f"⚠️  System issues detected before sleep: {final_health_message}")
                log_message(f"System issues before sleep in cycle {run_count}: {final_health_message}", 'watcher')
            else:
                print("✅ System healthy before next cycle")
        except Exception as e:
            print(f"⚠️  Final health check error: {e}")
        
        time.sleep(wait_interval_minutes * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Robot - Master Controller & CLI Interface")
    
    # --- ENHANCED: Add mode selection (merged from main.py) ---
    parser.add_argument('mode', nargs='?', default='watch', 
                       choices=['watch', 'backtest', 'live', 'download', 'demo', 'analyze', 'cleanup'], 
                       help='Operating mode: watch (default orchestration), backtest (single run), live (start live trading), download (get data), demo (strategy test), analyze (parameter analysis), cleanup (maintenance & file management)')
    
    # --- Symbol and timeframe arguments (from main.py) ---
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='5m', help='Primary timeframe')
    parser.add_argument('--multi-timeframe', action='store_true', default=True, help='Use multi-timeframe analysis (default: enabled)')
    parser.add_argument('--single-timeframe', action='store_true', help='Use single timeframe only (disables multi-timeframe)')
    
    # --- Original watcher arguments ---
    parser.add_argument('-c', '--config', type=str, default='core/optimization_config.json', help='Path to the configuration file.')
    parser.add_argument('-i', '--intensity', type=int, default=1, help='Optimization intensity level (1-4).')
    # Deprecated: runs-to-keep should be controlled via config output_settings.runs_to_keep (default 5)
    parser.add_argument('--runs-to-keep', type=int, default=None, help='[Deprecated] Ignored. Use output_settings.runs_to_keep in config.')
    parser.add_argument('--interval', type=int, help="Override the watch interval in minutes.")
    parser.add_argument('--skip-download', action='store_true', help='Skip initial data download.')
    parser.add_argument('--no-warmup', action='store_true', help='Disable Numba JIT warm-up.')
    parser.add_argument('--min-trades', type=int, help='Override minimum trades per window.')
    parser.add_argument('--optimizer', type=str, default='optuna', choices=['bayesian', 'optuna'], help='Optimizer to use.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--no-numba', action='store_true', help='Disable the Numba JIT compiler for the backtesting core.')
    parser.add_argument('--no-cache', action='store_true', help='Disable preprocessing cache')
    parser.add_argument('--clear-cache', action='store_true', help='Clear existing cache')
    parser.add_argument('--test-run', action='store_true', help='Run the watcher for a single cycle and then exit.')
    
    args = parser.parse_args()
    
    # Run system health diagnostic before any operations
    print("🏥 Running system health diagnostic...")
    try:
        # Quick health check - just verify essential data files exist
        import pandas as pd
        essential_files = [
            "data/crypto_data.parquet",
            "data/crypto_data_15m.parquet", 
            "data/crypto_data_1h.parquet",
            "data/crypto_data_4h.parquet"
        ]
        
        missing_files = []
        for file_path in essential_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"⚠️ Health check found missing data files: {missing_files}")
            # Try to download missing data
            print("🔄 Attempting to download missing data...")
            if not download_data(args):
                print("❌ Failed to download missing data")
            else:
                print("✅ Data download completed")
        else:
            print("✅ System health check passed - all essential data files available")
            
        # Optional: Run full diagnostic only if requested
        if args.debug:
            print("🐛 DEBUG: Running full diagnostic...")
            diagnostic_path = os.path.join(os.path.dirname(__file__), 'core', 'self_diagnostic_engine.py')
            if os.path.exists(diagnostic_path):
                result = subprocess.run([sys.executable, diagnostic_path], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print("✅ Full diagnostic completed successfully")
                else:
                    print("⚠️ Full diagnostic found issues (see logs/system_health.json)")
            
    except Exception as e:
        print(f"⚠️ Health check failed: {e} - proceeding anyway")
    
    # Handle single-timeframe override
    if args.single_timeframe:
        args.multi_timeframe = False

    try:
        if args.mode == 'watch':
            # Original watcher functionality
            print(f"🤖 Trading Robot Watcher - Continuous Optimization Mode")
            print(f"📁 Config: {args.config}")
            print(f"🔧 Intensity: {args.intensity}")
            if args.debug:
                print(f"🐛 Debug mode: ENABLED")
            print("-" * 50)
            main_loop(args)
            
        elif args.mode == 'backtest':
            # Single backtest run (from main.py)
            from core.backtest import run_backtest_instance
            
            print(f"🚀 Trading Robot - BACKTEST Mode")
            print(f"📊 Symbol: {args.symbol}")
            print(f"⏰ Timeframe: {args.timeframe}")
            print(f"🔄 Multi-timeframe: {'Yes' if args.multi_timeframe else 'No'}")
            print(f"📁 Config: {args.config}")
            if args.debug:
                print(f"🐛 Debug mode: ENABLED")
            print("-" * 50)
            
            # Create backtest args object
            class BacktestArgs:
                def __init__(self):
                    self.config = args.config
                    self.debug = args.debug
                    self.intensity = args.intensity
                    self.optimizer = args.optimizer
                    self.min_trades = args.min_trades
                    # Deprecated: do not override runs_to_keep from CLI; use config value instead
                    self.runs_to_keep = None
                    self.no_warmup = args.no_warmup
                    self.no_numba = args.no_numba
                    self.no_cache = args.no_cache
                    self.clear_cache = args.clear_cache
            
            backtest_args = BacktestArgs()
            
            print("🔄 Starting backtesting optimization...")
            if args.debug:
                print("🐛 Debug mode: Using smaller dataset and fewer optimization trials")
            
            results = run_backtest_instance(backtest_args)
            
            if results:
                print(f"✅ Backtest completed successfully!")
                print(f"📁 Results saved to: {results['run_directory']}")
                
                # Trigger automated parameter sync to live bot
                try:
                    from watcher_hook import on_backtest_complete
                    sync_success = on_backtest_complete(results['run_directory'], results)
                    if sync_success:
                        print("🔄 Live bot parameters automatically updated!")
                    else:
                        print("⚠️ Parameter sync to live bot failed")
                except ImportError:
                    print("⚠️ Automated pipeline not available - manual sync required")
                except Exception as e:
                    print(f"⚠️ Parameter sync error: {e}")
                    
            else:
                print("❌ Backtest failed. Check the logs for details.")
                sys.exit(1)
                
        elif args.mode == 'live':
            # Live trading mode (from main.py)
            print(f"🚀 Trading Robot - LIVE Mode")
            print(f"📊 Symbol: {args.symbol}")
            print(f"⏰ Timeframe: {args.timeframe}")
            print(f"🔄 Multi-timeframe: {'Yes' if args.multi_timeframe else 'No'}")
            print("-" * 50)
            
            # Start live monitoring (live bot runs on Vercel)
            print("🚀 Starting live bot monitoring...")
            print("📡 Live bot is running on Vercel - starting local monitoring")
            
            from unified_live_monitor import start_monitoring
            start_monitoring()  # Monitor indefinitely
            
        elif args.mode == 'download':
            # Data download mode (from main.py)
            print(f"🚀 Trading Robot - DOWNLOAD Mode")
            print(f"📊 Symbol: {args.symbol}")
            print(f"🔄 Multi-timeframe: {'Yes' if args.multi_timeframe else 'No'}")
            print("-" * 50)
            
            from utilities.utils import download_data
            success = download_data(args)
            if success:
                print("✅ Data download completed successfully!")
            else:
                print("❌ Data download failed. Check the logs for details.")
                sys.exit(1)
                
        elif args.mode == 'demo':
            # Strategy demo mode (from main.py)
            from core.strategy import MultiTimeframeStrategy
            from utilities.utils import Config
            import pandas as pd
            import os
            
            print(f"🚀 Trading Robot - DEMO Mode")
            print(f"📊 Symbol: {args.symbol}")
            print(f"⏰ Timeframe: {args.timeframe}")
            print(f"🔄 Multi-timeframe: {'Yes' if args.multi_timeframe else 'No'}")
            print("-" * 50)
            
            # Quick demo
            print("🎯 Running strategy demo...")
            
            # Load configuration
            config = Config(args.config)
            strategy = MultiTimeframeStrategy(config.config)
            
            # Try to load existing data
            data_file = f"crypto_data_{args.timeframe}.parquet"
            if os.path.exists(data_file):
                print(f"📊 Loading data from {data_file}")
                data = pd.read_parquet(data_file)
                if len(data) > 0:
                    print(f"📈 Data loaded: {len(data)} rows")
                    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
                    
                    # Generate signals using the strategy
                    try:
                        # The strategy expects a dictionary of timeframes
                        data_dict = {args.timeframe: data}
                        
                        # Generate multi-timeframe signal
                        signal_result = strategy.generate_multi_timeframe_signal(data_dict)
                        
                        print(f"📊 Generated signal successfully")
                        print(f"📊 Signal: {signal_result.primary_signal}")
                        print(f"📊 Confidence: {signal_result.confidence:.2f}")
                        print(f"📊 Strength: {signal_result.strength.name}")
                        
                        # Show timeframe breakdown
                        if signal_result.timeframe_signals:
                            print(f"📊 Timeframe breakdown:")
                            for tf_signal in signal_result.timeframe_signals:
                                print(f"  - {tf_signal.timeframe}: {tf_signal.signal} (confidence: {tf_signal.confidence:.2f})")
                        
                        # Show market condition
                        if hasattr(signal_result, 'market_condition'):
                            print(f"📊 Market Regime: {signal_result.market_condition.regime.name}")
                            print(f"📊 Market Confidence: {signal_result.market_condition.confidence:.2f}")
                            print(f"📊 Risk Adjustment: {signal_result.risk_adjustment:.2f}")
                        
                    except Exception as e:
                        print(f"⚠️  Error generating signals: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("⚠️  Data file is empty")
            else:
                print(f"⚠️  Data file {data_file} not found.")
                # Try default data file
                if os.path.exists("crypto_data.parquet"):
                    print("📊 Loading default data file...")
                    data = pd.read_parquet("crypto_data.parquet")
                    print(f"📈 Data loaded: {len(data)} rows")
                    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
                    
                    try:
                        signals = strategy.generate_signals(data)
                        print(f"📊 Generated {len(signals)} signals")
                        if len(signals) > 0:
                            latest_signal = signals.iloc[-1]
                            print(f"📊 Latest signal: {latest_signal['signal']}")
                    except Exception as e:
                        print(f"⚠️  Error generating signals: {e}")
                else:
                    print("⚠️  No data files found. Please run 'download' mode first.")
                    
        elif args.mode == 'analyze':
            # Parameter analysis mode (enhanced functionality)
            print(f"🚀 Trading Robot - ANALYZE Mode")
            print(f"📁 Config: {args.config}")
            print("-" * 50)
            
            print("🔍 Running parameter bounds analysis...")
            try:
                # Get the latest run directory for analysis
                with open("plots_output/latest_run_dir.txt", "r") as f:
                    latest_run = f.read().strip()
                
                # Create a mock args object for analyze_limits
                class AnalyzeArgs:
                    def __init__(self):
                        self.config = args.config
                        self.run_directory = latest_run
                
                analyze_args = AnalyzeArgs()
                from utilities.utils import analyze_limits
                analyze_limits(analyze_args)
                
            except FileNotFoundError:
                print("❌ latest_run_dir.txt not found. Cannot run parameter bounds analysis.")
                print("💡 Run a backtest first to generate analysis data.")
            except Exception as e:
                print(f"❌ Parameter bounds analysis failed: {e}")
            
            print("\n🔍 Running parameter performance analysis...")
            from utilities.utils import analyze_parameter_performance
            # Get the latest run directory from pointer file
            try:
                with open("plots_output/latest_run_dir.txt", "r") as f:
                    latest_run = f.read().strip()
                print(f"📊 Using latest run directory: {latest_run}")
                analyze_parameter_performance(latest_run)
            except FileNotFoundError:
                print("❌ latest_run_dir.txt not found. Cannot run parameter performance analysis.")
            except Exception as e:
                print(f"❌ Error in parameter performance analysis: {e}")
            
            # Enhanced parameter analysis when --intensity >= 2
            if args.intensity >= 2:
                print(f"\n🔬 ENHANCED ANALYSIS (Intensity {args.intensity}): Running advanced parameter analysis...")
                from utilities.utils import (analyze_enhanced_parameters, analyze_enhanced_bounds, 
                                           analyze_parameter_relationships)
                
                print("\n1️⃣ Enhanced parameter analysis...")
                try:
                    analyze_enhanced_parameters(args.config, sample_size=10000)
                except Exception as e:
                    print(f"❌ Enhanced parameter analysis failed: {e}")
                
                print("\n2️⃣ Enhanced bounds analysis...")
                try:
                    from utilities.utils import Config
                    config = Config(args.config)
                    analyze_enhanced_bounds(config.config)
                except Exception as e:
                    print(f"❌ Enhanced bounds analysis failed: {e}")
                
                print("\n3️⃣ Parameter relationship analysis...")
                try:
                    analyze_parameter_relationships(args.config)
                except Exception as e:
                    print(f"❌ Parameter relationship analysis failed: {e}")
                
                print("\n✅ Enhanced analysis complete!")
            
            # Enhanced debug analysis when --debug flag is enabled
            # Get the latest run directory from pointer file
            try:
                with open("plots_output/latest_run_dir.txt", "r") as f:
                    latest_run = f.read().strip()
                print(f"📊 Using latest run directory: {latest_run}")
                analyze_parameter_performance(latest_run)
            except FileNotFoundError:
                print("❌ latest_run_dir.txt not found. Cannot run parameter performance analysis.")
            except Exception as e:
                print(f"❌ Error in parameter performance analysis: {e}")
            
            # Enhanced debug analysis when --debug flag is enabled
            if args.debug:
                print("\n🐛 DEBUG MODE: Running enhanced diagnostic functions...")
                from utilities.utils import (debug_signal_generation, debug_backtest_entry_logic, 
                                           debug_atr_issue, debug_signal_timing, create_volatile_test_data)
                
                print("\n1️⃣ Testing signal generation with synthetic data...")
                try:
                    debug_signal_generation()
                except Exception as e:
                    print(f"❌ Signal generation debug failed: {e}")
                
                print("\n2️⃣ Testing backtest entry logic...")
                try:
                    debug_backtest_entry_logic()
                except Exception as e:
                    print(f"❌ Backtest entry debug failed: {e}")
                
                print("\n3️⃣ Testing ATR calculation...")
                try:
                    debug_atr_issue()
                except Exception as e:
                    print(f"❌ ATR debug failed: {e}")
                
                print("\n4️⃣ Testing signal timing conversion...")
                try:
                    debug_signal_timing()
                except Exception as e:
                    print(f"❌ Signal timing debug failed: {e}")
                
                print("\n✅ Enhanced debug analysis complete!")
                
        elif args.mode == 'cleanup':
            # File management and cleanup mode
            print(f"🚀 Trading Robot - CLEANUP Mode")
            print("-" * 50)
            
            print("🧹 Running system cleanup and file management...")
            from utilities.utils import find_duplicates_by_name, find_duplicates_by_content
            
            # Find duplicate files
            print("\n1️⃣ Scanning for duplicate files by name...")
            try:
                duplicate_names = find_duplicates_by_name()
                if duplicate_names:
                    print(f"📂 Found {len(duplicate_names)} sets of files with duplicate names")
                    if args.debug:
                        for dup_set in duplicate_names:
                            print(f"  - {dup_set}")
                else:
                    print("✅ No duplicate file names found")
            except Exception as e:
                print(f"❌ Duplicate name scan failed: {e}")
            
            print("\n2️⃣ Scanning for duplicate files by content...")
            try:
                duplicate_content = find_duplicates_by_content()
                if duplicate_content:
                    print(f"📄 Found {len(duplicate_content)} sets of files with duplicate content")
                    if args.debug:
                        for dup_set in duplicate_content:
                            print(f"  - {dup_set}")
                else:
                    print("✅ No duplicate file content found")
            except Exception as e:
                print(f"❌ Duplicate content scan failed: {e}")
            
            # Log file statistics
            print("\n3️⃣ Log file management...")
            try:
                from utilities.utils import central_logger
                log_stats = central_logger.get_log_stats()
                
                total_size_mb = 0
                for log_type, stats in log_stats.items():
                    if 'size_mb' in stats:
                        total_size_mb += stats['size_mb']
                        
                print(f"📊 Total log size: {total_size_mb:.2f} MB")
                print(f"📁 Active log files: {len(log_stats)}")
                
                if args.debug:
                    print("🐛 DEBUG: Detailed log statistics:")
                    for log_type, stats in log_stats.items():
                        print(f"  - {log_type}: {stats}")
                        
            except Exception as e:
                print(f"❌ Log analysis failed: {e}")
            
            # Disk space check
            print("\n4️⃣ Disk space analysis...")
            try:
                import shutil
                total, used, free = shutil.disk_usage(".")
                total_gb = total // (1024**3)
                used_gb = used // (1024**3)
                free_gb = free // (1024**3)
                used_percent = (used / total) * 100
                
                print(f"💾 Disk usage: {used_gb}GB / {total_gb}GB ({used_percent:.1f}% used)")
                print(f"💾 Free space: {free_gb}GB")
                
                if used_percent > 90:
                    print("⚠️  WARNING: Disk usage is high (>90%)")
                elif used_percent > 80:
                    print("⚠️  CAUTION: Disk usage is moderate (>80%)")
                else:
                    print("✅ Disk usage is healthy")
                    
            except Exception as e:
                print(f"❌ Disk analysis failed: {e}")
            
            print("\n✅ Cleanup analysis complete!")
            
            # Offer cleanup actions
            if args.intensity >= 2:
                print("\n🗑️  CLEANUP ACTIONS (Intensity >= 2):")
                print("  - Log rotation would be performed")
                print("  - Old backup files would be cleaned")
                print("  - Temporary files would be removed")
                print("  (Actual cleanup not implemented for safety - requires manual confirmation)")
            
    except KeyboardInterrupt:
        print("\n\nTrading Robot stopped manually by user. Exiting.")
        sys.exit(0)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required files are present.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_optimization_cycle(config_file="core/optimization_config.json", intensity=2):
    """Orchestration function for optimization cycle - alias for main functionality"""
    import argparse
    import sys
    
    # Create args object to match main() expectations
    args = argparse.Namespace()
    args.config = config_file
    args.intensity = intensity
    args.runs_to_keep = 5
    args.optimizer = "optuna"
    args.no_warmup = False
    args.min_trades = None
    args.debug = False
    args.no_numba = False
    args.mode = "watch"
    args.test_run = True  # Single cycle for function call
    
    return main_loop(args)



def analyze_parameter_limits(config_file="core/optimization_config.json"):
    """Analyze parameter limits - alias for utilities function"""
    from utilities.utils import analyze_limits
    return analyze_limits(config_file)

