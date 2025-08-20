#!/usr/bin/env python3
"""
Watcher Integration for Unified Live Bot Monitor
Simplified hook that uses the unified monitoring system
"""

import json
import sys
import os
from pathlib import Path

class WatcherHook:
    def __init__(self):
        self.base_dir = Path(__file__).parent
    
    def trigger_parameter_sync(self, run_directory=None):
        """Trigger parameter sync after successful backtest using unified monitor"""
        try:
            print("\n🔄 TRIGGERING AUTOMATED PARAMETER SYNC")
            print("=" * 60)
            
            if run_directory:
                print(f"📁 Backtest completed in: {run_directory}")
            
            # Use unified monitor for parameter sync
            from unified_live_monitor import on_backtest_complete
            
            success = on_backtest_complete(run_directory)
            
            if success:
                print("✅ Parameter sync completed successfully")
                print("🚀 Live bot will use new optimized parameters")
                return True
            else:
                print("❌ Parameter sync failed")
                return False
                
        except Exception as e:
            print(f"❌ Error triggering parameter sync: {e}")
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring using unified monitor"""
        try:
            print("\n📊 STARTING CONTINUOUS LIVE MONITORING")
            print("=" * 60)
            
            # Use unified monitor for continuous monitoring
            from unified_live_monitor import start_monitoring
            
            print("✅ Starting live monitoring in background")
            print("📈 Results will be tracked automatically")
            
            # Start 60-minute monitoring session (or until manually stopped)
            start_monitoring(60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
            return False

# Integration point for watcher.py
def on_backtest_complete(run_directory=None):
    """Called by watcher.py when backtest completes - simplified interface"""
    try:
        from unified_live_monitor import on_backtest_complete as unified_hook
        return unified_hook(run_directory)
    except Exception as e:
        print(f"❌ Watcher hook error: {e}")
        return False

if __name__ == "__main__":
    # Manual testing
    import sys
    
    if "--sync" in sys.argv:
        hook = WatcherHook()
        hook.trigger_parameter_sync()
    elif "--monitor" in sys.argv:
        from unified_live_monitor import start_monitoring
        start_monitoring()
    else:
        print("Usage: python watcher_hook.py [--sync|--monitor]")
