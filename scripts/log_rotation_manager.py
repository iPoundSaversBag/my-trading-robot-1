#!/usr/bin/env python3
"""
Log Rotation Manager - Phase 6H Comprehensive Log Management

This script provides manual control and monitoring for the comprehensive log rotation system.
It integrates with the enhanced CentralizedLogger to provide advanced log management capabilities.

FEATURES:
- Manual log rotation triggers
- Log statistics and health monitoring  
- Automated cleanup scheduling
- Emergency log archival
- Disk space monitoring
- Integration with watcher pipeline

USAGE:
    python scripts/log_rotation_manager.py --stats                    # Show log statistics
    python scripts/log_rotation_manager.py --rotate watcher           # Rotate specific log
    python scripts/log_rotation_manager.py --rotate-all               # Rotate all logs
    python scripts/log_rotation_manager.py --cleanup                  # Run manual cleanup
    python scripts/log_rotation_manager.py --emergency-archive        # Emergency archive
    python scripts/log_rotation_manager.py --monitor                  # Continuous monitoring

INTEGRATION:
- Called from watcher.py during pipeline execution
- Used by system_monitor.py for automated maintenance
- Provides API for live_bot.py log management
"""

import argparse
import sys
import os
import datetime
import json
import time

# Add parent directory to path for utilities import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.utils import central_logger

class LogRotationManager:
    """
    Advanced log rotation and management system for trading robot pipeline
    """
    
    def __init__(self):
        self.logger = central_logger
        self.start_time = datetime.datetime.now()
        
    def show_log_statistics(self):
        """Display comprehensive log statistics"""
        print("=" * 80)
        print(f"LOG ROTATION SYSTEM STATUS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        stats = self.logger.get_log_stats()
        
        total_size_mb = 0
        needs_rotation_count = 0
        
        print(f"{'Log Type':<20} {'File Path':<30} {'Size (MB)':<10} {'Max (MB)':<10} {'Status':<15} {'Last Modified'}")
        print("-" * 100)
        
        for log_type, info in stats.items():
            if 'exists' in info and not info['exists']:
                print(f"{log_type:<20} {info['file_path']:<30} {'N/A':<10} {'N/A':<10} {'Not Found':<15} {'N/A'}")
                continue
                
            size_mb = info['size_mb']
            max_mb = info.get('max_size_mb', 'N/A')
            needs_rotation = info.get('needs_rotation', False)
            last_modified = info.get('last_modified', 'Unknown')
            
            total_size_mb += size_mb
            if needs_rotation:
                needs_rotation_count += 1
                
            status = "NEEDS ROTATION" if needs_rotation else "OK"
            status_color = status
            
            print(f"{log_type:<20} {info['file_path']:<30} {size_mb:<10.2f} {max_mb:<10} {status_color:<15} {last_modified}")
        
        print("-" * 100)
        print(f"SUMMARY:")
        print(f"  Total log size: {total_size_mb:.2f} MB")
        print(f"  Logs needing rotation: {needs_rotation_count}")
        print(f"  Rotation system: {'ENABLED' if self.logger.enable_rotation else 'DISABLED'}")
        
        # Check backup directory
        backup_dir = "backups/logs/phase6h_comprehensive_rotation"
        if os.path.exists(backup_dir):
            backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.log')]
            backup_count = len(backup_files)
            print(f"  Backup files: {backup_count}")
        else:
            print(f"  Backup directory: Not found")
            
        # Disk space check
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            print(f"  Free disk space: {free_gb:.2f} GB")
            
            if free_gb < 1.0:
                print(f"  ⚠️  WARNING: Low disk space!")
        except:
            print(f"  Free disk space: Unable to determine")
            
        print("=" * 80)
    
    def rotate_specific_log(self, log_type):
        """Manually rotate a specific log type"""
        print(f"🔄 Manually rotating log: {log_type}")
        
        if self.logger.rotate_log_manually(log_type):
            print(f"✅ Successfully rotated {log_type} log")
            return True
        else:
            print(f"❌ Failed to rotate {log_type} log")
            return False
    
    def rotate_all_logs(self):
        """Rotate all logs that need rotation"""
        print("🔄 Rotating all logs that need rotation...")
        
        stats = self.logger.get_log_stats()
        rotated_count = 0
        
        for log_type, info in stats.items():
            if info.get('needs_rotation', False):
                print(f"  Rotating {log_type}...")
                if self.logger.rotate_log_manually(log_type):
                    rotated_count += 1
                    print(f"    ✅ {log_type} rotated successfully")
                else:
                    print(f"    ❌ {log_type} rotation failed")
        
        print(f"🔄 Rotation complete. {rotated_count} logs rotated.")
        return rotated_count
    
    def run_manual_cleanup(self):
        """Run manual cleanup and maintenance"""
        print("🧹 Running manual cleanup and maintenance...")
        
        try:
            self.logger._automated_cleanup()
            print("✅ Manual cleanup completed successfully")
            return True
        except Exception as e:
            print(f"❌ Manual cleanup failed: {e}")
            return False
    
    def emergency_archive(self):
        """Emergency archive of all current logs"""
        print("🚨 Running emergency archive of all logs...")
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_dir = f"backups/logs/emergency_archive_{timestamp}"
            os.makedirs(emergency_dir, exist_ok=True)
            
            archived_count = 0
            stats = self.logger.get_log_stats()
            
            for log_type, info in stats.items():
                if 'exists' in info and not info['exists']:
                    continue
                    
                log_file = info['file_path']
                if os.path.exists(log_file):
                    import shutil
                    backup_filename = f"{log_type}_{timestamp}.log"
                    backup_path = os.path.join(emergency_dir, backup_filename)
                    shutil.copy2(log_file, backup_path)
                    archived_count += 1
                    print(f"  📦 Archived {log_file} to {backup_path}")
            
            print(f"🚨 Emergency archive complete. {archived_count} files archived to {emergency_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Emergency archive failed: {e}")
            return False
    
    def continuous_monitor(self, check_interval=300):
        """Run continuous monitoring mode"""
        print(f"👁️  Starting continuous monitoring (checking every {check_interval} seconds)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                print(f"\n--- Monitor Check at {datetime.datetime.now().strftime('%H:%M:%S')} ---")
                
                # Check for logs needing rotation
                stats = self.logger.get_log_stats()
                needs_rotation = [log_type for log_type, info in stats.items() 
                                if info.get('needs_rotation', False)]
                
                if needs_rotation:
                    print(f"⚠️  Logs needing rotation: {', '.join(needs_rotation)}")
                    
                    # Auto-rotate if enabled
                    if self.logger.enable_rotation:
                        print("🔄 Auto-rotating logs...")
                        self.rotate_all_logs()
                else:
                    print("✅ All logs within size limits")
                
                # Check disk space
                try:
                    import shutil
                    total, used, free = shutil.disk_usage(".")
                    free_gb = free / (1024**3)
                    
                    if free_gb < 1.0:
                        print(f"⚠️  WARNING: Low disk space - {free_gb:.2f}GB remaining")
                except:
                    pass
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n👁️  Monitoring stopped by user")
    
    def integration_test(self):
        """Test integration with pipeline components"""
        print("🧪 Running integration tests...")
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Logger initialization
        tests_total += 1
        if self.logger.enable_rotation:
            print("✅ Test 1: Logger rotation enabled")
            tests_passed += 1
        else:
            print("❌ Test 1: Logger rotation disabled")
        
        # Test 2: Backup directory exists
        tests_total += 1
        backup_dir = "backups/logs/phase6h_comprehensive_rotation"
        if os.path.exists(backup_dir):
            print("✅ Test 2: Backup directory exists")
            tests_passed += 1
        else:
            print("❌ Test 2: Backup directory missing")
        
        # Test 3: Log directory structure
        tests_total += 1
        if os.path.exists("logs"):
            print("✅ Test 3: Logs directory exists")
            tests_passed += 1
        else:
            print("❌ Test 3: Logs directory missing")
        
        # Test 4: Test logging functionality
        tests_total += 1
        try:
            test_message = f"Integration test at {datetime.datetime.now()}"
            self.logger.log_message(test_message, 'system_monitor')
            print("✅ Test 4: Logging functionality works")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 4: Logging failed - {e}")
        
        print(f"\n🧪 Integration tests complete: {tests_passed}/{tests_total} passed")
        return tests_passed == tests_total

def main():
    """Main entry point for log rotation manager"""
    parser = argparse.ArgumentParser(description="Trading Robot Log Rotation Manager")
    
    parser.add_argument('--stats', action='store_true', 
                       help='Show log statistics')
    parser.add_argument('--rotate', type=str, 
                       help='Rotate specific log type (e.g., watcher, live_bot)')
    parser.add_argument('--rotate-all', action='store_true',
                       help='Rotate all logs that need rotation')
    parser.add_argument('--cleanup', action='store_true',
                       help='Run manual cleanup')
    parser.add_argument('--emergency-archive', action='store_true',
                       help='Emergency archive all logs')
    parser.add_argument('--monitor', action='store_true',
                       help='Run continuous monitoring')
    parser.add_argument('--test', action='store_true',
                       help='Run integration tests')
    
    args = parser.parse_args()
    
    manager = LogRotationManager()
    
    # Show stats by default if no other action specified
    if not any([args.rotate, args.rotate_all, args.cleanup, 
               args.emergency_archive, args.monitor, args.test]):
        args.stats = True
    
    if args.stats:
        manager.show_log_statistics()
    
    if args.rotate:
        manager.rotate_specific_log(args.rotate)
    
    if args.rotate_all:
        manager.rotate_all_logs()
    
    if args.cleanup:
        manager.run_manual_cleanup()
    
    if args.emergency_archive:
        manager.emergency_archive()
    
    if args.monitor:
        manager.continuous_monitor()
    
    if args.test:
        success = manager.integration_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
