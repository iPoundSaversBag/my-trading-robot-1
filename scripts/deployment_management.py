#!/usr/bin/env python3
"""
Deployment Management Scripts - Consolidated Deployment Operations
Combines deployment-related script functionality into a unified module.

Consolidated from:
- prepare_cloud_deployment.py (Cloud deployment preparation)
- log_rotation_manager.py (Log management and rotation)

Purpose: Unified deployment and maintenance operations including cloud
preparation, log management, and system monitoring capabilities.
"""

import argparse
import sys
import os
import shutil
import json
import datetime
import time
import glob
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for utilities import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utilities.utils import central_logger
except ImportError:
    import logging
    central_logger = logging.getLogger(__name__)

# ============================================================================
# CLOUD DEPLOYMENT PREPARATION
# ============================================================================

class CloudDeploymentManager:
    """Manage cloud deployment preparation and configuration."""
    
    def __init__(self, deploy_dir: str = "deploy"):
        self.deploy_dir = Path(deploy_dir)
        self.api_endpoints = {
            'live-data': '/api/live-data',
            'dashboard-data': '/api/dashboard-data',
            'portfolio': '/api/portfolio',
            'tearsheet': '/api/tearsheet'
        }
    
    def prepare_cloud_deployment(self) -> bool:
        """Prepare complete dashboard for cloud deployment."""
        print("🚀 Preparing Trading Dashboard for Cloud Deployment")
        print("=" * 60)
        
        try:
            # Clean and create deployment directory
            if self.deploy_dir.exists():
                shutil.rmtree(self.deploy_dir)
            self.deploy_dir.mkdir()
            
            print("📁 Creating deployment structure...")
            
            # Find and copy latest dashboard
            dashboard_copied = self._copy_dashboard()
            
            # Copy API files
            api_copied = self._copy_api_files()
            
            # Copy configuration files
            config_copied = self._copy_configuration_files()
            
            # Create deployment metadata
            self._create_deployment_metadata()
            
            # Generate deployment README
            self._generate_deployment_readme()
            
            success = dashboard_copied and api_copied and config_copied
            
            if success:
                print("=" * 60)
                print("✅ Cloud deployment preparation completed successfully!")
                print(f"📦 Deployment package ready in: {self.deploy_dir}")
                print("🚀 Ready for upload to Vercel, Netlify, or similar platforms")
            else:
                print("=" * 60)
                print("❌ Cloud deployment preparation failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Deployment preparation failed: {e}")
            central_logger.error(f"Cloud deployment preparation failed: {e}")
            return False
    
    def _find_latest_dashboard(self) -> Optional[Path]:
        """Find the latest performance report dashboard."""
        # Look for latest run directory
        plot_dirs = glob.glob('plots_output/20*')
        if not plot_dirs:
            return None
        
        latest_dir = max(plot_dirs, key=os.path.getctime)
        dashboard_path = Path(latest_dir) / 'performance_report.html'
        
        return dashboard_path if dashboard_path.exists() else None
    
    def _copy_dashboard(self) -> bool:
        """Copy dashboard files to deployment directory."""
        dashboard_source = self._find_latest_dashboard()
        
        if not dashboard_source:
            print("❌ No dashboard found to deploy")
            return False
        
        dashboard_dest = self.deploy_dir / "index.html"
        
        try:
            shutil.copy2(dashboard_source, dashboard_dest)
            print(f"✅ Copied dashboard: {dashboard_source} -> {dashboard_dest}")
            
            # Update API endpoints for cloud hosting
            self._update_api_endpoints(dashboard_dest)
            return True
            
        except Exception as e:
            print(f"❌ Failed to copy dashboard: {e}")
            return False
    
    def _copy_api_files(self) -> bool:
        """Copy API files to deployment directory."""
        api_dir = self.deploy_dir / "api"
        api_dir.mkdir(exist_ok=True)
        
        # Define API files to copy
        api_files = [
            ("api/live-data.py", "live-data.py"),
            ("api/dashboard-data.py", "dashboard-data.py"),
            ("api/portfolio.py", "portfolio.py"),
            ("api/tearsheet.py", "tearsheet.py"),
            ("api/utilities.py", "utilities.py")
        ]
        
        copied_count = 0
        for source_file, dest_file in api_files:
            source_path = Path(source_file)
            dest_path = api_dir / dest_file
            
            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ Copied API: {source_file}")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Failed to copy {source_file}: {e}")
            else:
                print(f"⚠️ API file not found: {source_file}")
        
        print(f"📡 Copied {copied_count} API endpoints")
        return copied_count > 0
    
    def _copy_configuration_files(self) -> bool:
        """Copy configuration files to deployment directory."""
        config_files = [
            ("vercel.json", "vercel.json"),
            ("package.json", "package.json"),
            ("requirements.txt", "requirements.txt")
        ]
        
        copied_count = 0
        for source_file, dest_file in config_files:
            source_path = Path(source_file)
            dest_path = self.deploy_dir / dest_file
            
            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ Copied config: {source_file}")
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Failed to copy {source_file}: {e}")
            else:
                print(f"⚠️ Config file not found: {source_file}")
        
        return copied_count > 0
    
    def _update_api_endpoints(self, dashboard_file: Path) -> None:
        """Update API endpoints in dashboard for cloud hosting."""
        try:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace local API endpoints with cloud endpoints
            replacements = {
                'http://localhost:8000/api/': '/api/',
                'localhost:8000': '',
                '127.0.0.1:8000': ''
            }
            
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated API endpoints for cloud hosting")
            
        except Exception as e:
            print(f"❌ Failed to update API endpoints: {e}")
    
    def _create_deployment_metadata(self) -> None:
        """Create deployment metadata file."""
        metadata = {
            "deployment_timestamp": datetime.datetime.now().isoformat(),
            "deployment_version": "1.0.0",
            "dashboard_source": str(self._find_latest_dashboard()),
            "api_endpoints": self.api_endpoints,
            "deployment_notes": "Automated cloud deployment package"
        }
        
        metadata_file = self.deploy_dir / "deployment.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Created deployment metadata")
    
    def _generate_deployment_readme(self) -> None:
        """Generate deployment README file."""
        readme_content = f"""# Trading Dashboard Cloud Deployment

## 📋 Deployment Information
- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Version**: 1.0.0
- **Status**: Ready for deployment

## 🚀 Deployment Instructions

### Vercel Deployment
1. Install Vercel CLI: `npm install -g vercel`
2. Deploy: `vercel --prod`
3. Configure environment variables if needed

### Netlify Deployment
1. Drag and drop this folder to Netlify dashboard
2. Configure build settings: Build command: (none), Publish directory: ./

### Manual Deployment
1. Upload all files to your web hosting provider
2. Ensure Python runtime is available for API endpoints
3. Set up proper routing for `/api/*` endpoints

## 📁 Package Contents
- `index.html` - Main dashboard (enhanced performance report)
- `api/` - Serverless API endpoints
- `vercel.json` - Vercel configuration
- `package.json` - Node.js dependencies
- `requirements.txt` - Python dependencies

## 🔧 API Endpoints
{chr(10).join([f"- `{endpoint}` - {description}" for endpoint, description in self.api_endpoints.items()])}

## 📊 Features
- Real-time trading performance dashboard
- Live data integration
- Portfolio analytics
- Performance tearsheets
- Responsive design for mobile/desktop

## 🎯 Next Steps
1. Deploy to your preferred platform
2. Configure any required environment variables
3. Test all API endpoints
4. Monitor deployment logs

---
Generated by Trading Robot Cloud Deployment Manager
"""
        
        readme_file = self.deploy_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print("✅ Generated deployment README")

# ============================================================================
# LOG ROTATION MANAGER
# ============================================================================

class LogRotationManager:
    """Advanced log rotation and management system for trading robot pipeline."""
    
    def __init__(self):
        self.logger = central_logger
        self.start_time = datetime.datetime.now()
        self.log_directories = ['logs', 'live_trading', '.']
        self.log_patterns = ['*.log', '*.txt', 'notifications.*']
        self.max_age_days = 30
        self.max_size_mb = 100
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log file statistics."""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'files_by_type': {},
            'files_by_directory': {},
            'old_files': [],
            'large_files': [],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        for directory in self.log_directories:
            if not os.path.exists(directory):
                continue
            
            dir_stats = {'count': 0, 'size_mb': 0}
            
            for pattern in self.log_patterns:
                log_files = glob.glob(os.path.join(directory, pattern))
                
                for log_file in log_files:
                    try:
                        file_stat = os.stat(log_file)
                        file_size_mb = file_stat.st_size / (1024 * 1024)
                        file_age_days = (time.time() - file_stat.st_mtime) / (24 * 3600)
                        
                        stats['total_files'] += 1
                        stats['total_size_mb'] += file_size_mb
                        dir_stats['count'] += 1
                        dir_stats['size_mb'] += file_size_mb
                        
                        # Track file types
                        file_ext = os.path.splitext(log_file)[1]
                        stats['files_by_type'][file_ext] = stats['files_by_type'].get(file_ext, 0) + 1
                        
                        # Track old files
                        if file_age_days > self.max_age_days:
                            stats['old_files'].append({
                                'file': log_file,
                                'age_days': round(file_age_days, 1),
                                'size_mb': round(file_size_mb, 2)
                            })
                        
                        # Track large files
                        if file_size_mb > self.max_size_mb:
                            stats['large_files'].append({
                                'file': log_file,
                                'size_mb': round(file_size_mb, 2),
                                'age_days': round(file_age_days, 1)
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Could not stat file {log_file}: {e}")
            
            stats['files_by_directory'][directory] = dir_stats
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats
    
    def rotate_log_file(self, log_file: str) -> bool:
        """Rotate a specific log file."""
        try:
            if not os.path.exists(log_file):
                return False
            
            # Create rotated filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(log_file)[0]
            extension = os.path.splitext(log_file)[1]
            rotated_name = f"{base_name}_{timestamp}{extension}.gz"
            
            # Compress and move the file
            with open(log_file, 'rb') as f_in:
                with gzip.open(rotated_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Create new empty log file
            open(log_file, 'w').close()
            
            self.logger.info(f"Rotated log file: {log_file} -> {rotated_name}")
            print(f"✅ Rotated: {log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate {log_file}: {e}")
            print(f"❌ Failed to rotate {log_file}: {e}")
            return False
    
    def rotate_all_logs(self) -> Dict[str, Any]:
        """Rotate all log files."""
        print("🔄 Rotating all log files...")
        
        rotated_files = []
        failed_files = []
        
        for directory in self.log_directories:
            if not os.path.exists(directory):
                continue
            
            for pattern in self.log_patterns:
                log_files = glob.glob(os.path.join(directory, pattern))
                
                for log_file in log_files:
                    # Skip already rotated files
                    if '.gz' in log_file or '_20' in log_file:
                        continue
                    
                    if self.rotate_log_file(log_file):
                        rotated_files.append(log_file)
                    else:
                        failed_files.append(log_file)
        
        result = {
            'rotated_count': len(rotated_files),
            'failed_count': len(failed_files),
            'rotated_files': rotated_files,
            'failed_files': failed_files,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        print(f"✅ Rotation complete: {len(rotated_files)} rotated, {len(failed_files)} failed")
        return result
    
    def cleanup_old_logs(self) -> Dict[str, Any]:
        """Clean up old log files."""
        print("🧹 Cleaning up old log files...")
        
        stats = self.get_log_statistics()
        removed_files = []
        failed_removals = []
        
        for old_file in stats['old_files']:
            try:
                os.remove(old_file['file'])
                removed_files.append(old_file)
                print(f"🗑️ Removed old file: {old_file['file']}")
            except Exception as e:
                failed_removals.append({'file': old_file['file'], 'error': str(e)})
                print(f"❌ Failed to remove {old_file['file']}: {e}")
        
        result = {
            'removed_count': len(removed_files),
            'failed_count': len(failed_removals),
            'removed_files': removed_files,
            'failed_removals': failed_removals,
            'space_freed_mb': sum(f['size_mb'] for f in removed_files),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        print(f"✅ Cleanup complete: {len(removed_files)} files removed, {result['space_freed_mb']:.2f} MB freed")
        return result
    
    def emergency_archive(self) -> bool:
        """Emergency archival of all logs."""
        print("🆘 Emergency archive mode activated...")
        
        try:
            # Create emergency archive directory
            archive_dir = f"emergency_archive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(archive_dir, exist_ok=True)
            
            archived_count = 0
            
            for directory in self.log_directories:
                if not os.path.exists(directory):
                    continue
                
                for pattern in self.log_patterns:
                    log_files = glob.glob(os.path.join(directory, pattern))
                    
                    for log_file in log_files:
                        try:
                            archive_path = os.path.join(archive_dir, os.path.basename(log_file))
                            shutil.copy2(log_file, archive_path)
                            archived_count += 1
                        except Exception as e:
                            print(f"❌ Failed to archive {log_file}: {e}")
            
            print(f"✅ Emergency archive complete: {archived_count} files archived to {archive_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Emergency archive failed: {e}")
            return False
    
    def monitor_logs(self, duration_minutes: int = 60) -> None:
        """Continuous log monitoring for specified duration."""
        print(f"👁️ Starting log monitoring for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            stats = self.get_log_statistics()
            
            print(f"\n📊 Log Status: {stats['total_files']} files, {stats['total_size_mb']} MB")
            
            if stats['large_files']:
                print(f"⚠️ Large files detected: {len(stats['large_files'])}")
            
            if stats['old_files']:
                print(f"🗓️ Old files detected: {len(stats['old_files'])}")
            
            time.sleep(300)  # Check every 5 minutes
        
        print("✅ Log monitoring session completed")

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deployment Management Scripts')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Cloud deployment commands
    deploy_parser = subparsers.add_parser('deploy', help='Cloud deployment operations')
    deploy_parser.add_argument('--prepare', action='store_true', help='Prepare cloud deployment')
    deploy_parser.add_argument('--dir', default='deploy', help='Deployment directory')
    
    # Log management commands
    log_parser = subparsers.add_parser('logs', help='Log management operations')
    log_parser.add_argument('--stats', action='store_true', help='Show log statistics')
    log_parser.add_argument('--rotate', type=str, help='Rotate specific log file')
    log_parser.add_argument('--rotate-all', action='store_true', help='Rotate all logs')
    log_parser.add_argument('--cleanup', action='store_true', help='Clean up old logs')
    log_parser.add_argument('--emergency-archive', action='store_true', help='Emergency archive')
    log_parser.add_argument('--monitor', type=int, help='Monitor logs for N minutes')
    
    args = parser.parse_args()
    
    if args.command == 'deploy':
        manager = CloudDeploymentManager(args.dir)
        if args.prepare:
            manager.prepare_cloud_deployment()
        else:
            deploy_parser.print_help()
    
    elif args.command == 'logs':
        manager = LogRotationManager()
        
        if args.stats:
            stats = manager.get_log_statistics()
            print("📊 LOG STATISTICS")
            print("=" * 40)
            print(f"Total files: {stats['total_files']}")
            print(f"Total size: {stats['total_size_mb']} MB")
            print(f"Old files: {len(stats['old_files'])}")
            print(f"Large files: {len(stats['large_files'])}")
            
        elif args.rotate:
            manager.rotate_log_file(args.rotate)
            
        elif args.rotate_all:
            manager.rotate_all_logs()
            
        elif args.cleanup:
            manager.cleanup_old_logs()
            
        elif args.emergency_archive:
            manager.emergency_archive()
            
        elif args.monitor:
            manager.monitor_logs(args.monitor)
            
        else:
            log_parser.print_help()
    
    else:
        parser.print_help()
