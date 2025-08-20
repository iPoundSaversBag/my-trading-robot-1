#!/usr/bin/env python3
"""
Data Management Scripts - Consolidated Data Operations
Combines data-related script functionality into a unified module.

Consolidated from:
- data_range_summary.py (Data range analysis)
- data_backfill.py (Data backfill operations)
- preflight_check.py (Empty file - implemented comprehensive checks)

Purpose: Unified data management operations including range analysis,
backfill operations, and comprehensive data health checks.
"""

import os
import sys
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add parent directory to path for utilities import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utilities.utils import central_logger
except ImportError:
    import logging
    central_logger = logging.getLogger(__name__)

# ============================================================================
# DATA RANGE ANALYSIS
# ============================================================================

class DataRangeAnalyzer:
    """Analyze data ranges and coverage for trading datasets."""
    
    DEFAULT_FILES = [
        'data/crypto_data_5m.parquet',
        'data/crypto_data_15m.parquet',
        'data/crypto_data_1h.parquet',
        'data/crypto_data_4h.parquet'
    ]
    
    @staticmethod
    def summarize_file(path: str) -> Dict[str, Any]:
        """Analyze a single data file and return comprehensive summary."""
        if not os.path.exists(path):
            return {'file': path, 'status': 'MISSING'}
        
        try:
            # Try to read the file
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                return {'file': path, 'status': 'unsupported_format'}
                
        except Exception as e:
            return {'file': path, 'status': f'read_failed: {e}'}
        
        # Try to ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ['timestamp', 'date', 'datetime', 'time']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                    except Exception:
                        continue
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'file': path, 'status': 'no_datetime_index'}
        
        # Calculate statistics
        start = df.index.min()
        end = df.index.max()
        span_days = (end - start).days
        years = span_days / 365.25
        
        # Check for gaps
        expected_freq = DataRangeAnalyzer._detect_frequency(df)
        gaps = DataRangeAnalyzer._detect_gaps(df, expected_freq)
        
        return {
            'file': path,
            'status': 'ok',
            'rows': len(df),
            'columns': list(df.columns),
            'start': start,
            'end': end,
            'span_days': span_days,
            'approx_years': round(years, 2),
            'expected_frequency': expected_freq,
            'gaps_detected': len(gaps),
            'gaps': gaps[:5] if gaps else [],  # Show first 5 gaps
            'data_quality': 'good' if len(gaps) < 10 else 'needs_attention'
        }
    
    @staticmethod
    def _detect_frequency(df: pd.DataFrame) -> str:
        """Detect the expected frequency of data."""
        if len(df) < 2:
            return 'unknown'
        
        # Calculate time differences
        diffs = df.index.to_series().diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= pd.Timedelta(minutes=1):
            return '1m'
        elif median_diff <= pd.Timedelta(minutes=5):
            return '5m'
        elif median_diff <= pd.Timedelta(minutes=15):
            return '15m'
        elif median_diff <= pd.Timedelta(hours=1):
            return '1h'
        elif median_diff <= pd.Timedelta(hours=4):
            return '4h'
        elif median_diff <= pd.Timedelta(days=1):
            return '1d'
        else:
            return 'custom'
    
    @staticmethod
    def _detect_gaps(df: pd.DataFrame, frequency: str) -> List[Dict[str, Any]]:
        """Detect gaps in time series data."""
        if frequency == 'unknown' or len(df) < 2:
            return []
        
        # Define expected timedelta
        freq_map = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        
        expected_delta = freq_map.get(frequency)
        if not expected_delta:
            return []
        
        # Find gaps
        diffs = df.index.to_series().diff().dropna()
        gaps = []
        
        for i, diff in enumerate(diffs):
            if diff > expected_delta * 1.5:  # Allow 50% tolerance
                gap_start = df.index[i-1]
                gap_end = df.index[i]
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': diff,
                    'missing_periods': int(diff / expected_delta) - 1
                })
        
        return gaps
    
    @staticmethod
    def analyze_all_files(file_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze all data files and generate comprehensive report."""
        files = file_list or DataRangeAnalyzer.DEFAULT_FILES
        
        print('=' * 60)
        print('📊 DATA RANGE ANALYSIS REPORT')
        print('=' * 60)
        
        results = []
        for file_path in files:
            result = DataRangeAnalyzer.summarize_file(file_path)
            results.append(result)
            
            # Print summary
            if result['status'] != 'ok':
                print(f"❌ {result['file']}: {result['status']}")
            else:
                print(f"✅ {result['file']}:")
                print(f"   📈 Rows: {result['rows']:,}")
                print(f"   📅 Range: {result['start']} to {result['end']}")
                print(f"   ⏱️  Span: {result['span_days']} days (~{result['approx_years']} years)")
                print(f"   🔍 Frequency: {result['expected_frequency']}")
                print(f"   🕳️  Gaps: {result['gaps_detected']} detected")
                print(f"   ✨ Quality: {result['data_quality']}")
                print()
        
        # Overall statistics
        successful_files = [r for r in results if r['status'] == 'ok']
        total_rows = sum(r['rows'] for r in successful_files)
        
        print('=' * 60)
        print('📋 SUMMARY')
        print('=' * 60)
        print(f"✅ Files processed: {len(results)}")
        print(f"🎯 Successful: {len(successful_files)}")
        print(f"📊 Total rows: {total_rows:,}")
        print(f"📈 Files with gaps: {sum(1 for r in successful_files if r['gaps_detected'] > 0)}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'files_analyzed': len(results),
            'successful_files': len(successful_files),
            'total_rows': total_rows,
            'results': results
        }

# ============================================================================
# DATA BACKFILL OPERATIONS
# ============================================================================

class DataBackfiller:
    """Handle data backfill operations for missing time periods."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3/klines"
    
    def fetch_data(self, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List[List]:
        """Fetch historical data from Binance API."""
        params = {
            'symbol': self.symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            central_logger.error(f"Failed to fetch data: {e}")
            return []
    
    def convert_to_dataframe(self, raw_data: List[List]) -> pd.DataFrame:
        """Convert raw Binance data to formatted DataFrame."""
        if not raw_data:
            return pd.DataFrame()
        
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        
        df = pd.DataFrame(raw_data, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[numeric_columns]  # Return only essential columns
    
    def backfill_gaps(self, file_path: str, gaps: List[Dict[str, Any]], interval: str) -> bool:
        """Backfill detected gaps in data file."""
        if not gaps:
            print(f"✅ No gaps to backfill in {file_path}")
            return True
        
        try:
            # Load existing data
            if file_path.endswith('.parquet'):
                existing_df = pd.read_parquet(file_path)
            else:
                existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            new_data_frames = []
            
            for gap in gaps:
                print(f"🔄 Backfilling gap: {gap['start']} to {gap['end']}")
                
                start_ms = int(gap['start'].timestamp() * 1000)
                end_ms = int(gap['end'].timestamp() * 1000)
                
                # Fetch data for gap
                raw_data = self.fetch_data(interval, start_ms, end_ms)
                if raw_data:
                    gap_df = self.convert_to_dataframe(raw_data)
                    new_data_frames.append(gap_df)
                    print(f"✅ Fetched {len(gap_df)} records for gap")
                else:
                    print(f"❌ Failed to fetch data for gap")
            
            if new_data_frames:
                # Combine all data
                all_data = [existing_df] + new_data_frames
                combined_df = pd.concat(all_data).sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                # Save updated data
                if file_path.endswith('.parquet'):
                    combined_df.to_parquet(file_path)
                else:
                    combined_df.to_csv(file_path)
                
                print(f"✅ Successfully backfilled {file_path}")
                return True
            
        except Exception as e:
            central_logger.error(f"Failed to backfill {file_path}: {e}")
            print(f"❌ Failed to backfill {file_path}: {e}")
        
        return False
    
    def auto_backfill(self, file_path: str) -> bool:
        """Automatically detect and backfill gaps in a data file."""
        analysis = DataRangeAnalyzer.summarize_file(file_path)
        
        if analysis['status'] != 'ok':
            print(f"❌ Cannot backfill {file_path}: {analysis['status']}")
            return False
        
        if analysis['gaps_detected'] == 0:
            print(f"✅ No gaps detected in {file_path}")
            return True
        
        # Determine interval from filename
        interval_map = {
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h'
        }
        
        interval = None
        for key, value in interval_map.items():
            if key in file_path:
                interval = value
                break
        
        if not interval:
            print(f"❌ Cannot determine interval for {file_path}")
            return False
        
        print(f"🔍 Found {analysis['gaps_detected']} gaps in {file_path}")
        return self.backfill_gaps(file_path, analysis['gaps'], interval)

# ============================================================================
# COMPREHENSIVE PREFLIGHT CHECKS
# ============================================================================

class PreflightChecker:
    """Comprehensive preflight checks for trading system readiness."""
    
    @staticmethod
    def check_data_files() -> Dict[str, Any]:
        """Check data file availability and quality."""
        analysis = DataRangeAnalyzer.analyze_all_files()
        
        issues = []
        for result in analysis['results']:
            if result['status'] != 'ok':
                issues.append(f"Data file issue: {result['file']} - {result['status']}")
            elif result['gaps_detected'] > 10:
                issues.append(f"Data quality concern: {result['file']} has {result['gaps_detected']} gaps")
        
        return {
            'status': 'pass' if not issues else 'warning',
            'issues': issues,
            'total_files': analysis['files_analyzed'],
            'successful_files': analysis['successful_files']
        }
    
    @staticmethod
    def check_dependencies() -> Dict[str, Any]:
        """Check system dependencies and imports."""
        issues = []
        
        # Check critical imports
        try:
            import pandas as pd
            import numpy as np
            import requests
        except ImportError as e:
            issues.append(f"Missing dependency: {e}")
        
        # Check data directory
        if not os.path.exists('data'):
            issues.append("Data directory not found")
        
        # Check configuration files
        config_files = ['monitoring_config.json', 'requirements.txt']
        for config_file in config_files:
            if not os.path.exists(config_file):
                issues.append(f"Configuration file missing: {config_file}")
        
        return {
            'status': 'pass' if not issues else 'fail',
            'issues': issues
        }
    
    @staticmethod
    def check_api_connectivity() -> Dict[str, Any]:
        """Check external API connectivity."""
        issues = []
        
        try:
            # Test Binance API
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=10
            )
            if response.status_code != 200:
                issues.append("Binance API not accessible")
        except Exception as e:
            issues.append(f"Binance API connectivity failed: {e}")
        
        return {
            'status': 'pass' if not issues else 'warning',
            'issues': issues
        }
    
    @staticmethod
    def run_comprehensive_check() -> Dict[str, Any]:
        """Run all preflight checks and generate report."""
        print('=' * 60)
        print('🚀 PREFLIGHT CHECK REPORT')
        print('=' * 60)
        
        checks = {
            'data_files': PreflightChecker.check_data_files(),
            'dependencies': PreflightChecker.check_dependencies(),
            'api_connectivity': PreflightChecker.check_api_connectivity()
        }
        
        overall_status = 'pass'
        total_issues = 0
        
        for check_name, result in checks.items():
            print(f"\n📋 {check_name.replace('_', ' ').title()}:")
            if result['status'] == 'pass':
                print(f"   ✅ PASS")
            elif result['status'] == 'warning':
                print(f"   ⚠️  WARNING")
                overall_status = 'warning' if overall_status == 'pass' else overall_status
            else:
                print(f"   ❌ FAIL")
                overall_status = 'fail'
            
            for issue in result['issues']:
                print(f"   • {issue}")
                total_issues += 1
        
        print('=' * 60)
        print(f"🎯 OVERALL STATUS: {overall_status.upper()}")
        print(f"⚠️  Total Issues: {total_issues}")
        print('=' * 60)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'total_issues': total_issues,
            'checks': checks
        }

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Management Scripts')
    parser.add_argument('command', choices=['analyze', 'backfill', 'preflight'], 
                       help='Command to execute')
    parser.add_argument('--file', type=str, help='Specific file to process')
    parser.add_argument('--auto', action='store_true', help='Auto-fix detected issues')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        if args.file:
            result = DataRangeAnalyzer.summarize_file(args.file)
            print(json.dumps(result, indent=2, default=str))
        else:
            DataRangeAnalyzer.analyze_all_files()
    
    elif args.command == 'backfill':
        backfiller = DataBackfiller()
        if args.file:
            success = backfiller.auto_backfill(args.file)
            print(f"Backfill {'successful' if success else 'failed'}")
        else:
            print("Please specify --file for backfill operation")
    
    elif args.command == 'preflight':
        PreflightChecker.run_comprehensive_check()
    
    else:
        parser.print_help()
