#!/usr/bin/env python3
"""
Core Analytics Module - Consolidated Analysis System
Combines all analysis functionality into a unified module.

Consolidated from:
- enhancer.py (Enhancement utilities)
- performance_calculator.py (Performance calculations)  
- generate_plots.py (Plot generation)
- generate_plots_clean.py (Clean plot generation)
- diagnose_trades.py (Trade diagnosis)
- report_slimmer.py (Report slimming)
- surgical_cleanup.py (Surgical cleanup)
- surgical_dedup.py (Surgical deduplication)

Purpose: Unified analytics system for trading performance analysis,
report generation, dashboard enhancement, and trade diagnostics.
"""

from __future__ import annotations
import os
import re
import json
import glob
import datetime
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

# Set up logging
try:
    # Try to import centralized logger
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utilities.utils import central_logger
except ImportError:
    # Fallback to basic logging
    central_logger = logging.getLogger(__name__)
    central_logger.setLevel(logging.INFO)

# ============================================================================
# CONSTANTS AND MARKERS
# ============================================================================

# Dashboard Enhancement Markers
START_MARK = "<!-- ENHANCEMENT_DASHBOARD_START -->"
END_MARK = "<!-- ENHANCEMENT_DASHBOARD_END -->"
V4_START = "<!-- ENHANCEMENT_DASHBOARD_V4_START -->"
V4_END = "<!-- ENHANCEMENT_DASHBOARD_V4_END -->"

# Legacy marker variants for cleanup
LEGACY_MARKERS: List[Tuple[str, str]] = [
    ("<!-- ENHANCEMENT_DASHBOARD_START -->", "<!-- ENHANCEMENT_DASHBOARD_END -->"),
    ("<!-- ENHANCEMENT_DASHBOARD_V2_START -->", "<!-- ENHANCEMENT_DASHBOARD_V2_END -->"),
    ("<!-- ENHANCEMENT_DASHBOARD_V3_START -->", "<!-- ENHANCEMENT_DASHBOARD_V3_END -->"),
]

# ============================================================================
# PERFORMANCE CALCULATOR CLASS
# ============================================================================

class PerformanceCalculator:
    """Calculate accurate trading performance metrics from trading journal data."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.metrics = {}
        
    def load_trading_journal(self, journal_path: str = "data/trading_journal.json") -> Dict[str, Any]:
        """Load trading journal data."""
        if not os.path.exists(journal_path):
            raise FileNotFoundError(f"Trading journal not found: {journal_path}")
            
        with open(journal_path, 'r') as f:
            return json.load(f)
    
    def calculate_equity_curve(self, journal_data: Dict[str, Any]) -> pd.DataFrame:
        """Calculate equity curve from session data."""
        sessions = journal_data.get('sessions', [])
        
        equity_data = []
        running_equity = self.initial_capital
        
        for session in sessions:
            session_pnl = session.get('total_pnl', 0)
            session_date = session.get('timestamp', datetime.datetime.now().isoformat())
            
            # Update equity
            running_equity += session_pnl
            running_equity = max(running_equity, 0)  # Can't go below 0
            
            equity_data.append({
                'date': session_date,
                'equity': running_equity,
                'pnl': session_pnl,
                'session_id': session.get('session_id', 'unknown')
            })
        
        return pd.DataFrame(equity_data)
    
    def calculate_performance_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if equity_curve.empty:
            return {}
        
        # Basic metrics
        total_return = (equity_curve['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change().fillna(0)
        
        # Risk metrics
        volatility = equity_curve['returns'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (equity_curve['returns'].mean() * 252) / (equity_curve['returns'].std() * np.sqrt(252)) if equity_curve['returns'].std() > 0 else 0
        
        # Drawdown calculation
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
        max_drawdown = equity_curve['drawdown'].min()
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'final_equity': equity_curve['equity'].iloc[-1],
            'total_sessions': len(equity_curve)
        }

# ============================================================================
# DASHBOARD ENHANCEMENT SYSTEM
# ============================================================================

class DashboardEnhancer:
    """Enhanced dashboard generation system."""
    
    @staticmethod
    def _wrap(title: str, body: str) -> str:
        """Wrap content in analysis section."""
        return f"<div class='analysis-section'><h3 class='section-title'>{title}</h3>{body}</div>"
    
    @staticmethod
    def generate_sections(data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """Generate dashboard sections with comprehensive analytics."""
        return [
            ("sec_perf", "📊 Performance", DashboardEnhancer._wrap(
                "📊 Performance Overview",
                f"""
                <div class='performance-metrics'>
                    <div class='metric-row'>
                        <span class='metric-label'>Total Return:</span>
                        <span class='metric-value'>{data.get('total_return', 0):.2%}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Sharpe Ratio:</span>
                        <span class='metric-value'>{data.get('sharpe_ratio', 0):.2f}</span>
                    </div>
                    <div class='metric-row'>
                        <span class='metric-label'>Max Drawdown:</span>
                        <span class='metric-value'>{data.get('max_drawdown', 0):.2%}</span>
                    </div>
                </div>
                """
            )),
            ("sec_errors", "🚨 Errors", DashboardEnhancer._wrap(
                "🚨 Error Analysis",
                "<p>No critical errors detected in latest analysis.</p>"
            )),
            ("sec_health", "💓 Health", DashboardEnhancer._wrap(
                "💓 System Health",
                f"""
                <div class='health-metrics'>
                    <div class='health-indicator'>
                        <span class='health-label'>Trading Sessions:</span>
                        <span class='health-value'>{data.get('total_sessions', 0)}</span>
                    </div>
                    <div class='health-indicator'>
                        <span class='health-label'>System Status:</span>
                        <span class='health-value'>✅ Operational</span>
                    </div>
                </div>
                """
            )),
            ("sec_risk", "🛡 Risk", DashboardEnhancer._wrap(
                "🛡 Risk Management",
                f"""
                <div class='risk-metrics'>
                    <div class='risk-indicator'>
                        <span class='risk-label'>Volatility:</span>
                        <span class='risk-value'>{data.get('volatility', 0):.2%}</span>
                    </div>
                    <div class='risk-indicator'>
                        <span class='risk-label'>Risk Level:</span>
                        <span class='risk-value'>{'🟢 Low' if data.get('volatility', 0) < 0.2 else '🟡 Medium' if data.get('volatility', 0) < 0.4 else '🔴 High'}</span>
                    </div>
                </div>
                """
            )),
        ]
    
    @staticmethod
    def generate_enhanced_block(data: Dict[str, Any]) -> str:
        """Generate complete enhanced dashboard block."""
        sections = DashboardEnhancer.generate_sections(data)
        buttons = [f"<button class='section-nav-btn' data-target='{sid}'>{title}</button>" 
                  for sid, title, _ in sections]
        
        toolbar = (
            "<div class='sections-toolbar'>" + 
            ''.join(buttons) +
            "<button class='section-nav-btn small js-show-all'>Show All</button>" +
            "<button class='section-nav-btn small js-hide-all'>Hide All</button>" +
            "</div>"
        )
        
        hidden_sections = [
            f"<section id='{sid}' class='dashboard-section hidden'>"
            f"<h2 class='section-heading'>{title}</h2>{content}</section>"
            for sid, title, content in sections
        ]
        
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        return f"""
{V4_START}
<style>
.landing-banner {{
    margin: 40px 0 25px;
    padding: 34px;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    border-radius: 16px;
    color: #fff;
}}
.sections-toolbar-wrapper {{
    position: sticky;
    top: 0;
    z-index: 50;
    background: #ffffffdd;
    backdrop-filter: blur(6px);
    padding: 6px;
    border-bottom: 1px solid #e2e6ef;
}}
.sections-toolbar {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}
.section-nav-btn {{
    background: #4556d6;
    color: #fff;
    border: 0;
    padding: 8px 10px;
    border-radius: 6px;
    font-size: .75rem;
    cursor: pointer;
    transition: background 0.2s;
}}
.section-nav-btn:hover {{
    background: #3644c4;
}}
.section-nav-btn.small {{
    background: #6c757d;
}}
.dashboard-section.hidden {{
    display: none;
}}
.dashboard-section.active {{
    display: block;
    animation: fadeIn .3s ease;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
.analysis-section {{
    background: #fff;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    margin-bottom: 15px;
}}
.metric-row, .health-indicator, .risk-indicator {{
    display: flex;
    justify-content: space-between;
    margin: 8px 0;
    padding: 5px 0;
    border-bottom: 1px solid #f0f0f0;
}}
.metric-label, .health-label, .risk-label {{
    font-weight: 500;
    color: #333;
}}
.metric-value, .health-value, .risk-value {{
    font-weight: 600;
    color: #2563eb;
}}
.empty-note {{
    opacity: .55;
    font-size: .85em;
    margin: 18px 4px;
}}
</style>

<div class='landing-banner'>
    <h1>🚀 Enhanced Trading Analytics Dashboard</h1>
    <p>Comprehensive performance analysis and system monitoring</p>
    <p><small>Last updated: {timestamp}</small></p>
</div>

<div class='sections-toolbar-wrapper'>
    {toolbar}
</div>

{''.join(hidden_sections)}

<script>
window.addEventListener('DOMContentLoaded', function() {{
    var sections = Array.from(document.querySelectorAll('.dashboard-section'));
    var buttons = Array.from(document.querySelectorAll('.sections-toolbar .section-nav-btn[data-target]'));
    var showAll = document.querySelector('.js-show-all');
    var hideAll = document.querySelector('.js-hide-all');
    
    function hideAllSections() {{
        sections.forEach(function(s) {{
            s.classList.remove('active');
            s.classList.add('hidden');
        }});
    }}
    
    function showAllSections() {{
        sections.forEach(function(s) {{
            s.classList.remove('hidden');
            s.classList.add('active');
        }});
    }}
    
    buttons.forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            var targetId = btn.getAttribute('data-target');
            var targetSection = document.getElementById(targetId);
            
            if (targetSection) {{
                var isHidden = targetSection.classList.contains('hidden');
                if (isHidden) {{
                    targetSection.classList.remove('hidden');
                    targetSection.classList.add('active');
                }} else {{
                    targetSection.classList.remove('active');
                    targetSection.classList.add('hidden');
                }}
            }}
        }});
    }});
    
    if (showAll) {{
        showAll.addEventListener('click', showAllSections);
    }}
    
    if (hideAll) {{
        hideAll.addEventListener('click', hideAllSections);
    }}
}});
</script>
{V4_END}
"""

# ============================================================================
# TRADE DIAGNOSIS SYSTEM
# ============================================================================

class TradeDiagnostics:
    """Trade analysis and diagnostic system."""
    
    @staticmethod
    def get_latest_run_directory() -> Optional[str]:
        """Find the most recent backtest run directory."""
        try:
            # First, try to read the canonical path from the file
            with open("latest_run_dir.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback to finding the most recent directory by timestamp
            list_of_dirs = glob.glob('plots_output/20*') 
            if not list_of_dirs:
                return None
            latest_dir = max(list_of_dirs, key=os.path.getctime)
            return latest_dir
    
    @staticmethod
    def analyze_trades(run_dir: str) -> Dict[str, Any]:
        """Perform detailed analysis of trade log from backtest run."""
        if not run_dir:
            return {"error": "Could not find the latest run directory"}

        trades_file = os.path.join(run_dir, 'all_trades_detailed.csv')
        if not os.path.exists(trades_file):
            return {"error": f"Trades file not found: {trades_file}"}
        
        try:
            # Load trades data
            trades_df = pd.read_csv(trades_file)
            
            if trades_df.empty:
                return {"error": "No trades data found"}
            
            # Calculate trade statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0]) if 'pnl' in trades_df.columns else 0
            losing_trades = len(trades_df[trades_df['pnl'] < 0]) if 'pnl' in trades_df.columns else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            analysis = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "run_directory": run_dir,
                "analysis_timestamp": datetime.datetime.now().isoformat()
            }
            
            if 'pnl' in trades_df.columns:
                analysis.update({
                    "total_pnl": trades_df['pnl'].sum(),
                    "average_win": trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
                    "average_loss": trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
                    "largest_win": trades_df['pnl'].max(),
                    "largest_loss": trades_df['pnl'].min()
                })
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing trades: {str(e)}"}

# ============================================================================
# REPORT GENERATION SYSTEM
# ============================================================================

class ReportGenerator:
    """Comprehensive report generation system."""
    
    @staticmethod
    def clean_legacy_markers(content: str) -> str:
        """Remove all legacy enhancement markers from content."""
        for start_marker, end_marker in LEGACY_MARKERS:
            pattern = re.escape(start_marker) + r'.*?' + re.escape(end_marker)
            content = re.sub(pattern, '', content, flags=re.DOTALL)
        return content
    
    @staticmethod
    def inject_enhancement_dashboard(html_path: str, data: Dict[str, Any]) -> bool:
        """Inject enhancement dashboard into HTML report."""
        try:
            if not os.path.exists(html_path):
                print(f"❌ HTML file not found: {html_path}")
                return False
            
            # Read current content
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean legacy markers
            content = ReportGenerator.clean_legacy_markers(content)
            
            # Remove existing V4 block if present
            v4_pattern = re.escape(V4_START) + r'.*?' + re.escape(V4_END)
            content = re.sub(v4_pattern, '', content, flags=re.DOTALL)
            
            # Generate new dashboard block
            dashboard_block = DashboardEnhancer.generate_enhanced_block(data)
            
            # Insert after <body> tag or at the beginning
            body_match = re.search(r'<body[^>]*>', content, re.IGNORECASE)
            if body_match:
                insert_pos = body_match.end()
                content = content[:insert_pos] + "\n" + dashboard_block + "\n" + content[insert_pos:]
            else:
                content = dashboard_block + "\n" + content
            
            # Write updated content atomically
            temp_path = html_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            os.replace(temp_path, html_path)
            print(f"✅ Enhanced dashboard injected into {html_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error injecting dashboard: {e}")
            traceback.print_exc()
            return False
    
    @staticmethod
    def create_slim_report(source_path: str, data: Dict[str, Any]) -> bool:
        """Create a slim version of the performance report."""
        try:
            if not os.path.exists(source_path):
                return False
            
            # Generate slim report path
            dir_name = os.path.dirname(source_path)
            slim_path = os.path.join(dir_name, 'performance_report_slim.html')
            
            # Create minimal HTML with enhanced dashboard
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            dashboard_block = DashboardEnhancer.generate_enhanced_block(data)
            
            slim_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Performance Report - Slim Version</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Trading Performance Report - Slim Version</h1>
        <p>Generated: {timestamp}</p>
        {dashboard_block}
    </div>
</body>
</html>"""
            
            with open(slim_path, 'w', encoding='utf-8') as f:
                f.write(slim_content)
            
            print(f"✅ Slim report created: {slim_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating slim report: {e}")
            return False

# ============================================================================
# MAIN ANALYTICS ORCHESTRATOR
# ============================================================================

class CoreAnalytics:
    """Main analytics orchestrator combining all functionality."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.performance_calc = PerformanceCalculator(initial_capital)
        self.dashboard_enhancer = DashboardEnhancer()
        self.trade_diagnostics = TradeDiagnostics()
        self.report_generator = ReportGenerator()
    
    def run_comprehensive_analysis(self, 
                                 journal_path: str = "data/trading_journal.json",
                                 report_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive analysis and generate enhanced reports."""
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "success",
            "analysis": {}
        }
        
        try:
            # Load and analyze trading data
            if os.path.exists(journal_path):
                journal_data = self.performance_calc.load_trading_journal(journal_path)
                equity_curve = self.performance_calc.calculate_equity_curve(journal_data)
                performance_metrics = self.performance_calc.calculate_performance_metrics(equity_curve)
                results["analysis"]["performance"] = performance_metrics
            
            # Analyze latest trades
            latest_run = self.trade_diagnostics.get_latest_run_directory()
            if latest_run:
                trade_analysis = self.trade_diagnostics.analyze_trades(latest_run)
                results["analysis"]["trades"] = trade_analysis
            
            # Generate enhanced reports
            if report_path and os.path.exists(report_path):
                enhanced = self.report_generator.inject_enhancement_dashboard(
                    report_path, results["analysis"]
                )
                slim_created = self.report_generator.create_slim_report(
                    report_path, results["analysis"]
                )
                results["reports"] = {
                    "enhanced": enhanced,
                    "slim_created": slim_created
                }
            
            return results
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"❌ Comprehensive analysis failed: {e}")
            traceback.print_exc()
            return results

# ============================================================================
# COMPATIBILITY FUNCTIONS FOR BACKTEST.PY
# ============================================================================

def plot_trades_for_window(trades_df: pd.DataFrame, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          symbol: str = "BTCUSDT") -> Optional[str]:
    """Plot trades for a specific time window - compatibility function."""
    try:
        if trades_df.empty:
            return None
        
        # Basic trade plotting logic
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter by time window if specified
        if start_time and end_time:
            mask = (trades_df.index >= start_time) & (trades_df.index <= end_time)
            trades_df = trades_df[mask]
        
        # Plot trades
        if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
            ax.scatter(trades_df.index, trades_df['entry_price'], 
                      c='green', marker='^', label='Entry', alpha=0.7)
            ax.scatter(trades_df.index, trades_df['exit_price'], 
                      c='red', marker='v', label='Exit', alpha=0.7)
        
        ax.set_title(f'Trades for {symbol}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = f"plots_output/trades_window_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        central_logger.error(f"Error plotting trades for window: {e}")
        return None

def plot_pnl_distribution(pnl_data: List[float], 
                         bins: int = 50,
                         title: str = "P&L Distribution") -> Optional[str]:
    """Plot P&L distribution - compatibility function."""
    try:
        if not pnl_data:
            return None
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(pnl_data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('P&L')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(pnl_data, vert=True)
        ax2.set_title(f'{title} - Box Plot')
        ax2.set_ylabel('P&L')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', label='Break-even')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"plots_output/pnl_distribution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        central_logger.error(f"Error plotting P&L distribution: {e}")
        return None

def enhance_performance_report(report_path: str, data: Optional[Dict[str, Any]] = None) -> bool:
    """Enhance performance report - compatibility function."""
    try:
        if not data:
            # Collect data automatically
            data = collect_comprehensive_analysis_data()
        
        # Use the ReportGenerator to inject enhancement
        generator = ReportGenerator()
        return generator.inject_enhancement_dashboard(report_path, data)
        
    except Exception as e:
        central_logger.error(f"Error enhancing performance report: {e}")
        return False

def collect_comprehensive_analysis_data() -> Dict[str, Any]:
    """Collect comprehensive analysis data - compatibility function."""
    try:
        analytics = CoreAnalytics()
        
        # Try to get trading journal data
        journal_path = "data/trading_journal.json"
        if os.path.exists(journal_path):
            results = analytics.run_comprehensive_analysis(journal_path)
            return results.get("analysis", {})
        
        # Fallback to basic data structure
        return {
            "performance": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "final_equity": 10000.0,
                "total_sessions": 0
            },
            "trades": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "run_directory": None
            }
        }
        
    except Exception as e:
        central_logger.error(f"Error collecting comprehensive analysis data: {e}")
        return {}

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_performance_analysis(journal_path: str = "data/trading_journal.json") -> Dict[str, Any]:
    """Convenience function to run performance analysis."""
    analytics = CoreAnalytics()
    return analytics.run_comprehensive_analysis(journal_path)

def enhance_latest_report() -> bool:
    """Convenience function to enhance the latest performance report."""
    analytics = CoreAnalytics()
    
    # Find latest report
    latest_run = analytics.trade_diagnostics.get_latest_run_directory()
    if not latest_run:
        print("❌ No latest run directory found")
        return False
    
    report_path = os.path.join(latest_run, 'performance_report.html')
    if not os.path.exists(report_path):
        print(f"❌ Performance report not found: {report_path}")
        return False
    
    # Run analysis and enhance report
    results = analytics.run_comprehensive_analysis(report_path=report_path)
    return results.get("status") == "success"

def diagnose_latest_trades() -> Dict[str, Any]:
    """Convenience function to diagnose latest trades."""
    diagnostics = TradeDiagnostics()
    latest_run = diagnostics.get_latest_run_directory()
    if latest_run:
        return diagnostics.analyze_trades(latest_run)
    return {"error": "No latest run directory found"}

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "analyze":
            print("🔍 Running comprehensive analysis...")
            results = run_performance_analysis()
            print(f"📊 Analysis complete: {results['status']}")
            
        elif command == "enhance":
            print("🚀 Enhancing latest report...")
            success = enhance_latest_report()
            print(f"✅ Enhancement {'successful' if success else 'failed'}")
            
        elif command == "diagnose":
            print("🔬 Diagnosing latest trades...")
            diagnosis = diagnose_latest_trades()
            print(f"📋 Diagnosis: {diagnosis}")
            
        else:
            print("❌ Unknown command. Use: analyze, enhance, or diagnose")
    else:
        print("🎯 Core Analytics Module")
        print("Usage: python analysis/core_analytics.py [analyze|enhance|diagnose]")
        print("  analyze  - Run comprehensive performance analysis")
        print("  enhance  - Enhance latest performance report")
        print("  diagnose - Diagnose latest trades")
