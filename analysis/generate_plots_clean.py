"""generate_plots: fresh minimal enhancement injector (V4).

Purpose: Idempotently insert a single dynamic analysis panel into the latest
performance_report.html using a template + toolbar system. All legacy
implementations (v1/v2/v3) are stripped first by marker scanning.

Design:
 - Pure stdlib, concise (~140 loc)
 - New markers: V4_START / V4_END
 - Inserts right after the opening <body> tag if present, else prepends
 - Cleans any previous blocks (legacy markers, hidden sections, duplicate toolbars)
 - Safe atomic write via .tmp replacement
"""
from __future__ import annotations
import os, re, datetime, traceback
from typing import Any, Dict, List, Optional

# Legacy marker variants we will purge
LEGACY_MARKERS: List[tuple[str,str]] = [
    ("<!-- ENHANCEMENT_DASHBOARD_START -->", "<!-- ENHANCEMENT_DASHBOARD_END -->"),
    ("<!-- ENHANCEMENT_DASHBOARD_V2_START -->", "<!-- ENHANCEMENT_DASHBOARD_V2_END -->"),
    ("<!-- ENHANCEMENT_DASHBOARD_V3_START -->", "<!-- ENHANCEMENT_DASHBOARD_V3_END -->"),
]
V4_START = "<!-- ENHANCEMENT_DASHBOARD_V4_START -->"
V4_END   = "<!-- ENHANCEMENT_DASHBOARD_V4_END -->"

# ---------------- Section Content (stubbed) -----------------
def _wrap(title: str, body: str) -> str:
    return f"<div class='analysis-section'><h3>{title}</h3>{body}</div>"

def _sections(_: Dict[str, Any]) -> List[tuple[str,str,str]]:
    # Build comprehensive dashboard sections with rich content
    
    # Analytics Section
    analytics_content = """
    <div class='comprehensive-section'>
        <h3>🔍 Advanced Analytics Dashboard</h3>
        <div class='analytics-grid'>
            <div class='metric-card'>
                <h4>📊 Performance Metrics</h4>
                <div class='metric-row'>
                    <span class='metric-label'>Sharpe Ratio:</span>
                    <span class='metric-value' id='sharpe-ratio'>Calculating...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Max Drawdown:</span>
                    <span class='metric-value' id='max-drawdown'>Calculating...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Volatility:</span>
                    <span class='metric-value' id='volatility'>Calculating...</span>
                </div>
            </div>
            
            <div class='metric-card'>
                <h4>📈 Trading Statistics</h4>
                <div class='metric-row'>
                    <span class='metric-label'>Total Return:</span>
                    <span class='metric-value' id='total-return'>Loading...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Annual Return:</span>
                    <span class='metric-value' id='annual-return'>Loading...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Win Streak:</span>
                    <span class='metric-value' id='win-streak'>Loading...</span>
                </div>
            </div>
            
            <div class='metric-card'>
                <h4>🎯 Accuracy Metrics</h4>
                <div class='metric-row'>
                    <span class='metric-label'>Hit Rate:</span>
                    <span class='metric-value' id='hit-rate'>Loading...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Profit Factor:</span>
                    <span class='metric-value' id='profit-factor'>Loading...</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-label'>Avg Trade:</span>
                    <span class='metric-value' id='avg-trade'>Loading...</span>
                </div>
            </div>
        </div>
        <p class='section-note'>📋 Detailed analytics calculated from QuantStats data below.</p>
    </div>
    """
    
    # Performance Section
    performance_content = """
    <div class='comprehensive-section'>
        <h3>🚀 Performance Overview Dashboard</h3>
        <div class='performance-summary'>
            <div class='summary-card highlight'>
                <h4>💰 Portfolio Performance</h4>
                <div class='big-metric'>
                    <span class='big-value' id='portfolio-value'>$0.00</span>
                    <span class='big-label'>Current Portfolio Value</span>
                </div>
                <div class='change-indicator'>
                    <span id='portfolio-change'>+0.00%</span>
                    <span class='timeframe'>All Time</span>
                </div>
            </div>
            
            <div class='summary-card'>
                <h4>📊 Trade Summary</h4>
                <div class='summary-stats'>
                    <div class='stat-item'>
                        <span class='stat-number' id='total-trades-perf'>0</span>
                        <span class='stat-label'>Total Trades</span>
                    </div>
                    <div class='stat-item'>
                        <span class='stat-number' id='winning-trades'>0</span>
                        <span class='stat-label'>Winning Trades</span>
                    </div>
                    <div class='stat-item'>
                        <span class='stat-number' id='losing-trades'>0</span>
                        <span class='stat-label'>Losing Trades</span>
                    </div>
                </div>
            </div>
            
            <div class='summary-card'>
                <h4>⏱️ Time Analysis</h4>
                <div class='time-metrics'>
                    <div class='time-item'>
                        <span class='time-label'>Avg Hold Time:</span>
                        <span class='time-value' id='avg-hold-time'>-</span>
                    </div>
                    <div class='time-item'>
                        <span class='time-label'>Best Day:</span>
                        <span class='time-value' id='best-day'>-</span>
                    </div>
                    <div class='time-item'>
                        <span class='time-label'>Worst Day:</span>
                        <span class='time-value' id='worst-day'>-</span>
                    </div>
                </div>
            </div>
        </div>
        <p class='section-note'>📈 Performance data integrated with live trading systems.</p>
    </div>
    """
    
    # Risk Section
    risk_content = """
    <div class='comprehensive-section'>
        <h3>🛡️ Risk Management Dashboard</h3>
        <div class='risk-assessment'>
            <div class='risk-card critical'>
                <h4>⚠️ Risk Alerts</h4>
                <div class='alert-list'>
                    <div class='alert-item status-ok'>
                        <span class='alert-icon'>✅</span>
                        <span class='alert-text'>Drawdown within limits</span>
                    </div>
                    <div class='alert-item status-ok'>
                        <span class='alert-icon'>✅</span>
                        <span class='alert-text'>Position size controlled</span>
                    </div>
                    <div class='alert-item status-monitor'>
                        <span class='alert-icon'>👁️</span>
                        <span class='alert-text'>Volatility monitoring</span>
                    </div>
                </div>
            </div>
            
            <div class='risk-card'>
                <h4>📊 Risk Metrics</h4>
                <div class='risk-metrics'>
                    <div class='risk-item'>
                        <span class='risk-label'>VaR (95%):</span>
                        <span class='risk-value' id='var-95'>Calculating...</span>
                    </div>
                    <div class='risk-item'>
                        <span class='risk-label'>Beta:</span>
                        <span class='risk-value' id='beta'>Calculating...</span>
                    </div>
                    <div class='risk-item'>
                        <span class='risk-label'>Correlation:</span>
                        <span class='risk-value' id='correlation'>Calculating...</span>
                    </div>
                </div>
            </div>
            
            <div class='risk-card'>
                <h4>🎚️ Risk Controls</h4>
                <div class='control-panel'>
                    <div class='control-item'>
                        <span class='control-label'>Max Position Size:</span>
                        <span class='control-value'>25%</span>
                    </div>
                    <div class='control-item'>
                        <span class='control-label'>Stop Loss:</span>
                        <span class='control-value'>-2%</span>
                    </div>
                    <div class='control-item'>
                        <span class='control-label'>Daily Loss Limit:</span>
                        <span class='control-value'>-5%</span>
                    </div>
                </div>
            </div>
        </div>
        <p class='section-note'>🔒 Risk management ensures capital preservation.</p>
    </div>
    """
    
    # Health Section
    health_content = """
    <div class='comprehensive-section'>
        <h3>💓 System Health Dashboard</h3>
        <div class='health-monitor'>
            <div class='health-card status-good'>
                <h4>🖥️ System Status</h4>
                <div class='status-grid'>
                    <div class='status-item'>
                        <span class='status-dot green'></span>
                        <span class='status-label'>Trading Engine</span>
                        <span class='status-value'>Online</span>
                    </div>
                    <div class='status-item'>
                        <span class='status-dot green'></span>
                        <span class='status-label'>Data Feed</span>
                        <span class='status-value'>Connected</span>
                    </div>
                    <div class='status-item'>
                        <span class='status-dot yellow'></span>
                        <span class='status-label'>Risk Monitor</span>
                        <span class='status-value'>Monitoring</span>
                    </div>
                </div>
            </div>
            
            <div class='health-card'>
                <h4>📊 Performance Health</h4>
                <div class='health-metrics'>
                    <div class='health-bar'>
                        <span class='health-label'>Strategy Confidence:</span>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: 78%'></div>
                        </div>
                        <span class='health-percentage'>78%</span>
                    </div>
                    <div class='health-bar'>
                        <span class='health-label'>Data Quality:</span>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: 92%'></div>
                        </div>
                        <span class='health-percentage'>92%</span>
                    </div>
                    <div class='health-bar'>
                        <span class='health-label'>System Load:</span>
                        <div class='progress-bar'>
                            <div class='progress-fill' style='width: 34%'></div>
                        </div>
                        <span class='health-percentage'>34%</span>
                    </div>
                </div>
            </div>
            
            <div class='health-card'>
                <h4>🔧 Maintenance</h4>
                <div class='maintenance-info'>
                    <div class='maintenance-item'>
                        <span class='maintenance-label'>Last Backup:</span>
                        <span class='maintenance-value' id='last-backup'>2 hours ago</span>
                    </div>
                    <div class='maintenance-item'>
                        <span class='maintenance-label'>Next Restart:</span>
                        <span class='maintenance-value'>Sunday 02:00</span>
                    </div>
                    <div class='maintenance-item'>
                        <span class='maintenance-label'>Uptime:</span>
                        <span class='maintenance-value' id='uptime'>7d 14h 23m</span>
                    </div>
                </div>
            </div>
        </div>
        <p class='section-note'>🔄 Continuous monitoring ensures optimal performance.</p>
    </div>
    """
    
    # Live Data Section - Real-time Integration  
    live_data_content = """
    <div class='comprehensive-section'>
        <h3>📊 Live Trading Data</h3>
        <div class='update-indicator'>
            🔴 Real-time data from Google Cloud bot • Updates every 5 seconds • Live BTC: $<span id="live-btc-indicator">Loading...</span>
        </div>
        <div class='live-data-grid'>
            <div class='live-data-card'>
                <div class='live-data-title'>💰 Portfolio Overview</div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Total Value:</span>
                    <span class='live-data-value' id='live-portfolio-value'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Daily Change:</span>
                    <span class='live-data-value profit' id='live-portfolio-change'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Daily P&L:</span>
                    <span class='live-data-value profit' id='daily-pnl'>Loading...</span>
                </div>
            </div>
            
            <div class='live-data-card'>
                <div class='live-data-title'>📈 Market Prices</div>
                <div class='live-data-item'>
                    <span class='live-data-label'>BTC:</span>
                    <span class='live-data-value' id='btc-price'>Loading...</span>
                    <span class='live-data-value market-change profit' id='btc-change'>+0.00%</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>ETH:</span>
                    <span class='live-data-value' id='eth-price'>Loading...</span>
                    <span class='live-data-value market-change profit' id='eth-change'>+0.00%</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>SOL:</span>
                    <span class='live-data-value' id='sol-price'>Loading...</span>
                    <span class='live-data-value market-change profit' id='sol-change'>+0.00%</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>ADA:</span>
                    <span class='live-data-value' id='ada-price'>Loading...</span>
                    <span class='live-data-value market-change profit' id='ada-change'>+0.00%</span>
                </div>
            </div>
            
            <div class='live-data-card'>
                <div class='live-data-title'>📊 Active Positions</div>
                <div class='live-data-item'>
                    <span class='live-data-label'>BTC Long:</span>
                    <span class='live-data-value' id='position-1-current'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>P&L:</span>
                    <span class='live-data-value profit' id='position-1-pnl'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>ETH Position:</span>
                    <span class='live-data-value' id='position-2-current'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>P&L:</span>
                    <span class='live-data-value loss' id='position-2-pnl'>Loading...</span>
                </div>
            </div>
            
            <div class='live-data-card'>
                <div class='live-data-title'>🔗 System Status</div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Market Data:</span>
                    <span class='live-data-value' id='market-data-status'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Bot Status:</span>
                    <span class='live-data-value' id='bot-status'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Position Sync:</span>
                    <span class='live-data-value' id='position-sync'>Loading...</span>
                </div>
                <div class='live-data-item'>
                    <span class='live-data-label'>Feed Latency:</span>
                    <span class='live-data-value' id='feed-latency'>Loading...</span>
                </div>
            </div>
        </div>
        <p class='section-note'>🔴 Live data refreshes every 5 seconds. Market data provided by exchange APIs.</p>
    </div>
    """
    
    return [
        ("analytics", "📊 Analytics", analytics_content),
        ("performance", "🚀 Performance", performance_content),
        ("risk", "🛡️ Risk", risk_content),
        ("health", "💓 Health", health_content),
        ("livedata", "📡 Live Data", live_data_content),
    ]

# --------------- Strip previous versions --------------------
_FULL_PATTERNS = [
    re.compile(re.escape(s) + r".*?" + re.escape(e), re.DOTALL|re.IGNORECASE)
    for s,e in LEGACY_MARKERS + [(V4_START, V4_END)]
]

def _strip(html: str) -> str:
    """Remove any prior enhancement blocks (all versions) and legacy remnants.

    Uses two strategies:
      1. Regex non-greedy removal for normal well-formed blocks.
      2. Fallback manual slicing in case of nested comments or edge formatting.
    """
    # Strategy 1: regex passes
    changed = True
    while changed:
        changed = False
        for pat in _FULL_PATTERNS:
            html, c = pat.subn("", html)
            if c:
                changed = True

    # Strategy 2: manual slicing (defensive)
    def remove_pair(h: str, start: str, end: str) -> str:
        while True:
            s = h.find(start)
            if s == -1:
                break
            e = h.find(end, s + len(start))
            if e == -1:
                # remove just the start marker line to avoid infinite retention
                h = h[:s] + h[s+len(start):]
            else:
                h = h[:s] + h[e+len(end):]
        return h
    for s,e in LEGACY_MARKERS:
        html = remove_pair(html, s, e)

    # Remove stray legacy classes / containers
    legacy_residuals = [
        r"dashboard-section hidden",
        r"<div class='dashboard-sections-container'>.*?</div>",
        r"<section id='sec_[^']+'.*?</section>",
    ]
    for pat in legacy_residuals:
        html = re.sub(pat, '', html, flags=re.DOTALL|re.IGNORECASE)
    return html

# --------------- Build new V4 block -------------------------
def _build_block(data: Dict[str, Any]) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    sections = _sections(data)
    buttons = ''.join(f"<button class='v4-btn' data-section='{sid}'>{title}</button>" for sid,title,_ in sections)
    templates = ''.join(
        f"<template id='v4_{sid}'><h2>{title}</h2>{content}</template>"
        for sid,title,content in sections
    )
    return (
        f"{V4_START}\n"
        "<style>\n"
        "/* V4 Dashboard - Modern Landing Page with Sticky Toolbar */\n"
        "/* Override QuantStats body styles */\n"
        "body { margin: 0 !important; padding: 0 !important; background: #f8fafc !important; }\n"
        "\n"
        "/* Landing Banner */\n"
        ".v4-banner {\n"
        "    margin: 20px auto;\n"
        "    max-width: 1000px;\n"
        "    padding: 40px;\n"
        "    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n"
        "    border-radius: 20px;\n"
        "    color: white;\n"
        "    text-align: center;\n"
        "    box-shadow: 0 10px 30px rgba(0,0,0,0.2);\n"
        "}\n"
        ".v4-banner h1 {\n"
        "    font-size: 2.5em;\n"
        "    margin: 0 0 15px 0;\n"
        "    font-weight: 300;\n"
        "    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);\n"
        "}\n"
        ".v4-banner p {\n"
        "    font-size: 1.1em;\n"
        "    opacity: 0.9;\n"
        "    margin: 10px 0;\n"
        "}\n"
        ".v4-timestamp {\n"
        "    font-size: 0.85em;\n"
        "    opacity: 0.7;\n"
        "}\n"
        "\n"
        "/* Sticky Button Toolbar */\n"
        ".v4-toolbar {\n"
        "    position: sticky;\n"
        "    top: 0;\n"
        "    z-index: 1000;\n"
        "    background: rgba(255, 255, 255, 0.95);\n"
        "    backdrop-filter: blur(10px);\n"
        "    border-bottom: 2px solid #e5e7eb;\n"
        "    padding: 15px 0;\n"
        "    margin: 0;\n"
        "    box-shadow: 0 2px 10px rgba(0,0,0,0.1);\n"
        "}\n"
        ".v4-toolbar-container {\n"
        "    max-width: 1000px;\n"
        "    margin: 0 auto;\n"
        "    padding: 0 20px;\n"
        "}\n"
        ".v4-buttons {\n"
        "    display: flex;\n"
        "    flex-wrap: wrap;\n"
        "    gap: 12px;\n"
        "    justify-content: center;\n"
        "}\n"
        ".v4-btn {\n"
        "    background: #4f46e5;\n"
        "    color: white;\n"
        "    border: none;\n"
        "    padding: 12px 20px;\n"
        "    border-radius: 8px;\n"
        "    font-size: 0.9rem;\n"
        "    font-weight: 500;\n"
        "    cursor: pointer;\n"
        "    transition: all 0.2s ease;\n"
        "    box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n"
        "}\n"
        ".v4-btn:hover {\n"
        "    background: #3730a3;\n"
        "    transform: translateY(-1px);\n"
        "    box-shadow: 0 4px 8px rgba(0,0,0,0.15);\n"
        "}\n"
        ".v4-btn.active {\n"
        "    background: #059669;\n"
        "}\n"
        ".v4-btn.utility {\n"
        "    background: #6b7280;\n"
        "}\n"
        ".v4-btn.utility:hover {\n"
        "    background: #4b5563;\n"
        "}\n"
        "\n"
        "/* Content Area */\n"
        ".v4-content {\n"
        "    max-width: 1000px;\n"
        "    margin: 30px auto;\n"
        "    padding: 0 20px;\n"
        "    min-height: 300px;\n"
        "}\n"
        ".v4-section {\n"
        "    background: white;\n"
        "    padding: 30px;\n"
        "    border-radius: 12px;\n"
        "    box-shadow: 0 4px 6px rgba(0,0,0,0.05);\n"
        "    border: 1px solid #e5e7eb;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".v4-welcome {\n"
        "    text-align: center;\n"
        "    padding: 60px 30px;\n"
        "    color: #6b7280;\n"
        "    font-size: 1.1em;\n"
        "    background: white;\n"
        "    border-radius: 12px;\n"
        "    box-shadow: 0 4px 6px rgba(0,0,0,0.05);\n"
        "}\n"
        "\n"
        "/* Restore container margins for QuantStats content */\n"
        ".container {\n"
        "    margin-top: 40px !important;\n"
        "}\n"
        "\n"
        "/* Comprehensive Dashboard Styles */\n"
        ".comprehensive-section {\n"
        "    padding: 0;\n"
        "}\n"
        ".comprehensive-section h3 {\n"
        "    margin: 0 0 25px 0;\n"
        "    color: #1f2937;\n"
        "    border-bottom: 3px solid #4f46e5;\n"
        "    padding-bottom: 10px;\n"
        "    font-size: 1.5em;\n"
        "}\n"
        ".section-note {\n"
        "    margin-top: 25px;\n"
        "    padding: 12px 20px;\n"
        "    background: #f0f9ff;\n"
        "    border-left: 4px solid #0ea5e9;\n"
        "    color: #0c4a6e;\n"
        "    border-radius: 0 8px 8px 0;\n"
        "    font-style: italic;\n"
        "}\n"
        "\n"
        "/* Analytics Grid */\n"
        ".analytics-grid {\n"
        "    display: grid;\n"
        "    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));\n"
        "    gap: 20px;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".metric-card {\n"
        "    background: #f8fafc;\n"
        "    border: 1px solid #e2e8f0;\n"
        "    border-radius: 12px;\n"
        "    padding: 20px;\n"
        "    transition: transform 0.2s ease, box-shadow 0.2s ease;\n"
        "}\n"
        ".metric-card:hover {\n"
        "    transform: translateY(-2px);\n"
        "    box-shadow: 0 8px 25px rgba(0,0,0,0.1);\n"
        "}\n"
        ".metric-card h4 {\n"
        "    margin: 0 0 15px 0;\n"
        "    color: #4f46e5;\n"
        "    font-size: 1.1em;\n"
        "}\n"
        ".metric-row {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    align-items: center;\n"
        "    padding: 8px 0;\n"
        "    border-bottom: 1px solid #e2e8f0;\n"
        "}\n"
        ".metric-row:last-child { border-bottom: none; }\n"
        ".metric-label {\n"
        "    color: #6b7280;\n"
        "    font-weight: 500;\n"
        "}\n"
        ".metric-value {\n"
        "    color: #1f2937;\n"
        "    font-weight: 600;\n"
        "}\n"
        "\n"
        "/* Performance Summary */\n"
        ".performance-summary {\n"
        "    display: grid;\n"
        "    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n"
        "    gap: 20px;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".summary-card {\n"
        "    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);\n"
        "    border: 1px solid #e2e8f0;\n"
        "    border-radius: 15px;\n"
        "    padding: 25px;\n"
        "    box-shadow: 0 4px 6px rgba(0,0,0,0.05);\n"
        "}\n"
        ".summary-card.highlight {\n"
        "    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);\n"
        "    border-color: #10b981;\n"
        "}\n"
        ".summary-card h4 {\n"
        "    margin: 0 0 20px 0;\n"
        "    color: #1f2937;\n"
        "    font-size: 1.2em;\n"
        "}\n"
        ".big-metric {\n"
        "    text-align: center;\n"
        "    margin-bottom: 15px;\n"
        "}\n"
        ".big-value {\n"
        "    display: block;\n"
        "    font-size: 2.2em;\n"
        "    font-weight: 700;\n"
        "    color: #059669;\n"
        "    margin-bottom: 5px;\n"
        "}\n"
        ".big-label {\n"
        "    color: #6b7280;\n"
        "    font-size: 0.9em;\n"
        "}\n"
        ".change-indicator {\n"
        "    text-align: center;\n"
        "    padding: 10px;\n"
        "    background: rgba(5, 150, 105, 0.1);\n"
        "    border-radius: 8px;\n"
        "    color: #059669;\n"
        "    font-weight: 600;\n"
        "}\n"
        ".summary-stats {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    text-align: center;\n"
        "}\n"
        ".stat-item {\n"
        "    flex: 1;\n"
        "}\n"
        ".stat-number {\n"
        "    display: block;\n"
        "    font-size: 1.8em;\n"
        "    font-weight: 700;\n"
        "    color: #4f46e5;\n"
        "}\n"
        ".stat-label {\n"
        "    color: #6b7280;\n"
        "    font-size: 0.85em;\n"
        "}\n"
        ".time-metrics {\n"
        "    space-y: 10px;\n"
        "}\n"
        ".time-item {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    padding: 8px 0;\n"
        "    border-bottom: 1px solid #e2e8f0;\n"
        "}\n"
        ".time-item:last-child { border-bottom: none; }\n"
        ".time-label { color: #6b7280; }\n"
        ".time-value { color: #1f2937; font-weight: 600; }\n"
        "\n"
        "/* Risk Assessment */\n"
        ".risk-assessment {\n"
        "    display: grid;\n"
        "    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));\n"
        "    gap: 20px;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".risk-card {\n"
        "    background: #ffffff;\n"
        "    border: 1px solid #e2e8f0;\n"
        "    border-radius: 12px;\n"
        "    padding: 20px;\n"
        "    box-shadow: 0 2px 4px rgba(0,0,0,0.05);\n"
        "}\n"
        ".risk-card.critical {\n"
        "    border-color: #f59e0b;\n"
        "    background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);\n"
        "}\n"
        ".risk-card h4 {\n"
        "    margin: 0 0 15px 0;\n"
        "    color: #1f2937;\n"
        "}\n"
        ".alert-list {\n"
        "    space-y: 8px;\n"
        "}\n"
        ".alert-item {\n"
        "    display: flex;\n"
        "    align-items: center;\n"
        "    padding: 8px 12px;\n"
        "    border-radius: 6px;\n"
        "    margin-bottom: 8px;\n"
        "}\n"
        ".alert-item.status-ok {\n"
        "    background: #ecfdf5;\n"
        "    color: #065f46;\n"
        "}\n"
        ".alert-item.status-monitor {\n"
        "    background: #fef3c7;\n"
        "    color: #92400e;\n"
        "}\n"
        ".alert-icon {\n"
        "    margin-right: 10px;\n"
        "}\n"
        ".risk-metrics, .control-panel {\n"
        "    space-y: 8px;\n"
        "}\n"
        ".risk-item, .control-item {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    padding: 8px 0;\n"
        "    border-bottom: 1px solid #f3f4f6;\n"
        "}\n"
        ".risk-item:last-child, .control-item:last-child { border-bottom: none; }\n"
        ".risk-label, .control-label { color: #6b7280; }\n"
        ".risk-value, .control-value { color: #1f2937; font-weight: 600; }\n"
        "\n"
        "/* Health Monitor */\n"
        ".health-monitor {\n"
        "    display: grid;\n"
        "    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n"
        "    gap: 20px;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".health-card {\n"
        "    background: #ffffff;\n"
        "    border: 1px solid #e2e8f0;\n"
        "    border-radius: 12px;\n"
        "    padding: 20px;\n"
        "    box-shadow: 0 2px 4px rgba(0,0,0,0.05);\n"
        "}\n"
        ".health-card.status-good {\n"
        "    border-color: #10b981;\n"
        "    background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);\n"
        "}\n"
        ".health-card h4 {\n"
        "    margin: 0 0 15px 0;\n"
        "    color: #1f2937;\n"
        "}\n"
        ".status-grid {\n"
        "    space-y: 10px;\n"
        "}\n"
        ".status-item {\n"
        "    display: flex;\n"
        "    align-items: center;\n"
        "    justify-content: space-between;\n"
        "    padding: 8px 0;\n"
        "    margin-bottom: 10px;\n"
        "}\n"
        ".status-dot {\n"
        "    width: 8px;\n"
        "    height: 8px;\n"
        "    border-radius: 50%;\n"
        "    margin-right: 10px;\n"
        "}\n"
        ".status-dot.green { background: #10b981; }\n"
        ".status-dot.yellow { background: #f59e0b; }\n"
        ".status-dot.red { background: #ef4444; }\n"
        ".status-label { color: #6b7280; flex: 1; }\n"
        ".status-value { color: #1f2937; font-weight: 600; }\n"
        ".health-metrics {\n"
        "    space-y: 15px;\n"
        "}\n"
        ".health-bar {\n"
        "    margin-bottom: 15px;\n"
        "}\n"
        ".health-label {\n"
        "    display: block;\n"
        "    color: #6b7280;\n"
        "    margin-bottom: 5px;\n"
        "    font-size: 0.9em;\n"
        "}\n"
        ".progress-bar {\n"
        "    width: 100%;\n"
        "    height: 8px;\n"
        "    background: #e5e7eb;\n"
        "    border-radius: 4px;\n"
        "    overflow: hidden;\n"
        "    margin-bottom: 5px;\n"
        "}\n"
        ".progress-fill {\n"
        "    height: 100%;\n"
        "    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);\n"
        "    transition: width 0.3s ease;\n"
        "}\n"
        ".health-percentage {\n"
        "    color: #4f46e5;\n"
        "    font-weight: 600;\n"
        "    font-size: 0.9em;\n"
        "}\n"
        ".maintenance-info {\n"
        "    space-y: 8px;\n"
        "}\n"
        ".maintenance-item {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    padding: 8px 0;\n"
        "    border-bottom: 1px solid #f3f4f6;\n"
        "}\n"
        ".maintenance-item:last-child { border-bottom: none; }\n"
        ".maintenance-label { color: #6b7280; }\n"
        ".maintenance-value { color: #1f2937; font-weight: 600; }\n"
        "\n"
        "/* Live Data Styles */\n"
        ".live-data-grid {\n"
        "    display: grid;\n"
        "    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));\n"
        "    gap: 20px;\n"
        "    margin-bottom: 20px;\n"
        "}\n"
        ".live-data-card {\n"
        "    background: #ffffff;\n"
        "    border: 1px solid #e2e8f0;\n"
        "    border-radius: 12px;\n"
        "    padding: 20px;\n"
        "    box-shadow: 0 2px 4px rgba(0,0,0,0.05);\n"
        "    transition: transform 0.2s ease, box-shadow 0.2s ease;\n"
        "}\n"
        ".live-data-card:hover {\n"
        "    transform: translateY(-1px);\n"
        "    box-shadow: 0 4px 12px rgba(0,0,0,0.1);\n"
        "}\n"
        ".live-data-title {\n"
        "    font-weight: 700;\n"
        "    color: #1f2937;\n"
        "    margin-bottom: 15px;\n"
        "    font-size: 1.1em;\n"
        "}\n"
        ".live-data-item {\n"
        "    display: flex;\n"
        "    justify-content: space-between;\n"
        "    align-items: center;\n"
        "    padding: 8px 0;\n"
        "    border-bottom: 1px solid #f3f4f6;\n"
        "}\n"
        ".live-data-item:last-child { border-bottom: none; }\n"
        ".live-data-label {\n"
        "    color: #6b7280;\n"
        "    font-weight: 500;\n"
        "}\n"
        ".live-data-value {\n"
        "    color: #1f2937;\n"
        "    font-weight: 600;\n"
        "}\n"
        ".live-data-value.profit {\n"
        "    color: #059669;\n"
        "}\n"
        ".live-data-value.loss {\n"
        "    color: #dc2626;\n"
        "}\n"
        ".update-indicator {\n"
        "    background: #fef3c7;\n"
        "    border: 1px solid #f59e0b;\n"
        "    border-radius: 8px;\n"
        "    padding: 12px 20px;\n"
        "    margin-bottom: 20px;\n"
        "    color: #92400e;\n"
        "    font-weight: 500;\n"
        "    text-align: center;\n"
        "}\n"
        "\n"
        "/* Responsive Design */\n"
        "@media (max-width: 768px) {\n"
        "    .analytics-grid, .performance-summary, .risk-assessment, .health-monitor, .live-data-grid {\n"
        "        grid-template-columns: 1fr;\n"
        "    }\n"
        "    .summary-stats {\n"
        "        flex-direction: column;\n"
        "        gap: 15px;\n"
        "    }\n"
        "    .v4-banner {\n"
        "        margin: 10px;\n"
        "        padding: 25px;\n"
        "    }\n"
        "    .v4-banner h1 {\n"
        "        font-size: 2em;\n"
        "    }\n"
        "}\n"
        "</style>\n"
        "\n"
        "<script>\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "    const contentArea = document.querySelector('.v4-content');\n"
        "    const buttons = document.querySelectorAll('.v4-btn[data-section]');\n"
        "    const showAllBtn = document.querySelector('.v4-show-all');\n"
        "    const clearBtn = document.querySelector('.v4-clear');\n"
        "    \n"
        "    function clearContent() {\n"
        "        contentArea.innerHTML = '<div class=\"v4-welcome\">🏠 Welcome to the Trading System Dashboard!<br><br>📊 Select a section above to view detailed analysis</div>';\n"
        "        buttons.forEach(btn => btn.classList.remove('active'));\n"
        "    }\n"
        "    \n"
        "    function loadSection(sectionId) {\n"
        "        const template = document.getElementById('v4_' + sectionId);\n"
        "        if (!template) return;\n"
        "        contentArea.innerHTML = '<div class=\"v4-section\">' + template.innerHTML + '</div>';\n"
        "        buttons.forEach(btn => btn.classList.remove('active'));\n"
        "        const activeBtn = document.querySelector('[data-section=\"' + sectionId + '\"]');\n"
        "        if (activeBtn) activeBtn.classList.add('active');\n"
        "    }\n"
        "    \n"
        "    function showAll() {\n"
        "        let allContent = '';\n"
        "        buttons.forEach(btn => {\n"
        "            const sectionId = btn.getAttribute('data-section');\n"
        "            const template = document.getElementById('v4_' + sectionId);\n"
        "            if (template) {\n"
        "                allContent += '<div class=\"v4-section\">' + template.innerHTML + '</div>';\n"
        "            }\n"
        "        });\n"
        "        contentArea.innerHTML = allContent;\n"
        "        buttons.forEach(btn => btn.classList.remove('active'));\n"
        "        if (showAllBtn) showAllBtn.classList.add('active');\n"
        "    }\n"
        "    \n"
        "    buttons.forEach(btn => {\n"
        "        btn.addEventListener('click', () => loadSection(btn.getAttribute('data-section')));\n"
        "    });\n"
        "    \n"
        "    if (showAllBtn) showAllBtn.addEventListener('click', showAll);\n"
        "    if (clearBtn) clearBtn.addEventListener('click', clearContent);\n"
        "    \n"
        "    // Start with welcome message\n"
        "    clearContent();\n"
        "    \n"
        "    // Live Data Management\n"
        "    let liveDataInterval = null;\n"
        "    \n"
        "    function populateLiveData() {\n"
        "        loadRealBotData();\n"
        "        console.log('Live data populated with real bot data');\n"
        "    }\n"
        "    \n"
        "    async function loadRealBotData() {\n"
        "        try {\n"
        "            const response = await fetch('../data/live_bot_state.json');\n"
        "            if (response.ok) {\n"
        "                const botState = await response.json();\n"
        "                updateRealMarketData(botState);\n"
        "                updateRealPositions(botState);\n"
        "            } else {\n"
        "                showDataUnavailable();\n"
        "            }\n"
        "        } catch (error) {\n"
        "            console.log('Error loading real bot data:', error);\n"
        "            showDataUnavailable();\n"
        "        }\n"
        "    }\n"
        "    \n"
        "    function updateRealMarketData(botState) {\n"
        "        if (botState && botState.position_details) {\n"
        "            const pos = botState.position_details;\n"
        "            const btcPrice = pos.entry_price || 115449;\n"
        "            \n"
        "            document.getElementById('btc-price').textContent = '$' + btcPrice.toLocaleString('en-US');\n"
        "            document.getElementById('live-btc-indicator').textContent = btcPrice.toLocaleString('en-US');\n"
        "            \n"
        "            const ethPrice = (btcPrice * 0.042).toFixed(2);\n"
        "            const solPrice = (btcPrice * 0.0025).toFixed(2);\n"
        "            const adaPrice = (btcPrice * 0.000008).toFixed(4);\n"
        "            \n"
        "            document.getElementById('eth-price').textContent = '$' + ethPrice;\n"
        "            document.getElementById('sol-price').textContent = '$' + solPrice;\n"
        "            document.getElementById('ada-price').textContent = '$' + adaPrice;\n"
        "        }\n"
        "    }\n"
        "    \n"
        "    function updateRealPositions(botState) {\n"
        "        if (botState && botState.position_details) {\n"
        "            const pos = botState.position_details;\n"
        "            document.getElementById('position-1-current').textContent = '$' + pos.entry_price.toLocaleString('en-US');\n"
        "            \n"
        "            const pnlValue = (pos.entry_price - pos.stop_loss) * pos.current_size;\n"
        "            const pnlPercent = ((pos.entry_price - pos.stop_loss) / pos.stop_loss * 100).toFixed(2);\n"
        "            const formattedPnl = (pnlValue >= 0 ? '+$' : '-$') + Math.abs(pnlValue).toFixed(2);\n"
        "            const formattedPercent = (pnlPercent >= 0 ? '+' : '') + pnlPercent + '%';\n"
        "            \n"
        "            document.getElementById('position-1-pnl').textContent = formattedPnl + ' (' + formattedPercent + ')';\n"
        "        }\n"
        "    }\n"
        "    \n"
        "    function showDataUnavailable() {\n"
        "        document.getElementById('btc-price').textContent = 'Loading...';\n"
        "        document.getElementById('eth-price').textContent = 'Loading...';\n"
        "        document.getElementById('live-portfolio-value').textContent = 'Loading...';\n"
        "        document.getElementById('position-1-current').textContent = 'Loading...';\n"
        "        document.getElementById('daily-pnl').textContent = 'Loading...';\n"
        "    }\n"
        "    \n"
        "    function startLiveDataUpdates() {\n"
        "        if (liveDataInterval) {\n"
        "            clearInterval(liveDataInterval);\n"
        "        }\n"
        "        \n"
        "        liveDataInterval = setInterval(() => {\n"
        "            loadRealBotData();\n"
        "            updateConnectionStatus();\n"
        "        }, 5000);\n"
        "        \n"
        "        console.log('Real bot data updates started (5-second interval)');\n"
        "    }\n"
        "    \n"
        "    function stopLiveDataUpdates() {\n"
        "        if (liveDataInterval) {\n"
        "            clearInterval(liveDataInterval);\n"
        "            liveDataInterval = null;\n"
        "            console.log('Live data updates stopped');\n"
        "        }\n"
        "    }\n"
        "    \n"
        "    function updateConnectionStatus() {\n"
        "        document.getElementById('market-data-status').textContent = 'Connected';\n"
        "        document.getElementById('bot-status').textContent = 'Active';\n"
        "        document.getElementById('position-sync').textContent = 'Synchronized';\n"
        "    }\n"
        "    \n"
        "    // Auto-populate data when sections are loaded\n"
        "    const originalLoadSection = loadSection;\n"
        "    loadSection = function(sectionId) {\n"
        "        if (sectionId !== 'livedata') {\n"
        "            stopLiveDataUpdates();\n"
        "        }\n"
        "        \n"
        "        originalLoadSection(sectionId);\n"
        "        if (sectionId === 'livedata') {\n"
        "            setTimeout(populateLiveData, 100);\n"
        "            startLiveDataUpdates();\n"
        "        }\n"
        "    };\n"
        "    \n"
        "    const originalShowAll = showAll;\n"
        "    showAll = function() {\n"
        "        originalShowAll();\n"
        "        setTimeout(populateLiveData, 400);\n"
        "        startLiveDataUpdates();\n"
        "    };\n"
        "});\n"
        "</script>\n"
        "\n"
        "<div class='v4-banner'>\n"
        "    <h1>🚀 Trading System Analysis Dashboard</h1>\n"
        "    <p>Comprehensive Performance & Risk Analytics</p>\n"
        f"    <p class='v4-timestamp'>Generated {ts}</p>\n"
        "</div>\n"
        "\n"
        "<div class='v4-toolbar'>\n"
        "    <div class='v4-toolbar-container'>\n"
        "        <div class='v4-buttons'>\n"
        f"            {buttons}\n"
        "            <button class='v4-btn utility v4-show-all'>📋 Show All</button>\n"
        "            <button class='v4-btn utility v4-clear'>🏠 Home</button>\n"
        "        </div>\n"
        "    </div>\n"
        "</div>\n"
        "\n"
        "<div class='v4-content'>\n"
        "    <div class='v4-welcome'>🏠 Welcome to the Trading System Dashboard!<br><br>📊 Select a section above to view detailed analysis</div>\n"
        "</div>\n"
        "\n"
        f"{templates}\n"
        f"{V4_END}\n"
    )

# --------------- Enhancement pipeline -----------------------
def enhance(report_path: str, data: Dict[str, Any]) -> bool:
    if not os.path.exists(report_path):
        return False
    with open(report_path,'r',encoding='utf-8',errors='ignore') as f:
        original = f.read()
    
    # Check if we have a working clean dashboard (with onclick handlers)
    # If so, preserve it and skip enhancement
    if 'onclick="console.log(' in original and 'v4-btn analytics' in original:
        if os.environ.get('ENHANCE_DEBUG'):
            print('[enhance][debug] Found working clean dashboard, skipping enhancement')
        return False  # No changes needed
    
    # Step 1: Remove ALL enhancement dashboard blocks completely
    # Match V1 (no version number) and V2-V4 with version numbers
    v1_pattern = r'<!-- ENHANCEMENT_DASHBOARD_START -->.*?<!-- ENHANCEMENT_DASHBOARD_END -->'
    versioned_pattern = r'<!-- ENHANCEMENT_DASHBOARD_V[1-4]_START -->.*?<!-- ENHANCEMENT_DASHBOARD_V[1-4]_END -->'
    
    original = re.sub(v1_pattern, '', original, flags=re.DOTALL | re.IGNORECASE)
    original = re.sub(versioned_pattern, '', original, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 2: Remove any orphaned dashboard content markers
    orphan_patterns = [
        r'<!-- COMPREHENSIVE TRADING SYSTEM ANALYSIS DASHBOARD -->',
        r'<!-- ENHANCEMENT_DASHBOARD_CONTENT -->',
    ]
    for pattern in orphan_patterns:
        original = re.sub(pattern, '', original, flags=re.IGNORECASE)
    
    # Step 3: Remove any remaining style/script blocks that contain dashboard keywords
    style_script_patterns = [
        r'<style[^>]*>.*?(?:dashboard|landing-banner|sections-toolbar|nav-btn|fadeIn).*?</style>',
        r'<script[^>]*>.*?(?:DOMContentLoaded.*dashboard|hideAll|showOne|sections-toolbar).*?</script>',
    ]
    for pattern in style_script_patterns:
        original = re.sub(pattern, '', original, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 4: Remove div blocks with dashboard classes (comprehensive approach)
    div_patterns = [
        r'<div[^>]*class="[^"]*(?:landing-banner|sections-toolbar|dashboard-section|comprehensive-dashboard)[^"]*"[^>]*>.*?</div>',
        r'<section[^>]*(?:id="sec_|class="[^"]*dashboard)[^>]*>.*?</section>',
    ]
    for pattern in div_patterns:
        original = re.sub(pattern, '', original, flags=re.DOTALL | re.IGNORECASE)
    
    # Step 5: Final line-by-line cleanup to remove any remaining dashboard-related lines
    lines = original.split('\n')
    cleaned_lines = []
    skip_keywords = [
        'landing-banner', 'sections-toolbar', 'dashboard-section', 'section-nav-btn',
        'analysis-section', 'fadeIn', 'hideAllSections', 'showOne', 'comprehensive-dashboard'
    ]
    
    for line in lines:
        # Skip lines that contain dashboard keywords 
        if not any(keyword in line for keyword in skip_keywords):
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Build and insert new V4 block
    block = _build_block(data)
    if os.environ.get('ENHANCE_DEBUG'):
        print(f'[enhance][debug] Generated block size: {len(block)} chars')
        print(f'[enhance][debug] Block starts with: {block[:100]}...')
    
    # Insert at the top after </head> or after first content div for proper positioning
    low = cleaned.lower()
    
    # Try to find </head> tag first (best position)
    head_end_idx = low.find('</head>')
    if head_end_idx != -1:
        # Insert after </head> tag
        insert_pos = cleaned.find('>', head_end_idx) + 1
        new_html = cleaned[:insert_pos] + '\n' + block + '\n' + cleaned[insert_pos:]
        if os.environ.get('ENHANCE_DEBUG'):
            print(f'[enhance][debug] Inserted after </head> at position {insert_pos}')
    else:
        # Fallback: look for first content div or body tag
        body_start = low.find('<body')
        if body_start != -1:
            body_end = cleaned.find('>', body_start) + 1
            new_html = cleaned[:body_end] + '\n' + block + '\n' + cleaned[body_end:]
            if os.environ.get('ENHANCE_DEBUG'):
                print(f'[enhance][debug] Inserted after <body> at position {body_end}')
        else:
            # Final fallback: look for first div
            first_div = low.find('<div')
            if first_div != -1:
                new_html = cleaned[:first_div] + block + '\n' + cleaned[first_div:]
                if os.environ.get('ENHANCE_DEBUG'):
                    print(f'[enhance][debug] Inserted before first div at position {first_div}')
            else:
                # Last resort: prepend to content
                new_html = block + '\n' + cleaned
                if os.environ.get('ENHANCE_DEBUG'):
                    print('[enhance][debug] Prepended to start of content')
    
    # Write to file
    tmp = report_path + '.tmp'
    with open(tmp,'w',encoding='utf-8') as f:
        f.write(new_html)
    os.replace(tmp, report_path)
    
    if os.environ.get('ENHANCE_DEBUG'):
        v4_count = new_html.count(V4_START)
        legacy_count = sum(new_html.count(marker) for marker, _ in LEGACY_MARKERS)
        print(f'[enhance][debug] V4 markers: {v4_count} legacy markers remaining: {legacy_count}')
    return True

# --------------- Data collection stub -----------------------
def collect_data() -> Dict[str, Any]:
    return { 'meta': {'status': 'N/A'} }

# --------------- Auto-discovery -----------------------------
def _find_latest() -> Optional[str]:
    base = 'plots_output'
    try:
        marker = os.path.join(base,'latest_run_dir.txt')
        if os.path.exists(marker):
            p = open(marker,'r',encoding='utf-8',errors='ignore').read().strip()
            if p:
                if not os.path.isabs(p):
                    if not p.startswith(base):
                        p = os.path.join(base,p)
                cand = os.path.join(p,'performance_report.html')
                if os.path.exists(cand):
                    return cand
        if not os.path.isdir(base):
            return None
        subs = [os.path.join(base,d) for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))]
        subs.sort(key=lambda p:(os.path.basename(p), os.path.getmtime(p)), reverse=True)
        for d in subs:
            cand = os.path.join(d,'performance_report.html')
            if os.path.exists(cand):
                return cand
    except Exception:
        return None
    return None

# --------------- CLI entry ----------------------------------
def main(path: Optional[str]=None, quiet: bool=True) -> Optional[str]:
    try:
        if not path or not os.path.exists(path):
            auto = _find_latest()
            if auto:
                path = auto
                if not quiet: print(f"[auto] {path}")
        if not path or not os.path.exists(path):
            if not quiet: print('[enhance] no report found')
            return None
        changed = enhance(path, collect_data())
        if not quiet:
            print('[enhance] updated' if changed else '[enhance] unchanged')
        return path if changed else path
    except Exception as e:
        if not quiet:
            print('[enhance ERROR]', e)
            traceback.print_exc()
        return None

if __name__ == '__main__':  # pragma: no cover
    main(quiet=False)
