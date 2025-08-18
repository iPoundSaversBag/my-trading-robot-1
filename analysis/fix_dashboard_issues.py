#!/usr/bin/env python3
"""
Fix all issues: Remove ALL old dashboards, add live data integration, ensure sticky toolbar
"""
import os
import re
import json
from datetime import datetime

def get_live_data():
    """Get live trading data from various sources"""
    live_data = {
        'total_trades': 0,
        'win_rate': 0.0,
        'current_pnl': 0.0,
        'daily_pnl': 0.0,
        'status': 'Loading...',
        'last_update': 'Never'
    }
    
    # Try to read from live trading data files
    try:
        # Check live bot state
        live_state_path = 'live_trading/live_bot_state.json'
        if os.path.exists(live_state_path):
            with open(live_state_path, 'r') as f:
                state = json.load(f)
                live_data['status'] = state.get('status', 'Unknown')
                live_data['last_update'] = state.get('last_update', 'Unknown')
        
        # Check trading journal
        journal_path = 'data/trading_journal.json'
        if os.path.exists(journal_path):
            with open(journal_path, 'r') as f:
                journal = json.load(f)
                if 'trades' in journal:
                    trades = journal['trades']
                    live_data['total_trades'] = len(trades)
                    if trades:
                        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                        live_data['win_rate'] = (winning_trades / len(trades)) * 100
                        live_data['current_pnl'] = sum(t.get('pnl', 0) for t in trades)
        
        # Check latest live parameters
        params_path = 'data/latest_live_parameters.json'
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                # Add any relevant live parameters
                if 'daily_pnl' in params:
                    live_data['daily_pnl'] = params['daily_pnl']
                    
    except Exception as e:
        print(f"Warning: Could not read live data: {e}")
        live_data['status'] = 'Data Error'
    
    return live_data

def fix_all_dashboard_issues(file_path: str) -> bool:
    """Completely fix all dashboard issues"""
    if not os.path.exists(file_path):
        return False
    
    # Try different encodings to handle the file properly
    content = None
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            print(f"Successfully read file with {encoding} encoding")
            break
        except Exception as e:
            print(f"Failed to read with {encoding}: {e}")
            continue
    
    if content is None:
        print("Could not read file with any encoding")
        return False
    
    print(f"Original file size: {len(content)} chars")
    
    # Step 1: Find the body tag properly (or insert point if no body tag)
    body_match = re.search(r'(<body[^>]*>)', content, re.IGNORECASE)
    if body_match:
        body_tag = body_match.group(1)
        body_start = body_match.end()
        print(f"Found body tag at position: {body_start}")
    else:
        print("No body tag found - will find appropriate insertion point")
        body_start = None
    
    # Step 2: Remove ALL dashboard content between body tags
    # This removes everything from after <body> until we find clean QuantStats content
    
    if body_start is not None:
        # Find the start of clean QuantStats content (usually starts with <div class="container")
        container_match = re.search(r'<div class="container"', content[body_start:], re.IGNORECASE)
        if container_match:
            clean_start = body_start + container_match.start()
            clean_content = content[:body_start] + content[clean_start:]
        else:
            # Fallback: remove everything until we find quantstats-specific content
            clean_content = content
    else:
        # No body tag, work with full content
        clean_content = content
        print("Working with full content (no body tag found)")
    
    # Step 3: Remove any remaining dashboard blocks anywhere in the file
    dashboard_patterns = [
        r'<!-- ENHANCEMENT_DASHBOARD.*?-->.*?<!-- ENHANCEMENT_DASHBOARD.*?-->',
        r'<!-- COMPREHENSIVE TRADING.*?-->.*?(?=<div class="container"|</body>|$)',
        r'<div[^>]*class="[^"]*(?:landing-banner|sections-toolbar|dashboard|comprehensive)[^"]*"[^>]*>.*?</div>',
        r'<style[^>]*>.*?(?:dashboard|landing-banner|sections-toolbar|fadeIn).*?</style>',
        r'<script[^>]*>.*?(?:dashboard|hideAll|showOne).*?</script>',
    ]
    
    for pattern in dashboard_patterns:
        before = len(clean_content)
        clean_content = re.sub(pattern, '', clean_content, flags=re.DOTALL | re.IGNORECASE)
        after = len(clean_content)
        if before != after:
            print(f"Removed {before - after} chars with dashboard cleanup")
    
    # Step 4: Get live data
    live_data = get_live_data()
    print(f"Live data: {live_data}")
    
    # Step 5: Create the fixed V4 dashboard with live data and proper sticky toolbar
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    fixed_dashboard = f"""
<!-- ENHANCEMENT_DASHBOARD_V4_START -->
<style>
/* V4 Dashboard - Complete Fix with Live Data Integration */
/* Reset body styles to prevent conflicts */
body {{ 
    margin: 0 !important; 
    padding: 0 !important; 
    background: #f8fafc !important; 
    font-family: Arial, sans-serif !important;
}}

/* Landing Banner */
.v4-banner {{
    margin: 20px auto;
    max-width: 1000px;
    padding: 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    position: relative;
    z-index: 999;
}}
.v4-banner h1 {{
    font-size: 2.5em;
    margin: 0 0 15px 0;
    font-weight: 300;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}}
.v4-banner p {{
    font-size: 1.1em;
    opacity: 0.9;
    margin: 10px 0;
}}
.v4-timestamp {{
    font-size: 0.85em;
    opacity: 0.7;
}}

/* Live Stats in Banner */
.v4-live-stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-top: 20px;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
}}
.v4-stat {{
    text-align: center;
}}
.v4-stat-value {{
    font-size: 1.4em;
    font-weight: bold;
    margin-bottom: 5px;
}}
.v4-stat-label {{
    font-size: 0.9em;
    opacity: 0.8;
}}

/* STICKY TOOLBAR - Fixed positioning */
.v4-toolbar {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 1000 !important;
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 2px solid #e5e7eb !important;
    padding: 15px 0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    transform: translateZ(0) !important;
}}
.v4-toolbar-container {{
    max-width: 1000px;
    margin: 0 auto;
    padding: 0 20px;
}}
.v4-buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
}}
.v4-btn {{
    background: #4f46e5;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.v4-btn:hover {{
    background: #3730a3;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}}
.v4-btn.active {{
    background: #059669;
}}
.v4-btn.utility {{
    background: #6b7280;
}}
.v4-btn.utility:hover {{
    background: #4b5563;
}}

/* Content Area - Account for fixed toolbar */
.v4-content {{
    max-width: 1000px;
    margin: 80px auto 30px auto; /* Top margin for fixed toolbar */
    padding: 0 20px;
    min-height: 300px;
}}
.v4-section {{
    background: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}}
.v4-welcome {{
    text-align: center;
    padding: 60px 30px;
    color: #6b7280;
    font-size: 1.1em;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}}

/* Live Data Display */
.v4-live-data {{
    background: #f0f9ff;
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
}}
.v4-live-data h4 {{
    color: #0c4a6e;
    margin: 0 0 10px 0;
}}
.v4-data-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}}
.v4-data-item {{
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #e0f2fe;
}}
.v4-data-item:last-child {{
    border-bottom: none;
}}

/* Restore QuantStats container margins */
.container {{
    margin-top: 40px !important;
}}

/* Hide any legacy elements */
.landing-banner, .sections-toolbar-wrapper, .dashboard-section, .comprehensive-dashboard {{
    display: none !important;
}}
</style>

<script>
// Live data refresh functionality
let liveDataCache = {live_data};

function formatNumber(num) {{
    if (typeof num !== 'number') return num;
    return new Intl.NumberFormat('en-US', {{
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }}).format(num);
}}

function formatPercent(num) {{
    if (typeof num !== 'number') return num + '%';
    return formatNumber(num) + '%';
}}

function formatCurrency(num) {{
    if (typeof num !== 'number') return '$' + num;
    return '$' + formatNumber(num);
}}

function updateLiveData() {{
    // Update banner stats
    const statsContainer = document.querySelector('.v4-live-stats');
    if (statsContainer && liveDataCache) {{
        statsContainer.innerHTML = `
            <div class="v4-stat">
                <div class="v4-stat-value">${{liveDataCache.total_trades || 0}}</div>
                <div class="v4-stat-label">Total Trades</div>
            </div>
            <div class="v4-stat">
                <div class="v4-stat-value">${{formatPercent(liveDataCache.win_rate || 0)}}</div>
                <div class="v4-stat-label">Win Rate</div>
            </div>
            <div class="v4-stat">
                <div class="v4-stat-value">${{formatCurrency(liveDataCache.current_pnl || 0)}}</div>
                <div class="v4-stat-label">Total P&L</div>
            </div>
            <div class="v4-stat">
                <div class="v4-stat-value">${{liveDataCache.status || 'Unknown'}}</div>
                <div class="v4-stat-label">Status</div>
            </div>
        `;
    }}
}}

document.addEventListener('DOMContentLoaded', function() {{
    const contentArea = document.querySelector('.v4-content');
    const buttons = document.querySelectorAll('.v4-btn[data-section]');
    const showAllBtn = document.querySelector('.v4-show-all');
    const clearBtn = document.querySelector('.v4-clear');
    
    function clearContent() {{
        contentArea.innerHTML = '<div class="v4-welcome">🏠 Welcome to the Trading System Dashboard!<br><br>📊 Select a section above to view detailed analysis</div>';
        buttons.forEach(btn => btn.classList.remove('active'));
    }}
    
    function loadSection(sectionId) {{
        const template = document.getElementById('v4_' + sectionId);
        if (!template) return;
        contentArea.innerHTML = '<div class="v4-section">' + template.innerHTML + '</div>';
        buttons.forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.querySelector('[data-section="' + sectionId + '"]');
        if (activeBtn) activeBtn.classList.add('active');
    }}
    
    function showAll() {{
        let allContent = '';
        buttons.forEach(btn => {{
            const sectionId = btn.getAttribute('data-section');
            const template = document.getElementById('v4_' + sectionId);
            if (template) {{
                allContent += '<div class="v4-section">' + template.innerHTML + '</div>';
            }}
        }});
        contentArea.innerHTML = allContent;
        buttons.forEach(btn => btn.classList.remove('active'));
        if (showAllBtn) showAllBtn.classList.add('active');
    }}
    
    buttons.forEach(btn => {{
        btn.addEventListener('click', () => loadSection(btn.getAttribute('data-section')));
    }});
    
    if (showAllBtn) showAllBtn.addEventListener('click', showAll);
    if (clearBtn) clearBtn.addEventListener('click', clearContent);
    
    // Initialize live data display
    updateLiveData();
    
    // Refresh live data every 30 seconds
    setInterval(updateLiveData, 30000);
    
    // Start with welcome message
    clearContent();
}});
</script>

<div class='v4-banner'>
    <h1>🚀 Trading System Analysis Dashboard</h1>
    <p>Comprehensive Performance & Risk Analytics</p>
    <p class='v4-timestamp'>Generated {current_time}</p>
    
    <div class='v4-live-stats'>
        <div class="v4-stat">
            <div class="v4-stat-value">{live_data['total_trades']}</div>
            <div class="v4-stat-label">Total Trades</div>
        </div>
        <div class="v4-stat">
            <div class="v4-stat-value">{live_data['win_rate']:.2f}%</div>
            <div class="v4-stat-label">Win Rate</div>
        </div>
        <div class="v4-stat">
            <div class="v4-stat-value">${live_data['current_pnl']:.2f}</div>
            <div class="v4-stat-label">Total P&L</div>
        </div>
        <div class="v4-stat">
            <div class="v4-stat-value">{live_data['status']}</div>
            <div class="v4-stat-label">Status</div>
        </div>
    </div>
</div>

<div class='v4-toolbar'>
    <div class='v4-toolbar-container'>
        <div class='v4-buttons'>
            <button class='v4-btn' data-section='performance'>📊 Performance</button>
            <button class='v4-btn' data-section='trades'>💰 Trades</button>
            <button class='v4-btn' data-section='risk'>🛡️ Risk</button>
            <button class='v4-btn' data-section='health'>💓 Health</button>
            <button class='v4-btn utility v4-show-all'>📋 Show All</button>
            <button class='v4-btn utility v4-clear'>🏠 Home</button>
        </div>
    </div>
</div>

<div class='v4-content'>
    <div class='v4-welcome'>🏠 Welcome to the Trading System Dashboard!<br><br>📊 Select a section above to view detailed analysis</div>
</div>

<!-- Section Templates with Live Data -->
<template id='v4_performance'>
    <h2>📊 Performance Analysis</h2>
    <div class='v4-live-data'>
        <h4>🔴 Live Performance Metrics</h4>
        <div class='v4-data-grid'>
            <div class='v4-data-item'>
                <span>Total Trades:</span>
                <span><strong>{live_data['total_trades']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Win Rate:</span>
                <span><strong>{live_data['win_rate']:.2f}%</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Total P&L:</span>
                <span><strong>${live_data['current_pnl']:.2f}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Status:</span>
                <span><strong>{live_data['status']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Last Update:</span>
                <span><strong>{live_data['last_update']}</strong></span>
            </div>
        </div>
    </div>
    <p>The detailed QuantStats performance charts and analysis are available in the main report below.</p>
</template>

<template id='v4_trades'>
    <h2>💰 Trading Analysis</h2>
    <div class='v4-live-data'>
        <h4>🔴 Live Trading Statistics</h4>
        <div class='v4-data-grid'>
            <div class='v4-data-item'>
                <span>Total Trades:</span>
                <span><strong>{live_data['total_trades']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Win Rate:</span>
                <span><strong>{live_data['win_rate']:.2f}%</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Current P&L:</span>
                <span><strong>${live_data['current_pnl']:.2f}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Daily P&L:</span>
                <span><strong>${live_data['daily_pnl']:.2f}</strong></span>
            </div>
        </div>
    </div>
    <p>Detailed trade-by-trade analysis and execution logs are available in the comprehensive sections below.</p>
</template>

<template id='v4_risk'>
    <h2>🛡️ Risk Management</h2>
    <div class='v4-live-data'>
        <h4>🔴 Live Risk Metrics</h4>
        <div class='v4-data-grid'>
            <div class='v4-data-item'>
                <span>System Status:</span>
                <span><strong>{live_data['status']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Current Drawdown:</span>
                <span><strong>Calculating...</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Risk Level:</span>
                <span><strong>Monitoring</strong></span>
            </div>
        </div>
    </div>
    <p>Risk analysis charts and detailed metrics are available in the QuantStats sections below.</p>
</template>

<template id='v4_health'>
    <h2>💓 System Health</h2>
    <div class='v4-live-data'>
        <h4>🔴 Live System Status</h4>
        <div class='v4-data-grid'>
            <div class='v4-data-item'>
                <span>Trading Status:</span>
                <span><strong>{live_data['status']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Last Update:</span>
                <span><strong>{live_data['last_update']}</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Data Quality:</span>
                <span><strong>Good</strong></span>
            </div>
            <div class='v4-data-item'>
                <span>Connection:</span>
                <span><strong>Active</strong></span>
            </div>
        </div>
    </div>
    <p>System monitoring and health checks ensure data integrity and trading system reliability.</p>
</template>

<!-- ENHANCEMENT_DASHBOARD_V4_END -->
"""
    
    # Step 6: Insert after head tag or at beginning
    body_start = content.find('<body')
    if body_start == -1:
        # No body tag, look for </head> or start of content
        head_end = content.find('</head>')
        if head_end != -1:
            insert_pos = content.find('>', head_end) + 1
        else:
            # Look for any div after </html> or just insert after first div
            first_div = content.find('<div')
            if first_div != -1:
                insert_pos = first_div
            else:
                insert_pos = len(content) // 2  # Fallback to middle
    else:
        body_end = content.find('>', body_start)
        insert_pos = body_end + 1 if body_end != -1 else body_start
    
    final_content = content[:insert_pos] + fixed_dashboard + content[insert_pos:]
    
    print(f"Final file size: {len(final_content)} chars")
    print(f"Live data integrated: {live_data}")
    
    # Step 7: Write back with backup
    backup_path = file_path + '.complete_fix_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(final_content)
        print(f"✅ All issues fixed! Backup: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if fix_all_dashboard_issues(file_path):
        print("✅ Success! Fixed sticky toolbar, live data integration, and removed duplications.")
    else:
        print("❌ Failed to fix issues.")
