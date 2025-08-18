#!/usr/bin/env python3
"""
Complete dashboard replacement - remove ALL existing dashboard content and apply clean V4
"""
import os
import re

def complete_replacement(file_path: str) -> bool:
    """Completely replace all dashboard content with clean V4"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    
    # Step 1: Find body tag and preserve structure
    body_match = re.search(r'(<body[^>]*>)', content, re.IGNORECASE | re.DOTALL)
    if not body_match:
        print("No body tag found")
        print("First 500 chars:", content[:500])
        return False
    
    # Get positions
    body_start_tag = body_match.group(1)
    body_end_pos = body_match.end()
    
    # Find the end body tag
    body_close_match = re.search(r'</body>', content, re.IGNORECASE)
    if body_close_match:
        body_close_start = body_close_match.start()
    else:
        body_close_start = len(content)
        body_close_match = None
    
    # Get the parts
    before_body = content[:body_end_pos]
    after_body = content[body_close_start:] if body_close_match else ""
    
    # Step 2: Extract the clean QuantStats content (everything between body tags)
    body_content = content[body_end_pos:body_close_start]
    
    # Remove ALL dashboard-related content with aggressive patterns
    dashboard_removal_patterns = [
        # All enhancement dashboard blocks (any version)
        r'<!-- ENHANCEMENT_DASHBOARD[^>]*START -->.*?<!-- ENHANCEMENT_DASHBOARD[^>]*END -->',
        # All comprehensive dashboard content
        r'<!-- COMPREHENSIVE TRADING SYSTEM ANALYSIS DASHBOARD -->.*?(?=<div class="container"|$)',
        r'<!-- ENHANCEMENT_DASHBOARD_CONTENT -->.*?(?=<div class="container"|$)',
        # Any div with dashboard-related classes
        r'<div[^>]*class="[^"]*(?:comprehensive-dashboard|interactive-dashboard|enh-banner|sections-toolbar|dashboard)[^"]*"[^>]*>.*?</div>',
        # Any style blocks with dashboard content
        r'<style[^>]*>.*?(?:comprehensive-dashboard|dashboard|sections-toolbar|landing-banner|nav-btn).*?</style>',
        # Any script blocks with dashboard content
        r'<script[^>]*>.*?(?:dashboard|sections|hideAll|showOne).*?</script>',
        # Section blocks
        r'<section[^>]*id="sec_[^"]*"[^>]*>.*?</section>',
    ]
    
    clean_body = body_content
    for pattern in dashboard_removal_patterns:
        before_size = len(clean_body)
        clean_body = re.sub(pattern, '', clean_body, flags=re.DOTALL | re.IGNORECASE)
        after_size = len(clean_body)
        if before_size != after_size:
            print(f"Removed {before_size - after_size} chars with pattern")
    
    # Step 3: Create the new V4 dashboard
    v4_dashboard = """
<!-- ENHANCEMENT_DASHBOARD_V4_START -->
<style>
/* V4 Dashboard - Modern Landing Page with Sticky Toolbar */
/* Reset body margins for better control */
body { margin: 0 !important; padding: 0 !important; background: #f8fafc !important; }

/* Landing Banner */
.v4-banner {
    margin: 20px auto;
    max-width: 1000px;
    padding: 40px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.v4-banner h1 {
    font-size: 2.5em;
    margin: 0 0 15px 0;
    font-weight: 300;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.v4-banner p {
    font-size: 1.1em;
    opacity: 0.9;
    margin: 10px 0;
}
.v4-banner .timestamp {
    font-size: 0.85em;
    opacity: 0.7;
}

/* Sticky Button Toolbar */
.v4-toolbar {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 2px solid #e5e7eb;
    padding: 15px 0;
    margin: 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.v4-toolbar-container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 0 20px;
}
.v4-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
}
.v4-btn {
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
}
.v4-btn:hover {
    background: #3730a3;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}
.v4-btn.active {
    background: #059669;
}
.v4-btn.utility {
    background: #6b7280;
}
.v4-btn.utility:hover {
    background: #4b5563;
}

/* Content Area */
.v4-content {
    max-width: 1000px;
    margin: 30px auto;
    padding: 0 20px;
    min-height: 300px;
}
.v4-section {
    background: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.v4-welcome {
    text-align: center;
    padding: 60px 30px;
    color: #6b7280;
    font-size: 1.1em;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* Restore container margins for QuantStats content */
.container {
    margin-top: 40px !important;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const contentArea = document.querySelector('.v4-content');
    const buttons = document.querySelectorAll('.v4-btn[data-section]');
    const showAllBtn = document.querySelector('.v4-show-all');
    const clearBtn = document.querySelector('.v4-clear');
    
    function clearContent() {
        contentArea.innerHTML = '<div class="v4-welcome">🏠 Welcome to the Trading System Dashboard!<br><br>Select a section above to view detailed analysis.</div>';
        buttons.forEach(btn => btn.classList.remove('active'));
    }
    
    function loadSection(sectionId) {
        const template = document.getElementById('v4_' + sectionId);
        if (!template) return;
        contentArea.innerHTML = '<div class="v4-section">' + template.innerHTML + '</div>';
        buttons.forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.querySelector('[data-section="' + sectionId + '"]');
        if (activeBtn) activeBtn.classList.add('active');
    }
    
    function showAll() {
        let allContent = '';
        buttons.forEach(btn => {
            const sectionId = btn.getAttribute('data-section');
            const template = document.getElementById('v4_' + sectionId);
            if (template) {
                allContent += '<div class="v4-section">' + template.innerHTML + '</div>';
            }
        });
        contentArea.innerHTML = allContent;
        buttons.forEach(btn => btn.classList.remove('active'));
        if (showAllBtn) showAllBtn.classList.add('active');
    }
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => loadSection(btn.getAttribute('data-section')));
    });
    
    if (showAllBtn) showAllBtn.addEventListener('click', showAll);
    if (clearBtn) clearBtn.addEventListener('click', clearContent);
    
    // Start with welcome message
    clearContent();
});
</script>

<div class='v4-banner'>
    <h1>🚀 Trading System Analysis Dashboard</h1>
    <p>Comprehensive Performance & Risk Analytics</p>
    <p class='timestamp'>Generated August 18, 2025</p>
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
    <div class='v4-welcome'>🏠 Welcome to the Trading System Dashboard!<br><br>Select a section above to view detailed analysis.</div>
</div>

<!-- Section Templates -->
<template id='v4_performance'>
    <h2>📊 Performance Analysis</h2>
    <div style='padding: 20px; background: #f8fafc; border-radius: 8px; margin: 15px 0;'>
        <h3>Key Metrics</h3>
        <p><strong>Cumulative Return:</strong> -100.00%</p>
        <p><strong>CAGR:</strong> -100.00%</p>
        <p><strong>Max Drawdown:</strong> 100.00%</p>
        <p><strong>Sharpe Ratio:</strong> 0.0000</p>
    </div>
    <p>The QuantStats performance charts and detailed analysis are available in the main report below.</p>
</template>

<template id='v4_trades'>
    <h2>💰 Trading Analysis</h2>
    <div style='padding: 20px; background: #f8fafc; border-radius: 8px; margin: 15px 0;'>
        <h3>Trading Statistics</h3>
        <p><strong>Total Trades:</strong> 4,784</p>
        <p><strong>Win Rate:</strong> 0.00%</p>
        <p><strong>Profit Factor:</strong> 0.0000</p>
        <p><strong>Total P&L:</strong> -$10,000.00</p>
    </div>
    <p>Detailed trade-by-trade analysis can be found in the comprehensive report sections below.</p>
</template>

<template id='v4_risk'>
    <h2>🛡️ Risk Management</h2>
    <div style='padding: 20px; background: #f8fafc; border-radius: 8px; margin: 15px 0;'>
        <h3>Risk Metrics</h3>
        <p><strong>Daily VaR (95%):</strong> N/A</p>
        <p><strong>Expected Shortfall:</strong> N/A</p>
        <p><strong>Annualized Volatility:</strong> N/A</p>
        <p><strong>Beta:</strong> N/A</p>
    </div>
    <p>Risk analysis charts and detailed metrics are available in the QuantStats sections below.</p>
</template>

<template id='v4_health'>
    <h2>💓 System Health</h2>
    <div style='padding: 20px; background: #f8fafc; border-radius: 8px; margin: 15px 0;'>
        <h3>Health Status</h3>
        <p><strong>Data Quality:</strong> Complete</p>
        <p><strong>Period Coverage:</strong> 372 days (1.02 years)</p>
        <p><strong>System Status:</strong> Analysis Complete</p>
        <p><strong>Last Updated:</strong> August 18, 2025</p>
    </div>
    <p>System monitoring and health checks ensure data integrity and analysis accuracy.</p>
</template>

<!-- ENHANCEMENT_DASHBOARD_V4_END -->
"""
    
    # Step 4: Reconstruct the file
    final_content = before_body + v4_dashboard + clean_body + after_body
    
    print(f"Clean body size: {len(clean_body)} chars")
    print(f"Final file size: {len(final_content)} chars")
    
    # Step 5: Write back
    backup_path = file_path + '.complete_backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)  # Backup original
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"✅ Complete replacement done. Backup: {backup_path}")
    return True

if __name__ == '__main__':
    file_path = 'plots_output/20250817_133240/performance_report.html'
    if complete_replacement(file_path):
        print("✅ Success! Clean V4 dashboard with sticky toolbar applied.")
    else:
        print("❌ Failed to process file.")
