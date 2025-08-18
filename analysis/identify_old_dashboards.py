#!/usr/bin/env python3
"""
Identify old dashboard patterns in the HTML file
"""
import re

def identify_old_dashboards():
    file_path = '../plots_output/20250817_133240/performance_report.html'
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print("=== OLD DASHBOARD PATTERN ANALYSIS ===\n")
    
    # Define old dashboard patterns to look for
    old_patterns = {
        'V1 Dashboard Markers': [
            'ENHANCEMENT_DASHBOARD_START',
            'ENHANCEMENT_DASHBOARD_END'
        ],
        'V2/V3 Dashboard Markers': [
            'ENHANCEMENT_DASHBOARD_V2_START',
            'ENHANCEMENT_DASHBOARD_V3_START'
        ],
        'Old Dashboard Classes': [
            'dashboard-section',
            'comprehensive-dashboard',
            'dashboard-header',
            'dashboard-title',
            'landing-banner',
            'sections-toolbar'
        ],
        'Old Dashboard Comments': [
            'COMPREHENSIVE TRADING SYSTEM ANALYSIS DASHBOARD',
            'ENHANCEMENT_DASHBOARD_CONTENT',
            'INTERACTIVE DASHBOARD FEATURES'
        ],
        'Old Dashboard CSS': [
            'dashboard-section.hidden',
            'dashboard-section.active',
            'comprehensive-dashboard',
            'interactive-dashboard'
        ],
        'Old Dashboard JS': [
            'hideAllSections',
            'showOne',
            'dashboard-sections-container',
            'InteractiveDashboard'
        ]
    }
    
    total_old_remnants = 0
    
    for category, patterns in old_patterns.items():
        print(f"📂 {category}:")
        category_total = 0
        
        for pattern in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            if matches > 0:
                print(f"  ❌ {pattern}: {matches} occurrences")
                category_total += matches
            else:
                print(f"  ✅ {pattern}: 0 occurrences")
        
        total_old_remnants += category_total
        print(f"  📊 Category Total: {category_total}\n")
    
    print(f"🔍 TOTAL OLD DASHBOARD REMNANTS: {total_old_remnants}")
    print(f"📄 File Size: {len(content):,} characters")
    
    # Check for current V4 dashboard
    v4_start = content.count('ENHANCEMENT_DASHBOARD_V4_START')
    v4_end = content.count('ENHANCEMENT_DASHBOARD_V4_END')
    print(f"\n✅ Current V4 Dashboard: {v4_start} start markers, {v4_end} end markers")
    
    if total_old_remnants > 0:
        print(f"\n⚠️  FOUND {total_old_remnants} OLD DASHBOARD REMNANTS - CLEANUP NEEDED!")
        
        # Show specific line numbers for major remnants
        major_patterns = ['ENHANCEMENT_DASHBOARD_START', 'comprehensive-dashboard', 'dashboard-section']
        for pattern in major_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                print(f"\n📍 {pattern} locations:")
                for i, match in enumerate(matches[:5]):  # Show first 5 matches
                    line_num = content[:match.start()].count('\n') + 1
                    print(f"  Line {line_num}: Position {match.start()}")
                if len(matches) > 5:
                    print(f"  ... and {len(matches) - 5} more")
    else:
        print("\n✅ NO OLD DASHBOARD REMNANTS FOUND - FILE IS CLEAN!")

if __name__ == '__main__':
    identify_old_dashboards()
