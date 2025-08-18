#!/usr/bin/env python3
"""
Check for duplicate V4 templates in the HTML file
"""
import re

def check_duplicates():
    file_path = '../plots_output/20250817_133240/performance_report.html'
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print("=== V4 DASHBOARD DUPLICATION ANALYSIS ===\n")
    
    # Check V4 markers
    v4_starts = [m.start() for m in re.finditer(r'ENHANCEMENT_DASHBOARD_V4_START', content)]
    v4_ends = [m.start() for m in re.finditer(r'ENHANCEMENT_DASHBOARD_V4_END', content)]
    print(f"V4 start markers: {len(v4_starts)} at positions {v4_starts}")
    print(f"V4 end markers: {len(v4_ends)} at positions {v4_ends}")
    
    # Check templates
    all_templates = re.findall(r"template id='([^']*)'", content)
    print(f"\nAll templates found: {all_templates}")
    
    # Check for duplicates
    from collections import Counter
    template_counts = Counter(all_templates)
    duplicates = {template: count for template, count in template_counts.items() if count > 1}
    
    if duplicates:
        print(f"\n❌ DUPLICATE TEMPLATES FOUND:")
        for template, count in duplicates.items():
            print(f"  {template}: {count} instances")
    else:
        print(f"\n✅ NO DUPLICATE TEMPLATES")
    
    # Check specific stat elements
    total_trades_count = len(re.findall(r'Total Trades', content, re.IGNORECASE))
    win_rate_count = len(re.findall(r'Win Rate', content, re.IGNORECASE))
    
    print(f"\nStat occurrences:")
    print(f"  Total Trades: {total_trades_count}")
    print(f"  Win Rate: {win_rate_count}")
    
    # If duplicates exist, find their positions
    if total_trades_count > 1:
        matches = list(re.finditer(r'Total Trades', content, re.IGNORECASE))
        print(f"\nTotal Trades positions:")
        for i, match in enumerate(matches):
            line_num = content[:match.start()].count('\n') + 1
            print(f"  Instance {i+1}: Line {line_num}, Position {match.start()}")
            
            # Find which V4 section this is in
            for j, start_pos in enumerate(v4_starts):
                if len(v4_ends) > j:
                    end_pos = v4_ends[j]
                    if start_pos <= match.start() <= end_pos:
                        print(f"    -> Inside V4 section {j+1} ({start_pos}-{end_pos})")
                        break
    
    return len(duplicates) == 0

if __name__ == '__main__':
    if check_duplicates():
        print("\n✅ No template duplications found")
    else:
        print("\n❌ Template duplications detected - cleanup needed")
