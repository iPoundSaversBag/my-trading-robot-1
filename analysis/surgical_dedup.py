#!/usr/bin/env python3
"""
Surgically remove only the second duplicate quick-stat section
"""
import re
import os

def remove_second_duplicate_only(file_path: str) -> bool:
    """Remove only the second occurrence of Total Trades quick-stats"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    
    # Find specific positions of Total Trades occurrences
    matches = list(re.finditer(r'Total Trades', content, re.IGNORECASE))
    print(f"Found {len(matches)} 'Total Trades' occurrences at positions: {[m.start() for m in matches]}")
    
    if len(matches) <= 1:
        print("No duplicates to remove")
        return True
    
    # Target only the second occurrence around position 12914
    second_match = matches[1]
    
    # Look for the quick-stat div that contains this second occurrence
    # Search backwards to find the opening div
    search_start = max(0, second_match.start() - 200)
    search_text = content[search_start:second_match.end() + 200]
    
    # Find the quick-stat div pattern around the second occurrence
    relative_pos = second_match.start() - search_start
    
    # Look for the div containing this Total Trades
    pattern = r'<div class="quick-stat">\s*<span class="stat-value">[^<]*</span>\s*<span class="stat-label">Total Trades</span>\s*</div>'
    
    # Find all quick-stat divs in the content
    quick_stat_matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
    print(f"Found {len(quick_stat_matches)} quick-stat Total Trades divs")
    
    if len(quick_stat_matches) <= 1:
        print("Only one quick-stat div found, nothing to remove")
        return True
    
    # Remove only the second quick-stat div
    second_div = quick_stat_matches[1]
    
    # Also look for the associated Win Rate div that follows
    win_rate_start = second_div.end()
    win_rate_text = content[win_rate_start:win_rate_start + 200]
    win_rate_pattern = r'\s*<div class="quick-stat">\s*<span class="stat-value">[^<]*</span>\s*<span class="stat-label">Win Rate</span>\s*</div>'
    win_rate_match = re.match(win_rate_pattern, win_rate_text, re.DOTALL | re.IGNORECASE)
    
    if win_rate_match:
        # Remove both Total Trades and Win Rate divs
        remove_start = second_div.start()
        remove_end = win_rate_start + win_rate_match.end()
        print(f"Removing second Total Trades + Win Rate divs: positions {remove_start}-{remove_end}")
    else:
        # Remove just the Total Trades div
        remove_start = second_div.start()
        remove_end = second_div.end()
        print(f"Removing second Total Trades div only: positions {remove_start}-{remove_end}")
    
    removed_text = content[remove_start:remove_end]
    print(f"Removing: {removed_text}")
    
    new_content = content[:remove_start] + content[remove_end:]
    
    removed_chars = len(content) - len(new_content)
    print(f"Removed {removed_chars} characters")
    print(f"Final file size: {len(new_content)} chars")
    
    # Verify the fix
    final_total_trades = len(re.findall(r'Total Trades', new_content, re.IGNORECASE))
    final_win_rate = len(re.findall(r'Win Rate', new_content, re.IGNORECASE))
    print(f"Final counts - Total Trades: {final_total_trades}, Win Rate: {final_win_rate}")
    
    # Create backup and save
    backup_path = file_path + '.surgical_dedup_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(new_content)
        print(f"✅ Surgical duplicate removal complete!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if remove_second_duplicate_only(file_path):
        print("✅ Success! Surgically removed second duplicate only.")
    else:
        print("❌ Failed to remove duplicate.")
