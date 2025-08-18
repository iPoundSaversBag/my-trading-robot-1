#!/usr/bin/env python3
"""
Remove duplicate quick-stat sections from QuantStats report
"""
import re
import os

def remove_duplicate_quick_stats(file_path: str) -> bool:
    """Remove duplicate quick-stat sections while keeping the first occurrence"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    
    # Find all quick-stat container blocks that contain Total Trades
    pattern = r'<div[^>]*class="[^"]*quick-stat[^"]*"[^>]*>.*?Total Trades.*?</div>'
    matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
    
    print(f"Found {len(matches)} quick-stat blocks with Total Trades")
    
    if len(matches) <= 1:
        print("No duplicates to remove")
        return True
    
    # Find the parent containers of these quick-stats to remove entire duplicate sections
    quick_stat_containers = []
    for match in matches:
        # Look backwards to find the parent container
        start_search = max(0, match.start() - 500)
        text_before = content[start_search:match.start()]
        
        # Look for container divs before this quick-stat
        container_matches = list(re.finditer(r'<div[^>]*>', text_before))
        if container_matches:
            # Find the most recent container div
            last_container = container_matches[-1]
            container_start = start_search + last_container.start()
            
            # Find the end of this container section
            container_end = match.end()
            # Look ahead to find more quick-stats in the same container
            remaining_content = content[match.end():match.end() + 500]
            more_quick_stats = re.search(r'(.*?</div>\s*</div>)', remaining_content, re.DOTALL)
            if more_quick_stats:
                container_end = match.end() + more_quick_stats.end()
            
            quick_stat_containers.append((container_start, container_end))
    
    print(f"Found {len(quick_stat_containers)} quick-stat containers")
    
    # Remove all but the first container
    new_content = content
    removed_total = 0
    
    for i, (start, end) in enumerate(reversed(quick_stat_containers[1:])):
        print(f"Removing container {len(quick_stat_containers) - i - 1}: positions {start}-{end}")
        section_text = new_content[start:end]
        print(f"Preview: {section_text[:100]}...")
        
        new_content = new_content[:start] + new_content[end:]
        removed_total += (end - start)
    
    print(f"Total removed: {removed_total} characters")
    print(f"Final file size: {len(new_content)} chars")
    
    # Verify the fix
    final_matches = len(re.findall(r'Total Trades', new_content, re.IGNORECASE))
    print(f"Remaining 'Total Trades' occurrences: {final_matches}")
    
    # Create backup and save
    backup_path = file_path + '.dedup_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(new_content)
        print(f"✅ Duplicate quick-stats removal complete!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if remove_duplicate_quick_stats(file_path):
        print("✅ Success! Removed duplicate quick-stat sections.")
    else:
        print("❌ Failed to remove duplicates.")
