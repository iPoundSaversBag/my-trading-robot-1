#!/usr/bin/env python3
"""
Remove duplicate V4 banner while preserving the first one
"""
import re
import os

def remove_duplicate_v4_banner(file_path: str) -> bool:
    """Remove duplicate V4 banner instances while keeping the first one"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    
    # Find all V4 banner blocks
    banner_pattern = r'<div class=[\'"]v4-banner[\'"]>.*?</div>'
    matches = list(re.finditer(banner_pattern, content, re.DOTALL))
    print(f"Found {len(matches)} V4 banner instances")
    
    if len(matches) <= 1:
        print("No duplicates to remove")
        return True
    
    # Remove all banners except the first one
    new_content = content
    removed_total = 0
    
    # Remove from the end to preserve position indices
    for i, match in enumerate(reversed(matches[1:])):
        start, end = match.span()
        removed_text = new_content[start:end]
        print(f"Removing banner {len(matches) - i - 1} at position {start}-{end}")
        print(f"Preview: {removed_text[:100]}...")
        
        new_content = new_content[:start] + new_content[end:]
        removed_total += (end - start)
    
    print(f"Total removed: {removed_total} characters")
    print(f"Final file size: {len(new_content)} chars")
    
    # Verify we still have exactly one banner
    final_banner_count = len(re.findall(banner_pattern, new_content, re.DOTALL))
    print(f"Final V4 banner count: {final_banner_count}")
    
    if final_banner_count != 1:
        print(f"❌ Error: Expected 1 banner, got {final_banner_count}")
        return False
    
    # Also check for duplicate toolbar instances
    toolbar_pattern = r'<div class=[\'"]v4-toolbar[\'"]>.*?</div>'
    toolbar_matches = list(re.finditer(toolbar_pattern, new_content, re.DOTALL))
    print(f"V4 toolbar instances: {len(toolbar_matches)}")
    
    if len(toolbar_matches) > 1:
        print("Also removing duplicate toolbars...")
        for match in reversed(toolbar_matches[1:]):
            start, end = match.span()
            new_content = new_content[:start] + new_content[end:]
            removed_total += (end - start)
    
    # Create backup and save
    backup_path = file_path + '.banner_fix_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(new_content)
        print(f"✅ Duplicate banner removal complete!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if remove_duplicate_v4_banner(file_path):
        print("✅ Success! Removed duplicate V4 banner.")
    else:
        print("❌ Failed to remove duplicates.")
