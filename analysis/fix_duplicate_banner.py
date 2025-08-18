#!/usr/bin/env python3
"""
Fix duplicate Trading System Analysis Dashboard banner
"""
import re
import os

def fix_duplicate_banner(file_path: str) -> bool:
    """Remove duplicate banner instances"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read file with proper encoding
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
    
    # Find all banner instances
    banner_pattern = r'<div class=[\'"]v4-banner[\'"]>.*?</div>'
    matches = list(re.finditer(banner_pattern, content, re.DOTALL))
    print(f"Found {len(matches)} banner instances")
    
    if len(matches) <= 1:
        print("No duplicates to remove")
        return True
    
    # Keep only the first banner, remove the rest
    new_content = content
    for match in reversed(matches[1:]):  # Remove from end to preserve positions
        start, end = match.span()
        print(f"Removing banner at position {start}-{end}")
        new_content = new_content[:start] + new_content[end:]
    
    # Also check for duplicate toolbar instances
    toolbar_pattern = r'<div class=[\'"]v4-toolbar[\'"]>.*?</div>'
    toolbar_matches = list(re.finditer(toolbar_pattern, new_content, re.DOTALL))
    print(f"Found {len(toolbar_matches)} toolbar instances")
    
    if len(toolbar_matches) > 1:
        # Keep only the first toolbar
        for match in reversed(toolbar_matches[1:]):
            start, end = match.span()
            print(f"Removing toolbar at position {start}-{end}")
            new_content = new_content[:start] + new_content[end:]
    
    removed_chars = len(content) - len(new_content)
    print(f"Removed {removed_chars} characters of duplicate content")
    
    # Create backup and write new content
    backup_path = file_path + '.duplicate_fix_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(new_content)
        print(f"✅ Fixed duplicates! Final size: {len(new_content)} chars")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if fix_duplicate_banner(file_path):
        print("✅ Success! Removed duplicate banner instances.")
    else:
        print("❌ Failed to fix duplicates.")
