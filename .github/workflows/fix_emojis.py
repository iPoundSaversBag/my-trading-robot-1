#!/usr/bin/env python3
"""
Script to replace emoji characters in YAML files with ASCII equivalents
"""

def fix_emojis_in_file(filename):
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Specific emoji replacements based on what we found
    replacements = [
        ('🚀', 'LAUNCH'),
        ('📊', 'ANALYTICS'), 
        ('✅', 'SUCCESS'),
        ('❌', 'ERROR'),
        ('⚠️', 'WARNING'),
        ('⚠', 'WARNING'),
        ('🔄', 'SYNC'),
        ('📢', 'NOTIFY'),
        ('📥', 'DOWNLOAD')
    ]
    
    # Track what we replace
    changes_made = []
    
    # Apply each replacement
    for emoji, replacement in replacements:
        count_before = content.count(emoji)
        if count_before > 0:
            content = content.replace(emoji, replacement)
            changes_made.append(f'Replaced {count_before} instances of {emoji} with {replacement}')
    
    # Write back to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return changes_made

if __name__ == "__main__":
    changes = fix_emojis_in_file('bidirectional-sync.yml')
    if changes:
        print('Changes made:')
        for change in changes:
            print(f'  {change}')
    else:
        print('No emoji characters found to replace')
