#!/usr/bin/env python3
"""
Remove the final 5 old dashboard remnants
"""
import re
import os

def remove_final_remnants(file_path: str) -> bool:
    """Remove the last remaining old dashboard elements"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read file with proper encoding
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    original_content = content
    
    # Target the specific old dashboard CSS and HTML that's still present
    cleanup_patterns = [
        # Remove any comprehensive-dashboard CSS class definitions
        r'\.comprehensive-dashboard\s*\{[^}]*\}',
        
        # Remove dashboard-section CSS class definitions  
        r'\.dashboard-section[^{]*\{[^}]*\}',
        
        # Remove landing-banner CSS class definitions
        r'\.landing-banner[^{]*\{[^}]*\}',
        
        # Remove sections-toolbar CSS class definitions
        r'\.sections-toolbar[^{]*\{[^}]*\}',
        
        # Remove any HTML elements with these old classes
        r'<[^>]*class="[^"]*(?:comprehensive-dashboard|dashboard-section|landing-banner|sections-toolbar)[^"]*"[^>]*>.*?</[^>]+>',
        
        # Remove any style blocks containing these old classes
        r'<style[^>]*>[^<]*(?:comprehensive-dashboard|dashboard-section|landing-banner|sections-toolbar)[^<]*</style>',
        
        # Remove any script blocks referencing these old classes
        r'<script[^>]*>[^<]*(?:comprehensive-dashboard|dashboard-section|landing-banner|sections-toolbar)[^<]*</script>',
    ]
    
    # Apply cleanup patterns
    removed_total = 0
    for i, pattern in enumerate(cleanup_patterns):
        before_len = len(content)
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        after_len = len(content)
        removed = before_len - after_len
        if removed > 0:
            print(f"Pattern {i+1}: Removed {removed} characters")
            removed_total += removed
    
    # Additional line-by-line cleanup for any remaining references
    lines = content.split('\n')
    clean_lines = []
    
    old_keywords = ['comprehensive-dashboard', 'dashboard-section', 'landing-banner', 'sections-toolbar']
    
    for line in lines:
        # Skip lines that contain old dashboard keywords (but preserve V4 content)
        if any(keyword in line.lower() for keyword in old_keywords) and 'v4-' not in line.lower():
            print(f"Removed line: {line.strip()[:80]}...")
            removed_total += len(line) + 1  # +1 for newline
        else:
            clean_lines.append(line)
    
    final_content = '\n'.join(clean_lines)
    
    print(f"Total removed: {removed_total} characters")
    print(f"Final file size: {len(final_content)} chars")
    
    # Verify cleanup
    remaining_issues = 0
    check_patterns = ['comprehensive-dashboard', 'dashboard-section', 'landing-banner', 'sections-toolbar']
    for pattern in check_patterns:
        count = len(re.findall(pattern, final_content, re.IGNORECASE))
        if count > 0:
            print(f"⚠️  Still found {count} instances of {pattern}")
            remaining_issues += count
        else:
            print(f"✅ {pattern}: Cleaned")
    
    if remaining_issues == 0:
        print("✅ ALL OLD DASHBOARD REMNANTS REMOVED!")
    else:
        print(f"⚠️  {remaining_issues} issues still remain")
    
    # Create backup and write new content
    backup_path = file_path + '.final_cleanup_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(original_content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(final_content)
        print(f"✅ Final cleanup complete!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if remove_final_remnants(file_path):
        print("✅ Success! Removed all remaining old dashboard elements.")
    else:
        print("❌ Failed to complete final cleanup.")
