#!/usr/bin/env python3
"""
Surgical removal of old dashboard elements without breaking the structure
"""
import re
import os

def surgical_old_dashboard_removal(file_path: str) -> bool:
    """Carefully remove only old dashboard elements without breaking structure"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    original_content = content
    
    # Only remove specific old dashboard blocks that we know are problematic
    # Be very conservative to avoid breaking the structure
    
    # 1. Remove V1 dashboard blocks (if any remain)
    v1_pattern = r'<!-- ENHANCEMENT_DASHBOARD_START -->.*?<!-- ENHANCEMENT_DASHBOARD_END -->'
    before = len(content)
    content = re.sub(v1_pattern, '', content, flags=re.DOTALL)
    if len(content) != before:
        print(f"Removed V1 dashboard block: {before - len(content)} chars")
    
    # 2. Remove orphaned dashboard comments
    comment_patterns = [
        r'<!-- COMPREHENSIVE TRADING SYSTEM ANALYSIS DASHBOARD -->',
        r'<!-- ENHANCEMENT_DASHBOARD_CONTENT -->',
        r'<!-- INTERACTIVE DASHBOARD FEATURES -->'
    ]
    
    for pattern in comment_patterns:
        before = len(content)
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        if len(content) != before:
            print(f"Removed comment: {before - len(content)} chars")
    
    # 3. Only remove CSS that contains old dashboard classes but preserve V4
    old_css_pattern = r'<style[^>]*>[^<]*(?:\.dashboard-section\.hidden|\.comprehensive-dashboard\s*\{|\.landing-banner\s*\{|\.sections-toolbar\s*\{)[^<]*</style>'
    before = len(content)
    content = re.sub(old_css_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    if len(content) != before:
        print(f"Removed old CSS: {before - len(content)} chars")
    
    # 4. Remove specific old HTML sections but be very targeted
    old_html_patterns = [
        r'<div[^>]*class="[^"]*dashboard-sections-container[^"]*"[^>]*>.*?</div>',
        r'<section[^>]*id="sec_[^"]*"[^>]*class="[^"]*dashboard-section[^"]*"[^>]*>.*?</section>'
    ]
    
    for pattern in old_html_patterns:
        before = len(content)
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        if len(content) != before:
            print(f"Removed old HTML section: {before - len(content)} chars")
    
    # 5. Remove old JavaScript functions but preserve V4
    old_js_pattern = r'<script[^>]*>[^<]*(?:hideAllSections|showOne|dashboard-sections-container)[^<]*</script>'
    before = len(content)
    content = re.sub(old_js_pattern, '', content, flags=re.DOTALL)
    if len(content) != before:
        print(f"Removed old JavaScript: {before - len(content)} chars")
    
    total_removed = len(original_content) - len(content)
    print(f"Total removed: {total_removed} characters")
    print(f"Final file size: {len(content)} chars")
    
    # Verify we didn't break anything critical
    critical_checks = {
        'V4 Dashboard': 'v4-banner' in content,
        'V4 Toolbar': 'v4-toolbar' in content,
        'QuantStats': 'quantstats' in content.lower(),
        'HTML Structure': '<html' in content.lower() and '</html>' in content.lower(),
        'Head Section': '<head' in content.lower() and '</head>' in content.lower()
    }
    
    all_good = True
    for check, status in critical_checks.items():
        if status:
            print(f"✅ {check}: Present")
        else:
            print(f"❌ {check}: Missing!")
            all_good = False
    
    if not all_good:
        print("❌ Critical content missing - not saving changes")
        return False
    
    # Create backup and save
    backup_path = file_path + '.surgical_backup'
    try:
        with open(backup_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(original_content)
        print(f"Backup created: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    try:
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        print(f"✅ Surgical cleanup complete!")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    file_path = '../plots_output/20250817_133240/performance_report.html'
    if surgical_old_dashboard_removal(file_path):
        print("✅ Success! Surgically removed old dashboard elements safely.")
    else:
        print("❌ Failed - file preserved unchanged.")
