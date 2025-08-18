#!/usr/bin/env python3
"""
Direct cleanup and V4 injection - removes ALL dashboard content and injects fresh V4
"""
import os
import re
from generate_plots import _build_block, collect_data

def clean_and_inject(file_path: str) -> bool:
    print(f"[debug] Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"[debug] File does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print(f"[debug] File size: {len(content)} characters")
    except Exception as e:
        print(f"[debug] Error reading file: {e}")
        return False
    
    # Step 1: Find body tag position for insertion point
    body_start = content.lower().find('<body')
    print(f"[debug] Body start position: {body_start}")
    if body_start == -1:
        print("[debug] No body tag found")
        return False
    
    body_end = content.find('>', body_start) + 1
    close_body = content.lower().find('</body>')
    print(f"[debug] Body end: {body_end}, Close body: {close_body}")
    
    if close_body == -1:
        close_body = len(content)
    
    # Step 2: Extract clean content (before body + body tag + after close body)
    before_content = content[:body_end]
    after_content = content[close_body:]
    print(f"[debug] Before content length: {len(before_content)}, After content length: {len(after_content)}")
    
    # Step 3: Get the content between body tags and strip all dashboard-related content
    body_content = content[body_end:close_body]
    print(f"[debug] Body content length: {len(body_content)}")
    
    try:
        # Step 4: Build new V4 dashboard
        data = collect_data()
        v4_block = _build_block(data)
        print(f"[debug] V4 block length: {len(v4_block)}")
    except Exception as e:
        print(f"[debug] Error building V4 block: {e}")
        return False
    
    # Remove everything that contains dashboard keywords
    dashboard_removal_patterns = [
        # Any HTML block with dashboard-related content
        r'<!--[^>]*ENHANCEMENT_DASHBOARD[^>]*-->.*?<!--[^>]*ENHANCEMENT_DASHBOARD[^>]*-->',
        r'<!--[^>]*COMPREHENSIVE TRADING[^>]*-->',
        r'<!--[^>]*ENHANCEMENT_DASHBOARD_CONTENT[^>]*-->',
        # Style and script blocks with dashboard content
        r'<style[^>]*>.*?(?:dashboard|landing-banner|sections-toolbar|nav-btn|analysis-section).*?</style>',
        r'<script[^>]*>.*?(?:dashboard|DOMContentLoaded.*section|hideAll|showOne).*?</script>',
        # Div blocks with dashboard classes
        r'<div[^>]*class="[^"]*(?:landing-banner|sections-toolbar|dashboard|analysis|comprehensive)[^"]*"[^>]*>.*?</div>',
        r'<section[^>]*(?:id="sec_|class="[^"]*dashboard)[^>]*>.*?</section>',
    ]
    
    clean_body = body_content
    for pattern in dashboard_removal_patterns:
        clean_body = re.sub(pattern, '', clean_body, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining lines with dashboard keywords
    lines = clean_body.split('\n')
    filtered_lines = []
    skip_keywords = [
        'dashboard', 'landing-banner', 'sections-toolbar', 'nav-btn', 'analysis-section',
        'ENHANCEMENT_DASHBOARD', 'comprehensive-dashboard', 'fadeIn', 'hideAll'
    ]
    
    for line in lines:
        if not any(keyword in line.lower() for keyword in skip_keywords):
            filtered_lines.append(line)
    
    clean_body = '\n'.join(filtered_lines)
    
    # Step 4: Build new V4 dashboard
    data = collect_data()
    v4_block = _build_block(data)
    
    # Step 5: Reconstruct file
    new_content = before_content + '\n' + v4_block + '\n' + clean_body + after_content
    
    # Step 6: Write atomically
    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    os.replace(tmp_path, file_path)
    
    print(f"[clean_inject] Successfully cleaned and injected V4 dashboard into {file_path}")
    return True

if __name__ == '__main__':
    # Auto-detect latest report
    base = 'plots_output'
    marker = os.path.join(base, 'latest_run_dir.txt')
    
    if os.path.exists(marker):
        with open(marker, 'r') as f:
            run_dir = f.read().strip()
        if not run_dir.startswith(base):
            run_dir = os.path.join(base, run_dir)
        report_path = os.path.join(run_dir, 'performance_report.html')
        
        if clean_and_inject(report_path):
            print(f"[success] Dashboard cleaned and updated: {report_path}")
        else:
            print(f"[error] Failed to process: {report_path}")
    else:
        print("[error] No latest_run_dir.txt found")
