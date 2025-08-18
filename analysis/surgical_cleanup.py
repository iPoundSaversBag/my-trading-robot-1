#!/usr/bin/env python3
"""
Manual surgical cleanup - remove ALL dashboard content and apply clean V4
"""
import os
import re

def surgical_cleanup(file_path: str) -> bool:
    """Surgically remove all dashboard content and apply clean V4"""
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Original file size: {len(content)} chars")
    
    # Step 1: Remove ALL enhancement dashboard blocks and their content
    # This is very aggressive - remove everything between any dashboard markers
    patterns_to_remove = [
        # All enhancement dashboard blocks
        r'<!-- ENHANCEMENT_DASHBOARD_START -->.*?<!-- ENHANCEMENT_DASHBOARD_END -->',
        r'<!-- ENHANCEMENT_DASHBOARD_V[1-4]_START -->.*?<!-- ENHANCEMENT_DASHBOARD_V[1-4]_END -->',
        # All comprehensive dashboard comments and their following content until next major section
        r'<!-- COMPREHENSIVE TRADING SYSTEM ANALYSIS DASHBOARD -->.*?(?=<div class="container"|</body>|$)',
        r'<!-- ENHANCEMENT_DASHBOARD_CONTENT -->.*?(?=<div class="container"|</body>|$)',
    ]
    
    cleaned = content
    for pattern in patterns_to_remove:
        before = len(cleaned)
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        after = len(cleaned)
        print(f"Pattern removed {before - after} chars")
    
    # Step 2: Remove any orphaned dashboard-related style/script blocks
    style_script_cleanup = [
        r'<style[^>]*>.*?(?:dashboard|landing-banner|sections-toolbar|nav-btn|fadeIn|comprehensive).*?</style>',
        r'<script[^>]*>.*?(?:DOMContentLoaded.*dashboard|hideAll|showOne|sections-toolbar).*?</script>',
    ]
    
    for pattern in style_script_cleanup:
        before = len(cleaned)
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        after = len(cleaned)
        if before != after:
            print(f"Style/script cleanup removed {before - after} chars")
    
    print(f"After cleanup: {len(cleaned)} chars")
    
    # Step 3: Apply fresh V4 dashboard
    v4_dashboard = """
<!-- ENHANCEMENT_DASHBOARD_V4_START -->
<style>
/* V4 Dashboard - Reset and Clean Styles */
body{font-family:Arial,sans-serif;margin:0;background:#f8fafc;}
/* Landing Banner */
.dashboard-banner{margin:30px auto 20px;max-width:900px;padding:40px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:20px;color:white;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.2);}
.banner-title{font-size:2.5em;margin:0 0 15px 0;font-weight:300;}
.banner-subtitle{font-size:1.1em;opacity:0.9;margin:0 0 10px 0;}
.banner-timestamp{font-size:0.85em;opacity:0.7;}
/* Sticky Button Toolbar */
.button-toolbar{position:sticky;top:0;z-index:1000;background:rgba(255,255,255,0.95);backdrop-filter:blur(10px);border-bottom:2px solid #e5e7eb;padding:15px 0;margin:0;}
.toolbar-container{max-width:900px;margin:0 auto;padding:0 20px;}
.toolbar-buttons{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;}
.nav-btn{background:#4f46e5;color:white;border:none;padding:12px 20px;border-radius:8px;font-size:0.9rem;font-weight:500;cursor:pointer;transition:all 0.2s ease;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.nav-btn:hover{background:#3730a3;transform:translateY(-1px);box-shadow:0 4px 8px rgba(0,0,0,0.15);}
.nav-btn.active{background:#059669;}
.nav-btn.utility{background:#6b7280;}
.nav-btn.utility:hover{background:#4b5563;}
/* Content Area */
.content-area{max-width:900px;margin:30px auto;padding:0 20px;min-height:300px;}
.section-content{background:white;padding:30px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.05);border:1px solid #e5e7eb;}
.landing-message{text-align:center;padding:60px 30px;color:#6b7280;font-size:1.1em;background:white;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.05);}
.analysis-section{margin:20px 0;}
.analysis-section h3{color:#1f2937;margin-bottom:15px;font-size:1.2em;}
.analysis-section p{color:#4b5563;line-height:1.6;}
/* Hide any legacy content that might still exist */
.comprehensive-dashboard, .interactive-dashboard, .enh-banner, .sections-toolbar-wrapper, .dashboard-sections-container{display:none !important;}
</style>
<script>
document.addEventListener('DOMContentLoaded', function() {
  const contentArea = document.querySelector('.content-area');
  const buttons = document.querySelectorAll('.nav-btn[data-section]');
  const showAllBtn = document.querySelector('.js-show-all');
  const clearBtn = document.querySelector('.js-clear');
  
  function clearContent() {
    contentArea.innerHTML = '<div class="landing-message">🏠 Welcome! Select a section above to view detailed analysis</div>';
    buttons.forEach(btn => btn.classList.remove('active'));
  }
  
  function loadSection(sectionId) {
    const template = document.getElementById('section_' + sectionId);
    if (!template) return;
    contentArea.innerHTML = '<div class="section-content">' + template.innerHTML + '</div>';
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector('[data-section="' + sectionId + '"]').classList.add('active');
  }
  
  function showAll() {
    let allContent = '';
    buttons.forEach(btn => {
      const sectionId = btn.getAttribute('data-section');
      const template = document.getElementById('section_' + sectionId);
      if (template) {
        allContent += '<div class="section-content" style="margin-bottom:20px;">' + template.innerHTML + '</div>';
      }
    });
    contentArea.innerHTML = allContent;
    buttons.forEach(btn => btn.classList.remove('active'));
    showAllBtn.classList.add('active');
  }
  
  buttons.forEach(btn => {
    btn.addEventListener('click', () => loadSection(btn.getAttribute('data-section')));
  });
  
  if (showAllBtn) showAllBtn.addEventListener('click', showAll);
  if (clearBtn) clearBtn.addEventListener('click', clearContent);
  
  clearContent();
});
</script>
<div class='dashboard-banner'>
  <h1 class='banner-title'>🚀 Trading System Analysis Dashboard</h1>
  <p class='banner-subtitle'>Comprehensive Performance & Risk Analytics</p>
  <p class='banner-timestamp'>Generated 2025-08-17 13:40:00 UTC</p>
</div>
<div class='button-toolbar'>
  <div class='toolbar-container'>
    <div class='toolbar-buttons'>
      <button class='nav-btn' data-section='performance'>📊 Performance</button>
      <button class='nav-btn' data-section='errors'>🚨 Errors</button>
      <button class='nav-btn' data-section='health'>💓 Health</button>
      <button class='nav-btn' data-section='risk'>🛡 Risk</button>
      <button class='nav-btn utility js-show-all'>📋 Show All</button>
      <button class='nav-btn utility js-clear'>🏠 Home</button>
    </div>
  </div>
</div>
<div class='content-area'>
  <div class='landing-message'>🏠 Welcome! Select a section above to view detailed analysis</div>
</div>
<template id='section_performance'><div class='analysis-section'><h3>📊 Performance</h3><p>Performance overview placeholder.</p></div></template>
<template id='section_errors'><div class='analysis-section'><h3>🚨 Errors</h3><p>No error analysis yet.</p></div></template>
<template id='section_health'><div class='analysis-section'><h3>💓 Health</h3><p>Health placeholder.</p></div></template>
<template id='section_risk'><div class='analysis-section'><h3>🛡 Risk</h3><p>Risk metrics placeholder.</p></div></template>
<!-- ENHANCEMENT_DASHBOARD_V4_END -->
"""
    
    # Step 4: Insert V4 dashboard after <body> tag
    body_match = re.search(r'<body[^>]*>', cleaned, re.IGNORECASE)
    if body_match:
        insert_pos = body_match.end()
        final_content = cleaned[:insert_pos] + v4_dashboard + cleaned[insert_pos:]
    else:
        # Fallback - insert before </body> or at end
        if '</body>' in cleaned.lower():
            final_content = re.sub(r'</body>', v4_dashboard + '\n</body>', cleaned, flags=re.IGNORECASE)
        else:
            final_content = cleaned + v4_dashboard
    
    print(f"Final file size: {len(final_content)} chars")
    
    # Step 5: Write back
    backup_path = file_path + '.pre_surgical_backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)  # Backup original
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"✅ Surgical cleanup complete. Backup saved to {backup_path}")
    return True

if __name__ == '__main__':
    file_path = 'plots_output/20250817_133240/performance_report.html'
    if surgical_cleanup(file_path):
        print("✅ Success! Dashboard cleaned and V4 applied.")
    else:
        print("❌ Failed to process file.")
