"""Standalone enhancement injector (temporary clean module).
Use this instead of corrupted generate_plots for landing-page dashboard injection.
"""
from __future__ import annotations
import os, re, datetime, traceback
from typing import Dict, Any, List

START_MARK = "<!-- ENHANCEMENT_DASHBOARD_START -->"
END_MARK = "<!-- ENHANCEMENT_DASHBOARD_END -->"

def _wrap(title: str, body: str) -> str:
    return "<div class='analysis-section'><h3 class='section-title'>{}</h3>{}</div>".format(title, body)

def generate_sections(data: Dict[str, Any]) -> List[tuple[str,str,str]]:
    return [
        ("sec_perf","📊 Performance", _wrap("📊 Performance","<p>Performance overview placeholder.</p>")),
        ("sec_errors","🚨 Errors", _wrap("🚨 Errors","<p>No error analysis yet.</p>")),
        ("sec_health","💓 Health", _wrap("💓 Health","<p>Health placeholder.</p>")),
        ("sec_risk","🛡 Risk", _wrap("🛡 Risk","<p>Risk metrics placeholder.</p>")),
    ]

def generate_block(data: Dict[str, Any]) -> str:
    sections = generate_sections(data)
    buttons = ["<button class='section-nav-btn' data-target='{}'>{}</button>".format(sid,title) for sid,title,_ in sections]
    toolbar = ("<div class='sections-toolbar'>" + ''.join(buttons) +
               "<button class='section-nav-btn small js-show-all'>Show All</button>"+
               "<button class='section-nav-btn small js-hide-all'>Hide All</button></div>")
    hidden = ["<section id='{}' class='dashboard-section hidden'><h2 class='section-heading'>{}</h2>{}</section>".format(sid,title,c) for sid,title,c in sections]
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    tpl = (
        "{start}\n"
        "<style>\n"
        ".landing-banner {{margin:40px 0 25px;padding:34px;background:linear-gradient(135deg,#1e3c72,#2a5298);border-radius:16px;color:#fff;}}\n"
        ".sections-toolbar-wrapper {{position:sticky;top:0;z-index:50;background:#ffffffdd;backdrop-filter:blur(6px);padding:6px;border-bottom:1px solid #e2e6ef;}}\n"
        ".sections-toolbar {{display:flex;flex-wrap:wrap;gap:6px;}}\n"
        ".section-nav-btn {{background:#4556d6;color:#fff;border:0;padding:8px 10px;border-radius:6px;font-size:.75rem;cursor:pointer;}}\n"
        ".section-nav-btn.small {{background:#6c757d;}}\n"
        ".dashboard-section.hidden {{display:none;}}\n"
        ".dashboard-section.active {{display:block;animation:fadeIn .3s ease;}}\n"
        "@keyframes fadeIn {{from {{opacity:0;transform:translateY(6px);}} to {{opacity:1;transform:translateY(0);}}}}\n"
        ".analysis-section {{background:#fff;padding:18px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}\n"
        ".empty-note {{opacity:.55;font-size:.85em;margin:18px 4px;}}\n"
        "</style>\n"
        "<script>\n"
        "window.addEventListener('DOMContentLoaded', function() {{\n"
        " var sections = Array.from(document.querySelectorAll('.dashboard-section'));\n"
        " var buttons = Array.from(document.querySelectorAll('.sections-toolbar .section-nav-btn[data-target]'));\n"
        " var showAll = document.querySelector('.js-show-all');\n"
        " var hideAll = document.querySelector('.js-hide-all');\n"
        " function hideAllSections() {{ sections.forEach(function(s) {{ s.classList.remove('active'); s.classList.add('hidden'); }}); }}\n"
        " function showOne(id) {{ var t=document.getElementById(id); if(!t) return; hideAllSections(); t.classList.remove('hidden'); t.classList.add('active'); t.scrollIntoView({{behavior:'smooth',block:'start'}}); }}\n"
        " buttons.forEach(function(b) {{ b.addEventListener('click', function() {{ showOne(b.dataset.target); }}); }});\n"
        " showAll.addEventListener('click', function() {{ sections.forEach(function(s) {{ s.classList.remove('hidden'); s.classList.add('active'); }}); window.scrollTo({{top:0,behavior:'smooth'}}); }});\n"
        " hideAll.addEventListener('click', function() {{ hideAllSections(); window.scrollTo({{top:0,behavior:'smooth'}}); }});\n"
        "}});\n"
        "</script>\n"
        "<div class='landing-banner'>\n"
        " <h1>🚀 Comprehensive Trading System Analysis</h1>\n"
        " <p style='margin:6px 0 10px;'>Landing page hub • Generated {ts}</p>\n"
        " <p style='font-size:.75em;opacity:.8;'>Use the buttons below to reveal dashboards.</p>\n"
        "</div>\n"
        "<div class='sections-toolbar-wrapper'>{toolbar}</div>\n"
        "<div class='dashboard-sections-container'>{sections_html}<div class='empty-note'>No sections active. Select a dashboard above.</div></div>\n"
        "{end}\n"
    )
    return tpl.format(start=START_MARK,end=END_MARK,ts=ts,toolbar=toolbar,sections_html=''.join(hidden))

def _strip_previous(html: str) -> str:
    if START_MARK in html and END_MARK in html:
        return re.sub(r"<!-- ENHANCEMENT_DASHBOARD_START -->(.|\n|\r)*?<!-- ENHANCEMENT_DASHBOARD_END -->","",html,flags=re.IGNORECASE)
    if START_MARK in html and END_MARK not in html:
        return re.sub(r"<!-- ENHANCEMENT_DASHBOARD_START -->(.|\n|\r){0,20000}","",html,flags=re.IGNORECASE)
    return html

def enhance_performance_report(report_path: str, data: Dict[str, Any]) -> str:
    with open(report_path,'r',encoding='utf-8',errors='ignore') as f: original = f.read()
    cleaned = _strip_previous(original)
    block = generate_block(data)
    lower = cleaned.lower()
    if '<body' in lower:
        m = re.search(r'<body[^>]*>', cleaned, re.IGNORECASE)
        if m:
            idx = m.end(); output = cleaned[:idx] + block + cleaned[idx:]
        else:
            output = block + cleaned
    else:
        output = block + cleaned
    tmp = report_path + '.tmp'
    with open(tmp,'w',encoding='utf-8') as f: f.write(output)
    os.replace(tmp, report_path)
    return report_path

def build_analysis_dataset() -> Dict[str, Any]:
    return {'system_health': {'overall_status':'Unknown'}}

def main(report_path='plots_output/performance_report.html', quiet=True):
    try:
        ds = build_analysis_dataset()
        return enhance_performance_report(report_path, ds)
    except Exception as e:
        if not quiet:
            print('[ENHANCE ERROR]', e)
            traceback.print_exc()
        return None

if __name__ == '__main__':
    p = main(quiet=False)
    print('✅ Updated' if p else '❌ Failed', p)
