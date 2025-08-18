"""report_slimmer: Produce a compact slim HTML version of the large performance_report.html
with only a minimal head + the new V4 enhancement dashboard (no bulky QuantStats body).

Usage (PowerShell):
  python analysis/report_slimmer.py  (auto-discovers latest report)
Output: creates sibling file performance_report_slim.html in the same run directory.
"""
from __future__ import annotations
import os, re, datetime, sys, traceback
from typing import Optional, Any, Dict, List

V4_START = "<!-- ENHANCEMENT_DASHBOARD_V4_START -->"
V4_END   = "<!-- ENHANCEMENT_DASHBOARD_V4_END -->"

# Basic section stubs (can be enriched later)
def _wrap(title: str, body: str) -> str:
    return f"<div class='analysis-section'><h3>{title}</h3>{body}</div>"

def _sections(data: Dict[str, Any]) -> List[tuple[str,str,str]]:
    return [
        ("perf",  "📊 Performance", _wrap("📊 Performance", "<p>Performance placeholder.</p>")),
        ("errors","🚨 Errors",      _wrap("🚨 Errors", "<p>No error data.</p>")),
        ("health","💓 Health",      _wrap("💓 Health", "<p>Health placeholder.</p>")),
        ("risk",  "🛡 Risk",        _wrap("🛡 Risk", "<p>Risk metrics placeholder.</p>")),
    ]

def _build_block(data: Dict[str, Any]) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    sections = _sections(data)
    buttons = ''.join(f"<button class='nav-btn' data-id='{sid}'>{title}</button>" for sid,title,_ in sections)
    templates = ''.join(
        f"<template id='tpl_{sid}'><div class='panel-section'><h2>{title}</h2>{content}</div></template>"
        for sid,title,content in sections
    )
    return (
        f"{V4_START}\n"
        "<style>\n"
        "body{font:13px/1.4 Arial,sans-serif;margin:26px;background:#f2f5f9;}\n"
        ".enh-banner{margin:10px 0 16px;padding:26px;background:linear-gradient(135deg,#1e3c72,#2a5298);border-radius:16px;color:#fff;}\n"
        ".toolbar-wrap{position:sticky;top:0;z-index:50;background:#ffffffee;backdrop-filter:blur(5px);padding:6px;border:1px solid #d9e0ec;border-radius:10px;margin-bottom:14px;}\n"
        ".toolbar{display:flex;flex-wrap:wrap;gap:6px;}\n"
        ".nav-btn{background:#4556d6;color:#fff;border:0;padding:6px 10px;border-radius:6px;font-size:.7rem;cursor:pointer;}\n"
        ".nav-btn.small{background:#6c757d;}\n"
        ".active-container{min-height:140px;border:2px dashed #cfd6e3;border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:18px;background:#fff;}\n"
        ".panel-section{background:#fff;padding:18px;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.05);}\n"
        ".analysis-section{background:#fff;padding:14px;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.04);}\n"
        ".placeholder{opacity:.55;font-size:.8rem;margin:4px;}\n"
        "</style>\n"
        "<script>\n"
        "window.addEventListener('DOMContentLoaded',()=>{const c=document.querySelector('.active-container');const all=document.querySelector('.js-all');const clr=document.querySelector('.js-clear');function one(id){const t=document.getElementById('tpl_'+id);if(!t)return;c.innerHTML=t.innerHTML;}function allLoad(){c.innerHTML=[...document.querySelectorAll('template[id^=tpl_]')].map(t=>t.innerHTML).join('');}function clearAll(){c.innerHTML='<div class=\\'placeholder\\'>No section selected.</div>';}document.querySelectorAll('.toolbar .nav-btn[data-id]').forEach(b=>b.addEventListener('click',()=>one(b.dataset.id)));all&&all.addEventListener('click',allLoad);clr&&clr.addEventListener('click',clearAll);clearAll();});\n"
        "</script>\n"
        "<div class='enh-banner'>\n"
        "  <h1>🚀 Slim Trading System Analysis</h1>\n"
        f"  <p style='margin:6px 0 8px;'>Generated {ts}</p>\n"
        "  <p style='font-size:.7rem;opacity:.85;'>Buttons load sections below.</p>\n"
        "</div>\n"
        f"<div class='toolbar-wrap'><div class='toolbar'>{buttons}<button class='nav-btn small js-all'>Show All</button><button class='nav-btn small js-clear'>Clear</button></div></div>\n"
        "<div class='active-container placeholder'>Preparing…</div>\n"
        f"{templates}\n"
        f"{V4_END}\n"
    )

def _find_latest() -> Optional[str]:
    base='plots_output'
    try:
        marker=os.path.join(base,'latest_run_dir.txt')
        if os.path.exists(marker):
            path=open(marker,'r',encoding='utf-8',errors='ignore').read().strip()
            if path and not os.path.isabs(path):
                if not path.startswith(base): path=os.path.join(base,path)
            cand=os.path.join(path,'performance_report.html')
            if os.path.exists(cand): return cand
        if not os.path.isdir(base): return None
        subs=[os.path.join(base,d) for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))]
        subs.sort(key=lambda p:(os.path.basename(p), os.path.getmtime(p)), reverse=True)
        for d in subs:
            cand=os.path.join(d,'performance_report.html')
            if os.path.exists(cand): return cand
    except Exception:
        return None
    return None

def _extract_head(html: str) -> str:
    m=re.search(r'<head[\s\S]*?</head>', html, re.IGNORECASE)
    if m: return m.group(0)
    # fallback minimal head
    return ("<head><meta charset='utf-8'><title>Slim Report</title></head>")

def collect_data() -> Dict[str, Any]:
    return {'meta': {'status':'N/A'}}

def build_slim(report_path: str, out_path: str) -> bool:
    if not os.path.exists(report_path):
        return False
    raw=open(report_path,'r',encoding='utf-8',errors='ignore').read()
    head=_extract_head(raw)
    block=_build_block(collect_data())
    slim_html=(
        "<!DOCTYPE html><html lang='en'>"+head+
        "<body>"+block+"</body></html>"\
    )
    with open(out_path,'w',encoding='utf-8') as f: f.write(slim_html)
    return True

def main():
    target=_find_latest()
    if not target:
        print('[slimmer] no performance_report.html found')
        return 1
    out=os.path.join(os.path.dirname(target),'performance_report_slim.html')
    ok=build_slim(target,out)
    if ok:
        print('[slimmer] wrote', out, 'size', os.path.getsize(out),'bytes (orig', os.path.getsize(target),'bytes)')
        return 0
    print('[slimmer] failed')
    return 2

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
