"""Lightweight fallback module to provide optimization analytics HTML snippet.
If real module is absent, this supplies basic summary from optimization_trial_data.json.
"""
import os, json
from pathlib import Path

def add_optimization_analytics_to_report(run_dir: str) -> str:
    try:
        trial_path = Path(run_dir) / 'optimization_trial_data.json'
        if not trial_path.exists():
            return '<div class="opt-analytics"><h3>Optimization Analytics</h3><p>No trial data file found.</p></div>'
        data = json.loads(trial_path.read_text())
        if not isinstance(data, list):
            return '<div class="opt-analytics"><h3>Optimization Analytics</h3><p>Unexpected trial data format.</p></div>'
        n = len(data)
        best = None
        for rec in data:
            val = rec.get('objective_value') or rec.get('value')
            if val is None:
                continue
            if best is None or val > best[0]:
                best = (val, rec.get('params',''))
        best_html = ''
        if best:
            best_html = f"<p><b>Best Objective:</b> {best[0]:.4f}</p><pre>{best[1]}</pre>"
        return f"<div class='opt-analytics'><h3>Optimization Analytics</h3><p>Trials: {n}</p>{best_html}</div>"
    except Exception as e:
        return f"<div class='opt-analytics'><h3>Optimization Analytics</h3><p>Error: {e}</p></div>"
