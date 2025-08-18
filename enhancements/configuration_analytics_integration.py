"""Fallback configuration analytics provider.
Summarizes final_config.json and optimized_params_per_window.json if present.
"""
import os, json
from pathlib import Path

def add_configuration_analytics_to_report(run_dir: str) -> str:
    try:
        parts = []
        final_cfg = Path(run_dir) / 'final_config.json'
        per_window = Path(run_dir) / 'optimized_params_per_window.json'
        if final_cfg.exists():
            try:
                cfg = json.loads(final_cfg.read_text())
                keys = list(cfg.keys())[:25]
                parts.append(f"<p><b>Final Config Keys ({len(cfg)} total):</b> {', '.join(keys)}{'...' if len(cfg)>25 else ''}</p>")
            except Exception as e:
                parts.append(f"<p>Could not parse final_config.json: {e}</p>")
        else:
            parts.append('<p>No final_config.json found.</p>')
        if per_window.exists():
            try:
                ow = json.loads(per_window.read_text())
                parts.append(f"<p><b>Windows Optimized:</b> {len(ow)}</p>")
            except Exception as e:
                parts.append(f"<p>Could not parse optimized_params_per_window.json: {e}</p>")
        else:
            parts.append('<p>No optimized_params_per_window.json found.</p>')
        html = "<div class='config-analytics'><h3>Configuration Analytics</h3>" + ''.join(parts) + "</div>"
        return html
    except Exception as e:
        return f"<div class='config-analytics'><h3>Configuration Analytics</h3><p>Error: {e}</p></div>"
