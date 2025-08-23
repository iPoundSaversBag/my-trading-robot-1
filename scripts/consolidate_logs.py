"""Phase 6C log consolidation: produce single summary + rotate oversize logs.
"""
import os, json, time, shutil, gzip
from pathlib import Path

LOG_DIR = Path('logs')
MAX_SIZE = 512 * 1024  # 512 KB rotation threshold
SUMMARY = LOG_DIR / 'summary_status.json'

PRIMARY_KEYS = ['system_health.json','strategy.log','live_bot.log','watcher.log','trade_alerts.log','error_alerts.log']


def rotate_if_large(path: Path):
    try:
        if path.exists() and path.is_file() and path.stat().st_size > MAX_SIZE:
            ts = time.strftime('%Y%m%dT%H%M%S')
            gz = path.with_suffix(path.suffix + f'.{ts}.gz')
            with open(path,'rb') as f_in, gzip.open(gz,'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            path.write_text('', encoding='utf-8')
            return str(gz)
    except Exception:
        pass
    return None

def build_summary():
    data = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'), 'files': {}}
    for name in PRIMARY_KEYS:
        p = LOG_DIR / name
        if p.exists():
            data['files'][name] = {'size': p.stat().st_size}
    SUMMARY.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    LOG_DIR.mkdir(exist_ok=True)
    rotations = {}
    for name in PRIMARY_KEYS:
        rotated = rotate_if_large(LOG_DIR / name)
        if rotated:
            rotations[name] = rotated
    build_summary()
    print('Summary built. Rotations:', rotations)

if __name__ == '__main__':
    main()
