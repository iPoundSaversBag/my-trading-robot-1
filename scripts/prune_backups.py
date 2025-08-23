"""Phase 6B backup pruning script.

Retains only latest N config backups & removes obsolete corrupted variants beyond threshold.
"""
import os, json, re, datetime, shutil
from pathlib import Path

BACKUP_DIR = Path('backups/configs')
RETAIN = 5

def parse_ts(name: str):
    m = re.search(r'(\d{8}T\d{6})Z', name)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
    except Exception:
        return None

def main():
    if not BACKUP_DIR.exists():
        print('No backup dir')
        return
    files = [p for p in BACKUP_DIR.iterdir() if p.suffix == '.json']
    annotated = [(p, parse_ts(p.name)) for p in files]
    annotated.sort(key=lambda x: x[1] or datetime.datetime.min, reverse=True)
    to_keep = set(p for p,_ in annotated[:RETAIN])
    removed = []
    for p,_ in annotated[RETAIN:]:
        try:
            p.unlink()
            removed.append(p.name)
        except Exception:
            pass
    print(f"Kept {len(to_keep)} backups, removed {len(removed)}: {removed}")

if __name__ == '__main__':
    main()
