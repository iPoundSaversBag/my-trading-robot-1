"""Verify integrity of master optimization config.

Recomputes SHA-256 hash of canonicalized JSON (excluding volatile _meta fields except schema_version)
and compares to stored _meta.content_hash.

Exit codes:
 0 - OK
 1 - Hash mismatch
 2 - File missing or unreadable
 3 - Other error
"""
from __future__ import annotations
import json, sys, hashlib, pathlib, argparse

CONFIG_PATH = pathlib.Path('core') / 'optimization_config.json'

def compute_hash(obj) -> str:
    # Copy while excluding volatile fields that should not affect integrity
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k,v in o.items() if not (k == '_meta' and isinstance(v, dict))}
        elif isinstance(o, list):
            return [strip(x) for x in o]
        return o
    sanitized = strip(obj)
    blob = json.dumps(sanitized, sort_keys=True, separators=(',',':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default=str(CONFIG_PATH), help='Path to master config JSON')
    args = ap.parse_args()
    p = pathlib.Path(args.path)
    if not p.exists():
        print(f"❌ Config not found: {p}")
        sys.exit(2)
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Failed reading config: {e}")
        sys.exit(2)
    stored = (data.get('_meta') or {}).get('content_hash')
    recomputed = compute_hash(data)
    if stored is None:
        print("⚠️ No stored content_hash; cannot verify. Recomputed:", recomputed)
        sys.exit(1)
    if stored != recomputed:
        print(f"❌ Hash mismatch. stored={stored} recomputed={recomputed}")
        sys.exit(1)
    # Avoid Unicode symbols for Windows legacy console compatibility
    print(f"INTEGRITY_OK hash={stored}")
    sys.exit(0)

if __name__ == '__main__':
    main()
