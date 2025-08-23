"""Utility script to ensure the optimization config has an up-to-date content hash.

Used by the config-integrity GitHub Action to auto-correct the hash
if the functional content (excluding _meta) changed without updating it.
"""
from __future__ import annotations

import sys
from utilities.utils import (
    load_master_config,
    atomic_update_master_config,
    _compute_config_hash,
)

CONFIG_PATH = "core/optimization_config.json"

def main() -> int:
    cfg = load_master_config(CONFIG_PATH)
    if not cfg:
        print(f"Missing config at {CONFIG_PATH}", file=sys.stderr)
        return 1

    body = {k: v for k, v in cfg.items() if k != "_meta"}
    recomputed = _compute_config_hash(body)
    meta = dict(cfg.get("_meta", {}))
    stored = meta.get("content_hash")

    print(f"Stored hash: {stored}")
    print(f"Recomputed: {recomputed}")

    if stored != recomputed:
        meta["content_hash"] = recomputed
        merged = dict(body)
        merged["_meta"] = meta
        atomic_update_master_config(
            merged,
            path=CONFIG_PATH,
            reason="ci_auto_hash_fix",
            suppress_trigger=True,
        )
        print("Hash updated")
    else:
        print("Hash OK")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
