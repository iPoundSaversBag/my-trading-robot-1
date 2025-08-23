"""Utility script to ensure the optimization config has an up-to-date content hash.

Primary goal: keep GitHub Action lightweight & resilient. We *attempt* to import
the rich helper functions from ``utilities.utils``; if that import fails (e.g.
missing heavy optional deps like matplotlib, ccxt, ta on a fresh CI runner), we
fall back to a minimal stdlib implementation so the workflow still succeeds in
verifying / auto-updating the hash.
"""
from __future__ import annotations

import sys, json, hashlib, datetime, os, tempfile

CONFIG_PATH = "core/optimization_config.json"

# ---------------------------------------------------------------------------
# Attempt full implementation import; fall back to minimal versions if needed
# ---------------------------------------------------------------------------
try:  # pragma: no cover - best effort import
    from utilities.utils import (
        load_master_config,  # type: ignore
        atomic_update_master_config,  # type: ignore
        _compute_config_hash,  # type: ignore
    )
    HAVE_UTILS = True
except Exception as e:  # pragma: no cover
    print(f"[sync_config_hash] Falling back to minimal implementation: {e}")
    HAVE_UTILS = False

    def load_master_config(path: str = CONFIG_PATH):  # minimal
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _compute_config_hash(cfg: dict) -> str:  # minimal
        try:
            canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(canonical).hexdigest()
        except Exception:
            return ""

    def atomic_update_master_config(updated_cfg: dict, path: str = CONFIG_PATH, reason: str = "ci_auto_hash_fix", suppress_trigger: bool = True):  # minimal
        try:
            cfg = dict(updated_cfg)
            meta = cfg.get("_meta", {}) if isinstance(cfg.get("_meta"), dict) else {}
            meta.update({
                "schema_version": 1,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "reason": reason,
            })
            cfg["_meta"] = meta
            meta["content_hash"] = _compute_config_hash({k: v for k, v in cfg.items() if k != "_meta"})
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".cfgtmp_", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    json.dump(cfg, tmp_f, indent=2)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            return True
        except Exception as e:
            print(f"Atomic update failed (fallback): {e}", file=sys.stderr)
            return False


def main() -> int:
    cfg = load_master_config(CONFIG_PATH)
    if not cfg:
        print(f"Missing or empty config at {CONFIG_PATH}", file=sys.stderr)
        return 1

    body = {k: v for k, v in cfg.items() if k != "_meta"}
    recomputed = _compute_config_hash(body)
    meta = dict(cfg.get("_meta", {}))
    stored = meta.get("content_hash")

    print(f"Stored hash: {stored}")
    print(f"Recomputed: {recomputed}")
    if HAVE_UTILS:
        print("Mode: full utils import")
    else:
        print("Mode: minimal fallback")

    if stored != recomputed:
        meta["content_hash"] = recomputed
        merged = dict(body)
        merged["_meta"] = meta
        ok = atomic_update_master_config(
            merged,
            path=CONFIG_PATH,
            reason="ci_auto_hash_fix",
            suppress_trigger=True,
        )
        print("Hash updated" if ok else "Hash update FAILED")
        return 0 if ok else 2
    else:
        print("Hash OK")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
