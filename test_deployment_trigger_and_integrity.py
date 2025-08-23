"""Lightweight validation for deployment trigger + master config integrity.

Simulates a harmless metadata-only update to master config, invokes atomic writer,
then asserts:
 1. deployment-trigger.json marked pending with provided reason
 2. verify_master_config_integrity exits with code 0

Usage:
  python test_deployment_trigger_and_integrity.py --reason test_run
Exit codes:
 0 success, 1 failure.
"""
from __future__ import annotations
import json, subprocess, sys, argparse, time, pathlib
from utilities.utils import load_master_config, atomic_update_master_config, mark_deployment_pending

TRIGGER_PATH = pathlib.Path('deployment-trigger.json')

def run_integrity_check() -> bool:
    proc = subprocess.run([sys.executable, 'verify_master_config_integrity.py'], capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
    return proc.returncode == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reason', default='validation_test', help='Reason tag for simulated update')
    args = ap.parse_args()
    cfg = load_master_config()
    # Inject a no-op meta bump (touch timestamp)
    cfg.setdefault('_meta', {})['self_test_touch'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    atomic_update_master_config(cfg, reason=f"self_test_{args.reason}")
    mark_deployment_pending(reason=f"self_test_{args.reason}")
    if not TRIGGER_PATH.exists():
        print('❌ deployment-trigger.json missing after update')
        sys.exit(1)
    try:
        trig = json.loads(TRIGGER_PATH.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ Failed reading trigger: {e}")
        sys.exit(1)
    if not trig.get('pending'):
        print('❌ Trigger not marked pending')
        sys.exit(1)
    if f"self_test_{args.reason}" not in trig.get('reason',''):
        print('❌ Trigger reason mismatch:', trig.get('reason'))
        sys.exit(1)
    if not run_integrity_check():
        print('❌ Integrity check failed')
        sys.exit(1)
    print('✅ Deployment trigger & integrity validation passed')
    sys.exit(0)

if __name__ == '__main__':
    main()
