import json, os, time
from utilities.utils import load_master_config, _compute_config_hash, atomic_update_master_config, mark_deployment_pending

MASTER_PATH = 'core/optimization_config.json'
TRIGGER_PATH = 'deployment-trigger.json'

def test_master_config_hash_integrity():
    cfg = load_master_config(MASTER_PATH)
    assert '_meta' in cfg, 'Missing _meta section'
    stored = cfg['_meta'].get('content_hash')
    recomputed = _compute_config_hash({k:v for k,v in cfg.items() if k != '_meta'})
    assert stored == recomputed, 'Content hash mismatch'


def test_ephemeral_reason_does_not_trigger():
    # ensure trigger removed
    if os.path.exists(TRIGGER_PATH):
        os.remove(TRIGGER_PATH)
    mark_deployment_pending(TRIGGER_PATH, reason='temp_ml_toggle_unit_test')
    if os.path.exists(TRIGGER_PATH):
        with open(TRIGGER_PATH,'r') as f:
            data = json.load(f)
            # Should not be pending because skipped (file might not exist at all)
            assert not data.get('pending', False), 'Ephemeral reason should not set pending'
    # Create a real reason and ensure pending
    mark_deployment_pending(TRIGGER_PATH, reason='config_update_real')
    assert os.path.exists(TRIGGER_PATH), 'Trigger file missing for real reason'
    with open(TRIGGER_PATH,'r') as f:
        data = json.load(f)
        assert data.get('pending', False) is True
        assert data.get('reason') == 'config_update_real'
