import os, json, time
from utilities.utils import mark_deployment_pending, clear_deployment_trigger
from core.backtest import IchimokuBacktester

TRIGGER_PATH = 'deployment-trigger.json'

def _read_trigger():
    if not os.path.exists(TRIGGER_PATH):
        return {}
    with open(TRIGGER_PATH,'r',encoding='utf-8') as f:
        return json.load(f)

def test_trigger_dedupe_window(tmp_path):
    # Ensure clean start
    if os.path.exists(TRIGGER_PATH):
        os.remove(TRIGGER_PATH)
    # First mark
    mark_deployment_pending(TRIGGER_PATH, reason='config_update_ci', dedupe_seconds=3)
    first = _read_trigger(); assert first.get('pending') is True
    # Second identical within window should not update timestamp (sleep < window)
    ts1 = first.get('last_update')
    time.sleep(1)
    mark_deployment_pending(TRIGGER_PATH, reason='config_update_ci', dedupe_seconds=3)
    second = _read_trigger(); assert second.get('last_update') == ts1
    # After window passes, should update
    time.sleep(3)
    mark_deployment_pending(TRIGGER_PATH, reason='config_update_ci', dedupe_seconds=3)
    third = _read_trigger(); assert third.get('last_update') != ts1
    clear_deployment_trigger(TRIGGER_PATH, note='ci_test')

def test_random_state_initialized_and_stable():
    # Instantiate twice with same config & debug seed -> same random_state
    bt1 = IchimokuBacktester('core/optimization_config.json', debug_mode=True)
    rs1 = bt1.random_state
    bt2 = IchimokuBacktester('core/optimization_config.json', debug_mode=True)
    rs2 = bt2.random_state
    assert isinstance(rs1, int) and isinstance(rs2, int)
    assert rs1 == rs2
