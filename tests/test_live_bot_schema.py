import json, os, types
import importlib.util

# Dynamically load the live-bot module without executing handler server pieces.
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'live-bot.py')

spec = importlib.util.spec_from_file_location('live_bot_mod', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_dashboard_status_schema_minimal():
    bot = mod.VercelLiveBot()
    data = bot.get_dashboard_status()
    # Required top-level keys
    for key in ['status', 'signal', 'account_balance', 'market_data', 'trading_summary']:
        assert key in data, f"Missing key: {key}"
    assert data['status'] == 'success'
    assert isinstance(data['account_balance'], dict)
    assert 'current_price' in data['market_data']
    assert 'total_trades' in data['trading_summary']

