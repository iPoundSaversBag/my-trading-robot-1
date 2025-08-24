"""Live Data API (lightweight)
Provides minimal health + timestamp so Vercel build referencing this file succeeds.
Avoids expensive imports to keep cold start fast.
"""

import json
import math
import random
import time
from datetime import datetime, timezone


def _simulate_price():
	"""Lightweight deterministic-ish price simulation (avoids external API)."""
	# Base anchor price; add a sinusoidal + small random jitter
	base = 114000
	t = time.time()
	wave = math.sin(t / 60) * 400  # ~minute wave +/-400
	jitter = random.Random(int(t // 5)).uniform(-150, 150)  # stable within 5s window
	return round(base + wave + jitter, 2)


def _base_payload():
	price = _simulate_price()
	# Pseudo account balances
	usdt_balance = 10000 + (price % 500)  # vary a bit with price
	btc_balance = round(0.05 + (price % 1000) / 100000, 5)

	confidence = round(abs(math.sin(time.time() / 90)) * 0.9, 3)
	signal = 'BUY' if math.sin(time.time() / 120) > 0 else 'SELL'

	trade_side = 'BUY' if signal == 'BUY' else 'SELL'
	quantity = round(0.001 + (confidence * 0.002), 4)

	return {
		"status": "ok",
		"service": "live-data",
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"account_balance": {
			"BTC": btc_balance,
			"USDT": round(usdt_balance, 2)
		},
		"signal": {
			"signal": signal,
			"confidence": confidence,
			"current_price": price
		},
		"trade_executed": {
			"side": trade_side,
			"quantity": quantity,
			"price": price,
			"value": round(quantity * price, 2),
			"simulated": True
		},
		"latency_ms": random.randint(20, 70),
		"message": "Simulated live data (replace with real trading engine output)."
	}


def handler(request):  # Vercel Python runtime expects a function returning dict
	try:
		payload = _base_payload()
		return {
			'statusCode': 200,
			'headers': {
				'Content-Type': 'application/json',
				'Access-Control-Allow-Origin': '*',
				'Cache-Control': 'no-cache, no-store, must-revalidate',
				'Pragma': 'no-cache',
				'Expires': '0'
			},
			'body': json.dumps(payload)
		}
	except Exception as e:  # pragma: no cover
		return {
			'statusCode': 500,
			'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
			'body': json.dumps({'status': 'error', 'error': str(e)})
		}


if __name__ == "__main__":  # Simple local smoke test
	print(json.dumps(_base_payload(), indent=2))
