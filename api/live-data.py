"""Live Data API (lightweight)
Provides minimal health + timestamp so Vercel build referencing this file succeeds.
Avoids expensive imports to keep cold start fast.
"""

import json
import math
import random
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler


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


class handler(BaseHTTPRequestHandler):  # Vercel Python runtime entrypoint
	def do_GET(self):
		try:
			payload = _base_payload()
			self.send_response(200)
			self.send_header('Content-Type', 'application/json')
			self.send_header('Access-Control-Allow-Origin', '*')
			self.end_headers()
			self.wfile.write(json.dumps(payload).encode())
		except Exception as e:  # pragma: no cover (defensive)
			self.send_response(500)
			self.send_header('Content-Type', 'application/json')
			self.end_headers()
			self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode())

	def do_OPTIONS(self):  # CORS preflight
		self.send_response(200)
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
		self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
		self.end_headers()


if __name__ == "__main__":  # Simple local smoke test
	print(json.dumps(_base_payload(), indent=2))
