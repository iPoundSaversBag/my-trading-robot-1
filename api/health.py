"""Lightweight health endpoint for dashboard & uptime checks."""
import json, time, os
from datetime import datetime, timezone

def _summarize_trades():
    path = 'trade_log.json'
    if not os.path.exists(path):
        return {"cumulative_realized_pnl": 0, "last_trade_ts": None}
    try:
        with open(path, 'r') as f:
            trades = json.load(f)
        cum = 0
        last_ts = None
        for t in trades:
            pnl = t.get('pnl')
            if pnl:
                cum += pnl
            if t.get('ts'):
                last_ts = max(last_ts or 0, t['ts'])
        return {"cumulative_realized_pnl": round(cum,2), "last_trade_ts": last_ts}
    except Exception:
        return {"cumulative_realized_pnl": None, "last_trade_ts": None}

def handler(request):
    # Basic metadata; could be extended to read a shared status file
    trade_summary = _summarize_trades()
    payload = {
        "status": "ok",
        "service": "health",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": os.environ.get('VERCEL_GIT_COMMIT_SHA', '')[:7],
        "region": os.environ.get('VERCEL_REGION', ''),
        "uptime_hint": int(time.time()),
        **trade_summary
    }
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Cache-Control': 'no-cache'
        },
        'body': json.dumps(payload)
    }
