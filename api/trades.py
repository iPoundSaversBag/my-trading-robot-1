"""Return recent persisted trades (sanitized)."""
import json, os
from datetime import datetime, timezone

TRADE_LOG_PATH = 'trade_log.json'


def handler(request):
    try:
        read_token = os.environ.get('TRADES_READ_TOKEN')
        auth = None
        # Attempt to extract Authorization header if present (Vercel provides request.headers maybe)
        if hasattr(request, 'headers') and request.headers:
            auth = request.headers.get('authorization') or request.headers.get('Authorization')
        if read_token and auth != f'Bearer {read_token}':
            return {
                'statusCode': 401,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'status': 'error', 'error': 'unauthorized'})
            }
        limit = 50
        if hasattr(request, 'query'):  # Vercel style may not provide; ignore
            try:
                limit = int(request.query.get('limit', [50])[0])
            except Exception:
                pass
        trades = []
        if os.path.exists(TRADE_LOG_PATH):
            with open(TRADE_LOG_PATH, 'r') as f:
                trades = json.load(f)
        trades = trades[-limit:]
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-cache'},
            'body': json.dumps({
                'status': 'ok',
                'count': len(trades),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trades': trades
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'status': 'error', 'error': str(e)})
        }
