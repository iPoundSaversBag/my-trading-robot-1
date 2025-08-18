import os
import pandas as pd
from datetime import datetime

def summarize_file(path: str):
    if not os.path.exists(path):
        return {'file': path, 'status': 'MISSING'}
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {'file': path, 'status': f'read_failed: {e}'}
    # Try to ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ['timestamp','date','datetime','time']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
                except Exception:
                    pass
    if not isinstance(df.index, pd.DatetimeIndex):
        return {'file': path, 'status': 'no_datetime_index'}
    start = df.index.min()
    end = df.index.max()
    span_days = (end - start).days
    years = span_days / 365.25
    return {
        'file': path,
        'status': 'ok',
        'rows': len(df),
        'start': start, 'end': end,
        'span_days': span_days,
        'approx_years': years
    }

FILES = [
    'data/crypto_data_5m.parquet',
    'data/crypto_data_15m.parquet',
    'data/crypto_data_1h.parquet',
    'data/crypto_data_4h.parquet'
]

print('--- DATA RANGE SUMMARY ---')
results = [summarize_file(f) for f in FILES]
for r in results:
    if r['status'] != 'ok':
        print(f"{r['file']}: {r['status']}")
    else:
        print(f"{r['file']}: rows={r['rows']:,} start={r['start']} end={r['end']} span_days={r['span_days']} (~{r['approx_years']:.2f}y)")

# Overall earliest start and latest end among available files
ok_results = [r for r in results if r['status'] == 'ok']
if ok_results:
    earliest = min(r['start'] for r in ok_results)
    latest = max(r['end'] for r in ok_results)
    total_days = (latest - earliest).days
    total_years = total_days / 365.25
    print(f"Overall span across available files: {total_days} days (~{total_years:.2f}y) {earliest} -> {latest}")
    if total_years >= 3.8:
        print("Coverage check: PASS (≈4 years of data available)")
    else:
        print("Coverage check: WARN (< ~3.8 years)")
else:
    print('No readable files with datetime index.')
