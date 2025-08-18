import pandas as pd, glob, os, math
from datetime import timezone

def load_ts(path):
    df=pd.read_parquet(path)
    # pick timestamp column heuristically
    for cand in ['timestamp','time','date','datetime','open_time']:
        if cand in df.columns:
            s=df[cand]
            break
    else:
        raise ValueError(f'No timestamp-like column in {path}, columns={df.columns.tolist()}')
    if pd.api.types.is_integer_dtype(s.dtype):
        # decide unit
        if s.iloc[0] > 10**14: # nanoseconds
            ts=pd.to_datetime(s, utc=True)
        elif s.iloc[0] > 10**12: # microseconds (unlikely)
            ts=pd.to_datetime(s, unit='us', utc=True)
        elif s.iloc[0] > 10**11: # milliseconds typical
            ts=pd.to_datetime(s, unit='ms', utc=True)
        else:
            ts=pd.to_datetime(s, unit='s', utc=True)
    else:
        ts=pd.to_datetime(s, utc=True)
    return ts

rows=[]
for f in sorted(glob.glob('data/crypto_data_*.parquet')):
    ts=load_ts(f)
    count=len(ts)
    start=ts.min(); end=ts.max(); span=end-start
    rows.append((os.path.basename(f), count, start, end, span))

print('\nCOVERAGE SUMMARY')
print('-'*80)
for name,count,start,end,span in rows:
    if '5m' in name: interval_min=5
    elif '15m' in name: interval_min=15
    elif '_1h' in name: interval_min=60
    elif '_4h' in name: interval_min=240
    else: interval_min=None
    if interval_min:
        theoretical = math.floor(span.total_seconds()/60/interval_min)+1
        completeness = 100*count/theoretical if theoretical>0 else 0
        print(f'{name:25s} start={start} end={end} rows={count:7d} span_days={span.days:4d} completeness={completeness:6.2f}%')
    else:
        print(f'{name:25s} start={start} end={end} rows={count:7d} span_days={span.days:4d}')
