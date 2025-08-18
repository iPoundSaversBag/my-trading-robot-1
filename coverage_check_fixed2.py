import pandas as pd, glob, os, math, re

INTERVAL_MAP = [('_15m',15), ('15m',15), ('_5m',5), ('5m',5), ('_1h',60), ('1h',60), ('_4h',240), ('4h',240)]

def infer_interval(name):
    for key,val in INTERVAL_MAP:
        if name.endswith(key) or key in name:
            return val
    return None

def load(path):
    df=pd.read_parquet(path)
    for c in ['timestamp','open_time','time','date','datetime']:
        if c in df.columns:
            s=df[c]; break
    else:
        raise ValueError(f'No ts col in {path}')
    if pd.api.types.is_integer_dtype(s.dtype):
        v=int(s.iloc[0])
        if 1e11 < v < 1e13:
            ts=pd.to_datetime(s, unit='ms', utc=True)
        elif v < 1e11:
            ts=pd.to_datetime(s, unit='s', utc=True)
        else:
            ts=pd.to_datetime(s, utc=True)
    else:
        ts=pd.to_datetime(s, utc=True)
    return ts

print('COVERAGE SUMMARY (corrected ordering)')
print('-'*110)
for f in sorted(glob.glob('data/crypto_data_*.parquet')):
    name=os.path.basename(f)
    ts=load(f)
    start,end=ts.min(),ts.max(); span=end-start; count=len(ts)
    interval=infer_interval(name)
    if interval:
        theoretical=math.floor(span.total_seconds()/60/interval)+1
        completeness=100*count/theoretical
        print(f'{name:25s} interval={interval:4d}m start={start} end={end} rows={count:7d} span_days={span.days:4d} completeness={completeness:7.2f}% (theoretical={theoretical})')
    else:
        print(f'{name:25s} start={start} end={end} rows={count:7d} span_days={span.days:4d}')
