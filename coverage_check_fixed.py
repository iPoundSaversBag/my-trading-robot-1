import pandas as pd, glob, os, math

def load(path):
    df=pd.read_parquet(path)
    # choose timestamp column
    for c in ['timestamp','open_time','time','date','datetime']:
        if c in df.columns:
            col=c; break
    else:
        raise ValueError(f'No ts column in {path}, cols={df.columns}')
    s=df[col]
    if pd.api.types.is_integer_dtype(s.dtype):
        v=int(s.iloc[0])
        # Epoch ms ~ 1.6e12; treat 1e11-1e13 as ms
        if 1e11 < v < 1e13:
            ts=pd.to_datetime(s, unit='ms', utc=True)
        elif v < 1e11:
            ts=pd.to_datetime(s, unit='s', utc=True)
        else:
            # assume already ns
            ts=pd.to_datetime(s, utc=True)
    else:
        ts=pd.to_datetime(s, utc=True)
    return ts

rows=[]
for f in sorted(glob.glob('data/crypto_data_*.parquet')):
    ts=load(f)
    name=os.path.basename(f)
    start,end=ts.min(),ts.max(); span=end-start; count=len(ts)
    if '5m' in name: interval_min=5
    elif '15m' in name: interval_min=15
    elif '_1h' in name: interval_min=60
    elif '_4h' in name: interval_min=240
    else: interval_min=None
    if interval_min:
        theoretical=math.floor(span.total_seconds()/60/interval_min)+1
        completeness=100*count/theoretical if theoretical>0 else 0
    else:
        theoretical=None; completeness=None
    rows.append((name,count,start,end,span.days,completeness))

print('COVERAGE SUMMARY (corrected)')
print('-'*95)
for name,count,start,end,days,comp in rows:
    if comp is not None:
        print(f'{name:25s} start={start} end={end} rows={count:7d} span_days={days:4d} completeness={comp:7.2f}%')
    else:
        print(f'{name:25s} start={start} end={end} rows={count:7d} span_days={days:4d}')
