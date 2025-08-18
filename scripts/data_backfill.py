"""
Data Backfill Utility
---------------------
Extends existing multi-timeframe parquet datasets back to the full
`years_of_data_to_download` specified in `core/optimization_config.json`.

Why this script?
- Current dataset spans ~3 years but config requests 4.
- Existing downloader replaces files starting from target start date (destructive refresh).
- This script NON-DESTRUCTIVELY prepends missing history only when needed.

Behavior:
1. Loads config to obtain symbol, exchange, years_of_data_to_download, and timeframe file map.
2. For each timeframe file:
   - If file missing -> skip (user should run standard downloader first) OR optionally full download.
   - Reads earliest timestamp present.
   - Computes required target_start = now - years*365.25 days.
   - If earliest <= target_start + tolerance -> skip (already sufficient).
   - Otherwise asynchronously fetches OHLCV candles from target_start up to (earliest - one timeframe) and merges.
3. Cleans data (remove zero volume, extreme wicks) and re-saves parquet sorted & deduped.

Safe Guards:
- Writes a temporary parquet (filename + ".backfill_tmp") first; only replaces original on success.
- Progress + summary printed.

Run:
    python scripts/data_backfill.py

Optional Flags:
    --force : Ignore sufficiency check and attempt full reconstruction.
    --dry   : Do not write changes; just report what would be done.

Dependencies: ccxt.pro (async), pandas.
"""
from __future__ import annotations
import asyncio
import datetime as dt
import json
import os
import sys
import traceback
from typing import Dict, List

import pandas as pd
import ccxt.pro as ccxt

CONFIG_PATH = os.path.join("core", "optimization_config.json")
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

MAX_FETCH_LIMIT = 1000
RETRIES = 5


def human_dt(ms: int) -> str:
    return pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def calc_required_start(years: float) -> dt.datetime:
    return dt.datetime.utcnow() - dt.timedelta(days=years * 365.25)


async def fetch_range(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> List[List[int]]:
    collected = []
    seen = set()
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    current = start_ms
    while current < end_ms:
        remaining_ms = end_ms - current
        # Log progress approximately every ~50k candles
        try:
            # Retry loop
            chunk = None
            for attempt in range(RETRIES):
                try:
                    chunk = await exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=MAX_FETCH_LIMIT)
                    break
                except ccxt.NetworkError as e:
                    wait = 2 ** attempt
                    print(f"[NET] {timeframe} retry {attempt+1}/{RETRIES} after error: {e} (sleep {wait}s)")
                    await asyncio.sleep(wait)
                except Exception as e:
                    print(f"[ERR] Unexpected fetch error @ {human_dt(current)}: {e}")
                    if attempt == RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
            if not chunk:
                print("[STOP] No more data returned (chunk empty).")
                break
            new_rows = 0
            for candle in chunk:
                ts = int(candle[0])
                if ts in seen or ts >= end_ms:
                    continue
                seen.add(ts)
                collected.append(candle)
                new_rows += 1
            if new_rows == 0:
                print("[STOP] Reached existing boundary (no new rows).")
                break
            current = chunk[-1][0] + tf_ms
            if len(collected) % 50000 < new_rows:  # crossed a multiple of 50k
                print(f"[PROGRESS] {timeframe} collected {len(collected):,} rows; current {human_dt(current)}")
            await asyncio.sleep(max(exchange.rateLimit / 1000, 0.15))
        except KeyboardInterrupt:
            print("[ABORT] User interrupted fetch.")
            break
    return collected


async def backfill_timeframe(symbol: str, exchange_name: str, timeframe: str, path: str, target_start: dt.datetime, force: bool, dry: bool) -> Dict:
    result = {"timeframe": timeframe, "status": "skipped", "added": 0, " earliest_before": None, "new_earliest": None}
    if timeframe not in TIMEFRAME_SECONDS:
        print(f"[SKIP] Unsupported timeframe {timeframe}")
        return result
    if not os.path.exists(path):
        print(f"[MISS] {path} not found. Run main downloader first (or implement full fetch here).")
        return result

    df = pd.read_parquet(path)
    if df.empty:
        print(f"[WARN] {path} empty. Re-run main downloader.")
        return result

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    earliest = df.index.min().to_pydatetime()
    target_ms = int(target_start.timestamp() * 1000)
    earliest_ms = int(earliest.timestamp() * 1000)
    result["earliest_before"] = earliest.isoformat()

    if not force and earliest <= target_start + dt.timedelta(days=1):
        print(f"[OK] {timeframe} already sufficient (earliest {earliest} <= target {target_start.date()}).")
        return result

    print(f"[BACKFILL] {timeframe}: need data from {target_start} to existing earliest {earliest}")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({'enableRateLimit': True})
    try:
        rows = await fetch_range(exchange, symbol, timeframe, target_ms, earliest_ms)
        if not rows:
            print(f"[NOOP] No rows fetched for {timeframe}; nothing to merge.")
            return result
        add_df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
        # Convert to timezone-naive UTC timestamps; avoid tz_convert on naive index
        add_df['datetime'] = pd.to_datetime(add_df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
        add_df.set_index('datetime', inplace=True)
        # Cleaning
        before = len(add_df)
        add_df = add_df[add_df['volume'] > 0]
        add_df = add_df[add_df['low'] > (add_df['high'] * 0.5)]
        if len(add_df) < before:
            print(f"[CLEAN] Removed {before - len(add_df)} anomalous rows ({timeframe}).")
        merged = pd.concat([add_df, df])
        merged.sort_index(inplace=True)
        merged = merged[~merged.index.duplicated(keep='first')]
        tmp_path = path + ".backfill_tmp"
        if dry:
            print(f"[DRY] Would add {len(add_df):,} rows; earliest would become {merged.index.min()}")
        else:
            merged.to_parquet(tmp_path)
            os.replace(tmp_path, path)
            print(f"[DONE] Added {len(add_df):,} rows. New span: {merged.index.min()} -> {merged.index.max()} ({len(merged):,} rows)")
        result.update({"status": "backfilled", "added": len(add_df), "new_earliest": str(merged.index.min())})
        return result
    except Exception as e:
        print(f"[ERROR] Backfill failed for {timeframe}: {e}")
        traceback.print_exc()
        result['status'] = 'error'
        return result
    finally:
        try:
            await exchange.close()
        except Exception:
            pass


async def main(force: bool = False, dry: bool = False):
    print("=== DATA BACKFILL START ===")
    cfg = load_config()
    bot = cfg.get('bot_settings', {})
    data_cfg = cfg.get('data_settings', {})
    years = bot.get('years_of_data_to_download', 4)
    exchange_name = bot.get('exchange_name', 'binance')
    symbol = bot.get('symbol', 'BTC/USDT')
    timeframe_files = data_cfg.get('timeframe_files') or {bot.get('timeframe','5m'): data_cfg.get('file_path','data/crypto_data_5m.parquet')}
    target_start = calc_required_start(years)

    print(f"Target years: {years} -> start {target_start.date()} (UTC)")
    print(f"Symbol: {symbol} Exchange: {exchange_name}")
    print(f"Force: {force} Dry: {dry}")

    results = []
    for tf, path in timeframe_files.items():
        results.append(await backfill_timeframe(symbol, exchange_name, tf, path, target_start, force, dry))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['timeframe']}: {r['status']} added={r['added']} earliest_before={r.get('earliest_before')} new_earliest={r.get('new_earliest')}")
    print("=== DATA BACKFILL END ===")


if __name__ == '__main__':
    force = '--force' in sys.argv
    dry = '--dry' in sys.argv
    try:
        asyncio.run(main(force=force, dry=dry))
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
