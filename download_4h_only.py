#!/usr/bin/env python3

"""
Download 4h Data Only
Simple script to download just the 4h timeframe data
"""

import asyncio
import datetime
import sys
import os

# Add parent directory to path to import data_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_manager import download_ohlcv_to_file

async def download_4h_only():
    """Download only 4h timeframe data"""
    print("=== DOWNLOADING 4H DATA ONLY ===")
    
    # Configuration
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '4h'
    YEARS_OF_DATA = 4
    FILENAME = 'data/crypto_data_4h.parquet'
    
    # Calculate start date
    start_date = datetime.datetime.now() - datetime.timedelta(days=YEARS_OF_DATA * 365.25)
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Start Date: {start_date_str}")
    print(f"Output File: {FILENAME}")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Download
    success = await download_ohlcv_to_file(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date_str=start_date_str,
        filename=FILENAME
    )
    
    if success:
        print(f"\n✅ SUCCESS: 4h data downloaded to {FILENAME}")
        # Verify file exists
        if os.path.exists(FILENAME):
            file_size = os.path.getsize(FILENAME)
            print(f"✅ File verified: {file_size:,} bytes")
        else:
            print(f"❌ ERROR: File {FILENAME} not found after download")
            return False
    else:
        print(f"\n❌ FAILED: 4h data download failed")
        return False
    
    return success

if __name__ == '__main__':
    try:
        result = asyncio.run(download_4h_only())
        exit_code = 0 if result else 1
        print(f"\nExiting with code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
