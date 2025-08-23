#!/usr/bin/env python3
"""
Test script to validate the multi-timeframe pipeline of the production regime detector.
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from core.production_regime_detector import ProductionRegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """Load 5m data for testing."""
    data_path = project_root / "data/crypto_data_5m.parquet"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded 5m data with {len(df)} rows.")
    return df

def main():
    """Run the validation test."""
    logger.info("🚀 Validating Multi-Timeframe Pipeline...")
    
    df = load_test_data()
    if df is None:
        sys.exit(1)
        
    # Take a recent slice of data for the test
    test_df = df.tail(5000) # Approx 17 days of 5m data
    
    detector = ProductionRegimeDetector()
    
    validation_result = detector.validate_multi_timeframe_pipeline(test_df)
    
    print("\n" + "="*60)
    print("🔬 MULTI-TIMEFRAME PIPELINE VALIDATION REPORT")
    print("="*60)
    
    print(f"Input Rows (5m): {validation_result['input_rows']}")
    
    print("\n📊 Aggregated Timeframes:")
    for tf, data in validation_result['timeframes'].items():
        print(f"  - {tf}: {data['rows']} rows (from {data['start']} to {data['end']})")
        
    print("\n⚙️ Sample Features (from 5m data):")
    for key, value in validation_result['sample_features'].items():
        print(f"  - {key}: {value}")
        
    print("\n⚠️ Issues:")
    if not validation_result['issues']:
        print("  - No issues found. ✅")
    else:
        for issue in validation_result['issues']:
            print(f"  - {issue} ❌")
            
    print("\n" + "="*60)
    
    if not validation_result['issues']:
        logger.info("✅ Validation successful! The multi-timeframe pipeline is working as expected.")
        sys.exit(0)
    else:
        logger.error("❌ Validation failed! Issues found in the pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()
