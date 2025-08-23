#!/usr/bin/env python3
"""
Validate the effectiveness of the production regime detector on the 5m timeframe.
This script verifies that the system still meets the 98% effectiveness target
after being recalibrated for the primary trading timeframe.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from core.production_regime_detector import ProductionRegimeDetector, MarketRegime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EffectivenessValidator:
    def __init__(self):
        self.detector = None
        self.data = None
        self.config = None

    def load_dependencies(self):
        """Load data, config, and initialize the detector."""
        logger.info("Loading dependencies...")
        
        # Load data
        data_path = project_root / "data/crypto_data_5m.parquet"
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False
        self.data = pd.read_parquet(data_path)
        logger.info(f"Loaded 5m data with {len(self.data)} rows.")

        # Load config
        config_path = project_root / "production_regime_config.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logger.info("Loaded production regime configuration.")

        # Initialize detector
        self.detector = ProductionRegimeDetector(self.config)
        return True

    def find_test_scenarios(self, num_scenarios=25):
        """Find clear examples of different regimes using raw price action."""
        logger.info(f"Finding {num_scenarios * 4} test scenarios using raw price action...")
        df = self.data.copy()
        
        scenarios = []
        window_size = 48  # 4 hours of 5m data
        step = 24 # 2 hour step to get overlapping windows

        # Calculate rolling features across the entire dataset once
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window_size).std()
        df['price_change'] = df['close'].rolling(window_size).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)

        df = df.dropna()

        # Define dynamic thresholds based on data distribution
        high_vol_threshold = df['volatility'].quantile(0.85)
        low_vol_threshold = df['volatility'].quantile(0.15)
        strong_trend_threshold = df['price_change'].abs().quantile(0.85)
        
        # Find Bull Trend scenarios
        bull_indices = df[(df['price_change'] > strong_trend_threshold) & (df['volatility'] > low_vol_threshold)].index
        if len(bull_indices) > num_scenarios:
            for idx in np.random.choice(bull_indices, num_scenarios, replace=False):
                start_loc = df.index.get_loc(idx)
                if start_loc >= window_size:
                    scenarios.append({'type': MarketRegime.TRENDING_BULL, 'data': df.iloc[start_loc-window_size:start_loc]})

        # Find Bear Trend scenarios
        bear_indices = df[(df['price_change'] < -strong_trend_threshold) & (df['volatility'] > low_vol_threshold)].index
        if len(bear_indices) > num_scenarios:
            for idx in np.random.choice(bear_indices, num_scenarios, replace=False):
                start_loc = df.index.get_loc(idx)
                if start_loc >= window_size:
                    scenarios.append({'type': MarketRegime.TRENDING_BEAR, 'data': df.iloc[start_loc-window_size:start_loc]})

        # Find High Volatility scenarios
        vol_indices = df[df['volatility'] > high_vol_threshold].index
        if len(vol_indices) > num_scenarios:
            for idx in np.random.choice(vol_indices, num_scenarios, replace=False):
                start_loc = df.index.get_loc(idx)
                if start_loc >= window_size:
                    scenarios.append({'type': MarketRegime.HIGH_VOLATILITY, 'data': df.iloc[start_loc-window_size:start_loc]})
            
        # Find Ranging scenarios
        ranging_indices = df[(df['volatility'] < low_vol_threshold) & (df['price_change'].abs() < strong_trend_threshold * 0.25)].index
        if len(ranging_indices) > num_scenarios:
            for idx in np.random.choice(ranging_indices, num_scenarios, replace=False):
                start_loc = df.index.get_loc(idx)
                if start_loc >= window_size:
                    scenarios.append({'type': MarketRegime.RANGING, 'data': df.iloc[start_loc-window_size:start_loc]})

        logger.info(f"Found {len(scenarios)} scenarios to test.")
        return scenarios

    def run_validation(self):
        """Run validation and report effectiveness."""
        if not self.load_dependencies():
            return

        scenarios = self.find_test_scenarios()
        if not scenarios:
            logger.error("Could not find any test scenarios.")
            return

        correct_predictions = 0
        total_predictions = len(scenarios)
        
        results = []

        for i, scenario in enumerate(scenarios):
            expected_regime = scenario['type']
            data_slice = scenario['data']
            
            if len(data_slice) < self.detector.min_data_points_production:
                continue

            detected_regime, confidence = self.detector.detect_regime(data_slice)
            
            is_correct = detected_regime == expected_regime
            if is_correct:
                correct_predictions += 1
            
            results.append({
                "scenario": i + 1,
                "expected": expected_regime.name,
                "detected": detected_regime.name,
                "confidence": confidence,
                "correct": is_correct
            })

        if total_predictions == 0:
            logger.error("No predictions were made, cannot calculate effectiveness.")
            return 0.0

        effectiveness = (correct_predictions / total_predictions) * 100
        
        self.print_report(effectiveness, results)
        
        return effectiveness

    def print_report(self, effectiveness, results):
        """Print a detailed report of the validation results."""
        print("\n" + "="*60)
        print("🔬 PRODUCTION REGIME DETECTOR EFFECTIVENESS REPORT (5m)")
        print("="*60)
        
        df_results = pd.DataFrame(results)
        
        print(f"\n🎯 Overall Effectiveness: {effectiveness:.2f}%")
        if effectiveness >= 98.0:
            print("✅ STATUS: PASSED (Effectiveness target of 98% met or exceeded)")
        else:
            print("❌ STATUS: FAILED (Effectiveness below 98% target)")

        if not df_results.empty:
            print("\n📊 Regime-wise Accuracy:")
            # Ensure all expected regimes are columns, even if not present in results
            all_regimes = [r.name for r in MarketRegime]
            accuracy_by_regime = df_results.groupby('expected')['correct'].value_counts(normalize=True).unstack().fillna(0)
            if 'correct' in accuracy_by_regime.columns and True in accuracy_by_regime['correct']:
                 print((accuracy_by_regime[True] * 100).reindex(all_regimes, fill_value=0).round(2))
            else:
                print("No correct predictions to report.")

            print("\n📋 Summary:")
            print(f"   - Total Scenarios Tested: {len(results)}")
            print(f"   - Correct Predictions: {df_results['correct'].sum()}")
            print(f"   - Incorrect Predictions: {len(results) - df_results['correct'].sum()}")

            incorrect = df_results[df_results['correct'] == False]
            if not incorrect.empty:
                print("\n🔍 Analysis of Incorrect Predictions:")
                print(incorrect.head())
        else:
            print("\n📋 No results to report.")
        
        print("\n" + "="*60)


def main():
    """Main function to run the validator."""
    validator = EffectivenessValidator()
    effectiveness = validator.run_validation()
    
    if effectiveness is None:
        sys.exit(1)
        
    if effectiveness >= 98.0:
        logger.info("Validation successful. The system is highly effective on the 5m timeframe.")
        sys.exit(0)
    else:
        logger.error("Validation failed. Effectiveness is below the required threshold.")
        sys.exit(1)

if __name__ == "__main__":
    main()
