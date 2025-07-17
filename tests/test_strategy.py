# ==============================================================================
#
#                               STRATEGY TESTS
#
# ==============================================================================
#
# FILE: tests/test_strategy.py
#
# PURPOSE:
#   This module contains unit tests for the `Strategy` class.
#
# ==============================================================================

import unittest
import numpy as np
import pandas as pd
from strategy import Strategy

class TestStrategy(unittest.TestCase):
    """
    A class to test the Strategy class.
    """
    def setUp(self):
        """
        Set up the test case.
        """
        self.params = {
            'TENKAN_SEN_PERIOD': 9,
            'KIJUN_SEN_PERIOD': 26,
            'SENKOU_SPAN_B_PERIOD': 52,
            'RSI_PERIOD': 14,
            'ADX_PERIOD': 14,
            'ATR_PERIOD': 14,
            'BBANDS_PERIOD': 20,
            'BBANDS_STD_DEV': 2,
            'RSI_OVERBOUGHT': 70,
            'RSI_OVERSOLD': 30,
            'ADX_TREND_THRESHOLD': 25,
            'BB_WIDTH_THRESHOLD': 0.01,
            'RSI_LOOKBACK': 1,
            'USE_ICHIMOKU_CLOUD_FILTER': True,
            'USE_ICHIMOKU_TENKAN_KIJUN_CROSS_FILTER': True,
            'USE_ICHIMOKU_CHIKOU_SPAN_FILTER': True,
            'USE_RSI_FILTER': True,
            'USE_ADX_FILTER': True,
            'USE_BBANDS_FILTER': True,
            'min_long_conditions': 6,
            'min_short_conditions': 6
        }
        self.strategy = Strategy(self.params)
        # Create a more realistic and sufficiently large DataFrame for testing
        data = {
            'high': np.random.random(200) * 100 + 1000,
            'low': np.random.random(200) * 100 + 900,
            'close': np.random.random(200) * 100 + 950,
            'open': np.random.random(200) * 100 + 950
        }
        self.df = pd.DataFrame(data)
        # Ensure low is always less than or equal to high
        self.df['low'] = self.df.apply(lambda row: min(row['low'], row['high']), axis=1)


    def test_generate_signals(self):
        """
        Test the generate_signals method.
        """
        signals_df = self.strategy.generate_signals(self.df)
        self.assertIsInstance(signals_df, pd.DataFrame)
        self.assertFalse(signals_df.empty)
        self.assertIn('long_signals', signals_df.columns)
        self.assertIn('short_signals', signals_df.columns)

if __name__ == '__main__':
    unittest.main()