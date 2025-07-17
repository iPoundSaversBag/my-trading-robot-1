# ==============================================================================
#
#                               PORTFOLIO TESTS
#
# ==============================================================================
#
# FILE: tests/test_portfolio.py
#
# PURPOSE:
#   This module contains unit tests for the `Portfolio` class.
#
# ==============================================================================

import unittest
import numpy as np
from portfolio import Portfolio

class TestPortfolio(unittest.TestCase):
    """
    A class to test the Portfolio class.
    """
    def setUp(self):
        """
        Set up the test case.
        """
        self.portfolio = Portfolio(initial_capital=10000)

    def test_add_trade(self):
        """
        Test the add_trade method.
        """
        trade = {
            'final_equity': 10100
        }
        self.portfolio.add_trade(trade)
        self.assertEqual(self.portfolio.capital, 10100)
        self.assertEqual(len(self.portfolio.history), 1)

    def test_get_equity_curve(self):
        """
        Test the get_equity_curve method.
        """
        trade1 = {
            'final_equity': 10100
        }
        trade2 = {
            'final_equity': 10050
        }
        self.portfolio.add_trade(trade1)
        self.portfolio.add_trade(trade2)
        equity_curve = self.portfolio.get_equity_curve()
        self.assertTrue(np.array_equal(equity_curve, np.array([10000, 10100, 10050])))

if __name__ == '__main__':
    unittest.main()