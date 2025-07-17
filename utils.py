# ==============================================================================
#
#                               UTILITY FUNCTIONS
#
# ==============================================================================
#
# FILE: utils.py
#
# PURPOSE:
#   This module provides a centralized collection of utility functions that are
#   shared across the entire trading robot pipeline. This includes data
#   preparation, performance calculations, and other helper functions.
#
# ==============================================================================

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, average_true_range

# (This function is now deprecated and has been moved into the Strategy class)