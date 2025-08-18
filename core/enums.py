from enum import Enum

class MarketRegime(Enum):
    """Enhanced market regime classification with 9 actionable types"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_BULLISH = "breakout_bullish"
    BREAKOUT_BEARISH = "breakout_bearish"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
