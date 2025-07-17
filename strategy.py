# ==============================================================================
#
#                               STRATEGY CLASS
#
# ==============================================================================
#
# FILE: strategy.py
#
# PURPOSE:
#   This module defines the base `Strategy` class, which encapsulates the
#   complete logic for a trading strategy. This includes the indicators, entry
#   signals, and exit signals.
#
# ==============================================================================

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator, ichimoku_a, ichimoku_b
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, average_true_range

class Strategy:
    """
    A base class for a trading strategy. It's responsible for generating
    all technical indicators and trading signals.
    """
    def __init__(self, params):
        self.params = params

    def generate_signals(self, df, realism_settings={}):
        """
        Calculates all necessary technical indicators and signals for a given DataFrame.
        """
        try:
            df_processed = df.copy()
            
            # Ensure params are integers where required
            for p in ['TENKAN_SEN_PERIOD', 'KIJUN_SEN_PERIOD', 'SENKOU_SPAN_B_PERIOD', 'RSI_PERIOD', 'ADX_PERIOD', 'ATR_PERIOD', 'BBANDS_PERIOD']:
                if p in self.params:
                    self.params[p] = int(self.params[p])

            # Ichimoku Cloud
            df_processed['tenkan'] = ichimoku_a(df_processed['high'], df_processed['low'], window1=self.params['TENKAN_SEN_PERIOD'])
            df_processed['kijun'] = ichimoku_b(df_processed['high'], df_processed['low'], window2=self.params['KIJUN_SEN_PERIOD'])
            df_processed['senkou_a'] = ((df_processed['tenkan'] + df_processed['kijun']) / 2).shift(self.params['KIJUN_SEN_PERIOD'])
            df_processed['senkou_b'] = ((df_processed['high'].rolling(window=self.params['SENKOU_SPAN_B_PERIOD']).max() + 
                                         df_processed['low'].rolling(window=self.params['SENKOU_SPAN_B_PERIOD']).min()) / 2).shift(self.params['KIJUN_SEN_PERIOD'])
            
            # Other Indicators
            df_processed['rsi'] = RSIIndicator(close=df_processed['close'], window=self.params['RSI_PERIOD']).rsi()
            df_processed['adx'] = ADXIndicator(high=df_processed['high'], low=df_processed['low'], close=df_processed['close'], window=self.params['ADX_PERIOD']).adx()
            df_processed['atr'] = average_true_range(high=df_processed['high'], low=df_processed['low'], close=df_processed['close'], window=self.params['ATR_PERIOD'])
            
            bbands = BollingerBands(close=df_processed['close'], window=self.params['BBANDS_PERIOD'], window_dev=self.params['BBANDS_STD_DEV'])
            df_processed['bb_high'] = bbands.bollinger_hband()
            df_processed['bb_low'] = bbands.bollinger_lband()
            df_processed['bb_width'] = bbands.bollinger_wband()
            df_processed['bb_mid'] = bbands.bollinger_mavg()

            # --- REALISM: Add ATR for variable slippage calculation ---
            if realism_settings.get('enabled', False):
                vol_atr_period = realism_settings.get('volatility_atr_period', 14)
                df_processed['volatility_atr'] = average_true_range(high=df_processed['high'], low=df_processed['low'], close=df_processed['close'], window=vol_atr_period)
            else:
                df_processed['volatility_atr'] = 0.0

            # Define Trend
            adx_threshold = self.params['ADX_TREND_THRESHOLD']
            bb_width_threshold = self.params['BB_WIDTH_THRESHOLD']
            df_processed['is_trending'] = (df_processed['adx'] > adx_threshold) & (df_processed['bb_width'] > bb_width_threshold)

            # --- Comprehensive Ichimoku Signal Generation ---
            chikou_span_period = self.params['KIJUN_SEN_PERIOD']
            
            is_cloud_bullish = df_processed['senkou_a'] > df_processed['senkou_b']
            is_cloud_bearish = df_processed['senkou_a'] < df_processed['senkou_b']
            is_tk_bullish = df_processed['tenkan'] > df_processed['kijun']
            is_tk_bearish = df_processed['tenkan'] < df_processed['kijun']
            is_price_above_cloud = df_processed['close'] > df_processed['senkou_a']
            is_price_below_cloud = df_processed['close'] < df_processed['senkou_b']
            is_chikou_above_price = df_processed['close'] > df_processed['close'].shift(chikou_span_period)
            is_chikou_below_price = df_processed['close'] < df_processed['close'].shift(chikou_span_period)

            rsi_lookback = self.params.get('RSI_LOOKBACK', 1)
            is_not_overbought = df_processed['rsi'] < self.params['RSI_OVERBOUGHT']
            is_not_oversold = df_processed['rsi'] > self.params['RSI_OVERSOLD']
            rsi_is_rising = df_processed['rsi'] > df_processed['rsi'].shift(rsi_lookback)
            rsi_is_falling = df_processed['rsi'] < df_processed['rsi'].shift(rsi_lookback)

            # Combine all conditions for a strong signal
            long_conditions = []
            short_conditions = []

            if self.params.get('USE_ICHIMOKU_CLOUD_FILTER', True):
                long_conditions.append(is_price_above_cloud)
                short_conditions.append(is_price_below_cloud)
            if self.params.get('USE_ICHIMOKU_TENKAN_KIJUN_CROSS_FILTER', True):
                long_conditions.append(is_tk_bullish)
                short_conditions.append(is_tk_bearish)
            if self.params.get('USE_ICHIMOKU_CHIKOU_SPAN_FILTER', True):
                long_conditions.append(is_chikou_above_price)
                short_conditions.append(is_chikou_below_price)
            if self.params.get('USE_RSI_FILTER', True):
                long_conditions.append(is_not_overbought)
                long_conditions.append(rsi_is_rising)
                short_conditions.append(is_not_oversold)
                short_conditions.append(rsi_is_falling)
            if self.params.get('USE_ADX_FILTER', True):
                long_conditions.append(df_processed['is_trending'])
                short_conditions.append(df_processed['is_trending'])
            if self.params.get('USE_BBANDS_FILTER', True):
                long_conditions.append(df_processed['close'] > df_processed['bb_mid'])
                short_conditions.append(df_processed['close'] < df_processed['bb_mid'])

            min_long_cond = self.params.get('min_long_conditions', len(long_conditions))
            min_short_cond = self.params.get('min_short_conditions', len(short_conditions))

            df_processed['long_signals'] = (pd.DataFrame(long_conditions).sum(axis=0) >= min_long_cond) if long_conditions else pd.Series(False, index=df_processed.index)
            df_processed['short_signals'] = (pd.DataFrame(short_conditions).sum(axis=0) >= min_short_cond) if short_conditions else pd.Series(False, index=df_processed.index)
            
            df_processed.dropna(inplace=True)
            return df_processed
        except Exception as e:
            print(f"Error preparing data in Strategy: {e}. Params: {self.params}")
            return pd.DataFrame()
