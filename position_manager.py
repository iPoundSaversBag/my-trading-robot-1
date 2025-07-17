# ==============================================================================
#
#                           UNIFIED POSITION MANAGER
#
# ==============================================================================
#
# FILE: position_manager.py
#
# PURPOSE:
#   This module defines the `PositionManager` class, which serves as the single
#   source of truth for all trade and position management logic. It is designed
#   to be used by both the backtester and the live trading bot to ensure that
#   the logic for entering, managing, and exiting trades is identical in both
#   simulation and production environments.
#
# ==============================================================================

class PositionManager:
    """
    A unified class to manage the state and logic of a single trade,
    to be used by both the backtester and the live bot.
    """
    def __init__(self, params):
        self.params = params
        self.in_position = False
        self.position_details = {}

    def load_state(self, state):
        """Loads the position state from a dictionary."""
        self.in_position = state.get('in_position', False)
        self.position_details = state.get('position_details', {})

    def get_state(self):
        """Returns the current position state as a dictionary."""
        return {'in_position': self.in_position, 'position_details': self.position_details}

    def check_for_entry(self, latest_candle):
        """
        Checks if an entry signal has occurred.
        
        Args:
            latest_candle (pd.Series): The most recent data candle.
        
        Returns:
            A tuple of (trade_type, entry_reason) or (None, None).
        """
        if self.in_position:
            return None, None

        if latest_candle.get('long_signal', False):
            return 'long', "Ichimoku Entry Signal"
        if latest_candle.get('short_signal', False):
            return 'short', "Ichimoku Entry Signal"
            
        return None, None

    def check_for_exit(self, latest_candle):
        """
        Checks for all possible exit conditions (stop loss, take profit, etc.).
        
        Args:
            latest_candle (pd.Series): The most recent data candle.
            
        Returns:
            The reason for the exit (str) or None if no exit condition is met.
        """
        if not self.in_position:
            return None

        pos_type = self.position_details['type']
        
        # 1. Check Hard Stop Loss
        if pos_type == 'long' and latest_candle['low'] <= self.position_details['hard_stop_loss']:
            return "Hard Stop Loss"
        if pos_type == 'short' and latest_candle['high'] >= self.position_details['hard_stop_loss']:
            return "Hard Stop Loss"

        # 2. Check Trailing Stop Loss
        if pos_type == 'long' and latest_candle['low'] <= self.position_details['trailing_stop_loss']:
            return "Trailing Stop Loss"
        if pos_type == 'short' and latest_candle['high'] >= self.position_details['trailing_stop_loss']:
            return "Trailing Stop Loss"
            
        # 3. Check Partial Take Profit
        if not self.position_details.get('partial_exit_taken', False):
            if pos_type == 'long' and latest_candle['high'] >= self.position_details['take_profit_price']:
                return "Partial Take Profit"
            if pos_type == 'short' and latest_candle['low'] <= self.position_details['take_profit_price']:
                return "Partial Take Profit"
        
        return None

    def update_trailing_stop(self, latest_candle):
        """Updates the trailing stop loss based on the latest price data."""
        if not self.in_position:
            return

        is_trending = latest_candle.get('is_trending', True)
        tsl_multiplier = self.params['TRENDING_TSL_ATR_MULTIPLIER'] if is_trending else self.params['RANGING_TSL_ATR_MULTIPLIER']
        
        if self.position_details['type'] == 'long':
            new_tsl = latest_candle['close'] - (latest_candle['atr'] * tsl_multiplier)
            self.position_details['trailing_stop_loss'] = max(self.position_details.get('trailing_stop_loss', -1), new_tsl)
        elif self.position_details['type'] == 'short':
            new_tsl = latest_candle['close'] + (latest_candle['atr'] * tsl_multiplier)
            self.position_details['trailing_stop_loss'] = min(self.position_details.get('trailing_stop_loss', float('inf')), new_tsl)

    def enter_position(self, trade_type, candle, balance_usd):
        """
        Calculates position details and updates the state to enter a new trade.
        
        Returns:
            bool: True if the position was successfully entered, False otherwise.
        """
        risk_per_trade = self.params.get('risk_per_trade_percentage', 0.01)
        risk_amount_usd = balance_usd * risk_per_trade
        entry_price = candle['close']
        atr = candle['atr']
        
        is_trending = candle.get('is_trending', True)
        tsl_multiplier = self.params['TRENDING_TSL_ATR_MULTIPLIER'] if is_trending else self.params['RANGING_TSL_ATR_MULTIPLIER']
        tp_multiplier = self.params['TRENDING_TP_ATR_MULTIPLIER'] if is_trending else self.params['RANGING_TP_ATR_MULTIPLIER']
        
        if trade_type == 'long':
            stop_loss_price = entry_price - (atr * tsl_multiplier)
            risk_per_unit = entry_price - stop_loss_price
        else: # short
            stop_loss_price = entry_price + (atr * tsl_multiplier)
            risk_per_unit = stop_loss_price - entry_price

        if risk_per_unit <= 0:
            print(f"Warning: Invalid stop-loss, risk is zero or negative. Skipping trade.")
            return False
            
        amount_to_trade = risk_amount_usd / risk_per_unit

        self.in_position = True
        self.position_details = {
            'type': trade_type,
            'entry_price': entry_price,
            'size': amount_to_trade,
            'partial_exit_taken': False,
            'hard_stop_loss': stop_loss_price,
            'trailing_stop_loss': stop_loss_price,
            'take_profit_price': entry_price + (atr * tp_multiplier) if trade_type == 'long' else entry_price - (atr * tp_multiplier)
        }
        return True

    def exit_position(self, exit_reason, exit_price):
        """
        Calculates P&L and updates the state to exit a trade (partially or fully).
        
        Returns:
            A tuple of (pnl, size_exited).
        """
        if not self.in_position:
            return 0, 0

        pos = self.position_details
        size_to_exit = pos['size']
        
        if exit_reason == "Partial Take Profit":
            size_to_exit *= self.params.get('PARTIAL_EXIT_PERCENTAGE', 0.5)
            self.position_details['size'] -= size_to_exit
            self.position_details['partial_exit_taken'] = True
        else:
            # Full exit
            self.in_position = False
            self.position_details = {}

        pnl = (exit_price - pos['entry_price']) * size_to_exit if pos['type'] == 'long' else (pos['entry_price'] - exit_price) * size_to_exit
        
        return pnl, size_to_exit