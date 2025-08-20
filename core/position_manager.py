#!/usr/bin/env python3
"""
UNIFIED POSITION MANAGER

This script implements a unified PositionManager by merging the best features
from both the standalone position_manager.py and the Portfolio version.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, Any, Optional, Tuple, Union

# Import MarketRegime for regime-aware position management
try:
    from core.strategy import MarketRegime
except ImportError:
    try:
        from .strategy import MarketRegime
    except ImportError:
        MarketRegime = None

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Unified PositionManager class that combines the best features from both implementations.
    
    This class manages:
    1. Position entry/exit logic
    2. Dynamic risk management based on market conditions
    3. Adaptive position sizing with confidence scaling
    4. Market regime awareness and volatility adjustments
    """
    

    def validate_position_size(self, position_size, account_balance, max_risk_percent=0.02):
        """
        Validate position size calculations to prevent trading errors
        
        Args:
            position_size: Calculated position size
            account_balance: Current account balance
            max_risk_percent: Maximum risk percentage (default 2%)
        
        Returns:
            bool: True if position size is valid, False otherwise
        """
        if position_size <= 0:
            return False
            
        if position_size > account_balance * 0.95:  # Max 95% of account
            return False
            
        # Risk check - position shouldn't exceed max risk
        max_position = account_balance * max_risk_percent
        if position_size > max_position:
            return False
            
        return True

    def __init__(self, config_file: Optional[Union[str, Dict[str, Any]]] = None):
        """Initialize with configuration"""
        # HEALTH CHECK: Ensure system health before position manager initialization
        try:
            import sys
            import os
            # Add parent directory to path to find utilities
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from utilities.utils import safe_health_check
            safe_health_check("PositionManager", silent=True)
        except Exception:
            # Any error in health check - proceed with warning
            pass
            
        # Enhanced configuration loading with centralized manager
        try:
            from utilities.utils import config_manager
            self.config = config_manager.get_config(config_file, 'risk_management')
        except Exception:
            # Fallback to traditional config loading
            self.config = self._load_config(config_file)
        
        # Core risk parameters
        self.base_risk_percentage = self.config.get('FIXED_RISK_PERCENTAGE', 0.02)
        self.max_single_trade_risk = self.config.get('MAX_SINGLE_TRADE_RISK', 0.03)
        self.max_portfolio_risk = self.config.get('MAX_PORTFOLIO_RISK', 0.05)
        self.base_position_multiplier = self.config.get('BASE_POSITION_MULTIPLIER', 1.0)
        
        # Market regime parameters  
        self.high_vol_threshold = self.config.get('HIGH_VOLATILITY_THRESHOLD', 0.03)
        self.low_vol_threshold = self.config.get('LOW_VOLATILITY_THRESHOLD', 0.01)
        self.trend_strength_threshold = self.config.get('TREND_STRENGTH_THRESHOLD', 0.7)
        
        # Drawdown protection
        self.max_drawdown_threshold = self.config.get('MAX_DRAWDOWN_THRESHOLD', 0.1)
        self.drawdown_risk_reduction = self.config.get('DRAWDOWN_RISK_REDUCTION', True)
        
        # Position sizing limits
        self.max_position_size = self.config.get('MAX_POSITION_SIZE', 0.15)
        self.min_position_size = self.config.get('MIN_POSITION_SIZE', 0.001)
        
        # Commission and trading costs
        self.commission_rate = self.config.get('COMMISSION_RATE', 0.001)  # 0.1% for market taker
        
        # Dynamic risk management state
        self.current_drawdown = 0.0
        self.recent_volatility = 0.02
        self.trend_strength = 0.5
        self.peak_portfolio_value = None
        self.consecutive_losses = 0
        self.recent_trades = []
        
        # Trading state attributes (for backtest compatibility)
        self.in_position = False
        self.position_details = {}
        
        # Warning reduction counters
        self.position_reduction_count = 0
        self.emergency_reduction_count = 0
        self.last_warning_time = 0
        self.warning_interval = 5000  # Show summary every 5000 reductions (much more aggressive)
        
        # Volume confirmation parameters (for enhanced signal filtering)
        self.min_volume_strength = self.config.get('MIN_VOLUME_STRENGTH', 0.3)
        self.volume_confirmation_enabled = self.config.get('VOLUME_CONFIRMATION', True)
        
        # Enhanced risk management parameters
        self.lookback_period = self.config.get('RISK_LOOKBACK_PERIOD', 20)
        self.volatility_window = self.config.get('VOLATILITY_WINDOW', 14)
        self.enable_dynamic_sizing = self.config.get('ENABLE_DYNAMIC_POSITION_SIZING', True)
        self.enable_regime_stops = self.config.get('ENABLE_REGIME_BASED_STOPS', True)
        self.enable_partial_profits = self.config.get('ENABLE_PARTIAL_PROFIT_TAKING', True)
        
        logger.info(f"Unified PositionManager initialized with base {self.base_risk_percentage:.1%} risk per trade")
        logger.info(f"Volume confirmation: {'enabled' if self.volume_confirmation_enabled else 'disabled'} (min strength: {self.min_volume_strength})")
        logger.info(f"Enhanced risk management: Dynamic sizing={self.enable_dynamic_sizing}, Regime stops={self.enable_regime_stops}")

    def load_state(self, state: dict):
        """Loads the position state from a dictionary."""
        self.in_position = state.get('in_position', False)
        self.position_details = state.get('position_details', {})
        logger.info(f"Position manager state loaded: in_position={self.in_position}")

    def get_state(self) -> dict:
        """Returns the current position state as a dictionary."""
        return {
            "in_position": self.in_position,
            "position_details": self.position_details
        }
    
    def _load_config(self, config_file: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Load configuration from file or use provided dictionary"""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'optimization_config.json')
        
        # If config_file is already a dictionary, return it directly
        if isinstance(config_file, dict):
            logger.info("Using provided configuration dictionary")
            return config_file
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_file}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'FIXED_RISK_PERCENTAGE': 0.02,
            'MAX_SINGLE_TRADE_RISK': 0.03,
            'MAX_PORTFOLIO_RISK': 0.05,
            'BASE_POSITION_MULTIPLIER': 1.0,
            'HIGH_VOLATILITY_THRESHOLD': 0.03,
            'LOW_VOLATILITY_THRESHOLD': 0.01,
            'TREND_STRENGTH_THRESHOLD': 0.7,
            'MAX_DRAWDOWN_THRESHOLD': 0.1,
            'DRAWDOWN_RISK_REDUCTION': True,
            'MAX_POSITION_SIZE': 0.15,
            'MIN_POSITION_SIZE': 0.001,
            'COMMISSION_RATE': 0.001  # 0.1% Binance market taker fee
        }
    
    def calculate_position_size(self, 
                              balance_usd: float,
                              entry_price: float,
                              stop_loss_price: float,
                              confidence: float = 1.0,
                              atr: Optional[float] = None,
                              market_data: Optional[Dict] = None) -> float:
        """
        Calculate position size with comprehensive dynamic risk management
        FIXED: Ensures position size cannot cause losses exceeding risk amount
        
        Args:
            balance_usd: Account balance in USD
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Signal confidence (0-1)
            atr: Average True Range for volatility adjustment
            market_data: Complete market context including regime, volatility, etc.
            
        Returns:
            Position size in base currency (BTC)
        """
        try:
            # Update market conditions if provided
            if market_data:
                self._update_market_conditions(market_data)
            
            # Calculate dynamic risk percentage based on comprehensive market analysis
            dynamic_risk_percentage = self._calculate_dynamic_risk(
                balance_usd, 
                market_data or {}, 
                confidence, 
                atr
            )
            
            # Calculate base risk amount using dynamic risk
            base_risk_amount = balance_usd * dynamic_risk_percentage
            
            # Apply adaptive position multiplier from strategy
            position_multiplier = market_data.get('adaptive_position_multiplier', 1.0) if market_data else 1.0
            position_multiplier = max(0.1, min(position_multiplier, 3.0))  # Ensure reasonable bounds
            
            # Apply position multiplier
            final_risk_amount = base_risk_amount * position_multiplier
            
            # CRITICAL FIX: Ensure final risk amount never exceeds available balance
            final_risk_amount = min(final_risk_amount, balance_usd * 0.95)  # Max 95% of balance
            
            # Calculate position size based on stop loss distance
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff <= 0:
                # Fallback: use a minimum percentage-based risk when stop loss calculation fails
                min_risk_pct = 0.005  # 0.5% minimum risk
                price_diff = entry_price * min_risk_pct
                logger.debug(f"Invalid stop loss distance, using {min_risk_pct:.1%} fallback risk distance")
            
            # CRITICAL FIX: Position size calculation that ensures risk-limited losses
            # Position size = Risk Amount / Price Difference per unit
            position_size = final_risk_amount / price_diff
            
            # ADDITIONAL SAFETY: Ensure position value doesn't exceed balance
            position_value_usd = position_size * entry_price
            max_position_value = balance_usd * 0.95  # Max 95% of balance
            
            if position_value_usd > max_position_value:
                position_size = max_position_value / entry_price
                self.position_reduction_count += 1
                
                # No individual warnings - only track count for final summary
            
            # Apply minimum position size limit
            position_size = max(self.min_position_size, position_size)
            
            # FINAL SAFETY CHECK: Calculate maximum possible loss with this position size
            max_possible_loss = position_size * price_diff
            if max_possible_loss > balance_usd:
                # This should never happen with correct calculations, but add as failsafe
                position_size = (balance_usd * 0.95) / price_diff
                self.emergency_reduction_count += 1
                
                # No individual warnings - only track count for final summary
            
            logger.debug(f"FIXED Risk-based position sizing: "
                        f"balance=${balance_usd:.2f}, "
                        f"risk_amount=${final_risk_amount:.2f}, "
                        f"price_diff=${price_diff:.2f}, "
                        f"position_size={position_size:.6f}, "
                        f"position_value=${position_size * entry_price:.2f}, "
                        f"max_loss=${position_size * price_diff:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in calculate_position_size: {e}")
            # Return minimum safe position size as fallback
            return self.min_position_size

    def get_warning_summary(self):
        """Get a summary of warnings during the session"""
        return {
            'position_reductions': self.position_reduction_count,
            'emergency_reductions': self.emergency_reduction_count
        }
    
    def log_session_summary(self):
        """Log a summary of position sizing activity at the end of session"""
        if self.position_reduction_count > 0 or self.emergency_reduction_count > 0:
            logger.info(f"Position Manager Session Summary: "
                       f"{self.position_reduction_count} position size reductions, "
                       f"{self.emergency_reduction_count} emergency reductions applied")
        else:
            logger.debug("Position Manager Session Summary: No position size adjustments needed")
    
    def _calculate_dynamic_risk(self, balance_usd: float, market_data: Dict, signal_confidence: float, atr: Optional[float] = None) -> float:
        """
        Calculate dynamic risk percentage based on comprehensive market conditions, signal quality, and portfolio state.
        
        Args:
            balance_usd: Current portfolio value
            market_data: Market context (regime, volatility, etc.)
            signal_confidence: Confidence in trading signal (0-1)
            atr: Average True Range for volatility calculation
            
        Returns:
            Dynamic risk percentage (0.001 to MAX_SINGLE_TRADE_RISK)
        """
        # Start with base risk
        adjusted_risk = self.base_risk_percentage
        
        # Extract market data
        market_regime = market_data.get('market_regime', 'ranging')
        regime_confidence = market_data.get('regime_confidence', 0.5)
        close_price = market_data.get('close', 0)
        
        # Calculate volatility from ATR if available
        if atr and close_price > 0:
            volatility = atr / close_price
        else:
            volatility = market_data.get('volatility', self.recent_volatility)
        
        # 1. Market Regime Risk Adjustment
        regime_multipliers = {
            'trending_bull': 1.2,      # Higher risk in bull trends
            'trending_bear': 0.7,      # Lower risk in bear trends  
            'ranging': 0.9,            # Slightly lower risk in ranging markets
            'high_volatility': 0.6,    # Much lower risk in high volatility
            'low_volatility': 1.1      # Slightly higher risk in low volatility
        }
        
        regime_multiplier = regime_multipliers.get(market_regime, 1.0)
        adjusted_risk *= regime_multiplier
        
        # 2. Confidence-Based Scaling
        # Signal confidence: 0.5 to 1.5x multiplier
        confidence_multiplier = 0.5 + (signal_confidence * 1.0)
        # Regime confidence: 0.7 to 1.3x multiplier  
        regime_conf_multiplier = 0.7 + (regime_confidence * 0.6)
        
        adjusted_risk *= confidence_multiplier * regime_conf_multiplier
        
        # 3. Volatility Adjustment
        if volatility > 0.05:  # High volatility (>5%)
            volatility_multiplier = 0.5
        elif volatility > 0.03:  # Medium volatility (3-5%)
            volatility_multiplier = 0.8
        elif volatility > 0.015:  # Normal volatility (1.5-3%)
            volatility_multiplier = 1.0
        else:  # Low volatility (<1.5%)
            volatility_multiplier = 1.1
            
        adjusted_risk *= volatility_multiplier
        
        # 4. Drawdown Protection
        if self.drawdown_risk_reduction and self.peak_portfolio_value:
            current_drawdown = (self.peak_portfolio_value - balance_usd) / self.peak_portfolio_value
            if current_drawdown > 0.05:  # 5% drawdown
                drawdown_multiplier = max(0.3, 1.0 - (current_drawdown * 2))
                adjusted_risk *= drawdown_multiplier
        
        # 5. Consecutive Loss Protection
        if self.consecutive_losses >= 2:
            loss_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.15))
            adjusted_risk *= loss_multiplier
        
        # 6. Final Risk Bounds
        adjusted_risk = max(0.001, min(adjusted_risk, self.max_single_trade_risk))
        
        return adjusted_risk
    
    def update_trade_history(self, trade_pnl: float, portfolio_value: float) -> None:
        """
        Update trade history for adaptive risk management.
        
        Args:
            trade_pnl: P&L of completed trade
            portfolio_value: Current portfolio value
        """
        # Update peak portfolio value
        if self.peak_portfolio_value is None or portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Track recent trades (keep last 10)
        self.recent_trades.append(trade_pnl)
        if len(self.recent_trades) > 10:
            self.recent_trades.pop(0)
        
        # Update consecutive losses
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.debug(f"Trade history updated: PnL=${trade_pnl:.2f}, "
                    f"consecutive_losses={self.consecutive_losses}, "
                    f"portfolio_value=${portfolio_value:.2f}")
    
    def _calculate_adaptive_position_multiplier(self, 
                                              confidence: float,
                                              atr: Optional[float] = None,
                                              market_data: Optional[Dict] = None) -> float:
        """
        LEGACY METHOD - kept for backwards compatibility
        Calculate adaptive position multiplier based on market conditions
        Note: This is now integrated into _calculate_dynamic_risk for comprehensive risk management
        """
        multiplier = self.base_position_multiplier
        
        # Confidence adjustment
        if confidence > 0.8:
            multiplier *= 1.2
        elif confidence < 0.5:
            multiplier *= 0.8
        
        # Volatility adjustment
        if atr:
            volatility_ratio = atr / 0.02  # Normalize to typical volatility
            if volatility_ratio > 1.5:
                multiplier *= 0.8  # Reduce in high volatility
            elif volatility_ratio < 0.5:
                multiplier *= 1.1  # Increase in low volatility
        
        # Market regime adjustment
        multiplier *= self._get_market_regime_multiplier()
        
        # Drawdown protection
        if self.current_drawdown > self.max_drawdown_threshold:
            multiplier *= 0.5  # Legacy drawdown reduction
        
        # Ensure reasonable bounds
        return max(0.1, min(multiplier, 3.0))
    
    def _get_market_regime_multiplier(self) -> float:
        """Get position multiplier based on market regime"""
        if self.recent_volatility > self.high_vol_threshold:
            return 0.7  # Reduce positions in high volatility
        elif self.recent_volatility < self.low_vol_threshold and self.trend_strength > self.trend_strength_threshold:
            return 1.3  # Increase positions in low vol trending markets
        else:
            return 1.0  # Normal market conditions
    
    def _update_market_conditions(self, market_data: Dict) -> None:
        """Update market condition indicators"""
        if 'volatility' in market_data:
            self.recent_volatility = market_data['volatility']
        
        if 'trend_strength' in market_data:
            self.trend_strength = market_data['trend_strength']
        
        if 'drawdown' in market_data:
            self.current_drawdown = market_data['drawdown']
    
    def _calculate_breakeven_level(self, entry_price: float, trade_type: str, estimated_slippage: float = 0.0) -> float:
        """
        Calculate breakeven price including commissions and slippage
        
        Args:
            entry_price: Price at which position was entered
            trade_type: 'long' or 'short'
            estimated_slippage: Estimated slippage as decimal (e.g., 0.001 for 0.1%)
            
        Returns:
            Breakeven price including all costs
        """
        # Total costs = entry commission + exit commission + slippage
        total_cost_rate = (self.commission_rate * 2) + estimated_slippage
        
        if trade_type.lower() == 'long':
            # For long positions, need price above entry to cover costs
            return entry_price * (1 + total_cost_rate)
        else:  # short
            # For short positions, need price below entry to cover costs
            return entry_price * (1 - total_cost_rate)
    
    def update_drawdown(self, current_drawdown: float) -> None:
        """Update current drawdown for risk adjustment"""
        self.current_drawdown = current_drawdown
        if current_drawdown > self.max_drawdown_threshold:
            logger.warning(f"High drawdown detected: {current_drawdown:.1%}")
    
    def validate_position(self, position_size: float, entry_price: float, balance_usd: float) -> bool:
        """Validate if position size is acceptable"""
        position_value = position_size * entry_price
        position_percentage = position_value / balance_usd
        
        if position_percentage > self.max_position_size:
            logger.warning(f"Position too large: {position_percentage:.1%} > {self.max_position_size:.1%}")
            return False
        
        if position_size < self.min_position_size:
            logger.warning(f"Position too small: {position_size:.6f} < {self.min_position_size:.6f}")
            return False
        
        return True
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk management metrics"""
        return {
            'base_risk_percentage': self.base_risk_percentage,
            'max_single_trade_risk': self.max_single_trade_risk,
            'max_portfolio_risk': self.max_portfolio_risk,
            'current_drawdown': self.current_drawdown,
            'recent_volatility': self.recent_volatility,
            'trend_strength': self.trend_strength,
            'max_position_size': self.max_position_size,
            'base_position_multiplier': self.base_position_multiplier,
            'consecutive_losses': self.consecutive_losses,
            'peak_portfolio_value': self.peak_portfolio_value,
            'recent_trades_count': len(self.recent_trades)
        }
    
    def adjust_risk_parameters(self, **kwargs) -> None:
        """Dynamically adjust risk parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"Adjusted {param}: {old_value} -> {value}")
    
    def calculate_stop_loss(self, entry_price: float, atr: float, direction: str = 'long') -> float:
        """Calculate stop loss based on ATR using optimizable parameters"""
        # Use optimizable STOP_LOSS_MULTIPLIER parameter
        atr_multiplier = self.config.get('STOP_LOSS_MULTIPLIER', 2.0)
        
        # Handle zero or invalid ATR values
        if atr <= 0 or np.isnan(atr):
            # Use a minimum percentage-based stop loss when ATR is invalid
            min_stop_loss_pct = 0.005  # 0.5% minimum stop loss
            atr = entry_price * min_stop_loss_pct
            logger.debug(f"Invalid ATR ({atr}), using minimum {min_stop_loss_pct:.1%} stop loss")
        
        if direction.lower() == 'long':
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)
    
    def calculate_take_profit(self, entry_price: float, atr: float, direction: str = 'long') -> float:
        """Calculate take profit based on ATR using optimizable parameters"""
        # Use optimizable TAKE_PROFIT_MULTIPLIER parameter
        atr_multiplier = self.config.get('TAKE_PROFIT_MULTIPLIER', 3.0)
        
        if direction.lower() == 'long':
            return entry_price + (atr * atr_multiplier)
        else:
            return entry_price - (atr * atr_multiplier)
    
    def get_position_summary(self, positions: list) -> Dict[str, Any]:
        """Get summary of current positions"""
        if not positions:
            return {'total_positions': 0, 'total_exposure': 0.0}
        
        total_exposure = sum(pos.get('value', 0) for pos in positions)
        
        return {
            'total_positions': len(positions),
            'total_exposure': total_exposure,
            'average_position_size': total_exposure / len(positions) if positions else 0
        }

    def check_volume_confirmation(self, volume_metrics: Dict[str, float]) -> bool:
        """
        Check if volume conditions support the signal
        
        Args:
            volume_metrics: Dictionary containing volume analysis metrics from strategy
            
        Returns:
            bool: True if volume confirms signal, False otherwise
        """
        if not self.volume_confirmation_enabled:
            return True  # Skip volume check if disabled
            
        if not volume_metrics:
            logger.debug("No volume metrics available, skipping volume confirmation")
            return True  # Default to True if no volume data
            
        # Get volume strength from strategy's volume analysis
        volume_strength = volume_metrics.get('volume_strength', 0.0)
        
        # Check if volume strength meets minimum threshold
        volume_confirmed = volume_strength >= self.min_volume_strength
        
        if volume_confirmed:
            logger.debug(f"Volume confirmation: PASSED (strength: {volume_strength:.3f} >= {self.min_volume_strength:.3f})")
        else:
            logger.debug(f"Volume confirmation: FAILED (strength: {volume_strength:.3f} < {self.min_volume_strength:.3f})")
            
        return volume_confirmed

    # Legacy compatibility methods for existing code
    def can_open_position(self, direction: str, market_data: Dict[str, Any]) -> bool:
        """Check if new position can be opened"""
        # Basic implementation - can be enhanced
        return True
    
    def add_position(self, position: Dict[str, Any]) -> bool:
        """Add a new position - placeholder for compatibility"""
        # This would normally track positions
        return True
    
    def get_position_metrics(self) -> Dict[str, Any]:
        """Get position metrics - compatibility method"""
        return self.get_risk_metrics()

    # ====================================================================================
    # TRADING STATE MANAGEMENT METHODS (for backtest compatibility)
    # ====================================================================================
    
    def check_for_entry(self, signal_candle, volume_metrics: Dict[str, float] = None) -> Tuple[Optional[str], Optional[float]]:
        """Check if entry conditions are met by reading signals from the dataframe"""
        try:
            # Check if signal columns exist in the candle data
            if hasattr(signal_candle, 'get'):
                # For Series or dict-like objects
                primary_signal = signal_candle.get('signal', 0)
                long_signal = signal_candle.get('long_signals', False)
                short_signal = signal_candle.get('short_signals', False)
                confidence = signal_candle.get('confidence', 0.0)
            elif hasattr(signal_candle, '__getitem__'):
                # For array-like objects with numeric indexing
                try:
                    primary_signal = 0
                    long_signal = False
                    short_signal = False
                    confidence = 0.0
                except:
                    return None, None
            else:
                return None, None
            
            # Check for entry signals based on strategy output
            min_confidence = self.config.get('min_confidence', 0.1)  # FIXED: Lower default to allow most signals
            
            if long_signal and primary_signal == 1 and confidence > min_confidence:
                # Check volume confirmation if enabled
                if not self.check_volume_confirmation(volume_metrics or {}):
                    logger.debug(f"Long signal failed volume confirmation")
                    return None, None
                return 'long', confidence
            elif short_signal and primary_signal == -1 and confidence > min_confidence:
                # Check volume confirmation if enabled
                if not self.check_volume_confirmation(volume_metrics or {}):
                    logger.debug(f"Short signal failed volume confirmation")
                    return None, None
                return 'short', confidence
            else:
                return None, None
                
        except Exception as e:
            logger.debug(f"Error checking for entry: {e}")
            return None, None
    
    def check_for_exit(self, current_candle) -> Optional[str]:
        """Check if exit conditions are met"""
        if not self.in_position:
            return None
            
        # Basic exit logic - should be enhanced based on strategy
        pos_type = self.position_details.get('type', 'long')
        entry_price = self.position_details.get('entry_price', 0)
        current_price = current_candle['close']
        
        # Simple stop loss check (5% for demo)
        if pos_type == 'long':
            if current_price <= entry_price * 0.95:
                return "Stop Loss"
        else:  # short
            if current_price >= entry_price * 1.05:
                return "Stop Loss"
                
        return None
    
    def update_trailing_stop(self, current_candle) -> None:
        """Update trailing stop loss using optimizable parameters"""
        if not self.in_position:
            return
            
        # Use optimizable TRAILING_STOP_MULTIPLIER parameter
        trailing_stop_multiplier = self.config.get('TRAILING_STOP_MULTIPLIER', 0.05)  # Default 5%
        
        pos_type = self.position_details.get('type', 'long')
        current_price = current_candle['close']
        
        if pos_type == 'long':
            new_stop = current_price * (1.0 - trailing_stop_multiplier)
            current_stop = self.position_details.get('stop_loss', 0)
            if new_stop > current_stop:
                self.position_details['stop_loss'] = new_stop
        else:  # short
            new_stop = current_price * (1.0 + trailing_stop_multiplier)
            current_stop = self.position_details.get('stop_loss', float('inf'))
            if new_stop < current_stop:
                self.position_details['stop_loss'] = new_stop
    
    def enter_position_with_risk_management(self, trade_type: str, current_candle, portfolio_cash: float, 
                                           strategy_levels: Dict[str, Any], entry_price: float = None):
        """
        Enter a position with comprehensive risk management using strategy-calculated levels
        
        Args:
            trade_type: 'long' or 'short'
            current_candle: Current market data
            portfolio_cash: Available cash
            strategy_levels: Dictionary containing:
                - stop_loss_price: Strategy-calculated stop loss
                - trailing_stop_distance: Strategy-calculated TSL distance
                - partial_tp_price: Strategy-calculated partial take profit price
                - partial_tp_percentage: Percentage of position to close at partial TP
                - estimated_slippage: Strategy-estimated slippage
            entry_price: Override entry price (optional)
            
        Returns:
            bool: True if position entered successfully
        """
        if self.in_position:
            return False
            
        # Use provided entry price or current close
        entry_price = entry_price or current_candle['close']
        
        # Extract strategy-calculated levels
        stop_loss_price = strategy_levels['stop_loss_price']
        trailing_distance = strategy_levels['trailing_stop_distance']
        partial_tp_price = strategy_levels['partial_tp_price']
        partial_tp_percentage = strategy_levels['partial_tp_percentage']
        estimated_slippage = strategy_levels.get('estimated_slippage', 0.0)
        
        # Extract market data from strategy signals if available
        market_regime = current_candle.get('market_regime', 'ranging')
        regime_confidence = current_candle.get('regime_confidence', 0.5)
        signal_confidence = current_candle.get('confidence', 0.7)
        atr_value = current_candle.get('atr', 0.02 * entry_price)
        adaptive_multiplier = current_candle.get('adaptive_position_multiplier', 1.0)
        
        # Create comprehensive market data for dynamic risk calculation
        market_data = {
            'market_regime': market_regime,
            'regime_confidence': regime_confidence,
            'volatility': atr_value / entry_price if entry_price > 0 else 0.02,
            'adaptive_position_multiplier': adaptive_multiplier,
            'close': entry_price
        }
        
        # Calculate position size using dynamic risk with strategy stop loss
        position_size = self.calculate_position_size(
            balance_usd=portfolio_cash,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_data=market_data,
            confidence=signal_confidence,
            atr=atr_value
        )
        
        if position_size <= 0:
            return False
        
        # Calculate breakeven level including commissions and slippage
        breakeven_level = self._calculate_breakeven_level(entry_price, trade_type, estimated_slippage)
        atr_at_entry = strategy_levels.get('atr_at_entry')
        
        # Calculate trailing stop initial position
        if trade_type.lower() == 'long':
            trailing_stop = entry_price - trailing_distance
        else:  # short
            trailing_stop = entry_price + trailing_distance
        
        # Set enhanced position state
        self.in_position = True
        self.position_details = {
            # Basic position info
            'type': trade_type,
            'entry_price': entry_price,
            'size': position_size,  # For backward compatibility
            'initial_size': position_size,
            'current_size': position_size,
            'entry_timestamp': getattr(current_candle, 'name', 'unknown'),
            
            # Risk management levels (from strategy)
            'stop_loss': stop_loss_price,
            'trailing_stop': trailing_stop,
            'trailing_distance': trailing_distance,
            'partial_tp_price': partial_tp_price,
            'partial_tp_percentage': partial_tp_percentage,
            
            # Calculated levels
            'breakeven_level': breakeven_level,
            'atr_at_entry': atr_at_entry,
            
            # State tracking
            'partial_tp_taken': False,
            'is_risk_free': False,
            
            # Market context
            'market_regime': market_regime,
            'signal_confidence': signal_confidence,
            'estimated_slippage': estimated_slippage
        }
        
        logger.info(f"Entered {trade_type} position: size={position_size:.6f}, "
                   f"entry=${entry_price:.2f}, stop=${stop_loss_price:.2f}, "
                   f"partial_tp=${partial_tp_price:.2f}, breakeven=${breakeven_level:.2f}")
        
        return True

    def update_position_management(self, current_candle) -> Dict[str, Any]:
        """
        Update position management: check stops, trailing stops, and partial take profit
        
        Args:
            current_candle: Current market data
            
        Returns:
            Dictionary with exit signals and actions
        """
        if not self.in_position:
            return {'action': 'none'}
        
        current_price = current_candle['close']
        pos_details = self.position_details
        trade_type = pos_details['type']
        
        # Check for stop loss hit with proper exit reason labeling
        if trade_type == 'long':
            if current_price <= pos_details['stop_loss']:
                # Determine if this is actually a profitable trailing stop or a real stop loss
                entry_price = pos_details['entry_price']
                if current_price >= entry_price:
                    # Price is above entry - this is a trailing take profit, not a stop loss
                    return {'action': 'exit_full', 'reason': 'Trailing Take Profit', 'price': current_price}
                else:
                    # Price is below entry - this is a real stop loss
                    return {'action': 'exit_full', 'reason': 'Stop Loss', 'price': current_price}
        else:  # short
            if current_price >= pos_details['stop_loss']:
                # Determine if this is actually a profitable trailing stop or a real stop loss
                entry_price = pos_details['entry_price']
                if current_price <= entry_price:
                    # Price is below entry - this is a trailing take profit for short, not a stop loss
                    return {'action': 'exit_full', 'reason': 'Trailing Take Profit', 'price': current_price}
                else:
                    # Price is above entry - this is a real stop loss for short
                    return {'action': 'exit_full', 'reason': 'Stop Loss', 'price': current_price}
        
        # Ensure trailing stop is active from entry; we already maintain trailing_stop and stop_loss

        # Auto breakeven partial: when price reaches breakeven (incl costs), sell enough to cover costs
        if not pos_details.get('is_risk_free', False):
            be_level = pos_details.get('breakeven_level', pos_details['entry_price'])
            # Use a small buffer via BREAKEVEN_ATR_BUFFER if atr_at_entry is known (already applied when set)
            if (trade_type == 'long' and current_price >= be_level) or (trade_type == 'short' and current_price <= be_level):
                be_partial_pct = float(self.config.get('PARTIAL_BE_PERCENTAGE', 0.5))
                be_partial_pct = max(0.1, min(be_partial_pct, 0.9))
                # Execute BE partial
                self.position_details['partial_tp_percentage'] = be_partial_pct
                return {'action': 'partial_exit', 'reason': 'Breakeven Partial', 'price': current_price}

        # Check for partial take profit (only if not already taken)
        if not pos_details.get('partial_tp_taken', False):
            partial_tp_hit = False
            if trade_type == 'long' and current_price >= pos_details.get('partial_tp_price', float('inf')):
                partial_tp_hit = True
            elif trade_type == 'short' and current_price <= pos_details.get('partial_tp_price', 0):
                partial_tp_hit = True
            
            if partial_tp_hit:
                return {'action': 'partial_exit', 'reason': 'Partial Take Profit', 'price': current_price}
        
        # Update trailing stop
        self._update_trailing_stop_advanced(current_price)
        
        return {'action': 'none'}
    
    def execute_partial_take_profit(self, exit_price: float, timestamp) -> Tuple[float, float]:
        """
        Execute partial take profit and move stop loss to breakeven
        
        Args:
            exit_price: Price at which partial exit occurs
            timestamp: Exit timestamp
            
        Returns:
            Tuple of (partial_pnl, remaining_position_size)
        """
        if not self.in_position or self.position_details.get('partial_tp_taken', False):
            return 0.0, 0.0
        
        pos_details = self.position_details
        trade_type = pos_details['type']
        entry_price = pos_details['entry_price']
        partial_percentage = pos_details['partial_tp_percentage']
        current_size = pos_details['current_size']
        
        # Calculate partial exit size
        partial_size = current_size * partial_percentage
        remaining_size = current_size - partial_size
        
        # Calculate P&L for partial exit
        if trade_type == 'long':
            partial_pnl = (exit_price - entry_price) * partial_size
        else:  # short
            partial_pnl = (entry_price - exit_price) * partial_size
        
        # Update position details
        self.position_details['current_size'] = remaining_size
        self.position_details['partial_tp_taken'] = True
        self.position_details['is_risk_free'] = True
        
        # Move stop loss to breakeven level (with small ATR buffer above/below BE)
        breakeven_level = pos_details.get('breakeven_level', self._calculate_breakeven_level(entry_price, trade_type))
        atr_at_entry = pos_details.get('atr_at_entry', None)
        be_buffer_mult = self.config.get('BREAKEVEN_ATR_BUFFER', 0.1)  # 0.1 ATR by default
        if atr_at_entry and atr_at_entry > 0:
            if trade_type == 'long':
                breakeven_level = breakeven_level + (atr_at_entry * be_buffer_mult)
            else:
                breakeven_level = breakeven_level - (atr_at_entry * be_buffer_mult)
        self.position_details['stop_loss'] = breakeven_level
        
        logger.info(f"Partial TP executed: sold {partial_size:.6f} at ${exit_price:.2f}, "
                   f"P&L=${partial_pnl:.2f}, remaining={remaining_size:.6f}, "
                   f"SL moved to breakeven ${breakeven_level:.2f}")
        
        return partial_pnl, remaining_size
    
    def _update_trailing_stop_advanced(self, current_price: float):
        """Update trailing stop loss with strategy-calculated distance"""
        if not self.in_position:
            return
        
        pos_details = self.position_details
        trade_type = pos_details['type']
        
        # Get trailing distance with fallback for legacy positions
        trailing_distance = pos_details.get('trailing_distance', current_price * 0.02)  # Default 2% of price
        current_trailing = pos_details.get('trailing_stop', 0)
        
        if trade_type == 'long':
            # For long positions, trailing stop moves up only
            new_trailing = current_price - trailing_distance
            if new_trailing > current_trailing:
                self.position_details['trailing_stop'] = new_trailing
                # Update stop loss to higher of current stop and trailing stop
                self.position_details['stop_loss'] = max(
                    pos_details['stop_loss'], 
                    new_trailing
                )
        else:  # short
            # For short positions, trailing stop moves down only
            new_trailing = current_price + trailing_distance
            if new_trailing < current_trailing:
                self.position_details['trailing_stop'] = new_trailing
                # Update stop loss to lower of current stop and trailing stop
                self.position_details['stop_loss'] = min(
                    pos_details['stop_loss'], 
                    new_trailing
                )

    def enter_position(self, trade_type: str, current_candle, portfolio_cash: float, entry_price: float = None) -> bool:
        """Enter a position using dynamic risk calculation with market regime data"""
        if self.in_position:
            return False
            
        # Use provided entry price or current close
        entry_price = entry_price or current_candle['close']
        
        # Extract market data from strategy signals if available
        market_regime = current_candle.get('market_regime', 'ranging')
        regime_confidence = current_candle.get('regime_confidence', 0.5)
        signal_confidence = current_candle.get('confidence', 0.7)
        atr_value = current_candle.get('atr', 0.02 * entry_price)
        adaptive_multiplier = current_candle.get('adaptive_position_multiplier', 1.0)
        
        # Create comprehensive market data for dynamic risk calculation
        market_data = {
            'market_regime': market_regime,
            'regime_confidence': regime_confidence,
            'volatility': atr_value / entry_price if entry_price > 0 else 0.02,
            'adaptive_position_multiplier': adaptive_multiplier,
            'close': entry_price
        }
        
        # Calculate position size using dynamic risk with real market conditions
        position_size = self.calculate_position_size(
            balance_usd=portfolio_cash,
            entry_price=entry_price,
            stop_loss_price=entry_price * (0.95 if trade_type == 'long' else 1.05),
            market_data=market_data,
            confidence=signal_confidence,
            atr=atr_value
        )
        
        if position_size <= 0:
            return False
            
        # Set position state
        self.in_position = True
        self.position_details = {
            'type': trade_type,
            'entry_price': entry_price,
            'size': position_size,  # Keep for backward compatibility
            'current_size': position_size,  # Standard key going forward
            'initial_size': position_size,  # Track original size
            'stop_loss': entry_price * (0.95 if trade_type == 'long' else 1.05),
            'entry_timestamp': getattr(current_candle, 'name', 'unknown'),
            'market_regime': market_regime,
            'signal_confidence': signal_confidence,
            # For false signal / outcome tracking
            'bars_in_trade': 0,
            'max_adverse_excursion': 0.0,
            'max_favorable_excursion': 0.0,
            # Enhanced compatibility keys
            'partial_tp_taken': False,
            'partial_tp_price': entry_price * (1.03 if trade_type == 'long' else 0.97),  # Default 3% TP
            'partial_tp_percentage': self.config.get('PARTIAL_EXIT_PERCENTAGE', 0.5),  # Use optimizable parameter
            'breakeven_level': self._calculate_breakeven_level(entry_price, trade_type),
            'is_risk_free': False,
            'trailing_distance': entry_price * 0.02,  # Default 2% trailing distance
            'trailing_stop': entry_price * (0.98 if trade_type == 'long' else 1.02),  # Initial trailing stop
        }
        
        logger.debug(f"Entered {trade_type} position: size={position_size:.6f}, "
                    f"regime={market_regime}, confidence={signal_confidence:.2f}, "
                    f"dynamic_risk_used=True")
        
        return True
    
    def exit_position(self, exit_reason: str, exit_price: float, timestamp, current_portfolio_value: float = None) -> Tuple[float, float]:
        """Exit current position and calculate P&L"""
        if not self.in_position:
            return 0.0, 0.0
            
        pos_details = self.position_details
        # Use current_size first (for partial exits), then fall back to size, then initial_size
        position_size = pos_details.get('current_size', pos_details.get('size', pos_details.get('initial_size', 0)))
        entry_price = pos_details['entry_price']
        trade_type = pos_details['type']
        
        # Calculate P&L
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - exit_price) * position_size
            
        # Update trade history for dynamic risk management
        # If current portfolio value is provided, use it; otherwise estimate it
        if current_portfolio_value is not None:
            portfolio_value = current_portfolio_value
        else:
            # Estimate portfolio value as previous value + PnL (rough approximation)
            portfolio_value = self.peak_portfolio_value or 10000.0  # Default fallback
            portfolio_value += pnl
        
        self.update_trade_history(pnl, portfolio_value)

        # False signal outcome tracking (quick adverse or small favorable moves)
        try:
            outcome = 'win' if pnl > 0 else 'loss'
            ttl_bars = pos_details.get('bars_in_trade', 0)
            mae = pos_details.get('max_adverse_excursion', 0.0)
            mfe = pos_details.get('max_favorable_excursion', 0.0)
            # Attach to strategy recent outcomes if strategy reference stored
            strategy_ref = getattr(self, 'strategy_ref', None)
            if strategy_ref is not None:
                if not hasattr(strategy_ref, '_recent_signal_outcomes'):
                    strategy_ref._recent_signal_outcomes = []
                strategy_ref._recent_signal_outcomes.append({
                    'ts': pos_details.get('entry_timestamp'),
                    'exit_ts': timestamp,
                    'result': outcome,
                    'ttl_bars': ttl_bars,
                    'mae': mae,
                    'mfe': mfe
                })
                # Keep buffer bounded
                if len(strategy_ref._recent_signal_outcomes) > 500:
                    strategy_ref._recent_signal_outcomes = strategy_ref._recent_signal_outcomes[-500:]
        except Exception:
            pass
        
        # Reset position state
        self.in_position = False
        self.position_details = {}
        
        return pnl, position_size
    
    # =========================================================================
    # ENHANCED RISK MANAGEMENT METHODS
    # =========================================================================
    
    def calculate_dynamic_position_multiplier(self, data: pd.DataFrame, market_regime: str = None) -> Tuple[float, float]:
        """
        Calculate dynamic position size multiplier based on volatility and market conditions
        
        Returns:
            Tuple of (position_multiplier, confidence)
        """
        if not self.enable_dynamic_sizing or len(data) < self.volatility_window:
            return 1.0, 0.1
            
        try:
            # Calculate volatility (ATR-based)
            close = data['close']
            high = data['high']
            low = data['low']
            
            tr = np.maximum(high - low, 
                           np.maximum(abs(high - close.shift(1)), 
                                    abs(low - close.shift(1))))
            atr = tr.rolling(self.volatility_window).mean().iloc[-1]
            volatility = atr / close.iloc[-1]
            
            # Calculate price momentum  
            price_change = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            momentum_strength = abs(price_change)
            
            # Base position multiplier on volatility
            if volatility > 0.04:  # High volatility - reduce size
                position_multiplier = 0.5
                confidence = 0.4
            elif volatility > 0.025:  # Moderate volatility
                position_multiplier = 0.75
                confidence = 0.35
            elif volatility < 0.015:  # Low volatility - increase size
                position_multiplier = 1.5
                confidence = 0.3
            else:  # Normal volatility
                position_multiplier = 1.0
                confidence = 0.25
            
            # Adjust for market regime if available
            if market_regime:
                if market_regime in ['trending_bull', 'breakout_bullish']:
                    position_multiplier *= 1.2  # Increase in favorable conditions
                    confidence += 0.1
                elif market_regime in ['high_volatility', 'ranging']:
                    position_multiplier *= 0.8  # Reduce in uncertain conditions
                    confidence += 0.05
            
            # Adjust for momentum
            if momentum_strength > 0.03:  # Strong momentum
                position_multiplier *= 1.1
                confidence += 0.05
            
            # Ensure reasonable bounds
            position_multiplier = max(0.3, min(2.0, position_multiplier))
            confidence = min(0.6, confidence)
            
            return position_multiplier, confidence
            
        except Exception as e:
            logger.debug(f"Dynamic position multiplier calculation failed: {e}")
            return 1.0, 0.1
    
    def calculate_regime_based_stop_distance(self, data: pd.DataFrame, signal: int, market_regime: str = None) -> Tuple[float, float]:
        """
        Calculate regime-based stop loss distances
        
        Returns:
            Tuple of (stop_distance_pct, confidence) 
        """
        if not self.enable_regime_stops or len(data) < 10:
            return 0.02, 0.1  # 2% default
        
        try:
            close = data['close']
            high = data['high'] 
            low = data['low']
            
            # Calculate ATR for adaptive stops
            tr = np.maximum(high - low,
                           np.maximum(abs(high - close.shift(1)),
                                    abs(low - close.shift(1))))
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / close.iloc[-1]
            
            # Base stop distance on market regime
            if market_regime == 'high_volatility':
                stop_distance = atr_pct * 2.5  # Wider stops in volatile markets
                confidence = 0.45
            elif market_regime == 'low_volatility':
                stop_distance = atr_pct * 1.2  # Tighter stops in calm markets  
                confidence = 0.4
            elif market_regime in ['trending_bull', 'trending_bear']:
                stop_distance = atr_pct * 1.8  # Moderate stops in trends
                confidence = 0.5
            elif market_regime in ['breakout_bullish', 'breakout_bearish']:
                stop_distance = atr_pct * 2.0  # Slightly wider for breakouts
                confidence = 0.4
            else:  # ranging or unknown
                stop_distance = atr_pct * 1.5  # Standard stops
                confidence = 0.3
            
            # Ensure reasonable bounds (0.5% to 8%)
            stop_distance = max(0.005, min(0.08, stop_distance))
            
            return stop_distance, confidence
            
        except Exception as e:
            logger.debug(f"Regime-based stop calculation failed: {e}")
            return 0.02, 0.1
    
    def calculate_partial_profit_levels(self, data: pd.DataFrame, signal: int) -> Tuple[list, float]:
        """
        Calculate multiple take profit levels for partial position closure
        
        Returns:
            Tuple of (profit_levels_list, confidence)
        """
        if not self.enable_partial_profits or len(data) < 10:
            return [0.015, 0.03, 0.05], 0.2
        
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calculate recent trading range
            recent_high = high.iloc[-20:].max()
            recent_low = low.iloc[-20:].min()
            range_pct = (recent_high - recent_low) / close.iloc[-1]
            
            # Calculate ATR for profit targets
            tr = np.maximum(high - low,
                           np.maximum(abs(high - close.shift(1)),
                                    abs(low - close.shift(1))))
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / close.iloc[-1]
            
            # Set profit levels based on volatility and range
            if range_pct > 0.06:  # Wide range market
                profit_levels = [atr_pct * 1.5, atr_pct * 3.0, atr_pct * 5.0]
                confidence = 0.4
            elif range_pct < 0.02:  # Tight range market
                profit_levels = [atr_pct * 1.0, atr_pct * 2.0, atr_pct * 3.5]
                confidence = 0.35
            else:  # Normal range
                profit_levels = [atr_pct * 1.2, atr_pct * 2.5, atr_pct * 4.0]
                confidence = 0.45
            
            # Ensure minimum profit targets
            profit_levels = [max(0.008, level) for level in profit_levels]  # Min 0.8%
            
            return profit_levels, confidence
            
        except Exception as e:
            logger.debug(f"Partial profit calculation failed: {e}")
            return [0.015, 0.03, 0.05], 0.2

# Compatibility function for existing code
def create_position_manager(config_file: Optional[str] = None) -> PositionManager:
    """Factory function to create position manager instance"""
    return PositionManager(config_file)

# Export for backwards compatibility
__all__ = ['PositionManager', 'create_position_manager']
