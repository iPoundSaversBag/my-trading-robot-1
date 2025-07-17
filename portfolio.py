# ==============================================================================
#
#                               PORTFOLIO CLASS
#
# ==============================================================================
#
# FILE: portfolio.py
#
# PURPOSE:
#   This module defines the `Portfolio` class, which is responsible for
#   managing the state of our trading account. This includes tracking capital,
#   positions, and calculating P&L.
#
# ==============================================================================

import numpy as np
import pandas as pd

class Portfolio:
    """
    A class to manage the portfolio and track its performance.
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.history = []

    def add_trade(self, trade):
        """
        Adds a trade to the portfolio's history.
        """
        self.history.append(trade)
        self.capital = trade['final_equity']

    def get_equity_curve(self):
        """
        Returns the equity curve of the portfolio.
        """
        if not self.history:
            return np.array([self.initial_capital])
        
        equity = [self.initial_capital]
        for trade in self.history:
            equity.append(trade['final_equity'])
        return np.array(equity)

    def calculate_performance_metrics(self, trades_df, initial_capital, final_equity):
        """
        Calculates key performance metrics from a DataFrame of trades.
        """
        if trades_df.empty:
            return {"Sharpe Ratio": 0.0, "Final Equity": initial_capital, "Sortino Ratio": 0.0, "Calmar Ratio": 0.0, "Max Drawdown": 0.0, "Total Trades": 0}

        equity_curve = trades_df.set_index('exit_timestamp')['pnl'].cumsum() + initial_capital
        if equity_curve.empty:
            return {"Sharpe Ratio": 0.0, "Final Equity": initial_capital, "Sortino Ratio": 0.0, "Calmar Ratio": 0.0, "Max Drawdown": 0.0, "Total Trades": 0}

        returns = equity_curve.pct_change().dropna()
        
        max_drawdown = 0.0

        if returns.empty or returns.std() == 0:
            return {"Sharpe Ratio": 0.0, "Final Equity": final_equity, "Sortino Ratio": 0.0, "Calmar Ratio": 0.0, "Max Drawdown": max_drawdown, "Total Trades": len(trades_df)}

        annualization_factor = np.sqrt(252 * (24 * 12)) # 5-min intervals
        sharpe = (returns.mean() / returns.std()) * annualization_factor if returns.std() > 0 else 0.0

        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        sortino = (returns.mean() / downside_std) * annualization_factor if downside_std > 0 else 0.0
        
        days_in_data = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days_in_data > 0:
            annual_return = (final_equity / initial_capital)**(365.25 / days_in_data) - 1
        else:
            annual_return = (final_equity / initial_capital) - 1

        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        return {
            "Sharpe Ratio": sharpe, 
            "Final Equity": final_equity, 
            "Sortino Ratio": sortino, 
            "Calmar Ratio": calmar, 
            "Max Drawdown": max_drawdown, 
            "Total Trades": len(trades_df)
        }
