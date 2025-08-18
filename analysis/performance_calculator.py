#!/usr/bin/env python3
"""
Performance Metrics Calculator
Properly calculate trading performance metrics from trading journal data
and export to JSON for reliable integration with performance reports.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class PerformanceCalculator:
    """Calculate accurate trading performance metrics from trading journal data."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.metrics = {}
        
    def load_trading_journal(self, journal_path: str = "data/trading_journal.json") -> Dict[str, Any]:
        """Load trading journal data."""
        if not os.path.exists(journal_path):
            raise FileNotFoundError(f"Trading journal not found: {journal_path}")
            
        with open(journal_path, 'r') as f:
            return json.load(f)
    
    def calculate_equity_curve(self, journal_data: Dict[str, Any]) -> pd.DataFrame:
        """Calculate equity curve from session data."""
        sessions = journal_data.get('sessions', [])
        
        equity_data = []
        running_equity = self.initial_capital
        
        for session in sessions:
            session_pnl = session.get('total_pnl', 0)
            session_date = session.get('timestamp', datetime.now().isoformat())
            
            # Update equity
            running_equity += session_pnl
            running_equity = max(running_equity, 0)  # Can't go below 0
            
            equity_data.append({
                'date': session_date,
                'equity': running_equity,
                'pnl': session_pnl,
                'session_id': session.get('session_id', 'unknown')
            })
        
        return pd.DataFrame(equity_data)
    
    def calculate_performance_metrics(self, journal_path: str = "data/trading_journal.json") -> Dict[str, Any]:
        """Calculate all performance metrics."""
        try:
            # Load data
            journal_data = self.load_trading_journal(journal_path)
            equity_df = self.calculate_equity_curve(journal_data)
            
            if equity_df.empty:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_trades = journal_data.get('metadata', {}).get('total_trades', 0)
            total_sessions = len(journal_data.get('sessions', []))
            
            # Calculate returns
            final_equity = equity_df['equity'].iloc[-1]
            total_pnl = final_equity - self.initial_capital
            
            # Cumulative Return (capped at -100%)
            cumulative_return = max((total_pnl / self.initial_capital) * 100, -100)
            
            # Calculate Max Drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
            max_drawdown = min(equity_df['drawdown'].max(), 100)  # Cap at 100%
            
            # Calculate CAGR
            # Estimate time period (assume recent trading over reasonable period)
            estimated_days = max(total_sessions * 2, 30)  # Assume sessions are every ~2 days
            estimated_years = estimated_days / 365.25
            
            if final_equity > 0 and estimated_years > 0:
                cagr = ((final_equity / self.initial_capital) ** (1/estimated_years) - 1) * 100
                cagr = max(cagr, -100)  # Cap at -100%
            else:
                cagr = -100
            
            # Risk Metrics
            daily_returns = equity_df['pnl'] / equity_df['equity'].shift(1) * 100
            daily_returns = daily_returns.dropna()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0
            var_95 = max(var_95, -50)  # Reasonable bound
            
            # Expected Shortfall (Conditional VaR)
            expected_shortfall = daily_returns[daily_returns <= var_95].mean() if len(daily_returns) > 0 else 0
            expected_shortfall = max(expected_shortfall, -50)  # Reasonable bound
            
            # Volatility
            volatility = daily_returns.std() if len(daily_returns) > 1 else 0
            annualized_volatility = volatility * np.sqrt(252)
            
            # Sharpe Ratio (assuming 0% risk-free rate)
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
            
            # Win Rate
            winning_sessions = sum(1 for session in journal_data.get('sessions', []) if session.get('total_pnl', 0) > 0)
            win_rate = (winning_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            # Drawdown dates
            max_dd_idx = equity_df['drawdown'].idxmax()
            max_dd_date = equity_df.loc[max_dd_idx, 'date'] if not pd.isna(max_dd_idx) else "N/A"
            
            # Profit factor
            gross_profit = sum(session.get('total_pnl', 0) for session in journal_data.get('sessions', []) if session.get('total_pnl', 0) > 0)
            gross_loss = abs(sum(session.get('total_pnl', 0) for session in journal_data.get('sessions', []) if session.get('total_pnl', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            self.metrics = {
                # Core Performance Metrics
                'cumulative_return': round(cumulative_return, 2),
                'cagr': round(cagr, 2),
                'max_drawdown': round(max_drawdown, 2),
                'max_drawdown_date': max_dd_date,
                
                # Risk Metrics  
                'daily_var_95': round(abs(var_95), 2),
                'expected_shortfall': round(abs(expected_shortfall), 2),
                'volatility_annualized': round(annualized_volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 4),
                
                # Trading Metrics
                'total_trades': total_trades,
                'total_sessions': total_sessions,
                'win_rate': round(win_rate, 2),
                'profit_factor': round(profit_factor, 4),
                
                # Equity Metrics
                'initial_capital': self.initial_capital,
                'final_equity': round(final_equity, 2),
                'total_pnl': round(total_pnl, 2),
                'gross_profit': round(gross_profit, 2),
                'gross_loss': round(gross_loss, 2),
                
                # Metadata
                'calculation_date': datetime.now().isoformat(),
                'data_quality': 'complete' if total_trades > 100 else 'limited',
                'estimated_period_days': estimated_days,
                'estimated_period_years': round(estimated_years, 2)
            }
            
            return self.metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty/default metrics when calculation fails."""
        return {
            'cumulative_return': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_date': 'N/A',
            'daily_var_95': 0.0,
            'expected_shortfall': 0.0,
            'volatility_annualized': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'total_sessions': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'calculation_date': datetime.now().isoformat(),
            'data_quality': 'no_data',
            'estimated_period_days': 0,
            'estimated_period_years': 0
        }
    
    def save_metrics(self, output_path: str = "data/calculated_performance_metrics.json") -> bool:
        """Save calculated metrics to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"✅ Performance metrics saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving metrics: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of calculated metrics."""
        if not self.metrics:
            print("❌ No metrics calculated yet. Run calculate_performance_metrics() first.")
            return
        
        print("\n" + "="*60)
        print("📊 CALCULATED PERFORMANCE METRICS SUMMARY")
        print("="*60)
        
        print(f"🎯 Core Performance:")
        print(f"   • Cumulative Return: {self.metrics['cumulative_return']:.2f}%")
        print(f"   • CAGR: {self.metrics['cagr']:.2f}%")
        print(f"   • Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        print(f"\n⚖️ Risk Metrics:")
        print(f"   • Daily VaR (95%): {self.metrics['daily_var_95']:.2f}%")
        print(f"   • Expected Shortfall: {self.metrics['expected_shortfall']:.2f}%")
        print(f"   • Annualized Volatility: {self.metrics['volatility_annualized']:.2f}%")
        print(f"   • Sharpe Ratio: {self.metrics['sharpe_ratio']:.4f}")
        
        print(f"\n📈 Trading Stats:")
        print(f"   • Total Trades: {self.metrics['total_trades']:,}")
        print(f"   • Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"   • Profit Factor: {self.metrics['profit_factor']:.4f}")
        
        print(f"\n💰 Financial Summary:")
        print(f"   • Initial Capital: ${self.metrics['initial_capital']:,.2f}")
        print(f"   • Final Equity: ${self.metrics['final_equity']:,.2f}")
        print(f"   • Total P&L: ${self.metrics['total_pnl']:,.2f}")
        
        print(f"\n📅 Data Quality:")
        print(f"   • Data Quality: {self.metrics['data_quality']}")
        print(f"   • Estimated Period: {self.metrics['estimated_period_days']} days ({self.metrics['estimated_period_years']:.2f} years)")
        print("="*60)


def main():
    """Main function to calculate and save performance metrics."""
    calculator = PerformanceCalculator()
    
    print("🔢 Calculating Performance Metrics...")
    metrics = calculator.calculate_performance_metrics()
    
    # Print summary
    calculator.print_summary()
    
    # Save to file
    success = calculator.save_metrics()
    
    if success:
        print(f"\n🎉 Performance metrics calculation complete!")
        print(f"📁 Metrics saved to: data/calculated_performance_metrics.json")
    else:
        print(f"\n❌ Failed to save performance metrics.")
    
    return success


if __name__ == "__main__":
    main()
