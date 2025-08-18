import pandas as pd
import numpy as np
import os
import glob

def get_latest_run_directory():
    """Finds the most recent backtest run directory."""
    try:
        # First, try to read the canonical path from the file
        with open("latest_run_dir.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to finding the most recent directory by timestamp
        list_of_dirs = glob.glob('plots_output/20*') 
        if not list_of_dirs:
            return None
        latest_dir = max(list_of_dirs, key=os.path.getctime)
        return latest_dir

def analyze_trades(run_dir):
    """
    Performs a detailed analysis of the trade log from a specific backtest run.
    """
    if not run_dir:
        print("❌ Could not find the latest run directory.")
        return

    trades_file = os.path.join(run_dir, 'all_trades_detailed.csv')
    if not os.path.exists(trades_file):
        print(f"❌ Trades file not found in directory: {trades_file}")
        return

    print(f"🔍 Analyzing trade data from: {trades_file}\n")
    df = pd.read_csv(trades_file)

    # --- Analysis ---

    # 1. Overall Performance
    total_pnl = df['pnl'].sum()
    total_trades = len(df)
    win_rate = (df['pnl'] > 0).sum() / total_trades if total_trades > 0 else 0
    
    print("--- Overall Performance ---")
    print(f"Total Trades: {total_trades}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print("-" * 27)

    # 2. Exit Reason Analysis
    exit_analysis = df.groupby('exit_reason')['pnl'].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=True)
    print("\n--- PnL by Exit Reason ---")
    print(exit_analysis)
    print("-" * 26)
    
    hard_stop_losses = exit_analysis.loc['Hard Stop Loss']['sum'] if 'Hard Stop Loss' in exit_analysis.index else 0
    trailing_stop_losses = exit_analysis.loc['Trailing Stop Loss']['sum'] if 'Trailing Stop Loss' in exit_analysis.index else 0

    # 3. Trade Type Analysis (Long vs. Short)
    trade_type_analysis = df.groupby('trade_type')['pnl'].agg(['sum', 'mean', 'count'])
    # Remap trade_type for readability
    trade_type_analysis.index = trade_type_analysis.index.map({1: 'Long', -1: 'Short'})
    print("\n--- PnL by Trade Type ---")
    print(trade_type_analysis)
    print("-" * 25)

    # 4. PnL Distribution
    print("\n--- PnL Distribution ---")
    print(df['pnl'].describe())
    print("-" * 24)

    # --- Summary & Diagnosis ---
    print("\n--- Diagnosis & Recommendations ---")
    if total_pnl < 0:
        print(f"🔴 CRITICAL: The strategy is unprofitable, losing ${-total_pnl:,.2f}.")
    else:
        print(f"🟢 The strategy is profitable, making ${total_pnl:,.2f}.")

    # Check if stop losses are the primary source of failure
    if hard_stop_losses + trailing_stop_losses < 0 and abs(hard_stop_losses + trailing_stop_losses) > abs(total_pnl) * 0.5:
        print(f"🔴 DIAGNOSIS: Stop losses are the main cause of capital drain, accounting for ${hard_stop_losses + trailing_stop_losses:,.2f} in losses.")
        print("   - This suggests that either the entry signals are poor, leading to immediate adverse movement, or the stop-loss placement is too tight, cutting trades before they can become profitable.")
        print("   - Given the high number of trades, poor entry quality is the most likely culprit.")
    
    # Check for imbalance in long/short performance
    if 'Long' in trade_type_analysis.index and 'Short' in trade_type_analysis.index:
        long_pnl = trade_type_analysis.loc['Long']['sum']
        short_pnl = trade_type_analysis.loc['Short']['sum']
        if long_pnl < 0 and short_pnl < 0:
            print("🔴 DIAGNOSIS: Both LONG and SHORT trades are losing money. The strategy's core logic is flawed in all market directions.")
        elif long_pnl < 0 and short_pnl > 0:
            print("🟡 DIAGNOSIS: LONG trades are failing while SHORT trades are profitable. The strategy is only effective in bearish conditions.")
        elif long_pnl > 0 and short_pnl < 0:
            print("🟡 DIAGNOSIS: SHORT trades are failing while LONG trades are profitable. The strategy is only effective in bullish conditions.")

    print("\n--- Next Steps ---")
    print("1. Focus on Entry Signal Quality: The current signals lead to trades that immediately hit their stops. We need to add more confirmation filters.")
    print("2. Add a Market Regime Filter: The strategy trades indiscriminately. We should add a filter (e.g., a long-term moving average) to only allow trades in the direction of the primary trend.")
    print("3. Review Signal Parameters: The underlying indicators (RSI, MACD, etc.) may need different periods or thresholds to be effective.")


if __name__ == "__main__":
    latest_run = get_latest_run_directory()
    analyze_trades(latest_run)
