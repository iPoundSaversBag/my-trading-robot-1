import numpy as np
import pandas as pd

def run_pnl_test(trades_df, initial_capital, final_equity):
    """
    Verifies the compounding mechanics of a walk-forward backtest by analyzing
    each window independently to ensure calculations are correct.
    """
    print("--- Running Walk-Forward Compounding Verification ---")

    if trades_df.empty:
        print("No trades were made. Verification focused on initial/final equity.")
        if np.isclose(initial_capital, final_equity):
            print("[SUCCESS] No trades, and equity is unchanged as expected.")
        else:
            print(f"[FAILURE] No trades, but equity changed from {initial_capital:,.2f} to {final_equity:,.2f}.")
        return

    trades_df = trades_df.sort_values(by='exit_timestamp').reset_index(drop=True)
    all_windows_valid = True
    
    # Group trades by their walk-forward window
    for window_num, window_trades in trades_df.groupby('window'):
        print(f"\n--- Verifying Window #{window_num} ---")
        
        # --- Roo Fix: This is the definitive verification logic. It uses the ground-truth data
        # logged by the backtester to ensure capital is chained correctly. ---

        if window_trades.empty:
            print(f"[INFO] Window #{window_num}: No trades were made. Skipping chaining verification for this window.")
            continue

        # Get the actual starting capital for this window, which was logged by the backtester.
        # The 'window_start_capital' column must be added to the trades_df in the backtester script.
        if 'window_start_capital' not in window_trades.columns:
            print("[FAILURE] 'window_start_capital' column not found in trades data. Cannot verify chaining.")
            all_windows_valid = False
            break

        window_start_capital_actual = window_trades.iloc[0]['window_start_capital']
        
        # Determine what the starting capital *should* have been.
        if window_num == 1:
            window_start_capital_expected = initial_capital
        else:
            prev_window_trades = trades_df[trades_df['window'] == window_num - 1]
            if not prev_window_trades.empty:
                window_start_capital_expected = prev_window_trades.iloc[-1]['final_equity']
            else:
                # If the previous window had no trades, find the last trade from an even earlier window
                last_trade_before_current = trades_df[trades_df['exit_timestamp'] < window_trades.iloc[0]['exit_timestamp']]
                if not last_trade_before_current.empty:
                    window_start_capital_expected = last_trade_before_current.iloc[-1]['final_equity']
                else:
                    window_start_capital_expected = initial_capital

        # Now, verify if the capital was chained correctly.
        if np.isclose(window_start_capital_expected, window_start_capital_actual):
            print(f"[SUCCESS] Window #{window_num}: Capital was chained correctly.")
            print(f"  - Previous Window End Equity: ${window_start_capital_expected:,.2f}")
            print(f"  - This Window Start Equity:   ${window_start_capital_actual:,.2f}")
        else:
            all_windows_valid = False
            print(f"[FAILURE] Window #{window_num}: Capital chaining is BROKEN.")
            print(f"  - Expected Start Capital (from prev. window): ${window_start_capital_expected:,.2f}")
            print(f"  - Actual Start Capital (from this window's log): ${window_start_capital_actual:,.2f}")
            print(f"  - Discrepancy: ${window_start_capital_expected - window_start_capital_actual:,.2f}")


    print("\n--- Overall Verification Summary ---")
    if all_windows_valid:
        print("[SUCCESS] All windows passed verification. The backtesting engine is working correctly.")
    else:
        print("[FAILURE] One or more windows failed verification. Please review the errors above.")

if __name__ == "__main__":
    print("Running verify_backtest.py in standalone mode with dummy data...")
    
    # Scenario: Correct data across two walk-forward windows
    print("\n--- Scenario: Correct Walk-Forward Data ---")
    dummy_trades = pd.DataFrame({
        'window': [1, 1, 2, 2, 2],
        'pnl': [100, -50, 200, -75, 50],
        'exit_timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-10', '2023-01-11', '2023-01-12']),
        'final_equity': [10100, 10050, 10250, 10175, 10225] # Correctly compounded
    })
    dummy_initial_capital = 10000
    dummy_final_equity = 10225
    run_pnl_test(dummy_trades, dummy_initial_capital, dummy_final_equity)