# ==============================================================================
#
#                           INTERACTIVE VISUALIZATION ENGINE
#
# ==============================================================================
#
# FILE: generate_plots.py
#
# PURPOSE:
#   This script is the dedicated visualization module for the trading pipeline.
#   It uses the Plotly library to generate rich, interactive charts that are
#   embedded directly into the final QuantStats HTML report.
#
# METHODOLOGY:
#   The script's functions are called by the main backtester. For each walk-
#   forward window, it:
#   1.  Calculates all necessary technical indicators (Ichimoku, BBands, RSI, ADX)
#       using the exact parameters that were found to be optimal for that window.
#   2.  Generates a multi-panel interactive chart:
#       - Candlestick chart with overlaid Ichimoku Cloud, Bollinger Bands, and trades.
#       - RSI indicator in a separate subplot.
#       - ADX indicator in a third subplot.
#   3.  Draws shapes and annotations for each trade, clearly marking the entry
#       and exit points and coloring them based on profitability.
#   4.  Returns the generated plot as an HTML div string to be embedded in the
#       main report.
#
# ==============================================================================

import pandas as pd
import numpy as np
import json
import os
import glob
import warnings
import ta
import datetime
import argparse
import io
import base64

# --- Ensure plotting libraries are available ---
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly is not installed. Please run 'pip install plotly'.")
    PLOTLY_AVAILABLE = False

def log_to_file(message, print_to_console=True):
    if print_to_console:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def plot_pnl_distribution(trades_df, return_html_div=False):
    """Generates a histogram of the Profit and Loss for all trades."""
    if not PLOTLY_AVAILABLE or trades_df.empty:
        return None

    fig = go.Figure(data=[go.Histogram(x=trades_df['pnl'], nbinsx=50, marker_color='#636EFA')])
    fig.update_layout(
        title_text='Distribution of Trade PnL',
        xaxis_title_text='Profit/Loss (USD)',
        yaxis_title_text='Number of Trades',
        bargap=0.1,
        template='plotly_dark'
    )
    
    if return_html_div:
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    return None


def plot_trades_for_window(full_df, window_trades, window_key, optimized_params, config, return_html_div=False):
    """Generates an interactive Plotly chart for a single walk-forward window."""
    if not PLOTLY_AVAILABLE:
        log_to_file("Plotting skipped: plotly not installed.")
        return None

    try:
        if window_trades.empty:
            log_to_file(f"No trades to plot for {window_key}.")
            return None

        plot_start = window_trades['entry_timestamp'].min() - pd.Timedelta(days=1)
        plot_end = window_trades['exit_timestamp'].max() + pd.Timedelta(days=1)
        plot_df = full_df.loc[plot_start:plot_end].copy()

        # --- Indicator Calculations for Plotting ---
        params = {**config.get('fixed_parameters', {}), **optimized_params[window_key]}
        
        # Ichimoku
        plot_df['tenkan'] = ta.trend.ichimoku_a(plot_df['high'], plot_df['low'], window1=params['TENKAN_SEN_PERIOD'])
        plot_df['kijun'] = ta.trend.ichimoku_b(plot_df['high'], plot_df['low'], window2=params['KIJUN_SEN_PERIOD'])
        plot_df['senkou_a'] = ((plot_df['tenkan'] + plot_df['kijun']) / 2).shift(params['KIJUN_SEN_PERIOD'])
        plot_df['senkou_b'] = ((plot_df['high'].rolling(window=params['SENKOU_SPAN_B_PERIOD']).max() + plot_df['low'].rolling(window=params['SENKOU_SPAN_B_PERIOD']).min()) / 2).shift(params['KIJUN_SEN_PERIOD'])
        
        # Bollinger Bands
        bbands = ta.volatility.BollingerBands(close=plot_df['close'], window=params['BBANDS_PERIOD'], window_dev=params['BBANDS_STD_DEV'])
        plot_df['bb_high'] = bbands.bollinger_hband()
        plot_df['bb_low'] = bbands.bollinger_lband()
        plot_df['bb_width'] = bbands.bollinger_wband()
        
        # RSI & ADX & ATR
        plot_df['rsi'] = ta.momentum.RSIIndicator(close=plot_df['close'], window=params['RSI_PERIOD']).rsi()
        plot_df['adx'] = ta.trend.ADXIndicator(high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], window=params['ADX_PERIOD']).adx()
        plot_df['atr'] = ta.volatility.average_true_range(high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], window=params['ATR_PERIOD'])
        plot_df['is_trending'] = (plot_df['adx'] > params['ADX_TREND_THRESHOLD']) & (plot_df['bb_width'] > params['BB_WIDTH_THRESHOLD'])


        # --- ROO: Robust data cleaning to handle infinity and NaN values ---
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        plot_df.dropna(inplace=True)
        
        # --- Create Interactive Plot ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f"Price and Trades for {window_key}", "RSI", "ADX"))

        # 1. Candlestick Chart
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
                                     low=plot_df['low'], close=plot_df['close'], name='Price'), row=1, col=1)

        # 2. Ichimoku Cloud & BBands
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['senkou_a'], name='Senkou A', line=dict(color='rgba(0, 255, 0, 0.4)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['senkou_b'], name='Senkou B', line=dict(color='rgba(255, 0, 0, 0.4)', width=1),
                                 fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['bb_high'], name='BB High', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['bb_low'], name='BB Low', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)

        # --- IMPROVEMENT: Re-calculate and plot Trailing Stop Loss ---
        tsl_series = pd.Series(np.nan, index=plot_df.index)
        for entry_ts, trade_group in window_trades.groupby('entry_timestamp'):
            trade_group = trade_group.sort_values('exit_timestamp')
            initial_trade = trade_group.iloc[0]
            entry_price = initial_trade['entry_price']
            is_long = initial_trade['trade_type'] == 1
            
            regime_at_entry = plot_df.loc[initial_trade['entry_timestamp']]['is_trending']
            tsl_multiplier_at_entry = params['TRENDING_TSL_ATR_MULTIPLIER'] if regime_at_entry else params['RANGING_TSL_ATR_MULTIPLIER']
            atr_at_entry = plot_df.loc[initial_trade['entry_timestamp']]['atr']
            current_tsl = entry_price - (atr_at_entry * tsl_multiplier_at_entry) if is_long else entry_price + (atr_at_entry * tsl_multiplier_at_entry)
            
            partial_exit_taken = False
            full_trade_slice = plot_df.loc[initial_trade['entry_timestamp']:trade_group.iloc[-1]['exit_timestamp']]

            for idx, candle in full_trade_slice.iterrows():
                tsl_series.loc[idx] = current_tsl
                if any((t['exit_reason'] == 'Partial TP' and t['exit_timestamp'] == idx) for _, t in trade_group.iterrows()):
                    partial_exit_taken = True
                
                regime = candle['is_trending']
                tsl_multiplier = params['TRENDING_TSL_ATR_MULTIPLIER'] if regime else params['RANGING_TSL_ATR_MULTIPLIER']
                
                if is_long:
                    new_tsl = candle['close'] - (candle['atr'] * tsl_multiplier)
                    current_tsl = max(current_tsl, new_tsl)
                    if partial_exit_taken: current_tsl = max(current_tsl, entry_price)
                else:
                    new_tsl = candle['close'] + (candle['atr'] * tsl_multiplier)
                    current_tsl = min(current_tsl, new_tsl)
                    if partial_exit_taken: current_tsl = min(current_tsl, entry_price)
        
        fig.add_trace(go.Scatter(x=tsl_series.index, y=tsl_series.ffill(), name='Trailing SL', mode='lines', line=dict(color='yellow', width=1, dash='dot')), row=1, col=1)

        # --- ROO: Refactored Plotting for Efficiency and Correctness ---
        # Collect all points first, then plot them in batches. This is much more efficient and avoids Plotly rendering bugs.
        long_entries_x, long_entries_y = [], []
        short_entries_x, short_entries_y = [], []
        win_exits_x, win_exits_y = [], []
        loss_exits_x, loss_exits_y = [], []

        shapes = []
        annotations = []
        for _, trade in window_trades.iterrows():
            if trade['trade_type'] == 1: # Long
                long_entries_x.append(trade['entry_timestamp'])
                long_entries_y.append(trade['entry_price'])
            else: # Short
                short_entries_x.append(trade['entry_timestamp'])
                short_entries_y.append(trade['entry_price'])

            if trade['pnl'] > 0: # Win
                win_exits_x.append(trade['exit_timestamp'])
                win_exits_y.append(trade['exit_price'])
            else: # Loss
                loss_exits_x.append(trade['exit_timestamp'])
                loss_exits_y.append(trade['exit_price'])

            # Batch shapes and annotations for performance
            exit_color = 'green' if trade['pnl'] > 0 else 'red'
            shapes.append(dict(type="line", xref="x", yref="y", x0=trade['entry_timestamp'], y0=trade['entry_price'], x1=trade['exit_timestamp'], y1=trade['exit_price'], line=dict(color=exit_color, width=1, dash='dash')))
            annotations.append(dict(x=trade['exit_timestamp'], y=trade['exit_price'], xref="x", yref="y", text=f"PnL: {trade['pnl']:.2f}<br>{trade['exit_reason']}", showarrow=True, arrowhead=1, ax=0, ay=-40, bgcolor="rgba(0,0,0,0.7)", bordercolor=exit_color, borderwidth=1, font=dict(color="white", size=9)))

        # Now, add the markers in efficient batches
        fig.add_trace(go.Scatter(x=long_entries_x, y=long_entries_y, mode='markers', marker=dict(color='blue', size=10, symbol='arrow-up', line=dict(width=1,color='white')), name='Long Entry'), row=1, col=1)
        fig.add_trace(go.Scatter(x=short_entries_x, y=short_entries_y, mode='markers', marker=dict(color='purple', size=10, symbol='arrow-down', line=dict(width=1,color='white')), name='Short Entry'), row=1, col=1)
        fig.add_trace(go.Scatter(x=win_exits_x, y=win_exits_y, mode='markers', marker=dict(color='green', size=8, symbol='x'), name='Winning Exit'), row=1, col=1)
        fig.add_trace(go.Scatter(x=loss_exits_x, y=loss_exits_y, mode='markers', marker=dict(color='red', size=8, symbol='x'), name='Losing Exit'), row=1, col=1)

        # 5. RSI Subplot
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['rsi'], name='RSI', line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=params['RSI_OVERBOUGHT'], line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=params['RSI_OVERSOLD'], line_dash="dash", line_color="green", row=2, col=1)

        # 6. ADX Subplot
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['adx'], name='ADX', line=dict(color='aqua')), row=3, col=1)
        fig.add_hline(y=params['ADX_TREND_THRESHOLD'], line_dash="dash", line_color="white", row=3, col=1)

        # --- Final Layout Touches ---
        fig.update_layout(template='plotly_dark', showlegend=False, height=800, margin=dict(l=20, r=20, t=40, b=20), shapes=shapes, annotations=annotations)
        fig.update_xaxes(rangeslider_visible=False)

        if return_html_div:
            log_to_file(f"Generated interactive plot for {window_key}")
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        return None

    except Exception as e:
        log_to_file(f"Error plotting trades for {window_key}: {e}")
        import traceback
        traceback.print_exc()
        return None