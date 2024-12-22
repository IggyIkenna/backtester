"""Analytics module for calculating backtest statistics"""
from typing import Dict
import pandas as pd
import numpy as np

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate detailed backtest statistics.
    Returns a dictionary of statistics.
    """
    stats = {}
    
    # Get trades
    trades = df[df['coin_position_change'] != 0].copy()
    winning_trades = trades[trades['trade_pnl'] > 0]
    losing_trades = trades[trades['trade_pnl'] < 0]
    
    # Calculate returns
    returns = df['total_pnl'].diff()
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    
    # Calculate drawdown
    rolling_max = df['total_pnl'].expanding().max()
    drawdown = df['total_pnl'] - rolling_max
    max_dd = drawdown.min()
    
    # Find max drawdown duration
    dd_start = None
    max_dd_duration = pd.Timedelta(0)
    
    for time, dd in drawdown.items():
        if dd < 0 and dd_start is None:
            dd_start = time
        elif dd == 0 and dd_start is not None:
            duration = time - dd_start
            max_dd_duration = max(max_dd_duration, duration)
            dd_start = None
    
    # Basic PnL metrics
    stats['total_pnl'] = df['total_pnl'].iloc[-1]
    stats['max_drawdown'] = max_dd
    stats['max_drawdown_duration'] = max_dd_duration.total_seconds() / 3600  # in hours
    
    # Risk metrics
    stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() > 0 else 0
    stats['sortino_ratio'] = returns.mean() / neg_returns.std() * np.sqrt(365 * 24) if len(neg_returns) > 0 else 0
    stats['volatility'] = returns.std() * np.sqrt(365 * 24)
    stats['var_95'] = np.percentile(returns, 5)
    stats['var_99'] = np.percentile(returns, 1)
    
    # Trade metrics
    stats['total_trades'] = len(trades)
    stats['win_rate'] = len(winning_trades) / len(trades) * 100 if len(trades) > 0 else 0
    stats['profit_factor'] = abs(winning_trades['trade_pnl'].sum() / losing_trades['trade_pnl'].sum()) if len(losing_trades) > 0 else np.inf
    stats['avg_trade_pnl'] = trades['trade_pnl'].mean() if len(trades) > 0 else 0
    stats['avg_winner'] = winning_trades['trade_pnl'].mean() if len(winning_trades) > 0 else 0
    stats['avg_loser'] = losing_trades['trade_pnl'].mean() if len(losing_trades) > 0 else 0
    stats['largest_winner'] = winning_trades['trade_pnl'].max() if len(winning_trades) > 0 else 0
    stats['largest_loser'] = losing_trades['trade_pnl'].min() if len(losing_trades) > 0 else 0
    
    # Execution metrics
    stats['maker_rate'] = trades['maker_trades'].sum() / len(trades) * 100 if len(trades) > 0 else 0
    stats['avg_trade_size'] = trades['trade_value'].mean() if len(trades) > 0 else 0
    stats['avg_trade_duration'] = trades.index.to_series().diff().mean().total_seconds() / 3600 if len(trades) > 0 else 0
    stats['max_position'] = df['coin_position'].abs().max()
    stats['missed_trades'] = df['missed_signals'].sum() if 'missed_signals' in df.columns else 0
    stats['avg_slippage_bps'] = trades['miss_bps'].mean() if 'miss_bps' in trades.columns else 0
    
    return stats