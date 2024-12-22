import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.utils.types import BacktestState, Trade
from backtester.configs.backtester_config import BacktestConfig
from backtester.utils.exceptions import DataOutputError, StrategyError, MetricsError
from backtester.utils.error_handler import BacktestErrorHandler

class OutputHandler:
    """Handles backtest output, logging, and performance analysis"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_dir = config.results_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(__name__)
        self.error_handler = BacktestErrorHandler(config, self.logger)
        self._log_output_config()
        
        # Create results directory
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results containers
        self.trades: List[Trade] = []
        self.state_history: List[Dict] = []
        
    def _log_output_config(self) -> None:
        """Log output configuration"""
        self.logger.info("Output Handler Configuration:")
        self.logger.info(f"Results Directory: {self.config.results_dir}")
        self.logger.info(f"Log Directory: {self.config.log_dir}")
        self.logger.info(f"Log Level: {self.config.log_level}")
        self.logger.info("Output Settings:")
        self.logger.info(f"  Save Trades: {self.config.save_trades}")
        self.logger.info(f"  Save Signals: {self.config.save_signals}")
    
    def log_state(self, state: BacktestState) -> None:
        """Log backtest state for analysis"""
        state_data = {
            'timestamp': state.timestamp.isoformat(),
            'market': {
                'open': state.market.open,
                'high': state.market.high,
                'low': state.market.low,
                'close': state.market.close,
                'volume': state.market.volume
            },
            'position': {
                'size': state.position.size,
                'entry_price': state.position.entry_price,
                'equity': state.position.equity,
                'leverage': state.position.leverage,
                'unrealized_pnl': state.position.unrealized_pnl,
                'realized_pnl': state.position.realized_pnl
            },
            'risk': state.risk_metrics.__dict__ if state.risk_metrics else {},
            'trading': {
                'state': state.trading_state.value,
                'reason': state.trading_disabled_reason
            }
        }
        self.state_history.append(state_data)
    
    def log_trade(self, trade: Trade) -> None:
        """Log executed trade"""
        self.trades.append(trade)
        
        log_data = {
            'timestamp': trade.timestamp.isoformat(),
            'trade_id': trade.trade_id,
            'signal_id': trade.signal_id,
            'side': trade.side.value,
            'size': trade.size,
            'price': trade.price,
            'fees': trade.fees,
            'is_maker': trade.is_maker
        }
        self.logger.info(f"TRADE_EXECUTED: {json.dumps(log_data)}")
    
    def save_results(self) -> None:
        """Save all results to files"""
        try:
            # Save state history
            try:
                state_df = pd.DataFrame(self.state_history)
                state_df.to_csv(self.run_dir / 'state_history.csv', index=False)
            except Exception as e:
                raise DataOutputError(
                    operation="save_state_history",
                    file_path=str(self.run_dir / 'state_history.csv'),
                    reason=str(e)
                )

            # Save trade history
            if self.trades:
                try:
                    trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
                    trades_df.to_csv(self.run_dir / 'trades.csv', index=False)
                except Exception as e:
                    raise DataOutputError(
                        operation="save_trades",
                        file_path=str(self.run_dir / 'trades.csv'),
                        reason=str(e)
                    )

            # Calculate and save metrics
            try:
                metrics = self._calculate_performance_metrics()
                with open(self.run_dir / 'metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                raise MetricsError(
                    metric_name="performance_metrics",
                    calculation="all",
                    reason=str(e)
                )

        except Exception as e:
            self.error_handler.handle_execution_error(e, None, None)
            if not self.config.continue_on_error:
                raise
    
    def create_visualizations(self) -> None:
        """Create all visualizations"""
        self._create_performance_chart()
        self._create_risk_metrics_chart()
        self._create_trade_analysis()
        self._create_drawdown_chart()
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics with error handling"""
        try:
            if not self.state_history:
                return {}

            metrics = {}
            
            try:
                metrics['returns'] = self._calculate_returns()
            except Exception as e:
                raise MetricsError(
                    metric_name="returns",
                    calculation="return_calculation",
                    reason=str(e)
                )

            try:
                metrics['risk'] = self._calculate_risk_metrics()
            except Exception as e:
                raise MetricsError(
                    metric_name="risk_metrics",
                    calculation="risk_calculation",
                    reason=str(e)
                )

            return metrics

        except Exception as e:
            self.error_handler.handle_execution_error(e, None, None)
            if not self.config.continue_on_error:
                raise
            return {}
    
    def _create_performance_chart(self) -> None:
        """Create main performance chart with trading state overlay"""
        df = pd.DataFrame(self.state_history)
        
        fig = make_subplots(rows=4, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price & Trading State', 'Equity', 'Leverage', 'Volume'),
                           row_heights=[0.4, 0.2, 0.2, 0.2])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['market.open'],
                high=df['market.high'],
                low=df['market.low'],
                close=df['market.close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add trading state overlay
        disabled_periods = df[df['trading.state'] != 'enabled']
        for _, period in disabled_periods.iterrows():
            fig.add_vrect(
                x0=period['timestamp'],
                x1=period['timestamp'],
                fillcolor="gray",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=1, col=1,
                hovertext=period['trading.reason']
            )
        
        # Add legend for trading states
        unique_states = df['trading.state'].unique()
        for state in unique_states:
            if state != 'enabled':
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='lines',
                        fill='toself',
                        fillcolor="gray",
                        opacity=0.3,
                        name=f"Trading {state.replace('_', ' ').title()}",
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add trades
        for trade in self.trades:
            color = 'green' if trade.side.value == 'BUY' else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade.timestamp],
                    y=[trade.price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.side.value == 'BUY' else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    name=f"{trade.side.value} {abs(trade.size):.4f}"
                ),
                row=1, col=1
            )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['position.equity'],
                name='Equity'
            ),
            row=2, col=1
        )
        
        # Leverage
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['position.leverage'],
                name='Leverage'
            ),
            row=3, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['market.volume'],
                name='Volume'
            ),
            row=4, col=1
        )
        
        fig.update_layout(height=1200, title='Backtest Performance')
        fig.write_html(self.run_dir / 'performance_chart.html')
    
    def _create_risk_metrics_chart(self) -> None:
        """Create risk metrics visualization"""
        df = pd.DataFrame(self.state_history)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Rolling Volatility', 'Value at Risk',
                                         'Rolling Sharpe', 'Drawdown'))
        
        # Volatility
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['risk.volatility'],
                      name='Volatility'),
            row=1, col=1
        )
        
        # VaR
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['risk.var_95'],
                      name='95% VaR'),
            row=1, col=2
        )
        
        # Sharpe
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['risk.sharpe_ratio'],
                      name='Rolling Sharpe'),
            row=2, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['risk.max_drawdown'],
                      name='Drawdown', fill='tozeroy'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title='Risk Metrics')
        fig.write_html(self.run_dir / 'risk_metrics.html')
    
    def _create_trade_analysis(self) -> None:
        """Create trade analysis visualization"""
        if not self.trades:
            return
            
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'size': abs(t.size),
                'price': t.price,
                'side': t.side.value,
                'is_maker': t.is_maker,
                'fees': t.fees
            }
            for t in self.trades
        ])
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Trade Size Distribution', 'Buy vs Sell',
                                         'Maker vs Taker', 'Fees Distribution'))
        
        # Size distribution
        fig.add_trace(
            go.Histogram(x=df['size'], name='Trade Sizes'),
            row=1, col=1
        )
        
        # Buy vs Sell
        side_counts = df['side'].value_counts()
        fig.add_trace(
            go.Pie(labels=side_counts.index, values=side_counts.values,
                  name='Trade Sides'),
            row=1, col=2
        )
        
        # Maker vs Taker
        maker_counts = df['is_maker'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Maker', 'Taker'], 
                  values=[maker_counts.get(True, 0), maker_counts.get(False, 0)],
                  name='Execution Type'),
            row=2, col=1
        )
        
        # Fees
        fig.add_trace(
            go.Histogram(x=df['fees'], name='Fees'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title='Trade Analysis')
        fig.write_html(self.run_dir / 'trade_analysis.html')
    
    def _create_drawdown_chart(self) -> None:
        """Create drawdown analysis"""
        df = pd.DataFrame(self.state_history)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['risk.max_drawdown'],
                fill='tozeroy',
                name='Drawdown'
            )
        )
        
        fig.update_layout(
            title='Drawdown Analysis',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        fig.write_html(self.run_dir / 'drawdown.html')
    
    def _calculate_returns(self) -> Dict[str, float]:
        """Calculate return metrics"""
        df = pd.DataFrame(self.state_history)
        
        # Calculate returns
        df['returns'] = df['position.equity'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Calculate annualized metrics
        trading_days = len(df)
        annualization_factor = np.sqrt(252 / trading_days)
        
        return {
            'total_return': df['cumulative_returns'].iloc[-1],
            'annualized_return': df['cumulative_returns'].iloc[-1] * (252 / trading_days),
            'volatility': df['returns'].std() * annualization_factor,
            'sharpe_ratio': (df['returns'].mean() / df['returns'].std()) * annualization_factor if df['returns'].std() > 0 else 0,
            'sortino_ratio': (df['returns'].mean() / df[df['returns'] < 0]['returns'].std()) * annualization_factor if len(df[df['returns'] < 0]) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(df['position.equity']),
            'win_rate': len(df[df['returns'] > 0]) / len(df[df['returns'] != 0]) if len(df[df['returns'] != 0]) > 0 else 0,
            'profit_factor': abs(df[df['returns'] > 0]['returns'].sum() / df[df['returns'] < 0]['returns'].sum()) if len(df[df['returns'] < 0]) > 0 else float('inf')
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics"""
        df = pd.DataFrame(self.state_history)
        
        # Position metrics
        position_metrics = {
            'avg_position_size': df['position.size'].abs().mean(),
            'max_position_size': df['position.size'].abs().max(),
            'avg_leverage': df['position.leverage'].mean(),
            'max_leverage': df['position.leverage'].max(),
            'time_in_market': len(df[df['position.size'] != 0]) / len(df)
        }
        
        # PnL metrics
        pnl_metrics = {
            'total_pnl': df['position.realized_pnl'].iloc[-1] + df['position.unrealized_pnl'].iloc[-1],
            'realized_pnl': df['position.realized_pnl'].iloc[-1],
            'unrealized_pnl': df['position.unrealized_pnl'].iloc[-1],
            'total_fees': df['position.maker_fees'].iloc[-1] + df['position.taker_fees'].iloc[-1],
            'maker_fees': df['position.maker_fees'].iloc[-1],
            'taker_fees': df['position.taker_fees'].iloc[-1]
        }
        
        # Risk metrics
        risk_metrics = {
            'var_95': np.percentile(df['returns'], 5) if 'returns' in df else 0,
            'expected_shortfall': df[df['returns'] <= np.percentile(df['returns'], 5)]['returns'].mean() if 'returns' in df else 0,
            'volatility_annual': df['returns'].std() * np.sqrt(252) if 'returns' in df else 0,
            'downside_deviation': df[df['returns'] < 0]['returns'].std() * np.sqrt(252) if len(df[df['returns'] < 0]) > 0 else 0
        }
        
        # Trading metrics
        if self.trades:
            trades_df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'size': abs(t.size),
                'price': t.price,
                'is_maker': t.is_maker,
                'fees': t.fees
            } for t in self.trades])
            
            trading_metrics = {
                'trade_count': len(trades_df),
                'maker_ratio': trades_df['is_maker'].mean(),
                'avg_trade_size': trades_df['size'].mean(),
                'avg_fees_per_trade': trades_df['fees'].mean()
            }
        else:
            trading_metrics = {
                'trade_count': 0,
                'maker_ratio': 0,
                'avg_trade_size': 0,
                'avg_fees_per_trade': 0
            }
        
        return {
            'position': position_metrics,
            'pnl': pnl_metrics,
            'risk': risk_metrics,
            'trading': trading_metrics
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        rolling_max = equity_series.expanding(min_periods=1).max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        return abs(drawdowns.min())