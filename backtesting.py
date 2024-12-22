from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging
from datetime import datetime

from backtester.utils.types import (
    MarketState, BacktestState, PositionState, OrderState,
    Trade, Signal, Side, OrderType, OrderStatus
)
from backtester.execution_engine import ExecutionEngine
from backtester.position_manager import PositionManager
from backtester.strategies.base_strategy import Strategy
from backtester.configs.backtester_config import BacktestConfig
from backtester.risk_manager import RiskManager
from backtester.output_handler import OutputHandler
from backtester.utils.error_handler import BacktestErrorHandler

class Backtester:
    """Main backtesting engine with state management"""
    
    def __init__(self,
                 strategy: Strategy,
                 execution_engine: ExecutionEngine,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 output_handler: OutputHandler,
                 config: BacktestConfig):
        
        self.strategy = strategy
        self.execution_engine = execution_engine
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.output_handler = output_handler
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handler
        self.error_handler = BacktestErrorHandler(config, self.logger)
        
        # Initialize state
        self.state: Optional[BacktestState] = None
        
        self._log_backtest_config()
    
    def _log_backtest_config(self) -> None:
        """Log backtest configuration"""
        self.logger.info("Backtest Configuration:")
        self.logger.info(f"Symbol: {self.config.symbol}")
        self.logger.info(f"Period: {self.config.time.start_date} to {self.config.time.end_date}")
        self.logger.info(f"Candle Period: {self.config.time.candle_period}")
        self.logger.info(f"Signal Period: {self.config.time.signal_period}")
        self.logger.info("Data Sources:")
        self.logger.info(f"  Market Data: {self.config.data_dir}")
        self.logger.info(f"  Features: {self.config.features_dir}")
        self.logger.info(f"Feature Groups: {[fg.name for fg in self.config.features]}")
        self.logger.info(f"Error Handling: {'Continue' if self.config.continue_on_error else 'Stop'} on error")
    
    def run(self, market_df: pd.DataFrame) -> BacktestState:
        """Run backtest with state management"""
        try:
            self._initialize_state(market_df)
            
            for timestamp, row in market_df.iterrows():
                try:
                    self._process_timestamp(pd.Timestamp(timestamp), row, market_df)
                except Exception as e:
                    self.error_handler.handle_execution_error(e, self.state.market, self.state)
                    if not self.config.continue_on_error:
                        raise
            
            return self.state
            
        except Exception as e:
            self.logger.error(f"Fatal backtest error: {str(e)}")
            raise
    
    def _initialize_state(self, market_df: pd.DataFrame) -> None:
        """Initialize backtest state"""
        initial_timestamp = market_df.index[0]
        
        # Create initial market state
        market_state = MarketState(
            timestamp=initial_timestamp,
            open=market_df.loc[initial_timestamp, 'open'],
            high=market_df.loc[initial_timestamp, 'high'],
            low=market_df.loc[initial_timestamp, 'low'],
            close=market_df.loc[initial_timestamp, 'close'],
            volume=market_df.loc[initial_timestamp, 'volume'],
            quote_volume=market_df.loc[initial_timestamp, 'quote_volume'],
            book_price=market_df.loc[initial_timestamp, 'close'],
            features=self._extract_features(market_df, initial_timestamp),
            prev_high=np.nan,
            prev_low=np.nan,
            prev_close=np.nan,
            prev_open=np.nan
        )
        
        # Create initial position state
        position_state = PositionState(
            timestamp=initial_timestamp,
            size=0.0,
            entry_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            equity=self.config.strategy.position_size,
            leverage=0.0,
            long_pnl=0.0,
            short_pnl=0.0,
            maker_fees=0.0,
            taker_fees=0.0
        )
        
        # Initialize complete backtest state
        self.state = BacktestState(
            market=market_state,
            position=position_state,
            active_orders=[],
            timestamp=initial_timestamp,
            high_water_mark=self.config.strategy.position_size,
            drawdown=0.0,
            trade_count=0,
            missed_trades=0,
            risk_metrics=None  # Will be updated by risk manager
        )
    
    def _process_timestamp(self, timestamp: pd.Timestamp, row: pd.Series, 
                          market_df: pd.DataFrame) -> None:
        """Process a single timestamp"""
        # Get previous candle data
        prev_idx = market_df.index.get_loc(timestamp) - 1
        if prev_idx >= 0:
            prev_row = market_df.iloc[prev_idx]
            prev_high = prev_row['high']
            prev_low = prev_row['low']
            prev_close = prev_row['close']
            prev_open = prev_row['open']
        else:
            prev_high = prev_low = prev_close = prev_open = None
        
        # Update market state
        self.state.market = MarketState(
            timestamp=timestamp,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            quote_volume=row['quote_volume'],
            book_price=market_df.shift(1).loc[timestamp, 'close'],
            features=self._extract_features(market_df, timestamp),
            prev_high=prev_high,
            prev_low=prev_low,
            prev_close=prev_close,
            prev_open=prev_open
        )
        
        # Process signals if it's a signal period
        if self._is_signal_period(timestamp):
            signals = self.strategy.generate_signals(self.state.market, self.state)
            valid_signals = self.risk_manager.filter_signals(signals, self.state)
            for signal in valid_signals:
                self.execution_engine.process_signal(signal, self.state)
        
        # Process executions
        trades = self.execution_engine.process_candle(self.state)
        
        # Update positions and risk
        for trade in trades:
            self.state.position = self.position_manager.update_position(trade, self.state.market)
            self.output_handler.log_trade(trade)
            self.state.trade_count += 1
        
        # Update risk metrics
        self.state.risk_metrics = self.risk_manager.calculate_metrics(self.state)
        
        # Log state
        self.output_handler.log_state(self.state)
    
    def _is_signal_period(self, timestamp: pd.Timestamp) -> bool:
        """Check if current timestamp is a signal period"""
        # TODO: signal data needs to resampled on a 4h basis
        signal_minutes = pd.Timedelta(self.config.time.signal_period).total_seconds() / 60
        candle_minutes = pd.Timedelta(self.config.time.candle_period).total_seconds() / 60
        if signal_minutes % candle_minutes != 0:
            raise ValueError("Signal period must be a multiple of candle period")
        return timestamp.minute % signal_minutes == 0
    
    def _extract_features(self, df: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Extract features from market data"""
        feature_cols = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume'
        ]]
        return {col: df.loc[timestamp, col] for col in feature_cols}