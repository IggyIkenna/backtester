from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
import pandas as pd
import logging
import numpy as np

from configs.backtester_config import BacktestConfig
from utils.types import Signal, Side, OrderType, SignalType, Trade, MarketState, BacktestState
from utils.exceptions import ValidationError, StrategyError
from utils.error_handler import BacktestErrorHandler


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Trading Time Configuration:
    -------------------------
    Trading time restrictions can be configured through several parameters:
    
    1. trading_days: List[str] or None
       - If None or empty list: Trading allowed on all days
       - If list: Only allow trading on specified days
       - Example: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    
    2. trading_hours: Tuple[int, int] or None
       - If None: Trading allowed at all hours
       - If tuple: Only allow trading between start_hour and end_hour
       - Example: (9, 16) for 9 AM to 4 PM
    
    3. allow_overnight: bool
       - If True: Allow positions to be held overnight
       - If False: Close positions at end of trading hours
    
    4. signal_start_day and signal_start_hour:
       - Controls when the first signal can be generated
       - Must align with trading_days if specified
       - Example: 'monday' and 0 for Monday at midnight
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)  # Will be 'strategies.base_strategy'
        self.error_handler = BacktestErrorHandler(config, self.logger)
        
        # Log trading time configuration
        self.logger.info(f"Initializing {self.__class__.__name__} strategy")
        self._log_trading_config()
        
        # Strategy state
        self.last_signal_time: pd.Timestamp = pd.Timestamp.min
        self.active_signals: Dict[str, Signal] = {}
        self.historical_trades: List[Trade] = []
    
    def _log_trading_config(self) -> None:
        """Log trading time configuration"""
        time_config = self.config.time
        
        # Log trading days configuration
        if time_config.trading_days is None or not time_config.trading_days:
            self.logger.info("Trading days: All days enabled")
        else:
            self.logger.info(f"Trading days: {', '.join(time_config.trading_days)}")
        
        # Log trading hours configuration
        if time_config.trading_hours:
            self.logger.info(
                f"Trading hours: {time_config.trading_hours[0]}:00 to "
                f"{time_config.trading_hours[1]}:00"
            )
        else:
            self.logger.info("Trading hours: 24/7 enabled")
        
        # Log signal initialization settings
        self.logger.info(
            f"Signal initialization: {time_config.signal_start_day.capitalize()} "
            f"at {time_config.signal_start_hour}:00"
        )
        
        # Log overnight trading
        self.logger.info(f"Overnight trading: {'Enabled' if time_config.allow_overnight else 'Disabled'}")
        
        # Log disabled periods
        if time_config.trading_disabled_datetime:
            self.logger.info("Disabled date ranges:")
            for start, end in time_config.trading_disabled_datetime:
                self.logger.info(f"  - {start} to {end}")
        
        if time_config.trading_disabled_hours:
            self.logger.info("Disabled hours:")
            for start, end in time_config.trading_disabled_hours:
                self.logger.info(f"  - {start}:00 to {end}:00")
    
    @abstractmethod
    def generate_signals(self, market_state: MarketState, backtest_state: BacktestState) -> List[Signal]:
        """Generate trading signals based on current market state"""
        try:
            if not self.should_generate_signals(market_state.timestamp):
                return []

            signals = self._generate_signals_impl(market_state, backtest_state)
            
            # Validate signals
            for signal in signals:
                try:
                    if not self._validate_signal(signal, market_state):
                        raise ValidationError(
                            field="signal",
                            value=signal.signal_id,
                            constraint="signal_validation"
                        )
                except ValidationError as e:
                    self.error_handler.handle_execution_error(e, market_state, backtest_state)
                    signals.remove(signal)

            return signals

        except Exception as e:
            self.error_handler.handle_execution_error(
                StrategyError(
                    strategy_name=self.__class__.__name__,
                    operation="generate_signals",
                    reason=str(e),
                    state_details={"timestamp": market_state.timestamp}
                ),
                market_state,
                backtest_state
            )
            if not self.config.continue_on_error:
                raise
            return []
    
    def calculate_target_position(self, market_state: MarketState, backtest_state: BacktestState) -> float:
        """Calculate desired position size"""
        return 0.0  # Default to flat
    
    def should_generate_signals(self, timestamp: pd.Timestamp) -> bool:
        """Check if it's time to generate new signals based on time restrictions"""
        
        # Special handling for first run
        if self.last_signal_time == pd.Timestamp.min:
            # Check if we're on the right day
            current_day = timestamp.strftime('%A').lower()
            if current_day != self.config.time.signal_start_day.lower():
                return False
            
            # Check if we're at the right hour
            if timestamp.hour != self.config.time.signal_start_hour:
                return False
            
            # Check if this is a valid trading day
            if self.config.time.trading_days:
                if current_day not in self.config.time.trading_days:
                    self.logger.warning(
                        f"Signal start day {current_day} is not in trading days "
                        f"{self.config.time.trading_days}"
                    )
                    return False
            
            # Check other initial conditions
            if not self._check_trading_conditions(timestamp):
                return False
            
            # If all initialization checks pass, we can proceed
            self.logger.info(f"First signal period initialized at {timestamp}")
            return True
        
        # For subsequent signals, check all conditions
        return self._check_trading_conditions(timestamp)
    
    def _check_trading_conditions(self, timestamp: pd.Timestamp) -> bool:
        """Check all trading conditions"""
        # 1. Check signal period alignment
        epoch_start = pd.Timestamp('1996-01-01 00:00:00')
        seconds_since_epoch = (timestamp - epoch_start).total_seconds()
        signal_seconds = pd.Timedelta(self.config.time.signal_period).total_seconds()
        
        if seconds_since_epoch % signal_seconds != 0:
            return False
        
        # 2. Check trading hours
        current_hour = timestamp.hour
        if self.config.time.trading_hours:
            start_hour, end_hour = self.config.time.trading_hours
            if not (start_hour <= current_hour < end_hour):
                if not self.config.time.allow_overnight:
                    return False
        
        # 3. Check trading days
        if self.config.time.trading_days:
            current_day = timestamp.strftime('%A').lower()
            if current_day not in self.config.time.trading_days:
                return False
        
        # 4. Check disabled datetime ranges
        if self.config.time.trading_disabled_datetime:
            for start_str, end_str in self.config.time.trading_disabled_datetime:
                start = pd.Timestamp(start_str)
                end = pd.Timestamp(end_str)
                if start <= timestamp <= end:
                    return False
        
        # 5. Check disabled hours
        if self.config.time.trading_disabled_hours:
            for start_hour, end_hour in self.config.time.trading_disabled_hours:
                # Handle ranges that cross midnight
                if start_hour > end_hour:
                    if current_hour >= start_hour or current_hour < end_hour:
                        return False
                else:
                    if start_hour <= current_hour < end_hour:
                        return False
        
        # 6. Check minimum time between signals
        if self.last_signal_time != pd.Timestamp.min:  # Skip for first signal
            time_since_last = (timestamp - self.last_signal_time).total_seconds()
            min_interval = pd.Timedelta(self.config.time.signal_period).total_seconds()
            if time_since_last < min_interval:
                return False
        
        return True
    
    def _create_signal(self, 
                      market_state: MarketState,
                      side: Side,
                      size: float,
                      signal_type: SignalType,
                      edge_bps: float,
                      order_type: OrderType = OrderType.MAKER,
                      params: Dict = None) -> Optional[Signal]:
        """Create a properly formatted signal with validation"""
        
        # Validate size
        position_value = abs(size * market_state.close)
        if position_value < self.config.execution.min_order_size_usd:
            self.logger.debug(
                f"Signal size ${position_value:.2f} below minimum "
                f"${self.config.execution.min_order_size_usd:.2f}"
            )
            return None
        
        # Calculate limit price with edge
        book_prices = market_state.calculate_book_prices(self.config.execution)
        if side == Side.BUY:
            # Place buy order below market
            limit_price = book_prices['close_bid'] * (1 - edge_bps/10000)
        else:
            # Place sell order above market
            limit_price = book_prices['close_ask'] * (1 + edge_bps/10000)
        
        # Create signal
        signal = Signal(
            signal_id="",  # Will be set in post_init
            timestamp=market_state.timestamp,
            side=side,
            size=size,
            limit_price=limit_price,
            order_type=order_type,
            signal_type=signal_type,
            signal_params=params or {}
        )
        
        self._log_signal(signal, market_state)
        return signal
    
    def _calculate_position_size(self, 
                               market_state: MarketState,
                               backtest_state: BacktestState,
                               target_notional: float) -> float:
        """Calculate position size respecting limits"""
        
        # Calculate maximum allowed position
        max_position_usd = min(
            self.config.execution.max_position_size_usd,
            backtest_state.position.equity * self.config.strategy.leverage
        )
        
        # Consider existing position
        current_position_value = abs(backtest_state.position.size * market_state.close)
        available_position = max_position_usd - current_position_value
        
        # Apply concentration limit
        max_concentration = backtest_state.position.equity * self.config.execution.concentration_limit
        available_position = min(available_position, max_concentration)
        
        # Calculate size in coins
        size = min(target_notional, available_position) / market_state.close
        
        # Apply coin-based limits
        size = min(size, self.config.execution.max_position_coins)
        
        return size
    
    def _check_rebalance_needs(self,
                              market_state: MarketState,
                              backtest_state: BacktestState) -> List[Signal]:
        """Check if position needs rebalancing"""
        signals = []
        
        # Get target position
        target_position = self.calculate_target_position(market_state, backtest_state)
        current_position = backtest_state.position.size
        
        # Check if difference exceeds threshold
        position_value = abs(current_position * market_state.close)
        min_change = max(
            self.config.execution.min_position_change_usd,
            position_value * self.config.strategy.rebalance_threshold
        )
        
        position_diff = target_position - current_position
        if abs(position_diff * market_state.close) > min_change:
            # Create rebalance signal
            signal = self._create_signal(
                market_state=market_state,
                side=Side.BUY if position_diff > 0 else Side.SELL,
                size=abs(position_diff),
                signal_type=SignalType.REBALANCE,
                edge_bps=self.config.strategy.exit_bps,
                order_type=OrderType.TAKER,
                params={'target_position': target_position}
            )
            if signal:
                signals.append(signal)
        
        return signals
    
    def _log_signal(self, signal: Signal, market_state: MarketState) -> None:
        """Log signal generation with context"""
        signal_value = abs(signal.size * market_state.close)
        
        log_data = {
            'timestamp': signal.timestamp.isoformat(),
            'signal_id': signal.signal_id,
            'type': signal.signal_type.value,
            'side': signal.side.value,
            'size': signal.size,
            'value_usd': signal_value,
            'limit_price': signal.limit_price,
            'market_price': market_state.close,
            'edge_bps': ((signal.limit_price / market_state.close - 1) * 10000 
                        if signal.side == Side.BUY else
                        (market_state.close / signal.limit_price - 1) * 10000),
            'params': signal.signal_params
        }
        
        self.logger.info(f"SIGNAL_GENERATED: {log_data}")
    
    def _generate_entry_signals(self,
                              macd_state: Dict,
                              market_state: MarketState,
                              backtest_state: BacktestState) -> List[Signal]:
        """Generate entry signals"""
        signals = []
        
        # Determine direction and size
        side = Side.BUY if macd_state['hist'] > 0 else Side.SELL
        size = abs(self.calculate_target_position(market_state, backtest_state))  # No volatility scaling
        
        if size > 0:
            signal = self._create_signal(
                market_state=market_state,
                side=side,
                size=size,
                signal_type=SignalType.ENTRY,
                edge_bps=self.config.strategy.entry_bps,
                params={
                    'macd': macd_state['macd'],
                    'signal': macd_state['signal'],
                    'hist': macd_state['hist']
                }
            )
            if signal:
                signals.append(signal)
        
        return signals