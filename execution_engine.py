"""Order execution and fill simulation against OHLCV data"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
import uuid

from backtester.utils.error_handler import BacktestErrorHandler
from backtester.utils.types import (
    MarketState, BacktestState, OrderState,
    Trade, Signal, Side, OrderType, OrderStatus, TradingState
)
from backtester.configs.backtester_config import BacktestConfig
from backtester.utils.exceptions import SignalError, OrderError


class ExecutionEngine:
    """Handles order execution and fill simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_handler = BacktestErrorHandler(config, self.logger)
        self._log_execution_config()
    
    def _log_execution_config(self) -> None:
        """Log execution configuration"""
        self.logger.info("Execution Engine Configuration:")
        self.logger.info(f"Fees: Maker {self.config.execution.maker_fee_bps} bps, "
                        f"Taker {self.config.execution.taker_fee_bps} bps")
        self.logger.info(f"Spreads: Base {self.config.execution.simulated_spread_bps} bps, "
                        f"Max {self.config.execution.max_spread_bps} bps")
        self.logger.info("Order Size Limits:")
        self.logger.info(f"  Min: ${self.config.execution.min_order_size_usd:,.2f}")
        self.logger.info(f"  Max: ${self.config.execution.max_order_size_usd:,.2f}")
        self.logger.info("Position Limits:")
        self.logger.info(f"  Max Size USD: ${self.config.execution.max_position_size_usd:,.2f}")
        self.logger.info(f"  Max Coins: {self.config.execution.max_position_coins:,.8f}")
        self.logger.info(f"  Max Leverage: {self.config.execution.max_leverage:.1f}x")
        self.logger.info(f"Order Behavior:")
        self.logger.info(f"  Price Update Threshold: {self.config.execution.price_update_threshold:.1%}")
        self.logger.info(f"  Taker Conversion: {self.config.execution.taker_conversion_hours} hours")
    
    def process_candle(self, state: BacktestState) -> List[Trade]:
        """Process order fills against current candle"""
        try:
            # First check and update trading state
            self._update_trading_state(state)
            
            # If trading is disabled, don't process orders
            if state.trading_state != TradingState.ENABLED:
                return []
            
            trades = []
            book_prices = state.market.calculate_book_prices(self.config)
            
            # Process each active order
            for order in list(state.active_orders):
                try:
                    # Check for taker conversion first
                    if self._should_convert_to_taker(order, state):
                        order.order_type = OrderType.TAKER
                        self.logger.info(
                            f"Order {order.order_id} converted to taker after "
                            f"{self.config.execution.taker_conversion_hours} hours"
                        )
                    
                    # Process based on order type
                    if order.order_type == OrderType.TAKER:
                        # Validate fill price
                        fill_price = state.market.close
                        if fill_price <= 0:
                            raise OrderError(
                                order_id=order.order_id,
                                reason=f"Invalid taker fill price: {fill_price}"
                            )
                        
                        # Create and process trade
                        trade = self._create_taker_trade(order, state)
                        trades.append(trade)
                        state.complete_order(order)
                        state.add_trade(trade)
                        continue
                    
                    # For maker orders, check limit fills
                    fill_price = self._check_limit_fill(order, state.market)
                    if fill_price:
                        # Validate fill price
                        if fill_price <= 0:
                            raise OrderError(
                                order_id=order.order_id,
                                reason=f"Invalid maker fill price: {fill_price}"
                            )
                        
                        # Create and process trade
                        trade = self._create_maker_trade(order, fill_price, state)
                        trades.append(trade)
                        state.complete_order(order)
                        state.add_trade(trade)
                        continue
                    
                    # Update limit price if needed
                    if order.order_type == OrderType.MAKER:
                        new_price = self._calculate_new_limit_price(order, state)
                        if self._should_update_limit_price(order, new_price):
                            order.limit_price = new_price
                            self.logger.info(
                                f"Updated order {order.order_id} price to {new_price:.2f} "
                                f"(change > {self.config.execution.price_update_threshold:.1%})"
                            )
                    
                except OrderError as e:
                    self.error_handler.handle_execution_error(e, state.market, state)
                    if not self.config.continue_on_error:
                        raise
                    continue  # Skip this order but continue processing others
            
            return trades
            
        except Exception as e:
            self.error_handler.handle_execution_error(e, state.market, state)
            if not self.config.continue_on_error:
                raise
            return []
    
    def process_signal(self, signal: Signal, state: BacktestState) -> None:
        """Process new signal and create order"""
        try:
            # Validate signal
            if not self._validate_signal(signal, state):
                raise SignalError(
                    signal_id=signal.signal_id,
                    reason="Signal validation failed"
                )
            
            # Create order
            order = OrderState(
                order_id="",  # Will be set in post_init
                signal_id=signal.signal_id,
                timestamp=state.market.timestamp,
                side=signal.side,
                size=signal.size,
                filled_size=0.0,
                remaining_size=signal.size,
                limit_price=signal.limit_price,
                order_type=signal.order_type,
                status=OrderStatus.ACTIVE,
                created_at=state.market.timestamp,
                updated_at=state.market.timestamp
            )
            
            # Validate order
            if order.size <= 0:
                raise OrderError(
                    order_id=order.order_id,
                    reason=f"Invalid order size: {order.size}"
                )
            
            # Add to state tracking
            state.add_order(order)
            self.logger.info(f"Created order {order.order_id} from signal {signal.signal_id}")
            
        except Exception as e:
            self.error_handler.handle_execution_error(e, state.market, state)
            if not self.config.continue_on_error:
                raise
    
    def _is_instant_fill(self, order: OrderState, book_prices: Dict[str, float]) -> bool:
        """Check if order should fill instantly as taker"""
        if order.side == Side.BUY:
            return order.limit_price >= book_prices['ask']
        else:
            return order.limit_price <= book_prices['bid']
    
    def _check_limit_fill(self, order: OrderState, market: MarketState) -> Optional[float]:
        """Check if limit order is filled during candle with spread-adjusted prices"""
        book_prices = market.calculate_book_prices(self.config)
        
        if order.side == Side.BUY:
            # For buys, we need to cross the ask
            if book_prices['low_ask'] <= order.limit_price <= book_prices['high_ask']:
                # Fill at limit price if it crosses the ask
                return min(order.limit_price, book_prices['high_ask'])
        else:  # SELL
            # For sells, we need to cross the bid
            if book_prices['low_bid'] <= order.limit_price <= book_prices['high_bid']:
                # Fill at limit price if it crosses the bid
                return max(order.limit_price, book_prices['low_bid'])
        return None
    
    def _should_convert_to_taker(self, order: OrderState, state: BacktestState) -> bool:
        """Check if maker order should convert to taker with dynamic conditions"""
        if order.order_type == OrderType.MAKER:
            conditions = []
            
            # Time threshold condition
            time_threshold = pd.Timedelta(hours=self.config.execution.taker_conversion_hours)
            conditions.append(state.market.timestamp - order.created_at > time_threshold)
            
            # Volatility condition if dynamic spread enabled
            if self.config.execution.use_dynamic_spread:
                execution_vol = state.market.get_execution_volatility()
                conditions.append(execution_vol > 0.02)
            
            # Momentum condition if enabled
            if self.config.execution.use_momentum:
                momentum = state.market.get_execution_momentum()
            if order.side == Side.BUY:
                    conditions.append(momentum > 0.01)  # Strong upward momentum
            else:
                    conditions.append(momentum < -0.01)  # Strong downward momentum
            
            # Stochastic condition if enabled
            if self.config.execution.use_stochastic_oscillator:
                stoch = state.market.get_execution_stochastic_oscillator()
                if order.side == Side.BUY:
                    conditions.append(stoch < 0.1)  # Deeply oversold
                else:
                    conditions.append(stoch > 0.9)  # Deeply overbought
            
            # Convert if any condition is met
            return any(conditions)
        
        return False
    
    def _create_taker_trade(self, order: OrderState, state: BacktestState) -> Trade:
        """Create trade for taker fill"""
        fill_price = state.market.close  # Use close price for taker fills
        
        return Trade(
            trade_id="",  # Will be set in post_init
            order_id=order.order_id,
            signal_id=order.signal_id,
            timestamp=state.market.timestamp,
            side=order.side,
            size=order.size,
            price=fill_price,
            fees=abs(order.size * fill_price) * self.config.execution.taker_fee_bps / 10000,
            is_maker=False
        )
    
    def _create_maker_trade(self, order: OrderState, fill_price: float, state: BacktestState) -> Trade:
        """Create trade for maker fill"""
        return Trade(
            trade_id="",  # Will be set in post_init
            order_id=order.order_id,
            signal_id=order.signal_id,
            timestamp=state.market.timestamp,
            side=order.side,
            size=order.size,
            price=fill_price,
            fees=abs(order.size * fill_price) * self.config.execution.maker_fee_bps / 10000,
            is_maker=True
        )
    
    def _validate_signal(self, signal: Signal, state: BacktestState) -> bool:
        """Validate signal parameters"""
        # Check minimum size
        if abs(signal.size * state.market.close) < self.config.execution.min_order_size_usd:
            self.logger.info(f"Signal size below minimum: ${abs(signal.size * state.market.close):,.2f}")
            return False
            
        # Check maximum size
        if abs(signal.size * state.market.close) > self.config.execution.max_order_size_usd:
            self.logger.warning(f"Signal size above maximum: ${abs(signal.size * state.market.close):,.2f}")
            return False
            
        # Validate limit price
        spread = self.config.execution.max_spread_bps / 10000
        if signal.side == Side.BUY:
            max_limit = state.market.close * (1 + spread)
            if signal.limit_price > max_limit:
                self.logger.warning(f"Buy limit price {signal.limit_price} above maximum spread")
                return False
        else:
            min_limit = state.market.close * (1 - spread)
            if signal.limit_price < min_limit:
                self.logger.warning(f"Sell limit price {signal.limit_price} below maximum spread")
                return False
                
        return True
    
    def _should_update_limit_price(self, order: OrderState, new_price: float) -> bool:
        """Check if price change justifies order update"""
        price_change = abs(new_price - order.limit_price) / order.limit_price
        return price_change > self.config.execution.price_update_threshold
    
    def _calculate_new_limit_price(self, order: OrderState, state: BacktestState) -> float:
        """Calculate new limit price for order update with dynamic adjustments"""
        book_prices = state.market.calculate_book_prices(self.config)
        total_edge = self.config.execution.simulated_spread_bps/10000
        
        # Apply dynamic spread adjustments if enabled
        if self.config.execution.use_dynamic_spread:
            execution_vol = state.market.get_execution_volatility()
            vol_adjustment = min(execution_vol * 5, self.config.execution.max_spread_bps/10000)
            total_edge += vol_adjustment
        
        # Apply stochastic oscillator adjustment if enabled
        if self.config.execution.use_stochastic_oscillator:
            stoch = state.market.get_execution_stochastic_oscillator()
            if order.side == Side.BUY:
                # More aggressive when oversold (stoch < 0.2)
                stoch_adjustment = max(0.2 - stoch, 0) * 0.001
            else:
                # More aggressive when overbought (stoch > 0.8)
                stoch_adjustment = max(stoch - 0.8, 0) * 0.001
            total_edge += stoch_adjustment
        
        # Apply momentum adjustment if enabled
        if self.config.execution.use_momentum:
            momentum = state.market.get_execution_momentum()
            momentum_sign = 1 if momentum > 0 else -1
            momentum_adjustment = min(abs(momentum) * 0.0001, 0.0002) * momentum_sign
            if order.side == Side.BUY:
                total_edge -= momentum_adjustment  # More aggressive on positive momentum
            else:
                total_edge += momentum_adjustment  # More aggressive on negative momentum
        
        # Apply final edge to price
        if order.side == Side.BUY:
            return book_prices['close_bid'] * (1 - total_edge)
        else:
            return book_prices['close_ask'] * (1 + total_edge)
    
    def _update_trading_state(self, state: BacktestState) -> None:
        """Update trading state based on current conditions"""
        timestamp = state.market.timestamp
        trading_state = TradingState.ENABLED
        reason = None
        
        # 1. Check trading days
        if self.config.time.trading_days:
            current_day = timestamp.strftime('%A').lower()
            if current_day not in self.config.time.trading_days:
                trading_state = TradingState.DISABLED_DAYS
                reason = f"Trading not allowed on {current_day}"
        
        # 2. Check trading hours
        current_hour = timestamp.hour
        if self.config.time.trading_hours:
            start_hour, end_hour = self.config.time.trading_hours
            if not (start_hour <= current_hour < end_hour):
                if not self.config.time.allow_overnight:
                    trading_state = TradingState.DISABLED_HOURS
                    reason = f"Outside trading hours ({start_hour}:00-{end_hour}:00)"
        
        # 3. Check disabled datetime ranges
        if self.config.time.trading_disabled_datetime:
            for start_str, end_str in self.config.time.trading_disabled_datetime:
                start = pd.Timestamp(start_str)
                end = pd.Timestamp(end_str)
                if start <= timestamp <= end:
                    trading_state = TradingState.DISABLED_RANGE
                    reason = f"Trading disabled during range {start} to {end}"
        
        # 4. Check disabled hours
        if self.config.time.trading_disabled_hours:
            for start_hour, end_hour in self.config.time.trading_disabled_hours:
                if start_hour > end_hour:  # Overnight range
                    if current_hour >= start_hour or current_hour < end_hour:
                        trading_state = TradingState.DISABLED_HOURS
                        reason = f"Trading disabled at hour {current_hour}"
                else:
                    if start_hour <= current_hour < end_hour:
                        trading_state = TradingState.DISABLED_HOURS
                        reason = f"Trading disabled at hour {current_hour}"
        
        # 5. Check overnight restrictions
        if not self.config.time.allow_overnight:
            if current_hour >= 16:  # After market hours
                trading_state = TradingState.DISABLED_OVERNIGHT
                reason = "Overnight trading not allowed"
        
        # Update state
        state.trading_state = trading_state
        state.trading_disabled_reason = reason
        
        if trading_state != TradingState.ENABLED:
            self.logger.info(f"Trading disabled: {reason}")