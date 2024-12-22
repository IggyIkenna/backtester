"""Position and PnL management with state tracking"""
from typing import Tuple, List
import logging

from backtester.utils.types import (
    MarketState, PositionState, Trade, Side
)
from backtester.configs.backtester_config import BacktestConfig
from backtester.utils.error_handler import BacktestErrorHandler
from backtester.utils.exceptions import PositionError


class PositionManager:
    """Manages position state and PnL calculations"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_handler = BacktestErrorHandler(config, self.logger)
        self._log_position_config()
        
        # Position tracking
        self.trade_history: List[Trade] = []
        self.position_history: List[PositionState] = []
    
    def _log_position_config(self) -> None:
        """Log position management configuration"""
        self.logger.info("Position Manager Configuration:")
        self.logger.info(f"Initial Position Size: ${self.config.strategy.position_size:,.2f}")
        self.logger.info(f"Base Leverage: {self.config.strategy.leverage:.1f}x")
        self.logger.info(f"Position Limits:")
        self.logger.info(f"  Max Size: ${self.config.execution.max_position_size_usd:,.2f}")
        self.logger.info(f"  Max Leverage: {self.config.execution.max_leverage:.1f}x")
        self.logger.info(f"  Concentration: {self.config.execution.concentration_limit:.1%}")
        self.logger.info(f"  Max Drawdown: {self.config.execution.max_drawdown:.1%}")
    
    def update_position(self, trade: Trade, market_state: MarketState) -> PositionState:
        """Update position state based on new trade"""
        try:
            current_position = self.position_history[-1] if self.position_history else None
            
            # Validate trade
            if trade.size == 0:
                raise PositionError(
                    reason="Zero size trade",
                    position_details={'current_size': current_position.size if current_position else 0}
                )
            
            # Initialize position if none exists
            if not current_position:
                current_position = PositionState(
                    timestamp=market_state.timestamp,
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
            
            # Calculate position changes
            new_size = current_position.size + trade.size
            new_entry_price = self._calculate_new_entry_price(current_position, trade, new_size)
            
            # Validate leverage before proceeding
            new_value = abs(new_size * market_state.close)
            if new_value > 0:
                new_leverage = new_value / current_position.equity
                if new_leverage > self.config.execution.max_leverage:
                    raise PositionError(
                        reason="Leverage limit exceeded",
                        position_details={
                            'leverage': new_leverage,
                            'limit': self.config.execution.max_leverage
                        }
                    )
            
            # Calculate PnL components
            realized_pnl = self._calculate_realized_pnl(current_position, trade)
            unrealized_pnl = self._calculate_unrealized_pnl(new_size, new_entry_price, market_state)
            
            # Update directional PnL
            long_pnl, short_pnl = self._update_directional_pnl(
                current_position, trade, realized_pnl
            )
            
            # Update fees
            maker_fees = current_position.maker_fees
            taker_fees = current_position.taker_fees
            if trade.is_maker:
                maker_fees += trade.fees
            else:
                taker_fees += trade.fees
            
            # Calculate equity and leverage
            total_pnl = realized_pnl + unrealized_pnl
            total_fees = maker_fees + taker_fees
            equity = self.config.strategy.position_size + total_pnl - total_fees
            leverage = abs(new_size * market_state.close) / equity if equity > 0 else 0
            
            # Create new position state
            new_position = PositionState(
                timestamp=market_state.timestamp,
                size=new_size,
                entry_price=new_entry_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                equity=equity,
                leverage=leverage,
                long_pnl=long_pnl,
                short_pnl=short_pnl,
                maker_fees=maker_fees,
                taker_fees=taker_fees
            )
            
            # Update history
            self.position_history.append(new_position)
            self.trade_history.append(trade)
            
            # Log update
            self._log_position_update(new_position, trade)
            
            return new_position
            
        except Exception as e:
            self.error_handler.handle_execution_error(e, market_state, state)
            if not self.config.continue_on_error:
                raise
            return current_position  # Return unchanged position on error
    
    def _calculate_new_entry_price(self, position: PositionState, trade: Trade, new_size: float) -> float:
        """Calculate new entry price with position change tracking"""
        if new_size == 0:
            return 0.0
        elif position.size == 0:
            return trade.price
        else:
            # Check if adding to or reducing position
            is_adding = (position.size > 0 and trade.size > 0) or \
                       (position.size < 0 and trade.size < 0)
            
            if is_adding:
                # Weighted average for position increase
                position_value = abs(position.size * position.entry_price)
                trade_value = abs(trade.size * trade.price)
                return (position_value + trade_value) / abs(new_size)
            else:
                # Keep entry price when reducing
                return position.entry_price
    
    def _calculate_realized_pnl(self, position: PositionState, trade: Trade) -> float:
        """Calculate realized PnL from trade"""
        if position.size == 0:
            return 0.0
        
        # Determine closing size
        closing_size = min(abs(trade.size), abs(position.size))
        if position.size > 0:
            closing_size *= -1 if trade.side == Side.SELL else 0
        else:
            closing_size *= -1 if trade.side == Side.BUY else 0
        
        # Calculate PnL
        price_diff = trade.price - position.entry_price
        if position.size < 0:
            price_diff = -price_diff
        
        return closing_size * price_diff - trade.fees
    
    def _calculate_unrealized_pnl(self, size: float, entry_price: float, market: MarketState) -> float:
        """Calculate unrealized PnL"""
        if size == 0 or entry_price == 0:
            return 0.0
        
        price_diff = market.close - entry_price
        if size < 0:
            price_diff = -price_diff
        
        return size * price_diff
    
    def _update_directional_pnl(self, position: PositionState, trade: Trade, realized_pnl: float) -> Tuple[float, float]:
        """Update long and short PnL components"""
        long_pnl = position.long_pnl
        short_pnl = position.short_pnl
        
        if position.size > 0:
            long_pnl += realized_pnl
        else:
            short_pnl += realized_pnl
            
        return long_pnl, short_pnl
    
    def _log_position_update(self, position: PositionState, trade: Trade) -> None:
        """Log position state changes"""
        log_data = {
            'timestamp': position.timestamp.isoformat(),
            'position': {
                'size': position.size,
                'entry_price': position.entry_price,
                'leverage': position.leverage
            },
            'pnl': {
                'unrealized': position.unrealized_pnl,
                'realized': position.realized_pnl,
                'long': position.long_pnl,
                'short': position.short_pnl
            },
            'fees': {
                'maker': position.maker_fees,
                'taker': position.taker_fees,
                'total': position.maker_fees + position.taker_fees
            },
            'equity': {
                'total': position.equity,
                'initial': self.config.strategy.position_size,
                'pnl': position.equity - self.config.strategy.position_size
            },
            'trade': {
                'id': trade.trade_id,
                'size': trade.size,
                'price': trade.price,
                'is_maker': trade.is_maker
            }
        }
        self.logger.info(f"POSITION_UPDATE: {log_data}")