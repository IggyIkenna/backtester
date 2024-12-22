"""
State management classes for backtesting system.
Provides dataclasses for tracking market, position, and order states.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Union, Any
import pandas as pd
import uuid


from configs.backtester_config import ExecutionConfig

class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MAKER = "maker"
    TAKER = "taker"

class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    STOP = "stop"
    RISK = "risk"

class TradingState(Enum):
    ENABLED = "enabled"
    DISABLED_TIME = "disabled_time"
    DISABLED_HOURS = "disabled_hours"
    DISABLED_DAYS = "disabled_days"
    DISABLED_OVERNIGHT = "disabled_overnight"
    DISABLED_RANGE = "disabled_range"


@dataclass
class MarketState:
    """Current market state including price and features"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    book_price: float  # Previous candle close
    features: Dict[str, float]
    prev_high: Optional[float] = None  # Previous candle high
    prev_low: Optional[float] = None   # Previous candle low
    prev_close: Optional[float] = None # Previous candle close
    prev_open: Optional[float] = None # Previous candle open
    
    def calculate_book_prices(self, config: ExecutionConfig) -> Dict[str, float]:
        """Calculate simulated book prices with spread adjustments"""
        # Base spread for current price
        base_spread = config.simulated_spread_bps / 10000
        
        # Calculate current volatility for matching/fills
        current_volatility = (self.high - self.low) / self.close
        spread_multiplier = 1 + (current_volatility * 10)  # Widen spread in volatile periods
        effective_spread = min(
            base_spread * spread_multiplier,
            config.max_spread_bps / 10000
        )
        
        # Calculate mid prices
        close_mid = self.close
        high_mid = self.high
        low_mid = self.low
        
        # Apply spreads
        half_spread = effective_spread / 2
        return {
            'close_ask': close_mid * (1 + half_spread),
            'close_bid': close_mid * (1 - half_spread),
            'high_ask': high_mid * (1 + half_spread),
            'high_bid': high_mid * (1 - half_spread),
            'low_ask': low_mid * (1 + half_spread),
            'low_bid': low_mid * (1 - half_spread),
            'spread_bps': effective_spread * 10000
        }
    
    def get_execution_volatility(self) -> float:
        """Calculate previous period volatility for execution decisions"""
        if self.prev_high is not None and self.prev_low is not None and self.prev_close is not None:
            return (self.prev_high - self.prev_low) / self.prev_close
        return 0.0
    
    def get_execution_stochastic_oscillator(self) -> float:
        """Calculate previous period stochastic oscillator for execution decisions"""
        if self.prev_high is not None and self.prev_low is not None and self.prev_close is not None:
            return (self.prev_close - self.prev_low) / (self.prev_high - self.prev_low)
        return 0.0
    
    def get_execution_momentum(self) -> float:
        """Calculate previous period momentum for execution decisions"""
        if self.prev_high is not None and self.prev_open is not None and self.prev_close is not None:
            return (self.prev_close - self.prev_open)
        return 0.0

@dataclass
class PositionState:
    """Current position and PnL state"""
    timestamp: pd.Timestamp
    size: float  # Current position size in coins
    entry_price: float  # Average entry price
    unrealized_pnl: float
    realized_pnl: float
    equity: float  # Current account equity
    leverage: float  # Current leverage
    
    # Detailed PnL tracking
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    maker_fees: float = 0.0
    taker_fees: float = 0.0
    
    def __post_init__(self):
        """Validate position state"""
        if self.entry_price < 0:
            raise ValueError(f"Invalid entry price: ${self.entry_price:.2f}")
        
        if self.equity <= 0:
            raise ValueError(f"Invalid equity: ${self.equity:.2f}")
        
        if self.leverage < 0:
            raise ValueError(f"Invalid leverage: {self.leverage:.2f}x")

@dataclass
class OrderState:
    """Current state of an order"""
    order_id: str
    signal_id: str
    status: OrderStatus
    side: Side
    size: float
    limit_price: float
    created_at: pd.Timestamp
    filled_at: Optional[pd.Timestamp] = None
    is_maker: bool = True
    fill_price: Optional[float] = None
    remaining_size: Optional[float] = None
    fees: Optional[float] = None
    
    def __post_init__(self):
        """Validate order state"""
        if self.size <= 0:
            raise ValueError(f"Invalid order size: {self.size}")
        
        if self.limit_price <= 0:
            raise ValueError(f"Invalid limit price: ${self.limit_price:.2f}")
        
        if self.filled_at and self.filled_at < self.created_at:
            raise ValueError("Fill time before creation time")


@dataclass
class RiskMetrics:
    """Current risk metrics state"""
    volatility: float
    var_95: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    leverage: float
    concentration: float
    overnight_exposure: float
    rolling_pnl: float
    rolling_vol: float



@dataclass
class BacktestState:
    """Complete state of the backtest at a point in time"""
    market: MarketState
    position: PositionState
    active_orders: List[OrderState]
    timestamp: pd.Timestamp
    high_water_mark: float
    drawdown: float
    trade_count: int
    missed_trades: int
    risk_metrics: Optional[RiskMetrics] = None  # Add risk metrics tracking
    trading_state: TradingState = TradingState.ENABLED
    trading_disabled_reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate backtest state"""
        if self.timestamp != self.market.timestamp:
            raise ValueError("Timestamp mismatch between market and backtest state")
        
        if self.high_water_mark < 0:
            raise ValueError(f"Invalid high water mark: ${self.high_water_mark:.2f}")
        
        if self.trade_count < 0:
            raise ValueError(f"Invalid trade count: {self.trade_count}") 



"""Common types and dataclasses used across the system"""






# Trading types

@dataclass
class PositionState:
    """Current position state"""
    timestamp: pd.Timestamp
    size: float
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    equity: float
    leverage: float
    long_pnl: float
    short_pnl: float
    maker_fees: float
    taker_fees: float




@dataclass
class Trade:
    """Executed trade details"""
    trade_id: str
    order_id: str
    signal_id: str
    timestamp: pd.Timestamp
    side: Side
    size: float
    price: float
    fees: float
    is_maker: bool
    trade_type: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketState:
    """Current market state"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    book_price: float
    features: Dict[str, float]
    prev_high: Optional[float] 
    prev_low: Optional[float] 
    prev_close: Optional[float] 
    prev_open: Optional[float] 

    def calculate_book_prices(self, config) -> Dict[str, float]:
        """Calculate simulated book prices"""
        spread = config.execution.simulated_spread_bps / 10000
        mid = self.close
        return {
            'mid': mid,
            'bid': mid * (1 - spread/2),
            'ask': mid * (1 + spread/2),
            'low_bid': self.low * (1 - spread/2),
            'high_ask': self.high * (1 + spread/2)
        }




@dataclass
class BacktestState:
    """Complete backtest state"""
    market: MarketState
    position: PositionState
    active_orders: List[OrderState]
    timestamp: pd.Timestamp
    high_water_mark: float
    drawdown: float
    trade_count: int
    missed_trades: int
    risk_metrics: Optional[RiskMetrics] = None
    
    # Add history tracking
    order_history: List[OrderState] = field(default_factory=list)
    trade_history: List[Trade] = field(default_factory=list)
    cancelled_orders: List[OrderState] = field(default_factory=list)
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to history"""
        self.trade_history.append(trade)
        self.trade_count += 1
    
    def add_order(self, order: OrderState) -> None:
        """Add order to history"""
        self.order_history.append(order)
        self.active_orders.append(order)
    
    def cancel_order(self, order: OrderState, reason: str) -> None:
        """Cancel order and move to history"""
        order.status = OrderStatus.CANCELLED
        order.cancel_reason = reason
        order.cancelled_at = self.timestamp
        self.active_orders.remove(order)
        self.cancelled_orders.append(order)
    
    def complete_order(self, order: OrderState) -> None:
        """Mark order as completed and remove from active"""
        order.status = OrderStatus.FILLED
        order.filled_at = self.timestamp
        self.active_orders.remove(order)

@dataclass
class Signal:
    """Trading signal with execution instructions"""
    signal_id: str
    timestamp: pd.Timestamp
    side: Side
    size: float
    limit_price: float
    order_type: OrderType
    signal_type: SignalType
    signal_params: Dict = None
    
    def __post_init__(self):
        if self.signal_params is None:
            self.signal_params = {}
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())

@dataclass
class OrderState:
    """Current state of an order"""
    order_id: str
    signal_id: str
    timestamp: pd.Timestamp
    side: Side
    size: float
    filled_size: float
    remaining_size: float
    limit_price: float
    order_type: OrderType
    status: OrderStatus
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    filled_at: Optional[pd.Timestamp] = None
    cancelled_at: Optional[pd.Timestamp] = None
    fill_price: Optional[float] = None
    fees: Optional[float] = None
    cancel_reason: Optional[str] = None
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        if self.updated_at is None:
            self.updated_at = self.created_at





