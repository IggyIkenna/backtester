from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import logging
import json

from utils.exceptions import MarketDataError, SignalError, OrderError, PositionError
from utils.types import MarketState, BacktestState

class BacktestErrorType(Enum):
    """Types of backtest errors"""
    MARKET_DATA = "market_data"          # Missing/invalid market data
    SIGNAL_GENERATION = "signal_generation"  # Strategy errors
    ORDER_EXECUTION = "order_execution"   # Execution issues
    POSITION_MANAGEMENT = "position_management"  # Position tracking errors
    STATE_VALIDATION = "state_validation"  # Invalid state transitions
    CONFIGURATION = "configuration"       # Config validation errors
    SYSTEM = "system"                    # Unexpected system errors

@dataclass
class BacktestError:
    """Structured error information with state context"""
    error_type: BacktestErrorType
    message: str
    timestamp: pd.Timestamp
    details: Dict[str, Any]
    recoverable: bool
    recovery_action: Optional[str] = None
    market_state: Optional[MarketState] = None
    backtest_state: Optional[BacktestState] = None

class BacktestErrorHandler:
    """Handles error detection, logging, and recovery"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.errors: List[BacktestError] = []
    
    def handle_execution_error(self, error: Exception, market_state: MarketState, 
                             backtest_state: BacktestState) -> None:
        """Handle execution-time errors"""
        if isinstance(error, MarketDataError):
            self._handle_market_data_error(error, market_state, backtest_state)
        elif isinstance(error, SignalError):
            self._handle_signal_error(error, market_state, backtest_state)
        elif isinstance(error, OrderError):
            self._handle_order_error(error, market_state, backtest_state)
        elif isinstance(error, PositionError):
            self._handle_position_error(error, market_state, backtest_state)
        else:
            self._handle_system_error(error, market_state, backtest_state)

    def _handle_market_data_error(self, error: 'MarketDataError', 
                                market_state: MarketState, 
                                backtest_state: BacktestState) -> None:
        """Handle market data related errors"""
        error_info = BacktestError(
            error_type=BacktestErrorType.MARKET_DATA,
            message=str(error),
            timestamp=market_state.timestamp,
            details={'missing_fields': error.missing_fields},
            recoverable=True,
            recovery_action="Skip timestamp",
            market_state=market_state,
            backtest_state=backtest_state
        )
        self._log_error(error_info)
        self.errors.append(error_info)

    def _handle_signal_error(self, error: 'SignalError',
                           market_state: MarketState,
                           backtest_state: BacktestState) -> None:
        """Handle strategy signal generation errors"""
        error_info = BacktestError(
            error_type=BacktestErrorType.SIGNAL_GENERATION,
            message=str(error),
            timestamp=market_state.timestamp,
            details={'signal_id': error.signal_id},
            recoverable=True,
            recovery_action="Skip signal",
            market_state=market_state,
            backtest_state=backtest_state
        )
        self._log_error(error_info)
        self.errors.append(error_info)

    def _handle_order_error(self, error: 'OrderError',
                          market_state: MarketState,
                          backtest_state: BacktestState) -> None:
        """Handle order execution errors"""
        error_info = BacktestError(
            error_type=BacktestErrorType.ORDER_EXECUTION,
            message=str(error),
            timestamp=market_state.timestamp,
            details={'order_id': error.order_id},
            recoverable=True,
            recovery_action="Cancel order",
            market_state=market_state,
            backtest_state=backtest_state
        )
        self._log_error(error_info)
        self.errors.append(error_info)

    def _handle_position_error(self, error: 'PositionError',
                             market_state: MarketState,
                             backtest_state: BacktestState) -> None:
        """Handle position management errors"""
        error_info = BacktestError(
            error_type=BacktestErrorType.POSITION_MANAGEMENT,
            message=str(error),
            timestamp=market_state.timestamp,
            details={'position_size': backtest_state.position.size},
            recoverable=False,  # Position errors are serious
            market_state=market_state,
            backtest_state=backtest_state
        )
        self._log_error(error_info)
        self.errors.append(error_info)

    def _handle_system_error(self, error: Exception,
                           market_state: MarketState,
                           backtest_state: BacktestState) -> None:
        """Handle unexpected system errors"""
        error_info = BacktestError(
            error_type=BacktestErrorType.SYSTEM,
            message=str(error),
            timestamp=market_state.timestamp,
            details={'error_type': type(error).__name__},
            recoverable=False,
            market_state=market_state,
            backtest_state=backtest_state
        )
        self._log_error(error_info)
        self.errors.append(error_info)

    def _log_error(self, error: BacktestError) -> None:
        """Log error with context"""
        log_data = {
            'timestamp': error.timestamp.isoformat(),
            'type': error.error_type.value,
            'message': error.message,
            'recoverable': error.recoverable,
            'details': error.details,
            'state': {
                'position_size': error.backtest_state.position.size if error.backtest_state else None,
                'equity': error.backtest_state.position.equity if error.backtest_state else None,
                'market_price': error.market_state.close if error.market_state else None
            }
        }
        self.logger.error(f"BACKTEST_ERROR: {json.dumps(log_data, indent=2)}")

