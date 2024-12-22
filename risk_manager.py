"""Risk management system for position and trade controls"""
from typing import List
import logging
import numpy as np
import pandas as pd

from backtester.utils.types import (
    RiskMetrics, BacktestState, Signal,
    Side, OrderType, SignalType, TradingState
)
from backtester.configs.backtester_config import BacktestConfig
from backtester.utils.exceptions import ValidationError, StateError
from backtester.utils.error_handler import BacktestErrorHandler


class RiskManager:
    """Manages risk limits and generates risk signals"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_handler = BacktestErrorHandler(config, self.logger)
        self._log_risk_config()
        
        # Risk state
        self.position_history: List[float] = []
        self.pnl_history: List[float] = []
        self.returns_history: List[float] = []
        
    def _log_risk_config(self) -> None:
        """Log risk management configuration"""
        self.logger.info("Risk Manager Configuration:")
        self.logger.info(f"Position Limits:")
        self.logger.info(f"  Max Leverage: {self.config.execution.max_leverage:.1f}x")
        self.logger.info(f"  Max Drawdown: {self.config.execution.max_drawdown:.1%}")
        self.logger.info(f"  Concentration: {self.config.execution.concentration_limit:.1%}")
        self.logger.info("Trading Restrictions:")
        if self.config.time.trading_hours:
            self.logger.info(f"  Hours: {self.config.time.trading_hours[0]}:00-{self.config.time.trading_hours[1]}:00")
        if self.config.time.trading_days:
            self.logger.info(f"  Days: {', '.join(self.config.time.trading_days)}")
        self.logger.info(f"  Overnight: {'Allowed' if self.config.time.allow_overnight else 'Not Allowed'}")
    
    def filter_signals(self, signals: List[Signal], state: BacktestState) -> List[Signal]:
        """Filter signals through risk checks"""
        try:
            valid_signals = []
            
            for signal in signals:
                try:
                    # Validate signal size
                    if not self._validate_signal_size(signal, state):
                        raise ValidationError(
                            field="signal_size",
                            value=signal.size,
                            constraint="position_limits"
                        )
                    
                    # Check position limits
                    if not self._check_position_limits(signal, state):
                        raise StateError(
                            state_type="position",
                            current_state=state.position.size,
                            attempted_transition=signal.size,
                            reason="Position limits exceeded"
                        )
                    
                    # Check trading restrictions
                    if not self._check_trading_restrictions(signal, state):
                        continue
                    
                    valid_signals.append(signal)
                    
                except (ValidationError, StateError) as e:
                    self.error_handler.handle_execution_error(e, state.market, state)
                    continue
            
            # Add risk management signals
            try:
                risk_signals = self._generate_risk_signals(state)
                valid_signals.extend(risk_signals)
            except Exception as e:
                self.error_handler.handle_execution_error(e, state.market, state)
            
            return valid_signals
            
        except Exception as e:
            self.error_handler.handle_execution_error(e, state.market, state)
            if not self.config.continue_on_error:
                raise
            return []
    
    def calculate_metrics(self, state: BacktestState) -> RiskMetrics:
        """Calculate current risk metrics"""
        try:
            # Update histories
            self.position_history.append(state.position.size)
            self.pnl_history.append(state.position.unrealized_pnl + state.position.realized_pnl)
            
            if len(self.pnl_history) > 1:
                returns = (self.pnl_history[-1] - self.pnl_history[-2]) / state.position.equity
                self.returns_history.append(returns)
            
            # Calculate metrics
            return RiskMetrics(
                volatility=self._calculate_volatility(),
                var_95=self._calculate_var_95(),
                expected_shortfall=self._calculate_expected_shortfall(),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                max_drawdown=self._calculate_max_drawdown(),
                leverage=state.position.leverage,
                concentration=abs(state.position.size * state.market.close) / state.position.equity,
                overnight_exposure=self._calculate_overnight_exposure(state),
                rolling_pnl=sum(self.pnl_history[-20:]) if len(self.pnl_history) >= 20 else 0,
                rolling_vol=np.std(self.returns_history[-20:]) * np.sqrt(252) if len(self.returns_history) >= 20 else 0
            )
            
        except Exception as e:
            self.error_handler.handle_execution_error(e, state.market, state)
            if not self.config.continue_on_error:
                raise
            return RiskMetrics(...)  # Return safe default metrics
    
    def _validate_signal_size(self, signal: Signal, state: BacktestState) -> bool:
        """Validate signal size against limits"""
        position_value = abs(signal.size * state.market.close)
        
        if position_value < self.config.execution.min_order_size_usd:
            return False
            
        if position_value > self.config.execution.max_order_size_usd:
            return False
            
        return True
    
    def _check_position_limits(self, signal: Signal, state: BacktestState) -> bool:
        """Check if signal would violate position limits"""
        new_size = state.position.size + signal.size
        position_value = abs(new_size * state.market.close)
        
        # Size limits
        if position_value > self.config.execution.max_position_size_usd:
            return False
            
        # Leverage limits
        new_leverage = position_value / state.position.equity
        if new_leverage > self.config.execution.max_leverage:
            return False
            
        # Concentration limits
        concentration = position_value / state.position.equity
        if concentration > self.config.execution.concentration_limit:
            return False
            
        return True
    
    def _check_trading_restrictions(self, signal: Signal, state: BacktestState) -> bool:
        """Check if trading is allowed"""
        return state.trading_state == TradingState.ENABLED
    
    def _generate_risk_signals(self, state: BacktestState) -> List[Signal]:
        """Generate risk management signals"""
        signals = []
        
        # Check stop loss
        if self._check_stop_loss(state):
            signals.append(self._create_stop_signal(state))
            
        # Check position reduction
        if self._check_position_reduction(state):
            signals.append(self._create_reduction_signal(state))
            
        return signals
    
    def _check_stop_loss(self, state: BacktestState) -> bool:
        """Check if position should be stopped out"""
        if state.position.size == 0:
            return False
            
        drawdown = state.drawdown / state.high_water_mark
        return abs(drawdown) > self.config.execution.max_drawdown
    
    def _check_position_reduction(self, state: BacktestState) -> bool:
        """Check if position size should be reduced"""
        if state.position.size == 0:
            return False
            
        return state.position.leverage > self.config.execution.max_leverage
    
    def _create_stop_signal(self, state: BacktestState) -> Signal:
        """Create signal to close position"""
        return Signal(
            signal_id="",  # Will be set in post_init
            timestamp=state.market.timestamp,
            side=Side.SELL if state.position.size > 0 else Side.BUY,
            size=-state.position.size,
            limit_price=state.market.close,
            order_type=OrderType.TAKER,
            signal_type=SignalType.STOP
        )
    
    def _create_reduction_signal(self, state: BacktestState) -> Signal:
        """Create signal to reduce position size"""
        target_size = (
            state.position.equity * self.config.execution.max_leverage / 
            state.market.close
        )
        reduction_size = target_size - abs(state.position.size)
        
        return Signal(
            signal_id="",  # Will be set in post_init
            timestamp=state.market.timestamp,
            side=Side.SELL if state.position.size > 0 else Side.BUY,
            size=reduction_size,
            limit_price=state.market.close,
            order_type=OrderType.TAKER,
            signal_type=SignalType.RISK
        )
    
    # Risk metric calculation helpers
    def _calculate_volatility(self) -> float:
        if len(self.returns_history) < 20:
            return 0.0
        return np.std(self.returns_history[-20:]) * np.sqrt(252)
    
    def _calculate_var_95(self) -> float:
        if len(self.returns_history) < 100:
            return 0.0
        return np.percentile(self.returns_history[-100:], 5)
    
    def _calculate_expected_shortfall(self) -> float:
        if len(self.returns_history) < 100:
            return 0.0
        var_95 = self._calculate_var_95()
        return np.mean([r for r in self.returns_history[-100:] if r <= var_95])
    
    def _calculate_sharpe_ratio(self) -> float:
        if len(self.returns_history) < 252:
            return 0.0
        returns = np.array(self.returns_history[-252:])
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        cumulative = np.maximum.accumulate(self.pnl_history)
        drawdowns = (cumulative - self.pnl_history) / cumulative
        return np.max(drawdowns)
    
    def _calculate_overnight_exposure(self, state: BacktestState) -> float:
        """Calculate overnight exposure"""
        if state.market.timestamp.hour < 9 or state.market.timestamp.hour >= 16:
            return abs(state.position.size * state.market.close)
        return 0.0