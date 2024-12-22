from typing import List, Dict, Optional
import numpy as np

from configs.backtester_config import BacktestConfig
from strategies.base_strategy import Strategy
from utils.exceptions import StrategyError
from utils.types import MarketState, BacktestState, Signal, Side, OrderType, SignalType


class MACDStrategy(Strategy):
    """
    MACD crossover strategy with volatility-adjusted positioning
    """
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.last_macd_state: Optional[Dict] = None
    
    def generate_signals(self, market_state: MarketState, backtest_state: BacktestState) -> List[Signal]:
        """Generate signals based on MACD and risk state"""
        signals = []
        
        # Check if we should generate signals
        if not self.should_generate_signals(market_state.timestamp):
            return signals
        
        # Check if we're in a risk-off state
        if backtest_state.risk_metrics and backtest_state.risk_metrics.volatility > self.config.execution.max_volatility:
            # Generate risk reduction signals if needed
            if backtest_state.position.size != 0:
                signals.append(self._create_exit_signal(market_state, backtest_state))
            return signals
        
        # Get current MACD state
        macd_state = self._get_macd_state(market_state.features)
        if not macd_state:
            return signals
            
        # Check for entry opportunities
        if self._check_entry_conditions(macd_state, backtest_state.position.size):
            entry_signals = self._generate_entry_signals(macd_state, market_state, backtest_state)
            signals.extend(entry_signals)
            
        # Check for exit conditions
        elif self._check_exit_conditions(macd_state, backtest_state.position.size):
            exit_signals = self._generate_exit_signals(market_state, backtest_state)
            signals.extend(exit_signals)
            
        # Check for rebalance needs
        else:
            rebalance_signals = self._check_rebalance_needs(market_state, backtest_state)
            signals.extend(rebalance_signals)
        
        # Only update last_signal_time if we actually generated signals or passed all checks
        self.last_signal_time = market_state.timestamp
        self.last_macd_state = macd_state
        
        return signals
    
    def calculate_target_position(self, market_state: MarketState, backtest_state: BacktestState) -> float:
        """Calculate ideal position size based on current conditions"""
        try:
            macd_state = self._get_macd_state(market_state.features)
            if not macd_state:
                return 0.0
            
            # Base position size
            base_size = self.config.strategy.position_size / market_state.close
            
            # Adjust for MACD strength
            macd_strength = abs(macd_state['hist']) / abs(macd_state['macd']) if abs(macd_state['macd']) > 0 else 0
            size_multiplier = min(macd_strength, 1.0)  # Cap at 100%
            
            # Adjust for volatility
            if backtest_state.risk_metrics and backtest_state.risk_metrics.volatility > 0:
                vol_adjustment = 1.0 - (backtest_state.risk_metrics.volatility / self.config.execution.max_volatility)
                size_multiplier *= max(vol_adjustment, 0.0)  # Don't let it go negative
            
            # Apply direction
            direction = 1 if macd_state['hist'] > 0 else -1
            
            return base_size * size_multiplier * direction
            
        except Exception as e:
            self.error_handler.handle_execution_error(
                StrategyError(
                    strategy_name=self.__class__.__name__,
                    operation="calculate_target_position",
                    reason=str(e),
                    state_details={"timestamp": market_state.timestamp}
                ),
                market_state,
                backtest_state
            )
            return 0.0
    
    def _get_macd_state(self, features: Dict[str, float]) -> Optional[Dict]:
        """Extract MACD state from features"""
        try:
            return {
                'macd': features.get('macd_line', 0),
                'signal': features.get('macd_signal', 0),
                'hist': features.get('macd_hist', 0),
                'weekly_zscore': features.get('macd_weekly_zscore', 0),
                'monthly_zscore': features.get('macd_monthly_zscore', 0)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get MACD state: {str(e)}")
            return None
    
    def _check_entry_conditions(self, macd_state: Dict, current_position: float) -> bool:
        """Check if entry conditions are met"""
        if current_position != 0:
            return False
            
        # Basic MACD cross check
        if abs(macd_state['hist']) < 0.0001:  # Too close to signal line
            return False
            
        # Check weekly zscore isn't extreme
        if abs(macd_state['weekly_zscore']) > self.config.strategy.zscore_weekly_threshold:
            return False
            
        # Check monthly zscore isn't extreme
        if abs(macd_state['monthly_zscore']) > self.config.strategy.zscore_monthly_threshold:
            return False
            
        # Confirm trend
        if macd_state['hist'] > 0:  # Bullish
            return macd_state['macd'] > macd_state['signal']
        else:  # Bearish
            return macd_state['macd'] < macd_state['signal']
    
    def _generate_entry_signals(self,
                              macd_state: Dict,
                              market_state: MarketState,
                              backtest_state: BacktestState) -> List[Signal]:
        """Generate entry signals"""
        signals = []
        
        # Determine direction and size
        side = Side.BUY if macd_state['hist'] > 0 else Side.SELL
        size = abs(self.calculate_target_position(market_state, backtest_state))
        
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
    
    def _check_exit_conditions(self, macd_state: Dict, current_position: float) -> bool:
        """Check if position should be closed"""
        if current_position == 0:
            return False
            
        # MACD cross
        if (current_position > 0 and macd_state['hist'] < 0) or \
           (current_position < 0 and macd_state['hist'] > 0):
            return True
            
        # Extreme zscores
        if abs(macd_state['weekly_zscore']) > self.config.strategy.zscore_weekly_threshold * 1.5:
            return True
            
        return False
    
    def _generate_exit_signals(self,
                             market_state: MarketState,
                             backtest_state: BacktestState) -> List[Signal]:
        """Generate exit signals"""
        signals = []
        
        current_position = backtest_state.position.size
        if current_position != 0:
            signal = self._create_signal(
                market_state=market_state,
                side=Side.SELL if current_position > 0 else Side.BUY,
                size=abs(current_position),
                signal_type=SignalType.EXIT,
                edge_bps=self.config.strategy.exit_bps,
                order_type=OrderType.TAKER
            )
            if signal:
                signals.append(signal)
        
        return signals
    
    def _check_rebalance_needs(self, market_state: MarketState, backtest_state: BacktestState) -> List[Signal]:
        """Check if position needs rebalancing"""
        signals = []
        current_position = backtest_state.position.size
        
        if current_position == 0:
            return signals
            
        # Calculate target position
        target_position = self.calculate_target_position(market_state, backtest_state)
        position_diff = target_position - current_position
        
        # Check if difference exceeds rebalance threshold
        if abs(position_diff) / abs(current_position) > self.config.strategy.rebalance_threshold:
            signals.append(self._create_signal(
                market_state=market_state,
                side=Side.BUY if position_diff > 0 else Side.SELL,
                size=abs(position_diff),
                signal_type=SignalType.REBALANCE,
                edge_bps=self.config.strategy.entry_bps/2  # Use half the normal edge for rebalancing
            ))
        
        return signals
    
    def _create_exit_signal(self, market_state: MarketState, backtest_state: BacktestState) -> Signal:
        """Create exit signal for risk management"""
        return self._create_signal(
            market_state=market_state,
            side=Side.SELL if backtest_state.position.size > 0 else Side.BUY,
            size=abs(backtest_state.position.size),
            signal_type=SignalType.RISK,
            edge_bps=0,  # No edge for risk exits
            order_type=OrderType.TAKER  # Force taker for risk exits
        )