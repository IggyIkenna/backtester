# Backtester Documentation

## Architecture Sketch

[Market Data] → [Backtester]
                    ↓
    ┌───────────────┴──────────────┐
    ↓               ↓              ↓
[Strategy] → [ExecutionEngine] → [PositionManager]
    ↓               ↓              ↓
    └───────────[RiskManager]──────┘
                    ↓
             [OutputHandler]

## Overview
A backtesting system for cryptocurrency trading strategies, focusing on MACD-based signals with limit order execution using OHLCV data.

## System Architecture

### Core Components
1. **Backtester** - Main orchestrator
2. **Strategy** (MACD) - Signal generation
3. **ExecutionEngine** - Order management and fills
4. **PositionManager** - Position and PnL tracking
5. **RiskManager** - Risk checks and limits
6. **OutputHandler** - Results and visualization

### State Management
- **MarketState**: Price and feature data
- **PositionState**: Current positions and PnL
- **OrderState**: Active orders
- **BacktestState**: Complete system state
- **RiskMetrics**: Risk measurements

## Event Flow

1. **Initialization**
   - Load configuration from `backtest_config.yaml`
   - Initialize components
   - Load market data and features

2. **Per Candle Processing**
   ```
   Market Data → Strategy → Signals → Risk Checks → Orders → Fills → Position Updates
   ```

3. **Order Execution**
   - Limit orders simulated against OHLCV bars
   - Fills occur at limit price if crossed by High/Low
   - Maker/Taker fees applied based on execution type
   - Orders convert to market after `taker_conversion_hours`

## Key Features

### Strategy (MACD)
- MACD crossover signals
- Position sizing based on signal strength
- Entry/exit with configurable edge requirements

### Risk Management
- Position size limits (USD and coins)
- Leverage limits
- Concentration checks
- Trading hour restrictions

### Execution
- Limit order simulation
- Spread simulation based on volatility
- Maker/Taker fee structure
- Order timeout conversion

## Configuration Structure

yaml
backtest:
# General settings
start_date, end_date, symbol, etc.
time:
# Time-related parameters
candle_period, signal_period, trading_hours
strategy:
# Strategy parameters
position_size, leverage, MACD parameters
execution:
# Execution parameters
fees, spreads, size limits, order behavior

## Known Limitations

1. OHLCV-only data (no order book depth)
2. No partial fills
3. Simplified spread simulation
4. Perfect fill assumption at limit prices

## Pre-Run Checklist

1. **Data Requirements**
   - OHLCV data in `data/raw/market_data_1h_2425d_20241217_022924/`
   - Optional features in `backtester/data/features/current/`

2. **Configuration Validation**
   - Check position size limits
   - Verify fee structure
   - Confirm trading hours if used

3. **Risk Parameters**
   - Validate leverage limits
   - Check position concentration limits
   - Verify stop-loss levels

4. **Directory Structure**
   ```
   ├── data/
   │   └── raw/
   ├── results/
   ├── logs/
   └── configs/
   ```

## Common Issues to Check

1. **Circular Dependencies**
   - Current circular import between state.py and risk_manager.py
   - Solution: Move shared types to utils/types.py
   - Ensure proper import order:
     ```
     utils/types.py <- state.py <- risk_manager.py
     ```

2. **State Management**
   - Verify state updates are atomic
   - Check state consistency after trades
   - Monitor risk metric calculations
   - Validate position state transitions

3. **Order Processing**
   - Check limit order price updates
   - Verify fill price calculations
   - Monitor maker/taker conversions
   - Validate fee calculations

4. **Data Handling**
   - Ensure OHLCV data completeness
   - Check feature calculation timing
   - Verify timestamp consistency
   - Monitor for data gaps

## Performance Considerations

1. Large position sizes relative to liquidity
2. Aggressive limit order placement
3. Frequent order updates
4. Risk limit violations
5. Trading hour restrictions impact

## Component Dependencies

utils.types
├── state.MarketState
├── state.PositionState
├── state.OrderState
└── state.BacktestState
└── risk_manager.RiskManager
└── Backtester



## Next Steps

1. Fix circular dependencies in state.py and risk_manager.py
2. Validate configuration parameters
3. Verify data file paths and formats
4. Run with minimal test data
5. Monitor execution engine behavior

## Running the Backtest

1. **Setup**
   ```bash
   # Ensure directory structure
   mkdir -p data/raw results logs
   
   # Copy configuration
   cp configs/backtest_config.yaml.example configs/backtest_config.yaml
   
   # Place data files
   cp your_data.csv data/raw/market_data_1h_2425d_20241217_022924/
   ```

2. **Configuration**
   - Edit configs/backtest_config.yaml
   - Set appropriate date range
   - Configure strategy parameters
   - Set risk limits

3. **Execution**
   ```bash
   python run_backtest.py
   ```

4. **Results**
   - Check results/ directory for output files
   - Review logs/ for execution details
   - Analyze performance metrics
