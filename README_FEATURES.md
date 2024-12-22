# Feature Calculator Documentation

## Overview
The feature calculation system provides a modular framework for computing technical and market structure features across multiple timeframes. Features are organized into groups, with support for both rolling windows and categorical (session-based) calculations.

## Feature Naming Convention
Features follow a standardized naming pattern:
`{group_name}_{symbol}_{feature_type}_{timeframe}_{metric}`

For example:
- `macd_BTCUSDT_1h_line` - MACD line for BTCUSDT on 1h timeframe
- `volume_metrics_ETHUSDT_4h_buy_ratio` - Buy ratio for ETHUSDT on 4h timeframe
- `seasonality_BTCUSDT_1h_quarter` - Quarter indicator for BTCUSDT on 1h timeframe

Z-scores follow the pattern:
`{group_name}_{symbol}_{feature_type}_{timeframe}_{metric}_zscore_{period}`

## Feature Groups

### 1. Seasonality Features
Time-based categorization features that identify market periods and cycles.

#### Features
- Quarter indicators (1-4)
- Cycle year position (1-4)
- Session indicators:
  - Weekend flags
  - US/Asia/London trading sessions
  - Special events (NFP, FOMC, etc.)

#### Example Usage
```python
from feature_calculators.seasonality import SeasonalityCalculator

calc = SeasonalityCalculator(
    timeframe="1h",
    config_path="configs/seasonality_config.json"
)
df_with_features = calc.calculate(df, symbol="BTCUSDT")
```

### 2. Volume Metrics
Quote volume-based features with support for both rolling windows and categorical periods.

#### Data Requirements
- `quote_volume`: Total volume in quote currency (USD)
- `taker_buy_quote_volume`: Buy volume in quote currency (USD)

#### Feature Types
1. Rolling Window Features:
   - Windows: [1h, 4h, 12h, 24h, 48h, 72h, 168h]
   - Metrics:
     - Total quote volume (USD)
     - Buy quote volume (USD)
     - Buy/Total ratio
     - Volume concentration (max/mean)

2. Categorical Features:
   - Modes: Discrete and Developing
   - Metrics:
     - Period quote volume totals
     - Buy ratios
     - Volume concentration
   - Z-scores (1w, 1m periods)

#### Normalization
All volume metrics use quote currency (USD) values for better cross-market comparison and to avoid price impact on volume metrics.

#### Example Usage
```python
from feature_calculators.volume_metrics import VolumeMetricsCalculator

# Rolling windows only
calc = VolumeMetricsCalculator(
    timeframe="1h",
    categorical_mode=None
)

# With categorical features
calc = VolumeMetricsCalculator(
    timeframe="1h",
    categorical_mode="developing",
    rolling_windows=[1, 4, 12, 24, 48, 72, 168],
    zscore_periods=[168, 720]
)
df_with_features = calc.calculate(df, symbol="BTCUSDT")
```

## Base Classes

### 1. FeatureCalculator
Base class for all feature calculators providing:
- Data validation
- Period alignment
- Z-score calculations
- Feature dependencies
- Symbol-aware feature naming

### 2. CategoricalFeatureCalculator
Extended base class for session-based features:
- Discrete vs developing calculations
- Proper period handling
- Session transition management
- Symbol-aware feature naming

## Feature Validation
Features are validated for:
- Missing values
- Range constraints
- Data availability
- Period alignment
- Symbol presence in feature names

## Performance Considerations
1. Memory Usage
   - Features are calculated efficiently in-place
   - Memory usage is tracked and logged
   - Large lookback periods are handled carefully

2. Computation Speed
   - Vectorized operations where possible
   - Efficient period handling
   - Progress logging for long calculations

## Adding New Features
1. Create new calculator class
2. Implement required methods
3. Add to feature catalog
4. Update feature processor
5. Add validation rules
6. Follow symbol-aware naming convention

## Usage in Backtesting
Features can be:
1. Pre-calculated and saved
2. Loaded on demand
3. Combined across timeframes
4. Used for signal generation

## Future Enhancements
1. More feature groups planned:
   - Price action patterns
   - Market structure
   - Order flow metrics
   - Sentiment indicators

2. Improvements planned:
   - Dynamic lookback periods
   - Adaptive z-scores
   - More validation rules
   - Performance optimizations