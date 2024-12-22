from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List
import pandas as pd
import yaml
import logging

"""Configuration system for backtesting framework"""

@dataclass
class FeatureGroupInfo:
    name: str
    prefix: str
    description: str


class FeatureGroupTypes(Enum):
    """Feature group enumeration with metadata"""
    SEASONALITY = FeatureGroupInfo(
        name='seasonality',
        prefix='seasonality_',
        description='Time-based categorization features'
    )
    MACD = FeatureGroupInfo(
        name='macd',
        prefix='macd_',
        description='Moving Average Convergence Divergence'
    )
    BOLLINGER = FeatureGroupInfo(
        name='bollinger',
        prefix='bb_',
        description='Bollinger Bands'
    )
    RSI = FeatureGroupInfo(
        name='rsi',
        prefix='rsi_',
        description='Relative Strength Index'
    )
    STOCHASTIC = FeatureGroupInfo(
        name='stochastic',
        prefix='stochastic_',
        description='Stochastic Oscillator'
    )
    CANDLE_PATTERNS = FeatureGroupInfo(
        name='candle_patterns',
        prefix='candle_patterns_',
        description='Candle pattern recognition features'
    )
    TIME_RANGES = FeatureGroupInfo(
        name='time_ranges',
        prefix='time_ranges_',
        description='Time range categorization features'
    )
    VOLUME_METRICS = FeatureGroupInfo(
        name='volume_metrics',
        prefix='volume_metrics_',
        description='Volume metrics features'
    )
    PRICE_LEVELS = FeatureGroupInfo(
        name='price_levels',
        prefix='price_levels_',
        description='Price level categorization features'
    )
    ALTERNATIVE = FeatureGroupInfo(
        name='alternative',
        prefix='alternative_',
        description='Alternative features'
    )
    VWAP = FeatureGroupInfo(
        name='vwap',
        prefix='vwap_',
        description='Volume Weighted Average Price'
    )
    VOLUME_PROFILE = FeatureGroupInfo(
        name='volume_profile',
        prefix='volume_profile_',
        description='Volume profile features'
    )
    MARKET_STRUCTURE = FeatureGroupInfo(
        name='market_structure',
        prefix='market_structure_',
        description='Market structure features'
    )
    STREAKS = FeatureGroupInfo(
        name='streaks',
        prefix='streaks_',
        description='Streaks features'
    )
    VOLATILITY_REGIME = FeatureGroupInfo(
        name='volatility_regime',
        prefix='volatility_regime_',
        description='Volatility regime features'
    )
    FUNDING_RATES = FeatureGroupInfo(
        name='funding_rates',
        prefix='funding_rates_',
        description='Funding rates features'
    )
    VOLUME_MOMENTUM = FeatureGroupInfo(
        name='volume_momentum',
        prefix='volume_momentum_',
        description='Volume momentum features'
    )
    RETURNS = FeatureGroupInfo(
        name='returns',
        prefix='returns_',
        description='Returns features'
    )
    RELATIONSHIP = FeatureGroupInfo(
        name='relationship',
        prefix='relationship_',
        description='Relationship features'
    )

    @classmethod
    def from_string(cls, value: str) -> 'FeatureGroupTypes':
        """Convert string to enum, case-insensitive"""
        try:
            return next(
                member for member in cls
                if member.value.name.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"Invalid feature group: {value}. "
                f"Valid options are: {[m.value.name for m in cls.__members__.values()]}"
            )

    @property
    def name(self) -> str:
        """Get the name for this group"""
        return self.value.name

    @property
    def prefix(self) -> str:
        """Get the feature prefix for this group"""
        return self.value.prefix

    @property
    def description(self) -> str:
        """Get the description for this group"""
        return self.value.description

    def __str__(self) -> str:
        return self.value.name

    @classmethod
    def list_all(cls) -> list:
        """List all available feature groups"""
        return list(cls.__members__.values())


@dataclass
class TimeConfig:
    """Time-related configuration"""
    candle_period: str
    signal_period: str
    signal_start_hour: int
    signal_start_day: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    allow_overnight: bool = True
    trading_hours: Optional[tuple[int, int]] = None  # e.g. (9, 16) for 9am-4pm
    trading_days: Optional[List[str]] = None  # List of enabled trading days
    trading_disabled_datetime: Optional[List[tuple[str, str]]] = None
    trading_disabled_hours: Optional[List[tuple[int, int]]] = None
    
    VALID_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    def __post_init__(self):
        """Validate time configuration"""
        # Validate trading hours
        if self.trading_hours:
            if not (0 <= self.trading_hours[0] < 24 and 0 <= self.trading_hours[1] < 24):
                raise ValueError("Trading hours must be between 0-23")
        
        # Validate signal start hour
        if not 0 <= self.signal_start_hour < 24:
            raise ValueError("Signal start hour must be between 0-23")
        
        # Validate signal start day
        if self.signal_start_day.lower() not in self.VALID_DAYS:
            raise ValueError(f"Signal start day must be one of {self.VALID_DAYS}")
        
        # Validate trading days
        if self.trading_days is not None:
            if not isinstance(self.trading_days, list):
                raise ValueError("Trading days must be a list or None")
            
            # Convert to lowercase for comparison
            self.trading_days = [day.lower() for day in self.trading_days]
            
            # Validate each day
            invalid_days = set(self.trading_days) - set(self.VALID_DAYS)
            if invalid_days:
                raise ValueError(f"Invalid trading days: {invalid_days}. Must be one of {self.VALID_DAYS}")
            
            # Warn if signal start day not in trading days
            if self.trading_days and self.signal_start_day.lower() not in self.trading_days:
                logging.warning(
                    f"Signal start day {self.signal_start_day} is not in enabled trading days {self.trading_days}. "
                    "This may prevent strategy initialization."
                )

@dataclass
class ExecutionConfig:
    """Execution parameters"""
    maker_fee_bps: float
    taker_fee_bps: float
    simulated_spread_bps: float
    max_spread_bps: float
    
    # Order size limits
    min_order_size_usd: float
    max_order_size_usd: float

    
    # Position limits
    min_position_size_usd: float
    max_position_size_usd: float
    max_position_coins: float
    max_leverage: float
    concentration_limit: float
    
    # Limit order behavior
    price_update_threshold: float
    taker_conversion_hours: int

    # PnL Limits
    max_drawdown: float

    max_volatility: float

    use_stochastic_oscillator: bool
    use_momentum: bool
    use_dynamic_spread: bool
    
    def __post_init__(self):
        if self.maker_fee_bps > self.taker_fee_bps:
            raise ValueError("Maker fees cannot exceed taker fees")
        if self.max_leverage < 1:
            raise ValueError("Max leverage must be >= 1")

@dataclass
class StrategyConfig:
    """Strategy-specific configuration"""
    # Position sizing
    position_size: float
    leverage: float
    rebalance_threshold: float
    
    # Entry/exit parameters
    entry_bps: float  # Required edge for entry
    exit_bps: float  # Acceptable cost for exit
    
    # MACD parameters
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    zscore_weekly_threshold: float = 2.0
    zscore_monthly_threshold: float = 3.0

@dataclass
class BacktestConfig:
    """Main backtest configuration"""
    # Paths
    data_dir: Path
    results_dir: Path
    features_dir: Path
    log_dir: Path
    
    # Components
    time: TimeConfig
    features: list[FeatureGroupTypes]
    execution: ExecutionConfig
    strategy: StrategyConfig
    symbol: str
    
    # Behavior
    continue_on_error: bool = False
    log_level: str = "INFO"
    save_trades: bool = True
    save_signals: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'BacktestConfig':
        """Load config from YAML file"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            

        # Convert paths
        config_dict['data_dir'] = Path(config_dict['backtest']['data_dir'])
        config_dict['results_dir'] = Path(config_dict['backtest']['results_dir'])
        config_dict['features_dir'] = Path(config_dict['backtest']['features_dir'])
        config_dict['log_dir'] = Path(config_dict['logging']['log_dir'])
        
        # Convert timestamps
        config_dict['backtest']['time']['start_date'] = pd.Timestamp(config_dict['backtest']['start_date'])
        config_dict['backtest']['time']['end_date'] = pd.Timestamp(config_dict['backtest']['end_date'])

        # Get features
        config_dict['features'] = [FeatureGroupTypes.from_string(fg) for fg in config_dict['backtest']['features']['feature_groups']]

        # Get symbol
        config_dict['symbol'] = config_dict['backtest']['symbol']
        
        # Create nested configs
        config_dict['time'] = TimeConfig(**config_dict['backtest']['time'])
        config_dict['execution'] = ExecutionConfig(**config_dict['execution'])
        config_dict['strategy'] = StrategyConfig(**config_dict['strategy'])

        # get log level
        config_dict['log_level'] = config_dict['logging']['level']

        config_dict.pop('backtest')
        config_dict.pop('logging')
        
        return cls(**config_dict)
