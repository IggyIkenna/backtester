import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Union
from backtester.feature_calculators.base import FeatureCalculator
from backtester.feature_calculators.macd import MACDFeatureCalculator
from backtester.feature_calculators.stochastic import StochasticFeatureCalculator
from backtester.feature_calculators.bollinger import BollingerBandsCalculator
from backtester.feature_calculators.candle_patterns import CandlePatternCalculator
from backtester.feature_calculators.seasonality import SeasonalityCalculator
from backtester.feature_calculators.time_ranges import TimeRangeCalculator
from backtester.feature_calculators.volume_metrics import VolumeMetricsCalculator
from backtester.feature_calculators.volume_profile import VolumeProfileCalculator
from backtester.feature_calculators.volume_momentum import VolumeMomentumCalculator
from backtester.feature_calculators.vwap import VWAPCalculator
from backtester.feature_calculators.alternative_data import AlternativeDataCalculator
from backtester.feature_calculators.rsi import RSIFeatureCalculator
from backtester.feature_calculators.price_levels import PriceLevelsCalculator
from backtester.feature_calculators.market_structure import MarketStructureCalculator
from backtester.feature_calculators.streaks import StreaksCalculator
from backtester.feature_calculators.volatility_regime import VolatilityRegimeCalculator
from backtester.feature_calculators.funding_rates import FundingRateCalculator
from backtester.feature_calculators.returns import ReturnFeatureCalculator
from backtester.feature_calculators.relationship import RelationshipFeatureCalculator

import numpy as np
import time
import json
from backtester.configs.backtester_config import FeatureGroupTypes

class FeatureProcessor:
    """Process and save multiple technical features"""
    
    def __init__(self, calculators: List[FeatureCalculator]):
        self.calculators = calculators
        self.setup_logging()
        self.feature_counts = {}
        self.memory_usage = {}
        self.calculation_times = {}
        self.total_memory = 0
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Add file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"feature_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _log_feature_stats(self, df: pd.DataFrame, calculator_name: str, original_cols: List[str]):
        """Log feature count and memory usage"""
        # Get new features added by this calculator
        new_features = [col for col in df.columns if col not in original_cols]
        feature_count = len(new_features)
        
        if feature_count == 0:
            self.logger.warning(f"No new features added by {calculator_name}")
            return
        
        # Calculate memory usage for new features
        memory_mb = df[new_features].memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Store stats
        self.feature_counts[calculator_name] = feature_count
        self.memory_usage[calculator_name] = memory_mb
        self.total_memory += memory_mb
        
        # Log stats
        self.logger.info(f"\nFeature Statistics for {calculator_name}:")
        self.logger.info(f"Features added: {feature_count}")
        self.logger.info(f"Memory usage: {memory_mb:.2f} MB")
        self.logger.info(f"Feature names: {new_features}")
        self.logger.info(f"Running total memory: {self.total_memory:.2f} MB")
        
        # Log sample statistics for each feature
        for feature in new_features:
            self.logger.info(f"\nStats for {feature}:")
            feature_data = df[feature]
            self.logger.info(f"NaN count: {feature_data.isna().sum()}")
            if pd.api.types.is_numeric_dtype(feature_data):
                self.logger.info(f"Mean: {feature_data.mean():.4f}")
                self.logger.info(f"Std: {feature_data.std():.4f}")
                self.logger.info(f"Min: {feature_data.min():.4f}")
                self.logger.info(f"Max: {feature_data.max():.4f}")
    
    def process_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate all features using registered calculators"""
        result_df = df.copy()
        
        for calculator in self.calculators:
            calculator_name = calculator.__class__.__name__
            self.logger.info(f"\nProcessing features using {calculator_name}")
            
            try:
                # Store original columns to track new features
                original_cols = list(result_df.columns)
                
                # Time the calculation
                start_time = time.time()
                
                # Calculate features with symbol
                result_df = calculator.calculate(result_df, symbol)
                
                # Store calculation time
                calc_time = time.time() - start_time
                self.calculation_times[calculator_name] = calc_time
                
                # Log feature statistics
                self._log_feature_stats(result_df, calculator_name, original_cols)
                
            except Exception as e:
                self.logger.error(f"Error calculating features: {str(e)}")
                raise
        
        # Log final summary with calculation times
        self.logger.info("\nFinal Feature Summary:")
        self.logger.info(f"Total features: {sum(self.feature_counts.values())}")
        self.logger.info(f"Total memory: {self.total_memory:.2f} MB")
        self.logger.info("\nBreakdown by calculator:")
        for calc, count in self.feature_counts.items():
            calc_time = self.calculation_times.get(calc, 0)
            self.logger.info(
                f"{calc}: {count} features, "
                f"{self.memory_usage[calc]:.2f} MB, "
                f"time: {calc_time:.2f}s"
            )
        
        return result_df
    
    def save_features(self, df: pd.DataFrame, symbol: str, group: FeatureGroupTypes) -> Path:
        """Save features for a specific group"""
        features_dir = Path("data/features/current")
        group_dir = features_dir / group.value.name
        output_file = group_dir / f"{symbol}_features.csv"
        metadata_file = group_dir / f"{symbol}_metadata.json"
        
        # Get all features for this group using the new naming convention
        group_features = []
        for calculator in self.calculators:
            if hasattr(calculator, 'created_features'):
                group_features.extend(calculator.created_features)
        
        self.logger.info(f"\nSaving features:")
        self.logger.info(f"Base directory: {features_dir}")
        self.logger.info(f"Group directory: {group_dir}")
        self.logger.info(f"Features file: {output_file}")
        self.logger.info(f"Metadata file: {metadata_file}")
        
        if not group_features:
            self.logger.warning(f"No features found for group {group.value.name}")
            return None
        
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # Add required base columns
        base_columns = ['open', 'high', 'low', 'close', 'quote_volume', 'taker_buy_quote_volume']
        output_columns = list(set(base_columns + group_features))  # Remove duplicates
        
        start_time = time.time()
        df[output_columns].to_csv(output_file)
        self.logger.info(f"Saved {group.value.name} features in {time.time() - start_time:.2f}s")
        
        # Save metadata
        metadata = {
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "symbol": symbol,
            "feature_group": group.value.name,
            "features": group_features,
            "num_periods": len(df),
            "date_range": {
                "start": df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                "end": df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved features to: {output_file}")
        self.logger.info(f"Metadata saved to: {metadata_file}")
        
        return group_dir

def get_calculator_group(group: Union[str, FeatureGroupTypes], timeframe_multipliers: list) -> List[FeatureCalculator]:
    """Get calculators for a specific feature group"""
    # Convert string to enum if needed
    if isinstance(group, str):
        group = FeatureGroupTypes.from_string(group)
    elif not isinstance(group, FeatureGroupTypes):
        raise ValueError(f"group must be str or FeatureGroupTypes, got {type(group)}")
    
    calculators = []
    logger = logging.getLogger(__name__)
    
    if group == FeatureGroupTypes.SEASONALITY:
        calc = SeasonalityCalculator(
            timeframe="1h",
            config_path=str(Path("configs/seasonality_config.json"))
        )
        calculators.append(calc)
        logger.info("Added Seasonality calculator")
            
    elif group == FeatureGroupTypes.BOLLINGER:
        bb_base_params = [
            {
                'period': 20, 
                'std_levels': [1, 2, 3, 4],
                'atr_levels': [1.8, 2.0, 2.5, 3.0],
                'ivol_levels': [1.8, 2.0, 2.5, 3.0]
            },
            {
                'period': 10, 
                'std_levels': [1, 2, 3, 4],
                'atr_levels': [1.8, 2.0, 2.5, 3.0],
                'ivol_levels': [1.8, 2.0, 2.5, 3.0]
            },
            {
                'period': 14, 
                'std_levels': [1, 2, 3, 4],
                'atr_levels': [1.8, 2.0, 2.5, 3.0],
                'ivol_levels': [1.8, 2.0, 2.5, 3.0]
            },
            {
                'period': 30, 
                'std_levels': [1, 2, 3, 4],
                'atr_levels': [1.8, 2.0, 2.5, 3.0],
                'ivol_levels': [1.8, 2.0, 2.5, 3.0]
            }
        ]
        for multiplier in timeframe_multipliers:
            for base_params in bb_base_params:
                params = {
                    'period': base_params['period'] * multiplier,
                    'std_levels': base_params['std_levels'],
                    'atr_levels': base_params['atr_levels'],
                    'ivol_levels': base_params['ivol_levels'],
                    'timeframe': f"{multiplier}h",
                    'zscore_periods': [168, 720]  # 1w, 1m in hours
                }
                calc = BollingerBandsCalculator(**params)
                calculators.append(calc)
                logger.info(f"Added Bollinger Bands calculator for {multiplier}h timeframe with period {params['period']}")
    
    elif group == FeatureGroupTypes.MACD:
        macd_base_params = [
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'fast_period': 8, 'slow_period': 21, 'signal_period': 5},
            {'fast_period': 21, 'slow_period': 55, 'signal_period': 9},
            {'fast_period': 5, 'slow_period': 35, 'signal_period': 5},
        ]
        for multiplier in timeframe_multipliers:
            for base_params in macd_base_params:
                params = {
                    'fast_period': base_params['fast_period'] * multiplier,
                    'slow_period': base_params['slow_period'] * multiplier,
                    'signal_period': base_params['signal_period'] * multiplier,
                    'timeframe': f"{multiplier}h"
                }
                calc = MACDFeatureCalculator(**params)
                calculators.append(calc)
                logger.info(f"Added MACD calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.STOCHASTIC:
        stoch_base_params = [
            {'k_period': 5, 'd_period': 3, 'smooth_k': 3},
            {'k_period': 8, 'd_period': 3, 'smooth_k': 3},
            {'k_period': 12, 'd_period': 3, 'smooth_k': 3},
            {'k_period': 21, 'd_period': 3, 'smooth_k': 3},
        ]
        for multiplier in timeframe_multipliers:
            for base_params in stoch_base_params:
                params = {
                    'k_period': base_params['k_period'] * multiplier,
                    'd_period': base_params['d_period'] * multiplier,
                    'smooth_k': base_params['smooth_k'] * multiplier,
                    'timeframe': f"{multiplier}h"
                }
                calc = StochasticFeatureCalculator(**params)
                calculators.append(calc)
                logger.info(f"Added Stochastic calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.CANDLE_PATTERNS:
        candle_base_params = [
            {
                'doji_threshold': 0.1,
                'shadow_threshold': 2.0,
                'pattern_window': 20,
                'zscore_periods': [168, 720]  # 1w, 1m in hours
            },
            {
                'doji_threshold': 0.15,
                'shadow_threshold': 2.5,
                'pattern_window': 30,
                'zscore_periods': [168, 720]
            }
        ]
        
        for multiplier in timeframe_multipliers:
            for base_params in candle_base_params:
                params = {
                    'timeframe': f"{multiplier}h",
                    'doji_threshold': base_params['doji_threshold'],
                    'shadow_threshold': base_params['shadow_threshold'],
                    'pattern_window': base_params['pattern_window'] * multiplier,
                    'zscore_periods': base_params['zscore_periods']
                }
                calc = CandlePatternCalculator(**params)
                calculators.append(calc)
                logger.info(
                    f"Added Candle Pattern calculator for {multiplier}h timeframe "
                    f"with window {params['pattern_window']}"
                )
    
    elif group == FeatureGroupTypes.TIME_RANGES:
        for multiplier in timeframe_multipliers:
            # Complete periods
            calc = TimeRangeCalculator(
                timeframe=f"{multiplier}h",
                mode="complete"
            )
            calculators.append(calc)
            logger.info(f"Added Time Range calculator (complete) for {multiplier}h timeframe")
            
            # Developing periods
            calc = TimeRangeCalculator(
                timeframe=f"{multiplier}h",
                mode="developing"
            )
            calculators.append(calc)
            logger.info(f"Added Time Range calculator (developing) for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.VOLUME_METRICS:
        for multiplier in timeframe_multipliers:
            # Rolling windows only (no categorical)
            calc = VolumeMetricsCalculator(
                timeframe=f"{multiplier}h",
                categorical_mode=None  # Only rolling windows
            )
            calculators.append(calc)
            logger.info(f"Added Volume Metrics calculator (rolling) for {multiplier}h timeframe")
            
            # Complete categorical periods
            calc = VolumeMetricsCalculator(
                timeframe=f"{multiplier}h",
                categorical_mode=None  # Categorical complete periods
            )
            calculators.append(calc)
            logger.info(f"Added Volume Metrics calculator (discrete) for {multiplier}h timeframe")
            
            # Developing categorical periods
            calc = VolumeMetricsCalculator(
                timeframe=f"{multiplier}h",
                categorical_mode=None  # Categorical developing periods
            )
            calculators.append(calc)
            logger.info(f"Added Volume Metrics calculator (developing) for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.ALTERNATIVE:
        for multiplier in timeframe_multipliers:
            calc = AlternativeDataCalculator(
                timeframe=f"{multiplier}h",
                rolling_windows=[
                    24,      # 1 day
                    72,      # 3 days
                    168,     # 1 week
                    336,     # 2 weeks
                    720      # 1 month
                ],
                zscore_periods=[168, 720]  # 1w, 1m in hours
            )
            calculators.append(calc)
            logger.info(f"Added Alternative Data calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.RSI:
        rsi_base_params = [
            {
                'period': 14,
                'oversold_levels': [20, 30, 40],
                'overbought_levels': [60, 70, 80]
            },
            {
                'period': 7,
                'oversold_levels': [20, 30, 40],
                'overbought_levels': [60, 70, 80]
            },
            {
                'period': 21,
                'oversold_levels': [20, 30, 40],
                'overbought_levels': [60, 70, 80]
            }
        ]
        
        for multiplier in timeframe_multipliers:
            for base_params in rsi_base_params:
                params = {
                    'period': base_params['period'] * multiplier,
                    'oversold_levels': base_params['oversold_levels'],
                    'overbought_levels': base_params['overbought_levels'],
                    'timeframe': f"{multiplier}h",
                    'zscore_periods': [168, 720]  # 1w, 1m in hours
                }
                calc = RSIFeatureCalculator(**params)
                calculators.append(calc)
                logger.info(
                    f"Added RSI calculator for {multiplier}h timeframe "
                    f"with period {params['period']}"
                )
    
    elif group == FeatureGroupTypes.PRICE_LEVELS:
        # Define price level parameters for different timeframes
        price_level_params = {
            'major_levels': [1000, 5000, 10000, 20000, 30000, 40000, 50000],
            'minor_levels': [1000, 2000, 3000, 4000, 5000],
            'level_margin': 0.01,  # 1% margin around levels
            'interaction_window': 20,  # periods to consider for level strength
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust interaction window based on timeframe
            params = price_level_params.copy()
            params['interaction_window'] = params['interaction_window'] * multiplier
            params['timeframe'] = f"{multiplier}h"
            
            calc = PriceLevelsCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Price Levels calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.VOLUME_PROFILE:
        # Define volume profile parameters for different timeframes
        volume_profile_params = {
            'num_bins': 100,  # Number of price bins for volume distribution
            'value_area_pct': 0.70,  # Value Area percentage (typically 70%)
            'hvn_threshold': 1.5,  # Threshold for High Volume Node
            'lvn_threshold': 0.5,  # Threshold for Low Volume Node
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust window size based on timeframe
            params = volume_profile_params.copy()
            params['window_size'] = 24 * multiplier  # 24 periods (1 day) worth of data
            params['timeframe'] = f"{multiplier}h"
            
            calc = VolumeProfileCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Volume Profile calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.VWAP:
        # Define VWAP parameters for different timeframes
        vwap_base_periods = [5, 8, 10, 12, 14, 20, 21, 26, 30, 50, 55, 100, 200]
        
        for multiplier in timeframe_multipliers:
            # Adjust periods based on timeframe
            params = {
                'timeframe': f"{multiplier}h",
                'periods': [p * multiplier for p in vwap_base_periods],
                'zscore_periods': [168, 720]  # 1w, 1m in hours
            }
            
            calc = VWAPCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added VWAP calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.MARKET_STRUCTURE:
        # Define market structure parameters for different timeframes
        market_structure_params = {
            'swing_periods': [5, 10, 20, 50],  # Periods to identify swing points
            'strength_threshold': [1.5, 2.0, 2.5],  # Thresholds for swing strength
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust periods based on timeframe
            params = market_structure_params.copy()
            params['swing_periods'] = [p * multiplier for p in params['swing_periods']]
            params['timeframe'] = f"{multiplier}h"
            
            calc = MarketStructureCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Market Structure calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.STREAKS:
        # Define streaks parameters for different timeframes
        streaks_params = {
            'volatility_threshold': 0.5,  # % threshold for high volatility
            'volume_threshold': 1.5,  # multiplier over average for high volume
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust thresholds based on timeframe
            params = streaks_params.copy()
            params['volatility_threshold'] = params['volatility_threshold'] * np.sqrt(multiplier)
            params['timeframe'] = f"{multiplier}h"
            
            calc = StreaksCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Streaks calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.VOLATILITY_REGIME:
        # Define volatility regime parameters for different timeframes
        vol_regime_params = {
            'short_window': 24,     # 1 day
            'medium_window': 168,   # 1 week
            'long_window': 720,     # 1 month
            'num_regimes': 3,       # Low, Medium, High volatility
            'transition_window': 12, # Hours to detect regime transitions
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust windows based on timeframe
            params = vol_regime_params.copy()
            params['short_window'] = params['short_window'] * multiplier
            params['medium_window'] = params['medium_window'] * multiplier
            params['long_window'] = params['long_window'] * multiplier
            params['transition_window'] = params['transition_window'] * multiplier
            params['timeframe'] = f"{multiplier}h"
            
            calc = VolatilityRegimeCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Volatility Regime calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.FUNDING_RATES:
        funding_rate_params = {
            'windows': [8, 24, 72, 168],  # 8h, 1d, 3d, 1w
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            params = funding_rate_params.copy()
            params['windows'] = [w * multiplier for w in params['windows']]
            params['timeframe'] = f"{multiplier}h"
            
            calc = FundingRateCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Funding Rates calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.VOLUME_MOMENTUM:
        volume_momentum_params = {
            'momentum_periods': [5, 10, 20, 50],
            'volume_ratio_periods': [24, 72, 168],  # 1d, 3d, 1w
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            params = volume_momentum_params.copy()
            params['momentum_periods'] = [p * multiplier for p in params['momentum_periods']]
            params['volume_ratio_periods'] = [p * multiplier for p in params['volume_ratio_periods']]
            params['timeframe'] = f"{multiplier}h"
            
            calc = VolumeMomentumCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Volume Momentum calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.RETURNS:
        # Define return parameters for different timeframes
        return_params = {
            'lags': [1, 2, 3, 4, 5, 10, 20, 50],
            'vol_windows': [5, 10, 20, 50],
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust parameters based on timeframe
            params = return_params.copy()
            params['lags'] = [lag * multiplier for lag in params['lags']]
            params['vol_windows'] = [w * multiplier for w in params['vol_windows']]
            params['timeframe'] = f"{multiplier}h"
            
            calc = ReturnFeatureCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Returns calculator for {multiplier}h timeframe")
    
    elif group == FeatureGroupTypes.RELATIONSHIP:
        # Define relationship parameters for different timeframes
        relationship_params = {
            'correlation_windows': [24, 72, 168, 336, 720],  # 1d, 3d, 1w, 2w, 1m
            'beta_windows': [24, 72, 168, 336, 720],
            'relative_strength_windows': [24, 72, 168, 336, 720],
            'zscore_periods': [168, 720]  # 1w, 1m in hours
        }
        
        for multiplier in timeframe_multipliers:
            # Adjust windows based on timeframe
            params = relationship_params.copy()
            params['correlation_windows'] = [w * multiplier for w in params['correlation_windows']]
            params['beta_windows'] = [w * multiplier for w in params['beta_windows']]
            params['relative_strength_windows'] = [w * multiplier for w in params['relative_strength_windows']]
            params['timeframe'] = f"{multiplier}h"
            
            calc = RelationshipFeatureCalculator(**params)
            calculators.append(calc)
            logger.info(f"Added Relationship calculator for {multiplier}h timeframe")
    
    return calculators

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Hard-coded parameters
    end_date = pd.Timestamp('2024-12-15').tz_localize(None)  # End date
    start_date = end_date - pd.Timedelta(days=2425)  # Last 5 years
    
    # List of all feature groups to process
    feature_groups = [
        # FeatureGroupTypes.MACD,
        # FeatureGroupTypes.SEASONALITY,
        # FeatureGroupTypes.VOLUME_METRICS,
        FeatureGroupTypes.ALTERNATIVE,
        # FeatureGroupTypes.BOLLINGER,
        # FeatureGroupTypes.CANDLE_PATTERNS,
        # FeatureGroupTypes.RSI,
        # FeatureGroupTypes.PRICE_LEVELS,
        # FeatureGroupTypes.VOLUME_PROFILE,
        # FeatureGroupTypes.VWAP,
        # FeatureGroupTypes.MARKET_STRUCTURE,
        # FeatureGroupTypes.STREAKS,
        # FeatureGroupTypes.VOLATILITY_REGIME,
        # FeatureGroupTypes.STOCHASTIC,
        # FeatureGroupTypes.TIME_RANGES,
        # FeatureGroupTypes.FUNDING_RATES,
        # FeatureGroupTypes.VOLUME_MOMENTUM,
        # FeatureGroupTypes.RETURNS,
        # FeatureGroupTypes.RELATIONSHIP  # Add Relationship feature group
    ]
    
    timeframe_multipliers = [1]
    
    all_symbols = ["BTCUSDT"]

    # Add dictionaries to track group stats
    group_times = {}
    group_sizes = {}
    group_feature_counts = {}
    
    for symbol in  ["BTCUSDT"]:
        logger.info(f"\nProcessing features for {symbol}")
    
        for group in feature_groups:
            start_time = time.time()
            logger.info(f"Processing {group.value.name} features from {start_date} to {end_date}")

            # Get calculators for specified group
            calculators = get_calculator_group(group, timeframe_multipliers)
            processor = FeatureProcessor(calculators)

            try:
                # Load market data
                data_dir = Path("data/raw/market_data_2425d_20241217_022924/binance-futures")
                file_path = data_dir / f"{symbol}_1h.csv"

                logger.info(f"\nData paths:")
                logger.info(f"Input directory: {data_dir}")
                logger.info(f"Input file: {file_path}")

                # Load data and filter date range
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Make timezone naive
                df.set_index('timestamp', inplace=True)

                # Include lookback period for calculations
                max_lookback = 365 * 24  # 365 days in hours for z-scores
                calc_start = start_date - pd.Timedelta(hours=max_lookback)

                # Filter data
                mask = (df.index >= calc_start) & (df.index <= end_date)
                df = df[mask].copy()

                logger.info(f"Loaded {len(df)} periods including lookback data")

                # Calculate features
                df_with_features = processor.process_features(df, symbol)

                # Filter to requested date range for output
                df_with_features = df_with_features[
                    (df_with_features.index >= start_date) &
                    (df_with_features.index <= end_date)
                ]

                logger.info(f"Final output contains {len(df_with_features)} periods")

                # Save features
                output_dir = processor.save_features(
                    df_with_features,
                    symbol,
                    group
                )

                if output_dir:
                    logger.info(f"\nFeature calculation complete.")
                    logger.info(f"Output directory: {output_dir}")
                    logger.info(f"Features file: {output_dir / f'{symbol}_features.csv'}")
                    logger.info(f"Metadata file: {output_dir / f'{symbol}_metadata.json'}")
                else:
                    logger.warning("No output directory created - no features were generated")

                # Track statistics
                group_time = time.time() - start_time
                group_times[group.value.name] = group_time
                
                # Get feature count and size for this group
                feature_count = sum(processor.feature_counts.values())
                memory_usage = processor.total_memory
                
                group_sizes[group.value.name] = memory_usage
                group_feature_counts[group.value.name] = feature_count
                
            except Exception as e:
                logger.error(f"Feature calculation failed: {str(e)}", exc_info=True)
                raise
    
    # Print final summary
    logger.info("\n" + "="*50)
    logger.info("FEATURE GENERATION SUMMARY")
    logger.info("="*50)
    
    # Sort groups by processing time
    sorted_groups = sorted(group_times.items(), key=lambda x: x[1], reverse=True)
    
    total_time = sum(group_times.values())
    total_size = sum(group_sizes.values())
    total_features = sum(group_feature_counts.values())
    
    logger.info("\nBreakdown by Feature Group:")
    logger.info(f"{'Group':<20} {'Time':<10} {'Size':<10} {'Features':<10} {'Time %':<10}")
    logger.info("-"*60)
    
    for group_name, time_taken in sorted_groups:
        size = group_sizes.get(group_name, 0)
        feature_count = group_feature_counts.get(group_name, 0)
        time_percent = (time_taken / total_time) * 100
        
        logger.info(
            f"{group_name:<20} "
            f"{time_taken:>8.1f}s "
            f"{size:>8.1f}MB "
            f"{feature_count:>8} "
            f"{time_percent:>8.1f}%"
        )
    
    logger.info("-"*60)
    logger.info(f"{'TOTAL':<20} {total_time:>8.1f}s {total_size:>8.1f}MB {total_features:>8}")
    
    logger.info("\nMost time-consuming groups:")
    for group_name, time_taken in sorted_groups[:3]:
        logger.info(f"- {group_name}: {time_taken:.1f}s ({(time_taken/total_time)*100:.1f}%)")
    
    logger.info("\nLargest feature groups:")
    sorted_by_size = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)
    for group_name, size in sorted_by_size[:3]:
        logger.info(f"- {group_name}: {size:.1f}MB ({(size/total_size)*100:.1f}%)")
    
    logger.info(f"\nTotal processing time: {total_time:.1f}s")
    logger.info(f"Total memory usage: {total_size:.1f}MB")
    logger.info(f"Total features generated: {total_features}")

if __name__ == "__main__":
    main() 