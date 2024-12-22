from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import logging
import numpy as np
from configs.backtester_config import FeatureGroupTypes
from utils.feature_names import get_feature_name


class FeatureCalculator(ABC):
    """Base class for all feature calculators"""
    
    EPOCH_START = pd.Timestamp('2000-01-01 00:00:00').tz_localize(None)
    ZSCORE_PERIODS = 365 * 24  # 365 days in hours for long-term z-scores
    
    def __init__(self, timeframe: str = "1h"):
        """Initialize the calculator with timeframe"""
        self.timeframe = timeframe
        self.created_features = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_required_columns(self) -> list:
        """Return list of required input columns"""
        pass

    def get_feature_name(self, feature_suffix: str, symbol: Optional[str] = None) -> str:
        """
        Generate standardized feature names with proper symbol prefixing.
        
        Args:
            feature_suffix: The feature-specific part of the name (e.g., '1h_close_ma_20')
            symbol: Optional trading symbol (e.g., 'BTCUSDT'). If None, uses self.symbol
        
        Returns:
            str: The complete feature name with proper prefix
        """
        # Use provided symbol or fall back to instance symbol
        symbol_to_use = symbol if symbol is not None else self.symbol
        return get_feature_name(symbol_to_use, feature_suffix)
    
    def validate(self, df: pd.DataFrame) -> None:
        """Validate input data"""
        # Check required columns
        missing = [col for col in self.get_required_columns() if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure data starts at or after epoch
        if df.index[0] < self.EPOCH_START:
            raise ValueError(f"Data starts before epoch ({self.EPOCH_START})")
    
    def _align_to_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align data to full periods starting from epoch"""
        # Find first full period after epoch
        first_valid = df.index[0]
        if first_valid < self.EPOCH_START:
            df = df[df.index >= self.EPOCH_START]
        
        return df
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Base calculate method that validates required columns"""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {missing_cols} "
                f"which are not present in the DataFrame"
            )
        
        return df.copy()
    
    def calculate_zscore(self, series: pd.Series, min_periods: int = 20) -> pd.Series:
        """Calculate z-score of a series using expanding window"""
        mean = series.expanding(min_periods=min_periods).mean()
        std = series.expanding(min_periods=min_periods).std()
        return ((series - mean) / std).fillna(0)
    
    def calculate_time_since(self, condition: np.ndarray, lookback: Optional[int] = None) -> np.ndarray:
        """
        Calculate the number of periods since a condition was last True.
        
        Args:
            condition: Boolean array of the condition
            lookback: Optional maximum number of periods to look back
        
        Returns:
            Array with the number of periods since condition was True
        """
        if not isinstance(condition, np.ndarray):
            condition = np.array(condition)
            
        # Initialize the result array with the maximum possible value
        result = np.full_like(condition, fill_value=len(condition), dtype=np.float32)
        
        # Find indices where condition is True
        true_indices = np.where(condition)[0]
        
        if len(true_indices) == 0:
            return result
            
        # For each position, find the distance to the most recent True value
        for i in range(len(condition)):
            # Find the closest previous True index
            prev_true = true_indices[true_indices <= i]
            if len(prev_true) > 0:
                result[i] = i - prev_true[-1]
                
            # Apply lookback limit if specified
            if lookback is not None and result[i] > lookback:
                result[i] = lookback
                
        return result
    
    def load_dependency_features(self, symbol: str, group) -> pd.DataFrame:
        """Load features from another group"""
        features_dir = Path("data/features/current")
        group_dir = features_dir / group.value.name
        feature_file = group_dir / f"{symbol}_features.csv"
        
        if not feature_file.exists():
            raise ValueError(f"Required features from {group.value.name} not found. Run that group first.")
            
        df = pd.read_csv(feature_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_required_features(self) -> List['FeatureGroupTypes']:
        """Override to specify required feature groups"""
        return []
    
    def load_seasonality_periods(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Load and filter seasonality data"""
        if symbol is None:
            raise ValueError("symbol parameter is required to load seasonality data")
        
        # Check if seasonality columns already exist
        existing_season_cols = [col for col in df.columns if 'seasonality_' in col]
        if existing_season_cols:
            self.logger.info("Seasonality columns already present, skipping load")
            return df
        
        seasonality_df = self.load_dependency_features(
            symbol,
            FeatureGroupTypes.SEASONALITY
        )
        
        # Filter out excluded seasons
        season_cols = [col for col in seasonality_df.columns 
                      if 'seasonality_' in col and 
                      not any(excluded in col for excluded in self.excluded_seasons)]
        
        self.logger.info(f"Loading seasonality periods: {season_cols}")
        
        # Use merge with indicator to check for duplicates
        result = df.merge(seasonality_df[season_cols], 
                         left_index=True, 
                         right_index=True,
                         how='left')
        
        return result
    
    def log_feature_creation(self, feature_name: str, df: pd.DataFrame) -> None:
        """Log feature creation and store in created_features list"""
        self.created_features.append(feature_name)
        self.logger.debug(f"Created feature: {feature_name}")
        
        # Log basic statistics if debug is enabled
        if self.logger.isEnabledFor(logging.DEBUG):
            feature_data = df[feature_name]
            self.logger.debug(f"Stats for {feature_name}:")
            self.logger.debug(f"NaN count: {feature_data.isna().sum()}")
            self.logger.debug(f"Unique count: {feature_data.nunique()}")
            if pd.api.types.is_numeric_dtype(feature_data):
                self.logger.debug(f"Mean: {feature_data.mean():.4f}")
                self.logger.debug(f"Std: {feature_data.std():.4f}")
                self.logger.debug(f"Min: {feature_data.min():.4f}")
                self.logger.debug(f"Max: {feature_data.max():.4f}")
    
    def shift_features(self, df: pd.DataFrame, shift_periods: int = 1) -> pd.DataFrame:
        """
        Shift features forward to prevent look-ahead bias.
        
        Args:
            df: DataFrame containing features
            shift_periods: Number of periods to shift features forward
        
        Returns:
            DataFrame with shifted features
        """
        if shift_periods <= 0:
            return df
            
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        df[feature_cols] = df[feature_cols].shift(shift_periods)
        return df