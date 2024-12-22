from feature_calculators.categorical_base import CategoricalFeatureCalculator
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class VolumeMetricsCalculator(CategoricalFeatureCalculator):
    """Calculate volume metrics with proper period handling"""
    
    def __init__(self, 
                 timeframe: str = "1h",
                 categorical_mode: Optional[str] = None,  # Can be None, 'discrete', or 'developing'
                 rolling_windows: List[int] = [1, 4, 12, 24, 48, 72, 168],  # Hours
                 zscore_periods: List[int] = [168, 720],  # 1 week, 1 month in hours
                 excluded_seasons: List[str] = None):
        
        # Initialize base class only if we're using categorical features
        if categorical_mode is not None:
            super().__init__(timeframe, categorical_mode, excluded_seasons)
        else:
            super().__init__(timeframe=timeframe)
            self.excluded_seasons = excluded_seasons or []
            
        self.categorical_mode = categorical_mode
        self.rolling_windows = rolling_windows
        self.zscore_periods = zscore_periods

    def get_required_columns(self) -> list:
        return ['quote_volume', 'taker_buy_quote_volume']
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate comprehensive volume metrics"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = f"{self.timeframe}_rolling"
        
        # Calculate rolling window metrics
        for window in self.rolling_windows:
            self._calculate_rolling_metrics(result_df, window, base_feature, symbol)
        
        # Calculate categorical metrics if needed
        if self.categorical_mode is not None:
            if symbol is None:
                raise ValueError("symbol parameter is required for categorical features")
            
            result_df = self.load_seasonality_periods(result_df, symbol)
            self._calculate_categorical_metrics(result_df, symbol)
        
        return result_df

    def _calculate_rolling_metrics(self, df: pd.DataFrame, window: int, base_feature: str, symbol: str):
        """Calculate rolling metrics using optimized vectorized operations"""
        # Shift data once for all calculations
        shifted_data = df.shift(1)
        
        # Pre-calculate rolling windows to avoid redundant computations
        volume_rolling = shifted_data['quote_volume'].rolling(window=window, min_periods=1)
        buy_volume_rolling = shifted_data['taker_buy_quote_volume'].rolling(window=window, min_periods=1)
        
        # Calculate metrics in bulk
        metrics = {
            'total': volume_rolling.sum(),
            'buy': buy_volume_rolling.sum(),
            'max_volume': volume_rolling.max(),
            'mean_volume': volume_rolling.mean()
        }
        
        # Assign total volume
        total_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_total")
        df[total_feature] = metrics['total']
        self.log_feature_creation(total_feature, df)
        
        # Assign buy volume
        buy_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_buy")
        df[buy_feature] = metrics['buy']
        self.log_feature_creation(buy_feature, df)
        
        # Calculate and assign buy ratio
        ratio_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_buy_ratio")
        df[ratio_feature] = metrics['buy'] / (metrics['total'] + 1e-10)  # Avoid division by zero
        self.log_feature_creation(ratio_feature, df)
        
        # Calculate and assign concentration
        conc_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_concentration")
        df[conc_feature] = metrics['max_volume'] / (metrics['mean_volume'] + 1e-10)
        self.log_feature_creation(conc_feature, df)
        
        # Calculate momentum features
        momentum_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_momentum")
        df[momentum_feature] = (
            shifted_data['quote_volume'].pct_change(window) * 
            (df[ratio_feature] - 0.5)  # Adjust for buy/sell imbalance
        )
        self.log_feature_creation(momentum_feature, df)
        
        # Calculate volume acceleration
        accel_feature = self.get_feature_name(symbol, f"{base_feature}_{window}_acceleration")
        df[accel_feature] = df[momentum_feature].pct_change()
        self.log_feature_creation(accel_feature, df)

    def _calculate_categorical_metrics(self, df: pd.DataFrame, symbol: str):
        """Calculate categorical metrics using vectorized operations"""
        shifted_data = df.shift(1)  # Use shifted data for forward-looking metrics
        
        # Map seasonality types to feature names
        seasonality_mapping = {
            'quarter': 'quarter',
            'is_weekend': 'weekend',
            'is_us_session': 'us_session',
            'is_asia_session': 'asia_session',
            'is_london_session': 'london_session',
            'is_session_1': 'session_1',
            'is_session_2': 'session_2',
            'is_session_3': 'session_3',
            'is_session_4': 'session_4',
            'is_session_5': 'session_5',
            'is_session_6': 'session_6',
            'is_NFP': 'nfp',
            'is_FOMC': 'fomc',
            'is_USD_GDP': 'gdp',
            'is_US_ELECTION': 'election'
        }
        
        # Get all seasonality columns for this timeframe
        season_cols = [col for col in df.columns 
                      if col.startswith(f'seasonality_{self.timeframe}_')]
        
        # Calculate metrics for each seasonality type
        for season_col in season_cols:
            # Extract the seasonality type and map to feature name
            season_type = season_col.replace(f'seasonality_{self.timeframe}_', '')
            feature_name = seasonality_mapping.get(season_type, season_type)
            
            # Create base name for this seasonality type
            base_feature = f"{self.timeframe}_{feature_name}"
            if self.categorical_mode == "developing":
                base_feature += "_dev"
            
            # Group by this seasonality type
            season_groups = shifted_data.groupby(df[season_col])
            
            # Calculate metrics
            total_feature = self.get_feature_name(symbol, f"{base_feature}_total")
            buy_feature = self.get_feature_name(symbol, f"{base_feature}_buy")
            ratio_feature = self.get_feature_name(symbol, f"{base_feature}_buy_ratio")
            conc_feature = self.get_feature_name(symbol, f"{base_feature}_concentration")
            
            df[total_feature] = season_groups['quote_volume'].transform('sum')
            df[buy_feature] = season_groups['taker_buy_quote_volume'].transform('sum')
            df[ratio_feature] = (
                df[buy_feature] / df[total_feature]
            )
            df[conc_feature] = (
                season_groups['quote_volume'].transform('max') /
                (season_groups['quote_volume'].transform('mean') + 1e-10)
            )
            
            # Log feature creation
            for feature in [total_feature, buy_feature, ratio_feature, conc_feature]:
                self.log_feature_creation(feature, df)
            
            # Calculate z-scores for this seasonality type
            for period in self.zscore_periods:
                for metric in ['total', 'buy_ratio', 'concentration']:
                    base_metric = self.get_feature_name(symbol, f"{base_feature}_{metric}")
                    zscore_feature = self.get_feature_name(symbol, f"{base_feature}_{metric}_zscore_{period}")
                    df[zscore_feature] = self.calculate_zscore(
                        df[base_metric],
                        min_periods=period
                    )
                    self.log_feature_creation(zscore_feature, df)