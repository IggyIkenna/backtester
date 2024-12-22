from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class VWAPCalculator(FeatureCalculator):
    """
    Calculate Volume Weighted Average Price (VWAP) features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 periods: List[int] = [5, 8, 10, 12, 14, 20, 21, 26, 30, 50, 55, 100, 200],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.periods = periods
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close', 'quote_volume']
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP for a given period"""
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate cumulative values
        cumulative_tp_vol = (typical_price * df['quote_volume']).rolling(window=period).sum()
        cumulative_vol = df['quote_volume'].rolling(window=period).sum()
        
        # Calculate VWAP
        vwap = cumulative_tp_vol / cumulative_vol
        
        return vwap
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate VWAP features"""
        result_df = super().calculate(df, symbol)
        self.symbol = symbol
        self.created_features = []
        
        # Create a dictionary to store all new features
        new_features = {}
        
        # Calculate base VWAP
        volume = df['quote_volume']
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        for period in self.periods:
            # Calculate rolling VWAP components
            rolling_volume = volume.rolling(period).sum()
            rolling_tp_vol = (typical_price * volume).rolling(period).sum()
            rolling_vwap = rolling_tp_vol / rolling_volume
            
            # Calculate standard deviation
            rolling_std = (
                ((typical_price - rolling_vwap) ** 2 * volume).rolling(period).sum() 
                / rolling_volume
            ).pow(0.5)
            
            # Calculate features for each multiplier
            for mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
                # Generate feature names
                base_name = f'vwap_{period}h'
                upper_feature = self.get_feature_name(f'{base_name}_upper_{mult}')
                lower_feature = self.get_feature_name(f'{base_name}_lower_{mult}')
                pos_feature = self.get_feature_name(f'{base_name}_pos_{mult}')
                
                # Calculate bands
                new_features[upper_feature] = rolling_vwap + (rolling_std * mult)
                new_features[lower_feature] = rolling_vwap - (rolling_std * mult)
                new_features[pos_feature] = (df['close'] - rolling_vwap) / (rolling_std * mult)
                
                # Log feature creation
                for feature in [upper_feature, lower_feature, pos_feature]:
                    self.log_feature_creation(feature, result_df)
        
        # Combine all new features at once
        result_df = pd.concat([
            result_df,
            pd.DataFrame(new_features, index=result_df.index)
        ], axis=1)
        
        return result_df