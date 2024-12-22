from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class RSIFeatureCalculator(FeatureCalculator):
    """
    Calculate RSI features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 period: int = 14,
                 oversold_levels: List[float] = [20, 30, 40],
                 overbought_levels: List[float] = [60, 70, 80],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.period = period
        self.oversold_levels = oversold_levels
        self.overbought_levels = overbought_levels
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['close']
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI using shifted data"""
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_time_since_condition(self, condition_array: np.ndarray, index: pd.Index) -> pd.Series:
        """Calculate time periods since a condition was last true"""
        # Initialize result series
        time_since = pd.Series(index=index, dtype=float)
        last_true_idx = None
        
        # Iterate through array
        for i in range(len(condition_array)):
            if condition_array[i]:
                last_true_idx = i
            if last_true_idx is not None:
                time_since.iloc[i] = i - last_true_idx
                
        return time_since
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate RSI features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = f'{self.timeframe}_{self.period}'
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate base RSI
        rsi = self._calculate_rsi(shifted_data)
        
        # Store base value
        base_rsi_feature = self.get_feature_name(symbol, base_feature)
        result_df[base_rsi_feature] = rsi
        self.log_feature_creation(base_rsi_feature, result_df)
        
        # Calculate changes
        change_feature = self.get_feature_name(symbol, f'{base_feature}_change')
        result_df[change_feature] = rsi.diff()
        self.log_feature_creation(change_feature, result_df)
        
        # Calculate overbought/oversold conditions and time since
        for level in self.oversold_levels:
            # Oversold condition
            oversold_feature = self.get_feature_name(symbol, f'{base_feature}_oversold_{level}')
            oversold_condition = (rsi <= level)
            result_df[oversold_feature] = oversold_condition.astype(int)
            self.log_feature_creation(oversold_feature, result_df)
            
            # Time since oversold
            time_since_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_oversold_{level}')
            result_df[time_since_feature] = self._calculate_time_since_condition(oversold_condition.values, df.index)
            self.log_feature_creation(time_since_feature, result_df)
            
            # Distance from level
            dist_feature = self.get_feature_name(symbol, f'{base_feature}_dist_oversold_{level}')
            result_df[dist_feature] = (rsi - level).clip(upper=0)
            self.log_feature_creation(dist_feature, result_df)
        
        for level in self.overbought_levels:
            # Overbought condition
            overbought_feature = self.get_feature_name(symbol, f'{base_feature}_overbought_{level}')
            overbought_condition = (rsi >= level)
            result_df[overbought_feature] = overbought_condition.astype(int)
            self.log_feature_creation(overbought_feature, result_df)
            
            # Time since overbought
            time_since_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_overbought_{level}')
            result_df[time_since_feature] = self._calculate_time_since_condition(overbought_condition.values, df.index)
            self.log_feature_creation(time_since_feature, result_df)
            
            # Distance from level
            dist_feature = self.get_feature_name(symbol, f'{base_feature}_dist_overbought_{level}')
            result_df[dist_feature] = (rsi - level).clip(lower=0)
            self.log_feature_creation(dist_feature, result_df)
        
        # Calculate z-scores
        for period in self.zscore_periods:
            zscore_feature = self.get_feature_name(symbol, f'{base_feature}_zscore_{period}')
            result_df[zscore_feature] = self.calculate_zscore(rsi, min_periods=period)
            self.log_feature_creation(zscore_feature, result_df)
            
            change_zscore_feature = self.get_feature_name(symbol, f'{base_feature}_change_zscore_{period}')
            result_df[change_zscore_feature] = self.calculate_zscore(
                result_df[change_feature],
                min_periods=period
            )
            self.log_feature_creation(change_zscore_feature, result_df)
        
        return result_df