from .base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List

class StochasticFeatureCalculator(FeatureCalculator):
    """
    Calculate Stochastic Oscillator features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 k_period: int = 14,
                 d_period: int = 3,
                 smooth_k: int = 3,
                 oversold_levels: List[float] = [20, 30, 40],
                 overbought_levels: List[float] = [60, 70, 80],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.oversold_levels = oversold_levels
        self.overbought_levels = overbought_levels
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close']
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic K and D values using shifted data"""
        # Calculate highest high and lowest low
        high_roll = df['high'].rolling(window=self.k_period).max()
        low_roll = df['low'].rolling(window=self.k_period).min()
        
        # Calculate raw K (0-100)
        k_raw = 100 * (df['close'] - low_roll) / (high_roll - low_roll)
        
        # Smooth K value if specified
        k = k_raw if self.smooth_k <= 1 else k_raw.rolling(window=self.smooth_k).mean()
        
        # Calculate D value (SMA of K)
        d = k.rolling(window=self.d_period).mean()
        
        return pd.DataFrame({
            'k': k,
            'd': d,
            'k_raw': k_raw
        })
    
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
        """Calculate Stochastic features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = f'{self.timeframe}_{self.k_period}_{self.d_period}'
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate base stochastic values
        stoch = self._calculate_stochastic(shifted_data)
        
        # Store base values
        k_feature = self.get_feature_name(symbol, f'{base_feature}_k')
        d_feature = self.get_feature_name(symbol, f'{base_feature}_d')
        result_df[k_feature] = stoch['k']
        result_df[d_feature] = stoch['d']
        self.log_feature_creation(k_feature, result_df)
        self.log_feature_creation(d_feature, result_df)
        
        # Calculate crossovers
        cross_feature = self.get_feature_name(symbol, f'{base_feature}_cross')
        cross_condition = np.where(
            (stoch['k'] > stoch['d']) & (stoch['k'].shift(1) <= stoch['d'].shift(1)), 1,
            np.where(
                (stoch['k'] < stoch['d']) & (stoch['k'].shift(1) >= stoch['d'].shift(1)), -1,
                0
            )
        )
        result_df[cross_feature] = cross_condition
        self.log_feature_creation(cross_feature, result_df)
        
        # Calculate time since last bullish/bearish crossover
        bull_cross = (cross_condition == 1)
        bear_cross = (cross_condition == -1)
        
        bull_cross_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_bull_cross')
        bear_cross_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_bear_cross')
        result_df[bull_cross_feature] = self._calculate_time_since_condition(bull_cross, df.index)
        result_df[bear_cross_feature] = self._calculate_time_since_condition(bear_cross, df.index)
        self.log_feature_creation(bull_cross_feature, result_df)
        self.log_feature_creation(bear_cross_feature, result_df)
        
        # Calculate overbought/oversold conditions
        for level in self.oversold_levels:
            # K value oversold
            k_feature = self.get_feature_name(symbol, f'{base_feature}_k_oversold_{level}')
            k_oversold = (stoch['k'] <= level)
            result_df[k_feature] = k_oversold.astype(int)
            self.log_feature_creation(k_feature, result_df)
            
            # Time since K oversold
            k_time_feature = self.get_feature_name(symbol, f'{base_feature}_k_time_since_oversold_{level}')
            result_df[k_time_feature] = self._calculate_time_since_condition(k_oversold.values, df.index)
            self.log_feature_creation(k_time_feature, result_df)
            
            # D value oversold
            d_feature = self.get_feature_name(symbol, f'{base_feature}_d_oversold_{level}')
            d_oversold = (stoch['d'] <= level)
            result_df[d_feature] = d_oversold.astype(int)
            self.log_feature_creation(d_feature, result_df)
            
            # Time since D oversold
            d_time_feature = self.get_feature_name(symbol, f'{base_feature}_d_time_since_oversold_{level}')
            result_df[d_time_feature] = self._calculate_time_since_condition(d_oversold.values, df.index)
            self.log_feature_creation(d_time_feature, result_df)
            
            # Distance from level
            k_dist = self.get_feature_name(symbol, f'{base_feature}_k_dist_oversold_{level}')
            result_df[k_dist] = (stoch['k'] - level).clip(upper=0)
            self.log_feature_creation(k_dist, result_df)
            
            d_dist = self.get_feature_name(symbol, f'{base_feature}_d_dist_oversold_{level}')
            result_df[d_dist] = (stoch['d'] - level).clip(upper=0)
            self.log_feature_creation(d_dist, result_df)
        
        for level in self.overbought_levels:
            # K value overbought
            k_feature = self.get_feature_name(symbol, f'{base_feature}_k_overbought_{level}')
            k_overbought = (stoch['k'] >= level)
            result_df[k_feature] = k_overbought.astype(int)
            self.log_feature_creation(k_feature, result_df)
            
            # Time since K overbought
            k_time_feature = self.get_feature_name(symbol, f'{base_feature}_k_time_since_overbought_{level}')
            result_df[k_time_feature] = self._calculate_time_since_condition(k_overbought.values, df.index)
            self.log_feature_creation(k_time_feature, result_df)
            
            # D value overbought
            d_feature = self.get_feature_name(symbol, f'{base_feature}_d_overbought_{level}')
            d_overbought = (stoch['d'] >= level)
            result_df[d_feature] = d_overbought.astype(int)
            self.log_feature_creation(d_feature, result_df)
            
            # Time since D overbought
            d_time_feature = self.get_feature_name(symbol, f'{base_feature}_d_time_since_overbought_{level}')
            result_df[d_time_feature] = self._calculate_time_since_condition(d_overbought.values, df.index)
            self.log_feature_creation(d_time_feature, result_df)
            
            # Distance from level
            k_dist = self.get_feature_name(symbol, f'{base_feature}_k_dist_overbought_{level}')
            result_df[k_dist] = (stoch['k'] - level).clip(lower=0)
            self.log_feature_creation(k_dist, result_df)
            
            d_dist = self.get_feature_name(symbol, f'{base_feature}_d_dist_overbought_{level}')
            result_df[d_dist] = (stoch['d'] - level).clip(lower=0)
            self.log_feature_creation(d_dist, result_df)
        
        # Calculate changes
        k_change = self.get_feature_name(symbol, f'{base_feature}_k_change')
        d_change = self.get_feature_name(symbol, f'{base_feature}_d_change')
        result_df[k_change] = stoch['k'].diff()
        result_df[d_change] = stoch['d'].diff()
        self.log_feature_creation(k_change, result_df)
        self.log_feature_creation(d_change, result_df)
        
        # Calculate z-scores
        for period in self.zscore_periods:
            # Z-scores for K and D values
            for value in ['k', 'd']:
                base_value = self.get_feature_name(symbol, f'{base_feature}_{value}')
                zscore_feature = self.get_feature_name(symbol, f'{base_feature}_{value}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    stoch[value],
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
                
                # Z-scores for changes
                change_feature = self.get_feature_name(symbol, f'{base_feature}_{value}_change')
                change_zscore = self.get_feature_name(symbol, f'{base_feature}_{value}_change_zscore_{period}')
                result_df[change_zscore] = self.calculate_zscore(
                    result_df[change_feature],
                    min_periods=period
                )
                self.log_feature_creation(change_zscore, result_df)
        
        return result_df