from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class MACDFeatureCalculator(FeatureCalculator):
    """
    Calculate MACD features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['close']
    
    def _calculate_macd_components(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD components using shifted data"""
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'line': macd_line,
            'signal': signal_line,
            'hist': histogram
        }
    
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
        """Calculate MACD features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = f'{self.timeframe}_{self.fast_period}_{self.slow_period}_{self.signal_period}'
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate MACD components
        macd_data = self._calculate_macd_components(shifted_data)
        
        # Store base values
        for component in ['line', 'signal', 'hist']:
            feature_name = self.get_feature_name(symbol, f'{base_feature}_{component}')
            result_df[feature_name] = macd_data[component]
            self.log_feature_creation(feature_name, result_df)
        
        # Calculate crossovers
        cross_feature = self.get_feature_name(symbol, f'{base_feature}_cross')
        cross_condition = np.where(
            (macd_data['line'] > macd_data['signal']) & 
            (macd_data['line'].shift(1) <= macd_data['signal'].shift(1)), 1,
            np.where(
                (macd_data['line'] < macd_data['signal']) & 
                (macd_data['line'].shift(1) >= macd_data['signal'].shift(1)), -1,
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
        
        # Calculate zero-line crossovers
        zero_cross_feature = self.get_feature_name(symbol, f'{base_feature}_zero_cross')
        zero_cross_condition = np.where(
            (macd_data['line'] > 0) & (macd_data['line'].shift(1) <= 0), 1,
            np.where(
                (macd_data['line'] < 0) & (macd_data['line'].shift(1) >= 0), -1,
                0
            )
        )
        result_df[zero_cross_feature] = zero_cross_condition
        self.log_feature_creation(zero_cross_feature, result_df)
        
        # Calculate time since last zero-line crossover
        zero_bull_cross = (zero_cross_condition == 1)
        zero_bear_cross = (zero_cross_condition == -1)
        
        zero_bull_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_zero_bull_cross')
        zero_bear_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_zero_bear_cross')
        
        result_df[zero_bull_feature] = self._calculate_time_since_condition(zero_bull_cross, df.index)
        result_df[zero_bear_feature] = self._calculate_time_since_condition(zero_bear_cross, df.index)
        self.log_feature_creation(zero_bull_feature, result_df)
        self.log_feature_creation(zero_bear_feature, result_df)
        
        # Calculate changes
        for component in ['line', 'signal', 'hist']:
            change_feature = self.get_feature_name(symbol, f'{base_feature}_{component}_change')
            result_df[change_feature] = macd_data[component].diff()
            self.log_feature_creation(change_feature, result_df)
        
        # Calculate z-scores
        for period in self.zscore_periods:
            for component in ['line', 'signal', 'hist']:
                # Z-score of value
                zscore_feature = self.get_feature_name(symbol, f'{base_feature}_{component}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    macd_data[component],
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
                
                # Z-score of changes
                change_zscore = self.get_feature_name(symbol, f'{base_feature}_{component}_change_zscore_{period}')
                result_df[change_zscore] = self.calculate_zscore(
                    result_df[self.get_feature_name(symbol, f'{base_feature}_{component}_change')],
                    min_periods=period
                )
                self.log_feature_creation(change_zscore, result_df)
        
        return result_df