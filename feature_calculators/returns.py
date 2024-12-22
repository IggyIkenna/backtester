from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ReturnFeatureCalculator(FeatureCalculator):
    """
    Calculate return-based features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 lags: List[int] = [1, 2, 3, 4, 5, 10, 20, 50],
                 vol_windows: List[int] = [5, 10, 20, 50],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.lags = sorted(lags)
        self.vol_windows = sorted(vol_windows)
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['close']
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate return features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = self.timeframe
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate log returns for each lag
        for lag in self.lags:
            # Log returns
            log_returns = np.log(shifted_data['close'] / shifted_data['close'].shift(lag))
            log_feature = self.get_feature_name(symbol, f'{base_feature}_log_{lag}')
            result_df[log_feature] = log_returns
            self.log_feature_creation(log_feature, result_df)
            
            # Percentage returns
            pct_returns = (shifted_data['close'] / shifted_data['close'].shift(lag) - 1) * 100
            pct_feature = self.get_feature_name(symbol, f'{base_feature}_pct_{lag}')
            result_df[pct_feature] = pct_returns.round(4)
            self.log_feature_creation(pct_feature, result_df)
            
            # Calculate z-scores for both types of returns
            for period in self.zscore_periods:
                # Log returns z-score
                log_zscore = self.get_feature_name(symbol, f'{base_feature}_log_{lag}_zscore_{period}')
                result_df[log_zscore] = self.calculate_zscore(
                    log_returns,
                    min_periods=period
                )
                self.log_feature_creation(log_zscore, result_df)
                
                # Percentage returns z-score
                pct_zscore = self.get_feature_name(symbol, f'{base_feature}_pct_{lag}_zscore_{period}')
                result_df[pct_zscore] = self.calculate_zscore(
                    pct_returns,
                    min_periods=period
                )
                self.log_feature_creation(pct_zscore, result_df)
        
        # Calculate return volatility for different windows
        log_returns_1p = np.log(shifted_data['close'] / shifted_data['close'].shift(1))
        
        for window in self.vol_windows:
            # Rolling standard deviation of returns
            vol = log_returns_1p.rolling(window=window).std() * np.sqrt(252 * 24)  # Annualized
            vol_feature = self.get_feature_name(symbol, f'{base_feature}_vol_{window}')
            result_df[vol_feature] = vol
            self.log_feature_creation(vol_feature, result_df)
            
            # Calculate z-scores for volatility
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{base_feature}_vol_{window}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    vol,
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # Calculate cumulative returns
        for window in self.vol_windows:
            # Cumulative log returns
            cum_log = log_returns_1p.rolling(window=window).sum()
            log_feature = self.get_feature_name(symbol, f'{base_feature}_cum_log_{window}')
            result_df[log_feature] = cum_log
            self.log_feature_creation(log_feature, result_df)
            
            # Cumulative percentage returns
            cum_pct = ((1 + log_returns_1p).rolling(window=window).apply(np.prod) - 1) * 100
            pct_feature = self.get_feature_name(symbol, f'{base_feature}_cum_pct_{window}')
            result_df[pct_feature] = cum_pct.round(4)
            self.log_feature_creation(pct_feature, result_df)
            
            # Calculate z-scores for cumulative returns
            for period in self.zscore_periods:
                # Log cumulative z-score
                log_zscore = self.get_feature_name(symbol, f'{base_feature}_cum_log_{window}_zscore_{period}')
                result_df[log_zscore] = self.calculate_zscore(
                    cum_log,
                    min_periods=period
                )
                self.log_feature_creation(log_zscore, result_df)
                
                # Percentage cumulative z-score
                pct_zscore = self.get_feature_name(symbol, f'{base_feature}_cum_pct_{window}_zscore_{period}')
                result_df[pct_zscore] = self.calculate_zscore(
                    cum_pct,
                    min_periods=period
                )
                self.log_feature_creation(pct_zscore, result_df)
        
        return result_df