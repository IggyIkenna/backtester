from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from backtester.feature_calculators.base import FeatureCalculator
import logging
import time

class FundingRateCalculator(FeatureCalculator):
    """Calculate funding rate based features"""
    
    def __init__(self, timeframe: str = "1h", windows: Optional[List[int]] = None, zscore_periods: Optional[List[int]] = None):
        """Initialize the calculator"""
        super().__init__(timeframe)
        # Default windows: 1d, 3d, 1w, 1m, 3m, 1y
        self.lookback_periods = windows or [
            24,     # 1 day
            72,     # 3 days
            168,    # 1 week
            720,    # 1 month (30 days)
            2160,   # 3 months (90 days)
            8760    # 1 year (365 days)
        ]
        # Default zscore periods: 1w, 1m, 3m, 1y
        self.zscore_periods = zscore_periods or [
            168,    # 1 week
            720,    # 1 month
            2160,   # 3 months
            8760    # 1 year
        ]
        
    def get_required_columns(self) -> list:
        return ['fundingRate']
        
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate funding rate features with performance logging"""
        start_time = time.time()
        logging.info(f"Starting funding rate calculations for {symbol}, shape: {df.shape}")
        
        result_df = super().calculate(df, symbol)
        self.symbol = symbol
        self.created_features = []
        
        result_df = result_df.copy()
        logging.info(f"Data copy created in {time.time() - start_time:.2f}s")
        
        # Pre-calculate rolling windows to avoid redundant computations
        calc_start = time.time()
        funding_rate = df['fundingRate']
        logging.info(f"Starting calculations for {len(self.lookback_periods)} periods")
        
        for period in self.lookback_periods:
            period_start = time.time()
            logging.info(f"Processing {period}h period")
            
            # Pre-calculate common rolling operations
            roll = funding_rate.rolling(period)
            roll_mean = roll.mean()
            roll_std = roll.std()
            
            # Mean funding rate (annualized)
            mean_feature = self.get_feature_name(f'funding_rate_mean_{period}h')
            result_df[mean_feature] = roll_mean * (24 * 365)
            self.log_feature_creation(mean_feature, result_df)
            
            # Std of funding rate
            std_feature = self.get_feature_name(f'funding_rate_std_{period}h')
            result_df[std_feature] = roll_std
            self.log_feature_creation(std_feature, result_df)
            
            # Z-score of current funding rate
            zscore_feature = self.get_feature_name(f'funding_rate_zscore_{period}h')
            result_df[zscore_feature] = (funding_rate - roll_mean) / (roll_std + 1e-10)
            self.log_feature_creation(zscore_feature, result_df)
            
            # Skew of funding rate
            skew_feature = self.get_feature_name(f'funding_rate_skew_{period}h')
            result_df[skew_feature] = roll.skew()
            self.log_feature_creation(skew_feature, result_df)
            
            # Positive ratio (vectorized)
            pos_ratio_feature = self.get_feature_name(f'funding_rate_pos_ratio_{period}h')
            result_df[pos_ratio_feature] = roll.apply(lambda x: (x > 0).mean())
            self.log_feature_creation(pos_ratio_feature, result_df)
            
            logging.info(f"Period {period}h completed in {time.time() - period_start:.2f}s")
        
        logging.info(f"All calculations completed in {time.time() - calc_start:.2f}s")
        logging.info(f"Created {len(self.created_features)} features")
        
        total_time = time.time() - start_time
        logging.info(f"Total funding rate calculation time: {total_time:.2f}s")
        
        return result_df
    
    def get_feature_name(self, feature: str) -> str:
        """Get standardized feature name"""
        return f"{self.symbol}_{feature}" if self.symbol else feature