from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

class VolumeMomentumCalculator(FeatureCalculator):
    """
    Calculate Volume Momentum features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    Features include volume momentum indicators and their z-scores.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 momentum_periods: List[int] = [5, 10, 20, 50],
                 volume_ratio_periods: List[int] = [24, 72, 168],  # 1d, 3d, 1w
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.momentum_periods = momentum_periods
        self.volume_ratio_periods = volume_ratio_periods
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_required_columns(self) -> list:
        return ['quote_volume', 'taker_buy_quote_volume']
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various volume momentum metrics"""
        results = {}
        
        # Calculate buy/sell volume ratio
        buy_volume = df['taker_buy_quote_volume']
        total_volume = df['quote_volume']
        buy_ratio = buy_volume / total_volume
        
        # Volume momentum (rate of change)
        for period in self.momentum_periods:
            # Volume momentum
            vol_momentum = total_volume.pct_change(periods=period)
            results[f'volume_momentum_{period}'] = vol_momentum
            
            # Buy ratio momentum
            buy_ratio_momentum = buy_ratio.pct_change(periods=period)
            results[f'buy_ratio_momentum_{period}'] = buy_ratio_momentum
            
            # Cumulative volume momentum
            cum_vol_momentum = total_volume.rolling(window=period).sum().pct_change()
            results[f'cum_volume_momentum_{period}'] = cum_vol_momentum
        
        # Volume ratios relative to moving averages
        for period in self.volume_ratio_periods:
            vol_ma = total_volume.rolling(window=period).mean()
            vol_ratio = total_volume / vol_ma
            results[f'volume_ratio_{period}'] = vol_ratio
            
            # Buy volume ratios
            buy_vol_ma = buy_volume.rolling(window=period).mean()
            buy_vol_ratio = buy_volume / buy_vol_ma
            results[f'buy_volume_ratio_{period}'] = buy_vol_ratio
        
        return results
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Volume Momentum features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate momentum metrics
        momentum_metrics = self._calculate_volume_momentum(shifted_data)
        
        # Store base momentum metrics
        for metric_name, metric_values in momentum_metrics.items():
            feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}')
            result_df[feature] = metric_values
            self.log_feature_creation(feature, result_df)
            
            # Calculate z-scores
            for zscore_period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_zscore_{zscore_period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    metric_values,
                    min_periods=zscore_period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        self.logger.info(f"Created {len(self.created_features)} features")
        return result_df 