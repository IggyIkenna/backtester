from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class PriceLevelsCalculator(FeatureCalculator):
    """
    Calculate Round Number and Key Price Level features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 major_levels: List[float] = [1000, 5000, 10000, 20000, 30000, 40000, 50000],
                 minor_levels: List[float] = [1000, 2000, 3000, 4000, 5000],
                 level_margin: float = 0.01,  # 1% margin around levels
                 interaction_window: int = 20,  # periods to consider for level strength
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.major_levels = np.array(sorted(major_levels))
        self.minor_levels = np.array(sorted(minor_levels))
        self.level_margin = level_margin
        self.interaction_window = interaction_window
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close']
    
    def _find_nearest_levels(self, price: float) -> Dict[str, float]:
        """Find nearest major and minor levels above and below a price"""
        major_up = self.major_levels[self.major_levels > price].min() if any(self.major_levels > price) else np.nan
        major_down = self.major_levels[self.major_levels < price].max() if any(self.major_levels < price) else np.nan
        minor_up = self.minor_levels[self.minor_levels > price].min() if any(self.minor_levels > price) else np.nan
        minor_down = self.minor_levels[self.minor_levels < price].max() if any(self.minor_levels < price) else np.nan
        
        return {
            'major_up': major_up,
            'major_down': major_down,
            'minor_up': minor_up,
            'minor_down': minor_down
        }
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float) -> pd.Series:
        """Calculate historical interaction strength for a price level"""
        margin = level * self.level_margin
        upper_bound = level + margin
        lower_bound = level - margin
        
        # Count touches of the level
        touches = ((df['high'] >= lower_bound) & (df['low'] <= upper_bound)).astype(int)
        
        # Calculate rolling interaction strength
        strength = touches.rolling(window=self.interaction_window).sum()
        
        return strength
    
    def _calculate_level_cluster(self, price: float) -> Dict[str, int]:
        """Calculate level clustering metrics"""
        margin = price * self.level_margin
        
        # Count levels within margin
        major_cluster = np.sum((self.major_levels >= price - margin) & 
                             (self.major_levels <= price + margin))
        minor_cluster = np.sum((self.minor_levels >= price - margin) & 
                             (self.minor_levels <= price + margin))
        
        return {
            'major_cluster': major_cluster,
            'minor_cluster': minor_cluster
        }
    
    def _calculate_dynamic_sr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        # Use local highs and lows
        highs = df['high'].rolling(window=self.interaction_window, center=True).max()
        lows = df['low'].rolling(window=self.interaction_window, center=True).min()
        
        # Calculate strength of levels
        high_strength = self._calculate_level_strength(df, highs)
        low_strength = self._calculate_level_strength(df, lows)
        
        return pd.DataFrame({
            'resistance': highs,
            'support': lows,
            'resistance_strength': high_strength,
            'support_strength': low_strength
        })
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Price Level features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate nearest levels for each price
        levels_data = pd.DataFrame([
            self._find_nearest_levels(price) for price in shifted_data['close']
        ], index=shifted_data.index)
        
        # Store base level values
        for level_type in ['major_up', 'major_down', 'minor_up', 'minor_down']:
            feature = self.get_feature_name(symbol, f'{self.timeframe}_{level_type}')
            result_df[feature] = levels_data[level_type]
            self.log_feature_creation(feature, result_df)
            
            # Calculate distance to level
            dist_feature = self.get_feature_name(symbol, f'{self.timeframe}_{level_type}_dist')
            result_df[dist_feature] = (
                (result_df[feature] - shifted_data['close']) / shifted_data['close'] * 100
            ).round(4)
            self.log_feature_creation(dist_feature, result_df)
        
        # Calculate level strength
        for level_type in ['major_up', 'major_down', 'minor_up', 'minor_down']:
            strength_feature = self.get_feature_name(symbol, f'{self.timeframe}_{level_type}_strength')
            result_df[strength_feature] = self._calculate_level_strength(
                shifted_data,
                levels_data[level_type]
            )
            self.log_feature_creation(strength_feature, result_df)
        
        # Calculate level clustering
        cluster_data = pd.DataFrame([
            self._calculate_level_cluster(price) for price in shifted_data['close']
        ], index=shifted_data.index)
        
        for cluster_type in ['major_cluster', 'minor_cluster']:
            feature = self.get_feature_name(symbol, f'{self.timeframe}_{cluster_type}')
            result_df[feature] = cluster_data[cluster_type]
            self.log_feature_creation(feature, result_df)
        
        # Calculate dynamic S/R levels
        sr_data = self._calculate_dynamic_sr(shifted_data)
        
        for sr_type in ['resistance', 'support']:
            # Base level
            feature = self.get_feature_name(symbol, f'{self.timeframe}_dynamic_{sr_type}')
            result_df[feature] = sr_data[sr_type]
            self.log_feature_creation(feature, result_df)
            
            # Level strength
            strength_feature = self.get_feature_name(symbol, f'{self.timeframe}_dynamic_{sr_type}_strength')
            result_df[strength_feature] = sr_data[f'{sr_type}_strength']
            self.log_feature_creation(strength_feature, result_df)
            
            # Distance to level
            dist_feature = self.get_feature_name(symbol, f'{self.timeframe}_dynamic_{sr_type}_dist')
            result_df[dist_feature] = (
                (result_df[feature] - shifted_data['close']) / shifted_data['close'] * 100
            ).round(4)
            self.log_feature_creation(dist_feature, result_df)
        
        # Calculate z-scores for distances
        for period in self.zscore_periods:
            # Static levels distance z-scores
            for level_type in ['major_up', 'major_down', 'minor_up', 'minor_down']:
                base_feature = self.get_feature_name(symbol, f'{self.timeframe}_{level_type}_dist')
                dist_zscore = self.get_feature_name(symbol, f'{self.timeframe}_{level_type}_dist_zscore_{period}')
                result_df[dist_zscore] = self.calculate_zscore(
                    result_df[base_feature],
                    min_periods=period
                )
                self.log_feature_creation(dist_zscore, result_df)
            
            # Dynamic S/R distance z-scores
            for sr_type in ['resistance', 'support']:
                base_feature = self.get_feature_name(symbol, f'{self.timeframe}_dynamic_{sr_type}_dist')
                dist_zscore = self.get_feature_name(symbol, f'{self.timeframe}_dynamic_{sr_type}_dist_zscore_{period}')
                result_df[dist_zscore] = self.calculate_zscore(
                    result_df[base_feature],
                    min_periods=period
                )
                self.log_feature_creation(dist_zscore, result_df)
        
        self.logger.info(f"Created {len(self.created_features)} features")
        return result_df