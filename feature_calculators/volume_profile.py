from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

class VolumeProfileCalculator(FeatureCalculator):
    """
    Calculate Volume Profile features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    Features include Point of Control (POC), Value Area High/Low (VAH/VAL),
    and High/Low Volume Nodes (HVN/LVN).
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 num_bins: int = 100,  # Number of price bins for volume distribution
                 value_area_pct: float = 0.70,  # Value Area percentage (typically 70%)
                 hvn_threshold: float = 1.5,  # Threshold for High Volume Node (relative to mean)
                 lvn_threshold: float = 0.5,  # Threshold for Low Volume Node (relative to mean)
                 window_size: int = 24,  # Rolling window for profile calculation
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        self.window_size = window_size
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close', 'quote_volume']
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        """Calculate volume profile for a given period using vectorized operations"""
        # Calculate price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Create weighted price points
        # We'll use the VWAP concept: split each candle into typical price points
        typical_prices = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate volume profile using numpy histogram
        volume_profile, _ = np.histogram(
            typical_prices,
            bins=price_bins,
            weights=df['quote_volume']
        )
        
        # Smooth the profile using a simple moving average
        window = 3  # Small window for smoothing
        volume_profile = np.convolve(volume_profile, np.ones(window)/window, mode='same')
        
        return pd.Series(volume_profile, index=bin_centers), price_bins
    
    def _find_value_area(self, 
                        volume_profile: pd.Series, 
                        poc_price: float) -> Tuple[float, float]:
        """Find Value Area High and Low prices using vectorized operations"""
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.value_area_pct
        
        # Sort volumes in descending order
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum_volume = sorted_profile.cumsum()
        
        # Find all price levels within the value area
        value_area_prices = sorted_profile[cumsum_volume <= target_volume].index
        
        # Get VAH and VAL
        val_price = value_area_prices.min()
        vah_price = value_area_prices.max()
        
        return val_price, vah_price
    
    def _find_volume_nodes(self, 
                          volume_profile: pd.Series) -> Tuple[List[float], List[float]]:
        """Find High and Low Volume Nodes using vectorized operations"""
        mean_volume = volume_profile.mean()
        
        # Find HVNs and LVNs using boolean indexing
        hvn_mask = volume_profile >= (mean_volume * self.hvn_threshold)
        lvn_mask = volume_profile <= (mean_volume * self.lvn_threshold)
        
        # Sort by volume to get the most significant nodes first
        hvn_prices = volume_profile[hvn_mask].sort_values(ascending=False).index.tolist()
        lvn_prices = volume_profile[lvn_mask].sort_values().index.tolist()
        
        return hvn_prices, lvn_prices
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Volume Profile features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = self.timeframe
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Pre-allocate arrays for features
        num_periods = len(shifted_data)
        result_df[self.get_feature_name(symbol, f'{base_feature}_poc')] = np.nan
        result_df[self.get_feature_name(symbol, f'{base_feature}_vah')] = np.nan
        result_df[self.get_feature_name(symbol, f'{base_feature}_val')] = np.nan
        for i in range(3):
            result_df[self.get_feature_name(symbol, f'{base_feature}_hvn_{i+1}')] = np.nan
            result_df[self.get_feature_name(symbol, f'{base_feature}_lvn_{i+1}')] = np.nan
        
        # Calculate rolling volume profiles using a sliding window
        for i in range(self.window_size-1, num_periods):
            window_data = shifted_data.iloc[i-self.window_size+1:i+1]
            
            # Calculate volume profile for the window
            volume_profile, _ = self._calculate_volume_profile(window_data)
            
            # Find Point of Control (price level with highest volume)
            poc_price = volume_profile.idxmax()
            
            # Find Value Area
            val_price, vah_price = self._find_value_area(volume_profile, poc_price)
            
            # Find Volume Nodes
            hvn_prices, lvn_prices = self._find_volume_nodes(volume_profile)
            
            # Store results
            idx = shifted_data.index[i]
            result_df.loc[idx, self.get_feature_name(symbol, f'{base_feature}_poc')] = poc_price
            result_df.loc[idx, self.get_feature_name(symbol, f'{base_feature}_vah')] = vah_price
            result_df.loc[idx, self.get_feature_name(symbol, f'{base_feature}_val')] = val_price
            
            # Store top 3 HVNs and LVNs
            for j in range(min(3, len(hvn_prices))):
                result_df.loc[idx, self.get_feature_name(symbol, f'{base_feature}_hvn_{j+1}')] = hvn_prices[j]
            
            for j in range(min(3, len(lvn_prices))):
                result_df.loc[idx, self.get_feature_name(symbol, f'{base_feature}_lvn_{j+1}')] = lvn_prices[j]
        
        # Log feature creation for base features
        base_features = ['poc', 'vah', 'val']
        for feature in base_features:
            feature_name = self.get_feature_name(symbol, f'{base_feature}_{feature}')
            self.log_feature_creation(feature_name, result_df)
        
        # Log feature creation for volume nodes
        for i in range(3):
            hvn_feature = self.get_feature_name(symbol, f'{base_feature}_hvn_{i+1}')
            lvn_feature = self.get_feature_name(symbol, f'{base_feature}_lvn_{i+1}')
            self.log_feature_creation(hvn_feature, result_df)
            self.log_feature_creation(lvn_feature, result_df)
        
        # Vectorized calculation of distances from current price
        close_price = shifted_data['close'].values.reshape(-1, 1)
        for feature in base_features + [f'hvn_{i+1}' for i in range(3)] + [f'lvn_{i+1}' for i in range(3)]:
            base_feature_name = self.get_feature_name(symbol, f'{base_feature}_{feature}')
            feature_values = result_df[base_feature_name].values
            dist_feature = self.get_feature_name(symbol, f'{base_feature}_{feature}_dist')
            result_df[dist_feature] = ((feature_values - close_price.flatten()) / close_price.flatten() * 100).round(4)
            self.log_feature_creation(dist_feature, result_df)
        
        # Vectorized calculation of Value Area width
        width_feature = self.get_feature_name(symbol, f'{base_feature}_va_width')
        vah_feature = self.get_feature_name(symbol, f'{base_feature}_vah')
        val_feature = self.get_feature_name(symbol, f'{base_feature}_val')
        result_df[width_feature] = (
            (result_df[vah_feature] - result_df[val_feature]) /
            shifted_data['close'] * 100
        ).round(4)
        self.log_feature_creation(width_feature, result_df)
        
        # Calculate z-scores
        for period in self.zscore_periods:
            # Distance z-scores
            for feature in base_features + [f'hvn_{i+1}' for i in range(3)] + [f'lvn_{i+1}' for i in range(3)]:
                dist_feature = self.get_feature_name(symbol, f'{base_feature}_{feature}_dist')
                dist_zscore = self.get_feature_name(symbol, f'{base_feature}_{feature}_dist_zscore_{period}')
                result_df[dist_zscore] = self.calculate_zscore(
                    result_df[dist_feature],
                    min_periods=period
                )
                self.log_feature_creation(dist_zscore, result_df)
            
            # Value Area width z-score
            width_zscore = self.get_feature_name(symbol, f'{base_feature}_va_width_zscore_{period}')
            result_df[width_zscore] = self.calculate_zscore(
                result_df[width_feature],
                min_periods=period
            )
            self.log_feature_creation(width_zscore, result_df)
        
        return result_df