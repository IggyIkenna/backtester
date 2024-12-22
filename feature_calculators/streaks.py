from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

class StreaksCalculator(FeatureCalculator):
    """
    Calculate Consecutive Movement Analysis features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    Features include up/down streaks, volatility streaks, and volume streaks.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 volatility_threshold: float = 0.5,  # % threshold for high volatility
                 volume_threshold: float = 1.5,  # multiplier over average for high volume
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close', 'quote_volume']
    
    def _calculate_streaks(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive streaks (positive numbers for up, negative for down)"""
        # Initialize streak series
        streaks = pd.Series(0, index=series.index)
        
        # Calculate signs of changes
        signs = np.sign(series)
        
        # Calculate streaks
        current_streak = 0
        for i in range(len(series)):
            if signs.iloc[i] == 0:
                # No change
                streaks.iloc[i] = current_streak
            else:
                if current_streak * signs.iloc[i] >= 0:
                    # Continuing streak
                    current_streak += signs.iloc[i]
                else:
                    # New streak
                    current_streak = signs.iloc[i]
                streaks.iloc[i] = current_streak
        
        return streaks
    
    def _calculate_volatility_streaks(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate high/low volatility streaks"""
        # Calculate true range
        tr = pd.Series(0.0, index=df.index)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Calculate average true range
        atr = tr.rolling(window=window).mean()
        
        # Identify high volatility periods
        high_vol = (tr > atr * (1 + self.volatility_threshold)).astype(int)
        
        # Calculate streaks
        return self._calculate_streaks(high_vol)
    
    def _calculate_volume_streaks(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate high/low volume streaks"""
        # Calculate average volume
        avg_volume = df['quote_volume'].rolling(window=window).mean()
        
        # Identify high volume periods
        high_vol = (df['quote_volume'] > avg_volume * self.volume_threshold).astype(int)
        
        # Calculate streaks
        return self._calculate_streaks(high_vol)
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Streak features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate price change streaks
        returns = shifted_data['close'].pct_change()
        price_streaks = self._calculate_streaks(returns)
        
        # Store price streaks
        price_feature = self.get_feature_name(symbol, f'{self.timeframe}_price')
        result_df[price_feature] = price_streaks
        self.log_feature_creation(price_feature, result_df)
        
        # Calculate absolute streak lengths
        length_feature = self.get_feature_name(symbol, f'{self.timeframe}_price_length')
        result_df[length_feature] = abs(price_streaks)
        self.log_feature_creation(length_feature, result_df)
        
        # Calculate streak returns
        streak_returns = pd.Series(0.0, index=df.index)
        current_streak_start = 0
        for i in range(1, len(price_streaks)):
            if abs(price_streaks.iloc[i]) < abs(price_streaks.iloc[i-1]):
                # Streak ended, calculate return
                streak_returns.iloc[i-1] = (
                    shifted_data['close'].iloc[i-1] / 
                    shifted_data['close'].iloc[current_streak_start] - 1
                ) * 100
                current_streak_start = i
        
        returns_feature = self.get_feature_name(symbol, f'{self.timeframe}_returns')
        result_df[returns_feature] = streak_returns
        self.log_feature_creation(returns_feature, result_df)
        
        # Calculate volatility streaks for different windows
        for window in [20, 50, 100]:
            vol_streaks = self._calculate_volatility_streaks(shifted_data, window)
            
            # Store volatility streaks
            vol_feature = self.get_feature_name(symbol, f'{self.timeframe}_volatility_{window}')
            result_df[vol_feature] = vol_streaks
            self.log_feature_creation(vol_feature, result_df)
            
            # Store absolute streak lengths
            length_feature = self.get_feature_name(symbol, f'{self.timeframe}_volatility_{window}_length')
            result_df[length_feature] = abs(vol_streaks)
            self.log_feature_creation(length_feature, result_df)
        
        # Calculate volume streaks for different windows
        for window in [20, 50, 100]:
            vol_streaks = self._calculate_volume_streaks(shifted_data, window)
            
            # Store volume streaks
            vol_feature = self.get_feature_name(symbol, f'{self.timeframe}_volume_{window}')
            result_df[vol_feature] = vol_streaks
            self.log_feature_creation(vol_feature, result_df)
            
            # Store absolute streak lengths
            length_feature = self.get_feature_name(symbol, f'{self.timeframe}_volume_{window}_length')
            result_df[length_feature] = abs(vol_streaks)
            self.log_feature_creation(length_feature, result_df)
        
        # Calculate combined signals
        for window in [20, 50, 100]:
            vol_streaks = result_df[self.get_feature_name(symbol, f'{self.timeframe}_volatility_{window}')]
            volume_streaks = result_df[self.get_feature_name(symbol, f'{self.timeframe}_volume_{window}')]
            
            # High volatility and high volume streaks
            combined_feature = self.get_feature_name(symbol, f'{self.timeframe}_combined_{window}')
            result_df[combined_feature] = np.where(
                (vol_streaks > 0) & (volume_streaks > 0), 1,
                np.where((vol_streaks < 0) & (volume_streaks < 0), -1, 0)
            )
            self.log_feature_creation(combined_feature, result_df)
        
        # Calculate z-scores
        for zscore_period in self.zscore_periods:
            # Price streak z-scores
            base_length_feature = self.get_feature_name(symbol, f'{self.timeframe}_price_length')
            zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_price_length_zscore_{zscore_period}')
            result_df[zscore_feature] = self.calculate_zscore(
                result_df[base_length_feature],
                min_periods=zscore_period
            )
            self.log_feature_creation(zscore_feature, result_df)
            
            # Returns z-scores
            base_returns_feature = self.get_feature_name(symbol, f'{self.timeframe}_returns')
            returns_zscore = self.get_feature_name(symbol, f'{self.timeframe}_returns_zscore_{zscore_period}')
            result_df[returns_zscore] = self.calculate_zscore(
                result_df[base_returns_feature],
                min_periods=zscore_period
            )
            self.log_feature_creation(returns_zscore, result_df)
            
            # Volatility and volume streak z-scores
            for window in [20, 50, 100]:
                for streak_type in ['volatility', 'volume']:
                    base_length_feature = self.get_feature_name(symbol, f'{self.timeframe}_{streak_type}_{window}_length')
                    length_zscore = self.get_feature_name(symbol, f'{self.timeframe}_{streak_type}_{window}_length_zscore_{zscore_period}')
                    result_df[length_zscore] = self.calculate_zscore(
                        result_df[base_length_feature],
                        min_periods=zscore_period
                    )
                    self.log_feature_creation(length_zscore, result_df)
        
        self.logger.info(f"Created {len(self.created_features)} features")
        return result_df