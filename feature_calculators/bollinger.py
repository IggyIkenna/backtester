from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class BollingerBandsCalculator(FeatureCalculator):
    """
    Calculate Bollinger Bands features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self,
                 timeframe: str = "1h",
                 period: int = 20,
                 std_levels: List[float] = [1, 2, 3, 4],
                 atr_levels: List[float] = [1.8, 2.0, 2.5, 3.0],
                 ivol_levels: List[float] = [1.8, 2.0, 2.5, 3.0],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.period = period
        self.std_levels = sorted(std_levels)
        self.atr_levels = sorted(atr_levels)
        self.ivol_levels = sorted(ivol_levels)
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close']
    
    def _calculate_time_since_condition(self, condition_array: np.ndarray, index: pd.Index) -> pd.Series:
        """Calculate time periods since a condition was last true"""
        time_since = pd.Series(index=index, dtype=float)
        last_true_idx = None
        
        for i in range(len(condition_array)):
            if condition_array[i]:
                last_true_idx = i
            if last_true_idx is not None:
                time_since.iloc[i] = i - last_true_idx
                
        return time_since
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Bollinger Bands features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = f'{self.timeframe}_{self.period}'
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate base metrics using shifted data
        bb_mean = shifted_data['close'].rolling(
            window=self.period,
            min_periods=self.period
        ).mean()
        
        rolling_std = shifted_data['close'].rolling(
            window=self.period,
            min_periods=self.period
        ).std()
        
        # Calculate True Range and ATR
        tr = pd.concat([
            shifted_data['high'] - shifted_data['low'],
            abs(shifted_data['high'] - shifted_data['close'].shift(1)),
            abs(shifted_data['low'] - shifted_data['close'].shift(1))
        ], axis=1).max(axis=1)
        
        rolling_atr = tr.rolling(
            window=self.period,
            min_periods=self.period
        ).mean()
        
        # Store base mean
        mean_feature = self.get_feature_name(symbol, f'{base_feature}_mean')
        result_df[mean_feature] = bb_mean
        self.log_feature_creation(mean_feature, result_df)
        
        # Calculate bands for each standard deviation level
        for std_level in self.std_levels:
            level_str = str(std_level).replace('.', '')
            
            # Calculate bands
            bb_upper = bb_mean + (rolling_std * std_level)
            bb_lower = bb_mean - (rolling_std * std_level)
            
            # Store bands
            upper_feature = self.get_feature_name(symbol, f'{base_feature}_upper_{level_str}')
            lower_feature = self.get_feature_name(symbol, f'{base_feature}_lower_{level_str}')
            result_df[upper_feature] = bb_upper
            result_df[lower_feature] = bb_lower
            self.log_feature_creation(upper_feature, result_df)
            self.log_feature_creation(lower_feature, result_df)
            
            # Calculate band touches
            upper_touch = (shifted_data['high'] >= bb_upper)
            lower_touch = (shifted_data['low'] <= bb_lower)
            
            upper_touch_feature = self.get_feature_name(symbol, f'{base_feature}_upper_touch_{level_str}')
            lower_touch_feature = self.get_feature_name(symbol, f'{base_feature}_lower_touch_{level_str}')
            result_df[upper_touch_feature] = upper_touch.astype(int)
            result_df[lower_touch_feature] = lower_touch.astype(int)
            self.log_feature_creation(upper_touch_feature, result_df)
            self.log_feature_creation(lower_touch_feature, result_df)
            
            # Calculate time since band touches
            upper_time_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_upper_touch_{level_str}')
            lower_time_feature = self.get_feature_name(symbol, f'{base_feature}_time_since_lower_touch_{level_str}')
            result_df[upper_time_feature] = self._calculate_time_since_condition(upper_touch.values, df.index)
            result_df[lower_time_feature] = self._calculate_time_since_condition(lower_touch.values, df.index)
            self.log_feature_creation(upper_time_feature, result_df)
            self.log_feature_creation(lower_time_feature, result_df)
            
            # Calculate band width
            width_feature = self.get_feature_name(symbol, f'{base_feature}_width_{level_str}')
            band_width = (bb_upper - bb_lower) / bb_mean * 100
            result_df[width_feature] = band_width
            self.log_feature_creation(width_feature, result_df)
            
            # Calculate position within bands
            position_feature = self.get_feature_name(symbol, f'{base_feature}_position_{level_str}')
            position = (shifted_data['close'] - bb_lower) / (bb_upper - bb_lower) * 100
            result_df[position_feature] = position.round(2)
            self.log_feature_creation(position_feature, result_df)
            
            # Calculate distances from bands
            upper_dist = ((bb_upper - shifted_data['close']) / shifted_data['close'] * 100).round(4)
            lower_dist = ((shifted_data['close'] - bb_lower) / shifted_data['close'] * 100).round(4)
            
            upper_dist_feature = self.get_feature_name(symbol, f'{base_feature}_upper_dist_{level_str}')
            lower_dist_feature = self.get_feature_name(symbol, f'{base_feature}_lower_dist_{level_str}')
            result_df[upper_dist_feature] = upper_dist
            result_df[lower_dist_feature] = lower_dist
            self.log_feature_creation(upper_dist_feature, result_df)
            self.log_feature_creation(lower_dist_feature, result_df)
        
        # Calculate ATR-based features
        atr_feature = self.get_feature_name(symbol, f'{base_feature}_atr')
        atr_pct_feature = self.get_feature_name(symbol, f'{base_feature}_atr_pct')
        result_df[atr_feature] = rolling_atr
        result_df[atr_pct_feature] = (rolling_atr / shifted_data['close'] * 100).round(4)
        self.log_feature_creation(atr_feature, result_df)
        self.log_feature_creation(atr_pct_feature, result_df)
        
        # Calculate ATR bands
        for atr_level in self.atr_levels:
            level_str = str(atr_level).replace('.', '')
            
            # Calculate ATR bands
            atr_upper = bb_mean + (rolling_atr * atr_level)
            atr_lower = bb_mean - (rolling_atr * atr_level)
            
            # Store ATR bands
            atr_upper_feature = self.get_feature_name(symbol, f'{base_feature}_atr_upper_{level_str}')
            atr_lower_feature = self.get_feature_name(symbol, f'{base_feature}_atr_lower_{level_str}')
            result_df[atr_upper_feature] = atr_upper
            result_df[atr_lower_feature] = atr_lower
            self.log_feature_creation(atr_upper_feature, result_df)
            self.log_feature_creation(atr_lower_feature, result_df)
            
            # Calculate ATR band positions
            atr_position = (shifted_data['close'] - atr_lower) / (atr_upper - atr_lower) * 100
            atr_pos_feature = self.get_feature_name(symbol, f'{base_feature}_atr_position_{level_str}')
            result_df[atr_pos_feature] = atr_position.round(2)
            self.log_feature_creation(atr_pos_feature, result_df)
        
        # Calculate IVOL (Implied Volatility) bands
        rolling_ivol = rolling_std / bb_mean * np.sqrt(252 * 24)  # Annualized hourly volatility
        
        for ivol_level in self.ivol_levels:
            level_str = str(ivol_level).replace('.', '')
            
            # Calculate IVOL bands
            ivol_upper = bb_mean * (1 + rolling_ivol * ivol_level)
            ivol_lower = bb_mean * (1 - rolling_ivol * ivol_level)
            
            # Store IVOL bands
            ivol_upper_feature = self.get_feature_name(symbol, f'{base_feature}_ivol_upper_{level_str}')
            ivol_lower_feature = self.get_feature_name(symbol, f'{base_feature}_ivol_lower_{level_str}')
            result_df[ivol_upper_feature] = ivol_upper
            result_df[ivol_lower_feature] = ivol_lower
            self.log_feature_creation(ivol_upper_feature, result_df)
            self.log_feature_creation(ivol_lower_feature, result_df)
            
            # Calculate IVOL band positions
            ivol_position = (shifted_data['close'] - ivol_lower) / (ivol_upper - ivol_lower) * 100
            ivol_pos_feature = self.get_feature_name(symbol, f'{base_feature}_ivol_position_{level_str}')
            result_df[ivol_pos_feature] = ivol_position.round(2)
            self.log_feature_creation(ivol_pos_feature, result_df)
        
        # Calculate z-scores
        for period in self.zscore_periods:
            # Width z-score for each std level
            for std_level in self.std_levels:
                level_str = str(std_level).replace('.', '')
                width_feature = self.get_feature_name(symbol, f'{base_feature}_width_{level_str}')
                width_zscore = self.get_feature_name(symbol, f'{base_feature}_width_{level_str}_zscore_{period}')
                result_df[width_zscore] = self.calculate_zscore(
                    result_df[width_feature],
                    min_periods=period
                )
                self.log_feature_creation(width_zscore, result_df)
                
                # Position z-score
                pos_feature = self.get_feature_name(symbol, f'{base_feature}_position_{level_str}')
                pos_zscore = self.get_feature_name(symbol, f'{base_feature}_position_{level_str}_zscore_{period}')
                result_df[pos_zscore] = self.calculate_zscore(
                    result_df[pos_feature],
                    min_periods=period
                )
                self.log_feature_creation(pos_zscore, result_df)
            
            # ATR z-score
            atr_zscore = self.get_feature_name(symbol, f'{base_feature}_atr_zscore_{period}')
            result_df[atr_zscore] = self.calculate_zscore(
                rolling_atr,
                min_periods=period
            )
            self.log_feature_creation(atr_zscore, result_df)
            
            # IVOL z-score
            ivol_zscore = self.get_feature_name(symbol, f'{base_feature}_ivol_zscore_{period}')
            result_df[ivol_zscore] = self.calculate_zscore(
                rolling_ivol,
                min_periods=period
            )
            self.log_feature_creation(ivol_zscore, result_df)
        
        return result_df