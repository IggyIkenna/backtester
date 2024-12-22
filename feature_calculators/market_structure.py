from feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

class MarketStructureCalculator(FeatureCalculator):
    """
    Calculate Market Structure features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    Features include Higher Highs (HH), Lower Lows (LL), Higher Lows (HL),
    Lower Highs (LH), and structure breaks.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 swing_periods: List[int] = [5, 10, 20, 50],  # Periods to identify swing points
                 strength_threshold: List[float] = [1.5, 2.0, 2.5],  # Thresholds for swing strength
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.swing_periods = swing_periods
        self.strength_threshold = strength_threshold
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close']
    
    def _find_swing_points(self, 
                          df: pd.DataFrame, 
                          period: int) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows using local maxima/minima"""
        # Calculate local maxima and minima
        high_series = df['high']
        low_series = df['low']
        
        # Find swing highs (local maxima)
        swing_highs = pd.Series(False, index=df.index)
        for i in range(period, len(df) - period):
            if high_series.iloc[i] == high_series.iloc[i-period:i+period+1].max():
                swing_highs.iloc[i] = True
        
        # Find swing lows (local minima)
        swing_lows = pd.Series(False, index=df.index)
        for i in range(period, len(df) - period):
            if low_series.iloc[i] == low_series.iloc[i-period:i+period+1].min():
                swing_lows.iloc[i] = True
        
        return swing_highs, swing_lows
    
    def _calculate_swing_strength(self, 
                                df: pd.DataFrame, 
                                swing_points: pd.Series,
                                period: int,
                                is_high: bool = True) -> pd.Series:
        """Calculate the strength of swing points relative to surrounding price action"""
        strength = pd.Series(0.0, index=df.index)
        price_series = df['high'] if is_high else df['low']
        
        for i in range(period, len(df) - period):
            if swing_points.iloc[i]:
                # Calculate average range of surrounding periods
                surrounding_range = (
                    df['high'].iloc[i-period:i+period+1].max() -
                    df['low'].iloc[i-period:i+period+1].min()
                )
                
                if surrounding_range > 0:
                    if is_high:
                        # For swing highs, compare to surrounding highs
                        diff = price_series.iloc[i] - price_series.iloc[i-period:i+period+1].mean()
                    else:
                        # For swing lows, compare to surrounding lows
                        diff = price_series.iloc[i-period:i+period+1].mean() - price_series.iloc[i]
                    
                    strength.iloc[i] = (diff / surrounding_range * 100).round(2)
        
        return strength
    
    def _identify_structure(self, 
                          df: pd.DataFrame,
                          swing_highs: pd.Series,
                          swing_lows: pd.Series,
                          period: int) -> Dict[str, pd.Series]:
        """Identify market structure patterns (HH, LL, HL, LH)"""
        # Initialize result series
        hh = pd.Series(False, index=df.index)  # Higher High
        ll = pd.Series(False, index=df.index)  # Lower Low
        hl = pd.Series(False, index=df.index)  # Higher Low
        lh = pd.Series(False, index=df.index)  # Lower High
        
        # Find previous swing points for comparison
        for i in range(period * 2, len(df)):
            if swing_highs.iloc[i]:
                # Compare with previous swing high
                prev_highs = df['high'][
                    swing_highs & 
                    (df.index < df.index[i]) & 
                    (df.index >= df.index[i - period * 2])
                ]
                
                if len(prev_highs) > 0:
                    hh.iloc[i] = df['high'].iloc[i] > prev_highs.max()
                    lh.iloc[i] = df['high'].iloc[i] < prev_highs.min()
            
            if swing_lows.iloc[i]:
                # Compare with previous swing low
                prev_lows = df['low'][
                    swing_lows & 
                    (df.index < df.index[i]) & 
                    (df.index >= df.index[i - period * 2])
                ]
                
                if len(prev_lows) > 0:
                    ll.iloc[i] = df['low'].iloc[i] < prev_lows.min()
                    hl.iloc[i] = df['low'].iloc[i] > prev_lows.max()
        
        return {
            'hh': hh,
            'll': ll,
            'hl': hl,
            'lh': lh
        }
    
    def _detect_structure_breaks(self,
                               df: pd.DataFrame,
                               structure: Dict[str, pd.Series],
                               period: int) -> pd.Series:
        """Detect market structure breaks based on pattern changes"""
        breaks = pd.Series(0, index=df.index)
        
        # Bullish structure break: HL followed by HH
        for i in range(period, len(df)):
            if structure['hh'].iloc[i]:
                # Look back for HL
                if structure['hl'].iloc[i-period:i].any():
                    breaks.iloc[i] = 1
        
        # Bearish structure break: LH followed by LL
        for i in range(period, len(df)):
            if structure['ll'].iloc[i]:
                # Look back for LH
                if structure['lh'].iloc[i-period:i].any():
                    breaks.iloc[i] = -1
        
        return breaks
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Market Structure features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate features for each swing period
        for period in self.swing_periods:
            # Find swing points
            swing_highs, swing_lows = self._find_swing_points(shifted_data, period)
            
            # Calculate swing strength
            high_strength = self._calculate_swing_strength(
                shifted_data, swing_highs, period, is_high=True
            )
            low_strength = self._calculate_swing_strength(
                shifted_data, swing_lows, period, is_high=False
            )
            
            # Store swing point indicators
            result_df[self.get_feature_name(symbol, f'{self.timeframe}_{period}_swing_high')] = swing_highs.astype(int)
            result_df[self.get_feature_name(symbol, f'{self.timeframe}_{period}_swing_low')] = swing_lows.astype(int)
            self.log_feature_creation(self.get_feature_name(symbol, f'{self.timeframe}_{period}_swing_high'), result_df)
            self.log_feature_creation(self.get_feature_name(symbol, f'{self.timeframe}_{period}_swing_low'), result_df)
            
            # Store swing strength
            result_df[self.get_feature_name(symbol, f'{self.timeframe}_{period}_high_strength')] = high_strength
            result_df[self.get_feature_name(symbol, f'{self.timeframe}_{period}_low_strength')] = low_strength
            self.log_feature_creation(self.get_feature_name(symbol, f'{self.timeframe}_{period}_high_strength'), result_df)
            self.log_feature_creation(self.get_feature_name(symbol, f'{self.timeframe}_{period}_low_strength'), result_df)
            
            # Identify market structure
            structure = self._identify_structure(
                shifted_data, swing_highs, swing_lows, period
            )
            
            # Store structure patterns
            for pattern in ['hh', 'll', 'hl', 'lh']:
                feature = self.get_feature_name(symbol, f'{self.timeframe}_{period}_{pattern}')
                result_df[feature] = structure[pattern].astype(int)
                self.log_feature_creation(feature, result_df)
            
            # Detect structure breaks
            breaks = self._detect_structure_breaks(shifted_data, structure, period)
            break_feature = self.get_feature_name(symbol, f'{self.timeframe}_{period}_break')
            result_df[break_feature] = breaks
            self.log_feature_creation(break_feature, result_df)
            
            # Calculate trend context (bullish/bearish based on recent structure)
            window = period * 2
            hh_count = structure['hh'].rolling(window=window).sum()
            ll_count = structure['ll'].rolling(window=window).sum()
            hl_count = structure['hl'].rolling(window=window).sum()
            lh_count = structure['lh'].rolling(window=window).sum()
            
            trend_context = pd.Series(0, index=df.index)
            trend_context[hh_count + hl_count > lh_count + ll_count] = 1
            trend_context[lh_count + ll_count > hh_count + hl_count] = -1
            
            trend_feature = self.get_feature_name(symbol, f'{self.timeframe}_{period}_trend')
            result_df[trend_feature] = trend_context
            self.log_feature_creation(trend_feature, result_df)
            
            # Calculate z-scores
            for zscore_period in self.zscore_periods:
                # Strength z-scores
                for strength_type in ['high', 'low']:
                    base_feature = self.get_feature_name(symbol, f'{self.timeframe}_{period}_{strength_type}_strength')
                    zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_{period}_{strength_type}_strength_zscore_{zscore_period}')
                    result_df[zscore_feature] = self.calculate_zscore(
                        result_df[base_feature],
                        min_periods=zscore_period
                    )
                    self.log_feature_creation(zscore_feature, result_df)
            
            # Calculate strength-based signals
            for threshold in self.strength_threshold:
                threshold_str = str(threshold).replace('.', '')
                
                # Strong swing high signal
                high_signal = self.get_feature_name(symbol, f'{self.timeframe}_{period}_strong_high_{threshold_str}')
                result_df[high_signal] = (high_strength > threshold).astype(int)
                self.log_feature_creation(high_signal, result_df)
                
                # Strong swing low signal
                low_signal = self.get_feature_name(symbol, f'{self.timeframe}_{period}_strong_low_{threshold_str}')
                result_df[low_signal] = (low_strength > threshold).astype(int)
                self.log_feature_creation(low_signal, result_df)
        
        self.logger.info(f"Created {len(self.created_features)} features")
        return result_df