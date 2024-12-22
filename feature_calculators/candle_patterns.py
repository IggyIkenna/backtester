from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class CandlePatternCalculator(FeatureCalculator):
    """
    Calculate candlestick patterns with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 doji_threshold: float = 0.1,
                 shadow_threshold: float = 2.0,
                 pattern_window: int = 20,
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.doji_threshold = doji_threshold
        self.shadow_threshold = shadow_threshold
        self.pattern_window = pattern_window
        self.zscore_periods = zscore_periods
    
    def get_required_columns(self) -> list:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def _calculate_candle_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic candle metrics using shifted data"""
        # Use previous completed candle
        shifted_data = df.shift(1)
        
        # Calculate basic metrics
        body = shifted_data['close'] - shifted_data['open']
        body_size = abs(body)
        upper_shadow = shifted_data['high'] - pd.concat(
            [shifted_data['open'], shifted_data['close']], axis=1
        ).max(axis=1)
        lower_shadow = pd.concat(
            [shifted_data['open'], shifted_data['close']], axis=1
        ).min(axis=1) - shifted_data['low']
        total_range = shifted_data['high'] - shifted_data['low']
        
        # Calculate relative metrics
        avg_body = body_size.rolling(self.pattern_window, min_periods=1).mean()
        avg_shadow = (upper_shadow + lower_shadow).rolling(self.pattern_window, min_periods=1).mean()
        avg_range = total_range.rolling(self.pattern_window, min_periods=1).mean()
        
        return pd.DataFrame({
            'body': body,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'total_range': total_range,
            'avg_body': avg_body,
            'avg_shadow': avg_shadow,
            'avg_range': avg_range,
            'open': shifted_data['open'],
            'high': shifted_data['high'],
            'low': shifted_data['low'],
            'close': shifted_data['close'],
            'volume': shifted_data['volume']
        })
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate candle pattern features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Calculate candle metrics
        metrics = self._calculate_candle_metrics(df)
        base_feature = self.timeframe
        
        # 1. Single Candle Patterns
        # Doji
        doji_feature = self.get_feature_name(symbol, f'{base_feature}_doji')
        doji = (metrics['body_size'] <= metrics['avg_body'] * self.doji_threshold)
        result_df[doji_feature] = doji.astype(int)
        self.log_feature_creation(doji_feature, result_df)
        
        # Hammer
        hammer_feature = self.get_feature_name(symbol, f'{base_feature}_hammer')
        hammer_cond = (
            (metrics['lower_shadow'] > metrics['body_size'] * self.shadow_threshold) &
            (metrics['upper_shadow'] < metrics['body_size']) &
            (metrics['body'] > 0)
        )
        result_df[hammer_feature] = hammer_cond.astype(int)
        self.log_feature_creation(hammer_feature, result_df)
        
        # Shooting Star
        star_feature = self.get_feature_name(symbol, f'{base_feature}_shooting_star')
        star_cond = (
            (metrics['upper_shadow'] > metrics['body_size'] * self.shadow_threshold) &
            (metrics['lower_shadow'] < metrics['body_size']) &
            (metrics['body'] < 0)
        )
        result_df[star_feature] = star_cond.astype(int)
        self.log_feature_creation(star_feature, result_df)
        
        # 2. Pattern Strength Metrics
        # Hammer strength
        hammer_strength = self.get_feature_name(symbol, f'{base_feature}_hammer_strength')
        result_df[hammer_strength] = np.where(
            hammer_cond,
            metrics['lower_shadow'] / metrics['body_size'],
            0
        )
        self.log_feature_creation(hammer_strength, result_df)
        
        # Star strength
        star_strength = self.get_feature_name(symbol, f'{base_feature}_star_strength')
        result_df[star_strength] = np.where(
            star_cond,
            metrics['upper_shadow'] / metrics['body_size'],
            0
        )
        self.log_feature_creation(star_strength, result_df)
        
        # Calculate z-scores for pattern strengths
        for pattern in ['hammer', 'star']:
            strength_feature = self.get_feature_name(symbol, f'{base_feature}_{pattern}_strength')
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{base_feature}_{pattern}_strength_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    result_df[strength_feature],
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # 3. Multi-Candle Patterns
        # Engulfing
        bull_engulf_feature = self.get_feature_name(symbol, f'{base_feature}_bullish_engulfing')
        bullish_engulfing = (
            (metrics['body'].shift(1) < 0) &  # Previous red
            (metrics['body'] > 0) &           # Current green
            (metrics['open'] <= metrics['close'].shift(1)) &
            (metrics['close'] >= metrics['open'].shift(1))
        )
        result_df[bull_engulf_feature] = bullish_engulfing.astype(int)
        self.log_feature_creation(bull_engulf_feature, result_df)
        
        bear_engulf_feature = self.get_feature_name(symbol, f'{base_feature}_bearish_engulfing')
        bearish_engulfing = (
            (metrics['body'].shift(1) > 0) &  # Previous green
            (metrics['body'] < 0) &           # Current red
            (metrics['open'] >= metrics['close'].shift(1)) &
            (metrics['close'] <= metrics['open'].shift(1))
        )
        result_df[bear_engulf_feature] = bearish_engulfing.astype(int)
        self.log_feature_creation(bear_engulf_feature, result_df)
        
        # Three White Soldiers / Black Crows
        white_soldiers_feature = self.get_feature_name(symbol, f'{base_feature}_three_white_soldiers')
        three_white = (
            (metrics['body'] > 0) &
            (metrics['body'].shift(1) > 0) &
            (metrics['body'].shift(2) > 0) &
            (metrics['close'] > metrics['close'].shift(1)) &
            (metrics['close'].shift(1) > metrics['close'].shift(2))
        )
        result_df[white_soldiers_feature] = three_white.astype(int)
        self.log_feature_creation(white_soldiers_feature, result_df)
        
        black_crows_feature = self.get_feature_name(symbol, f'{base_feature}_three_black_crows')
        three_black = (
            (metrics['body'] < 0) &
            (metrics['body'].shift(1) < 0) &
            (metrics['body'].shift(2) < 0) &
            (metrics['close'] < metrics['close'].shift(1)) &
            (metrics['close'].shift(1) < metrics['close'].shift(2))
        )
        result_df[black_crows_feature] = three_black.astype(int)
        self.log_feature_creation(black_crows_feature, result_df)
        
        # 4. Pattern Frequency Analysis
        bullish_patterns = (hammer_cond | bullish_engulfing | three_white)
        bearish_patterns = (star_cond | bearish_engulfing | three_black)
        
        # Rolling pattern counts
        bull_count_feature = self.get_feature_name(symbol, f'{base_feature}_bullish_count')
        result_df[bull_count_feature] = bullish_patterns.rolling(
            self.pattern_window, min_periods=1
        ).sum()
        self.log_feature_creation(bull_count_feature, result_df)
        
        bear_count_feature = self.get_feature_name(symbol, f'{base_feature}_bearish_count')
        result_df[bear_count_feature] = bearish_patterns.rolling(
            self.pattern_window, min_periods=1
        ).sum()
        self.log_feature_creation(bear_count_feature, result_df)
        
        # Calculate z-scores for pattern frequencies
        for pattern_type in ['bullish', 'bearish']:
            count_feature = self.get_feature_name(symbol, f'{base_feature}_{pattern_type}_count')
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{base_feature}_{pattern_type}_count_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    result_df[count_feature],
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # 5. Volume Confirmation
        vol_intensity_feature = self.get_feature_name(symbol, f'{base_feature}_volume_intensity')
        result_df[vol_intensity_feature] = (
            metrics['volume'] / metrics['volume'].rolling(self.pattern_window).mean()
        )
        self.log_feature_creation(vol_intensity_feature, result_df)
        
        # Calculate volume intensity z-scores
        for period in self.zscore_periods:
            zscore_feature = self.get_feature_name(symbol, f'{base_feature}_volume_intensity_zscore_{period}')
            result_df[zscore_feature] = self.calculate_zscore(
                result_df[vol_intensity_feature],
                min_periods=period
            )
            self.log_feature_creation(zscore_feature, result_df)
        
        # 6. Pattern Clustering
        cluster_feature = self.get_feature_name(symbol, f'{base_feature}_pattern_cluster')
        result_df[cluster_feature] = (
            result_df[bull_count_feature] + 
            result_df[bear_count_feature]
        )
        self.log_feature_creation(cluster_feature, result_df)
        
        # Calculate pattern cluster z-scores
        for period in self.zscore_periods:
            zscore_feature = self.get_feature_name(symbol, f'{base_feature}_pattern_cluster_zscore_{period}')
            result_df[zscore_feature] = self.calculate_zscore(
                result_df[cluster_feature],
                min_periods=period
            )
            self.log_feature_creation(zscore_feature, result_df)
        
        return result_df