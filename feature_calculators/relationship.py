from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path

class RelationshipFeatureCalculator(FeatureCalculator):
    """
    Calculate relationship metrics between an asset and BTC.
    Features include correlation, beta, relative strength, and cointegration metrics.
    All calculations use shifted data to prevent lookahead bias.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 correlation_windows: List[int] = [24, 72, 168, 336, 720],  # 1d, 3d, 1w, 2w, 1m
                 beta_windows: List[int] = [24, 72, 168, 336, 720],
                 relative_strength_windows: List[int] = [24, 72, 168, 336, 720],
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.correlation_windows = sorted(correlation_windows)
        self.beta_windows = sorted(beta_windows)
        self.relative_strength_windows = sorted(relative_strength_windows)
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"Initialized {self.__class__.__name__} with "
                        f"correlation_windows={correlation_windows}, "
                        f"beta_windows={beta_windows}, "
                        f"relative_strength_windows={relative_strength_windows}")
    
    def get_required_columns(self) -> list:
        return ['close']  # Only need asset's close price
    
    def _load_btc_data(self, df: pd.DataFrame) -> pd.Series:
        """Load BTC data and align it with the input DataFrame's index"""
        try:
            # Load BTC data from the same directory as the input data
            data_dir = Path("data/raw/market_data_1h_2425d_20241217_022924/binance-futures")
            btc_file = data_dir / "BTCUSDT.csv"
            
            self.logger.info(f"Loading BTC data from: {btc_file}")
            
            # Load BTC data
            btc_df = pd.read_csv(btc_file)
            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
            btc_df.set_index('timestamp', inplace=True)
            
            # Align with input data's index
            btc_close = btc_df['close'].reindex(df.index)
            
            # Check for missing data
            missing = btc_close.isna().sum()
            if missing > 0:
                self.logger.warning(f"Missing {missing} BTC price points")
            
            return btc_close
            
        except Exception as e:
            self.logger.error(f"Error loading BTC data: {str(e)}")
            raise
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate relationship features with proper shifting"""
        if symbol == "BTCUSDT":
            self.logger.info("Skipping relationship features for BTC")
            return df
            
        result_df = super().calculate(df, symbol)
        
        # Load BTC data
        btc_close = self._load_btc_data(df)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        shifted_btc = btc_close.shift(1)
        
        # Calculate returns for both asset and BTC
        asset_returns = np.log(shifted_data['close'] / shifted_data['close'].shift(1))
        btc_returns = np.log(shifted_btc / shifted_btc.shift(1))
        
        # Calculate correlations for different windows
        for window in self.correlation_windows:
            # Rolling correlation
            correlation = asset_returns.rolling(window=window).corr(btc_returns)
            feature = self.get_feature_name(symbol, f'{self.timeframe}_correlation_{window}')
            result_df[feature] = correlation
            self.log_feature_creation(feature, result_df)
            
            # Calculate z-scores for correlation
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_correlation_{window}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    correlation,
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # Calculate betas for different windows
        for window in self.beta_windows:
            # Calculate rolling beta (regression slope)
            cov = asset_returns.rolling(window=window).cov(btc_returns)
            var = btc_returns.rolling(window=window).var()
            beta = cov / var
            
            feature = self.get_feature_name(symbol, f'{self.timeframe}_beta_{window}')
            result_df[feature] = beta
            self.log_feature_creation(feature, result_df)
            
            # Calculate z-scores for beta
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_beta_{window}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    beta,
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # Calculate relative strength metrics
        for window in self.relative_strength_windows:
            # Relative strength (asset performance vs BTC)
            asset_perf = (shifted_data['close'] / shifted_data['close'].shift(window) - 1) * 100
            btc_perf = (shifted_btc / shifted_btc.shift(window) - 1) * 100
            rel_strength = asset_perf - btc_perf
            
            feature = self.get_feature_name(symbol, f'{self.timeframe}_rel_strength_{window}')
            result_df[feature] = rel_strength.round(4)
            self.log_feature_creation(feature, result_df)
            
            # Calculate z-scores for relative strength
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_rel_strength_{window}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    rel_strength,
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
            
            # Calculate relative volatility
            asset_vol = asset_returns.rolling(window=window).std() * np.sqrt(252 * 24)
            btc_vol = btc_returns.rolling(window=window).std() * np.sqrt(252 * 24)
            rel_vol = asset_vol / btc_vol
            
            feature = self.get_feature_name(symbol, f'{self.timeframe}_rel_vol_{window}')
            result_df[feature] = rel_vol.round(4)
            self.log_feature_creation(feature, result_df)
            
            # Calculate z-scores for relative volatility
            for period in self.zscore_periods:
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_rel_vol_{window}_zscore_{period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    rel_vol,
                    min_periods=period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        # Calculate spread and its z-score
        log_spread = np.log(shifted_data['close']) - np.log(shifted_btc)
        spread_feature = self.get_feature_name(symbol, f'{self.timeframe}_log_spread')
        result_df[spread_feature] = log_spread
        self.log_feature_creation(spread_feature, result_df)
        
        # Calculate spread z-scores for different periods
        for period in self.zscore_periods:
            zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_log_spread_zscore_{period}')
            result_df[zscore_feature] = self.calculate_zscore(
                log_spread,
                min_periods=period
            )
            self.log_feature_creation(zscore_feature, result_df)
        
        return result_df