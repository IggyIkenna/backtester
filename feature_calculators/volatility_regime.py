from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

class VolatilityRegimeCalculator(FeatureCalculator):
    """
    Calculate Volatility Regime features with proper shifting and z-scores.
    All calculations use shifted data to prevent lookahead bias.
    Features include regime classification, transitions, and persistence metrics.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 short_window: int = 24,    # 1 day
                 medium_window: int = 168,   # 1 week
                 long_window: int = 720,     # 1 month
                 num_regimes: int = 3,       # Low, Medium, High volatility
                 transition_window: int = 12, # Hours to detect regime transitions
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.num_regimes = num_regimes
        self.transition_window = transition_window
        self.zscore_periods = zscore_periods
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='mean')
    
    def get_required_columns(self) -> list:
        return ['high', 'low', 'close', 'quote_volume']
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various volatility metrics"""
        # Parkinson volatility (uses high-low range)
        high_low_ratio = np.log(df['high'] / df['low'])
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * high_low_ratio ** 2)
        
        # Garman-Klass volatility (uses OHLC)
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['close'].shift(1)) ** 2
        garman_klass_vol = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        
        # True Range and ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Volume-weighted volatility
        volume_ratio = df['quote_volume'] / df['quote_volume'].rolling(window=self.short_window).mean()
        vol_weighted_tr = tr * volume_ratio
        
        return {
            'parkinson': parkinson_vol,
            'garman_klass': garman_klass_vol,
            'true_range': tr,
            'vol_weighted_tr': vol_weighted_tr
        }
    
    def _detect_regimes(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility regimes using Gaussian Mixture Model"""
        # Prepare data for GMM
        X = volatility.values.reshape(-1, 1)
        
        # Handle NaN values using imputer
        X_imputed = self.imputer.fit_transform(X)
        
        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.num_regimes,
            random_state=42,
            covariance_type='full',
            n_init=5
        )
        
        # Predict regimes
        regimes = pd.Series(gmm.fit_predict(X_imputed), index=volatility.index)
        
        # Sort regimes by volatility level (0 = low, 1 = medium, 2 = high)
        means = [gmm.means_[i][0] for i in range(self.num_regimes)]
        regime_map = {i: rank for rank, i in enumerate(np.argsort(means))}
        regimes = regimes.map(regime_map)
        
        # Set NaN values in original data to NaN in regimes
        regimes[volatility.isna()] = np.nan
        
        return regimes
    
    def _calculate_regime_metrics(self, 
                                regimes: pd.Series,
                                volatility: pd.Series) -> Dict[str, pd.Series]:
        """Calculate regime-related metrics"""
        # Regime duration
        regime_changes = regimes != regimes.shift(1)
        regime_groups = (~regime_changes).cumsum()
        regime_duration = regime_groups.groupby(regime_groups).cumcount() + 1
        
        # Regime stability (inverse of regime changes in window)
        regime_stability = 1 - regime_changes.rolling(
            window=self.transition_window
        ).mean()
        
        # Regime intensity (how far current volatility is from regime mean)
        regime_means = volatility.groupby(regimes).transform('mean')
        regime_intensity = (volatility - regime_means) / regime_means
        
        # Transition probability
        def calc_transition_prob(x):
            if len(x) < 2:
                return 0
            return (x != x.shift(1)).mean()
        
        transition_prob = regimes.rolling(
            window=self.transition_window
        ).apply(calc_transition_prob)
        
        return {
            'duration': regime_duration,
            'stability': regime_stability,
            'intensity': regime_intensity,
            'transition_prob': transition_prob
        }
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate Volatility Regime features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Use shifted data for all calculations
        shifted_data = df.shift(1)
        
        # Calculate volatility metrics
        vol_metrics = self._calculate_volatility_metrics(shifted_data)
        
        # Calculate rolling volatilities for different windows
        for window, window_name in [
            (self.short_window, 'short'),
            (self.medium_window, 'medium'),
            (self.long_window, 'long')
        ]:
            # Store base volatility metrics
            for metric_name, metric_values in vol_metrics.items():
                # Rolling volatility
                feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_{window_name}')
                result_df[feature] = metric_values.rolling(window=window).mean()
                self.log_feature_creation(feature, result_df)
                
                # Detect regimes for each metric and window
                regimes = self._detect_regimes(result_df[feature])
                regime_feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_{window_name}_regime')
                result_df[regime_feature] = regimes
                self.log_feature_creation(regime_feature, result_df)
                
                # Calculate regime metrics
                regime_metrics = self._calculate_regime_metrics(
                    regimes,
                    result_df[feature]
                )
                
                # Store regime metrics
                for r_metric_name, r_metric_values in regime_metrics.items():
                    metric_feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_{window_name}_regime_{r_metric_name}')
                    result_df[metric_feature] = r_metric_values
                    self.log_feature_creation(metric_feature, result_df)
        
        # Calculate composite regime
        composite_vol = pd.DataFrame([
            result_df[self.get_feature_name(symbol, f'{self.timeframe}_{m}_medium')] for m in vol_metrics.keys()
        ]).mean()
        
        composite_regimes = self._detect_regimes(composite_vol)
        composite_regime_feature = self.get_feature_name(symbol, f'{self.timeframe}_composite_regime')
        result_df[composite_regime_feature] = composite_regimes
        self.log_feature_creation(composite_regime_feature, result_df)
        
        # Calculate composite regime metrics
        composite_metrics = self._calculate_regime_metrics(
            composite_regimes,
            composite_vol
        )
        
        for metric_name, metric_values in composite_metrics.items():
            feature = self.get_feature_name(symbol, f'{self.timeframe}_composite_{metric_name}')
            result_df[feature] = metric_values
            self.log_feature_creation(feature, result_df)
        
        # Calculate z-scores
        for zscore_period in self.zscore_periods:
            # Volatility metric z-scores
            for window_name in ['short', 'medium', 'long']:
                for metric_name in vol_metrics.keys():
                    base_feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_{window_name}')
                    zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_{metric_name}_{window_name}_zscore_{zscore_period}')
                    result_df[zscore_feature] = self.calculate_zscore(
                        result_df[base_feature],
                        min_periods=zscore_period
                    )
                    self.log_feature_creation(zscore_feature, result_df)
            
            # Regime metric z-scores
            for metric_name in ['duration', 'stability', 'intensity']:
                base_feature = self.get_feature_name(symbol, f'{self.timeframe}_composite_{metric_name}')
                zscore_feature = self.get_feature_name(symbol, f'{self.timeframe}_composite_{metric_name}_zscore_{zscore_period}')
                result_df[zscore_feature] = self.calculate_zscore(
                    result_df[base_feature],
                    min_periods=zscore_period
                )
                self.log_feature_creation(zscore_feature, result_df)
        
        self.logger.info(f"Created {len(self.created_features)} features")
        return result_df