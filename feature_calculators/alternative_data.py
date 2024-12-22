from backtester.feature_calculators.base import FeatureCalculator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
from pathlib import Path

class AlternativeDataCalculator(FeatureCalculator):
    """
    Calculate alternative data features including cycle metrics and special events.
    Features are shifted by 1 period to reflect real-world collection delays.
    """
    
    def __init__(self, 
                 timeframe: str = "1h",
                 rolling_windows: List[int] = [24, 72, 168],  # 1d, 3d, 7d in hours
                 zscore_periods: List[int] = [168, 720]):  # 1w, 1m in hours
        
        super().__init__(timeframe=timeframe)
        self.rolling_windows = rolling_windows
        self.zscore_periods = zscore_periods
        
        # BTC Halving dates
        self.halving_dates = [
            pd.Timestamp("2012-11-28"),
            pd.Timestamp("2016-07-09"),
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-18")
        ]
        
        # High MRV Z-score periods
        self.high_mrv_periods = [
            (pd.Timestamp("2017-12-05"), pd.Timestamp("2017-12-21")),
            (pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-12")),
            (pd.Timestamp("2021-02-01"), pd.Timestamp("2021-02-20")),
            (pd.Timestamp("2021-03-10"), pd.Timestamp("2021-03-15"))
        ]
        
        # Load data sources
        self.vix_data = self._load_vix_data()
        self.fng_data = self._load_fng_data()
        
        # Fear and Greed API endpoint
        self.fng_url = "https://api.alternative.me/fng/"
    
    def get_required_columns(self) -> list:
        return ['close']
    
    def _load_vix_data(self) -> pd.DataFrame:
        """Load VIX data from hourly file"""
        try:
            vix_file = Path("/Users/ikennaigboaka/Documents/repos/central-market-data-service/backtester/data/raw/market_data_2425d_20241217_022924/cboe/VIX_full_1hour.txt")


            if not vix_file.exists():
                self.logger.warning(f"VIX file not found at expected path: {vix_file}")
                return pd.DataFrame()
                
            # Read without headers and assign column names
            df = pd.read_csv(vix_file, 
                           header=None, 
                           names=['timestamp', 'open', 'high', 'low', 'close'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Log successful load and data info
            self.logger.info(f"Successfully loaded VIX data:")
            self.logger.info(f"- Total rows: {len(df)}")
            self.logger.info(f"- Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"- Columns: {df.columns.tolist()}")
            self.logger.info(f"- Sample data:\n{df.head()}")
            
            # Check for data quality
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Found null values in VIX data:\n{null_counts}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading VIX data: {str(e)}")
            self.logger.warning("VIX features will not be calculated due to data loading error")
            return pd.DataFrame()

    def _load_fng_data(self) -> pd.DataFrame:
        """Load Fear and Greed index data from API"""
        try:
            params = {
                "limit": 100000,  # Get maximum available history
                "format": "json"
            }

            # Make API request
            response = requests.get(self.fng_url, params=params)
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch Fear & Greed data from API: {response.status_code}")
                return pd.DataFrame()

            # Parse response
            data = response.json()
            if 'data' not in data:
                self.logger.warning("Invalid API response format - missing 'data' field")
                return pd.DataFrame()

            # Convert to DataFrame and process
            df = pd.DataFrame(data['data'])

            # Convert timestamp to datetime and shift forward by one day
            # This aligns with the convention that data for day X is available at midnight of day X
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') + timedelta(days=1)

            # Convert value to numeric and rename
            df['fear_greed_value'] = pd.to_numeric(df['value'])

            # Sort by date
            df = df.sort_values('timestamp')

            # The data point labeled as "today" represents data from end of previous day
            # Subtract 2 days to get last complete day
            last_complete_date = df['timestamp'].max() - timedelta(days=2)
            self.logger.info(
                f"Last data point is labeled as {df['timestamp'].max().strftime('%Y-%m-%d')}, "
                f"representing data from {(df['timestamp'].max() - timedelta(days=1)).strftime('%Y-%m-%d')}"
            )
            self.logger.info(f"Setting last complete day to: {last_complete_date.strftime('%Y-%m-%d')}")
            df = df[df['timestamp'] <= last_complete_date]

            # Set index and select final columns
            df.set_index('timestamp', inplace=True)
            df = df[['fear_greed_value']].copy()

            # Log successful load and data info
            self.logger.info(f"Successfully loaded Fear & Greed data from API:")
            self.logger.info(f"- Total rows: {len(df)}")
            self.logger.info(f"- Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"- Columns: {df.columns.tolist()}")

            # Check for data quality
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Found null values in Fear & Greed data:\n{null_counts}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading Fear & Greed data from API: {str(e)}")
            self.logger.warning("Fear & Greed features will not be calculated due to API error")
            return pd.DataFrame()
    
    def _calculate_vix_metrics(self, df: pd.DataFrame, symbol: str, base_feature: str) -> pd.DataFrame:
        """Calculate VIX metrics with proper shifting"""
        if self.vix_data.empty:
            self.logger.warning("VIX data is empty, skipping VIX metrics calculation")
            return df
        
        # Add debug logging
        self.logger.info(f"VIX data shape: {self.vix_data.shape}")
        self.logger.info(f"VIX data index range: {self.vix_data.index.min()} to {self.vix_data.index.max()}")
        self.logger.info(f"Input df index range: {df.index.min()} to {df.index.max()}")
        
        # Check for index overlap
        vix_range = pd.Interval(self.vix_data.index.min(), self.vix_data.index.max(), closed='both')
        df_range = pd.Interval(df.index.min(), df.index.max(), closed='both')
        overlap = (
            max(vix_range.left, df_range.left),
            min(vix_range.right, df_range.right)
        )
        overlap_pct = (overlap[1] - overlap[0]) / (df_range.right - df_range.left)
        if overlap_pct < 0.9:  # Less than 90% overlap
            self.logger.warning(
                f"Limited overlap between VIX and input data: {overlap_pct:.1%}\n"
                f"This may result in missing or incomplete VIX features"
            )
        
        # Shift OHLC data forward by 1 period
        features_created = 0
        for col in ['open', 'high', 'low', 'close']:
            shifted_col = self.vix_data[col].shift(1)
            
            # Basic OHLC metrics
            feature_name = self.get_feature_name(symbol, f'{base_feature}_vix_{col}')
            df[feature_name] = shifted_col
            self.log_feature_creation(feature_name, df)
            features_created += 1
        
        # Calculate additional metrics
        shifted_high = self.vix_data['high'].shift(1)
        shifted_low = self.vix_data['low'].shift(1)
        shifted_close = self.vix_data['close'].shift(1)
        shifted_open = self.vix_data['open'].shift(1)
        
        # Daily range
        range_feature = self.get_feature_name(symbol, f'{base_feature}_vix_daily_range')
        df[range_feature] = ((shifted_high - shifted_low) / shifted_low * 100).round(2)
        self.log_feature_creation(range_feature, df)
        features_created += 1
        
        # Body size
        body_feature = self.get_feature_name(symbol, f'{base_feature}_vix_body_size')
        df[body_feature] = ((shifted_close - shifted_open) / shifted_open * 100).round(2)
        self.log_feature_creation(body_feature, df)
        features_created += 1
        
        # Upper/Lower shadows
        upper_shadow = self.get_feature_name(symbol, f'{base_feature}_vix_upper_shadow')
        lower_shadow = self.get_feature_name(symbol, f'{base_feature}_vix_lower_shadow')
        df[upper_shadow] = ((shifted_high - shifted_close.where(shifted_close > shifted_open, shifted_open)) / shifted_close * 100).round(2)
        df[lower_shadow] = ((shifted_close.where(shifted_close < shifted_open, shifted_open) - shifted_low) / shifted_close * 100).round(2)
        self.log_feature_creation(upper_shadow, df)
        self.log_feature_creation(lower_shadow, df)
        features_created += 2
        
        # Calculate rolling metrics for close price
        for window in self.rolling_windows:
            # Moving averages
            ma_feature = self.get_feature_name(symbol, f'{base_feature}_vix_ma_{window}')
            ma = shifted_close.rolling(window=window).mean()
            df[ma_feature] = ma
            self.log_feature_creation(ma_feature, df)
            features_created += 1
            
            # Position vs MA
            vs_ma_feature = self.get_feature_name(symbol, f'{base_feature}_vix_vs_ma_{window}')
            df[vs_ma_feature] = ((shifted_close - ma) / ma * 100).round(2)
            self.log_feature_creation(vs_ma_feature, df)
            features_created += 1
        
        # Calculate changes
        daily_change = self.get_feature_name(symbol, f'{base_feature}_vix_daily_change')
        weekly_change = self.get_feature_name(symbol, f'{base_feature}_vix_weekly_change')
        df[daily_change] = shifted_close.pct_change(24).round(4)
        df[weekly_change] = shifted_close.pct_change(168).round(4)
        self.log_feature_creation(daily_change, df)
        self.log_feature_creation(weekly_change, df)
        features_created += 2
        
        # Calculate z-scores
        for period in self.zscore_periods:
            # Close price z-score
            zscore = self.get_feature_name(symbol, f'{base_feature}_vix_zscore_{period}')
            df[zscore] = self.calculate_zscore(shifted_close, min_periods=period)
            self.log_feature_creation(zscore, df)
            features_created += 1
            
            # Range z-score
            range_zscore = self.get_feature_name(symbol, f'{base_feature}_vix_range_zscore_{period}')
            df[range_zscore] = self.calculate_zscore(df[range_feature], min_periods=period)
            self.log_feature_creation(range_zscore, df)
            features_created += 1
            
            # Change z-score
            change_zscore = self.get_feature_name(symbol, f'{base_feature}_vix_change_zscore_{period}')
            df[change_zscore] = self.calculate_zscore(df[daily_change], min_periods=period)
            self.log_feature_creation(change_zscore, df)
            features_created += 1
        
        # Check for null values in created features
        vix_features = [col for col in df.columns if 'vix' in col]
        null_counts = df[vix_features].isnull().sum()
        if null_counts.any():
            self.logger.warning(
                f"Found null values in VIX features:\n"
                f"{null_counts[null_counts > 0]}"
            )
        
        # Log feature creation summary
        self.logger.info(f"Created {features_created} VIX-related features")
        self.logger.info(f"VIX features memory usage: {df[vix_features].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return df
    
    def _calculate_fng_metrics(self, df: pd.DataFrame, symbol: str, base_feature: str) -> pd.DataFrame:
        """Calculate Fear and Greed metrics with proper shifting"""
        if self.fng_data.empty:
            self.logger.warning("Fear & Greed data is empty, skipping F&G metrics calculation")
            return df
        
        # Add debug logging
        self.logger.info(f"F&G data shape: {self.fng_data.shape}")
        self.logger.info(f"F&G data index range: {self.fng_data.index.min()} to {self.fng_data.index.max()}")
        
        # Check for index overlap
        fng_range = pd.Interval(self.fng_data.index.min(), self.fng_data.index.max(), closed='both')
        df_range = pd.Interval(df.index.min(), df.index.max(), closed='both')
        overlap = (
            max(fng_range.left, df_range.left),
            min(fng_range.right, df_range.right)
        )
        overlap_pct = (overlap[1] - overlap[0]) / (df_range.right - df_range.left)
        if overlap_pct < 0.9:  # Less than 90% overlap
            self.logger.warning(
                f"Limited overlap between F&G and input data: {overlap_pct:.1%}\n"
                f"This may result in missing or incomplete F&G features"
            )
        
        features_created = 0
        
        # Shift FNG data forward by 1 period
        shifted_fng = self.fng_data['fear_greed_value'].shift(1)
        
        # Basic FNG metrics
        fng_value = self.get_feature_name(symbol, f'{base_feature}_fng_value')
        df[fng_value] = shifted_fng
        self.log_feature_creation(fng_value, df)
        features_created += 1
        
        # Calculate rolling metrics
        for window in self.rolling_windows:
            # Moving averages
            ma_feature = self.get_feature_name(symbol, f'{base_feature}_fng_ma_{window}')
            ma = shifted_fng.rolling(window=window).mean()
            df[ma_feature] = ma
            self.log_feature_creation(ma_feature, df)
            features_created += 1
            
            # Position vs MA
            vs_ma_feature = self.get_feature_name(symbol, f'{base_feature}_fng_vs_ma_{window}')
            df[vs_ma_feature] = ((shifted_fng - ma) / ma * 100).round(2)
            self.log_feature_creation(vs_ma_feature, df)
            features_created += 1
        
        # Calculate changes
        daily_change = self.get_feature_name(symbol, f'{base_feature}_fng_daily_change')
        weekly_change = self.get_feature_name(symbol, f'{base_feature}_fng_weekly_change')
        df[daily_change] = shifted_fng.pct_change(24).round(4)
        df[weekly_change] = shifted_fng.pct_change(168).round(4)
        self.log_feature_creation(daily_change, df)
        self.log_feature_creation(weekly_change, df)
        features_created += 2
        
        # Calculate z-scores
        for period in self.zscore_periods:
            zscore = self.get_feature_name(symbol, f'{base_feature}_fng_zscore_{period}')
            df[zscore] = self.calculate_zscore(shifted_fng, min_periods=period)
            self.log_feature_creation(zscore, df)
            features_created += 1
            
            change_zscore = self.get_feature_name(symbol, f'{base_feature}_fng_change_zscore_{period}')
            df[change_zscore] = self.calculate_zscore(df[daily_change], min_periods=period)
            self.log_feature_creation(change_zscore, df)
            features_created += 1
        
        # Check for null values in created features
        fng_features = [col for col in df.columns if 'fng' in col]
        null_counts = df[fng_features].isnull().sum()
        if null_counts.any():
            self.logger.warning(
                f"Found null values in F&G features:\n"
                f"{null_counts[null_counts > 0]}"
            )
        
        # Log feature creation summary
        self.logger.info(f"Created {features_created} Fear & Greed related features")
        self.logger.info(f"F&G features memory usage: {df[fng_features].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        return df
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate alternative data features with proper shifting"""
        result_df = super().calculate(df, symbol)
        
        # Base feature name without group prefix (added by get_feature_name)
        base_feature = self.timeframe
        
        # Calculate VIX metrics
        result_df = self._calculate_vix_metrics(result_df, symbol, base_feature)
        
        # Calculate Fear and Greed metrics
        result_df = self._calculate_fng_metrics(result_df, symbol, base_feature)
        
        # Calculate cycle metrics with proper shifting
        shifted_dates = df.index.shift(-1, freq='1H')  # Shift dates back by 1 period
        
        cycle_metrics = [self._get_cycle_metrics(date) for date in shifted_dates]
        metrics_df = pd.DataFrame(cycle_metrics, index=df.index)
        
        # Add cycle metrics to result
        features_created = 0
        for metric in ['cycle_year', 'days_to_halving', 'is_halving', 'high_mrv']:
            feature_name = self.get_feature_name(symbol, f'{base_feature}_{metric}')
            result_df[feature_name] = metrics_df[metric]
            self.log_feature_creation(feature_name, result_df)
            features_created += 1
        
        # Log final summary
        self.logger.info(f"Total features created: {features_created}")
        
        return result_df

    def _get_cycle_metrics(self, date: pd.Timestamp) -> Dict[str, int]:
        """Calculate cycle metrics for a given date"""
        # Calculate cycle year
        cycle_starts = [
            pd.Timestamp("2012-11-28"),
            pd.Timestamp("2016-07-09"),
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-18"),
            pd.Timestamp("2028-04-18")  # Estimated
        ]
        
        cycle_year = 0
        for i in range(len(cycle_starts)-1):
            if cycle_starts[i] <= date < cycle_starts[i+1]:
                days_into_cycle = (date - cycle_starts[i]).days
                total_cycle_days = (cycle_starts[i+1] - cycle_starts[i]).days
                cycle_year = int((days_into_cycle / total_cycle_days * 4) + 1)
                break
        
        # Calculate days until next halving
        days_to_halving = 0
        for halving_date in self.halving_dates:
            if date < halving_date:
                days_to_halving = (halving_date - date).days
                break
        if days_to_halving == 0:  # Beyond last known halving
            last_halving = self.halving_dates[-1]
            next_halving = last_halving + pd.Timedelta(days=1460)  # ~4 years
            days_to_halving = (next_halving - date).days
        
        return {
            'cycle_year': cycle_year,
            'days_to_halving': days_to_halving,
            'is_halving': 1 if date in self.halving_dates else 0,
            'high_mrv': 1 if any(start <= date <= end for start, end in self.high_mrv_periods) else 0
        }