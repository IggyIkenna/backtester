from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from feature_calculators.base import FeatureCalculator
import logging

class TimeRangeCalculator(FeatureCalculator):
    """Calculate time-based range features with proper period handling"""
    
    def __init__(self, timeframe: str = "1h", mode: str = "complete"):
        """
        Initialize the calculator.
        
        Args:
            timeframe: Data timeframe (e.g., '1h', '4h')
            mode: Either 'complete' for finished periods or 'developing' for in-progress
        """
        super().__init__(timeframe)
        self.mode = mode
        
        # Session definitions (for grouping, not for data access)
        self.sessions = {
            'us': (lambda x: (x.hour >= 14) & (x.hour < 21)),  # Vectorized condition
            'asia': (lambda x: (x.hour >= 22) | (x.hour < 7)),  # Vectorized condition
            'london': (lambda x: (x.hour >= 8) & (x.hour < 16)),  # Vectorized condition
            'day': (lambda x: pd.Series(True, index=x))  # Full day, vectorized
        }
        
        # Z-score periods for normalization
        self.zscore_periods = [168, 720]  # 1w, 1m in hours
    
    def get_required_columns(self) -> list:
        """Return required input columns"""
        return ['open', 'high', 'low', 'close']
    
    def _calculate_ath_atl_metrics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate ATH/ATL metrics with proper shifting"""
        # Use shifted data for all calculations
        shifted_data = df[['high', 'low', 'close']].shift(1)
        
        # Calculate expanding max/min (ATH/ATL) - already vectorized
        ath = shifted_data['high'].expanding().max()
        atl = shifted_data['low'].expanding().min()
        
        # Calculate distances as percentage from current close - vectorized operations
        close = shifted_data['close']
        ath_distance = ((ath - close) / close).round(4)
        atl_distance = ((close - atl) / close).round(4)
        
        # Calculate time since ATH/ATL - vectorized conditions
        is_ath = shifted_data['high'] >= ath
        is_atl = shifted_data['low'] <= atl
        
        periods_since_ath = self.calculate_time_since(is_ath)
        periods_since_atl = self.calculate_time_since(is_atl)
        
        # Calculate z-scores for distances - vectorized operations
        ath_distance_zscores = {
            period: self.calculate_zscore(ath_distance.rolling(period).mean())
            for period in self.zscore_periods
        }
        atl_distance_zscores = {
            period: self.calculate_zscore(atl_distance.rolling(period).mean())
            for period in self.zscore_periods
        }
        
        return {
            'ath': ath,
            'atl': atl,
            'ath_distance': ath_distance,
            'atl_distance': atl_distance,
            'periods_since_ath': periods_since_ath,
            'periods_since_atl': periods_since_atl,
            'ath_distance_zscores': ath_distance_zscores,
            'atl_distance_zscores': atl_distance_zscores
        }
    
    def _calculate_range_metrics(self, df: pd.DataFrame, period_mask: pd.Series) -> Dict[str, pd.Series]:
        """Calculate range metrics for a given period - vectorized operations"""
        if not period_mask.any():
            return None
        
        # Use shifted data for completed candles
        period_data = df[['open', 'high', 'low', 'close']].shift(1)[period_mask]
        
        if len(period_data) == 0:
            return None
        
        # Calculate range metrics - vectorized operations
        period_high = period_data['high'].max()
        period_low = period_data['low'].min()
        period_range = period_high - period_low
        period_open = period_data['open'].iloc[0]
        period_close = period_data['close'].iloc[-1]
        
        # Calculate range percentage - scalar operations
        period_range_pct = round(period_range / period_low, 4)
        period_body = abs(period_close - period_open)
        period_body_pct = round(period_body / period_open, 4)
        
        # Calculate wicks - scalar operations
        if period_close >= period_open:
            upper_wick = period_high - period_close
            lower_wick = period_open - period_low
        else:
            upper_wick = period_high - period_open
            lower_wick = period_close - period_low
        
        total_wick = upper_wick + lower_wick
        wick_ratio = round(upper_wick / total_wick, 4) if total_wick > 0 else 0
        
        # Create series for z-score calculations - reuse existing series
        range_series = pd.Series(period_range_pct, index=df.index[period_mask])
        body_series = pd.Series(period_body_pct, index=df.index[period_mask])
        
        # Calculate z-scores - vectorized operations
        range_zscores = {
            period: self.calculate_zscore(range_series.rolling(period).mean())
            for period in self.zscore_periods
        }
        body_zscores = {
            period: self.calculate_zscore(body_series.rolling(period).mean())
            for period in self.zscore_periods
        }
        
        # Create series for all metrics at once
        metrics_df = pd.DataFrame(index=df.index[period_mask])
        metrics_df['high'] = period_high
        metrics_df['low'] = period_low
        metrics_df['range'] = period_range
        metrics_df['range_pct'] = period_range_pct
        metrics_df['body'] = period_body
        metrics_df['body_pct'] = period_body_pct
        metrics_df['upper_wick'] = upper_wick
        metrics_df['lower_wick'] = lower_wick
        metrics_df['wick_ratio'] = wick_ratio
        
        return {
            **{col: metrics_df[col] for col in metrics_df.columns},
            'range_zscores': range_zscores,
            'body_zscores': body_zscores
        }
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate range features with proper shifting"""
        result_df = super().calculate(df, symbol)
        self.symbol = symbol
        
        # Calculate ATH/ATL metrics
        ath_atl_metrics = self._calculate_ath_atl_metrics(df)
        
        # Add ATH/ATL metrics to result - vectorized assignments
        metric_mapping = {
            'ath': ath_atl_metrics['ath'],
            'atl': ath_atl_metrics['atl'],
            'ath_distance': ath_atl_metrics['ath_distance'],
            'atl_distance': ath_atl_metrics['atl_distance'],
            'periods_since_ath': ath_atl_metrics['periods_since_ath'],
            'periods_since_atl': ath_atl_metrics['periods_since_atl']
        }
        
        for name, values in metric_mapping.items():
            result_df[self.get_feature_name(name)] = values
        
        # Add ATH/ATL z-scores - vectorized assignments
        for period in self.zscore_periods:
            result_df[self.get_feature_name(f'ath_distance_{period}h_zscore')] = ath_atl_metrics['ath_distance_zscores'][period]
            result_df[self.get_feature_name(f'atl_distance_{period}h_zscore')] = ath_atl_metrics['atl_distance_zscores'][period]
        
        # Process each session
        for session_name, session_condition in self.sessions.items():
            # Create session mask - vectorized condition
            session_mask = session_condition(df.index)
            
            if self.mode == "developing":
                # Only calculate for current developing session - vectorized mask
                current_date = df.index[-1].date()
                current_mask = session_mask & (df.index.date == current_date)
                if current_mask.any():
                    metrics = self._calculate_range_metrics(df, current_mask)
                    if metrics:
                        self._add_session_metrics(result_df, metrics, session_name)
            else:
                # Calculate for each completed session
                dates = pd.Series(df.index.date).unique()
                # Process dates in chunks for better performance
                chunk_size = 100
                for i in range(0, len(dates), chunk_size):
                    date_chunk = dates[i:i + chunk_size]
                    # Create mask for all dates in chunk - vectorized operation using numpy
                    dates_mask = np.isin(df.index.date, date_chunk)
                    chunk_mask = session_mask & dates_mask
                    if chunk_mask.any():
                        metrics = self._calculate_range_metrics(df, chunk_mask)
                        if metrics:
                            self._add_session_metrics(result_df, metrics, session_name)
        
        # Store created feature names - vectorized operation
        self.created_features = [col for col in result_df.columns 
                               if col.startswith(f"{self.symbol}_") and 
                               col not in df.columns]
        
        return self.shift_features(result_df)
    
    def _add_session_metrics(self, df: pd.DataFrame, metrics: Dict, session_name: str) -> None:
        """Add session metrics to DataFrame with proper naming - vectorized assignments"""
        base = f"{self.timeframe}_{session_name}"
        
        # Add all metrics at once using a dictionary comprehension
        metric_mapping = {
            f'{base}_{metric}': values
            for metric, values in metrics.items()
            if not isinstance(values, dict)  # Skip nested dictionaries (z-scores)
        }
        
        # Add basic metrics - vectorized assignment
        for feature_suffix, values in metric_mapping.items():
            df[self.get_feature_name(feature_suffix)] = values
        
        # Add z-scores - vectorized assignment
        for period in self.zscore_periods:
            df[self.get_feature_name(f'{base}_range_{period}h_zscore')] = metrics['range_zscores'][period]
            df[self.get_feature_name(f'{base}_body_{period}h_zscore')] = metrics['body_zscores'][period]
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based range features with improved logging"""
        logging.info(f"Calculating time range features for {len(df)} periods")
        
        try:
            # Create copy of input data
            result = df.copy()
            features_added = 0
            
            # Calculate features for each symbol
            for symbol in self.symbols:
                logging.info(f"Processing {symbol}")
                
                # Get price data
                price_data = result[result['symbol'] == symbol]
                if price_data.empty:
                    logging.warning(f"No data found for {symbol}")
                    continue
                    
                # Calculate session ranges
                for session in ['asia', 'london', 'ny']:
                    logging.info(f"Calculating {session} session features")
                    
                    # Calculate session high/low/range
                    session_data = self._calculate_session_range(price_data, session)
                    if session_data is not None:
                        for col in session_data.columns:
                            result[f"{symbol}_{session}_{col}"] = session_data[col]
                            features_added += 1
                            
                # Calculate daily ranges
                logging.info("Calculating daily features")
                daily_data = self._calculate_daily_range(price_data)
                if daily_data is not None:
                    for col in daily_data.columns:
                        result[f"{symbol}_24h_{col}"] = daily_data[col]
                        features_added += 1
                        
                # Calculate z-scores
                for lookback in [168, 720]:  # 1 week and 1 month
                    logging.info(f"Calculating {lookback}h z-scores")
                    zscore_data = self._calculate_zscores(price_data, lookback)
                    if zscore_data is not None:
                        for col in zscore_data.columns:
                            result[f"{symbol}_{col}_{lookback}h_zscore"] = zscore_data[col]
                            features_added += 1
                            
            logging.info(f"Added {features_added} time range features")
            return result
            
        except Exception as e:
            logging.error(f"Error calculating time range features: {str(e)}")
            logging.exception("Full traceback:")
            raise
    
    def _calculate_session_range(self, df: pd.DataFrame, session: str) -> Optional[pd.DataFrame]:
        """Calculate price ranges for a trading session"""
        try:
            # Get session hours
            hours = {
                'asia': (0, 8),
                'london': (8, 16),
                'ny': (16, 24)
            }
            start_hour, end_hour = hours[session]
            
            # Filter for session hours
            mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
            session_data = df[mask]
            
            if session_data.empty:
                logging.warning(f"No data for {session} session")
                return None
                
            # Resample to get session OHLC
            resampled = session_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Calculate ranges
            result = pd.DataFrame()
            result[f'{session}_high'] = resampled['high']
            result[f'{session}_low'] = resampled['low']
            result[f'{session}_range'] = resampled['high'] - resampled['low']
            result[f'{session}_range_pct'] = result[f'{session}_range'] / resampled['open']
            result[f'{session}_body'] = abs(resampled['close'] - resampled['open'])
            result[f'{session}_body_pct'] = result[f'{session}_body'] / resampled['open']
            
            # Calculate wicks
            result[f'{session}_upper_wick'] = resampled['high'] - resampled[['open', 'close']].max(axis=1)
            result[f'{session}_lower_wick'] = resampled[['open', 'close']].min(axis=1) - resampled['low']
            result[f'{session}_wick_ratio'] = result[f'{session}_upper_wick'] / (result[f'{session}_lower_wick'] + 1e-10)
            
            logging.info(f"Calculated {len(result.columns)} features for {session} session")
            return result
            
        except Exception as e:
            logging.error(f"Error calculating {session} session features: {str(e)}")
            return None