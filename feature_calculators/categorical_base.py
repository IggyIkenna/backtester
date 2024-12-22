from typing import List, Callable
import pandas as pd
from feature_calculators.base import FeatureCalculator

class CategoricalFeatureCalculator(FeatureCalculator):
    def __init__(self, timeframe: str, mode: str = "discrete", excluded_seasons: List[str] = None):
        super().__init__(timeframe=timeframe)
        if mode not in ["discrete", "developing"]:
            raise ValueError("Mode must be either 'discrete' or 'developing'")
        self.mode = mode
        self.excluded_seasons = excluded_seasons or []
        
    def calculate_categorical_feature(self, df: pd.DataFrame, season_col: str, 
                                    metric_func: Callable) -> pd.Series:
        """Calculate feature for a categorical period"""
        if self.mode == "developing":
            return self._calculate_developing(df, season_col, metric_func)
        return self._calculate_discrete(df, season_col, metric_func)
    
    def _calculate_developing(self, df: pd.DataFrame, season_col: str, 
                            metric_func: Callable) -> pd.Series:
        """Calculate developing metrics within categorical periods"""
        shifted_data = df.shift(1)  # Account for data availability
        session_changes = df[season_col] != df[season_col].shift(1)
        result = pd.Series(index=df.index, dtype=float)
        
        current_period_start = None
        current_period_data = []
        
        for i, (idx, row) in enumerate(shifted_data.iterrows()):
            if session_changes.iloc[i]:
                # New period starts
                if current_period_data:
                    # Finalize previous period
                    result.iloc[i-1] = metric_func(pd.concat(current_period_data))
                current_period_start = idx
                current_period_data = [row]
            else:
                current_period_data.append(row)
                # Calculate developing metric
                if current_period_data:
                    period_length = len(df[df[season_col] == df[season_col].iloc[i]])
                    current_length = len(current_period_data)
                    raw_metric = metric_func(pd.concat(current_period_data))
                    # Scale for incomplete periods
                    result.iloc[i] = raw_metric * (period_length/current_length)
        
        return result
    
    def _calculate_discrete(self, df: pd.DataFrame, season_col: str, metric_func: Callable) -> pd.Series:
        """Calculate discrete metrics for completed categorical periods"""
        shifted_data = df.shift(1)  # Account for data availability
        session_changes = df[season_col] != df[season_col].shift(1)
        result = pd.Series(index=df.index, dtype=float)
        
        current_period_data = []
        
        for i, (idx, row) in enumerate(shifted_data.iterrows()):
            if session_changes.iloc[i] and current_period_data:
                # Period completed, calculate metric
                metric_value = metric_func(pd.concat(current_period_data))
                # Apply to all rows in the period
                result.iloc[i-len(current_period_data):i] = metric_value
                current_period_data = []
            current_period_data.append(row)
        
        return result