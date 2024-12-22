"""
Utility functions for feature generation and validation.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from backtester.utils.exceptions import ValidationError, ConfigurationError


def period_to_seconds(period: str) -> int:
    """Convert a period string to seconds with validation"""
    try:
        multiplier = int(''.join(filter(str.isdigit, period)))
        unit = ''.join(filter(str.isalpha, period)).lower()
        
        conversion = {
            's': 1, 'sec': 1,
            'm': 60, 'min': 60,
            'h': 3600, 'hour': 3600,
            'd': 86400, 'day': 86400,
            'w': 604800, 'week': 604800
        }
        
        if unit not in conversion:
            raise ValidationError(
                field="period_unit",
                value=unit,
                constraint=f"must be one of {list(conversion.keys())}"
            )
        
        return multiplier * conversion[unit]
        
    except Exception as e:
        raise ConfigurationError(
            parameter="period",
            value=period,
            reason=str(e)
        )


def get_expected_timestamps_count(start_time: pd.Timestamp,
                                end_time: pd.Timestamp,
                                base_candle_size: str) -> int:
    """Calculate expected number of timestamps between start and end time."""
    duration = (end_time - start_time).total_seconds()
    base_seconds = period_to_seconds(base_candle_size)
    return int(duration / base_seconds) + 1 


class PeriodType(Enum):
    FIXED_TIME = 'fixed_time'
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    ALLTIME = 'alltime'

# Global mapping of allowed window types per period type
ALLOWED_WINDOW_TYPES: Dict[PeriodType, List[str]] = {
    PeriodType.FIXED_TIME: ['discrete', 'continuous'],
    PeriodType.NUMERIC: ['discrete', 'continuous'],
    PeriodType.CATEGORICAL: ['discrete', 'developing'],  # No continuous for categorical
    PeriodType.ALLTIME: ['expanding']  # Only expanding for alltime
}

def get_period_type(period: str) -> PeriodType:
    """Determine period type from period string"""
    period = str(period).lower()
    if period in ['alltime', None]:
        return PeriodType.ALLTIME
    elif period[1:].isdigit():
        return PeriodType.NUMERIC
    elif any(period.startswith(p) for p in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                                          'saturday', 'sunday', 'weekday', 'weekend', 'monthly', 
                                          'quarterly', 'yearly']) or period.endswith('_session'):
        return PeriodType.CATEGORICAL
    else:
        return PeriodType.FIXED_TIME

def validate_window_type(period: Union[str, int], window_type: str) -> bool:
    """Validate window type with error handling"""
    try:
        if isinstance(period, int):
            period = f'{period}'
        period_type = get_period_type(period)
        
        if window_type not in ALLOWED_WINDOW_TYPES[period_type]:
            raise ValidationError(
                field="window_type",
                value=window_type,
                constraint=f"must be one of {ALLOWED_WINDOW_TYPES[period_type]}"
            )
        return True
        
    except Exception as e:
        raise ConfigurationError(
            parameter="window_type",
            value=window_type,
            reason=str(e)
        )