# Custom Error Types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class MarketDataError(Exception):
    """Missing or invalid market data"""
    missing_fields: List[str]
    timestamp: pd.Timestamp

@dataclass
class SignalError(Exception):
    """Invalid signal generation"""
    signal_id: str
    reason: str

@dataclass
class OrderError(Exception):
    """Order execution error"""
    order_id: str
    reason: str

@dataclass
class PositionError(Exception):
    """Position management error"""
    reason: str
    position_details: Dict[str, Any]

@dataclass
class ConfigurationError(Exception):
    """Configuration validation error"""
    parameter: str
    value: Any
    reason: str

@dataclass
class ValidationError(Exception):
    """Data validation error"""
    field: str
    value: Any
    constraint: str

@dataclass
class StateError(Exception):
    """Invalid state transition or state validation error"""
    state_type: str
    current_state: Any
    attempted_transition: str
    reason: str

@dataclass
class DataOutputError(Exception):
    """Error during data output/saving"""
    operation: str  # e.g., 'save_trades', 'create_visualization'
    file_path: str
    reason: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class StrategyError(Exception):
    """Strategy-specific errors"""
    strategy_name: str
    operation: str  # e.g., 'signal_generation', 'position_sizing'
    reason: str
    state_details: Optional[Dict[str, Any]] = None

@dataclass
class MetricsError(Exception):
    """Error calculating metrics"""
    metric_name: str
    calculation: str
    reason: str
    data_details: Optional[Dict[str, Any]] = None