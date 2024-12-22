from .base import FeatureCalculator
import pandas as pd
import json
from pathlib import Path

class SeasonalityCalculator(FeatureCalculator):
    """Calculate pure seasonality features (time-based categorization only)"""
    
    def __init__(self, timeframe: str = "1h", config_path: str = None):
        super().__init__(timeframe=timeframe)
        self.config = self._load_config(config_path)
        self._validate_sessions(self.config['time_features']['sessions'])
        
        # Calendar-based session definitions (known in advance)
        self.sessions = {
            'us': (lambda x: 14 <= x.hour < 21),
            'asia': (lambda x: x.hour >= 22 or x.hour < 7),
            'london': (lambda x: 8 <= x.hour < 16),
            'first': (lambda x: 0 <= x.hour < 6),
            'second': (lambda x: 6 <= x.hour < 12),
            'third': (lambda x: 12 <= x.hour < 18),
            'last': (lambda x: 18 <= x.hour < 24)
        }
    
    def get_required_columns(self) -> list:
        return []  # No price/volume data needed for pure seasonality
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load seasonality configuration from JSON or use defaults"""
        default_config = {
            'time_features': {
                'quarter': True,
                'sessions': {
                    'session_1': ['00:00', '06:00'],
                    'session_2': ['06:00', '12:00'],
                    'session_3': ['12:00', '18:00'],
                    'session_4': ['18:00', '00:00']
                }
            },
            'events': {
                'economic_events': {
                    'FOMC': 'first_friday',
                    'USD_GDP': ['2023-01-26', '2023-04-27', '2023-07-27', '2023-10-26'],
                    'US_ELECTION': ['2020-11-03', '2024-11-05']
                },
                'major_events': {
                    # 2020 COVID-19 related
                    'COVID_EMERGENCY': ['2020-03-13'],  # US National Emergency Declaration
                    'FED_UNLIMITED_QE': ['2020-03-23'],  # Fed announces unlimited QE
                    # 2023 Banking Crisis
                    'SVB_COLLAPSE': ['2023-03-10'],  # Silicon Valley Bank collapse
                    'SIGNATURE_CLOSURE': ['2023-03-12'],  # Signature Bank closure
                    'CREDIT_SUISSE_DEAL': ['2023-03-19'],  # UBS agrees to buy Credit Suisse
                    # Other significant events
                    'RUSSIA_UKRAINE': ['2022-02-24'],  # Russia-Ukraine conflict begins
                    'FTX_BANKRUPTCY': ['2022-11-11'],  # FTX bankruptcy filing
                    'CRYPTO_SEC': ['2023-06-13']  # SEC sues Binance and Coinbase
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.info(f"Loaded events: {config['events']['economic_events'].keys()}")
                return config
        
        return default_config
    
    def calculate(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Calculate pure seasonality features"""
        result_df = super().calculate(df, symbol)
        base_feature = self.timeframe
        
        # Quarter
        if self.config['time_features']['quarter']:
            quarter_feature = self.get_feature_name(symbol, f'{base_feature}_quarter')
            result_df[quarter_feature] = df.index.quarter
            self.log_feature_creation(quarter_feature, result_df)
        
        # Halving cycle year (1-4)
        cycle_feature = self.get_feature_name(symbol, f'{base_feature}_cycle_year')
        result_df[cycle_feature] = df.index.year.map(
            lambda x: ((x - 2016) % 4) + 1
        )
        self.log_feature_creation(cycle_feature, result_df)
        
        # Weekend flags (Friday 23:00 to Sunday 17:00)
        weekend_feature = self.get_feature_name(symbol, f'{base_feature}_is_weekend')
        result_df[weekend_feature] = (
            # Friday after 23:00
            ((df.index.dayofweek == 4) & (df.index.hour >= 23)) |
            # All of Saturday
            (df.index.dayofweek == 5) |
            # Sunday until 17:00
            ((df.index.dayofweek == 6) & (df.index.hour < 17))
        ).astype(int)
        self.log_feature_creation(weekend_feature, result_df)
        
        # Session markers (using >= start and < end to avoid overlaps)
        for session_name, hours in self.config['time_features']['sessions'].items():
            start_time = pd.to_datetime(hours[0]).time()
            end_time = pd.to_datetime(hours[1]).time()
            
            session_feature = self.get_feature_name(symbol, f'{base_feature}_is_{session_name}')
            
            # Handle sessions that cross midnight
            if start_time > end_time:
                result_df[session_feature] = (
                    (df.index.time >= start_time) | 
                    (df.index.time < end_time)
                ).astype(int)
            else:
                result_df[session_feature] = (
                    (df.index.time >= start_time) & 
                    (df.index.time < end_time)
                ).astype(int)
            
            self.log_feature_creation(session_feature, result_df)
        
        # Event features
        for event_category in ['economic_events', 'major_events']:
            for event_name, dates in self.config['events'][event_category].items():
                event_feature = self.get_feature_name(symbol, f'{base_feature}_is_{event_name}')
                
                if dates == "first_friday":
                    result_df[event_feature] = (
                        (df.index.day <= 7) & 
                        (df.index.dayofweek == 4)
                    ).astype(int)
                else:
                    # Convert dates to datetime and normalize to midnight
                    event_dates = pd.to_datetime(dates).normalize()
                    # Convert index dates to midnight for comparison
                    index_dates = df.index.normalize()
                    
                    # Create event flag
                    result_df[event_feature] = (
                        index_dates.isin(event_dates)
                    ).astype(int)
                    
                    # For FOMC, GDP, Elections, and major events, mark the entire day
                    if event_name in ['FOMC', 'USD_GDP', 'US_ELECTION'] or event_category == 'major_events':
                        event_mask = result_df[event_feature] == 1
                        date_mask = df.index.normalize().isin(index_dates[event_mask])
                        result_df.loc[date_mask, event_feature] = 1
                
                self.log_feature_creation(event_feature, result_df)
                
                # Debug log for specific events
                if event_name in ['US_ELECTION', 'COVID_EMERGENCY', 'FTX_BANKRUPTCY']:
                    self.logger.info(f"{event_name} dates in data: {event_dates}")
                    self.logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
                    self.logger.info(f"{event_name} flags set: {result_df[event_feature].sum()}")
        
        return result_df
    
    def _validate_sessions(self, sessions: dict) -> None:
        """Validate that sessions don't overlap"""
        def time_to_minutes(t: str) -> int:
            h, m = map(int, t.split(':'))
            return h * 60 + m
        
        # Convert all sessions to minute ranges
        session_ranges = []
        for name, (start, end) in sessions.items():
            start_mins = time_to_minutes(start)
            end_mins = time_to_minutes(end)
            if end_mins <= start_mins:  # Handle overnight sessions
                end_mins += 24 * 60
            session_ranges.append((name, start_mins, end_mins))
        
        # Check for overlaps in non-market sessions (1-6)
        day_sessions = [s for s in session_ranges if s[0].startswith('session_')]
        for i, (name1, start1, end1) in enumerate(day_sessions):
            for name2, start2, end2 in day_sessions[i+1:]:
                if (start1 < end2 and start2 < end1):
                    raise ValueError(f"Sessions {name1} and {name2} overlap")