import logging
import traceback

import pandas as pd
from pathlib import Path
import pickle
from typing import List, Optional, Dict
from tqdm import tqdm
import requests
from binance.client import Client
from datetime import datetime, timedelta
import time
import numpy as np
import json
import yfinance as yf
import calendar
import databento as db


class BinanceDataCollector:
    def __init__(
            self,
            symbols: List[str] = None,
            interval: str = None,
            lookback_days: int = None,
            client_id: str = "",
            client_secret: str = ""
    ):
        """
        Initialize the data collector with trading pairs and timeframe parameters

        Args:
            symbols: List of trading pairs to collect
            interval: Timeframe for the candles ('1m', '5m', '15m', '1h', '4h', '1d', etc)
            lookback_days: Number of days of historical data to fetch
            client_id: Binance API key (optional)
            client_secret: Binance API secret (optional)
        """
        self.symbols = symbols
        self.interval = interval
        self.lookback_days = lookback_days
        self.client = Client(client_id, client_secret)
        self.databento_client = db.Historical("db-CLnuRBBp7tNPexMqAW3iqmvVEA7PK")
        self.deribit_session = requests.Session()  # Add session for Deribit API calls
        self.setup_logging()

    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def fetch_vix_history(self, start_str: str, end_str: str) -> pd.DataFrame:
        """
        Fetch VIX historical data from Yahoo Finance
        
        Args:
            start_str: Start date in 'YYYY-MM-DD' format
            end_str: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with timestamp index and VIX data
        """
        try:
            logging.info(f"Fetching VIX data from {start_str} to {end_str}")
            
            # Try to fetch VIX data with 1h interval first
            vix = yf.download('^VIX', start=start_str, end=end_str, interval='1h')
            
            if not vix.empty:
                logging.info("Successfully fetched hourly VIX data")
                # Log unique hours in the data
                unique_hours = sorted(vix.index.hour.unique())
                logging.info(f"VIX data hours: {unique_hours}")
                # Log unique days of week (0=Monday, 6=Sunday)
                unique_days = sorted(vix.index.dayofweek.unique())
                logging.info(f"VIX trading days: {[calendar.day_name[d] for d in unique_days]}")
                # Log average records per day
                days_covered = (vix.index.max() - vix.index.min()).days
                avg_records_per_day = len(vix) / days_covered if days_covered > 0 else 0
                logging.info(f"VIX average records per day: {avg_records_per_day:.1f}")
            else:
                logging.warning("No hourly VIX data available, falling back to daily data")
                vix = yf.download('^VIX', start=start_str, end=end_str, interval='1d')
            
            if vix.empty:
                raise ValueError("No VIX data retrieved")
            
            # Rename columns for consistency
            vix = vix.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate returns
            vix['returns'] = vix['close'].pct_change()
            
            # Log statistics
            logging.info(f"Retrieved {len(vix)} VIX records")
            logging.info(f"VIX date range: {vix.index[0]} to {vix.index[-1]}")
            logging.info(f"Data frequency: {vix.index.freq if vix.index.freq else '1d'}")
            logging.info(f"Average VIX: {vix['close'].mean():.2f}")
            logging.info(f"Min VIX: {vix['close'].min():.2f}")
            logging.info(f"Max VIX: {vix['close'].max():.2f}")
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            vix_percentiles = vix['close'].quantile([p/100 for p in percentiles])
            
            # Log percentiles
            for p, v in zip(percentiles, vix_percentiles):
                logging.info(f"VIX P{p:02d}: {v:.2f}")
            
            return vix
            
        except Exception as e:
            logging.error(f"Error fetching VIX data: {str(e)}")
            return pd.DataFrame()

    def fetch_funding_rate_history(self, symbol: str, start_str: str, end_str: str):
        """Fetch historical funding rates using public endpoint"""
        try:
            start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)

            all_funding_rates = []
            current_start = start_ts

            # Initialize progress bar
            pbar = tqdm(
                total=(end_ts - start_ts) / (1000 * 3600 * 8),  # Funding rate every 8 hours
                desc=f"Fetching {symbol} funding rates",
                unit="rates"
            )

            while current_start < end_ts:
                try:
                    funding_rates = self.client.futures_funding_rate(
                        symbol=symbol,
                        startTime=current_start,
                        endTime=end_ts,
                        limit=1000
                    )

                    if not funding_rates:
                        break

                    all_funding_rates.extend(funding_rates)
                    pbar.update(len(funding_rates))

                    # Update start time for next batch
                    current_start = int(funding_rates[-1]['fundingTime']) + 1

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    logging.error(f"Error in funding rate batch: {e}")
                    break

            pbar.close()

            if not all_funding_rates:
                logging.warning(f"No funding rate data available for {symbol}")
                return pd.DataFrame()

            # Create DataFrame with more robust error handling
            df = pd.DataFrame(all_funding_rates)

            # Convert funding time with error handling and round to nearest 8-hour interval
            df['fundingTime'] = pd.to_datetime(df['fundingTime'].astype(float), unit='ms', errors='coerce')

            # Create function to round to nearest 8-hour interval
            def round_to_8hours(ts):
                hour = ts.hour
                rounded_hour = (hour // 8) * 8  # Round down to nearest 8-hour interval
                return ts.replace(hour=rounded_hour, minute=0, second=0, microsecond=0, nanosecond=0)

            # Apply rounding to fundingTime
            df['fundingTime'] = df['fundingTime'].apply(round_to_8hours)

            # Convert fundingRate with error handling
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')

            # Drop any rows with NaT in fundingTime
            df = df.dropna(subset=['fundingTime'])

            if df.empty:
                logging.warning(f"No valid funding rate data after processing for {symbol}")
                return pd.DataFrame()

            # Set index and sort, ensuring no duplicates after rounding
            df = df.groupby('fundingTime')['fundingRate'].mean().reset_index()  # Handle any duplicate times after rounding
            df = df.set_index('fundingTime')
            df = df.sort_index()

            logging.info(f"Retrieved {len(df)} funding rates for {symbol}")

            return df[['fundingRate']]

        except Exception as e:
            logging.error(f"Error fetching funding rates for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_deribit_dvol(self, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch BTC 30-day implied volatility index (DVOL) from Deribit"""
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

        # DVOL launch date
        dvol_launch = datetime.strptime('2021-03-24', '%Y-%m-%d')
        start_date = datetime.strptime(start_str, '%Y-%m-%d')

        # If requested start date is before DVOL launch, adjust to launch date
        if start_date < dvol_launch:
            logging.info(f"Adjusting DVOL start date from {start_str} to 2021-03-24 (DVOL launch date)")
            start_str = '2021-03-24'

        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)

            logging.info(f"Fetching DVOL data from {start_str} to {end_str}")

            # Calculate expected number of data points
            total_hours = (end_ts - start_ts) / (1000 * 3600)  # Convert ms to hours
            expected_points = int(total_hours)

            all_dvol_data = []
            current_end = end_ts

            # Initialize progress bar
            pbar = tqdm(
                total=expected_points,
                desc="Fetching DVOL data",
                unit="points"
            )

            while current_end > start_ts:
                params = {
                    "currency": "BTC",  # BTC DVOL index
                    "start_timestamp": start_ts,
                    "end_timestamp": current_end,
                    "resolution": "3600",  # 1 hour in seconds
                    "count": 1000  # Maximum points per request
                }

                response = self.deribit_session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                logging.debug(f"DVOL API Response: {data}")  # Add this line

                if not data.get('result', {}).get('data'):
                    logging.warning("No data in DVOL response")
                    break

                dvol_data = data['result']['data']
                if not dvol_data:
                    logging.warning("Empty DVOL data array")
                    break

                # Log first and last points in this batch
                logging.info(f"DVOL batch: {len(dvol_data)} points from "
                             f"{datetime.fromtimestamp(dvol_data[0][0] / 1000)} to "
                             f"{datetime.fromtimestamp(dvol_data[-1][0] / 1000)}")

                # Prepend data since we're going backwards in time
                all_dvol_data = dvol_data + all_dvol_data

                # Update progress bar
                points_received = len(dvol_data)
                pbar.update(points_received)

                # Update end timestamp for next batch
                current_end = dvol_data[0][0] - 1

                time.sleep(0.1)

            pbar.close()

            if not all_dvol_data:
                logging.warning("No DVOL data collected")
                return pd.DataFrame(columns=['timestamp', 'implied_volatility'])

            logging.info(f"Total DVOL points collected: {len(all_dvol_data)}")

            # Convert to DataFrame with proper column names
            df = pd.DataFrame(all_dvol_data, columns=['timestamp', 'open', 'high', 'low', 'close'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            # Use close price as the implied volatility value
            df['implied_volatility'] = df['close']

            # Sort index to ensure chronological order
            df = df.sort_index()

            # Log the date range of the final DataFrame
            logging.info(f"DVOL data range: {df.index[0]} to {df.index[-1]}")

            # Resample to 1H and forward fill any gaps
            df = df.resample('1H').ffill()

            return df[['implied_volatility']]

        except Exception as e:
            logging.error(f"Error fetching DVOL data: {str(e)}")
            logging.exception("Full traceback:")  # This will log the full stack trace
            return pd.DataFrame(columns=['timestamp', 'implied_volatility'])

    def fetch_historical_klines(self, symbol: str, start_str: str, end_str: Optional[str] = None):
        """Fetch historical klines with error handling"""
        try:
            logging.info(f"Fetching {self.interval} data for {symbol}")
            start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000) if end_str else None

            # Calculate expected number of klines based on interval and time range
            interval_seconds = {
                '1m': 60,
                '3m': 180,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '2h': 7200,
                '4h': 14400,
                '6h': 21600,
                '8h': 28800,
                '12h': 43200,
                '1d': 86400,
                '3d': 259200,
                '1w': 604800
            }

            interval_secs = interval_seconds.get(self.interval)
            if not interval_secs:
                raise ValueError(f"Unsupported interval: {self.interval}")

            total_seconds = (end_ts - start_ts) / 1000  # Convert ms to seconds
            expected_klines = int(total_seconds / interval_secs)

            # Get base klines data
            all_klines = []

            # Initialize progress bar with expected total
            pbar = tqdm(
                total=expected_klines,
                desc=f"Fetching {symbol} klines",
                unit="klines"
            )

            current_klines = 0
            while start_ts < end_ts:
                klines = self.client.futures_historical_klines(
                    symbol=symbol,
                    interval=self.interval,
                    start_str=start_ts,
                    end_str=end_ts,
                    limit=1500
                )

                if not klines:
                    break

                all_klines.extend(klines)
                current_klines += len(klines)
                pbar.update(len(klines))  # Update by actual number of klines received

                start_ts = klines[-1][0] + 1
                time.sleep(0.1)  # Rate limiting

            pbar.close()

            if not all_klines:
                raise ValueError(f"No data retrieved for {symbol}")

            logging.info(f"Retrieved {current_klines}/{expected_klines} klines for {symbol}")

            # Create base DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                        'trades', 'taker_buy_volume', 'taker_buy_quote_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate taker sell volumes
            df['taker_sell_quote_volume'] = df['quote_volume'] - df['taker_buy_quote_volume']

            df['volume'] = df['volume'] * df['close']  # Convert volume to USD
            df = df.set_index('timestamp')

            # Initialize funding rate column with NaN
            df['fundingRate'] = np.nan

            # Add funding rates if available
            funding_df = self.fetch_funding_rate_history(symbol, start_str, end_str)
            if not funding_df.empty:
                try:
                    # Resample to 8H frequency (funding rate interval) and forward fill
                    funding_df = funding_df.resample('8h').ffill()

                    # Update funding rate values directly instead of using merge
                    df.loc[funding_df.index, 'fundingRate'] = funding_df['fundingRate']

                    # Now resample to match kline interval if needed
                    if self.interval != '8h':
                        df['fundingRate'] = df['fundingRate'].fillna(method='ffill')

                    logging.info(f"Successfully added funding rates for {symbol}")
                    logging.info(f"Funding rate range: {df['fundingRate'].min():.6f} to {df['fundingRate'].max():.6f}")

                except Exception as e:
                    logging.error(f"Error processing funding rates for {symbol}: {e}")
                    df['fundingRate'] = np.nan

            # Fill NaN values with -1 only after all processing
            df['fundingRate'] = df['fundingRate'].fillna(-1.0)

            # Log summary statistics
            self._log_summary_statistics(df, symbol)

            # Prepare the base columns list
            base_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                            'trades', 'taker_buy_quote_volume',
                            'taker_sell_quote_volume', 'fundingRate']

            df['implied_volatility'] = np.nan

            # Fetch DVOL data if this is BTC
            if symbol == 'BTCUSDT':
                try:
                    dvol_df = self.fetch_deribit_dvol(start_str, end_str)
                    if not dvol_df.empty and 'implied_volatility' in dvol_df.columns:
                        # Ensure indices are timezone-naive for proper alignment
                        dvol_df.index = dvol_df.index.tz_localize(None)

                        # Resample to match our interval and forward fill
                        dvol_df = dvol_df.resample(self.interval).ffill()

                        # Only update values where indices intersect
                        common_idx = df.index.intersection(dvol_df.index)
                        df.loc[common_idx, 'implied_volatility'] = dvol_df.loc[common_idx, 'implied_volatility']

                        base_columns.append('implied_volatility')

                        logging.info(f"Added DVOL data from {dvol_df.index[0]} onwards")
                    else:
                        logging.warning("No valid DVOL data available, skipping implied volatility")
                except Exception as e:
                    logging.error(f"Error processing DVOL data: {str(e)}")
                    # Continue without implied volatility data

            # Return only the columns we have successfully processed
            available_columns = [col for col in base_columns if col in df.columns]
            return df[available_columns]

        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def _log_summary_statistics(self, df: pd.DataFrame, symbol: str):
        """Log summary statistics for the given symbol's data"""
        try:
            print(f"\n{symbol} Statistics:")
            print("-" * 30)
            print(f"Records: {len(df)}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Average Price: ${df['close'].mean():.2f}")
            print(f"Min Price: ${df['close'].min():.2f}")
            print(f"Max Price: ${df['close'].max():.2f}")
            
            # Calculate price percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            price_percentiles = df['close'].quantile([p/100 for p in percentiles])
            
            print("\nPrice Percentiles:")
            for p, v in zip(percentiles, price_percentiles):
                print(f"P{p:02d}: ${v:.2f}")
                
            # Volume statistics
            print("\nVolume Statistics:")
            print(f"Average Daily Volume: ${df['volume'].mean():,.2f}")
            print(f"Max Daily Volume: ${df['volume'].max():,.2f}")
            
            # Funding rate statistics if available
            if 'fundingRate' in df.columns:
                funding_rates = df['fundingRate'][df['fundingRate'] != -1]  # Exclude placeholder values
                if not funding_rates.empty:
                    print("\nFunding Rate Statistics:")
                    print(f"Average Funding Rate: {funding_rates.mean():.4%}")
                    print(f"Min Funding Rate: {funding_rates.min():.4%}")
                    print(f"Max Funding Rate: {funding_rates.max():.4%}")
                    
                    # Funding rate percentiles
                    funding_percentiles = funding_rates.quantile([p/100 for p in percentiles])
                    print("\nFunding Rate Percentiles:")
                    for p, v in zip(percentiles, funding_percentiles):
                        print(f"P{p:02d}: {v:.4%}")
            
            print("\n" + "=" * 50)

        except Exception as e:
            logging.error(f"Error calculating summary statistics for {symbol}: {str(e)}")

    def save_to_csv(self, data_dict: Dict[str, pd.DataFrame], params: dict) -> str:
        """Save market data to CSV files with metadata"""
        # Create data directory if it doesn't exist
        base_dir = Path("data/raw")  # Updated base path
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp-based directory for this dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_dir = base_dir / f"market_data_{self.interval}_{self.lookback_days}d_{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories only for the exchanges we have data for
        if any(symbol == 'VIX' for symbol in data_dict.keys()):
            cboe_dir = dataset_dir / 'cboe'
            cboe_dir.mkdir(parents=True, exist_ok=True)

        if any(symbol != 'VIX' for symbol in data_dict.keys()):
            binance_dir = dataset_dir / 'binance-futures'
            binance_dir.mkdir(parents=True, exist_ok=True)

        # Get the schema from the first dataframe (they should all have the same schema)
        first_symbol = next(iter(data_dict))
        first_df = data_dict[first_symbol]
        schema = {
            'columns': list(first_df.columns),
            'dtypes': {col: str(dtype) for col, dtype in first_df.dtypes.items()}
        }

        # Add schema to params
        params['schema'] = schema

        # Save metadata as CSV
        metadata_file = dataset_dir / "metadata.csv"
        metadata_df = pd.DataFrame([params])
        metadata_df.to_csv(metadata_file, index=False)
        logging.info(f"Saved metadata to {metadata_file}")

        # Save each symbol's data to a separate CSV file
        with tqdm(total=len(data_dict), desc="Saving data to CSV files") as pbar:
            for symbol, df in data_dict.items():
                if symbol == 'VIX':
                    filename = dataset_dir / 'cboe' / f"{symbol}.csv"
                else:
                    filename = dataset_dir / 'binance-futures' / f"{symbol}.csv"
                df.to_csv(filename)
                pbar.update(1)
                logging.info(f"Saved {symbol} data to {filename}")

        return str(dataset_dir)

    def load_from_csv(self, directory: str) -> tuple:
        """Load data from CSV files and return data dict and params"""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} not found")

        # Load parameters from JSON
        params_file = directory / "parameters.json"
        with open(params_file, 'r') as f:
            params = json.load(f)

        # Load all symbol data files from both directories
        data = {}
        
        # Get all possible exchange directories
        exchange_dirs = [d for d in directory.iterdir() if d.is_dir()]
        
        # Count total files to process (excluding parameters.json)
        total_files = sum(len(list(d.glob("*.csv"))) for d in exchange_dirs)

        with tqdm(total=total_files, desc="Loading data from CSV files") as pbar:
            # Load data from each exchange directory
            for exchange_dir in exchange_dirs:
                for file in exchange_dir.glob("*.csv"):
                    symbol = file.stem
                    df = pd.read_csv(file, index_col=0, parse_dates=True)
                    data[symbol] = df
                    pbar.update(1)
                    logging.info(f"Loaded {len(df)} records for {symbol}")

        return data, params

    def fetch_and_store_data(self) -> str:
        """Fetch data for configured symbols and store locally"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        data_dict = {}
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                if symbol == 'VIX':
                    # Fetch VIX data from Yahoo Finance
                    df = self.fetch_vix_history(start_str, end_str)
                    if not df.empty:
                        data_dict['VIX'] = df
                        logging.info(f"Successfully fetched {len(df)} VIX records")
                elif symbol == 'NASDAQ':
                    # Fetch NASDAQ futures data from Yahoo Finance
                    df = self.fetch_nasdaq_history(start_str, end_str)
                    if not df.empty:
                        data_dict['NASDAQ'] = df
                        logging.info(f"Successfully fetched {len(df)} NASDAQ futures records")
                else:
                    # Fetch Binance data
                    df = self.fetch_historical_klines(
                        symbol=symbol,
                        start_str=start_str,
                        end_str=end_str
                    )
                    data_dict[symbol] = df
                    self._log_summary_statistics(df, symbol)
                    logging.info(f"Successfully fetched {len(df)} records for {symbol}")

            except Exception as e:
                logging.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"data/raw/market_data_{self.interval}_{self.lookback_days}d_{timestamp}")
        
        # Create subdirectories for each exchange
        binance_dir = output_dir / "binance-futures"
        cboe_dir = output_dir / "cboe"
        nasdaq_dir = output_dir / "nasdaq"
        
        # Create all directories
        binance_dir.mkdir(parents=True, exist_ok=True)
        cboe_dir.mkdir(parents=True, exist_ok=True)
        nasdaq_dir.mkdir(parents=True, exist_ok=True)

        # Save data for each symbol
        for symbol, df in data_dict.items():
            try:
                if symbol == 'VIX':
                    # Save VIX data to CBOE directory
                    filepath = cboe_dir / f"{symbol}.csv"
                elif symbol == 'NASDAQ':
                    # Save NASDAQ futures data to NASDAQ directory
                    filepath = nasdaq_dir / f"{symbol}.csv"
                else:
                    # Save Binance data to Binance directory
                    filepath = binance_dir / f"{symbol}.csv"
                
                df.to_csv(filepath)
                logging.info(f"Saved {symbol} data to {filepath}")
                
            except Exception as e:
                logging.error(f"Failed to save data for {symbol}: {e}")
                continue

        # Save parameters
        params = {
            'interval': self.interval,
            'lookback_days': self.lookback_days,
            'symbols': self.symbols,
            'start_date': start_str,
            'end_date': end_str,
            'timestamp': timestamp
        }
        
        with open(output_dir / 'parameters.json', 'w') as f:
            json.dump(params, f, indent=4)
            
        return str(output_dir)

    def load_data(self, directory: str) -> tuple:
        """Load data from CSV files and return data dict and params"""
        return self.load_from_csv(directory)

    def fetch_nasdaq_history(self, start_str: str, end_str: str) -> pd.DataFrame:
        """
        Fetch NASDAQ futures (NQ) historical data from Yahoo Finance
        
        Args:
            start_str: Start date in 'YYYY-MM-DD' format
            end_str: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with timestamp index and NASDAQ futures data
        """
        try:
            logging.info(f"Fetching NASDAQ futures data from {start_str} to {end_str}")
            
            # Try to fetch NASDAQ futures data with 1h interval
            nasdaq = yf.download('NQ=F', start=start_str, end=end_str, interval='1h')
            
            if not nasdaq.empty:
                logging.info("Successfully fetched hourly NASDAQ futures data")
                # Log unique hours in the data
                unique_hours = sorted(nasdaq.index.hour.unique())
                logging.info(f"NASDAQ futures data hours: {unique_hours}")
                # Log unique days of week (0=Monday, 6=Sunday)
                unique_days = sorted(nasdaq.index.dayofweek.unique())
                logging.info(f"NASDAQ futures trading days: {[calendar.day_name[d] for d in unique_days]}")
                # Log average records per day
                days_covered = (nasdaq.index.max() - nasdaq.index.min()).days
                avg_records_per_day = len(nasdaq) / days_covered if days_covered > 0 else 0
                logging.info(f"NASDAQ futures average records per day: {avg_records_per_day:.1f}")
            else:
                logging.warning("No hourly NASDAQ futures data available, falling back to daily data")
                nasdaq = yf.download('NQ=F', start=start_str, end=end_str, interval='1d')
            
            if nasdaq.empty:
                raise ValueError("No NASDAQ futures data retrieved")
            
            # Rename columns for consistency
            nasdaq = nasdaq.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Calculate returns
            nasdaq['returns'] = nasdaq['close'].pct_change()
            
            # Log statistics
            logging.info(f"Retrieved {len(nasdaq)} NASDAQ futures records")
            logging.info(f"NASDAQ futures date range: {nasdaq.index[0]} to {nasdaq.index[-1]}")
            logging.info(f"Data frequency: {nasdaq.index.freq if nasdaq.index.freq else '1d'}")
            logging.info(f"Average NASDAQ futures: {nasdaq['close'].mean():.2f}")
            logging.info(f"Min NASDAQ futures: {nasdaq['close'].min():.2f}")
            logging.info(f"Max NASDAQ futures: {nasdaq['close'].max():.2f}")
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            nasdaq_percentiles = nasdaq['close'].quantile([p/100 for p in percentiles])
            
            # Log percentiles
            for p, v in zip(percentiles, nasdaq_percentiles):
                logging.info(f"NASDAQ futures P{p:02d}: {v:.2f}")
            
            return nasdaq
            
        except Exception as e:
            logging.error(f"Error fetching NASDAQ futures data: {str(e)}")
            return pd.DataFrame()

    def fetch_klines(self, symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch historical klines (candlestick) data from Binance Futures
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_str: Start date in 'YYYY-MM-DD' format
            end_str: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)
            
            logging.info(f"Fetching {symbol} klines from {start_str} to {end_str}")
            
            # Calculate expected number of candles
            interval_seconds = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '8h': 28800,
                '12h': 43200, '1d': 86400, '3d': 259200, '1w': 604800
            }
            seconds_per_candle = interval_seconds.get(self.interval, 3600)  # Default to 1h if interval not found
            expected_candles = int((end_ts - start_ts) / (seconds_per_candle * 1000))
            
            all_klines = []
            current_start = start_ts
            
            # Initialize progress bar
            pbar = tqdm(
                total=expected_candles,
                desc=f"Fetching {symbol} klines",
                unit="candles"
            )
            
            while current_start < end_ts:
                try:
                    # Fetch batch of klines
                    klines = self.client.futures_historical_klines(
                        symbol=symbol,
                        interval=self.interval,
                        start_str=str(current_start),
                        end_str=str(end_ts),
                        limit=1000
                    )
                    
                    if not klines:
                        break
                        
                    all_klines.extend(klines)
                    pbar.update(len(klines))
                    
                    # Update start time for next batch
                    current_start = int(klines[-1][0]) + 1
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logging.error(f"Error in klines batch: {e}")
                    break
                    
            pbar.close()
            
            if not all_klines:
                logging.warning(f"No kline data available for {symbol}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_volume', 'trades', 'taker_buy_volume',
                             'taker_buy_quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set index and sort
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Calculate taker sell quote volume
            df['taker_sell_quote_volume'] = df['quote_volume'] - df['taker_buy_quote_volume']
            
            # Drop unnecessary columns
            df = df.drop(['close_time', 'ignore', 'taker_buy_volume'], axis=1)
            
            # Add funding rates
            df['fundingRate'] = np.nan
            funding_df = self.fetch_funding_rate_history(symbol, start_str, end_str)
            if not funding_df.empty:
                try:
                    # Resample to 8H frequency (funding rate interval) and forward fill
                    funding_df = funding_df.resample('8H').ffill()
                    
                    # Update funding rate values
                    df.loc[funding_df.index, 'fundingRate'] = funding_df['fundingRate']
                    
                    # Forward fill funding rates
                    df['fundingRate'] = df['fundingRate'].fillna(method='ffill')
                    
                    logging.info(f"Successfully added funding rates for {symbol}")
                    
                except Exception as e:
                    logging.error(f"Error processing funding rates for {symbol}: {e}")
            
            # Fill any remaining NaN funding rates with -1
            df['fundingRate'] = df['fundingRate'].fillna(-1)
            
            # Add implied volatility column (only for BTC)
            df['implied_volatility'] = np.nan
            if symbol == 'BTCUSDT':
                try:
                    dvol_df = self.fetch_deribit_dvol(start_str, end_str)
                    if not dvol_df.empty and 'implied_volatility' in dvol_df.columns:
                        # Ensure indices are timezone-naive
                        dvol_df.index = dvol_df.index.tz_localize(None)
                        
                        # Resample to match interval and forward fill
                        dvol_df = dvol_df.resample(self.interval).ffill()
                        
                        # Update implied volatility values
                        common_idx = df.index.intersection(dvol_df.index)
                        df.loc[common_idx, 'implied_volatility'] = dvol_df.loc[common_idx, 'implied_volatility']
                        
                        logging.info(f"Added DVOL data from {dvol_df.index[0]} onwards")
                        
                except Exception as e:
                    logging.error(f"Error processing DVOL data: {e}")
            
            # Log statistics
            logging.info(f"Retrieved {len(df)} klines for {symbol}")
            logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logging.info(f"Average price: {df['close'].mean():.2f}")
            logging.info(f"Min price: {df['close'].min():.2f}")
            logging.info(f"Max price: {df['close'].max():.2f}")
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            price_percentiles = df['close'].quantile([p/100 for p in percentiles])
            
            # Log percentiles
            for p, v in zip(percentiles, price_percentiles):
                logging.info(f"{symbol} P{p:02d}: {v:.2f}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_databento_data(self, symbol: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch and resample data from Databento"""
        # Map symbols to Databento dataset and symbols for front-month futures
        symbol_mapping = {
            "NASDAQ": ("GLBX.MDP3", ["NQ.c.1"]),  # CME E-mini NASDAQ-100 front month
        }
        
        if symbol not in symbol_mapping:
            raise ValueError(f"Symbol {symbol} not supported for Databento")
            
        dataset, db_symbols = symbol_mapping[symbol]
        
        # Get OHLCV data directly in 1-hour intervals
        data = self.databento_client.timeseries.get_range(
            dataset=dataset,
            start=start_str,
            end=end_str,
            symbols=db_symbols,
            stype_in="continuous",  # Use continuous front-month contract
            schema="ohlcv-1h"
        )
        
        # Convert to DataFrame and format
        df = data.to_df()
        
        # Rename columns to match our standard format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'quote_volume'
        })
        
        # Add required columns with default values
        df['taker_buy_quote_volume'] = df['quote_volume'] * 0.5  # Estimate
        
        return df


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Define collection parameters
    crypto_symbols = ["BTCUSDT", "ETHUSDT", 'LTCUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'MATICUSDT', 'DOTUSDT']

    trad_symbols = ["NASDAQ"]
    symbols = crypto_symbols + trad_symbols
    interval = "1h"
    lookback_days = 2425
    
    # Initialize collector
    collector = BinanceDataCollector(symbols=symbols, interval=interval, lookback_days=lookback_days)
    
    # Calculate date range
    end_date = datetime(2024, 12, 15)
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"data/raw/market_data_{lookback_days}d_{timestamp}")
        
        # Create exchange directories
        exchange_dirs = {
            "binance": output_dir / "binance-futures",
            "cme": output_dir / "cme-futures",
        }
        
        for directory in exchange_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize collection tracking
        collected_data = {
            "binance-futures": {"symbols": [], "data_types": set()},
            "cme-futures": {"symbols": [], "data_types": set()},
            "cboe-futures": {"symbols": [], "data_types": set()}
        }
        
        # Process each symbol
        for symbol in symbols:
            logging.info(f"\nProcessing {symbol}")
            
            try:
                if symbol == "NASDAQ":
                    df = collector.fetch_databento_data(symbol, start_str, end_str)
                    if not df.empty:
                        output_file = exchange_dirs["cme"] / f"NQ1_{interval}.csv"
                        df.to_csv(output_file)
                        logging.info(f"Saved NASDAQ futures (NQ1) data")
                        collected_data["cme-futures"]["symbols"].append("NQ1")
                        collected_data["cme-futures"]["data_types"].add("OHLCV")

                
                else:  # Crypto symbols
                    df = collector.fetch_klines(symbol, start_str, end_str)
                    if not df.empty:
                        output_file = exchange_dirs["binance"] / f"{symbol}_{interval}.csv"
                        df.to_csv(output_file)
                        logging.info(f"Saved {symbol} OHLCV data")
                        collected_data["binance-futures"]["symbols"].append(symbol)
                        collected_data["binance-futures"]["data_types"].add("OHLCV")
                        
                        # Get funding rates for crypto
                        funding_df = collector.fetch_funding_rate_history(symbol, start_str, end_str)
                        if not funding_df.empty:
                            funding_file = exchange_dirs["binance"] / f"{symbol}_funding.csv"
                            funding_df.to_csv(funding_file)
                            logging.info(f"Saved {symbol} funding rate data")
                            collected_data["binance-futures"]["data_types"].add("funding_rates")
            
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Clean up metadata - remove empty exchanges and convert sets to lists
        collected_data = {
            exchange: {
                "symbols": data["symbols"],
                "data_types": list(data["data_types"])
            }
            for exchange, data in collected_data.items()
            if data["symbols"]  # Only keep exchanges with collected data
        }
        
        # Create metadata
        metadata = {
            "collection_time": timestamp,
            "interval": interval,
            "lookback_days": lookback_days,
            "start_date": start_str,
            "end_date": end_str,
            "exchanges": list(collected_data.keys()),
            "data_sources": {
                "binance-futures": "Binance Futures API",
                "cme-futures": "Databento CME MDP3",
                "cboe-futures": "Databento CBOE MDP3"
            },
            "collected_data": collected_data
        }
        
        # Save metadata
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        logging.info("\nData collection complete!")
        logging.info(f"Output directory: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error in data collection: {e}", exc_info=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
