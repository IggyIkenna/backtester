"""Coinglass Liquidation Heatmap Data Collector

This script collects liquidation heatmap data from Coinglass API to visualize
price levels where liquidations may occur.

Key concepts:
- Liquidation levels indicate where leveraged trades might be forcibly closed
- Higher concentration of liquidations (yellow in heatmap) suggests potential price magnets
- These levels can act as support/resistance zones due to high liquidity
- The heatmap predicts initiation points of liquidations, not their full extent
"""

import logging
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from configs.data_catalog import DataCatalog

class CoinglassCollector:
    def __init__(self, api_key: str = "dc210e2151d94a2f9cfcaac2aad85370", min_liquidation_intensity: float = 0.5e10):
        """Initialize the Coinglass data collector
        
        Args:
            api_key: Coinglass API key
            min_liquidation_intensity: Minimum liquidation intensity to show in heatmap (default: 5B)
        """
        self.base_url = "https://open-api-v3.coinglass.com/api"
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "CG-API-KEY": self.api_key
        }
        self.catalog = DataCatalog()
        self.min_liquidation_intensity = min_liquidation_intensity
        
        # Only use the longest time range
        self.time_range = "1y"
        
        # Supported symbols (only BTC now)
        self.symbols = ["BTC"]
        
        # Binance data path
        self.binance_data_path = Path("backtester/data/raw/market_data_1h_1825d_20241215_050021/binance-futures")

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to the Coinglass API"""
        try:
            url = f"{self.base_url}/{endpoint}"
            logging.info(f"Making request to {url} with params: {params}")
            response = requests.get(url, headers=self.headers, params=params)
            
            logging.info(f"Response status code: {response.status_code}")
            
            try:
                data = response.json()
                logging.debug(f"Raw response data: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON response. Raw response: {response.text}")
                return None
            
            if not data.get("success"):
                logging.error(f"API error: Request unsuccessful")
                return None
                
            if not data.get("data"):
                logging.error(f"Empty data response from API: {data}")
                return None
                
            return data.get("data")
            
        except Exception as e:
            logging.error(f"Error making request to {endpoint}: {str(e)}")
            return None

    def _load_binance_data(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        """Load hourly candle data from Binance futures"""
        try:
            # Load Binance data
            binance_file = self.binance_data_path / f"{symbol}USDT.csv"
            if not binance_file.exists():
                logging.error(f"Binance data file not found: {binance_file}")
                return pd.DataFrame()
                
            # Read the CSV file
            df = pd.read_csv(binance_file)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to the required date range
            df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
            
            if df.empty:
                logging.error("No Binance data found in the specified date range")
                return df
                
            logging.info(f"Loaded {len(df)} hourly candles from Binance")
            logging.info(f"Binance data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading Binance data: {str(e)}")
            return pd.DataFrame()

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to daily OHLCV"""
        try:
            # Set timestamp as index for resampling
            df = df.set_index('timestamp')
            
            # Resample to daily frequency
            daily = pd.DataFrame()
            daily['open'] = df['open'].resample('D').first()
            daily['high'] = df['high'].resample('D').max()
            daily['low'] = df['low'].resample('D').min()
            daily['close'] = df['close'].resample('D').last()
            daily['volume'] = df['volume'].resample('D').sum()
            
            # Reset index to get timestamp as column
            daily = daily.reset_index()
            
            logging.info(f"Resampled {len(df)} hourly candles to {len(daily)} daily candles")
            return daily
            
        except Exception as e:
            logging.error(f"Error resampling to daily: {str(e)}")
            return df

    def fetch_liquidation_heatmap(self, symbol: str = "BTC") -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Fetch liquidation heatmap data from Coinglass and align with Binance hourly data
        
        Returns:
            Tuple of (price_df, heatmap_matrix, metadata) where:
            - price_df: DataFrame with OHLCV data from Binance
            - heatmap_matrix: 2D array of liquidation intensities
            - metadata: Additional information about the data
        """
        if symbol not in self.symbols:
            raise ValueError(f"Invalid symbol. Must be one of: {', '.join(self.symbols)}")
            
        try:
            endpoint = "futures/liquidation/model2/heatmap"
            params = {
                "exchange": "Binance",
                "symbol": f"{symbol}USDT",
                "range": self.time_range
            }
            
            response_data = self._make_request(endpoint, params)
            if not response_data:
                return pd.DataFrame(), np.array([]), {}
            
            # Extract data components
            y = response_data.get('y', [])  # Price levels
            liq = response_data.get('liq', [])  # Liquidation points [x_idx, y_idx, value]
            prices = response_data.get('prices', [])  # Daily OHLCV data
            
            if not y or not liq or not prices:
                logging.error("Missing required data in response")
                return pd.DataFrame(), np.array([]), {}
            
            # Create temporary DataFrame for date range
            temp_df = pd.DataFrame(prices, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='s')
            
            # Get date range from Coinglass data
            start_time = temp_df['timestamp'].min()
            end_time = temp_df['timestamp'].max()
            
            # Load hourly candles from Binance
            price_df = self._load_binance_data(symbol, start_time, end_time)
            if price_df.empty:
                return pd.DataFrame(), np.array([]), {}
            
            # Create empty heatmap matrix with hourly granularity
            heatmap_matrix = np.zeros((len(y), len(price_df)))
            
            # Create timestamp to index mapping for hourly data
            timestamp_to_idx = {ts: idx for idx, ts in enumerate(price_df['timestamp'])}
            
            # Fill heatmap matrix with liquidation values
            for x_idx, y_idx, value in liq:
                try:
                    x_idx = int(x_idx)
                    y_idx = int(y_idx)
                    if 0 <= x_idx < len(temp_df) and 0 <= y_idx < len(y):
                        # Get timestamp for this liquidation point
                        ts = temp_df.iloc[x_idx]['timestamp']
                        # Get the corresponding hour's start
                        ts_hour = ts.replace(minute=0, second=0, microsecond=0)
                        
                        # Find the corresponding index in hourly data
                        if ts_hour in timestamp_to_idx:
                            h_idx = timestamp_to_idx[ts_hour]
                            heatmap_matrix[y_idx, h_idx] = float(value)
                except (ValueError, TypeError, IndexError) as e:
                    logging.warning(f"Invalid liquidation point [{x_idx}, {y_idx}, {value}]: {e}")
                    continue
            
            # Forward fill gaps in the heatmap matrix
            for i in range(len(y)):
                row = heatmap_matrix[i, :]
                last_value = 0
                for j in range(len(row)):
                    if row[j] == 0:
                        heatmap_matrix[i, j] = last_value
                    else:
                        last_value = row[j]
            
            # Filter out small liquidation values
            heatmap_matrix[heatmap_matrix < self.min_liquidation_intensity] = 0
            logging.info(f"Filtered liquidation points below {self.min_liquidation_intensity:.2e}")
            logging.info(f"Remaining non-zero points: {np.count_nonzero(heatmap_matrix)}")
            
            # Extract metadata
            metadata = {
                'precision': response_data.get('precision', 0),
                'range_low': response_data.get('rangeLow', 0),
                'range_high': response_data.get('rangeHigh', 0),
                'instrument': response_data.get('instrument', {}),
                'price_levels': y,
                'min_liquidation_intensity': self.min_liquidation_intensity
            }
            
            if not price_df.empty:
                logging.info(f"Successfully processed {len(price_df)} hourly candles")
                logging.info(f"Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
                logging.info(f"Price range: {price_df['low'].min():.2f} to {price_df['high'].max():.2f}")
                logging.info(f"Heatmap shape: {heatmap_matrix.shape}")
                logging.info(f"Non-zero liquidation points: {np.count_nonzero(heatmap_matrix)}")
            
            return price_df, heatmap_matrix, metadata
            
        except Exception as e:
            logging.error(f"Error fetching liquidation heatmap: {str(e)}")
            return pd.DataFrame(), np.array([]), {}

    def plot_liquidation_heatmap(self, price_df: pd.DataFrame, heatmap_matrix: np.ndarray, metadata: Dict, save_path: Optional[str] = None):
        """Plot the liquidation heatmap with price data overlay using both static and interactive plots
        
        Args:
            price_df: DataFrame with OHLCV data
            heatmap_matrix: 2D array of liquidation intensities
            metadata: Dictionary with metadata about the data
            save_path: Optional path to save the plot to
        """
        try:
            if price_df.empty or heatmap_matrix.size == 0:
                logging.error("No data to plot")
                return
            
            # Create static matplotlib plot
            self._create_static_plot(price_df, heatmap_matrix, metadata, save_path)
            
            # Create interactive plotly plot
            self._create_interactive_plot(price_df, heatmap_matrix, metadata, save_path)
                
        except Exception as e:
            logging.error(f"Error plotting liquidation heatmap: {str(e)}")

    def _create_static_plot(self, price_df: pd.DataFrame, heatmap_matrix: np.ndarray, metadata: Dict, save_path: Optional[str] = None):
        """Create static matplotlib plot"""
        try:
            # Create figure with multiple subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
            
            # Plot candlestick chart
            ax1.plot(price_df['timestamp'], price_df['close'], color='white', alpha=0.8, label='Price')
            
            # Get price levels from metadata
            price_levels = metadata.get('price_levels', [])
            if not price_levels:
                price_levels = np.linspace(price_df['low'].min(), price_df['high'].max(), heatmap_matrix.shape[0])
            
            # Plot heatmap with improved color scheme
            sns.heatmap(
                heatmap_matrix,
                cmap='YlOrRd',  # Changed to YlOrRd for better visibility of liquidations
                ax=ax1,
                cbar_kws={'label': 'Liquidation Intensity'},
                xticklabels=price_df['timestamp'][::len(price_df)//10],  # Show 10 time labels
                yticklabels=[f"{float(p):.0f}" for p in price_levels[::len(price_levels)//10]],  # Show 10 price labels
                vmin=self.min_liquidation_intensity,  # Set minimum value for color scale
                robust=True  # Use robust quantiles for color scaling
            )
            
            ax1.set_title('BTC Price with Liquidation Heatmap')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price (USDT)')
            ax1.legend()
            
            # Plot volume if available
            if 'volume' in price_df.columns:
                ax2.bar(price_df['timestamp'], price_df['volume'], color='lightblue', alpha=0.7, label='Volume')
                ax2.set_title('Trading Volume')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Volume (USDT)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logging.info(f"Static plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating static plot: {str(e)}")

    def _create_interactive_plot(self, price_df: pd.DataFrame, heatmap_matrix: np.ndarray, metadata: Dict, save_path: Optional[str] = None):
        """Create interactive plotly plot"""
        try:
            # Create daily resampled data
            daily_df = self._resample_to_daily(price_df.copy())
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=('BTC Price with Liquidation Heatmap', 'Trading Volume')
            )
            
            # Get price levels
            price_levels = metadata.get('price_levels', [])
            if not price_levels:
                price_levels = np.linspace(price_df['low'].min(), price_df['high'].max(), heatmap_matrix.shape[0])
            
            # Add heatmap with improved color scheme
            heatmap = go.Heatmap(
                z=heatmap_matrix,
                x=price_df['timestamp'],
                y=[float(p) for p in price_levels],
                colorscale=[
                    [0, 'rgba(0,0,0,0)'],  # Transparent for zero values
                    [0.1, 'rgba(255,255,0,0.3)'],  # Light yellow for low values
                    [0.5, 'rgba(255,165,0,0.6)'],  # Orange for medium values
                    [1, 'rgba(255,0,0,0.9)']  # Red for high values
                ],
                name='Liquidation Intensity',
                showscale=True,
                colorbar=dict(
                    title='Liquidation Intensity',
                    tickformat='.2e'  # Scientific notation for large numbers
                ),
                zmin=self.min_liquidation_intensity,  # Set minimum value for color scale
                zauto=True,  # Automatically adjust color scale
                hoverongaps=False,  # Don't show hover for gaps
                hovertemplate=(
                    "Date: %{x}<br>" +
                    "Price Level: $%{y:,.0f}<br>" +
                    "Liquidation Value: %{z:.2e}<br>" +
                    "<extra></extra>"  # Remove secondary box
                )
            )
            fig.add_trace(heatmap, row=1, col=1)
            
            # Create hourly candlesticks
            hourly_candlesticks = go.Candlestick(
                x=price_df['timestamp'],
                open=price_df['open'],
                high=price_df['high'],
                low=price_df['low'],
                close=price_df['close'],
                name='Hourly Price',
                increasing=dict(line=dict(color='#26A69A')),  # Green
                decreasing=dict(line=dict(color='#EF5350')),  # Red
                showlegend=True,
                opacity=0.7,
                visible=True,  # Show hourly by default
                hoverlabel=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    font_size=12,
                    font_family="monospace"
                ),
                text=[
                    f"Date: {ts}<br>" +
                    f"Open: ${o:,.2f}<br>" +
                    f"High: ${h:,.2f}<br>" +
                    f"Low: ${l:,.2f}<br>" +
                    f"Close: ${c:,.2f}"
                    for ts, o, h, l, c in zip(
                        price_df['timestamp'],
                        price_df['open'],
                        price_df['high'],
                        price_df['low'],
                        price_df['close']
                    )
                ]
            )
            
            # Create daily candlesticks
            daily_candlesticks = go.Candlestick(
                x=daily_df['timestamp'],
                open=daily_df['open'],
                high=daily_df['high'],
                low=daily_df['low'],
                close=daily_df['close'],
                name='Daily Price',
                increasing=dict(line=dict(color='#26A69A')),  # Green
                decreasing=dict(line=dict(color='#EF5350')),  # Red
                showlegend=True,
                opacity=0.7,
                visible=False,  # Hide daily by default
                hoverlabel=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    font_size=12,
                    font_family="monospace"
                ),
                text=[
                    f"Date: {ts}<br>" +
                    f"Open: ${o:,.2f}<br>" +
                    f"High: ${h:,.2f}<br>" +
                    f"Low: ${l:,.2f}<br>" +
                    f"Close: ${c:,.2f}"
                    for ts, o, h, l, c in zip(
                        daily_df['timestamp'],
                        daily_df['open'],
                        daily_df['high'],
                        daily_df['low'],
                        daily_df['close']
                    )
                ]
            )
            
            fig.add_trace(hourly_candlesticks, row=1, col=1)
            fig.add_trace(daily_candlesticks, row=1, col=1)
            
            # Add volume bars for hourly data
            hourly_colors = ['#26A69A' if close >= open else '#EF5350' 
                           for close, open in zip(price_df['close'], price_df['open'])]
            hourly_volume = go.Bar(
                x=price_df['timestamp'],
                y=price_df['volume'],
                name='Hourly Volume',
                marker_color=hourly_colors,
                opacity=0.7,
                visible=True,  # Show hourly by default
                hovertemplate=(
                    "Date: %{x}<br>" +
                    "Volume: %{y:,.0f} USDT<br>" +
                    "<extra></extra>"
                )
            )
            
            # Add volume bars for daily data
            daily_colors = ['#26A69A' if close >= open else '#EF5350' 
                          for close, open in zip(daily_df['close'], daily_df['open'])]
            daily_volume = go.Bar(
                x=daily_df['timestamp'],
                y=daily_df['volume'],
                name='Daily Volume',
                marker_color=daily_colors,
                opacity=0.7,
                visible=False,  # Hide daily by default
                hovertemplate=(
                    "Date: %{x}<br>" +
                    "Volume: %{y:,.0f} USDT<br>" +
                    "<extra></extra>"
                )
            )
            
            fig.add_trace(hourly_volume, row=2, col=1)
            fig.add_trace(daily_volume, row=2, col=1)
            
            # Add buttons for timeframe selection
            updatemenus = [
                dict(
                    type="buttons",
                    direction="right",
                    x=0.1,
                    y=1.15,
                    showactive=True,
                    buttons=[
                        dict(
                            label="Hourly",
                            method="update",
                            args=[
                                {"visible": [True, True, False, True, False]},  # [heatmap, hourly_candles, daily_candles, hourly_volume, daily_volume]
                                {"title": "BTC Price with Liquidation Heatmap (Hourly)"}
                            ]
                        ),
                        dict(
                            label="Daily",
                            method="update",
                            args=[
                                {"visible": [True, False, True, False, True]},  # [heatmap, hourly_candles, daily_candles, hourly_volume, daily_volume]
                                {"title": "BTC Price with Liquidation Heatmap (Daily)"}
                            ]
                        )
                    ]
                )
            ]
            
            # Update layout with improved styling
            fig.update_layout(
                title=dict(
                    text='BTC Liquidation Heatmap Analysis',
                    font=dict(size=24)
                ),
                updatemenus=updatemenus,
                template='plotly_dark',
                showlegend=True,
                height=900,
                xaxis_title='Time',
                yaxis_title='Price (USDT)',
                xaxis2_title='Time',
                yaxis2_title='Volume (USDT)',
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                hoverlabel=dict(
                    bgcolor='rgba(0,0,0,0.8)',  # Semi-transparent black background
                    font_size=12,
                    font_family="monospace"
                )
            )
            
            # Update axes with improved styling
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=2, col=1)
            
            # Update y-axis range to match price range
            fig.update_yaxes(
                range=[price_df['low'].min() * 0.99, price_df['high'].max() * 1.01],
                row=1, col=1
            )
            
            if save_path:
                # Save interactive HTML
                html_path = str(save_path).replace('.png', '.html')
                fig.write_html(html_path)
                logging.info(f"Interactive plot saved to {html_path}")
            
        except Exception as e:
            logging.error(f"Error creating interactive plot: {str(e)}")

def main():
    """Main function to run the Coinglass data collector"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize collector with minimum liquidation intensity filter
        collector = CoinglassCollector(min_liquidation_intensity=0.5e10)  # Filter out values below 10B
        
        # Fetch liquidation heatmap data
        price_df, heatmap_matrix, metadata = collector.fetch_liquidation_heatmap("BTC")
        
        if price_df.empty or heatmap_matrix.size == 0:
            logging.error("No data retrieved")
            return
            
        # Create output directory if it doesn't exist
        output_dir = Path("backtester/data/coinglass")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot and save both static and interactive plots
        save_path = output_dir / "liquidation_heatmap.png"
        collector.plot_liquidation_heatmap(price_df, heatmap_matrix, metadata, str(save_path))
        
        # Save the data
        price_csv_path = output_dir / "price_data.csv"
        heatmap_csv_path = output_dir / "liquidation_heatmap.csv"
        metadata_path = output_dir / "metadata.json"
        
        price_df.to_csv(price_csv_path, index=False)
        np.savetxt(heatmap_csv_path, heatmap_matrix, delimiter=',')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Price data saved to {price_csv_path}")
        logging.info(f"Heatmap data saved to {heatmap_csv_path}")
        logging.info(f"Metadata saved to {metadata_path}")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 