import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from configs.backtester_config import FeatureGroupTypes
from typing import Union, List
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates

class FeatureAnalyzer:
    def __init__(self, group: FeatureGroupTypes, symbol: str):
        self.features_dir = Path(f"backtester/data/features/current/{group.name}")
        self.symbol = symbol
        self.load_data()
    
    def load_data(self):
        """Load feature data and metadata"""
        metadata_file = self.features_dir / f"{self.symbol}_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load feature data
        self.df = pd.read_csv(self.features_dir / f"{self.symbol}_features.csv")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
    
    def plot_feature_group(self, group: FeatureGroupTypes):
        """Plot all features from a specific group"""
        if not isinstance(group, FeatureGroupTypes):
            raise TypeError(f"Expected FeatureGroup, got {type(group)}")
        
        # Separate base columns and feature columns
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in self.df.columns if group.name in col]
        
        print("\nColumns in DataFrame:", self.df.columns.tolist())
        print("\nFeature columns found:", feature_columns)
        
        # Data validation
        print("\nData validation:")
        for feature in feature_columns:
            missing = self.df[feature].isna().sum()
            unique = self.df[feature].nunique()
            print(f"{feature}:")
            print(f"  Missing values: {missing}")
            print(f"  Unique values: {unique}")
            print(f"  First value: {self.df[feature].iloc[0]}")
            print(f"  Last value: {self.df[feature].iloc[-1]}")
        
        # Create subplots with different row heights
        n_features = len(feature_columns)
        fig = make_subplots(
            rows=n_features + 1,
            cols=1,
            subplot_titles=['Price Chart'] + feature_columns,
            vertical_spacing=0.03,
            row_heights=[0.3] + [0.7/n_features] * n_features,  # Larger height for price chart
            shared_xaxes=True  # Synchronize x-axes
        )
        
        # Add price plot in first row
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['open'],
                high=self.df['high'],
                low=self.df['low'],
                close=self.df['close'],
                name='Price',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add feature plots in subsequent rows
        for i, feature in enumerate(feature_columns, 2):  # Start from row 2
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[feature],
                    name=feature,
                    mode='lines',
                    line=dict(shape='hv')  # Step interpolation for categorical data
                ),
                row=i, col=1
            )
            
            # Set y-axis range based on feature type
            if 'is_' in feature:  # Binary features
                fig.update_yaxes(range=[-0.1, 1.1], row=i, col=1)
            else:  # Non-binary features (auto-scale with padding)
                y_min = self.df[feature].min()
                y_max = self.df[feature].max()
                padding = (y_max - y_min) * 0.2  # 20% padding
                fig.update_yaxes(
                    range=[y_min - padding, y_max + padding],
                    row=i, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=300*(n_features + 1),
            title=f"{group.description} Features",
            showlegend=True,
            xaxis_rangeslider_visible=False  # Disable rangeslider for cleaner look
        )
        
        # Update x-axis labels
        for i in range(1, n_features + 1):
            fig.update_xaxes(title_text="", row=i, col=1)  # Remove x-axis labels except last
        fig.update_xaxes(title_text="Date", row=n_features + 1, col=1)  # Add label on last subplot
        
        fig.show()
    
    def analyze_correlations(self, group: FeatureGroupTypes):
        """Analyze feature correlations"""
        if not isinstance(group, FeatureGroupTypes):
            raise TypeError(f"Expected FeatureGroup, got {type(group)}. Use FeatureGroup enum instead of {type(group)}.")

        features = [col for col in self.df.columns if group.name.lower() in col]
        
        # Calculate correlations
        corr = self.df[features].corr()
        
        # Plot correlation matrix
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=features,
            y=features,
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title=f"Feature Correlations - {group.name.lower()}",
            height=800
        )
        fig.show()
    
    def analyze_zscore_distribution(self, group: FeatureGroupTypes):
        """Analyze z-score distributions"""
        zscore_features = [col for col in self.df.columns if 'zscore' in col and group.name.lower() in col]
        
        if not zscore_features:
            print(f"No z-score features found for group {group}")
            return
        
        fig = make_subplots(rows=len(zscore_features), cols=1)
        
        for i, feature in enumerate(zscore_features, 1):
            fig.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    name=feature,
                    nbinsx=50
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            height=300*len(zscore_features),
            title=f"Z-score Distributions - {group}"
        )
        fig.show()
    
    def _plot_price_and_volume(self, features: pd.DataFrame, ax):
        """Plot price candlesticks and volume using vectorized operations"""
        # Ensure data is properly aligned
        df = features.copy()
        
        # Create figure with secondary y-axis
        ax2 = ax.twinx()
        
        # Calculate bar width in datetime units
        time_delta = pd.Timedelta(hours=1)
        bar_width = 0.8  # Width as a fraction of the time delta
        
        # Plot candlesticks
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        
        # Plot price ranges (wicks)
        ax.vlines(df.index, df.low, df.high, color='black', linewidth=0.5)
        
        # Convert timestamps to matplotlib dates for width calculation
        def create_rect(row, color):
            width = bar_width * time_delta / pd.Timedelta(days=1)  # Convert to days for matplotlib
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            return Rectangle((mdates.date2num(row.name), bottom), 
                           width, height,
                           facecolor=color,
                           edgecolor=color)
        
        # Create rectangles for up and down candles
        rect_up = [create_rect(row, 'green') for _, row in up.iterrows()]
        rect_down = [create_rect(row, 'red') for _, row in down.iterrows()]
        
        # Add candlestick bodies to plot
        collection = PatchCollection(rect_up + rect_down, match_original=True)
        ax.add_collection(collection)
        
        # Plot volume on a separate axis with actual values
        if 'quote_volume' in df.columns:
            # Format volume in billions for readability
            volume_in_b = df['quote_volume'] / 1e9
            colors = np.where(df.close >= df.open, 'lightgreen', 'lightcoral')
            
            ax2.bar(df.index, volume_in_b,
                    width=time_delta,
                    color=colors, alpha=0.3)
            ax2.set_ylabel('Volume (Billion USDT)')
        
        # Formatting
        ax.set_title('Price and Volume')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Set proper axis limits
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(df['low'].min() * 0.999, df['high'].max() * 1.001)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Add legend
        ax.legend(['Price Range', 'Price'], loc='upper left')
        ax2.legend(['Volume'], loc='upper right')
    
    def _plot_time_features(self, features: pd.DataFrame, ax, feature_list: List[str]):
        """Plot selected time-based features"""
        # Sort features by time to ensure proper alignment
        df = features.sort_index()
        
        for feature in feature_list:
            if feature in df.columns:
                # Plot with step interpolation for categorical features
                ax.step(df.index, df[feature], label=feature, where='post')
                
        ax.set_title('Time Features')
        ax.set_ylabel('Value')
        ax.grid(True)
        if feature_list:
            ax.legend(loc='upper left')
        
        # Set y-axis limits for binary features
        if all(df[feature].isin([0, 1]) for feature in feature_list if feature in df.columns):
            ax.set_ylim(-0.1, 1.1)

def main():
    ACTIVE_FEATURE_GROUP = FeatureGroupTypes.SEASONALITY
    symbol = 'BTCUSDT'
    analyzer = FeatureAnalyzer(ACTIVE_FEATURE_GROUP, symbol=symbol)
    
    # Example usage:
    analyzer.plot_feature_group(ACTIVE_FEATURE_GROUP)
    analyzer.analyze_correlations(ACTIVE_FEATURE_GROUP)
    
    # Only analyze z-scores if they exist
    if any('zscore' in col for col in analyzer.df.columns):
        analyzer.analyze_zscore_distribution(ACTIVE_FEATURE_GROUP)

if __name__ == "__main__":
    main() 