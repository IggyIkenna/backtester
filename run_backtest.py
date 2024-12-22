"""Main backtest runner script"""
import logging
from pathlib import Path
import pandas as pd
from typing import Dict

from backtesting import Backtester
from strategies.macd_strategy import MACDStrategy
from execution_engine import ExecutionEngine
from position_manager import PositionManager
from risk_manager import RiskManager
from output_handler import OutputHandler
from configs.backtester_config import BacktestConfig
from utils.types import MarketState, BacktestState


def setup_logging(config: BacktestConfig) -> None:
    """Setup centralized logging configuration"""
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Basic config
    logging.basicConfig(
        level=config.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_dir / 'backtest.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set levels for specific components if needed
    logging.getLogger('strategies.base_strategy').setLevel(config.log_level)
    logging.getLogger('strategies.macd_strategy').setLevel(config.log_level)
    logging.getLogger('execution_engine').setLevel(config.log_level)
    logging.getLogger('position_manager').setLevel(config.log_level)
    logging.getLogger('risk_manager').setLevel(config.log_level)
    logging.getLogger('output_handler').setLevel(config.log_level)
    logging.getLogger('backtesting').setLevel(config.log_level)


def load_market_data(config: BacktestConfig) -> pd.DataFrame:
    """Load and prepare market data"""
    # Load OHLCV data
    data_path = config.data_dir / f"{config.symbol}_{config.time.candle_period}.csv"
    df = pd.read_csv(data_path)
    
    # Convert timestamp and set index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Filter to backtest period
    mask = (df.index >= config.time.start_date) & (df.index <= config.time.end_date)
    df = df[mask].copy()
    
    # Load features if they exist
    features_path = config.features_dir / f"{config.symbol}_features.csv"
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        features_df.set_index('timestamp', inplace=True)
        df = df.join(features_df)
    
    return df


def main():
    # Load config
    config = BacktestConfig.from_yaml(Path('backtester/configs/backtest_config.yaml'))
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting backtest")
    
    try:
        # Load data
        market_df = load_market_data(config)
        logger.info(f"Loaded {len(market_df)} candles")
        
        # Initialize components
        strategy = MACDStrategy(config)
        execution_engine = ExecutionEngine(config)
        position_manager = PositionManager(config)
        risk_manager = RiskManager(config)
        output_handler = OutputHandler(config)
        
        # Create and run backtester
        backtester = Backtester(
            strategy=strategy,
            execution_engine=execution_engine,
            position_manager=position_manager,
            risk_manager=risk_manager,
            output_handler=output_handler,
            config=config
        )
        
        # Run backtest
        results: BacktestState = run(market_df)
        
        # Generate output
        output_handler.save_results()
        output_handler.create_visualizations()
        
        logger.info(f"Results saved to {config.results_dir}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        raise
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main()
