import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import glob

class DataCatalog:
    def __init__(self, catalog_path: str = "backtester/data/catalog.json"):
        """Initialize the data catalog"""
        self.catalog_path = Path(catalog_path)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog = self._load_catalog()
        
        # Define collection priorities and constraints
        self.collection_priorities = {
            "coinglass_liquidation_heatmap": {
                "priority": 1,  # Fast API call
                "granularity": "dynamic",  # Depends on range parameter
                "ranges": ["12h", "24h", "3d", "7d", "30d", "90d", "180d", "1y"],
                "timing": "t-1",  # Backward looking
                "symbols": ["BTCUSDT"],
                "lookback_days": 1825  # 5 years
            },
            "coinglass_fear_greed": {
                "priority": 1,
                "granularity": "1d",
                "timing": "t-1",
                "lookback_days": 1825  # 5 years
            },
            "vix": {
                "priority": 2,
                "granularity": "1h",
                "timing": "t+1",  # Forward looking (raw market data)
                "lookback_days": 720
            },
            "nasdaq": {
                "priority": 2,
                "granularity": "1h",
                "timing": "t+1",
                "lookback_days": 720
            },
            "options": {
                "priority": 3,
                "granularity": "15m",
                "timing": "t-1",
                "start_date": "2023-05-23",
                "end_date": "2024-10-25"
            },
            "liquidations": {
                "priority": 3,
                "granularity": "5m",
                "timing": "t-1",
                "start_date": "2023-05-23",
                "end_date": "2024-10-25"
            },
            "derivative_tickers": {
                "priority": 3,
                "granularity": "5m",
                "timing": "t-1",
                "start_date": "2023-05-23",
                "end_date": "2024-10-25"
            },
            "binance_futures": {
                "priority": 4,  # Most time-consuming
                "granularities": ["1h", "4h"],
                "timing": "t+1",
                "lookback_days": 1825  # 5 years
            }
        }
        
    def _load_catalog(self) -> dict:
        """Load existing catalog or create new one if it doesn't exist"""
        if self.catalog_path.exists():
            with open(self.catalog_path, 'r') as f:
                return json.load(f)
        return {
            "last_updated": datetime.now().isoformat(),
            "data_sources": {}
        }
        
    def _save_catalog(self):
        """Save catalog to JSON file"""
        self.catalog["last_updated"] = datetime.now().isoformat()
        with open(self.catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=4)
            
    def update_source(
        self,
        source_id: str,
        nickname: str,
        directory: str,
        granularities: List[str],
        date_range: Dict[str, str],
        headers: List[str],
        fetcher_script: str,
        data_type: str = "raw",  # raw or feature
        exchange: Optional[str] = None,
        timing: str = "t+1",  # t+1 or t-1
        period_length: Optional[str] = None,  # e.g., "1d", "1h"
        last_retrieval: Optional[str] = None  # Timestamp of last data point
    ):
        """Update or add a data source in the catalog"""
        self.catalog["data_sources"][source_id] = {
            "nickname": nickname,
            "directory": directory,
            "granularities": granularities,
            "date_range": date_range,
            "headers": headers,
            "last_updated": datetime.now().isoformat(),
            "fetcher_script": fetcher_script,
            "data_type": data_type,
            "exchange": exchange,
            "timing": timing,
            "period_length": period_length,
            "last_retrieval": last_retrieval
        }
        self._save_catalog()
        
    def scan_directory(self):
        """Scan data directories and update catalog with found data"""
        base_dir = Path("backtester/data")
        
        # Scan raw data
        raw_dir = base_dir / "raw"
        if raw_dir.exists():
            for market_dir in raw_dir.glob("market_data_*"):
                self._process_raw_data_directory(market_dir)
                
        # Scan feature data
        features_dir = base_dir / "features"
        if features_dir.exists():
            self._process_features_directory(features_dir)
                
    def _process_raw_data_directory(self, market_dir: Path):
        """Process a raw data directory and update catalog"""
        # Extract interval and lookback from directory name
        dir_parts = market_dir.name.split('_')
        if len(dir_parts) >= 3:
            interval = dir_parts[1]
            lookback = dir_parts[2].replace('d', '')
            
            # Process each exchange directory
            for exchange_dir in market_dir.glob("*"):
                if exchange_dir.is_dir():
                    exchange = exchange_dir.name
                    
                    # Process each symbol file
                    for symbol_file in exchange_dir.glob("*.csv"):
                        try:
                            df = pd.read_csv(symbol_file, nrows=1)
                            headers = list(df.columns)
                            
                            # Get date range from actual data
                            df_full = pd.read_csv(symbol_file)
                            date_range = {
                                "start": df_full.iloc[0]['timestamp'] if 'timestamp' in df_full else str(df_full.index[0]),
                                "end": df_full.iloc[-1]['timestamp'] if 'timestamp' in df_full else str(df_full.index[-1])
                            }
                            
                            source_id = f"{exchange}_{symbol_file.stem}_{interval}"
                            self.update_source(
                                source_id=source_id,
                                nickname=symbol_file.stem,
                                directory=str(market_dir),
                                granularities=[interval],
                                date_range=date_range,
                                headers=headers,
                                fetcher_script="data_collector.py",
                                data_type="raw",
                                exchange=exchange
                            )
                        except Exception as e:
                            logging.error(f"Error processing {symbol_file}: {str(e)}")
                            
    def _process_features_directory(self, features_dir: Path):
        """Process features directory and update catalog"""
        feature_types = {
            "options": {"script": "fetch_all_vols.py", "interval": "15m"},
            "liquidations": {"script": "fetch_liquidations.py", "interval": "5m"},
            "derivative_tickers": {"script": "fetch_derivative_tickers.py", "interval": "5m"},
            "fear_and_greed": {"script": "featch_fear_and_greed_index.py", "interval": "1d"}
        }
        
        # First check for fear and greed data which has a different structure
        fear_greed_dir = features_dir / "fear_and_greed"
        if fear_greed_dir.exists():
            for csv_file in fear_greed_dir.glob("fear_and_greed_index_*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    headers = list(df.columns)
                    
                    # Get date range from actual data
                    date_range = {
                        "start": df.iloc[0]['date'] if 'date' in df else str(df.index[0]),
                        "end": df.iloc[-1]['date'] if 'date' in df else str(df.index[-1])
                    }
                    
                    source_id = "fear_and_greed_index"
                    self.update_source(
                        source_id=source_id,
                        nickname="fear_and_greed_index",
                        directory=str(fear_greed_dir),
                        granularities=["1d"],
                        date_range=date_range,
                        headers=headers,
                        fetcher_script="featch_fear_and_greed_index.py",
                        data_type="feature"
                    )
                except Exception as e:
                    logging.error(f"Error processing fear and greed data: {str(e)}")
        
        # Process other feature directories
        for feature_dir in features_dir.glob("*"):
            if not feature_dir.is_dir() or feature_dir.name == "fear_and_greed":
                continue
                
            # Identify feature type
            feature_type = None
            for ft in feature_types:
                if ft in feature_dir.name:
                    feature_type = ft
                    break
                    
            if not feature_type:
                continue
                
            # Process each exchange directory if it exists
            for exchange_dir in feature_dir.glob("*"):
                if not exchange_dir.is_dir():
                    continue
                    
                # Process each symbol directory
                for symbol_dir in exchange_dir.glob("*"):
                    if not symbol_dir.is_dir():
                        continue
                        
                    # Process CSV files
                    for csv_file in symbol_dir.glob("*.csv"):
                        try:
                            df = pd.read_csv(csv_file, nrows=1)
                            headers = list(df.columns)
                            
                            # Get date range from actual data
                            df_full = pd.read_csv(csv_file)
                            date_range = {
                                "start": df_full.iloc[0]['timestamp'] if 'timestamp' in df_full else str(df_full.index[0]),
                                "end": df_full.iloc[-1]['timestamp'] if 'timestamp' in df_full else str(df_full.index[-1])
                            }
                            
                            source_id = f"{feature_type}_{exchange_dir.name}_{symbol_dir.name}"
                            self.update_source(
                                source_id=source_id,
                                nickname=f"{feature_type}_{symbol_dir.name}",
                                directory=str(feature_dir),
                                granularities=[feature_types[feature_type]["interval"]],
                                date_range=date_range,
                                headers=headers,
                                fetcher_script=feature_types[feature_type]["script"],
                                data_type="feature",
                                exchange=exchange_dir.name
                            )
                        except Exception as e:
                            logging.error(f"Error processing {csv_file}: {str(e)}")
                            
    def get_latest_data_info(self, source_id: str) -> Optional[dict]:
        """Get information about the latest data for a source"""
        return self.catalog["data_sources"].get(source_id)
        
    def get_all_sources(self) -> dict:
        """Get all data sources"""
        return self.catalog["data_sources"]
        
    def print_summary(self):
        """Print a summary of all data sources"""
        print("\nData Catalog Summary")
        print("=" * 80)
        
        for source_id, info in self.catalog["data_sources"].items():
            print(f"\nSource: {source_id}")
            print("-" * 40)
            print(f"Nickname: {info['nickname']}")
            print(f"Type: {info['data_type']}")
            if info['exchange']:
                print(f"Exchange: {info['exchange']}")
            print(f"Granularities: {', '.join(info['granularities'])}")
            print(f"Date Range: {info['date_range']['start']} to {info['date_range']['end']}")
            print(f"Last Updated: {info['last_updated']}")
            print(f"Fetcher: {info['fetcher_script']}")
            print(f"Headers: {', '.join(info['headers'][:5])}...")

    def get_universal_overlap(self) -> Tuple[str, str, List[str]]:
        """Find the universal overlapping date range across all data sources
        
        Returns:
            Tuple containing:
            - Start date (YYYY-MM-DD)
            - End date (YYYY-MM-DD)
            - List of missing sources
        """
        if not self.catalog["data_sources"]:
            return None, None, []

        start_dates = []
        end_dates = []
        missing_sources = []
        
        # Check each required source
        for source_type, config in self.collection_priorities.items():
            sources = [s for s in self.catalog["data_sources"].values() 
                      if source_type in s["nickname"].lower()]
            
            if not sources:
                missing_sources.append(source_type)
                continue
            
            for source in sources:
                try:
                    start = pd.to_datetime(source["date_range"]["start"])
                    end = pd.to_datetime(source["date_range"]["end"])
                    start_dates.append(start)
                    end_dates.append(end)
                except (KeyError, ValueError) as e:
                    logging.warning(f"Invalid date range for {source['nickname']}: {e}")
                    missing_sources.append(source["nickname"])

        if not start_dates or not end_dates:
            return None, None, missing_sources

        # Find the overlap
        latest_start = max(start_dates)
        earliest_end = min(end_dates)
        
        # Format dates
        overlap_start = latest_start.strftime('%Y-%m-%d')
        overlap_end = earliest_end.strftime('%Y-%m-%d')
        
        return overlap_start, overlap_end, missing_sources

    def get_collection_status(self) -> Dict:
        """Get collection status for all data sources
        
        Returns:
            Dictionary with collection status for each source type
        """
        status = {}
        
        for source_type, config in self.collection_priorities.items():
            sources = [s for s in self.catalog["data_sources"].values() 
                      if source_type in s["nickname"].lower()]
            
            status[source_type] = {
                "priority": config["priority"],
                "granularity": config.get("granularity") or config.get("granularities"),
                "required_range": {
                    "start": config.get("start_date") or (datetime.now() - timedelta(days=config["lookback_days"])).strftime('%Y-%m-%d'),
                    "end": config.get("end_date") or datetime.now().strftime('%Y-%m-%d')
                },
                "available_sources": len(sources),
                "missing": len(sources) == 0,
                "current_range": {
                    "start": min([pd.to_datetime(s["date_range"]["start"]) for s in sources]).strftime('%Y-%m-%d') if sources else None,
                    "end": max([pd.to_datetime(s["date_range"]["end"]) for s in sources]).strftime('%Y-%m-%d') if sources else None
                } if sources else {"start": None, "end": None}
            }
            
        return status

    def print_collection_status(self):
        """Print collection status summary with timing information"""
        status = self.get_collection_status()
        overlap_start, overlap_end, missing = self.get_universal_overlap()
        
        print("\nData Collection Status")
        print("=" * 80)
        
        # Print source-specific status
        for source_type, info in sorted(status.items(), key=lambda x: x[1]["priority"]):
            print(f"\n{source_type.upper()}")
            print("-" * 40)
            print(f"Priority: {info['priority']}")
            print(f"Granularity: {info['granularity']}")
            print(f"Timing: {info.get('timing', 'Not specified')}")
            if source_type == "coinglass_liquidation_heatmap":
                print(f"Available ranges: {', '.join(info['ranges'])}")
            print(f"Required Range: {info['required_range']['start']} to {info['required_range']['end']}")
            print(f"Current Range: {info['current_range']['start']} to {info['current_range']['end']}")
            if 'period_length' in info:
                print(f"Period Length: {info['period_length']}")
            if 'last_retrieval' in info:
                print(f"Last Retrieval: {info['last_retrieval']}")
            print(f"Status: {'🟢 Available' if not info['missing'] else '🔴 Missing'}")
        
        # Print overlap information
        print("\nUniversal Data Overlap")
        print("-" * 40)
        if overlap_start and overlap_end:
            print(f"Start: {overlap_start}")
            print(f"End: {overlap_end}")
        else:
            print("No universal overlap found")
        
        if missing:
            print("\nMissing Sources:")
            for source in missing:
                print(f"- {source}")

    def get_binance_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get Binance data from catalog, returns None if not found"""
        source_id = f"binance-futures_{symbol}_{interval}"
        source_info = self.get_latest_data_info(source_id)
        
        if not source_info:
            return None
            
        try:
            data_path = Path(source_info["directory"]) / "binance-futures" / f"{symbol}.csv"
            if not data_path.exists():
                return None
                
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            return df
            
        except Exception as e:
            logging.error(f"Error reading Binance data: {str(e)}")
            return None


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize catalog
    catalog = DataCatalog()
    
    # Scan directories and update catalog
    logging.info("Scanning data directories...")
    catalog.scan_directory()
    
    # Print collection status
    catalog.print_collection_status()
    
    logging.info(f"Catalog saved to {catalog.catalog_path}")


if __name__ == "__main__":
    main() 