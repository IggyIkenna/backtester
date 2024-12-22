from typing import Optional

def get_feature_name(symbol: Optional[str], feature_suffix: str) -> str:
    """
    Generate standardized feature names with proper symbol prefixing.
    
    Args:
        symbol: The trading symbol (e.g., 'BTCUSDT'). If None, no prefix is added.
        feature_suffix: The feature-specific part of the name (e.g., '1h_close_ma_20')
    
    Returns:
        str: The complete feature name with proper prefix if symbol is provided
    
    Example:
        >>> get_feature_name('BTCUSDT', '1h_close_ma_20')
        'btcusdt_1h_close_ma_20'
        >>> get_feature_name(None, '1h_close_ma_20')
        '1h_close_ma_20'
    """
    if symbol is None:
        return feature_suffix
    
    # Convert symbol to lowercase and remove any special characters
    clean_symbol = symbol.lower().replace('usdt', '')
    
    return f"{clean_symbol}_{feature_suffix}" 