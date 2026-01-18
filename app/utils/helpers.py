"""
Helper utilities for Crypto Regime Intelligence Engine.
Contains data validation, normalization, and mathematical helpers.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd


# Asset ID mapping for CoinGecko
ASSET_ID_MAP = {
    # Common tickers to CoinGecko IDs
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ada": "cardano",
    "doge": "dogecoin",
    "dot": "polkadot",
    "matic": "matic-network",
    "link": "chainlink",
    "avax": "avalanche-2",
    "atom": "cosmos",
    "uni": "uniswap",
    "ltc": "litecoin",
    "etc": "ethereum-classic",
    "xlm": "stellar",
    "algo": "algorand",
    "near": "near",
    "ftm": "fantom",
    "aave": "aave",
}


def normalize_asset_id(asset: str) -> str:
    """
    Normalize asset ticker/name to CoinGecko ID.
    
    Args:
        asset: Asset ticker or name (e.g., 'BTC', 'bitcoin')
        
    Returns:
        CoinGecko asset ID
    """
    asset_lower = asset.lower().strip()
    
    # Check if it's a known ticker
    if asset_lower in ASSET_ID_MAP:
        return ASSET_ID_MAP[asset_lower]
    
    # Assume it's already a CoinGecko ID
    return asset_lower


def normalize_asset_list(assets: Union[str, List[str]]) -> List[str]:
    """
    Normalize a comma-separated string or list of assets.
    
    Args:
        assets: Comma-separated string or list of asset identifiers
        
    Returns:
        List of normalized CoinGecko IDs
    """
    if isinstance(assets, str):
        asset_list = [a.strip() for a in assets.split(",") if a.strip()]
    else:
        asset_list = assets
    
    return [normalize_asset_id(a) for a in asset_list]


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    default: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division fails
        
    Returns:
        Division result or default
    """
    # Handle pandas Series
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)
    
    # Handle numpy arrays
    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / denominator, default)
        return result
    
    # Handle scalars
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate percentage returns from price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of percentage returns
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Series of prices
        
    Returns:
        Series of log returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def normalize_to_range(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 0.0,
    target_max: float = 100.0
) -> float:
    """
    Normalize a value to a target range.
    
    Args:
        value: Value to normalize
        min_val: Original minimum
        max_val: Original maximum
        target_min: Target minimum
        target_max: Target maximum
        
    Returns:
        Normalized value
    """
    if max_val == min_val:
        return (target_min + target_max) / 2
    
    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """Clip value to specified range."""
    return max(min_val, min(max_val, value))


def ewma(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponentially Weighted Moving Average.
    
    Args:
        series: Input series
        span: EWMA span
        
    Returns:
        EWMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling z-score.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return safe_divide(series - rolling_mean, rolling_std)


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """
    Validate OHLCV data structure.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ["open", "high", "low", "close"]
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if data.empty:
        raise ValueError("Data is empty")
    
    if data.isnull().all().any():
        raise ValueError("Data contains columns with all null values")
    
    return True


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    return f"{value:,.{decimals}f}"
