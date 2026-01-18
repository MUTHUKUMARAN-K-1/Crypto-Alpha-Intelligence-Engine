"""
CoinGecko API client for Crypto Regime Intelligence Engine.
Handles all data fetching with rate limiting and error handling.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import httpx
import pandas as pd
import numpy as np

from app.config import settings
from app.data.data_cache import price_cache, market_cache
from app.utils.logger import app_logger, log_error
from app.utils.helpers import normalize_asset_id


class CoinGeckoAPIError(Exception):
    """Custom exception for CoinGecko API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimitError(CoinGeckoAPIError):
    """Exception for rate limit errors."""
    pass


class CoinGeckoClient:
    """
    Async client for CoinGecko API.
    
    Handles price history, market data, and OHLC fetching
    with built-in rate limiting and caching.
    """
    
    def __init__(self):
        """Initialize the CoinGecko client."""
        self.base_url = settings.coingecko_base_url
        self.api_key = settings.coingecko_api_key
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / settings.max_requests_per_minute
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optional API key."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoRegimeEngine/1.0"
        }
        
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        
        return headers
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < self._min_request_interval:
            wait_time = self._min_request_interval - elapsed
            await asyncio.sleep(wait_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a rate-limited request to CoinGecko API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            CoinGeckoAPIError: On API errors
            RateLimitError: On rate limit exceeded
        """
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        url,
                        params=params,
                        headers=self._get_headers()
                    )
                    
                    if response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = self.retry_delay * (2 ** attempt)
                        app_logger.warning(
                            f"Rate limited, waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        raise CoinGeckoAPIError(
                            f"API error: {response.text}",
                            status_code=response.status_code
                        )
                    
                    return response.json()
                    
                except httpx.RequestError as e:
                    if attempt == self.max_retries - 1:
                        log_error("CoinGecko request", e)
                        raise CoinGeckoAPIError(f"Request failed: {str(e)}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise RateLimitError("Max retries exceeded due to rate limiting")
    
    async def get_price_history(
        self,
        asset_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get historical price data for an asset.
        
        Args:
            asset_id: CoinGecko asset ID
            days: Number of days of history
            vs_currency: Quote currency
            
        Returns:
            DataFrame with timestamp, price, volume, market_cap columns
        """
        # Check cache first
        cache_key = f"price_history_{asset_id}_{days}"
        cached = price_cache.get(cache_key)
        if cached is not None:
            return cached
        
        normalized_id = normalize_asset_id(asset_id)
        
        data = await self._request(
            f"/coins/{normalized_id}/market_chart",
            params={
                "vs_currency": vs_currency,
                "days": days,
                "interval": "daily" if days > 90 else None
            }
        )
        
        # Parse response into DataFrame
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        market_caps = data.get("market_caps", [])
        
        if not prices:
            raise CoinGeckoAPIError(f"No price data for {asset_id}")
        
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Add volume and market cap
        if volumes:
            vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
            vol_df.set_index("timestamp", inplace=True)
            df = df.join(vol_df["volume"], how="left")
        
        if market_caps:
            mcap_df = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
            mcap_df["timestamp"] = pd.to_datetime(mcap_df["timestamp"], unit="ms")
            mcap_df.set_index("timestamp", inplace=True)
            df = df.join(mcap_df["market_cap"], how="left")
        
        # Cache the result
        price_cache.set(cache_key, df)
        
        return df
    
    async def get_ohlc(
        self,
        asset_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Get OHLC (candlestick) data for an asset.
        
        Args:
            asset_id: CoinGecko asset ID
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            vs_currency: Quote currency
            
        Returns:
            DataFrame with open, high, low, close columns
        """
        cache_key = f"ohlc_{asset_id}_{days}"
        cached = price_cache.get(cache_key)
        if cached is not None:
            return cached
        
        normalized_id = normalize_asset_id(asset_id)
        
        data = await self._request(
            f"/coins/{normalized_id}/ohlc",
            params={
                "vs_currency": vs_currency,
                "days": days
            }
        )
        
        if not data:
            raise CoinGeckoAPIError(f"No OHLC data for {asset_id}")
        
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Cache the result
        price_cache.set(cache_key, df)
        
        return df
    
    async def get_market_data(
        self,
        asset_id: str,
        vs_currency: str = "usd"
    ) -> Dict[str, Any]:
        """
        Get current market data for an asset.
        
        Args:
            asset_id: CoinGecko asset ID
            vs_currency: Quote currency
            
        Returns:
            Dictionary with market data
        """
        cache_key = f"market_{asset_id}"
        cached = market_cache.get(cache_key)
        if cached is not None:
            return cached
        
        normalized_id = normalize_asset_id(asset_id)
        
        data = await self._request(
            f"/coins/{normalized_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false"
            }
        )
        
        market_data = data.get("market_data", {})
        
        result = {
            "id": data.get("id"),
            "symbol": data.get("symbol", "").upper(),
            "name": data.get("name"),
            "current_price": market_data.get("current_price", {}).get(vs_currency),
            "market_cap": market_data.get("market_cap", {}).get(vs_currency),
            "total_volume": market_data.get("total_volume", {}).get(vs_currency),
            "high_24h": market_data.get("high_24h", {}).get(vs_currency),
            "low_24h": market_data.get("low_24h", {}).get(vs_currency),
            "price_change_24h": market_data.get("price_change_24h"),
            "price_change_percentage_24h": market_data.get("price_change_percentage_24h"),
            "price_change_percentage_7d": market_data.get("price_change_percentage_7d"),
            "price_change_percentage_30d": market_data.get("price_change_percentage_30d"),
            "ath": market_data.get("ath", {}).get(vs_currency),
            "ath_date": market_data.get("ath_date", {}).get(vs_currency),
            "atl": market_data.get("atl", {}).get(vs_currency),
            "circulating_supply": market_data.get("circulating_supply"),
            "total_supply": market_data.get("total_supply"),
        }
        
        # Cache the result
        market_cache.set(cache_key, result)
        
        return result
    
    async def get_multiple_prices(
        self,
        asset_ids: List[str],
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price history for multiple assets concurrently.
        
        Args:
            asset_ids: List of asset IDs
            days: Number of days of history
            
        Returns:
            Dictionary mapping asset ID to price DataFrame
        """
        tasks = [
            self.get_price_history(asset_id, days)
            for asset_id in asset_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for asset_id, result in zip(asset_ids, results):
            if isinstance(result, Exception):
                app_logger.warning(f"Failed to fetch {asset_id}: {result}")
                continue
            output[normalize_asset_id(asset_id)] = result
        
        return output
    
    async def ping(self) -> bool:
        """
        Check API connectivity.
        
        Returns:
            True if API is reachable
        """
        try:
            await self._request("/ping")
            return True
        except Exception as e:
            app_logger.error(f"Ping failed: {e}")
            return False


# Global client instance
coingecko_client = CoinGeckoClient()
