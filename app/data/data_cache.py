"""
Data caching layer for Crypto Regime Intelligence Engine.
Implements TTL-based in-memory caching to respect API rate limits.
"""

import time
import hashlib
from typing import Any, Optional, Dict, Callable, TypeVar
from functools import wraps
from cachetools import TTLCache
from threading import Lock

from app.config import settings
from app.utils.logger import log_cache_event, app_logger

T = TypeVar("T")


class DataCache:
    """
    Thread-safe TTL cache for API responses.
    
    Attributes:
        cache: Internal cache storage
        ttl: Time-to-live in seconds
        hits: Cache hit counter
        misses: Cache miss counter
    """
    
    def __init__(
        self,
        maxsize: int = 1000,
        ttl: Optional[int] = None
    ):
        """
        Initialize the cache.
        
        Args:
            maxsize: Maximum number of items to cache
            ttl: Time-to-live in seconds (defaults to settings)
        """
        self.ttl = ttl or settings.cache_ttl
        self.cache = TTLCache(maxsize=maxsize, ttl=self.ttl)
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hash-based cache key
        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self.lock:
            value = self.cache.get(key)
            
            if value is not None:
                self.hits += 1
                log_cache_event(key[:16], hit=True)
                return value
            
            self.misses += 1
            log_cache_event(key[:16], hit=False)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            try:
                del self.cache[key]
                return True
            except KeyError:
                return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit rate and counts
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "ttl_seconds": self.ttl
        }
    
    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        
        return wrapper


class AsyncDataCache(DataCache):
    """
    Async-compatible version of DataCache.
    Supports async functions with caching.
    """
    
    def cached_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to cache async function results.
        
        Args:
            func: Async function to cache
            
        Returns:
            Wrapped async function with caching
        """
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = self._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute async function and cache result
            result = await func(*args, **kwargs)
            self.set(key, result)
            return result
        
        return wrapper


# Global cache instances
price_cache = AsyncDataCache(maxsize=500, ttl=settings.cache_ttl)
market_cache = AsyncDataCache(maxsize=200, ttl=settings.cache_ttl)
feature_cache = DataCache(maxsize=1000, ttl=60)  # Shorter TTL for computed features
