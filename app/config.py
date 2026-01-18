"""
Configuration module for Crypto Regime Intelligence Engine.
Loads environment variables and provides application settings.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # CoinGecko API Configuration
    coingecko_api_key: Optional[str] = None
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    
    # OpenRouter LLM Configuration
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "deepseek/deepseek-r1-distill-llama-70b:free"
    
    # WEEX API Configuration (for AI Wars Hackathon)
    weex_api_key: Optional[str] = None
    weex_secret_key: Optional[str] = None
    weex_passphrase: Optional[str] = None
    weex_base_url: str = "https://api-contract.weex.com"
    
    # Trading Configuration
    max_leverage: int = 20  # Hackathon cap
    max_position_pct: float = 0.1  # Max 10% of portfolio per trade
    paper_trading: bool = True  # Start in paper trading mode
    
    # Cache Configuration
    cache_ttl: int = 300  # 5 minutes default
    
    # Logging Configuration
    log_level: str = "INFO"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Rate Limiting
    max_requests_per_minute: int = 10  # Conservative for free tier
    
    # Model Configuration
    model_confidence_threshold: float = 0.6
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience accessor
settings = get_settings()
