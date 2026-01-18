"""
Logging utilities for Crypto Regime Intelligence Engine.
Provides structured logging with configurable levels.
"""

import logging
import sys
from typing import Optional

from app.config import settings


def setup_logger(
    name: str,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a structured logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level from settings or override
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Structured format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Default application logger
app_logger = setup_logger("crypto_regime")


def log_api_request(
    endpoint: str,
    params: dict,
    status: str = "started"
) -> None:
    """Log API request details."""
    app_logger.info(
        f"API Request | endpoint={endpoint} | params={params} | status={status}"
    )


def log_api_response(
    endpoint: str,
    status_code: int,
    duration_ms: float
) -> None:
    """Log API response details."""
    app_logger.info(
        f"API Response | endpoint={endpoint} | status={status_code} | duration={duration_ms:.2f}ms"
    )


def log_model_inference(
    model_name: str,
    input_shape: tuple,
    output: str,
    confidence: float
) -> None:
    """Log ML model inference details."""
    app_logger.info(
        f"Model Inference | model={model_name} | input_shape={input_shape} | "
        f"output={output} | confidence={confidence:.3f}"
    )


def log_cache_event(
    cache_key: str,
    hit: bool
) -> None:
    """Log cache hit/miss events."""
    event = "HIT" if hit else "MISS"
    app_logger.debug(f"Cache {event} | key={cache_key}")


def log_error(
    context: str,
    error: Exception,
    extra: Optional[dict] = None
) -> None:
    """Log error with context."""
    extra_str = f" | extra={extra}" if extra else ""
    app_logger.error(
        f"Error | context={context} | error={type(error).__name__}: {str(error)}{extra_str}"
    )
