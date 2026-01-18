"""
FastAPI routes for the Crypto Regime Intelligence Engine.
Defines all REST API endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException

from app.api.schemas import (
    RegimeResponse,
    TradabilityResponse,
    HealthResponse,
    ErrorResponse,
    MetricsSummary,
    MarketData,
    RiskFactor
)
from app.intelligence.regime_engine import regime_engine
from app.data.coingecko_client import CoinGeckoAPIError
from app.utils.logger import log_api_request, log_api_response, log_error
import time


# Create router with tags for API docs
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and its components.",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of the API, CoinGecko connection, and ML models.
    """
    start_time = time.time()
    log_api_request("/health", {})
    
    try:
        result = await regime_engine.health_check()
        
        duration_ms = (time.time() - start_time) * 1000
        log_api_response("/health", 200, duration_ms)
        
        return HealthResponse(**result)
    
    except Exception as e:
        log_error("health_check", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/regime",
    response_model=RegimeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "External API unavailable"}
    },
    summary="Market Regime Detection",
    description="Detect the current market regime (TREND, RANGE, or HIGH-RISK) based on multiple assets.",
    tags=["Intelligence"]
)
async def detect_regime(
    assets: str = Query(
        ...,
        description="Comma-separated list of asset IDs or tickers (e.g., 'btc,eth,sol')",
        example="btc,eth,sol"
    ),
    days: int = Query(
        30,
        ge=7,
        le=365,
        description="Number of days of history to analyze"
    ),
    include_details: bool = Query(
        False,
        description="Include detailed metrics in response"
    )
) -> RegimeResponse:
    """
    Detect market regime based on multiple cryptocurrency assets.
    
    Analyzes volatility, correlation, and liquidity patterns to classify
    the current market into one of three regimes:
    
    - **TREND**: Directional market movement with momentum
    - **RANGE**: Sideways/consolidating market
    - **HIGH-RISK**: Volatile/uncertain conditions
    
    The response includes a confidence score, detailed metrics,
    and a human-readable explanation.
    """
    start_time = time.time()
    log_api_request("/regime", {"assets": assets, "days": days})
    
    # Parse assets
    asset_list = [a.strip() for a in assets.split(",") if a.strip()]
    
    if not asset_list:
        raise HTTPException(
            status_code=400,
            detail="At least one asset must be provided"
        )
    
    if len(asset_list) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 assets allowed per request"
        )
    
    try:
        result = await regime_engine.analyze_regime(asset_list, days=days)
        
        # Optionally exclude detailed metrics
        if not include_details:
            result.pop("detailed_metrics", None)
        
        duration_ms = (time.time() - start_time) * 1000
        log_api_response("/regime", 200, duration_ms)
        
        # Build response
        return RegimeResponse(
            regime=result["regime"],
            confidence=result["confidence"],
            regime_probabilities=result["regime_probabilities"],
            metrics=MetricsSummary(**result["metrics"]),
            detailed_metrics=result.get("detailed_metrics"),
            explanation=result["explanation"],
            assets_analyzed=result["assets_analyzed"],
            analysis_period_days=result["analysis_period_days"]
        )
    
    except CoinGeckoAPIError as e:
        log_error("detect_regime", e)
        raise HTTPException(
            status_code=503,
            detail=f"CoinGecko API error: {e.message}"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        log_error("detect_regime", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@router.get(
    "/tradability",
    response_model=TradabilityResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Asset not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
        503: {"model": ErrorResponse, "description": "External API unavailable"}
    },
    summary="Asset Tradability Score",
    description="Get a tradability score and risk assessment for a single asset.",
    tags=["Intelligence"]
)
async def get_tradability(
    asset: str = Query(
        ...,
        description="Asset ID or ticker (e.g., 'btc' or 'bitcoin')",
        example="btc"
    ),
    days: int = Query(
        30,
        ge=7,
        le=365,
        description="Number of days of history to analyze"
    )
) -> TradabilityResponse:
    """
    Get tradability score and risk assessment for a cryptocurrency.
    
    Returns:
    - **tradability_score**: 0-100 score indicating how favorable conditions are for trading
    - **risk_level**: LOW, MEDIUM, or HIGH
    - **current_regime**: The detected market regime for this asset
    - **reasoning**: Human-readable explanation of the assessment
    
    Higher tradability scores indicate better trading conditions:
    - 70-100: Favorable conditions
    - 40-69: Acceptable with caution
    - 0-39: Unfavorable conditions
    """
    start_time = time.time()
    log_api_request("/tradability", {"asset": asset, "days": days})
    
    if not asset or not asset.strip():
        raise HTTPException(
            status_code=400,
            detail="Asset parameter is required"
        )
    
    try:
        result = await regime_engine.analyze_tradability(asset.strip(), days=days)
        
        duration_ms = (time.time() - start_time) * 1000
        log_api_response("/tradability", 200, duration_ms)
        
        # Build response
        return TradabilityResponse(
            asset=result["asset"],
            asset_name=result["asset_name"],
            tradability_score=result["tradability_score"],
            risk_level=result["risk_level"],
            current_regime=result["current_regime"],
            reasoning=result["reasoning"],
            market_data=MarketData(**result["market_data"]),
            risk_factors=[RiskFactor(**rf) for rf in result["risk_factors"]],
            analysis_period_days=result["analysis_period_days"]
        )
    
    except CoinGeckoAPIError as e:
        log_error("get_tradability", e)
        if "404" in str(e.status_code) or "not found" in str(e.message).lower():
            raise HTTPException(
                status_code=404,
                detail=f"Asset not found: {asset}"
            )
        raise HTTPException(
            status_code=503,
            detail=f"CoinGecko API error: {e.message}"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        log_error("get_tradability", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@router.get(
    "/",
    summary="API Root",
    description="Welcome endpoint with API information.",
    tags=["System"]
)
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Crypto Regime Intelligence Engine",
        "version": "1.0.0",
        "description": "AI-powered market regime detection and tradability scoring",
        "endpoints": {
            "health": "/health",
            "regime": "/regime?assets=btc,eth,sol",
            "tradability": "/tradability?asset=btc"
        },
        "documentation": "/docs"
    }
