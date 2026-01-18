"""
Pydantic schemas for API request/response validation.
Defines all data models for the REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# Response Models

class MetricsSummary(BaseModel):
    """Summary of analysis metrics."""
    volatility: str = Field(..., description="Volatility summary")
    correlation: str = Field(..., description="Correlation summary")
    liquidity: str = Field(..., description="Liquidity summary")


class RegimeResponse(BaseModel):
    """Response for regime detection endpoint."""
    regime: str = Field(
        ...,
        description="Detected market regime: TREND, RANGE, or HIGH-RISK"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score for the regime classification"
    )
    regime_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution over all regimes"
    )
    metrics: MetricsSummary = Field(
        ...,
        description="Summary of key metrics"
    )
    detailed_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed metrics (optional, based on include_details param)"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the regime classification"
    )
    assets_analyzed: List[str] = Field(
        ...,
        description="List of assets included in the analysis"
    )
    analysis_period_days: int = Field(
        ...,
        description="Number of days analyzed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "regime": "TREND",
                "confidence": 0.82,
                "regime_probabilities": {
                    "TREND": 0.82,
                    "RANGE": 0.12,
                    "HIGH-RISK": 0.06
                },
                "metrics": {
                    "volatility": "Moderate volatility (45.2% annualized, expanding)",
                    "correlation": "Diverging (avg: 0.45)",
                    "liquidity": "Strong liquidity and stability (scores: 72, 68)"
                },
                "explanation": "Market is in a trending phase with bullish momentum. Volatility is expanding, indicating strengthening directional movement.",
                "assets_analyzed": ["bitcoin", "ethereum", "solana"],
                "analysis_period_days": 30
            }
        }


class RiskFactor(BaseModel):
    """Individual risk factor details."""
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    importance: float = Field(..., description="Feature importance in the model")


class MarketData(BaseModel):
    """Current market data for an asset."""
    current_price: Optional[float] = Field(None, description="Current price in USD")
    price_change_24h: Optional[float] = Field(None, description="24h price change %")
    volume_24h: Optional[float] = Field(None, description="24h trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")


class TradabilityResponse(BaseModel):
    """Response for tradability score endpoint."""
    asset: str = Field(..., description="Asset symbol")
    asset_name: str = Field(..., description="Full asset name")
    tradability_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Tradability score from 0-100"
    )
    risk_level: str = Field(
        ...,
        description="Risk level: LOW, MEDIUM, or HIGH"
    )
    current_regime: str = Field(
        ...,
        description="Current market regime for the asset"
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the tradability assessment"
    )
    market_data: MarketData = Field(
        ...,
        description="Current market data"
    )
    risk_factors: List[RiskFactor] = Field(
        default=[],
        description="Top risk factors influencing the score"
    )
    analysis_period_days: int = Field(
        ...,
        description="Number of days analyzed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "asset": "BTC",
                "asset_name": "Bitcoin",
                "tradability_score": 78,
                "risk_level": "LOW",
                "current_regime": "TREND",
                "reasoning": "Conditions are favorable for trading with controlled risk. Trading conditions are particularly favorable right now. Strong liquidity supports efficient trade execution.",
                "market_data": {
                    "current_price": 42150.00,
                    "price_change_24h": 2.35,
                    "volume_24h": 28500000000,
                    "market_cap": 825000000000
                },
                "risk_factors": [
                    {"feature": "liquidity_score", "value": 75.2, "importance": 0.23},
                    {"feature": "stability_index", "value": 68.5, "importance": 0.18}
                ],
                "analysis_period_days": 30
            }
        }


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, str] = Field(
        ...,
        description="Status of individual components"
    )
    cache: Optional[Dict[str, Any]] = Field(
        None,
        description="Cache statistics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "components": {
                    "coingecko_api": "connected",
                    "regime_classifier": "ready",
                    "risk_model": "ready"
                },
                "cache": {
                    "price_cache": {"hits": 45, "misses": 12, "hit_rate": 0.79},
                    "market_cache": {"hits": 20, "misses": 5, "hit_rate": 0.80}
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Asset not found",
                "detail": "Could not fetch data for asset 'xyz'",
                "code": "ASSET_NOT_FOUND"
            }
        }
