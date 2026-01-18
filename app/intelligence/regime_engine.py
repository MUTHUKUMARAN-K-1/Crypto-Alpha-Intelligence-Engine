"""
Regime Engine - Main orchestrator for the intelligence system.
Coordinates data fetching, feature extraction, model inference, and explanation generation.
"""

import asyncio
from typing import Dict, Any, List, Optional

import pandas as pd

from app.data.coingecko_client import coingecko_client, CoinGeckoAPIError
from app.features.volatility import volatility_analyzer
from app.features.correlation import correlation_analyzer
from app.features.liquidity import liquidity_analyzer
from app.models.regime_model import regime_classifier
from app.models.risk_model import risk_model
from app.intelligence.explanation import (
    explanation_generator,
    ExplanationContext
)
from app.utils.helpers import normalize_asset_list, normalize_asset_id
from app.utils.logger import app_logger, log_error


class RegimeEngine:
    """
    Main orchestration engine for market regime intelligence.
    
    Coordinates the entire analysis pipeline:
    1. Data fetching from CoinGecko
    2. Feature extraction (volatility, correlation, liquidity)
    3. ML model inference (regime classification, risk scoring)
    4. Explanation generation
    """
    
    def __init__(self):
        """Initialize the regime engine."""
        self.client = coingecko_client
        self.volatility = volatility_analyzer
        self.correlation = correlation_analyzer
        self.liquidity = liquidity_analyzer
        self.regime_model = regime_classifier
        self.risk_model = risk_model
        self.explainer = explanation_generator
    
    async def analyze_regime(
        self,
        assets: List[str],
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Perform complete market regime analysis.
        
        Args:
            assets: List of asset IDs or tickers
            days: Days of history to analyze
            
        Returns:
            Complete analysis result with regime, confidence, metrics, and explanation
        """
        try:
            # Normalize asset list
            normalized_assets = normalize_asset_list(assets)
            
            if not normalized_assets:
                raise ValueError("No valid assets provided")
            
            app_logger.info(f"Analyzing regime for assets: {normalized_assets}")
            
            # Fetch data for all assets
            price_data = await self.client.get_multiple_prices(
                normalized_assets,
                days=days
            )
            
            if not price_data:
                raise CoinGeckoAPIError("Failed to fetch price data for any asset")
            
            # Fetch market data for primary asset (for liquidity analysis)
            primary_asset = normalized_assets[0]
            try:
                market_data = await self.client.get_market_data(primary_asset)
                market_cap = market_data.get("market_cap", 0)
            except Exception as e:
                app_logger.warning(f"Could not fetch market data: {e}")
                market_data = {}
                market_cap = 0
            
            # Try to get OHLC for better volatility analysis
            try:
                ohlc_data = await self.client.get_ohlc(primary_asset, days=days)
            except Exception:
                ohlc_data = None
            
            # Extract features
            # Volatility analysis (use OHLC if available)
            primary_prices = price_data[primary_asset]["close"]
            if isinstance(ohlc_data, pd.DataFrame) and len(ohlc_data) > 0:
                volatility_metrics = self.volatility.analyze(ohlc=ohlc_data)
            else:
                volatility_metrics = self.volatility.analyze(prices=primary_prices)
            
            # Correlation analysis (needs multiple assets)
            price_series_dict = {
                asset: df["close"] for asset, df in price_data.items()
            }
            correlation_metrics = self.correlation.analyze(
                price_series_dict,
                benchmark_id="bitcoin"
            )
            
            # Liquidity analysis
            if "volume" in price_data[primary_asset].columns:
                volume_series = price_data[primary_asset]["volume"]
            else:
                volume_series = pd.Series()
            
            liquidity_metrics = self.liquidity.analyze(
                prices=primary_prices,
                volume=volume_series,
                market_cap=market_cap
            )
            
            # Regime classification
            regime, confidence, feature_importances = self.regime_model.predict(
                volatility_metrics,
                correlation_metrics,
                liquidity_metrics
            )
            
            # Get regime probabilities
            regime_probs = self.regime_model.get_regime_probabilities(
                volatility_metrics,
                correlation_metrics,
                liquidity_metrics
            )
            
            # Generate explanation
            context = ExplanationContext(
                regime=regime,
                confidence=confidence,
                volatility_metrics=volatility_metrics,
                correlation_metrics=correlation_metrics,
                liquidity_metrics=liquidity_metrics,
                feature_importances=feature_importances
            )
            explanation = self.explainer.generate_regime_explanation(context)
            
            # Generate metrics summary
            metrics_summary = self.explainer.generate_metrics_summary(
                volatility_metrics,
                correlation_metrics,
                liquidity_metrics
            )
            
            return {
                "regime": regime,
                "confidence": round(confidence, 3),
                "regime_probabilities": {
                    k: round(v, 3) for k, v in regime_probs.items()
                },
                "metrics": metrics_summary,
                "detailed_metrics": {
                    "volatility": volatility_metrics,
                    "correlation": correlation_metrics,
                    "liquidity": liquidity_metrics
                },
                "explanation": explanation,
                "assets_analyzed": list(price_data.keys()),
                "analysis_period_days": days
            }
            
        except CoinGeckoAPIError as e:
            log_error("RegimeEngine.analyze_regime", e)
            raise
        except Exception as e:
            log_error("RegimeEngine.analyze_regime", e)
            raise
    
    async def analyze_tradability(
        self,
        asset: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze tradability and risk for a single asset.
        
        Args:
            asset: Asset ID or ticker
            days: Days of history to analyze
            
        Returns:
            Tradability analysis with score, risk level, and reasoning
        """
        try:
            normalized_asset = normalize_asset_id(asset)
            
            app_logger.info(f"Analyzing tradability for: {normalized_asset}")
            
            # Fetch price data
            price_df = await self.client.get_price_history(normalized_asset, days=days)
            
            if price_df.empty:
                raise CoinGeckoAPIError(f"No data available for {asset}")
            
            # Fetch market data
            try:
                market_data = await self.client.get_market_data(normalized_asset)
                market_cap = market_data.get("market_cap", 0)
            except Exception as e:
                app_logger.warning(f"Could not fetch market data: {e}")
                market_data = {}
                market_cap = 0
            
            # Try to get OHLC
            try:
                ohlc_data = await self.client.get_ohlc(normalized_asset, days=days)
            except Exception:
                ohlc_data = None
            
            # Feature extraction
            prices = price_df["close"]
            
            if isinstance(ohlc_data, pd.DataFrame) and len(ohlc_data) > 0:
                volatility_metrics = self.volatility.analyze(ohlc=ohlc_data)
            else:
                volatility_metrics = self.volatility.analyze(prices=prices)
            
            # For single asset, correlation is computed against BTC if not BTC itself
            if normalized_asset != "bitcoin":
                try:
                    btc_df = await self.client.get_price_history("bitcoin", days=days)
                    price_dict = {
                        normalized_asset: prices,
                        "bitcoin": btc_df["close"]
                    }
                    correlation_metrics = self.correlation.analyze(
                        price_dict,
                        benchmark_id="bitcoin"
                    )
                except Exception:
                    correlation_metrics = self.correlation.analyze({normalized_asset: prices})
            else:
                correlation_metrics = self.correlation.analyze({normalized_asset: prices})
            
            # Liquidity analysis
            if "volume" in price_df.columns:
                volume_series = price_df["volume"]
            else:
                volume_series = pd.Series()
            liquidity_metrics = self.liquidity.analyze(
                prices=prices,
                volume=volume_series,
                market_cap=market_cap
            )
            
            # First get regime
            regime, regime_confidence, _ = self.regime_model.predict(
                volatility_metrics,
                correlation_metrics,
                liquidity_metrics
            )
            
            # Then get tradability score
            tradability_score, risk_level, risk_details = self.risk_model.predict(
                volatility_metrics,
                correlation_metrics,
                liquidity_metrics,
                regime
            )
            
            # Generate explanation
            reasoning = self.explainer.generate_tradability_explanation(
                tradability_score=tradability_score,
                risk_level=risk_level,
                regime=regime,
                volatility_metrics=volatility_metrics,
                liquidity_metrics=liquidity_metrics,
                top_factors=risk_details.get("top_factors", [])
            )
            
            return {
                "asset": market_data.get("symbol", normalized_asset.upper()),
                "asset_name": market_data.get("name", normalized_asset),
                "tradability_score": tradability_score,
                "risk_level": risk_level,
                "current_regime": regime,
                "reasoning": reasoning,
                "market_data": {
                    "current_price": market_data.get("current_price"),
                    "price_change_24h": market_data.get("price_change_percentage_24h"),
                    "volume_24h": market_data.get("total_volume"),
                    "market_cap": market_data.get("market_cap")
                },
                "risk_factors": risk_details.get("top_factors", []),
                "analysis_period_days": days
            }
            
        except CoinGeckoAPIError as e:
            log_error("RegimeEngine.analyze_tradability", e)
            raise
        except Exception as e:
            log_error("RegimeEngine.analyze_tradability", e)
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the engine and its dependencies.
        
        Returns:
            Health status dictionary
        """
        status = {
            "status": "healthy",
            "components": {}
        }
        
        # Check CoinGecko API
        try:
            api_ok = await self.client.ping()
            status["components"]["coingecko_api"] = "connected" if api_ok else "disconnected"
        except Exception as e:
            status["components"]["coingecko_api"] = f"error: {str(e)}"
            status["status"] = "degraded"
        
        # Check ML models
        status["components"]["regime_classifier"] = (
            "ready" if self.regime_model.is_trained else "not_trained"
        )
        status["components"]["risk_model"] = (
            "ready" if self.risk_model.is_trained else "not_trained"
        )
        
        # Check cache stats
        from app.data.data_cache import price_cache, market_cache
        status["cache"] = {
            "price_cache": price_cache.get_stats(),
            "market_cache": market_cache.get_stats()
        }
        
        return status


# Global engine instance
regime_engine = RegimeEngine()
