"""
Tests for the Crypto Regime Intelligence Engine.
Contains unit tests for features, models, and integration tests for the API.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestVolatilityAnalyzer:
    """Tests for volatility feature extraction."""
    
    def test_calculate_returns(self):
        """Test return calculation."""
        from app.utils.helpers import calculate_returns
        
        prices = pd.Series([100, 105, 103, 108, 110])
        returns = calculate_returns(prices)
        
        assert len(returns) == 4
        assert returns.iloc[0] == pytest.approx(0.05, rel=0.01)
    
    def test_volatility_analysis(self):
        """Test complete volatility analysis."""
        from app.features.volatility import volatility_analyzer
        
        # Create sample price data
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 2))
        
        result = volatility_analyzer.analyze(prices=prices)
        
        assert "current_volatility" in result
        assert "volatility_regime" in result
        assert "spikes" in result
        assert result["volatility_regime"] in ["EXPANDING", "CONTRACTING", "STABLE"]
    
    def test_ohlc_volatility(self):
        """Test volatility analysis with OHLC data."""
        from app.features.volatility import volatility_analyzer
        
        # Create sample OHLC data
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        
        ohlc = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n)) * 1.5,
            "low": close - abs(np.random.randn(n)) * 1.5,
            "close": close
        })
        
        result = volatility_analyzer.analyze(ohlc=ohlc)
        
        assert "atr" in result
        assert "atr_percentage" in result
        assert result["atr"] is not None


class TestCorrelationAnalyzer:
    """Tests for correlation feature extraction."""
    
    def test_pairwise_correlation(self):
        """Test pairwise correlation calculation."""
        from app.features.correlation import correlation_analyzer
        
        np.random.seed(42)
        btc = pd.Series(100 + np.cumsum(np.random.randn(100) * 2))
        eth = pd.Series(50 + np.cumsum(np.random.randn(100) * 1.5))
        
        corr_matrix = correlation_analyzer.calculate_pairwise_correlations({
            "bitcoin": btc,
            "ethereum": eth
        })
        
        assert not corr_matrix.empty
        assert "bitcoin" in corr_matrix.columns
        assert "ethereum" in corr_matrix.columns
    
    def test_correlation_regime_detection(self):
        """Test correlation regime detection."""
        from app.features.correlation import correlation_analyzer
        
        np.random.seed(42)
        base = np.cumsum(np.random.randn(100) * 2)
        
        # High correlation scenario
        btc = pd.Series(100 + base)
        eth = pd.Series(50 + base * 0.8 + np.random.randn(100) * 0.5)
        
        result = correlation_analyzer.analyze({
            "bitcoin": btc,
            "ethereum": eth
        })
        
        assert "average_correlation" in result
        assert "correlation_regime" in result
        assert result["correlation_regime"] in ["HIGH_CORRELATION", "LOW_CORRELATION", "DIVERGING", "UNDEFINED"]


class TestLiquidityAnalyzer:
    """Tests for liquidity feature extraction."""
    
    def test_volume_profile(self):
        """Test volume profile calculation."""
        from app.features.liquidity import liquidity_analyzer
        
        np.random.seed(42)
        volume = pd.Series(np.random.uniform(1e6, 5e6, 50))
        
        result = liquidity_analyzer.calculate_volume_profile(volume)
        
        assert "avg_volume" in result
        assert "volume_trend" in result
        assert result["volume_trend"] in ["INCREASING", "DECREASING", "STABLE"]
    
    def test_liquidity_analysis(self):
        """Test complete liquidity analysis."""
        from app.features.liquidity import liquidity_analyzer
        
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 2))
        volume = pd.Series(np.random.uniform(1e6, 5e6, 50))
        
        result = liquidity_analyzer.analyze(
            prices=prices,
            volume=volume,
            market_cap=1e9
        )
        
        assert "liquidity_score" in result
        assert "stability_index" in result
        assert "liquidity_regime" in result
        assert 0 <= result["liquidity_score"] <= 100


class TestRegimeClassifier:
    """Tests for the regime classification model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        from app.models.regime_model import regime_classifier
        
        assert regime_classifier.is_trained
        assert regime_classifier.model is not None
        assert regime_classifier.scaler is not None
    
    def test_regime_prediction(self):
        """Test regime prediction."""
        from app.models.regime_model import regime_classifier
        
        # Create mock metrics
        volatility_metrics = {
            "current_volatility": 0.6,
            "volatility_regime": "EXPANDING",
            "atr_percentage": 3.5,
            "spikes": {"spike_rate": 0.05},
            "returns_mean_pct": 1.2,
            "returns_std_pct": 2.5
        }
        
        correlation_metrics = {
            "average_correlation": 0.5,
            "correlation_regime": "DIVERGING"
        }
        
        liquidity_metrics = {
            "liquidity_score": 70,
            "stability_index": 65,
            "market_depth_proxy": 60,
            "volume_profile": {"volume_trend": "INCREASING"}
        }
        
        regime, confidence, importances = regime_classifier.predict(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics
        )
        
        assert regime in ["TREND", "RANGE", "HIGH-RISK"]
        assert 0 <= confidence <= 1
        assert len(importances) > 0


class TestRiskModel:
    """Tests for the risk/tradability model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        from app.models.risk_model import risk_model
        
        assert risk_model.is_trained
        assert risk_model.model is not None
    
    def test_tradability_prediction(self):
        """Test tradability prediction."""
        from app.models.risk_model import risk_model
        
        volatility_metrics = {
            "current_volatility": 0.4,
            "volatility_regime": "STABLE",
            "atr_percentage": 2.0,
            "spikes": {"spike_rate": 0.02},
            "max_drawdown_pct": -8
        }
        
        correlation_metrics = {
            "average_correlation": 0.6
        }
        
        liquidity_metrics = {
            "liquidity_score": 75,
            "stability_index": 70,
            "market_depth_proxy": 65,
            "volume_profile": {"volume_trend": "STABLE"}
        }
        
        score, risk_level, details = risk_model.predict(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics,
            regime="RANGE"
        )
        
        assert 0 <= score <= 100
        assert risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert "top_factors" in details


class TestExplanationGenerator:
    """Tests for explanation generation."""
    
    def test_regime_explanation(self):
        """Test regime explanation generation."""
        from app.intelligence.explanation import (
            explanation_generator,
            ExplanationContext
        )
        
        context = ExplanationContext(
            regime="TREND",
            confidence=0.85,
            volatility_metrics={
                "current_volatility": 0.6,
                "volatility_regime": "EXPANDING",
                "spikes": {"recent_spikes": 0},
                "returns_mean_pct": 1.5
            },
            correlation_metrics={
                "average_correlation": 0.5,
                "correlation_regime": "DIVERGING"
            },
            liquidity_metrics={
                "liquidity_score": 70,
                "volume_profile": {"volume_trend": "INCREASING"}
            },
            feature_importances={}
        )
        
        explanation = explanation_generator.generate_regime_explanation(context)
        
        assert len(explanation) > 0
        assert "trend" in explanation.lower() or "momentum" in explanation.lower()


class TestHelpers:
    """Tests for helper utilities."""
    
    def test_asset_normalization(self):
        """Test asset ID normalization."""
        from app.utils.helpers import normalize_asset_id, normalize_asset_list
        
        assert normalize_asset_id("BTC") == "bitcoin"
        assert normalize_asset_id("eth") == "ethereum"
        assert normalize_asset_id("bitcoin") == "bitcoin"
        
        assets = normalize_asset_list("btc,eth,sol")
        assert assets == ["bitcoin", "ethereum", "solana"]
    
    def test_safe_divide(self):
        """Test safe division."""
        from app.utils.helpers import safe_divide
        
        assert safe_divide(10, 2) == 5
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, default=-1) == -1


# Integration tests (require network, skip in CI)
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access to CoinGecko")
    async def test_regime_analysis_pipeline(self):
        """Test full regime analysis pipeline."""
        from app.intelligence.regime_engine import regime_engine
        
        result = await regime_engine.analyze_regime(
            assets=["bitcoin", "ethereum"],
            days=14
        )
        
        assert "regime" in result
        assert "confidence" in result
        assert "explanation" in result
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access to CoinGecko")
    async def test_tradability_pipeline(self):
        """Test full tradability analysis pipeline."""
        from app.intelligence.regime_engine import regime_engine
        
        result = await regime_engine.analyze_tradability(
            asset="bitcoin",
            days=14
        )
        
        assert "tradability_score" in result
        assert "risk_level" in result
        assert "reasoning" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
