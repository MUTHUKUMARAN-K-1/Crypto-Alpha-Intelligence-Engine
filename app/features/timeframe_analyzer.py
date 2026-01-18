"""
Multi-Timeframe Regime Analysis Module.
Detects regimes at multiple time scales and computes confluence scores.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from app.features.volatility import VolatilityAnalyzer
from app.features.correlation import CorrelationAnalyzer
from app.features.liquidity import LiquidityAnalyzer
from app.models.regime_model import regime_classifier
from app.utils.logger import app_logger


class TimeframeAnalyzer:
    """
    Performs regime detection across multiple timeframes.
    
    Timeframes analyzed:
    - Short-term: 7 days (1W)
    - Medium-term: 14 days (2W)  
    - Standard: 30 days (1M)
    - Long-term: 90 days (3M)
    
    Creates confluence scores when multiple timeframes agree.
    """
    
    # Define timeframe configurations
    TIMEFRAMES = {
        "1W": {"days": 7, "label": "Short-term (1 Week)", "weight": 0.15},
        "2W": {"days": 14, "label": "Medium-term (2 Weeks)", "weight": 0.25},
        "1M": {"days": 30, "label": "Standard (1 Month)", "weight": 0.35},
        "3M": {"days": 90, "label": "Long-term (3 Months)", "weight": 0.25},
    }
    
    def __init__(self):
        """Initialize timeframe analyzer with feature extractors."""
        self.volatility_analyzer = VolatilityAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
    
    def analyze_single_timeframe(
        self,
        prices: pd.Series,
        volume: pd.Series = None,
        market_cap: float = 0,
        correlation_data: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze regime for a single timeframe.
        
        Args:
            prices: Price series for the timeframe
            volume: Optional volume series
            market_cap: Optional market cap
            correlation_data: Optional dict of price series for correlation
            
        Returns:
            Regime analysis for this timeframe
        """
        # Ensure we have enough data
        if len(prices) < 5:
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {"TREND": 0.33, "RANGE": 0.33, "HIGH-RISK": 0.33}
            }
        
        # Extract features
        volatility_metrics = self.volatility_analyzer.analyze(prices=prices)
        
        # Handle volume
        if isinstance(volume, pd.Series) and len(volume) > 0:
            liquidity_metrics = self.liquidity_analyzer.analyze(
                prices=prices,
                volume=volume,
                market_cap=market_cap
            )
        else:
            liquidity_metrics = self.liquidity_analyzer.analyze(
                prices=prices,
                volume=None,
                market_cap=market_cap
            )
        
        # Handle correlation
        if correlation_data and len(correlation_data) > 1:
            correlation_metrics = self.correlation_analyzer.analyze(
                correlation_data,
                benchmark_id="bitcoin"
            )
        else:
            correlation_metrics = self.correlation_analyzer.analyze(
                {"asset": prices}
            )
        
        # Get regime prediction
        regime, confidence, _ = regime_classifier.predict(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics
        )
        
        # Get probabilities
        probabilities = regime_classifier.get_regime_probabilities(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics
        )
        
        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in probabilities.items()},
            "volatility": round(volatility_metrics.get("current_volatility", 0), 4),
            "volatility_regime": volatility_metrics.get("volatility_regime", "STABLE")
        }
    
    def analyze_multi_timeframe(
        self,
        full_prices: pd.Series,
        full_volume: pd.Series = None,
        market_cap: float = 0,
        full_correlation_data: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze regimes across all timeframes and compute confluence.
        
        Args:
            full_prices: Full price series (at least 90 days)
            full_volume: Optional full volume series
            market_cap: Current market cap
            full_correlation_data: Optional dict of price series
            
        Returns:
            Multi-timeframe analysis with confluence score
        """
        results = {}
        regime_votes = {"TREND": 0, "RANGE": 0, "HIGH-RISK": 0}
        weighted_confidence = 0
        total_weight = 0
        
        for tf_key, tf_config in self.TIMEFRAMES.items():
            days = tf_config["days"]
            weight = tf_config["weight"]
            
            # Slice data for this timeframe
            if len(full_prices) >= days:
                tf_prices = full_prices.tail(days)
                tf_volume = full_volume.tail(days) if isinstance(full_volume, pd.Series) and len(full_volume) >= days else None
                
                # Slice correlation data
                tf_corr_data = None
                if full_correlation_data:
                    tf_corr_data = {
                        k: v.tail(days) for k, v in full_correlation_data.items()
                        if len(v) >= days
                    }
                
                # Analyze this timeframe
                tf_result = self.analyze_single_timeframe(
                    prices=tf_prices,
                    volume=tf_volume,
                    market_cap=market_cap,
                    correlation_data=tf_corr_data
                )
                
                results[tf_key] = {
                    "label": tf_config["label"],
                    "days": days,
                    **tf_result
                }
                
                # Accumulate votes weighted by confidence and timeframe weight
                regime = tf_result["regime"]
                conf = tf_result["confidence"]
                if regime in regime_votes:
                    regime_votes[regime] += weight * conf
                    weighted_confidence += conf * weight
                    total_weight += weight
            else:
                results[tf_key] = {
                    "label": tf_config["label"],
                    "days": days,
                    "regime": "INSUFFICIENT_DATA",
                    "confidence": 0,
                    "probabilities": {}
                }
        
        # Determine dominant regime
        dominant_regime = max(regime_votes, key=regime_votes.get)
        
        # Calculate confluence score (0-100)
        # High confluence = most timeframes agree on the same regime
        regime_values = list(regime_votes.values())
        if sum(regime_values) > 0:
            max_vote = max(regime_values)
            total_votes = sum(regime_values)
            confluence = (max_vote / total_votes) * 100
        else:
            confluence = 0
        
        # Agreement count
        timeframes_agreeing = sum(
            1 for tf in results.values()
            if tf.get("regime") == dominant_regime
        )
        
        # Signal strength based on confluence
        if confluence >= 80:
            signal_strength = "VERY_STRONG"
        elif confluence >= 65:
            signal_strength = "STRONG"
        elif confluence >= 50:
            signal_strength = "MODERATE"
        else:
            signal_strength = "WEAK"
        
        return {
            "timeframes": results,
            "confluence": {
                "dominant_regime": dominant_regime,
                "confluence_score": round(confluence, 1),
                "signal_strength": signal_strength,
                "timeframes_agreeing": timeframes_agreeing,
                "total_timeframes": len([r for r in results.values() if r.get("regime") != "INSUFFICIENT_DATA"]),
                "weighted_confidence": round(weighted_confidence / total_weight, 3) if total_weight > 0 else 0
            },
            "recommendation": self._generate_mtf_recommendation(
                dominant_regime, confluence, signal_strength
            )
        }
    
    def _generate_mtf_recommendation(
        self,
        regime: str,
        confluence: float,
        strength: str
    ) -> str:
        """Generate human-readable multi-timeframe recommendation."""
        
        if strength == "VERY_STRONG":
            conf_text = "Multiple timeframes strongly agree"
        elif strength == "STRONG":
            conf_text = "Most timeframes agree"
        elif strength == "MODERATE":
            conf_text = "Some disagreement across timeframes"
        else:
            conf_text = "Timeframes show conflicting signals"
        
        if regime == "TREND":
            action = "favorable for trend-following strategies"
        elif regime == "RANGE":
            action = "suitable for mean-reversion or range-bound strategies"
        else:
            action = "suggests increased caution and reduced position sizes"
        
        return f"{conf_text} on {regime} regime ({confluence:.0f}% confluence). Market conditions {action}."


# Global analyzer instance
timeframe_analyzer = TimeframeAnalyzer()
