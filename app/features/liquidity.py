"""
Liquidity feature extraction module.
Computes volume profile, liquidity score, stability index, and market depth proxy.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from app.utils.helpers import safe_divide, ewma, clip_value, normalize_to_range


class LiquidityAnalyzer:
    """
    Analyzes market liquidity using volume and price data.
    
    Metrics computed:
    - Volume profile and trends
    - Liquidity score (volume/market cap ratio)
    - Price stability index
    - Market depth proxy
    """
    
    def __init__(
        self,
        volume_window: int = 20,
        stability_window: int = 14,
        trend_window: int = 7
    ):
        """
        Initialize the liquidity analyzer.
        
        Args:
            volume_window: Window for volume analysis
            stability_window: Window for stability calculations
            trend_window: Short window for trend detection
        """
        self.volume_window = volume_window
        self.stability_window = stability_window
        self.trend_window = trend_window
    
    def calculate_volume_profile(
        self,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze volume patterns and trends.
        
        Args:
            volume: Volume series
            
        Returns:
            Dictionary with volume profile metrics
        """
        if len(volume) == 0 or len(volume) < self.volume_window:
            return {
                "avg_volume": 0,
                "volume_trend": "STABLE",
                "volume_volatility": 0,
                "relative_volume": 1.0
            }
        
        # Average volume
        avg_volume = volume.mean()
        
        # Recent vs historical volume
        recent_avg = volume.tail(self.trend_window).mean()
        historical_avg = volume.iloc[:-self.trend_window].mean()
        
        relative_volume = safe_divide(recent_avg, historical_avg, default=1.0)
        
        # Volume trend
        if relative_volume > 1.2:
            volume_trend = "INCREASING"
        elif relative_volume < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"
        
        # Volume volatility (coefficient of variation)
        volume_std = volume.rolling(window=self.volume_window).std().iloc[-1]
        volume_mean = volume.rolling(window=self.volume_window).mean().iloc[-1]
        volume_volatility = safe_divide(volume_std, volume_mean, default=0.0)
        
        return {
            "avg_volume": float(avg_volume) if not np.isnan(avg_volume) else 0,
            "volume_trend": volume_trend,
            "volume_volatility": float(volume_volatility) if not np.isnan(volume_volatility) else 0,
            "relative_volume": float(relative_volume) if not np.isnan(relative_volume) else 1.0
        }
    
    def calculate_liquidity_score(
        self,
        volume: pd.Series,
        market_cap: float
    ) -> float:
        """
        Calculate liquidity score based on volume/market cap ratio.
        
        Higher ratio = more liquid market.
        
        Args:
            volume: Volume series
            market_cap: Current market cap
            
        Returns:
            Liquidity score (0-100)
        """
        if market_cap <= 0 or len(volume) == 0:
            return 50.0  # Default middle score
        
        avg_volume = volume.tail(self.trend_window).mean()
        
        # Volume to market cap ratio
        vol_mcap_ratio = safe_divide(avg_volume, market_cap)
        
        # Normalize to 0-100 scale
        # Typical crypto vol/mcap ratio ranges from 0.01 to 0.5
        score = normalize_to_range(
            vol_mcap_ratio,
            min_val=0.01,
            max_val=0.3,
            target_min=0,
            target_max=100
        )
        
        return clip_value(score, 0, 100)
    
    def calculate_stability_index(
        self,
        prices: pd.Series
    ) -> float:
        """
        Calculate price stability index.
        
        Lower volatility = higher stability.
        
        Args:
            prices: Price series
            
        Returns:
            Stability index (0-100, higher = more stable)
        """
        if len(prices) == 0 or len(prices) < self.stability_window:
            return 50.0
        
        # Calculate rolling volatility
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=self.stability_window).std().iloc[-1]
        
        if np.isnan(volatility):
            return 50.0
        
        # Convert volatility to stability (inverse relationship)
        # Typical daily volatility ranges from 0.01 to 0.1
        stability = normalize_to_range(
            volatility,
            min_val=0.01,
            max_val=0.1,
            target_min=100,  # Low vol = high stability
            target_max=0     # High vol = low stability
        )
        
        return clip_value(stability, 0, 100)
    
    def calculate_market_depth_proxy(
        self,
        prices: pd.Series,
        volume: pd.Series
    ) -> float:
        """
        Calculate proxy for market depth (ability to absorb large orders).
        
        Based on price impact estimation: how much does price move per unit volume?
        
        Args:
            prices: Price series
            volume: Volume series
            
        Returns:
            Market depth proxy (0-100, higher = deeper market)
        """
        if len(prices) < 10 or len(volume) < 10:
            return 50.0
        
        # Calculate price changes
        price_changes = prices.pct_change().abs()
        
        # Align series
        aligned = pd.concat([price_changes, volume], axis=1).dropna()
        if len(aligned) < 5:
            return 50.0
        
        aligned.columns = ["price_change", "volume"]
        
        # Price impact = price change / volume (lower = better)
        # Use median to be robust to outliers
        avg_volume = aligned["volume"].mean()
        if avg_volume <= 0:
            return 50.0
        
        # Normalize volume and calculate impact ratio
        aligned["norm_volume"] = aligned["volume"] / avg_volume
        aligned["impact"] = safe_divide(
            aligned["price_change"],
            aligned["norm_volume"],
            default=0.01
        )
        
        avg_impact = aligned["impact"].median()
        
        if np.isnan(avg_impact):
            return 50.0
        
        # Convert to depth score (inverse relationship)
        depth = normalize_to_range(
            avg_impact,
            min_val=0.001,
            max_val=0.05,
            target_min=100,  # Low impact = high depth
            target_max=0
        )
        
        return clip_value(depth, 0, 100)
    
    def detect_liquidity_regime(
        self,
        liquidity_score: float,
        stability_index: float,
        depth_proxy: float
    ) -> Tuple[str, float]:
        """
        Detect overall liquidity regime.
        
        Args:
            liquidity_score: Liquidity score
            stability_index: Price stability
            depth_proxy: Market depth
            
        Returns:
            Tuple of (regime, confidence)
        """
        # Weighted average
        composite = (
            liquidity_score * 0.4 +
            stability_index * 0.3 +
            depth_proxy * 0.3
        )
        
        if composite >= 70:
            return "HIGH_LIQUIDITY", min(1.0, (composite - 70) / 30 + 0.7)
        elif composite >= 40:
            return "MODERATE_LIQUIDITY", 0.6
        else:
            return "LOW_LIQUIDITY", min(1.0, (40 - composite) / 40 + 0.6)
    
    def analyze(
        self,
        prices: pd.Series,
        volume: pd.Series = None,
        market_cap: float = None
    ) -> Dict[str, Any]:
        """
        Perform complete liquidity analysis.
        
        Args:
            prices: Price series
            volume: Optional volume series
            market_cap: Optional current market cap
            
        Returns:
            Dictionary with all liquidity metrics
        """
        # Volume profile
        has_volume = False
        if isinstance(volume, pd.Series) and len(volume) > 0:
            volume_profile = self.calculate_volume_profile(volume)
            has_volume = True
        else:
            volume_profile = {
                "avg_volume": 0,
                "volume_trend": "UNKNOWN",
                "volume_volatility": 0,
                "relative_volume": 1.0
            }
        
        # Liquidity score
        if has_volume and market_cap is not None and market_cap > 0:
            liquidity_score = self.calculate_liquidity_score(volume, market_cap)
        else:
            liquidity_score = 50.0
        
        # Stability index
        stability_index = self.calculate_stability_index(prices)
        
        # Market depth proxy
        if has_volume:
            depth_proxy = self.calculate_market_depth_proxy(prices, volume)
        else:
            depth_proxy = 50.0
        
        # Liquidity regime
        regime, regime_confidence = self.detect_liquidity_regime(
            liquidity_score,
            stability_index,
            depth_proxy
        )
        
        return {
            "volume_profile": volume_profile,
            "liquidity_score": float(liquidity_score),
            "stability_index": float(stability_index),
            "market_depth_proxy": float(depth_proxy),
            "liquidity_regime": regime,
            "regime_confidence": float(regime_confidence),
            "is_liquid": liquidity_score >= 60,
            "is_stable": stability_index >= 60
        }


# Default analyzer instance
liquidity_analyzer = LiquidityAnalyzer()
