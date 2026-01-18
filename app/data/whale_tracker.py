"""
Whale Tracking Module.
Tracks large volume anomalies and market flow to estimate whale activity.
Uses REAL market data (Volume/Price Divergence) as a proxy for on-chain keys.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from app.utils.logger import app_logger


class WhaleTracker:
    """
    Tracks 'Whale' activity using Volume Anomaly Detection.
    
    Since we do not have paid on-chain APIs (Glassnode/WhaleAlert), we use
    proven market microstructure proxies:
    1. Volume Spikes > 2σ (Standard Deviations) = Institutional Activity
    2. Price/Volume Divergence = Accumulation/Distribution
    3. Large Trade Estimation based on volume blocks
    """
    
    def __init__(self):
        self._cache = {}
    
    async def get_whale_sentiment(
        self, 
        asset: str, 
        price_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze whale sentiment using real market data.
        
        Args:
            asset: Asset ID
            price_data: DataFrame with 'close' and 'volume' columns
            
        Returns:
            Whale sentiment analysis based on volume flows
        """
        if price_data is None or price_data.empty:
            # Fallback if no data provided (should not happen in proper flow)
            return self._get_fallback_data(asset)
            
        try:
            # 1. Detect Volume Anomalies
            analysis = self._analyze_volume_flow(price_data)
            
            # 2. Derive Sentiment from Flow
            sentiment_score = analysis["sentiment_score"]
            
            return {
                "asset": asset,
                "whale_sentiment_score": round(sentiment_score, 1),
                "sentiment_label": self._score_to_label(sentiment_score),
                "net_flow": analysis["net_money_flow"],
                "net_flow_str": f"${analysis['net_money_flow']:,.0f}",
                "activity_level": analysis["activity_level"],
                "signals": analysis["signals"],
                "regime_adjustments": self._calculate_regime_adjustments(sentiment_score, analysis["activity_level"]),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            app_logger.error(f"Whale tracking failed: {e}")
            return self._get_fallback_data(asset)

    def _analyze_volume_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume and price for institutional footprints."""
        # Ensure we have data
        if "volume" not in df.columns or "close" not in df.columns:
            raise ValueError("Missing volume/close data")
            
        # Look at last 14 periods
        recent = df.tail(14).copy()
        
        # Calculate Volume Moving Average and Std Dev
        recent["vol_ma"] = recent["volume"].rolling(20).mean() # Approximate if short data
        vol_mean = recent["volume"].mean()
        vol_std = recent["volume"].std()
        
        # Identify "Whale Candles" (Volume > 1.5 sigma)
        recent["is_whale"] = recent["volume"] > (vol_mean + 1.5 * vol_std)
        
        # Calculate Money Flow for Whale Candles
        # If Price went UP on Whale Volume -> Buying (Accumulation)
        # If Price went DOWN on Whale Volume -> Selling (Distribution)
        recent["price_change"] = recent["close"].diff()
        
        whale_buy_vol = recent[recent["is_whale"] & (recent["price_change"] > 0)]["volume"].sum()
        whale_sell_vol = recent[recent["is_whale"] & (recent["price_change"] < 0)]["volume"].sum()
        
        # Net Flow (Volume * Avg Price roughly estimates $)
        avg_price = recent["close"].mean()
        net_vol_flow = whale_buy_vol - whale_sell_vol
        net_money_flow = net_vol_flow * avg_price
        
        # Calculate Activity Level
        whale_candles_count = recent["is_whale"].sum()
        if whale_candles_count >= 4:
            activity = "VERY_HIGH"
        elif whale_candles_count >= 2:
            activity = "HIGH"
        elif whale_candles_count >= 1:
            activity = "MODERATE"
        else:
            activity = "LOW"
            
        # Calculate Sentiment Score (0-100)
        # 50 is neutral. 
        total_whale_vol = whale_buy_vol + whale_sell_vol
        if total_whale_vol > 0:
            buy_ratio = whale_buy_vol / total_whale_vol
            sentiment_score = buy_ratio * 100
        else:
            sentiment_score = 50
            
        # Generate Signals
        signals = []
        if net_money_flow > 0 and activity in ["HIGH", "VERY_HIGH"]:
             signals.append({"description": "Strong volume accumulation detected", "type": "BUY"})
        elif net_money_flow < 0 and activity in ["HIGH", "VERY_HIGH"]:
             signals.append({"description": "Heavy institutional selling pressure", "type": "SELL"})
        elif activity == "LOW":
             signals.append({"description": "Low institutional participation", "type": "NEUTRAL"})
        else:
             signals.append({"description": "Mixed whale activity", "type": "NEUTRAL"})

        return {
            "sentiment_score": sentiment_score,
            "net_money_flow": net_money_flow,
            "activity_level": activity,
            "signals": signals
        }

    def _score_to_label(self, score: float) -> str:
        if score >= 65: return "BULLISH"
        if score >= 55: return "SLIGHTLY_BULLISH"
        if score <= 35: return "BEARISH"
        if score <= 45: return "SLIGHTLY_BEARISH"
        return "NEUTRAL"

    def _calculate_regime_adjustments(self, score: float, activity: str) -> Dict[str, float]:
        # Simple adjust: High Score + High Activity = Favors Trend
        val = (score - 50) / 100 # -0.5 to 0.5
        mult = 1.5 if activity == "HIGH" else 1.0
        return {"TREND": val * 0.2 * mult} # Placeholder

    def _get_fallback_data(self, asset: str) -> Dict[str, Any]:
        return {
            "asset": asset,
            "whale_sentiment_score": 50,
            "sentiment_label": "NEUTRAL",
            "net_flow": 0,
            "activity_level": "LOW",
            "signals": [{"description": "Insufficient volume data for whale tracking", "type": "NEUTRAL"}],
            "regime_adjustments": {}
        }

whale_tracker = WhaleTracker()
