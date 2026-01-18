"""
Volatility feature extraction module.
Computes ATR, standard deviation, volatility regime, and spike detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from app.utils.helpers import (
    calculate_returns,
    calculate_log_returns,
    safe_divide,
    clip_value,
    ewma,
    rolling_zscore
)


class VolatilityAnalyzer:
    """
    Analyzes price volatility using multiple metrics.
    
    Metrics computed:
    - ATR (Average True Range)
    - Rolling standard deviation
    - Volatility regime (expanding/contracting)
    - Price spike detection
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        std_period: int = 20,
        spike_threshold: float = 2.5
    ):
        """
        Initialize the volatility analyzer.
        
        Args:
            atr_period: Period for ATR calculation
            std_period: Period for rolling std dev
            spike_threshold: Z-score threshold for spike detection
        """
        self.atr_period = atr_period
        self.std_period = std_period
        self.spike_threshold = spike_threshold
    
    def calculate_true_range(self, ohlc: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for OHLC data.
        
        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        
        Args:
            ohlc: DataFrame with high, low, close columns
            
        Returns:
            Series of True Range values
        """
        high = ohlc["high"]
        low = ohlc["low"]
        prev_close = ohlc["close"].shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range
    
    def calculate_atr(self, ohlc: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            ohlc: DataFrame with high, low, close columns
            
        Returns:
            Series of ATR values
        """
        true_range = self.calculate_true_range(ohlc)
        atr = true_range.rolling(window=self.atr_period).mean()
        return atr
    
    def calculate_atr_percentage(self, ohlc: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR as percentage of price.
        
        Args:
            ohlc: DataFrame with OHLC data
            
        Returns:
            ATR percentage series
        """
        atr = self.calculate_atr(ohlc)
        atr_pct = safe_divide(atr, ohlc["close"]) * 100
        return atr_pct
    
    def calculate_rolling_volatility(
        self,
        prices: pd.Series,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate rolling standard deviation of returns.
        
        Args:
            prices: Price series
            annualize: Whether to annualize (multiply by sqrt(365))
            
        Returns:
            Rolling volatility series
        """
        returns = calculate_log_returns(prices)
        volatility = returns.rolling(window=self.std_period).std()
        
        if annualize:
            volatility = volatility * np.sqrt(365)
        
        return volatility
    
    def detect_volatility_regime(
        self,
        prices: pd.Series
    ) -> Tuple[str, float]:
        """
        Detect if volatility is expanding or contracting.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (regime, strength)
            regime: "EXPANDING", "CONTRACTING", or "STABLE"
            strength: 0-1 indicating regime strength
        """
        volatility = self.calculate_rolling_volatility(prices, annualize=False)
        
        if len(volatility.dropna()) < self.std_period * 2:
            return "STABLE", 0.5
        
        # Compare recent volatility to historical
        recent_vol = volatility.tail(5).mean()
        historical_vol = volatility.iloc[:-5].mean()
        
        if historical_vol == 0:
            return "STABLE", 0.5
        
        vol_ratio = recent_vol / historical_vol
        
        if vol_ratio > 1.3:
            strength = min(1.0, (vol_ratio - 1.0) / 0.5)
            return "EXPANDING", strength
        elif vol_ratio < 0.7:
            strength = min(1.0, (1.0 - vol_ratio) / 0.5)
            return "CONTRACTING", strength
        else:
            return "STABLE", 1.0 - abs(vol_ratio - 1.0)
    
    def detect_spikes(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Detect abnormal price movements (spikes).
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with spike information
        """
        returns = calculate_returns(prices)
        zscore = rolling_zscore(returns, window=self.std_period)
        
        # Find spikes above threshold
        spike_mask = zscore.abs() > self.spike_threshold
        spike_count = spike_mask.sum()
        
        # Recent spike activity (last 7 periods)
        recent_spikes = spike_mask.tail(7).sum()
        
        # Calculate spike intensity
        if len(zscore.dropna()) > 0:
            max_spike = zscore.abs().max()
            avg_spike = zscore[spike_mask].abs().mean() if spike_count > 0 else 0
        else:
            max_spike = 0
            avg_spike = 0
        
        return {
            "total_spikes": int(spike_count),
            "recent_spikes": int(recent_spikes),
            "max_spike_zscore": float(max_spike) if not np.isnan(max_spike) else 0,
            "avg_spike_zscore": float(avg_spike) if not np.isnan(avg_spike) else 0,
            "spike_rate": float(spike_count / len(returns)) if len(returns) > 0 else 0
        }
    
    def analyze(
        self,
        ohlc: pd.DataFrame = None,
        prices: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Perform complete volatility analysis.
        
        Args:
            ohlc: Optional OHLC DataFrame
            prices: Optional price series (use close from OHLC if not provided)
            
        Returns:
            Dictionary with all volatility metrics
        """
        # Extract prices if OHLC provided
        if isinstance(ohlc, pd.DataFrame) and len(ohlc) > 0 and not isinstance(prices, pd.Series):
            prices = ohlc["close"]
        
        if not isinstance(prices, pd.Series) or len(prices) == 0:
            raise ValueError("Either ohlc or prices must be provided")
        
        # Calculate metrics
        returns = calculate_returns(prices)
        log_returns = calculate_log_returns(prices)
        
        # Current volatility (annualized)
        current_vol = self.calculate_rolling_volatility(prices).iloc[-1]
        
        # ATR metrics (if OHLC available)
        if ohlc is not None:
            atr = self.calculate_atr(ohlc).iloc[-1]
            atr_pct = self.calculate_atr_percentage(ohlc).iloc[-1]
        else:
            atr = None
            atr_pct = None
        
        # Regime detection
        regime, regime_strength = self.detect_volatility_regime(prices)
        
        # Spike detection
        spikes = self.detect_spikes(prices)
        
        # Additional statistics
        returns_mean = returns.mean() * 100
        returns_std = returns.std() * 100
        max_drawdown = self._calculate_max_drawdown(prices)
        
        return {
            "current_volatility": float(current_vol) if not np.isnan(current_vol) else 0,
            "atr": float(atr) if atr is not None and not np.isnan(atr) else None,
            "atr_percentage": float(atr_pct) if atr_pct is not None and not np.isnan(atr_pct) else None,
            "volatility_regime": regime,
            "regime_strength": float(regime_strength),
            "spikes": spikes,
            "returns_mean_pct": float(returns_mean) if not np.isnan(returns_mean) else 0,
            "returns_std_pct": float(returns_std) if not np.isnan(returns_std) else 0,
            "max_drawdown_pct": float(max_drawdown),
            "is_high_volatility": current_vol > 0.8 if not np.isnan(current_vol) else False
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min() * 100
        return float(max_drawdown) if not np.isnan(max_drawdown) else 0


# Default analyzer instance
volatility_analyzer = VolatilityAnalyzer()
