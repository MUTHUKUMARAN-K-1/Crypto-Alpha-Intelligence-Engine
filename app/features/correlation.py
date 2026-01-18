"""
Correlation feature extraction module.
Computes cross-asset correlation, beta, and decorrelation events.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from app.utils.helpers import calculate_returns, safe_divide


class CorrelationAnalyzer:
    """
    Analyzes cross-asset correlations and market relationships.
    
    Metrics computed:
    - Rolling correlation between assets
    - Correlation regime (high/low/diverging)
    - Beta to market benchmark
    - Decorrelation event detection
    """
    
    def __init__(
        self,
        correlation_window: int = 30,
        short_window: int = 7,
        correlation_threshold_high: float = 0.7,
        correlation_threshold_low: float = 0.3
    ):
        """
        Initialize the correlation analyzer.
        
        Args:
            correlation_window: Window for rolling correlation
            short_window: Short window for recent trends
            correlation_threshold_high: Threshold for high correlation
            correlation_threshold_low: Threshold for low correlation
        """
        self.correlation_window = correlation_window
        self.short_window = short_window
        self.threshold_high = correlation_threshold_high
        self.threshold_low = correlation_threshold_low
    
    def calculate_rolling_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> pd.Series:
        """
        Calculate rolling Pearson correlation between two series.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            Rolling correlation series
        """
        returns1 = calculate_returns(series1)
        returns2 = calculate_returns(series2)
        
        # Align the series
        aligned = pd.concat([returns1, returns2], axis=1).dropna()
        if len(aligned) < self.correlation_window:
            return pd.Series([np.nan])
        
        aligned.columns = ["r1", "r2"]
        
        correlation = aligned["r1"].rolling(
            window=self.correlation_window
        ).corr(aligned["r2"])
        
        return correlation
    
    def calculate_pairwise_correlations(
        self,
        price_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlations for all assets.
        
        Args:
            price_dict: Dictionary of asset_id -> price series
            
        Returns:
            Correlation matrix DataFrame
        """
        # Convert to returns
        returns_dict = {
            asset: calculate_returns(prices)
            for asset, prices in price_dict.items()
        }
        
        # Align all series
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        if len(returns_df) < 5:
            return pd.DataFrame()
        
        return returns_df.corr()
    
    def calculate_average_correlation(
        self,
        price_dict: Dict[str, pd.Series]
    ) -> float:
        """
        Calculate average pairwise correlation across all assets.
        
        Args:
            price_dict: Dictionary of asset prices
            
        Returns:
            Average correlation value
        """
        corr_matrix = self.calculate_pairwise_correlations(price_dict)
        
        if corr_matrix.empty:
            return 0.0
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_values = corr_matrix.where(mask).stack()
        
        if len(upper_values) == 0:
            return 0.0
        
        return float(upper_values.mean())
    
    def calculate_beta(
        self,
        asset_prices: pd.Series,
        benchmark_prices: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate beta of asset relative to benchmark.
        
        Beta = Cov(asset, benchmark) / Var(benchmark)
        
        Args:
            asset_prices: Asset price series
            benchmark_prices: Benchmark (e.g., BTC) price series
            
        Returns:
            Tuple of (beta, r_squared)
        """
        asset_returns = calculate_returns(asset_prices)
        benchmark_returns = calculate_returns(benchmark_prices)
        
        # Align
        aligned = pd.concat(
            [asset_returns, benchmark_returns],
            axis=1
        ).dropna()
        
        if len(aligned) < 10:
            return 1.0, 0.0
        
        aligned.columns = ["asset", "benchmark"]
        
        covariance = aligned["asset"].cov(aligned["benchmark"])
        variance = aligned["benchmark"].var()
        
        beta = safe_divide(covariance, variance, default=1.0)
        
        # R-squared
        correlation = aligned["asset"].corr(aligned["benchmark"])
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
        
        return float(beta), float(r_squared)
    
    def detect_correlation_regime(
        self,
        price_dict: Dict[str, pd.Series]
    ) -> Tuple[str, float]:
        """
        Detect the current correlation regime.
        
        Args:
            price_dict: Dictionary of asset prices
            
        Returns:
            Tuple of (regime, confidence)
            regime: "HIGH_CORRELATION", "LOW_CORRELATION", "DIVERGING"
        """
        avg_corr = self.calculate_average_correlation(price_dict)
        
        # Also check trend of correlation
        # Need at least 2 assets for meaningful analysis
        if len(price_dict) < 2:
            return "UNDEFINED", 0.5
        
        if avg_corr >= self.threshold_high:
            confidence = min(1.0, (avg_corr - self.threshold_high) / 0.2 + 0.7)
            return "HIGH_CORRELATION", confidence
        elif avg_corr <= self.threshold_low:
            confidence = min(1.0, (self.threshold_low - avg_corr) / 0.2 + 0.7)
            return "LOW_CORRELATION", confidence
        else:
            # Middle ground - "DIVERGING" or transitioning
            return "DIVERGING", 0.6
    
    def detect_decorrelation_events(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Detect sudden decorrelation events between two series.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            List of decorrelation events with dates and magnitudes
        """
        rolling_corr = self.calculate_rolling_correlation(series1, series2)
        
        if len(rolling_corr.dropna()) < self.correlation_window * 2:
            return []
        
        # Calculate change in correlation
        corr_change = rolling_corr.diff()
        
        # Significant drops in correlation
        threshold = corr_change.std() * 2
        events = []
        
        for idx, change in corr_change.items():
            if not np.isnan(change) and change < -threshold:
                events.append({
                    "date": str(idx),
                    "correlation_drop": float(change),
                    "correlation_before": float(rolling_corr.get(idx - pd.Timedelta(days=1), np.nan)),
                    "correlation_after": float(rolling_corr.get(idx, np.nan))
                })
        
        return events[-5:]  # Return last 5 events
    
    def analyze(
        self,
        price_dict: Dict[str, pd.Series],
        benchmark_id: str = "bitcoin"
    ) -> Dict[str, Any]:
        """
        Perform complete correlation analysis.
        
        Args:
            price_dict: Dictionary of asset_id -> price series
            benchmark_id: ID of benchmark asset for beta calculation
            
        Returns:
            Dictionary with all correlation metrics
        """
        if len(price_dict) < 1:
            return {
                "average_correlation": 0.0,
                "correlation_regime": "UNDEFINED",
                "regime_confidence": 0.0,
                "betas": {},
                "correlation_matrix": {}
            }
        
        # Average correlation
        avg_corr = self.calculate_average_correlation(price_dict)
        
        # Correlation regime
        regime, regime_confidence = self.detect_correlation_regime(price_dict)
        
        # Beta calculations (if benchmark available)
        betas = {}
        benchmark_prices = price_dict.get(benchmark_id)
        
        if isinstance(benchmark_prices, pd.Series) and len(benchmark_prices) > 0:
            for asset_id, prices in price_dict.items():
                if asset_id != benchmark_id:
                    beta, r_squared = self.calculate_beta(prices, benchmark_prices)
                    betas[asset_id] = {
                        "beta": beta,
                        "r_squared": r_squared
                    }
        
        # Correlation matrix
        corr_matrix = self.calculate_pairwise_correlations(price_dict)
        corr_dict = {}
        if not corr_matrix.empty:
            for col in corr_matrix.columns:
                corr_dict[col] = {
                    k: float(v) if not np.isnan(v) else 0.0
                    for k, v in corr_matrix[col].items()
                }
        
        return {
            "average_correlation": float(avg_corr) if not np.isnan(avg_corr) else 0.0,
            "correlation_regime": regime,
            "regime_confidence": float(regime_confidence),
            "betas": betas,
            "correlation_matrix": corr_dict,
            "is_highly_correlated": avg_corr >= self.threshold_high,
            "asset_count": len(price_dict)
        }


# Default analyzer instance
correlation_analyzer = CorrelationAnalyzer()
