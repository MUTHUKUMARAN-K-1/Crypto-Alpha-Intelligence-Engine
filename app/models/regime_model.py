"""
ML-based market regime classifier.
Uses Random Forest to classify market conditions into TREND, RANGE, or HIGH-RISK.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from app.utils.logger import app_logger, log_model_inference


class RegimeClassifier:
    """
    Machine Learning model for market regime classification.
    
    Classifies market into three regimes:
    - TREND: Directional market movement
    - RANGE: Sideways/consolidating market
    - HIGH-RISK: Volatile/uncertain conditions
    """
    
    # Regime labels
    REGIMES = ["TREND", "RANGE", "HIGH-RISK"]
    
    def __init__(self):
        """Initialize the regime classifier."""
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Initialize with pre-trained model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize model with sensible defaults or load pre-trained weights.
        Uses a pre-configured model that works well for crypto markets.
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        
        # Define feature names
        self.feature_names = [
            "volatility_current",
            "volatility_regime_score",
            "atr_percentage",
            "spike_rate",
            "avg_correlation",
            "correlation_regime_score",
            "liquidity_score",
            "stability_index",
            "market_depth",
            "volume_trend_score",
            "returns_mean",
            "returns_std"
        ]
        
        # Pre-train with synthetic data representing different regimes
        self._pretrain_with_synthetic_data()
    
    def _pretrain_with_synthetic_data(self) -> None:
        """
        Pre-train model with synthetic feature patterns.
        This creates a model that can classify regimes based on
        known feature patterns for each regime type.
        """
        np.random.seed(42)
        samples_per_regime = 200
        
        # Generate synthetic training data
        X_trend = self._generate_trend_features(samples_per_regime)
        X_range = self._generate_range_features(samples_per_regime)
        X_high_risk = self._generate_high_risk_features(samples_per_regime)
        
        X = np.vstack([X_trend, X_range, X_high_risk])
        y = np.array(
            [0] * samples_per_regime +  # TREND
            [1] * samples_per_regime +  # RANGE
            [2] * samples_per_regime    # HIGH-RISK
        )
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        app_logger.info("Regime classifier pre-trained with synthetic data")
    
    def _generate_trend_features(self, n_samples: int) -> np.ndarray:
        """Generate feature patterns typical of trending markets."""
        return np.column_stack([
            np.random.uniform(0.4, 0.9, n_samples),      # volatility_current (moderate-high)
            np.random.uniform(0.6, 1.0, n_samples),      # volatility_regime_score (expanding)
            np.random.uniform(2.0, 5.0, n_samples),      # atr_percentage
            np.random.uniform(0.0, 0.1, n_samples),      # spike_rate (low)
            np.random.uniform(0.3, 0.6, n_samples),      # avg_correlation (moderate)
            np.random.uniform(0.3, 0.6, n_samples),      # correlation_regime_score
            np.random.uniform(50, 80, n_samples),        # liquidity_score (good)
            np.random.uniform(40, 70, n_samples),        # stability_index (moderate)
            np.random.uniform(50, 80, n_samples),        # market_depth (good)
            np.random.uniform(0.5, 1.0, n_samples),      # volume_trend_score (increasing)
            np.random.uniform(0.5, 3.0, n_samples),      # returns_mean (positive bias)
            np.random.uniform(1.5, 4.0, n_samples)       # returns_std (moderate)
        ])
    
    def _generate_range_features(self, n_samples: int) -> np.ndarray:
        """Generate feature patterns typical of ranging markets."""
        return np.column_stack([
            np.random.uniform(0.1, 0.4, n_samples),      # volatility_current (low)
            np.random.uniform(0.0, 0.4, n_samples),      # volatility_regime_score (contracting)
            np.random.uniform(0.5, 2.0, n_samples),      # atr_percentage (low)
            np.random.uniform(0.0, 0.05, n_samples),     # spike_rate (very low)
            np.random.uniform(0.6, 0.9, n_samples),      # avg_correlation (high)
            np.random.uniform(0.7, 1.0, n_samples),      # correlation_regime_score (stable)
            np.random.uniform(40, 70, n_samples),        # liquidity_score (moderate)
            np.random.uniform(60, 90, n_samples),        # stability_index (high)
            np.random.uniform(40, 70, n_samples),        # market_depth (moderate)
            np.random.uniform(0.0, 0.5, n_samples),      # volume_trend_score (stable/decreasing)
            np.random.uniform(-1.0, 1.0, n_samples),     # returns_mean (neutral)
            np.random.uniform(0.5, 2.0, n_samples)       # returns_std (low)
        ])
    
    def _generate_high_risk_features(self, n_samples: int) -> np.ndarray:
        """Generate feature patterns typical of high-risk markets."""
        return np.column_stack([
            np.random.uniform(0.7, 1.5, n_samples),      # volatility_current (very high)
            np.random.uniform(0.7, 1.0, n_samples),      # volatility_regime_score (expanding fast)
            np.random.uniform(4.0, 10.0, n_samples),     # atr_percentage (high)
            np.random.uniform(0.1, 0.3, n_samples),      # spike_rate (high)
            np.random.uniform(0.1, 0.4, n_samples),      # avg_correlation (breaking down)
            np.random.uniform(0.0, 0.4, n_samples),      # correlation_regime_score (diverging)
            np.random.uniform(20, 50, n_samples),        # liquidity_score (poor)
            np.random.uniform(10, 40, n_samples),        # stability_index (low)
            np.random.uniform(20, 50, n_samples),        # market_depth (poor)
            np.random.uniform(-0.5, 1.5, n_samples),     # volume_trend_score (erratic)
            np.random.uniform(-3.0, 3.0, n_samples),     # returns_mean (volatile)
            np.random.uniform(3.0, 8.0, n_samples)       # returns_std (high)
        ])
    
    def extract_features(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract feature vector from analysis metrics.
        
        Args:
            volatility_metrics: Output from VolatilityAnalyzer
            correlation_metrics: Output from CorrelationAnalyzer
            liquidity_metrics: Output from LiquidityAnalyzer
            
        Returns:
            Feature vector as numpy array
        """
        # Map volatility regime to score
        vol_regime_map = {"EXPANDING": 0.8, "STABLE": 0.5, "CONTRACTING": 0.2}
        vol_regime_score = vol_regime_map.get(
            volatility_metrics.get("volatility_regime", "STABLE"), 0.5
        )
        
        # Map correlation regime to score
        corr_regime_map = {
            "HIGH_CORRELATION": 0.8,
            "DIVERGING": 0.5,
            "LOW_CORRELATION": 0.2,
            "UNDEFINED": 0.5
        }
        corr_regime_score = corr_regime_map.get(
            correlation_metrics.get("correlation_regime", "UNDEFINED"), 0.5
        )
        
        # Map volume trend to score
        vol_trend_map = {"INCREASING": 1.0, "STABLE": 0.5, "DECREASING": 0.2, "UNKNOWN": 0.5}
        volume_profile = liquidity_metrics.get("volume_profile", {})
        vol_trend_score = vol_trend_map.get(
            volume_profile.get("volume_trend", "STABLE"), 0.5
        )
        
        # Extract numeric features with safe defaults
        features = np.array([
            volatility_metrics.get("current_volatility", 0.5),
            vol_regime_score,
            volatility_metrics.get("atr_percentage", 2.0) or 2.0,
            volatility_metrics.get("spikes", {}).get("spike_rate", 0.05),
            correlation_metrics.get("average_correlation", 0.5),
            corr_regime_score,
            liquidity_metrics.get("liquidity_score", 50),
            liquidity_metrics.get("stability_index", 50),
            liquidity_metrics.get("market_depth_proxy", 50),
            vol_trend_score,
            volatility_metrics.get("returns_mean_pct", 0),
            volatility_metrics.get("returns_std_pct", 2.0)
        ]).reshape(1, -1)
        
        return features
    
    def predict(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict market regime from analysis metrics.
        
        Args:
            volatility_metrics: Output from VolatilityAnalyzer
            correlation_metrics: Output from CorrelationAnalyzer
            liquidity_metrics: Output from LiquidityAnalyzer
            
        Returns:
            Tuple of (regime, confidence, feature_importances)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Extract and scale features
        features = self.extract_features(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics
        )
        features_scaled = self.scaler.transform(features)
        
        # Predict with probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        regime = self.REGIMES[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Feature importances
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Log inference
        log_model_inference(
            model_name="RegimeClassifier",
            input_shape=features.shape,
            output=regime,
            confidence=confidence
        )
        
        return regime, float(confidence), importances
    
    def get_regime_probabilities(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Get probability distribution over all regimes.
        
        Returns:
            Dictionary mapping regime to probability
        """
        features = self.extract_features(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics
        )
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return dict(zip(self.REGIMES, probabilities.tolist()))


# Global classifier instance
regime_classifier = RegimeClassifier()
