"""
ML-based risk and tradability scoring model.
Uses Gradient Boosting to compute tradability scores and risk levels.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from app.utils.logger import app_logger, log_model_inference
from app.utils.helpers import clip_value


class RiskModel:
    """
    Machine Learning model for tradability and risk scoring.
    
    Outputs:
    - Tradability score (0-100): Higher = better trading conditions
    - Risk level: LOW, MEDIUM, HIGH
    """
    
    RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
    
    def __init__(self):
        """Initialize the risk model."""
        self.model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Initialize with pre-trained model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize model with sensible defaults."""
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            learning_rate=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
        # Define feature names
        self.feature_names = [
            "volatility_current",
            "volatility_regime_score",
            "atr_percentage",
            "spike_rate",
            "avg_correlation",
            "liquidity_score",
            "stability_index",
            "market_depth",
            "volume_trend_score",
            "max_drawdown",
            "regime_encoded"
        ]
        
        # Pre-train with synthetic data
        self._pretrain_with_synthetic_data()
    
    def _pretrain_with_synthetic_data(self) -> None:
        """Pre-train with synthetic tradability patterns."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate features
        X = np.column_stack([
            np.random.uniform(0.1, 1.2, n_samples),      # volatility
            np.random.uniform(0.0, 1.0, n_samples),      # vol regime
            np.random.uniform(0.5, 8.0, n_samples),      # atr pct
            np.random.uniform(0.0, 0.25, n_samples),     # spike rate
            np.random.uniform(0.1, 0.9, n_samples),      # correlation
            np.random.uniform(20, 90, n_samples),        # liquidity
            np.random.uniform(20, 90, n_samples),        # stability
            np.random.uniform(20, 90, n_samples),        # depth
            np.random.uniform(0.0, 1.0, n_samples),      # volume trend
            np.random.uniform(-30, 0, n_samples),        # max drawdown
            np.random.choice([0, 1, 2], n_samples)       # regime (0=trend, 1=range, 2=high-risk)
        ])
        
        # Generate target (tradability score) based on feature logic
        y = self._generate_tradability_targets(X)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        app_logger.info("Risk model pre-trained with synthetic data")
    
    def _generate_tradability_targets(self, X: np.ndarray) -> np.ndarray:
        """
        Generate target tradability scores based on feature values.
        
        Higher tradability when:
        - Moderate volatility (not too low, not too high)
        - Good liquidity
        - High stability
        - Low spike rate
        - Favorable regime (TREND > RANGE > HIGH-RISK)
        """
        scores = []
        
        for row in X:
            vol, vol_regime, atr, spike_rate, corr, liq, stab, depth, vol_trend, drawdown, regime = row
            
            # Base score from liquidity and stability
            base = (liq * 0.3 + stab * 0.25 + depth * 0.2)
            
            # Volatility penalty (too high or too low is bad)
            if vol < 0.2:
                vol_factor = 0.7  # Too quiet
            elif vol > 0.9:
                vol_factor = 0.6  # Too volatile
            else:
                vol_factor = 1.0  # Goldilocks zone
            
            # Spike penalty
            spike_penalty = max(0, 1 - spike_rate * 3)
            
            # Regime bonus/penalty
            regime_factor = {0: 1.1, 1: 0.9, 2: 0.6}[int(regime)]
            
            # Drawdown penalty
            drawdown_factor = max(0.5, 1 + drawdown / 50)
            
            # Calculate final score
            score = base * vol_factor * spike_penalty * regime_factor * drawdown_factor
            
            # Add some noise
            score += np.random.normal(0, 5)
            
            scores.append(clip_value(score, 0, 100))
        
        return np.array(scores)
    
    def extract_features(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any],
        regime: str
    ) -> np.ndarray:
        """
        Extract feature vector for risk prediction.
        
        Args:
            volatility_metrics: Output from VolatilityAnalyzer
            correlation_metrics: Output from CorrelationAnalyzer
            liquidity_metrics: Output from LiquidityAnalyzer
            regime: Predicted regime from RegimeClassifier
            
        Returns:
            Feature vector as numpy array
        """
        # Map volatility regime to score
        vol_regime_map = {"EXPANDING": 0.8, "STABLE": 0.5, "CONTRACTING": 0.2}
        vol_regime_score = vol_regime_map.get(
            volatility_metrics.get("volatility_regime", "STABLE"), 0.5
        )
        
        # Map volume trend to score
        vol_trend_map = {"INCREASING": 1.0, "STABLE": 0.5, "DECREASING": 0.2, "UNKNOWN": 0.5}
        volume_profile = liquidity_metrics.get("volume_profile", {})
        vol_trend_score = vol_trend_map.get(
            volume_profile.get("volume_trend", "STABLE"), 0.5
        )
        
        # Encode regime
        regime_map = {"TREND": 0, "RANGE": 1, "HIGH-RISK": 2}
        regime_encoded = regime_map.get(regime, 1)
        
        features = np.array([
            volatility_metrics.get("current_volatility", 0.5),
            vol_regime_score,
            volatility_metrics.get("atr_percentage", 2.0) or 2.0,
            volatility_metrics.get("spikes", {}).get("spike_rate", 0.05),
            correlation_metrics.get("average_correlation", 0.5),
            liquidity_metrics.get("liquidity_score", 50),
            liquidity_metrics.get("stability_index", 50),
            liquidity_metrics.get("market_depth_proxy", 50),
            vol_trend_score,
            volatility_metrics.get("max_drawdown_pct", -10),
            regime_encoded
        ]).reshape(1, -1)
        
        return features
    
    def predict(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any],
        regime: str
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Predict tradability score and risk level.
        
        Args:
            volatility_metrics: Output from VolatilityAnalyzer
            correlation_metrics: Output from CorrelationAnalyzer
            liquidity_metrics: Output from LiquidityAnalyzer
            regime: Predicted regime from RegimeClassifier
            
        Returns:
            Tuple of (tradability_score, risk_level, details)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Extract and scale features
        features = self.extract_features(
            volatility_metrics,
            correlation_metrics,
            liquidity_metrics,
            regime
        )
        features_scaled = self.scaler.transform(features)
        
        # Predict tradability score
        raw_score = self.model.predict(features_scaled)[0]
        tradability_score = int(clip_value(raw_score, 0, 100))
        
        # Determine risk level
        if tradability_score >= 70:
            risk_level = "LOW"
        elif tradability_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Feature importance for this prediction
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Additional details
        details = {
            "raw_score": float(raw_score),
            "feature_importances": importances,
            "top_factors": self._get_top_factors(features[0], importances)
        }
        
        # Log inference
        log_model_inference(
            model_name="RiskModel",
            input_shape=features.shape,
            output=f"{tradability_score} ({risk_level})",
            confidence=tradability_score / 100
        )
        
        return tradability_score, risk_level, details
    
    def _get_top_factors(
        self,
        features: np.ndarray,
        importances: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Get top contributing factors for the prediction.
        
        Returns:
            List of top 3 factors with their values and contributions
        """
        # Sort by importance
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        top_factors = []
        for feat_name, importance in sorted_features:
            idx = self.feature_names.index(feat_name)
            top_factors.append({
                "feature": feat_name,
                "value": float(features[idx]),
                "importance": float(importance)
            })
        
        return top_factors


# Global risk model instance
risk_model = RiskModel()
