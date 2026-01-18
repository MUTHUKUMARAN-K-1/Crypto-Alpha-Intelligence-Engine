"""
Funding Rate Predictor - Novel AI for perpetual futures trading.
Predicts funding rate direction to profit from carry trades.

When funding rate is negative: Shorts pay longs (go long to collect)
When funding rate is positive: Longs pay shorts (go short to collect)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from app.utils.logger import app_logger


@dataclass
class FundingPrediction:
    """Funding rate prediction result."""
    direction: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    probability: float  # Probability of predicted direction
    current_rate: float
    predicted_rate: float
    suggested_action: str  # "LONG", "SHORT", "WAIT"
    confidence: float
    reasoning: str


class FundingRatePredictor:
    """
    Neural network-style model to predict funding rate direction.
    
    Uses historical funding rates, open interest, volume momentum,
    and price action to predict the next funding rate direction.
    
    Trading Logic:
    - Negative funding predicted → Go long (shorts will pay longs)
    - Positive funding predicted → Go short (longs will pay shorts)
    - Neutral/uncertain → Wait for clearer signal
    """
    
    def __init__(self):
        """Initialize the funding rate predictor."""
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            "funding_rate_ma_8h",
            "funding_rate_ma_24h", 
            "funding_rate_momentum",
            "funding_rate_volatility",
            "open_interest_change",
            "volume_momentum",
            "price_momentum_24h",
            "price_volatility",
            "long_short_ratio",
            "btc_correlation"
        ]
        
        # Initialize with synthetic training
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with synthetic training data."""
        # Generate synthetic training data based on market patterns
        np.random.seed(42)
        n_samples = 2000
        
        # Create feature matrix
        X = np.zeros((n_samples, len(self.feature_names)))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Simulate different market conditions
            market_type = np.random.choice(["bullish", "bearish", "neutral"])
            
            if market_type == "bullish":
                # Bullish: positive momentum, increasing OI, funding tends positive
                X[i, 0] = np.random.uniform(0.001, 0.01)  # funding_rate_ma_8h
                X[i, 1] = np.random.uniform(0.0005, 0.008)  # funding_rate_ma_24h
                X[i, 2] = np.random.uniform(0, 0.005)  # funding_rate_momentum
                X[i, 3] = np.random.uniform(0.001, 0.003)  # funding_rate_volatility
                X[i, 4] = np.random.uniform(0, 0.1)  # open_interest_change
                X[i, 5] = np.random.uniform(0, 0.2)  # volume_momentum
                X[i, 6] = np.random.uniform(0, 0.05)  # price_momentum_24h
                X[i, 7] = np.random.uniform(0.01, 0.03)  # price_volatility
                X[i, 8] = np.random.uniform(1.0, 2.0)  # long_short_ratio
                X[i, 9] = np.random.uniform(0.3, 0.8)  # btc_correlation
                y[i] = 1  # Positive funding likely
                
            elif market_type == "bearish":
                # Bearish: negative momentum, funding tends negative
                X[i, 0] = np.random.uniform(-0.01, -0.001)
                X[i, 1] = np.random.uniform(-0.008, -0.0005)
                X[i, 2] = np.random.uniform(-0.005, 0)
                X[i, 3] = np.random.uniform(0.002, 0.005)
                X[i, 4] = np.random.uniform(-0.1, 0)
                X[i, 5] = np.random.uniform(-0.2, 0)
                X[i, 6] = np.random.uniform(-0.05, 0)
                X[i, 7] = np.random.uniform(0.02, 0.05)
                X[i, 8] = np.random.uniform(0.5, 1.0)
                X[i, 9] = np.random.uniform(0.5, 0.9)
                y[i] = 0  # Negative funding likely
                
            else:
                # Neutral: mixed signals
                X[i, 0] = np.random.uniform(-0.003, 0.003)
                X[i, 1] = np.random.uniform(-0.002, 0.002)
                X[i, 2] = np.random.uniform(-0.002, 0.002)
                X[i, 3] = np.random.uniform(0.001, 0.002)
                X[i, 4] = np.random.uniform(-0.05, 0.05)
                X[i, 5] = np.random.uniform(-0.1, 0.1)
                X[i, 6] = np.random.uniform(-0.02, 0.02)
                X[i, 7] = np.random.uniform(0.01, 0.02)
                X[i, 8] = np.random.uniform(0.9, 1.1)
                X[i, 9] = np.random.uniform(0.4, 0.7)
                y[i] = np.random.choice([0, 1])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        app_logger.info("Funding Rate Predictor initialized with synthetic training")
    
    def extract_features(
        self,
        funding_history: List[Dict[str, Any]],
        open_interest_history: List[float],
        volume_history: List[float],
        price_history: List[float],
        btc_price_history: List[float] = None
    ) -> np.ndarray:
        """
        Extract features from market data.
        
        Args:
            funding_history: List of historical funding rate records
            open_interest_history: Open interest values
            volume_history: Volume values
            price_history: Price values
            btc_price_history: BTC prices for correlation
            
        Returns:
            Feature array
        """
        features = np.zeros(len(self.feature_names))
        
        # Extract funding rates
        funding_rates = [f.get("fundingRate", f.get("rate", 0)) for f in funding_history]
        if len(funding_rates) < 3:
            funding_rates = [0.0001] * 8  # Default
        
        # Funding rate moving averages
        features[0] = np.mean(funding_rates[-3:])  # ~8h MA (3 x 8h periods)
        features[1] = np.mean(funding_rates[-9:]) if len(funding_rates) >= 9 else np.mean(funding_rates)  # ~24h MA
        
        # Funding rate momentum and volatility
        features[2] = funding_rates[-1] - features[1] if len(funding_rates) > 0 else 0
        features[3] = np.std(funding_rates[-9:]) if len(funding_rates) >= 9 else 0.001
        
        # Open interest change
        if len(open_interest_history) >= 2:
            features[4] = (open_interest_history[-1] - open_interest_history[0]) / (open_interest_history[0] + 1e-10)
        
        # Volume momentum
        if len(volume_history) >= 2:
            features[5] = (volume_history[-1] - np.mean(volume_history[:-1])) / (np.mean(volume_history) + 1e-10)
        
        # Price momentum and volatility
        if len(price_history) >= 2:
            features[6] = (price_history[-1] - price_history[0]) / (price_history[0] + 1e-10)
            features[7] = np.std(np.diff(price_history) / (np.array(price_history[:-1]) + 1e-10))
        
        # Long/short ratio (approximated from funding)
        current_funding = funding_rates[-1] if funding_rates else 0
        features[8] = 1.0 + current_funding * 100  # Higher funding = more longs
        
        # BTC correlation
        if btc_price_history and len(btc_price_history) == len(price_history) and len(price_history) > 1:
            price_returns = np.diff(price_history) / (np.array(price_history[:-1]) + 1e-10)
            btc_returns = np.diff(btc_price_history) / (np.array(btc_price_history[:-1]) + 1e-10)
            if len(price_returns) > 1:
                features[9] = np.corrcoef(price_returns, btc_returns)[0, 1]
        else:
            features[9] = 0.7  # Default high correlation
        
        return features
    
    def predict(
        self,
        funding_history: List[Dict[str, Any]],
        open_interest_history: List[float] = None,
        volume_history: List[float] = None,
        price_history: List[float] = None,
        btc_price_history: List[float] = None
    ) -> FundingPrediction:
        """
        Predict next funding rate direction.
        
        Args:
            funding_history: Historical funding rate records
            open_interest_history: Open interest values
            volume_history: Volume values
            price_history: Price values
            btc_price_history: BTC prices for correlation
            
        Returns:
            FundingPrediction with direction and suggested action
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Default histories if not provided
        if open_interest_history is None:
            open_interest_history = [1e9] * 10
        if volume_history is None:
            volume_history = [1e8] * 10
        if price_history is None:
            price_history = [50000] * 10
        
        # Extract features
        features = self.extract_features(
            funding_history,
            open_interest_history,
            volume_history,
            price_history,
            btc_price_history
        )
        
        # Scale and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction probabilities
        proba = self.model.predict_proba(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        # Current funding rate
        current_rate = funding_history[-1].get("fundingRate", funding_history[-1].get("rate", 0)) if funding_history else 0
        
        # Determine direction and confidence
        if prediction == 1:
            direction = "POSITIVE"
            probability = proba[1]
        else:
            direction = "NEGATIVE"
            probability = proba[0]
        
        # Handle neutral cases
        confidence = abs(proba[1] - 0.5) * 2  # 0 to 1 scale
        if confidence < 0.2:
            direction = "NEUTRAL"
        
        # Determine suggested action
        if direction == "NEGATIVE" and confidence > 0.3:
            suggested_action = "LONG"  # Collect funding from shorts
            predicted_rate = current_rate - 0.001 * confidence
        elif direction == "POSITIVE" and confidence > 0.3:
            suggested_action = "SHORT"  # Collect funding from longs
            predicted_rate = current_rate + 0.001 * confidence
        else:
            suggested_action = "WAIT"
            predicted_rate = current_rate
        
        # Generate reasoning
        if suggested_action == "LONG":
            reasoning = (
                f"Funding rate predicted to go negative ({predicted_rate*100:.4f}%). "
                f"Short positions will pay {abs(predicted_rate)*100:.4f}% to longs. "
                f"Long position recommended to collect funding with {confidence*100:.1f}% confidence."
            )
        elif suggested_action == "SHORT":
            reasoning = (
                f"Funding rate predicted to stay positive ({predicted_rate*100:.4f}%). "
                f"Long positions will pay {predicted_rate*100:.4f}% to shorts. "
                f"Short position recommended to collect funding with {confidence*100:.1f}% confidence."
            )
        else:
            reasoning = (
                f"Funding rate prediction uncertain (confidence: {confidence*100:.1f}%). "
                f"Current rate: {current_rate*100:.4f}%. Waiting for clearer signal."
            )
        
        return FundingPrediction(
            direction=direction,
            probability=probability,
            current_rate=current_rate,
            predicted_rate=predicted_rate,
            suggested_action=suggested_action,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if not self.is_trained:
            return {}
        
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


# Global instance
funding_predictor = FundingRatePredictor()
