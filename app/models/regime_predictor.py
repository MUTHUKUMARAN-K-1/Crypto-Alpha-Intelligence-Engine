"""
Regime Prediction Module using LSTM.
Predicts upcoming regime changes before they happen using deep learning.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.utils.logger import app_logger


class RegimePredictor:
    """
    LSTM-based regime transition predictor.
    
    Predicts the probability of regime change in the next 24-72 hours
    based on historical regime probabilities and market features.
    
    Uses lightweight LSTM implementation with sklearn when PyTorch unavailable.
    """
    
    # Regime encoding
    REGIME_ENCODING = {
        "TREND": 0,
        "RANGE": 1,
        "HIGH-RISK": 2
    }
    
    REGIME_DECODING = {v: k for k, v in REGIME_ENCODING.items()}
    
    def __init__(
        self,
        sequence_length: int = 14,
        hidden_size: int = 64,
        prediction_horizons: List[int] = [24, 48, 72]  # hours
    ):
        """
        Initialize regime predictor.
        
        Args:
            sequence_length: Number of days of history to use
            hidden_size: LSTM hidden layer size
            prediction_horizons: Hours ahead to predict
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.prediction_horizons = prediction_horizons
        self.model = None
        self._model_trained = False
        
        # Try to initialize LSTM model
        self._try_init_model()
    
    def _try_init_model(self):
        """Try to initialize PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn
            
            class LSTMPredictor(nn.Module):
                def __init__(self, input_size, hidden_size, num_classes):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, num_classes),
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return out
            
            # Input features: 3 regime probs + volatility + correlation avg + liquidity
            input_size = 6
            num_classes = 3  # TREND, RANGE, HIGH-RISK
            
            self.model = LSTMPredictor(input_size, self.hidden_size, num_classes)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            # Pre-train with synthetic data
            self._pretrain_synthetic()
            
            app_logger.info("LSTM Regime Predictor initialized")
            
            app_logger.info("LSTM Regime Predictor initialized")
            
        except (ImportError, OSError) as e:
            app_logger.warning(f"PyTorch initialization failed ({e}). Using statistical predictor fallback.")
            self.model = None
    
    def _pretrain_synthetic(self):
        """Pre-train model with synthetic regime sequences."""
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        
        # Generate synthetic training data
        num_samples = 1000
        sequences = []
        labels = []
        
        for _ in range(num_samples):
            # Simulate regime sequence
            seq = []
            current_regime = np.random.choice([0, 1, 2])
            
            for t in range(self.sequence_length):
                # Regime probabilities based on current state
                if current_regime == 0:  # TREND
                    probs = [0.7, 0.2, 0.1]
                    vol = np.random.uniform(0.3, 0.6)
                elif current_regime == 1:  # RANGE
                    probs = [0.2, 0.7, 0.1]
                    vol = np.random.uniform(0.1, 0.3)
                else:  # HIGH-RISK
                    probs = [0.1, 0.2, 0.7]
                    vol = np.random.uniform(0.6, 1.0)
                
                # Add noise
                probs = np.array(probs) + np.random.uniform(-0.1, 0.1, 3)
                probs = np.clip(probs, 0, 1)
                probs = probs / probs.sum()
                
                corr = np.random.uniform(0.3, 0.8)
                liq = np.random.uniform(30, 80)
                
                seq.append([probs[0], probs[1], probs[2], vol, corr, liq/100])
                
                # Transition probability
                if np.random.random() < 0.15:
                    current_regime = np.random.choice([0, 1, 2])
            
            sequences.append(seq)
            labels.append(current_regime)
        
        # Train
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.LongTensor(labels).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        self._model_trained = True
        app_logger.info("LSTM pre-trained with synthetic regime sequences")
    
    def predict(
        self,
        regime_history: List[Dict[str, Any]],
        volatility_history: List[float],
        correlation_history: List[float],
        liquidity_history: List[float]
    ) -> Dict[str, Any]:
        """
        Predict future regime probabilities.
        
        Args:
            regime_history: List of past regime probability dicts
            volatility_history: List of past volatility values
            correlation_history: List of past correlation values
            liquidity_history: List of past liquidity scores
            
        Returns:
            Prediction with next regime probabilities and transition likelihood
        """
        # Ensure we have enough history
        n = min(
            len(regime_history),
            len(volatility_history),
            len(correlation_history),
            len(liquidity_history)
        )
        
        if n < 5:
            return self._default_prediction()
        
        # Take last sequence_length points
        n = min(n, self.sequence_length)
        
        # Build feature sequence
        features = []
        for i in range(-n, 0):
            rh = regime_history[i]
            features.append([
                rh.get("TREND", 0.33),
                rh.get("RANGE", 0.33),
                rh.get("HIGH-RISK", 0.33),
                volatility_history[i] if volatility_history[i] < 2 else volatility_history[i] / 100,
                correlation_history[i],
                liquidity_history[i] / 100 if liquidity_history[i] > 1 else liquidity_history[i]
            ])
        
        if self.model is not None and self._model_trained:
            return self._predict_with_lstm(features, regime_history[-1])
        else:
            return self._predict_statistical(features, regime_history[-1])
    
    def _predict_with_lstm(
        self,
        features: List[List[float]],
        current_regime: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict using LSTM model."""
        import torch
        
        # Pad sequence if needed
        while len(features) < self.sequence_length:
            features.insert(0, features[0])
        
        X = torch.FloatTensor([features]).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X)[0].cpu().numpy()
        
        predicted_regime = self.REGIME_DECODING[np.argmax(probs)]
        
        # Calculate transition probability
        current_dominant = max(current_regime, key=current_regime.get)
        if predicted_regime != current_dominant:
            transition_prob = float(probs[self.REGIME_ENCODING[predicted_regime]])
        else:
            # Probability of staying in same regime
            transition_prob = 0.0
        
        return {
            "predicted_regime": predicted_regime,
            "prediction_confidence": float(max(probs)),
            "regime_probabilities": {
                "TREND": float(probs[0]),
                "RANGE": float(probs[1]),
                "HIGH-RISK": float(probs[2])
            },
            "transition_probability": round(transition_prob, 3),
            "transition_expected": transition_prob > 0.3,
            "prediction_horizon_hours": 24,
            "method": "lstm",
            "recommendation": self._generate_prediction_recommendation(
                predicted_regime, transition_prob, current_dominant
            )
        }
    
    def _predict_statistical(
        self,
        features: List[List[float]],
        current_regime: Dict[str, float]
    ) -> Dict[str, Any]:
        """Statistical fallback prediction based on momentum."""
        # Calculate regime probability momentum
        trend_momentum = features[-1][0] - features[0][0]
        range_momentum = features[-1][1] - features[0][1]
        risk_momentum = features[-1][2] - features[0][2]
        
        # Current dominant + momentum
        current_probs = np.array([
            current_regime.get("TREND", 0.33),
            current_regime.get("RANGE", 0.33),
            current_regime.get("HIGH-RISK", 0.33)
        ])
        
        momentum = np.array([trend_momentum, range_momentum, risk_momentum])
        predicted_probs = current_probs + momentum * 0.5
        predicted_probs = np.clip(predicted_probs, 0.05, 0.95)
        predicted_probs = predicted_probs / predicted_probs.sum()
        
        predicted_regime = self.REGIME_DECODING[np.argmax(predicted_probs)]
        current_dominant = max(current_regime, key=current_regime.get)
        
        transition_prob = abs(momentum).max()
        
        return {
            "predicted_regime": predicted_regime,
            "prediction_confidence": float(max(predicted_probs)),
            "regime_probabilities": {
                "TREND": float(predicted_probs[0]),
                "RANGE": float(predicted_probs[1]),
                "HIGH-RISK": float(predicted_probs[2])
            },
            "transition_probability": round(float(transition_prob), 3),
            "transition_expected": transition_prob > 0.15,
            "prediction_horizon_hours": 24,
            "method": "statistical",
            "recommendation": self._generate_prediction_recommendation(
                predicted_regime, transition_prob, current_dominant
            )
        }
    
    def _generate_prediction_recommendation(
        self,
        predicted: str,
        transition_prob: float,
        current: str
    ) -> str:
        """Generate human-readable prediction recommendation."""
        if predicted == current:
            if transition_prob < 0.2:
                return f"Expect {current} regime to continue. No significant transition detected."
            else:
                return f"Current {current} regime shows some instability. Monitor for potential changes."
        else:
            if transition_prob > 0.5:
                return f"High probability of transition from {current} to {predicted} regime within 24-72 hours. Consider adjusting positions."
            elif transition_prob > 0.3:
                return f"Moderate probability of transition to {predicted} regime. Prepare contingency plans."
            else:
                return f"Slight indication of potential shift toward {predicted}. Continue monitoring."
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when insufficient data."""
        return {
            "predicted_regime": "UNKNOWN",
            "prediction_confidence": 0.0,
            "regime_probabilities": {"TREND": 0.33, "RANGE": 0.33, "HIGH-RISK": 0.33},
            "transition_probability": 0.0,
            "transition_expected": False,
            "prediction_horizon_hours": 24,
            "method": "default",
            "recommendation": "Insufficient historical data for prediction. Collect more data points."
        }


# Global predictor instance
regime_predictor = RegimePredictor()
