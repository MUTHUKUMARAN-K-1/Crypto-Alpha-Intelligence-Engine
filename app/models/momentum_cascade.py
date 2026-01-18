"""
Momentum Cascade Detector - Cross-asset momentum spillover prediction.
Detects when BTC momentum spills over to altcoins and predicts timing.

Key innovations:
1. Lead-lag relationship detection between BTC and altcoins
2. Momentum velocity scoring
3. Cascade timing prediction
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from sklearn.linear_model import LinearRegression
from scipy import stats

from app.utils.logger import app_logger


@dataclass
class CascadeSignal:
    """Cascade signal for an altcoin."""
    asset: str
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0-1 signal strength
    lag_minutes: float  # Expected lag before altcoin follows
    probability: float  # Probability of cascade occurring
    reasoning: str


@dataclass
class MomentumCascadePrediction:
    """Complete cascade prediction result."""
    btc_momentum: float
    btc_momentum_direction: str
    btc_move_significant: bool
    cascade_expected: bool
    cascade_signals: List[CascadeSignal]
    overall_market_bias: str
    confidence: float
    timing_recommendation: str


class MomentumCascadeDetector:
    """
    Detects momentum spillover from BTC to altcoins.
    
    Trading Logic:
    - When BTC makes a significant move, altcoins often follow
    - Different altcoins have different lag times
    - By detecting BTC momentum early, we can position in altcoins
      BEFORE they move
    
    Research shows:
    - SOL, ETH typically lag BTC by 5-15 minutes
    - DOGE, XRP can lag by 10-30 minutes
    - High beta coins amplify BTC moves
    """
    
    # Historical lag relationships (minutes)
    DEFAULT_LAGS = {
        "ethereum": 8,
        "solana": 10,
        "cardano": 15,
        "dogecoin": 12,
        "ripple": 18,
        "litecoin": 12,
        "binancecoin": 6
    }
    
    # Historical beta to BTC
    DEFAULT_BETAS = {
        "ethereum": 0.85,
        "solana": 1.4,
        "cardano": 1.2,
        "dogecoin": 1.6,
        "ripple": 1.1,
        "litecoin": 0.9,
        "binancecoin": 0.75
    }
    
    def __init__(self, lookback_periods: int = 50):
        """
        Initialize cascade detector.
        
        Args:
            lookback_periods: Number of periods for momentum calculation
        """
        self.lookback = lookback_periods
        self.price_history: Dict[str, deque] = {}
        self.momentum_threshold = 0.02  # 2% move is significant
        
        # Dynamically learned lags
        self.learned_lags: Dict[str, float] = {}
        self.learned_betas: Dict[str, float] = {}
    
    def update_prices(self, asset: str, price: float, timestamp: datetime = None):
        """
        Update price history for an asset.
        
        Args:
            asset: Asset identifier
            price: Current price
            timestamp: Price timestamp
        """
        if asset not in self.price_history:
            self.price_history[asset] = deque(maxlen=self.lookback)
        
        self.price_history[asset].append({
            "price": price,
            "timestamp": timestamp or datetime.now()
        })
    
    def calculate_momentum(self, prices: List[float], periods: int = 10) -> float:
        """
        Calculate price momentum.
        
        Args:
            prices: Price series
            periods: Lookback periods
            
        Returns:
            Momentum as percentage change
        """
        if len(prices) < periods:
            return 0.0
        
        start_price = prices[-periods]
        end_price = prices[-1]
        
        return (end_price - start_price) / (start_price + 1e-10)
    
    def calculate_momentum_velocity(self, prices: List[float], periods: int = 10) -> float:
        """
        Calculate momentum acceleration (velocity of price change).
        
        Args:
            prices: Price series
            periods: Lookback periods
            
        Returns:
            Momentum velocity
        """
        if len(prices) < periods + 1:
            return 0.0
        
        returns = np.diff(prices[-periods-1:]) / np.array(prices[-periods-1:-1])
        
        # Fit linear regression to returns
        if len(returns) > 1:
            X = np.arange(len(returns)).reshape(-1, 1)
            reg = LinearRegression().fit(X, returns)
            return reg.coef_[0]
        
        return 0.0
    
    def estimate_lead_lag(
        self,
        leader_returns: List[float],
        follower_returns: List[float],
        max_lag: int = 20
    ) -> Tuple[int, float]:
        """
        Estimate lead-lag relationship using cross-correlation.
        
        Args:
            leader_returns: Returns of leading asset (BTC)
            follower_returns: Returns of following asset
            max_lag: Maximum lag to test
            
        Returns:
            (optimal_lag, correlation_at_lag)
        """
        if len(leader_returns) < max_lag + 5 or len(follower_returns) < max_lag + 5:
            return 0, 0.0
        
        best_lag = 0
        best_corr = 0.0
        
        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr, _ = stats.pearsonr(leader_returns[:-1], follower_returns[1:])
            else:
                # Leader leads by 'lag' periods
                corr, _ = stats.pearsonr(
                    leader_returns[:-lag-1],
                    follower_returns[lag+1:]
                )
            
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        return best_lag, best_corr
    
    def predict_cascade(
        self,
        btc_prices: List[float],
        altcoin_prices: Dict[str, List[float]],
        momentum_periods: int = 10
    ) -> MomentumCascadePrediction:
        """
        Predict momentum cascade from BTC to altcoins.
        
        Args:
            btc_prices: BTC price history
            altcoin_prices: Dict of altcoin prices
            momentum_periods: Periods for momentum calculation
            
        Returns:
            MomentumCascadePrediction with signals for each altcoin
        """
        # Calculate BTC momentum
        btc_momentum = self.calculate_momentum(btc_prices, momentum_periods)
        btc_velocity = self.calculate_momentum_velocity(btc_prices, momentum_periods)
        
        # Determine BTC direction and significance
        if btc_momentum > self.momentum_threshold:
            btc_direction = "BULLISH"
            btc_significant = True
        elif btc_momentum < -self.momentum_threshold:
            btc_direction = "BEARISH"
            btc_significant = True
        else:
            btc_direction = "NEUTRAL"
            btc_significant = False
        
        # If BTC momentum is accelerating, move is more likely to cascade
        momentum_accelerating = btc_velocity > 0 if btc_momentum > 0 else btc_velocity < 0
        
        cascade_signals = []
        
        if btc_significant:
            # Calculate BTC returns for correlation
            btc_returns = list(np.diff(btc_prices[-self.lookback:]) / np.array(btc_prices[-self.lookback:-1]))
            
            for asset, prices in altcoin_prices.items():
                if len(prices) < momentum_periods:
                    continue
                
                # Calculate altcoin momentum
                alt_momentum = self.calculate_momentum(prices, momentum_periods)
                
                # Check if altcoin has already moved (no cascade opportunity)
                if (btc_direction == "BULLISH" and alt_momentum > self.momentum_threshold * 0.5) or \
                   (btc_direction == "BEARISH" and alt_momentum < -self.momentum_threshold * 0.5):
                    # Altcoin already following, reduced opportunity
                    signal_strength = 0.3
                else:
                    # Altcoin hasn't moved yet - cascade opportunity
                    signal_strength = 0.7 + (0.3 if momentum_accelerating else 0)
                
                # Get lag estimate
                lag_minutes = self.DEFAULT_LAGS.get(asset, 15)
                
                # Try to learn lag from data
                if len(prices) >= self.lookback:
                    alt_returns = list(np.diff(prices[-self.lookback:]) / np.array(prices[-self.lookback:-1]))
                    learned_lag, lag_corr = self.estimate_lead_lag(btc_returns, alt_returns)
                    if lag_corr > 0.3:
                        lag_minutes = learned_lag * 5  # Convert periods to approx minutes
                        self.learned_lags[asset] = lag_minutes
                
                # Get beta (expected amplification)
                beta = self.DEFAULT_BETAS.get(asset, 1.0)
                
                # Calculate probability of cascade
                base_prob = 0.6 if btc_significant else 0.3
                prob = min(0.95, base_prob + (abs(btc_momentum) * 5) + (0.1 if momentum_accelerating else 0))
                
                # Determine signal direction
                signal_direction = btc_direction
                
                # Generate reasoning
                if signal_direction == "BULLISH":
                    reasoning = (
                        f"BTC showing {btc_momentum*100:.2f}% bullish momentum. "
                        f"{asset.title()} typically follows with {lag_minutes:.0f} min lag and {beta:.1f}x beta. "
                        f"Expected move: +{abs(btc_momentum * beta * 100):.2f}%. "
                        f"{'Momentum accelerating - strong cascade likely.' if momentum_accelerating else ''}"
                    )
                elif signal_direction == "BEARISH":
                    reasoning = (
                        f"BTC showing {btc_momentum*100:.2f}% bearish momentum. "
                        f"{asset.title()} typically follows with {lag_minutes:.0f} min lag and {beta:.1f}x beta. "
                        f"Expected move: {btc_momentum * beta * 100:.2f}%. "
                        f"{'Momentum accelerating - strong cascade likely.' if momentum_accelerating else ''}"
                    )
                else:
                    reasoning = f"BTC momentum neutral. No cascade expected for {asset.title()}."
                
                cascade_signals.append(CascadeSignal(
                    asset=asset,
                    direction=signal_direction,
                    strength=signal_strength,
                    lag_minutes=lag_minutes,
                    probability=prob,
                    reasoning=reasoning
                ))
        
        # Sort signals by strength
        cascade_signals.sort(key=lambda x: x.strength, reverse=True)
        
        # Overall market bias
        if btc_significant:
            overall_bias = btc_direction
        else:
            overall_bias = "NEUTRAL"
        
        # Overall confidence
        confidence = abs(btc_momentum) / self.momentum_threshold if btc_significant else 0.0
        confidence = min(1.0, confidence)
        
        # Timing recommendation
        if btc_significant and cascade_signals:
            avg_lag = np.mean([s.lag_minutes for s in cascade_signals])
            timing = f"Enter altcoin positions within {avg_lag:.0f} minutes for optimal timing."
        else:
            timing = "Wait for significant BTC momentum before positioning."
        
        return MomentumCascadePrediction(
            btc_momentum=btc_momentum,
            btc_momentum_direction=btc_direction,
            btc_move_significant=btc_significant,
            cascade_expected=btc_significant and len(cascade_signals) > 0,
            cascade_signals=cascade_signals,
            overall_market_bias=overall_bias,
            confidence=confidence,
            timing_recommendation=timing
        )
    
    def get_best_cascade_opportunities(
        self,
        prediction: MomentumCascadePrediction,
        min_strength: float = 0.5,
        max_positions: int = 3
    ) -> List[CascadeSignal]:
        """
        Get best cascade trading opportunities.
        
        Args:
            prediction: Cascade prediction result
            min_strength: Minimum signal strength
            max_positions: Maximum positions to recommend
            
        Returns:
            Top cascade signals
        """
        qualified = [
            s for s in prediction.cascade_signals
            if s.strength >= min_strength and s.probability >= 0.5
        ]
        
        return qualified[:max_positions]


# Global instance
momentum_cascade = MomentumCascadeDetector()
