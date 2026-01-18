"""
Strategy Selector - Regime-adaptive strategy switching.
Automatically selects optimal trading strategy based on market regime.

Key innovations:
1. Dynamic strategy switching based on regime
2. Multi-model signal aggregation
3. Risk-adjusted position sizing
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.utils.logger import app_logger


class Strategy(Enum):
    """Available trading strategies."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM_CASCADE = "momentum_cascade"
    FUNDING_ARBITRAGE = "funding_arbitrage"
    DEFENSIVE = "defensive"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: Strategy
    max_leverage: int
    position_size_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    entry_threshold: float  # Minimum signal confidence to enter
    suitable_regimes: List[str]
    description: str


@dataclass
class StrategyRecommendation:
    """Strategy recommendation result."""
    primary_strategy: Strategy
    secondary_strategy: Optional[Strategy]
    config: StrategyConfig
    reasoning: str
    confidence: float
    signals_to_use: List[str]
    risk_adjustment: float  # 0.5-1.5 multiplier


class StrategySelector:
    """
    Selects optimal trading strategy based on market conditions.
    
    Strategy Selection Logic:
    - TREND regime → Trend Following with Momentum Cascade
    - RANGE regime → Mean Reversion with Orderbook signals
    - HIGH-RISK regime → Defensive mode with Funding Arbitrage only
    """
    
    def __init__(self):
        """Initialize strategy selector."""
        self.strategies = self._define_strategies()
        self.current_strategy = Strategy.DEFENSIVE
        self.strategy_history: List[Tuple[datetime, Strategy]] = []
    
    def _define_strategies(self) -> Dict[Strategy, StrategyConfig]:
        """Define all available strategies."""
        return {
            Strategy.TREND_FOLLOWING: StrategyConfig(
                name=Strategy.TREND_FOLLOWING,
                max_leverage=10,
                position_size_multiplier=1.0,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                entry_threshold=0.6,
                suitable_regimes=["TREND"],
                description="Follow strong directional moves with momentum confirmation"
            ),
            
            Strategy.MEAN_REVERSION: StrategyConfig(
                name=Strategy.MEAN_REVERSION,
                max_leverage=5,
                position_size_multiplier=0.7,
                stop_loss_pct=1.5,
                take_profit_pct=2.0,
                entry_threshold=0.65,
                suitable_regimes=["RANGE"],
                description="Trade reversals at range boundaries with tight stops"
            ),
            
            Strategy.MOMENTUM_CASCADE: StrategyConfig(
                name=Strategy.MOMENTUM_CASCADE,
                max_leverage=8,
                position_size_multiplier=0.8,
                stop_loss_pct=2.5,
                take_profit_pct=5.0,
                entry_threshold=0.55,
                suitable_regimes=["TREND"],
                description="Ride momentum spillover from BTC to altcoins"
            ),
            
            Strategy.FUNDING_ARBITRAGE: StrategyConfig(
                name=Strategy.FUNDING_ARBITRAGE,
                max_leverage=3,
                position_size_multiplier=1.2,
                stop_loss_pct=3.0,
                take_profit_pct=1.5,
                entry_threshold=0.5,
                suitable_regimes=["TREND", "RANGE", "HIGH-RISK"],
                description="Collect funding rate premium with low-risk carry trades"
            ),
            
            Strategy.DEFENSIVE: StrategyConfig(
                name=Strategy.DEFENSIVE,
                max_leverage=2,
                position_size_multiplier=0.3,
                stop_loss_pct=1.0,
                take_profit_pct=1.5,
                entry_threshold=0.8,
                suitable_regimes=["HIGH-RISK"],
                description="Minimal exposure, capital preservation focus"
            )
        }
    
    def select_strategy(
        self,
        regime: str,
        regime_confidence: float,
        tradability_score: float,
        volatility_level: str = "MODERATE",
        funding_signal: Optional[str] = None,
        orderbook_signal: Optional[str] = None,
        cascade_signal: Optional[str] = None
    ) -> StrategyRecommendation:
        """
        Select optimal strategy based on market conditions.
        
        Args:
            regime: Market regime (TREND, RANGE, HIGH-RISK)
            regime_confidence: Confidence in regime detection
            tradability_score: Tradability score (0-100)
            volatility_level: Current volatility (LOW, MODERATE, HIGH, EXTREME)
            funding_signal: Funding rate predictor signal
            orderbook_signal: Orderbook predictor signal
            cascade_signal: Momentum cascade signal
            
        Returns:
            StrategyRecommendation with selected strategy and config
        """
        # Default to defensive
        primary = Strategy.DEFENSIVE
        secondary = None
        signals_to_use = []
        reasoning_parts = []
        
        # Calculate risk adjustment based on conditions
        risk_adjustment = 1.0
        
        if volatility_level == "EXTREME":
            risk_adjustment *= 0.5
            reasoning_parts.append("Extreme volatility - reducing exposure")
        elif volatility_level == "HIGH":
            risk_adjustment *= 0.7
            reasoning_parts.append("High volatility detected")
        elif volatility_level == "LOW":
            risk_adjustment *= 1.2
            reasoning_parts.append("Low volatility - can increase exposure")
        
        if tradability_score < 40:
            risk_adjustment *= 0.6
            reasoning_parts.append("Low tradability - caution advised")
        elif tradability_score > 70:
            risk_adjustment *= 1.1
            reasoning_parts.append("High tradability conditions")
        
        if regime_confidence < 0.5:
            risk_adjustment *= 0.7
            reasoning_parts.append("Low regime confidence")
        
        # Strategy selection based on regime
        if regime == "TREND":
            if tradability_score >= 50:
                primary = Strategy.TREND_FOLLOWING
                signals_to_use = ["regime", "momentum_cascade", "orderbook"]
                reasoning_parts.append(
                    "TREND regime detected - using trend following strategy"
                )
                
                # Check for cascade opportunities
                if cascade_signal in ["BULLISH", "BEARISH"]:
                    secondary = Strategy.MOMENTUM_CASCADE
                    signals_to_use.append("cascade")
                    reasoning_parts.append("Cascade signal active - adding momentum cascade")
                    
            else:
                # Low tradability in trend - use funding arbitrage
                primary = Strategy.FUNDING_ARBITRAGE
                signals_to_use = ["funding", "regime"]
                reasoning_parts.append(
                    "TREND but low tradability - using funding arbitrage only"
                )
                
        elif regime == "RANGE":
            if tradability_score >= 50:
                primary = Strategy.MEAN_REVERSION
                signals_to_use = ["orderbook", "regime"]
                reasoning_parts.append(
                    "RANGE regime detected - using mean reversion strategy"
                )
                
                # Strong orderbook signals are key for mean reversion
                if orderbook_signal in ["UP", "DOWN"]:
                    signals_to_use.append("orderbook_primary")
                    reasoning_parts.append("Using orderbook imbalance for entries")
                    
                # Add funding as secondary income
                secondary = Strategy.FUNDING_ARBITRAGE
                    
            else:
                primary = Strategy.DEFENSIVE
                secondary = Strategy.FUNDING_ARBITRAGE
                signals_to_use = ["funding"]
                reasoning_parts.append(
                    "RANGE with low tradability - defensive with funding only"
                )
                
        else:  # HIGH-RISK
            primary = Strategy.DEFENSIVE
            risk_adjustment *= 0.5
            reasoning_parts.append(
                "HIGH-RISK regime - switching to defensive mode"
            )
            
            # Only funding arbitrage is allowed in high-risk
            if funding_signal in ["LONG", "SHORT"]:
                secondary = Strategy.FUNDING_ARBITRAGE
                signals_to_use = ["funding"]
                reasoning_parts.append("Using funding arbitrage for low-risk income")
            else:
                signals_to_use = []
                reasoning_parts.append("Waiting for clearer signals")
        
        # Get strategy config
        config = self.strategies[primary]
        
        # Calculate confidence
        base_confidence = regime_confidence
        signal_count = len([s for s in [funding_signal, orderbook_signal, cascade_signal] if s])
        confidence = min(0.95, base_confidence + (signal_count * 0.1))
        
        # Update strategy history
        if primary != self.current_strategy:
            self.strategy_history.append((datetime.now(), primary))
            self.current_strategy = primary
            app_logger.info(f"Strategy switched to {primary.value}")
        
        return StrategyRecommendation(
            primary_strategy=primary,
            secondary_strategy=secondary,
            config=config,
            reasoning=" | ".join(reasoning_parts),
            confidence=confidence,
            signals_to_use=signals_to_use,
            risk_adjustment=min(1.5, max(0.3, risk_adjustment))
        )
    
    def get_strategy_config(self, strategy: Strategy) -> StrategyConfig:
        """Get configuration for a specific strategy."""
        return self.strategies[strategy]
    
    def get_effective_leverage(
        self,
        recommendation: StrategyRecommendation,
        max_allowed: int = 20
    ) -> int:
        """
        Calculate effective leverage for a trade.
        
        Args:
            recommendation: Strategy recommendation
            max_allowed: Maximum allowed leverage (hackathon cap: 20)
            
        Returns:
            Effective leverage to use
        """
        base_leverage = recommendation.config.max_leverage
        adjusted = int(base_leverage * recommendation.risk_adjustment)
        return min(max_allowed, max(1, adjusted))
    
    def get_position_size_factor(
        self,
        recommendation: StrategyRecommendation,
        signal_confidence: float
    ) -> float:
        """
        Calculate position size multiplier.
        
        Args:
            recommendation: Strategy recommendation
            signal_confidence: Confidence of the trading signal
            
        Returns:
            Position size multiplier (0.1 - 2.0)
        """
        base = recommendation.config.position_size_multiplier
        
        # Adjust by signal confidence
        if signal_confidence < 0.5:
            confidence_mult = 0.5
        elif signal_confidence < 0.7:
            confidence_mult = 0.8
        else:
            confidence_mult = 1.0 + (signal_confidence - 0.7)
        
        # Apply risk adjustment
        factor = base * confidence_mult * recommendation.risk_adjustment
        
        return min(2.0, max(0.1, factor))
    
    def should_enter_trade(
        self,
        recommendation: StrategyRecommendation,
        signal_confidence: float
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be entered.
        
        Args:
            recommendation: Strategy recommendation
            signal_confidence: Confidence of the trading signal
            
        Returns:
            (should_enter, reason)
        """
        threshold = recommendation.config.entry_threshold
        
        if signal_confidence < threshold:
            return False, f"Signal confidence {signal_confidence:.2f} below threshold {threshold:.2f}"
        
        if recommendation.primary_strategy == Strategy.DEFENSIVE:
            return False, "Defensive mode - no new entries"
        
        if recommendation.risk_adjustment < 0.4:
            return False, f"Risk adjustment too low ({recommendation.risk_adjustment:.2f})"
        
        return True, f"Entry conditions met: confidence={signal_confidence:.2f}, strategy={recommendation.primary_strategy.value}"


# Global instance
strategy_selector = StrategySelector()
