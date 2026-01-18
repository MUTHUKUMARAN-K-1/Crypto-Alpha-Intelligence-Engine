"""
Adaptive Position Sizing Module.
Calculates position sizes based on detected regime and risk metrics.
Generates explainable AI trading rationales.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class PositionAction(Enum):
    """Trading action recommendations."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    STRONG_REDUCE = "STRONG_REDUCE"


@dataclass
class PositionRecommendation:
    """Position sizing recommendation with rationale."""
    position_multiplier: float  # 0.0 to 2.0
    action: PositionAction
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade_pct: float
    rationale: str
    factors: List[Dict[str, Any]]


class AdaptivePositionSizer:
    """
    Calculates optimal position sizes based on market regime.
    
    Position multipliers:
    - TREND + Low Risk: 1.5x (capitalize on momentum)
    - TREND + Medium Risk: 1.0x (standard position)
    - RANGE + Low Risk: 0.8x (smaller, range-bound trades)
    - RANGE + Medium Risk: 0.6x (reduced exposure)
    - HIGH-RISK: 0.3x-0.5x (capital preservation)
    """
    
    # Regime-based position multipliers
    REGIME_MULTIPLIERS = {
        ("TREND", "LOW"): 1.5,
        ("TREND", "MEDIUM"): 1.0,
        ("TREND", "HIGH"): 0.6,
        ("RANGE", "LOW"): 0.8,
        ("RANGE", "MEDIUM"): 0.6,
        ("RANGE", "HIGH"): 0.4,
        ("HIGH-RISK", "LOW"): 0.5,
        ("HIGH-RISK", "MEDIUM"): 0.3,
        ("HIGH-RISK", "HIGH"): 0.2,
    }
    
    # Action thresholds
    ACTION_THRESHOLDS = {
        (1.3, 2.0): PositionAction.STRONG_BUY,
        (1.0, 1.3): PositionAction.BUY,
        (0.7, 1.0): PositionAction.HOLD,
        (0.4, 0.7): PositionAction.REDUCE,
        (0.0, 0.4): PositionAction.STRONG_REDUCE,
    }
    
    def __init__(
        self,
        base_risk_per_trade: float = 2.0,
        max_leverage: float = 20.0  # WEEX hackathon limit
    ):
        """
        Initialize position sizer.
        
        Args:
            base_risk_per_trade: Base risk percentage per trade
            max_leverage: Maximum allowed leverage (20x for WEEX)
        """
        self.base_risk = base_risk_per_trade
        self.max_leverage = max_leverage
    
    def calculate_position(
        self,
        regime: str,
        risk_level: str,
        tradability_score: float,
        volatility: float,
        confluence_score: float = 50.0,
        sentiment_score: float = 50.0
    ) -> PositionRecommendation:
        """
        Calculate optimal position size and parameters.
        
        Args:
            regime: Current market regime (TREND/RANGE/HIGH-RISK)
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            tradability_score: Tradability score (0-100)
            volatility: Current annualized volatility
            confluence_score: Multi-timeframe confluence (0-100)
            sentiment_score: Sentiment score (0-100)
            
        Returns:
            PositionRecommendation with full rationale
        """
        factors = []
        
        # Base multiplier from regime + risk
        key = (regime, risk_level)
        base_mult = self.REGIME_MULTIPLIERS.get(key, 0.5)
        factors.append({
            "factor": "Regime & Risk",
            "value": f"{regime} + {risk_level}",
            "impact": f"{base_mult}x base"
        })
        
        # Tradability adjustment (-20% to +20%)
        trad_adj = (tradability_score - 50) / 250  # -0.2 to +0.2
        factors.append({
            "factor": "Tradability Score",
            "value": f"{tradability_score:.0f}/100",
            "impact": f"{trad_adj:+.1%}"
        })
        
        # Confluence adjustment (-15% to +15%)
        conf_adj = (confluence_score - 50) / 333  # -0.15 to +0.15
        factors.append({
            "factor": "Timeframe Confluence",
            "value": f"{confluence_score:.0f}%",
            "impact": f"{conf_adj:+.1%}"
        })
        
        # Sentiment adjustment (-10% to +10%)
        sent_adj = (sentiment_score - 50) / 500  # -0.1 to +0.1
        factors.append({
            "factor": "Market Sentiment",
            "value": f"{sentiment_score:.0f}/100",
            "impact": f"{sent_adj:+.1%}"
        })
        
        # Final multiplier
        final_mult = base_mult * (1 + trad_adj + conf_adj + sent_adj)
        final_mult = max(0.1, min(2.0, final_mult))  # Clamp to 0.1-2.0
        
        # Determine action
        action = PositionAction.HOLD
        for (low, high), act in self.ACTION_THRESHOLDS.items():
            if low <= final_mult < high:
                action = act
                break
        
        # Calculate stop loss based on volatility
        # Higher volatility = wider stop loss
        base_stop = 2.0  # 2% base
        vol_adjustment = volatility * 100 * 0.5  # Add 0.5% per 1% volatility
        stop_loss = min(10.0, base_stop + vol_adjustment)
        
        # Take profit based on regime
        if regime == "TREND":
            take_profit = stop_loss * 3  # 3:1 RR in trends
        elif regime == "RANGE":
            take_profit = stop_loss * 1.5  # 1.5:1 RR in ranges
        else:
            take_profit = stop_loss * 2  # 2:1 RR in high-risk
        
        # Risk per trade adjusted by multiplier
        risk_per_trade = self.base_risk * final_mult
        
        # USE LLM FOR RATIONALE (via ExplanationGenerator)
        # We need to import inside method or at top if circular imports handled
        # For now, let's assume we can import the global instance
        from app.intelligence.explanation import explanation_generator

        # Create a simplified prompt for the LLM
        prompt_data = {
            "regime": regime,
            "risk_level": risk_level,
            "multiplier": f"{final_mult:.2f}x",
            "action": action.value,
            "metrics": {
                "tradability": f"{tradability_score:.0f}/100",
                "confluence": f"{confluence_score:.0f}%",
                "sentiment": f"{sentiment_score:.0f}/100",
                "volatility": f"{volatility:.1%} annualized"
            },
            "parameters": {
                "stop_loss": f"-{stop_loss:.1f}%",
                "take_profit": f"+{take_profit:.1f}%"
            }
        }
        
        # Use a new method on explanation_generator or reuse existing generic caller
        # We'll add a helper method to ExplanationGenerator OR just call the generic LLM method if we exposed it.
        # Since _call_llm is internal, let's try to add a specific method to ExplanationGenerator first.
        # But to avoid touching two files in one step if possible, I will perform a direct call here using the public methods if suitable,
        # OR better: I'll accept that PositionSizer just constructs the data and passes it to a new method I'll add to ExplanationGenerator later.
        # Actually, let's keep it simple: PositionSizer builds a string for now, because I haven't added `generate_position_rationale` to `ExplanationGenerator` yet.
        # WAIT: I can just invoke the LLM directly here if I instantiate the client, but that duplicates code.
        # Best approach: Add `generate_position_rationale` to `ExplanationGenerator` in next step, then call it here.
        # For this step, I will replace the logic to PREPARE for that call.
        
        # ... actually, looking at the plan, I should have updated ExplanationGenerator first to support this.
        # I'll update PositionSizer to call a new method `generate_position_rationale` which I WILL add to ExplanationGenerator.
        
        rationale = explanation_generator.generate_position_rationale(prompt_data)
        
        return PositionRecommendation(
            position_multiplier=round(final_mult, 2),
            action=action,
            stop_loss_pct=round(stop_loss, 2),
            take_profit_pct=round(take_profit, 2),
            risk_per_trade_pct=round(risk_per_trade, 2),
            rationale=rationale,
            factors=factors
        )

    # Removed _generate_rationale method as it is no longer used internally

    
    def to_dict(self, rec: PositionRecommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary for API response."""
        return {
            "position_multiplier": rec.position_multiplier,
            "action": rec.action.value,
            "risk_parameters": {
                "stop_loss_pct": rec.stop_loss_pct,
                "take_profit_pct": rec.take_profit_pct,
                "risk_per_trade_pct": rec.risk_per_trade_pct,
                "risk_reward_ratio": round(rec.take_profit_pct / rec.stop_loss_pct, 2) if rec.stop_loss_pct > 0 else 0
            },
            "rationale": rec.rationale,
            "factors": rec.factors
        }


# Global position sizer instance
position_sizer = AdaptivePositionSizer()
