"""
LLM Explainer - Generate human-readable trade explanations using LLM.
Produces compliant AI logs for WEEX hackathon requirements.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

import httpx

from app.config import settings
from app.utils.logger import app_logger


@dataclass
class TradeExplanation:
    """Structured trade explanation for AI logs."""
    summary: str
    reasoning: str
    factors: List[str]
    risk_assessment: str
    confidence_justification: str
    timestamp: str


class LLMExplainer:
    """
    Generates human-readable explanations for AI trading decisions.
    Uses OpenRouter/DeepSeek (free tier) or falls back to template-based.
    """
    
    SYSTEM_PROMPT = """You are an AI trading assistant explaining trading decisions for a crypto algorithmic trading system.

Your explanations must be:
1. Clear and professional
2. Mention specific data points (prices, percentages, indicators)
3. Justify why the AI made this decision
4. Acknowledge risks

Format: Provide a 2-3 sentence summary followed by key factors.
Do NOT use markdown formatting. Keep it plain text."""

    # Fallback models in priority order (all free tier)
    FALLBACK_MODELS = [
        "deepseek/deepseek-r1-distill-llama-70b:free",
        "google/gemma-2-9b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "qwen/qwen-2-7b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ]

    def __init__(self):
        """Initialize the LLM explainer."""
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.base_url = "https://openrouter.ai/api/v1"
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def _call_llm_with_model(self, prompt: str, model: str) -> Optional[str]:
        """
        Call LLM API with a specific model.
        
        Args:
            prompt: The prompt to send
            model: Model to use
            
        Returns:
            Generated text or None on failure
        """
        if not self.api_key:
            return None
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/MUTHUKUMARAN-K-1/Crypto-Alpha-Intelligence-Engine",
                    "X-Title": "Crypto Alpha Intelligence"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            app_logger.debug(f"LLM call failed with {model}: {e}")
            return None
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API with fallback models.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text or None if all models fail
        """
        if not self.api_key:
            app_logger.info("No OpenRouter API key - using template fallback")
            return None
        
        # Try primary model first
        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        
        for model in models_to_try:
            result = await self._call_llm_with_model(prompt, model)
            if result:
                app_logger.debug(f"LLM response from {model}")
                return result
            await asyncio.sleep(0.5)  # Brief delay between retries
        
        app_logger.warning("All LLM models failed, using template fallback")
        return None
    
    def _generate_template_explanation(
        self,
        signal: str,
        asset: str,
        regime: str,
        regime_confidence: float,
        tradability_score: float,
        factors: Dict[str, Any]
    ) -> str:
        """Generate explanation using templates (fallback)."""
        
        # Build explanation based on signal and factors
        if signal == "BUY" or signal == "LONG":
            action = "opening a long position"
            direction = "bullish"
        elif signal == "SELL" or signal == "SHORT":
            action = "opening a short position"
            direction = "bearish"
        else:
            action = "holding current position"
            direction = "neutral"
        
        parts = [
            f"AI analysis recommends {action} on {asset.upper()}. "
        ]
        
        # Add regime context
        if regime == "TREND":
            parts.append(
                f"Market is in TREND regime ({regime_confidence*100:.0f}% confidence), "
                f"supporting directional trades. "
            )
        elif regime == "RANGE":
            parts.append(
                f"Market is in RANGE regime ({regime_confidence*100:.0f}% confidence), "
                f"favoring mean reversion strategies. "
            )
        else:
            parts.append(
                f"Market is in HIGH-RISK regime ({regime_confidence*100:.0f}% confidence), "
                f"suggesting defensive positioning. "
            )
        
        # Add tradability context
        if tradability_score >= 70:
            parts.append(f"Tradability score of {tradability_score:.0f} indicates favorable conditions. ")
        elif tradability_score >= 50:
            parts.append(f"Moderate tradability ({tradability_score:.0f}) suggests caution on position size. ")
        else:
            parts.append(f"Low tradability ({tradability_score:.0f}) warrants reduced exposure. ")
        
        # Add specific factors
        factor_parts = []
        
        if factors.get("funding_rate"):
            fr = factors["funding_rate"]
            if fr < 0:
                factor_parts.append(f"negative funding rate ({fr*100:.4f}%) favors longs")
            else:
                factor_parts.append(f"positive funding rate ({fr*100:.4f}%) favors shorts")
        
        if factors.get("orderbook_imbalance"):
            imb = factors["orderbook_imbalance"]
            if imb > 1.2:
                factor_parts.append(f"bid-heavy orderbook ({imb:.2f}x)")
            elif imb < 0.8:
                factor_parts.append(f"ask-heavy orderbook ({imb:.2f}x)")
        
        if factors.get("momentum"):
            mom = factors["momentum"]
            factor_parts.append(f"{abs(mom)*100:.1f}% momentum {'upward' if mom > 0 else 'downward'}")
        
        if factor_parts:
            parts.append("Key factors: " + ", ".join(factor_parts) + ".")
        
        return "".join(parts)
    
    async def generate_explanation(
        self,
        signal: str,
        asset: str,
        regime: str,
        regime_confidence: float,
        tradability_score: float,
        strategy: str,
        factors: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> TradeExplanation:
        """
        Generate comprehensive trade explanation.
        
        Args:
            signal: Trade signal (BUY/SELL/HOLD)
            asset: Asset being traded
            regime: Market regime
            regime_confidence: Regime confidence
            tradability_score: Tradability score
            strategy: Selected strategy name
            factors: Dict of analysis factors
            market_data: Current market data
            
        Returns:
            TradeExplanation with all components
        """
        # Build prompt for LLM
        prompt = f"""Generate a trading explanation for the following AI decision:

DECISION: {signal} {asset.upper()}
STRATEGY: {strategy}

MARKET CONDITIONS:
- Regime: {regime} (confidence: {regime_confidence*100:.0f}%)
- Tradability Score: {tradability_score:.0f}/100
- Current Price: ${market_data.get('price', 'N/A') if market_data else 'N/A'}

KEY FACTORS:
{json.dumps(factors, indent=2)}

Explain why this decision was made, mention specific data points, and note any risks."""

        # Try LLM first
        llm_explanation = await self._call_llm(prompt)
        
        if llm_explanation:
            summary = llm_explanation
        else:
            # Fallback to template
            summary = self._generate_template_explanation(
                signal, asset, regime, regime_confidence,
                tradability_score, factors
            )
        
        # Build factor list
        factor_list = []
        if factors.get("regime_signal"):
            factor_list.append(f"Regime signal: {factors['regime_signal']}")
        if factors.get("funding_prediction"):
            factor_list.append(f"Funding prediction: {factors['funding_prediction']}")
        if factors.get("orderbook_signal"):
            factor_list.append(f"Orderbook signal: {factors['orderbook_signal']}")
        if factors.get("cascade_signal"):
            factor_list.append(f"Momentum cascade: {factors['cascade_signal']}")
        
        # Risk assessment
        if tradability_score >= 70 and regime_confidence >= 0.7:
            risk = "Low risk - favorable conditions with high confidence"
        elif tradability_score >= 50 and regime_confidence >= 0.5:
            risk = "Medium risk - moderate conditions, standard position sizing"
        else:
            risk = "High risk - unfavorable conditions, reduced exposure recommended"
        
        # Confidence justification
        avg_confidence = (regime_confidence + tradability_score/100) / 2
        if avg_confidence >= 0.7:
            confidence_just = f"High confidence ({avg_confidence*100:.0f}%) based on strong regime signal and favorable tradability"
        elif avg_confidence >= 0.5:
            confidence_just = f"Moderate confidence ({avg_confidence*100:.0f}%) with some uncertainty in market conditions"
        else:
            confidence_just = f"Low confidence ({avg_confidence*100:.0f}%) - recommend smaller position or waiting"
        
        return TradeExplanation(
            summary=summary,
            reasoning=summary,  # Same as summary for simplicity
            factors=factor_list,
            risk_assessment=risk,
            confidence_justification=confidence_just,
            timestamp=datetime.now().isoformat()
        )
    
    def format_for_ai_log(self, explanation: TradeExplanation) -> str:
        """
        Format explanation for WEEX AI log upload.
        
        Args:
            explanation: TradeExplanation object
            
        Returns:
            Formatted string for AI log
        """
        parts = [
            explanation.summary,
            "",
            f"Risk Assessment: {explanation.risk_assessment}",
            f"Confidence: {explanation.confidence_justification}",
        ]
        
        if explanation.factors:
            parts.append("")
            parts.append("Key Factors:")
            for factor in explanation.factors:
                parts.append(f"- {factor}")
        
        return "\n".join(parts)
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
llm_explainer = LLMExplainer()
