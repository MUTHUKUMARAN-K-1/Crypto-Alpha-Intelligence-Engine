"""
AI-powered explanation generator.
Produces human-readable explanations of market regime and risk assessments using OpenRouter LLM.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging
from openai import OpenAI
from app.config import settings

logger = logging.getLogger("crypto_regime")

@dataclass
class ExplanationContext:
    """Context data for generating explanations."""
    regime: str
    confidence: float
    volatility_metrics: Dict[str, Any]
    correlation_metrics: Dict[str, Any]
    liquidity_metrics: Dict[str, Any]
    feature_importances: Dict[str, float]


class ExplanationGenerator:
    """
    Generates human-readable explanations for market analysis using LLMs.
    Uses OpenRouter API for generation with a template-based fallback.
    """
    
    def __init__(self):
        """Initialize the explanation generator with OpenRouter client."""
        self.client = None
        if settings.openrouter_api_key:
            try:
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                )
                logger.info(f"OpenRouter LLM initialized with model: {settings.openrouter_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter client: {e}")
    
    def _create_prompt(self, context_data: Dict[str, Any], context_type: str) -> str:
        """Create a prompt for the LLM based on context."""
        
        system_instructions = """
        You are an expert crypto market analyst AI. Your task is to analyze the provided market metrics 
        and generate a concise, professional, and explainable rationale for the user.
        
        Guidelines:
        - Be direct and professional. Use financial terminology correctly but accessible.
        - Focus on the "WHY". Explain why the regime or score is what it is.
        - Mention specific metrics (e.g. volatility %, correlation) to back up your claims.
        - Keep it under 2-3 sentences.
        - Do not use markdown headers or lists. Just a paragraph.
        """
        
        full_prompt = f"{system_instructions}\n\nDATA CONTEXT ({context_type}):\n{json.dumps(context_data, indent=2)}\n\nGenerate Rationale:"
        return full_prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call OpenRouter LLM."""
        if not self.client:
            return None
            
        try:
            completion = self.client.chat.completions.create(
                model=settings.openrouter_model,
                messages=[
                    {"role": "system", "content": "You are a helpful crypto market analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return None

    def generate_regime_explanation(self, context: ExplanationContext) -> str:
        """Generate explanation using LLM or fallback."""
        
        # Prepare data for LLM
        data = {
            "current_regime": context.regime,
            "confidence": f"{context.confidence:.1%}",
            "volatility": context.volatility_metrics,
            "liquidity": context.liquidity_metrics
        }
        
        # 1. Try LLM
        prompt = self._create_prompt(data, "Market Regime Analysis")
        explanation = self._call_llm(prompt)
        
        if explanation:
            return explanation
            
        # 2. Fallback to simple template
        return f"Market is currently in {context.regime} regime (Confidence: {context.confidence:.0%}). automated analysis based on volatility and liquidity metrics."

    def generate_tradability_explanation(
        self,
        tradability_score: int,
        risk_level: str,
        regime: str,
        volatility_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any],
        top_factors: List[Dict[str, Any]]
    ) -> str:
        """Generate tradability explanation using LLM or fallback."""
        
        data = {
            "score": tradability_score,
            "risk_level": risk_level,
            "market_regime": regime,
            "key_factors": top_factors[:3] # Top 3 factors
        }
        
        prompt = self._create_prompt(data, "Tradability & Risk Scoring")
        explanation = self._call_llm(prompt)
        
        if explanation:
            return explanation
            
        return f"Tradability Score: {tradability_score}/100. Risk Level: {risk_level}. This score is based on {regime} market conditions and current volatility levels."

    def generate_position_rationale(self, data: Dict[str, Any]) -> str:
        """Generate rationale for position sizing decision."""
        
        prompt = self._create_prompt(data, "Position Sizing & Risk Management")
        explanation = self._call_llm(prompt)
        
        if explanation:
            return explanation
            
        # Fallback
        return (f"Recommended Action: {data.get('action', 'HOLD')}. "
                f"Position Multiplier: {data.get('multiplier', '1.0x')}. "
                f"Based on {data.get('regime', 'current')} regime and {data.get('risk_level', 'MEDIUM')} risk.")

    def generate_metrics_summary(
        self,
        volatility_metrics: Dict[str, Any],
        correlation_metrics: Dict[str, Any],
        liquidity_metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate summaries. Uses simple logic to avoid LLM latency for simple fields."""
        
        # Volatility summary
        vol = volatility_metrics.get("current_volatility", 0.5)
        vol_summary = f"{vol:.1%} Annualized"
        
        # Correlation summary
        avg_corr = correlation_metrics.get("average_correlation", 0.5)
        corr_summary = f"{avg_corr:.2f} Avg Correlation"
        
        # Liquidity summary
        liq_score = liquidity_metrics.get("liquidity_score", 50)
        liq_summary = f"Score: {liq_score:.0f}/100"
        
        return {
            "volatility": vol_summary,
            "correlation": corr_summary,
            "liquidity": liq_summary
        }


# Global explanation generator instance
explanation_generator = ExplanationGenerator()
