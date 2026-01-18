"""
Sentiment Analysis Module using CryptoBERT.
Analyzes crypto-related text for market sentiment using HuggingFace transformers.
"""

from typing import Dict, Any, List, Optional
import asyncio
from functools import lru_cache

from app.utils.logger import app_logger


class SentimentAnalyzer:
    """
    NLP-based sentiment analyzer for cryptocurrency content.
    
    Uses CryptoBERT (ElKulako/cryptobert) - a model trained on 3.2M+ crypto
    social media posts for accurate crypto-specific sentiment detection.
    
    Fallback: Uses rule-based sentiment when transformers unavailable.
    """
    
    # Keywords for rule-based fallback
    BULLISH_KEYWORDS = [
        "bullish", "moon", "pump", "buy", "long", "breakout", "ath", "all-time high",
        "accumulate", "hodl", "rocket", "gains", "rally", "surge", "soaring",
        "institutional", "adoption", "milestone", "partnership", "launch"
    ]
    
    BEARISH_KEYWORDS = [
        "bearish", "dump", "crash", "sell", "short", "breakdown", "fear",
        "panic", "rug", "scam", "hack", "exploit", "regulation", "ban",
        "fud", "correction", "decline", "plunge", "loss", "warning"
    ]
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_transformers: Whether to use HuggingFace transformers
        """
        self.use_transformers = use_transformers
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        if use_transformers:
            self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load CryptoBERT model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            app_logger.info("Loading CryptoBERT sentiment model...")
            
            # Use the fine-tuned crypto sentiment model
            model_name = "ElKulako/cryptobert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            # Check if CUDA available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self._model_loaded = True
            app_logger.info(f"CryptoBERT loaded successfully on {self.device}")
            
        except ImportError:
            app_logger.warning("transformers not installed. Using rule-based sentiment.")
            self._model_loaded = False
        except Exception as e:
            app_logger.warning(f"Failed to load CryptoBERT: {e}. Using rule-based sentiment.")
            self._model_loaded = False
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores and label
        """
        if not text or len(text.strip()) < 3:
            return self._neutral_response()
        
        if self._model_loaded:
            return self._analyze_with_transformer(text)
        else:
            return self._analyze_with_rules(text)
    
    def _analyze_with_transformer(self, text: str) -> Dict[str, Any]:
        """Analyze using CryptoBERT transformer model."""
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
            
            # CryptoBERT has 3 classes: negative, neutral, positive
            scores = {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
            
            # Determine label
            max_label = max(scores, key=scores.get)
            
            # Convert to unified score (0-100)
            # positive = 100, neutral = 50, negative = 0
            sentiment_score = (
                scores["positive"] * 100 +
                scores["neutral"] * 50 +
                scores["negative"] * 0
            )
            
            return {
                "label": max_label.upper(),
                "scores": scores,
                "sentiment_score": round(sentiment_score, 1),
                "confidence": round(max(scores.values()), 3),
                "method": "cryptobert"
            }
            
        except Exception as e:
            app_logger.error(f"Transformer analysis failed: {e}")
            return self._analyze_with_rules(text)
    
    def _analyze_with_rules(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based sentiment analysis."""
        text_lower = text.lower()
        
        # Count keyword matches
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        
        if total == 0:
            return self._neutral_response()
        
        # Calculate scores
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        
        if bullish_ratio > 0.6:
            label = "POSITIVE"
            sentiment_score = 50 + (bullish_ratio * 50)
        elif bearish_ratio > 0.6:
            label = "NEGATIVE"
            sentiment_score = 50 - (bearish_ratio * 50)
        else:
            label = "NEUTRAL"
            sentiment_score = 50
        
        return {
            "label": label,
            "scores": {
                "positive": bullish_ratio,
                "neutral": 1 - max(bullish_ratio, bearish_ratio),
                "negative": bearish_ratio
            },
            "sentiment_score": round(sentiment_score, 1),
            "confidence": round(max(bullish_ratio, bearish_ratio), 3),
            "method": "rule_based"
        }
    
    def _neutral_response(self) -> Dict[str, Any]:
        """Return neutral sentiment response."""
        return {
            "label": "NEUTRAL",
            "scores": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            "sentiment_score": 50.0,
            "confidence": 0.34,
            "method": "default"
        }
    
    def analyze_multiple(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple texts and aggregate sentiment.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Aggregated sentiment analysis
        """
        if not texts:
            return {
                "aggregate_score": 50.0,
                "aggregate_label": "NEUTRAL",
                "total_analyzed": 0,
                "distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
            }
        
        results = [self.analyze_text(t) for t in texts]
        
        # Aggregate scores
        total_score = sum(r["sentiment_score"] for r in results)
        avg_score = total_score / len(results)
        
        # Count labels
        distribution = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        for r in results:
            label = r["label"]
            if label in distribution:
                distribution[label] += 1
        
        # Determine aggregate label
        if avg_score >= 60:
            agg_label = "POSITIVE"
        elif avg_score <= 40:
            agg_label = "NEGATIVE"
        else:
            agg_label = "NEUTRAL"
        
        return {
            "aggregate_score": round(avg_score, 1),
            "aggregate_label": agg_label,
            "total_analyzed": len(results),
            "distribution": distribution,
            "individual_results": results[:5]  # Return first 5 for reference
        }
    
    def get_sentiment_regime_adjustment(
        self,
        sentiment_score: float
    ) -> Dict[str, float]:
        """
        Calculate regime probability adjustments based on sentiment.
        
        Args:
            sentiment_score: Aggregate sentiment score (0-100)
            
        Returns:
            Dict of regime probability adjustments
        """
        # Sentiment affects regime probabilities
        # Positive sentiment boosts TREND, reduces HIGH-RISK
        # Negative sentiment boosts HIGH-RISK, reduces TREND
        
        normalized = (sentiment_score - 50) / 50  # -1 to +1
        
        return {
            "TREND": normalized * 0.15,      # +/- 15%
            "RANGE": -abs(normalized) * 0.05,  # Always slightly reduce
            "HIGH-RISK": -normalized * 0.15   # Inverse of TREND
        }


# Global sentiment analyzer (lazy loading)
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer(use_transformers=True)
    return _sentiment_analyzer
