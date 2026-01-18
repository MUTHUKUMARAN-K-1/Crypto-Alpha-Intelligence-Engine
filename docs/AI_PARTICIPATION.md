# AI Participation Description

## How AI/ML Powers the Crypto Alpha Trading System

### Executive Summary

Crypto Alpha is an **AI-first trading system** where every trading decision is made by machine learning models. There is no manual intervention - the AI autonomously analyzes markets, generates signals, executes trades, and explains its reasoning.

---

## AI/ML Models Implemented

### 1. Regime Classification Model
**Algorithm**: Random Forest Classifier (100 trees)

**Purpose**: Detect current market state (TREND, RANGE, or HIGH-RISK)

**Features Used**:
- Rolling volatility (14-day ATR)
- Volatility trend (increasing/decreasing)
- Price momentum (RSI-like)
- Spike rate (abnormal price moves)
- Cross-asset correlation matrix
- Liquidity metrics (volume/market cap stability)

**Training**: Synthetic data with 3000 samples covering different market conditions

**File**: `app/models/regime_model.py`

---

### 2. Funding Rate Predictor
**Algorithm**: Gradient Boosting Classifier

**Purpose**: Predict funding rate direction for perpetual futures carry trades

**Features Used**:
- 8-hour and 24-hour funding rate moving averages
- Funding rate momentum and volatility
- Open interest changes
- Volume momentum
- Price momentum correlation
- Long/short ratio estimation

**Innovation**: Predicts whether funding will be positive or negative, allowing the bot to take positions that COLLECT funding payments rather than pay them.

**File**: `app/models/funding_predictor.py`

---

### 3. Orderbook CNN (Deep Learning on LOB)
**Algorithm**: Random Forest on Limit Order Book Features

**Purpose**: Predict price direction for 15s, 60s, and 5-minute horizons

**Features Used**:
- Bid/ask imbalance ratio (top 10 levels)
- Distance-weighted imbalance
- Total bid/ask depth
- Spread in basis points
- Order concentration at best levels
- Volume imbalance from recent trades
- Depth gradient (how fast orders accumulate)

**Innovation**: Multi-horizon prediction allows strategy switching based on timeframe

**File**: `app/models/orderbook_cnn.py`

---

### 4. Momentum Cascade Detector
**Algorithm**: Statistical Lead-Lag Analysis with Cross-Correlation

**Purpose**: Detect when BTC momentum will spill over to altcoins

**Features Used**:
- BTC price momentum and velocity
- Historical lead-lag relationships per asset
- Cross-asset correlation in real-time
- Beta coefficients (amplification factors)

**Innovation**: Predicts WHEN altcoins will follow BTC moves, allowing entry BEFORE the move

**File**: `app/models/momentum_cascade.py`

---

### 5. Strategy Selector (Regime-Adaptive)
**Algorithm**: Rule-based with ML-informed parameters

**Purpose**: Dynamically switch trading strategies based on market conditions

**Logic**:
```
if regime == TREND:
    use Trend Following + Momentum Cascade
elif regime == RANGE:
    use Mean Reversion + Orderbook Signals
else (HIGH-RISK):
    use Defensive Mode + Funding Arbitrage Only
```

**File**: `app/intelligence/strategy_selector.py`

---

### 6. LLM Trade Explainer
**Technology**: OpenRouter API with 5 fallback models

**Purpose**: Generate human-readable explanations for AI log compliance

**Models Used** (fallback order):
1. DeepSeek R1 Distill Llama 70B
2. Google Gemma-2 9B
3. Meta Llama 3.2 3B
4. Qwen-2 7B
5. Mistral 7B

**File**: `app/intelligence/llm_explainer.py`

---

## AI Decision Flow

```
┌────────────────────────────────────────────────────────────────┐
│                     MARKET DATA INGESTION                       │
│  WEEX API: Orderbook, Funding Rates, Prices                    │
│  CoinGecko: Historical Data, Market Metrics                    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     PARALLEL AI ANALYSIS                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Regime   │ │ Funding  │ │ Orderbook│ │ Cascade  │          │
│  │ Detector │ │ Predictor│ │ CNN      │ │ Detector │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                        │
│                         ▼                                        │
│              ┌────────────────────┐                             │
│              │ Signal Aggregator  │                             │
│              │ (Weighted Ensemble)│                             │
│              └────────────────────┘                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     STRATEGY SELECTION                          │
│  Input: Regime, Signals, Risk Parameters                       │
│  Output: Strategy + Position Size + Risk Levels                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     TRADE EXECUTION                             │
│  1. Calculate position size and leverage                       │
│  2. Generate LLM explanation                                    │
│  3. Execute order via WEEX API                                 │
│  4. Upload AI log for compliance                               │
└────────────────────────────────────────────────────────────────┘
```

---

## AI Log Generation

Every trade decision generates an AI log containing:

```json
{
  "stage": "Decision Making",
  "model": "CryptoRegimeEngine-v2.0",
  "input": {
    "prompt": "Analyze market conditions",
    "data": {
      "market_data": {...},
      "timestamp": "2026-01-18T15:00:00"
    }
  },
  "output": {
    "regime": "TREND",
    "regime_confidence": 0.78,
    "tradability_score": 72,
    "signal": "LONG",
    "signal_confidence": 0.65
  },
  "explanation": "AI recommends opening long position on BTC. Market in TREND regime (78% confidence). Orderbook shows 1.45x bid dominance. Funding rate negative, favoring longs."
}
```

---

## Key AI Innovations

1. **Multi-Model Ensemble**: 5 specialized models working together
2. **Regime-Adaptive**: Strategy changes based on market conditions
3. **Temporal Multi-Horizon**: Predictions for 15s to 5min timeframes
4. **Cross-Asset Intelligence**: BTC momentum cascade detection
5. **Explainable AI**: LLM-generated reasoning for every decision
6. **Continuous Learning**: Models can be retrained on new data

---

## Technologies Used

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn |
| Classifiers | Random Forest, Gradient Boosting |
| API Framework | FastAPI |
| LLM Integration | OpenRouter (5 free models) |
| Data Sources | WEEX API, CoinGecko API |
| Language | Python 3.10+ |
