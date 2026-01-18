# Trading Policy Document

## Crypto Alpha Intelligence Engine - Trading Logic & Rules

### 1. Trading Logic Overview

The Crypto Alpha system uses a **multi-model ensemble approach** to generate trading signals. Each model provides independent analysis, and signals are aggregated using weighted voting.

```
Signal Flow:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Market Data     │──│ AI Models       │──│ Trade Execution │
│ (WEEX + CG)     │  │ (5 Models)      │  │ (WEEX API)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 2. Signal Generation

#### Model 1: Regime Classification
- **Input**: Price volatility, volume patterns, market correlation
- **Output**: TREND / RANGE / HIGH-RISK classification
- **Weight**: 25%

#### Model 2: Funding Rate Prediction
- **Input**: Historical funding rates, open interest, price momentum
- **Output**: LONG (negative funding) / SHORT (positive funding) / WAIT
- **Weight**: 20%

#### Model 3: Orderbook Analysis
- **Input**: Bid/ask depth, imbalance ratio, spread
- **Output**: UP / DOWN / NEUTRAL (15s-5min prediction)
- **Weight**: 30%

#### Model 4: Momentum Cascade
- **Input**: BTC price momentum, altcoin correlation
- **Output**: Cascade signals for altcoins with timing
- **Weight**: 25%

### 3. Signal Aggregation

```python
final_score = Σ (signal_direction × confidence × weight)

if final_score > 0.2 and tradability >= 40:
    action = LONG
elif final_score < -0.2 and tradability >= 40:
    action = SHORT
else:
    action = HOLD
```

### 4. Trading Rules

| Rule | Value | Rationale |
|------|-------|-----------|
| Max Leverage | 20x | Hackathon requirement |
| Max Position Size | 10% of portfolio | Risk management |
| Stop Loss | 1-3% (strategy-dependent) | Capital protection |
| Take Profit | 1.5-5% (strategy-dependent) | Profit capture |
| Min Signal Confidence | 50% | Avoid weak signals |
| Min Tradability Score | 40/100 | Trade favorable conditions only |

### 5. Strategy Selection

The system dynamically selects strategy based on detected market regime:

| Regime | Primary Strategy | Secondary Strategy |
|--------|------------------|-------------------|
| TREND | Trend Following | Momentum Cascade |
| RANGE | Mean Reversion | Funding Arbitrage |
| HIGH-RISK | Defensive | Funding Arbitrage Only |

### 6. Risk Management

1. **Position Sizing**: Based on signal confidence and regime
2. **Leverage Adjustment**: Reduced in HIGH-RISK regime
3. **Stop Loss**: Always set on entry
4. **Maximum Drawdown**: Exit all positions if >15%
5. **Correlation Check**: Avoid overexposure to correlated assets

### 7. Order Execution

1. Market orders for high-urgency signals (momentum cascade)
2. Limit orders for mean reversion and funding arbitrage
3. All orders include pre-set TP/SL levels
4. AI log uploaded for every trade decision

### 8. Allowed Trading Pairs

As per hackathon rules:
- cmt_btcusdt
- cmt_ethusdt
- cmt_solusdt
- cmt_dogeusdt
- cmt_xrpusdt
- cmt_adausdt
- cmt_bnbusdt
- cmt_ltcusdt

### 9. Compliance

- Minimum 10 trades required ✓
- Maximum 20x leverage enforced ✓
- AI log upload for every decision ✓
- Paper trading mode for testing ✓
