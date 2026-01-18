# System Architecture

## Crypto Alpha Intelligence Engine - Technical Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│                     Dashboard (HTML/CSS/JS)                             │
│                     http://localhost:8000/dashboard                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                   │
│                         FastAPI (27 routes)                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐            │
│  │ routes.py   │  │ novelty_routes  │  │ trading_routes   │            │
│  │ (Core API)  │  │ (Advanced AI)   │  │ (Trading Bot)    │            │
│  └─────────────┘  └─────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌────────────────────────────┐    ┌────────────────────────────────────┐
│      DATA LAYER            │    │         INTELLIGENCE LAYER          │
│  ┌──────────────────────┐  │    │  ┌─────────────────────────────┐   │
│  │ weex_client.py       │  │    │  │ regime_engine.py            │   │
│  │ (WEEX Futures API)   │  │    │  │ (Main Orchestrator)         │   │
│  └──────────────────────┘  │    │  └─────────────────────────────┘   │
│  ┌──────────────────────┐  │    │  ┌─────────────────────────────┐   │
│  │ coingecko_client.py  │  │    │  │ trade_executor.py           │   │
│  │ (Market Data API)    │  │    │  │ (Trade Orchestration)       │   │
│  └──────────────────────┘  │    │  └─────────────────────────────┘   │
│  ┌──────────────────────┐  │    │  ┌─────────────────────────────┐   │
│  │ websocket_handler.py │  │    │  │ strategy_selector.py        │   │
│  │ (Real-time Streams)  │  │    │  │ (Regime-Adaptive Strategy)  │   │
│  └──────────────────────┘  │    │  └─────────────────────────────┘   │
│  ┌──────────────────────┐  │    │  ┌─────────────────────────────┐   │
│  │ data_cache.py        │  │    │  │ llm_explainer.py            │   │
│  │ (TTL Caching)        │  │    │  │ (AI Trade Explanations)     │   │
│  └──────────────────────┘  │    │  └─────────────────────────────┘   │
└────────────────────────────┘    └────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ML MODELS LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ regime_     │  │ funding_    │  │ orderbook_  │  │ momentum_   │    │
│  │ model.py    │  │ predictor   │  │ cnn.py      │  │ cascade.py  │    │
│  │ (RF)        │  │ (GB)        │  │ (RF+LOB)    │  │ (Stats)     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│  ┌─────────────┐                                                        │
│  │ risk_model  │                                                        │
│  │ (GB)        │                                                        │
│  └─────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
│  │ volatility  │  │ correlation │  │ liquidity   │                     │
│  │ features    │  │ features    │  │ features    │                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
crypto-regime-engine/
│
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Environment configuration
│   ├── trading_bot.py             # Main async trading loop
│   │
│   ├── api/
│   │   ├── routes.py              # Core analysis endpoints
│   │   ├── novelty_routes.py      # Advanced AI endpoints
│   │   └── trading_routes.py      # Trading & bot control
│   │
│   ├── data/
│   │   ├── weex_client.py         # WEEX API (auth, orders)
│   │   ├── coingecko_client.py    # CoinGecko API
│   │   ├── websocket_handler.py   # Real-time streaming
│   │   └── data_cache.py          # TTL caching layer
│   │
│   ├── models/
│   │   ├── regime_model.py        # Random Forest classifier
│   │   ├── risk_model.py          # Gradient Boosting regressor
│   │   ├── funding_predictor.py   # Funding rate ML
│   │   ├── orderbook_cnn.py       # LOB deep learning
│   │   └── momentum_cascade.py    # Cross-asset momentum
│   │
│   ├── intelligence/
│   │   ├── regime_engine.py       # Main orchestrator
│   │   ├── trade_executor.py      # Trade management
│   │   ├── strategy_selector.py   # Strategy switching
│   │   ├── llm_explainer.py       # LLM integration
│   │   └── position_sizer.py      # Risk-based sizing
│   │
│   ├── features/
│   │   ├── volatility.py          # Volatility analysis
│   │   ├── correlation.py         # Cross-asset correlation
│   │   └── liquidity.py           # Liquidity metrics
│   │
│   └── utils/
│       └── logger.py              # Logging configuration
│
├── docs/
│   ├── TRADING_POLICY.md          # Trading rules
│   ├── AI_PARTICIPATION.md        # AI/ML description
│   └── ARCHITECTURE.md            # This file
│
├── scripts/
│   └── api_test.py                # Hackathon API testing
│
├── frontend/
│   ├── index.html                 # Dashboard UI
│   ├── app.js                     # Frontend logic
│   └── styles.css                 # Styling
│
├── public/
│   └── logo.png                   # Project logo
│
├── requirements.txt
├── .env.example
├── run.bat / run.sh
└── README.md
```

### API Endpoints (27 Total)

#### Core Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/regime` | GET | Detect market regime |
| `/tradability` | GET | Get tradability score |
| `/analyze` | GET | Full market analysis |

#### Trading
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trading/status` | GET | WEEX connection status |
| `/trading/summary` | GET | Trade count & status |
| `/trading/funding-prediction` | GET | Predict funding rate |
| `/trading/orderbook-prediction` | GET | Predict price direction |
| `/trading/cascade-prediction` | GET | Momentum cascade |
| `/trading/strategy` | GET | Get strategy recommendation |
| `/trading/bot/start` | POST | Start trading bot |
| `/trading/bot/stop` | POST | Stop trading bot |

#### Advanced AI
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/advanced/multi-timeframe` | GET | Multi-TF analysis |
| `/advanced/sentiment` | GET | Crypto sentiment |
| `/advanced/whale-tracking` | GET | Whale movements |

### Data Flow

```
1. DATA INGESTION
   ├── WEEX API → Orderbook, Funding Rates, Positions
   └── CoinGecko → Prices, Volume, Market Cap

2. FEATURE EXTRACTION
   ├── Volatility features (ATR, spikes, trend)
   ├── Correlation matrix (cross-asset)
   └── Liquidity metrics (stability, depth)

3. ML PREDICTIONS (Parallel)
   ├── Regime Classifier → TREND/RANGE/HIGH-RISK
   ├── Funding Predictor → LONG/SHORT/WAIT
   ├── Orderbook CNN → UP/DOWN/NEUTRAL
   └── Momentum Cascade → Asset signals

4. SIGNAL AGGREGATION
   └── Weighted ensemble → Final signal

5. STRATEGY SELECTION
   └── Regime-based strategy → Parameters

6. TRADE EXECUTION
   ├── Position sizing
   ├── Order placement (WEEX API)
   └── AI log upload

7. MONITORING
   └── Dashboard updates
```

### Security

- API keys stored in `.env` (never committed)
- HMAC-SHA256 request signing for WEEX
- Rate limiting on API calls
- Paper trading mode for testing
