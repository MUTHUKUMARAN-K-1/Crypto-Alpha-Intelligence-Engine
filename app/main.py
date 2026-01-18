from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.api.novelty_routes import novelty_router
from app.api.trading_routes import trading_router
from app.config import settings
from app.utils.logger import app_logger

# Get the frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    app_logger.info("=" * 50)
    app_logger.info("Crypto Alpha Intelligence Engine Starting...")
    app_logger.info(f"Log Level: {settings.log_level}")
    app_logger.info(f"CoinGecko API: {settings.coingecko_base_url}")
    app_logger.info(f"WEEX API: {settings.weex_base_url}")
    app_logger.info(f"Paper Trading: {settings.paper_trading}")
    app_logger.info(f"Cache TTL: {settings.cache_ttl}s")
    app_logger.info(f"Frontend Dir: {FRONTEND_DIR}")
    app_logger.info("=" * 50)
    
    # Initialize models (they self-initialize on import, but log it)
    from app.models.regime_model import regime_classifier
    from app.models.risk_model import risk_model
    from app.models.funding_predictor import funding_predictor
    from app.models.orderbook_cnn import orderbook_cnn
    from app.models.momentum_cascade import momentum_cascade
    
    app_logger.info(f"Regime Classifier: {'ready' if regime_classifier.is_trained else 'not trained'}")
    app_logger.info(f"Risk Model: {'ready' if risk_model.is_trained else 'not trained'}")
    app_logger.info(f"Funding Predictor: {'ready' if funding_predictor.is_trained else 'not trained'}")
    app_logger.info(f"Orderbook CNN: {'ready' if orderbook_cnn.is_trained else 'not trained'}")
    app_logger.info("Momentum Cascade: ready")
    
    # Log novelty features
    app_logger.info("Advanced AI Features: Funding Prediction, Orderbook CNN, Momentum Cascade, Strategy Selector")
    
    app_logger.info("Application ready to serve requests")
    app_logger.info(f"Dashboard: http://localhost:{settings.port}")
    app_logger.info(f"API Docs: http://localhost:{settings.port}/docs")
    
    yield
    
    # Shutdown
    app_logger.info("Crypto Alpha Intelligence Engine Shutting Down...")
    from app.data.weex_client import weex_client
    await weex_client.close()
    app_logger.info("Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="Crypto Alpha Intelligence Engine",
    description="""
## AI-Powered Cryptocurrency Trading System for WEEX AI Hackathon

This API provides machine learning-based market regime detection, tradability scoring,
and automated trading on WEEX futures using real-time data from CoinGecko and WEEX.

### 🏆 Hackathon Novelty Features

- **Funding Rate Predictor**: Neural network predicting funding rate direction for carry trades
- **Orderbook CNN**: Deep learning on limit order book for 15s-5min price prediction
- **Momentum Cascade**: Detect BTC→altcoin momentum spillover with timing
- **Regime-Adaptive Strategy**: Auto-switch between trend/mean-reversion/defensive
- **LLM Explainer**: AI-generated trade reasoning for compliance

### Core Features

- **Market Regime Detection**: Classify conditions as TREND, RANGE, or HIGH-RISK
- **Tradability Scoring**: 0-100 scores for trading favorability
- **Multi-Timeframe Analysis**: Confluence scoring across timeframes
- **Whale Tracking**: On-chain whale movement analysis

### Data Sources

- **WEEX API**: Real-time futures data, orderbook, funding rates, trade execution
- **CoinGecko API**: Historical prices, market data, WebSocket streaming

### ML Models

- **Regime Classifier**: Random Forest on volatility, correlation, liquidity
- **Risk Model**: Gradient Boosting for tradability
- **Funding Predictor**: Gradient Boosting for funding rate direction
- **Orderbook CNN**: Random Forest on LOB features
- **Momentum Cascade**: Statistical lead-lag detection
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(novelty_router)
app.include_router(trading_router)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        """Serve the frontend dashboard."""
        return FileResponse(FRONTEND_DIR / "index.html")
    
    @app.get("/app.js", include_in_schema=False)
    async def serve_js():
        """Serve the frontend JavaScript."""
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")
    
    @app.get("/styles.css", include_in_schema=False)
    async def serve_css():
        """Serve the frontend CSS."""
        return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


# Main entry point for running with `python -m app.main`
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
