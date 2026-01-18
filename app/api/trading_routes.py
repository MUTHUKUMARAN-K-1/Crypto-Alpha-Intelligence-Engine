"""
Trading API Routes - REST endpoints for trading functionality.
Provides API access to trading bot and execution features.
"""

from typing import List, Optional
from datetime import datetime
import asyncio

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.data.weex_client import weex_client
from app.intelligence.trade_executor import trade_executor, TradeAction
from app.intelligence.strategy_selector import strategy_selector, Strategy
from app.models.funding_predictor import funding_predictor
from app.models.orderbook_cnn import orderbook_cnn, TimeHorizon
from app.models.momentum_cascade import momentum_cascade
from app.trading_bot import TradingBot
from app.utils.logger import app_logger


# Create router
trading_router = APIRouter(prefix="/trading", tags=["Trading"])


# ============== Response Models ==============

class FundingPredictionResponse(BaseModel):
    """Funding rate prediction response."""
    direction: str
    probability: float
    current_rate: float
    predicted_rate: float
    suggested_action: str
    confidence: float
    reasoning: str


class OrderbookPredictionResponse(BaseModel):
    """Orderbook prediction response."""
    direction: str
    probability: float
    horizon_seconds: int
    imbalance_ratio: float
    bid_pressure: float
    ask_pressure: float
    confidence: float
    reasoning: str


class CascadeSignalResponse(BaseModel):
    """Cascade signal for an asset."""
    asset: str
    direction: str
    strength: float
    lag_minutes: float
    probability: float
    reasoning: str


class CascadePredictionResponse(BaseModel):
    """Momentum cascade prediction response."""
    btc_momentum: float
    btc_direction: str
    cascade_expected: bool
    signals: List[CascadeSignalResponse]
    market_bias: str
    confidence: float
    timing: str


class StrategyResponse(BaseModel):
    """Strategy recommendation response."""
    primary_strategy: str
    secondary_strategy: Optional[str]
    leverage: int
    position_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    reasoning: str
    confidence: float


class TradeSummaryResponse(BaseModel):
    """Trading summary response."""
    total_trades: int
    minimum_required: int
    requirement_met: bool
    active_positions: int
    paper_trading: bool


class WEEXStatusResponse(BaseModel):
    """WEEX connection status response."""
    connected: bool
    server_time: Optional[int]
    account_available: Optional[float]


# ============== Endpoints ==============

@trading_router.get("/status", response_model=WEEXStatusResponse)
async def get_weex_status():
    """
    Check WEEX API connection status.
    
    Returns connection status and account info if connected.
    """
    try:
        connected = await weex_client.ping()
        server_time = await weex_client.get_server_time() if connected else None
        
        account_available = None
        if connected:
            try:
                account = await weex_client.get_account_info()
                account_available = float(account.get("available", 0))
            except Exception:
                pass
        
        return WEEXStatusResponse(
            connected=connected,
            server_time=server_time,
            account_available=account_available
        )
    except Exception as e:
        return WEEXStatusResponse(
            connected=False,
            server_time=None,
            account_available=None
        )


@trading_router.get("/summary", response_model=TradeSummaryResponse)
async def get_trade_summary():
    """
    Get trading activity summary.
    
    Returns trade counts and status for hackathon requirements.
    """
    summary = trade_executor.get_trade_summary()
    return TradeSummaryResponse(**summary)


@trading_router.get("/funding-prediction", response_model=FundingPredictionResponse)
async def get_funding_prediction(
    asset: str = Query(..., description="Asset to analyze", examples=["btc"])
):
    """
    Get funding rate prediction for an asset.
    
    Predicts whether funding rate will be positive or negative,
    suggesting long/short positions to collect funding.
    """
    try:
        symbol = weex_client.normalize_symbol(asset)
        funding_history = await weex_client.get_funding_history(symbol, page_size=20)
        
        if not funding_history:
            raise HTTPException(status_code=400, detail="No funding history available")
        
        prediction = funding_predictor.predict(funding_history)
        
        return FundingPredictionResponse(
            direction=prediction.direction,
            probability=prediction.probability,
            current_rate=prediction.current_rate,
            predicted_rate=prediction.predicted_rate,
            suggested_action=prediction.suggested_action,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Funding prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/orderbook-prediction", response_model=OrderbookPredictionResponse)
async def get_orderbook_prediction(
    asset: str = Query(..., description="Asset to analyze", examples=["btc"]),
    horizon: str = Query("60", description="Prediction horizon in seconds", examples=["15", "60", "300"])
):
    """
    Get short-term price prediction from orderbook analysis.
    
    Uses CNN-style model on orderbook depth to predict
    price direction for 15s, 60s, or 5min horizons.
    """
    try:
        symbol = weex_client.normalize_symbol(asset)
        orderbook = await weex_client.get_orderbook(symbol, limit=20)
        
        if not orderbook:
            raise HTTPException(status_code=400, detail="Orderbook not available")
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Parse horizon
        horizon_map = {
            "15": TimeHorizon.SECONDS_15,
            "60": TimeHorizon.SECONDS_60,
            "300": TimeHorizon.MINUTES_5
        }
        time_horizon = horizon_map.get(horizon, TimeHorizon.SECONDS_60)
        
        prediction = orderbook_cnn.predict(bids, asks, time_horizon)
        
        return OrderbookPredictionResponse(
            direction=prediction.direction,
            probability=prediction.probability,
            horizon_seconds=prediction.horizon_seconds,
            imbalance_ratio=prediction.imbalance_ratio,
            bid_pressure=prediction.bid_pressure,
            ask_pressure=prediction.ask_pressure,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Orderbook prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/cascade-prediction", response_model=CascadePredictionResponse)
async def get_cascade_prediction():
    """
    Get momentum cascade prediction.
    
    Detects when BTC momentum may spill over to altcoins,
    providing timing recommendations for altcoin entries.
    """
    try:
        from app.data.coingecko_client import coingecko_client
        
        # Get price histories
        assets = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"]
        price_data = await coingecko_client.get_multiple_prices(assets, days=7)
        
        btc_prices = price_data.get("bitcoin", {}).get("price", []).tolist() if "bitcoin" in price_data else []
        
        altcoin_prices = {}
        for asset in assets:
            if asset != "bitcoin" and asset in price_data:
                altcoin_prices[asset] = price_data[asset]["price"].tolist()
        
        if not btc_prices or not altcoin_prices:
            raise HTTPException(status_code=400, detail="Insufficient price data")
        
        prediction = momentum_cascade.predict_cascade(btc_prices, altcoin_prices)
        
        signals = [
            CascadeSignalResponse(
                asset=s.asset,
                direction=s.direction,
                strength=s.strength,
                lag_minutes=s.lag_minutes,
                probability=s.probability,
                reasoning=s.reasoning
            )
            for s in prediction.cascade_signals
        ]
        
        return CascadePredictionResponse(
            btc_momentum=prediction.btc_momentum,
            btc_direction=prediction.btc_momentum_direction,
            cascade_expected=prediction.cascade_expected,
            signals=signals,
            market_bias=prediction.overall_market_bias,
            confidence=prediction.confidence,
            timing=prediction.timing_recommendation
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Cascade prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/strategy", response_model=StrategyResponse)
async def get_strategy_recommendation(
    regime: str = Query(..., description="Current market regime", examples=["TREND", "RANGE", "HIGH-RISK"]),
    confidence: float = Query(0.7, description="Regime confidence (0-1)"),
    tradability: float = Query(70, description="Tradability score (0-100)")
):
    """
    Get strategy recommendation based on market conditions.
    
    Returns optimal strategy, leverage, and risk parameters
    based on current regime and market conditions.
    """
    try:
        recommendation = strategy_selector.select_strategy(
            regime=regime.upper(),
            regime_confidence=confidence,
            tradability_score=tradability
        )
        
        leverage = strategy_selector.get_effective_leverage(recommendation)
        
        return StrategyResponse(
            primary_strategy=recommendation.primary_strategy.value,
            secondary_strategy=recommendation.secondary_strategy.value if recommendation.secondary_strategy else None,
            leverage=leverage,
            position_multiplier=recommendation.config.position_size_multiplier,
            stop_loss_pct=recommendation.config.stop_loss_pct,
            take_profit_pct=recommendation.config.take_profit_pct,
            reasoning=recommendation.reasoning,
            confidence=recommendation.confidence
        )
    except Exception as e:
        app_logger.error(f"Strategy recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background trading bot task
_bot_task = None


@trading_router.post("/bot/start")
async def start_trading_bot(
    background_tasks: BackgroundTasks,
    mode: str = Query("paper", description="Trading mode", examples=["paper", "live"]),
    duration: int = Query(None, description="Duration in minutes"),
    assets: str = Query("btc,eth,sol", description="Comma-separated assets")
):
    """
    Start the trading bot in the background.
    
    Bot will run continuously, generating signals and executing trades
    based on all AI models.
    """
    global _bot_task
    
    if _bot_task and not _bot_task.done():
        raise HTTPException(status_code=400, detail="Bot already running")
    
    asset_list = [a.strip() for a in assets.split(",")]
    
    bot = TradingBot(
        assets=asset_list,
        paper_trading=(mode.lower() == "paper")
    )
    
    async def run_bot():
        await bot.initialize()
        await bot.run(duration_minutes=duration)
    
    _bot_task = asyncio.create_task(run_bot())
    
    return {
        "status": "started",
        "mode": mode,
        "assets": asset_list,
        "duration": duration
    }


@trading_router.post("/bot/stop")
async def stop_trading_bot():
    """Stop the running trading bot."""
    global _bot_task
    
    if _bot_task and not _bot_task.done():
        _bot_task.cancel()
        return {"status": "stopped"}
    
    return {"status": "not_running"}


@trading_router.get("/bot/status")
async def get_bot_status():
    """Get current bot status."""
    global _bot_task
    
    running = _bot_task and not _bot_task.done() if _bot_task else False
    summary = trade_executor.get_trade_summary()
    
    return {
        "running": running,
        **summary
    }
