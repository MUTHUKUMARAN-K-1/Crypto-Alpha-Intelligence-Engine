"""
Trade Executor - Orchestrates trade execution with all AI models.
Combines signals from multiple models, applies risk management, and executes trades.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from app.data.weex_client import weex_client, OrderSide, OrderType, WEEXAPIError
from app.models.funding_predictor import funding_predictor, FundingPrediction
from app.models.orderbook_cnn import orderbook_cnn, OrderbookPrediction, TimeHorizon
from app.models.momentum_cascade import momentum_cascade, MomentumCascadePrediction
from app.intelligence.strategy_selector import strategy_selector, Strategy, StrategyRecommendation
from app.intelligence.llm_explainer import llm_explainer, TradeExplanation
from app.intelligence.position_sizer import position_sizer
from app.intelligence.regime_engine import regime_engine
from app.config import settings
from app.utils.logger import app_logger


class TradeAction(Enum):
    """Trade actions."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Aggregated trade signal from all models."""
    asset: str
    action: TradeAction
    confidence: float
    source_signals: Dict[str, str]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutedTrade:
    """Record of an executed trade."""
    trade_id: str
    order_id: str
    asset: str
    symbol: str
    action: TradeAction
    size: float
    leverage: int
    entry_price: float
    stop_loss: float
    take_profit: float
    signal: TradeSignal
    explanation: TradeExplanation
    ai_log_uploaded: bool
    timestamp: datetime


class TradeExecutor:
    """
    Main trade execution engine.
    
    Coordinates:
    1. Signal generation from all AI models
    2. Strategy selection based on regime
    3. Position sizing and risk management
    4. Order execution via WEEX API
    5. AI log upload for hackathon compliance
    """
    
    def __init__(self):
        """Initialize trade executor."""
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[ExecutedTrade] = []
        self.paper_trading = settings.paper_trading
        self.max_position_pct = settings.max_position_pct
        self.max_leverage = min(settings.max_leverage, 20)  # Hackathon cap
        
        # Track trade count for hackathon minimum
        self.trade_count = 0
        self.min_trades_required = 10
    
    async def gather_signals(
        self,
        asset: str,
        btc_prices: List[float] = None,
        altcoin_prices: Dict[str, List[float]] = None
    ) -> Dict[str, Any]:
        """
        Gather signals from all AI models.
        
        Args:
            asset: Asset to analyze
            btc_prices: BTC price history for cascade detection
            altcoin_prices: Other altcoin prices
            
        Returns:
            Dict of signals from each model
        """
        signals = {}
        
        try:
            # 1. Get regime analysis
            regime_result = await regime_engine.analyze_regime([asset])
            signals["regime"] = {
                "regime": regime_result.get("regime", "UNKNOWN"),
                "confidence": regime_result.get("confidence", 0.5),
                "tradability_score": regime_result.get("tradability_score", 50)
            }
        except Exception as e:
            app_logger.warning(f"Regime analysis failed: {e}")
            signals["regime"] = {"regime": "UNKNOWN", "confidence": 0.5}
        
        try:
            # 2. Get funding rate prediction
            symbol = weex_client.normalize_symbol(asset)
            funding_history = await weex_client.get_funding_history(symbol)
            
            if funding_history:
                funding_pred = funding_predictor.predict(funding_history)
                signals["funding"] = {
                    "direction": funding_pred.direction,
                    "action": funding_pred.suggested_action,
                    "confidence": funding_pred.confidence,
                    "current_rate": funding_pred.current_rate,
                    "reasoning": funding_pred.reasoning
                }
        except Exception as e:
            app_logger.warning(f"Funding prediction failed: {e}")
            signals["funding"] = {"action": "WAIT", "confidence": 0}
        
        try:
            # 3. Get orderbook prediction
            symbol = weex_client.normalize_symbol(asset)
            orderbook = await weex_client.get_orderbook(symbol)
            
            if orderbook:
                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                
                ob_pred = orderbook_cnn.predict(bids, asks, TimeHorizon.SECONDS_60)
                signals["orderbook"] = {
                    "direction": ob_pred.direction,
                    "confidence": ob_pred.confidence,
                    "imbalance": ob_pred.imbalance_ratio,
                    "reasoning": ob_pred.reasoning
                }
        except Exception as e:
            app_logger.warning(f"Orderbook prediction failed: {e}")
            signals["orderbook"] = {"direction": "NEUTRAL", "confidence": 0}
        
        try:
            # 4. Get momentum cascade prediction
            if btc_prices and altcoin_prices:
                cascade_pred = momentum_cascade.predict_cascade(
                    btc_prices, altcoin_prices
                )
                
                signals["cascade"] = {
                    "btc_momentum": cascade_pred.btc_momentum,
                    "btc_direction": cascade_pred.btc_momentum_direction,
                    "cascade_expected": cascade_pred.cascade_expected,
                    "market_bias": cascade_pred.overall_market_bias,
                    "confidence": cascade_pred.confidence
                }
                
                # Find signal for this specific asset
                for sig in cascade_pred.cascade_signals:
                    if sig.asset.lower() == asset.lower():
                        signals["cascade"]["asset_signal"] = sig.direction
                        signals["cascade"]["asset_strength"] = sig.strength
                        break
        except Exception as e:
            app_logger.warning(f"Cascade prediction failed: {e}")
            signals["cascade"] = {"cascade_expected": False, "confidence": 0}
        
        return signals
    
    def aggregate_signals(
        self,
        signals: Dict[str, Any],
        asset: str
    ) -> TradeSignal:
        """
        Aggregate signals from all models into a single trade decision.
        
        Args:
            signals: Dict of signals from each model
            asset: Asset being analyzed
            
        Returns:
            TradeSignal with final decision
        """
        # Weight each signal source
        weights = {
            "regime": 0.25,
            "funding": 0.20,
            "orderbook": 0.30,
            "cascade": 0.25
        }
        
        # Calculate directional score (-1 to 1)
        score = 0.0
        total_weight = 0.0
        source_signals = {}
        reasoning_parts = []
        
        # Regime signal
        regime = signals.get("regime", {})
        if regime.get("regime") == "TREND":
            score += weights["regime"] * 0.3  # Slight bullish bias in trends
            source_signals["regime"] = "TREND"
        elif regime.get("regime") == "HIGH-RISK":
            score -= weights["regime"] * 0.5  # Bias toward caution
            source_signals["regime"] = "HIGH-RISK"
        total_weight += weights["regime"]
        
        # Funding signal
        funding = signals.get("funding", {})
        if funding.get("action") == "LONG":
            score += weights["funding"] * funding.get("confidence", 0.5)
            source_signals["funding"] = "LONG"
            reasoning_parts.append(f"Funding rate favors longs ({funding.get('current_rate', 0)*100:.4f}%)")
        elif funding.get("action") == "SHORT":
            score -= weights["funding"] * funding.get("confidence", 0.5)
            source_signals["funding"] = "SHORT"
            reasoning_parts.append(f"Funding rate favors shorts ({funding.get('current_rate', 0)*100:.4f}%)")
        total_weight += weights["funding"]
        
        # Orderbook signal
        orderbook = signals.get("orderbook", {})
        if orderbook.get("direction") == "UP":
            score += weights["orderbook"] * orderbook.get("confidence", 0.5)
            source_signals["orderbook"] = "UP"
            reasoning_parts.append(f"Orderbook bullish (imbalance: {orderbook.get('imbalance', 1):.2f}x)")
        elif orderbook.get("direction") == "DOWN":
            score -= weights["orderbook"] * orderbook.get("confidence", 0.5)
            source_signals["orderbook"] = "DOWN"
            reasoning_parts.append(f"Orderbook bearish (imbalance: {orderbook.get('imbalance', 1):.2f}x)")
        total_weight += weights["orderbook"]
        
        # Cascade signal
        cascade = signals.get("cascade", {})
        if cascade.get("cascade_expected"):
            asset_signal = cascade.get("asset_signal", cascade.get("market_bias"))
            if asset_signal == "BULLISH":
                score += weights["cascade"] * cascade.get("confidence", 0.5)
                source_signals["cascade"] = "BULLISH"
                reasoning_parts.append(f"BTC momentum cascade bullish")
            elif asset_signal == "BEARISH":
                score -= weights["cascade"] * cascade.get("confidence", 0.5)
                source_signals["cascade"] = "BEARISH"
                reasoning_parts.append(f"BTC momentum cascade bearish")
        total_weight += weights["cascade"]
        
        # Normalize score
        if total_weight > 0:
            score /= total_weight
        
        # Determine action
        tradability = regime.get("tradability_score", 50)
        regime_confidence = regime.get("confidence", 0.5)
        
        if score > 0.2 and tradability >= 40:
            action = TradeAction.LONG
            confidence = min(0.95, abs(score) + regime_confidence * 0.2)
        elif score < -0.2 and tradability >= 40:
            action = TradeAction.SHORT
            confidence = min(0.95, abs(score) + regime_confidence * 0.2)
        else:
            action = TradeAction.HOLD
            confidence = 0.5
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No strong signals detected"
        
        return TradeSignal(
            asset=asset,
            action=action,
            confidence=confidence,
            source_signals=source_signals,
            reasoning=reasoning
        )
    
    async def execute_trade(
        self,
        signal: TradeSignal,
        account_balance: float = 1000.0
    ) -> Optional[ExecutedTrade]:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: TradeSignal from aggregation
            account_balance: Current account balance
            
        Returns:
            ExecutedTrade if successful, None otherwise
        """
        if signal.action == TradeAction.HOLD:
            app_logger.info(f"HOLD signal for {signal.asset} - no trade executed")
            return None
        
        # Get strategy recommendation
        regime = signal.source_signals.get("regime", "UNKNOWN")
        strategy_rec = strategy_selector.select_strategy(
            regime=regime,
            regime_confidence=signal.confidence,
            tradability_score=50,
            funding_signal=signal.source_signals.get("funding"),
            orderbook_signal=signal.source_signals.get("orderbook"),
            cascade_signal=signal.source_signals.get("cascade")
        )
        
        # Check if we should enter
        should_enter, reason = strategy_selector.should_enter_trade(
            strategy_rec, signal.confidence
        )
        
        if not should_enter:
            app_logger.info(f"Trade rejected: {reason}")
            return None
        
        # Calculate position size
        leverage = strategy_selector.get_effective_leverage(strategy_rec, self.max_leverage)
        position_factor = strategy_selector.get_position_size_factor(strategy_rec, signal.confidence)
        position_value = account_balance * self.max_position_pct * position_factor
        
        # Get current price
        symbol = weex_client.normalize_symbol(signal.asset)
        
        try:
            ticker = await weex_client.get_ticker(symbol)
            current_price = float(ticker.get("last", ticker.get("close", 0)))
        except Exception as e:
            app_logger.error(f"Failed to get ticker: {e}")
            return None
        
        if current_price <= 0:
            app_logger.error("Invalid price, cannot execute")
            return None
        
        # Calculate size in contracts
        size = position_value / current_price
        
        # Calculate stop loss and take profit
        config = strategy_rec.config
        if signal.action == TradeAction.LONG:
            stop_loss = current_price * (1 - config.stop_loss_pct / 100)
            take_profit = current_price * (1 + config.take_profit_pct / 100)
            order_side = OrderSide.BUY_OPEN
        else:
            stop_loss = current_price * (1 + config.stop_loss_pct / 100)
            take_profit = current_price * (1 - config.take_profit_pct / 100)
            order_side = OrderSide.SELL_OPEN
        
        # Generate explanation
        explanation = await llm_explainer.generate_explanation(
            signal=signal.action.value,
            asset=signal.asset,
            regime=regime,
            regime_confidence=signal.confidence,
            tradability_score=50,
            strategy=strategy_rec.primary_strategy.value,
            factors={
                "funding_prediction": signal.source_signals.get("funding"),
                "orderbook_signal": signal.source_signals.get("orderbook"),
                "cascade_signal": signal.source_signals.get("cascade"),
                "regime_signal": regime
            },
            market_data={"price": current_price}
        )
        
        trade_id = str(uuid.uuid4())[:8]
        
        if self.paper_trading:
            # Paper trade - simulate execution
            app_logger.info(
                f"[PAPER] {signal.action.value} {signal.asset} | "
                f"Size: {size:.4f} | Price: {current_price:.2f} | "
                f"Leverage: {leverage}x | SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
            )
            order_id = f"paper_{trade_id}"
            ai_log_uploaded = True  # Simulated
            
        else:
            # Real trade execution
            try:
                # Set leverage
                await weex_client.set_leverage(symbol, leverage)
                
                # Place order
                order_result = await weex_client.place_order(
                    symbol=symbol,
                    side=order_side,
                    size=size,
                    order_type=OrderType.MARKET,
                    client_oid=trade_id,
                    take_profit=take_profit,
                    stop_loss=stop_loss
                )
                
                order_id = order_result.get("order_id", trade_id)
                
                app_logger.info(
                    f"[LIVE] {signal.action.value} {signal.asset} | "
                    f"Order ID: {order_id} | Size: {size:.4f}"
                )
                
                # Upload AI log
                try:
                    await weex_client.generate_and_upload_ai_log(
                        regime=regime,
                        regime_confidence=signal.confidence,
                        tradability_score=50,
                        signal=signal.action.value,
                        signal_confidence=signal.confidence,
                        market_data={"price": current_price, "symbol": symbol},
                        reasoning=llm_explainer.format_for_ai_log(explanation),
                        order_id=order_id
                    )
                    ai_log_uploaded = True
                    app_logger.info(f"AI log uploaded for order {order_id}")
                except Exception as e:
                    app_logger.error(f"AI log upload failed: {e}")
                    ai_log_uploaded = False
                    
            except WEEXAPIError as e:
                app_logger.error(f"Order execution failed: {e}")
                return None
        
        # Record trade
        executed = ExecutedTrade(
            trade_id=trade_id,
            order_id=order_id,
            asset=signal.asset,
            symbol=symbol,
            action=signal.action,
            size=size,
            leverage=leverage,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal=signal,
            explanation=explanation,
            ai_log_uploaded=ai_log_uploaded,
            timestamp=datetime.now()
        )
        
        self.trade_history.append(executed)
        self.trade_count += 1
        
        # Track position
        self.active_positions[signal.asset] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": signal.action,
            "size": size,
            "entry_price": current_price
        }
        
        return executed
    
    async def run_trading_cycle(
        self,
        assets: List[str] = None,
        btc_prices: List[float] = None,
        altcoin_prices: Dict[str, List[float]] = None,
        account_balance: float = 1000.0
    ) -> List[ExecutedTrade]:
        """
        Run a complete trading cycle for all assets.
        
        Args:
            assets: Assets to trade (default: competition pairs)
            btc_prices: BTC price history
            altcoin_prices: Altcoin price histories
            account_balance: Current balance
            
        Returns:
            List of executed trades
        """
        if assets is None:
            assets = ["bitcoin", "ethereum", "solana", "cardano"]
        
        executed_trades = []
        
        for asset in assets:
            try:
                # Gather all signals
                signals = await self.gather_signals(asset, btc_prices, altcoin_prices)
                
                # Aggregate into trade decision
                trade_signal = self.aggregate_signals(signals, asset)
                
                # Execute if actionable
                if trade_signal.action != TradeAction.HOLD:
                    trade = await self.execute_trade(trade_signal, account_balance)
                    if trade:
                        executed_trades.append(trade)
                
                # Rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                app_logger.error(f"Error in trading cycle for {asset}: {e}")
        
        return executed_trades
    
    def get_trade_count(self) -> int:
        """Get total trade count for hackathon minimum requirement."""
        return self.trade_count
    
    def meets_minimum_trades(self) -> bool:
        """Check if minimum trade requirement is met."""
        return self.trade_count >= self.min_trades_required
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of trading activity."""
        return {
            "total_trades": self.trade_count,
            "minimum_required": self.min_trades_required,
            "requirement_met": self.meets_minimum_trades(),
            "active_positions": len(self.active_positions),
            "paper_trading": self.paper_trading
        }


# Global instance
trade_executor = TradeExecutor()
