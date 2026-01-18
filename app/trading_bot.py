"""
Crypto Alpha Trading Bot - Main Trading Loop.
Orchestrates all AI models for real-time trading on WEEX.
"""

import asyncio
import signal
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import sys

from app.data.weex_client import weex_client
from app.data.coingecko_client import coingecko_client
from app.intelligence.trade_executor import trade_executor, TradeAction
from app.intelligence.regime_engine import regime_engine
from app.config import settings
from app.utils.logger import app_logger


class TradingBot:
    """
    Main trading bot for WEEX AI Hackathon.
    
    Features:
    - Continuous trading loop with configurable interval
    - Multi-asset signal generation
    - Real-time market data from WEEX + CoinGecko
    - Automatic AI log upload for compliance
    - Paper trading mode for testing
    """
    
    # Competition assets
    COMPETITION_ASSETS = [
        "bitcoin", "ethereum", "solana", "cardano",
        "dogecoin", "ripple", "litecoin", "binancecoin"
    ]
    
    def __init__(
        self,
        assets: List[str] = None,
        loop_interval: int = 60,
        paper_trading: bool = None
    ):
        """
        Initialize trading bot.
        
        Args:
            assets: Assets to trade (default: competition assets)
            loop_interval: Seconds between trading cycles
            paper_trading: Paper trading mode
        """
        self.assets = assets or self.COMPETITION_ASSETS[:4]  # Start with top 4
        self.loop_interval = loop_interval
        self.paper_trading = paper_trading if paper_trading is not None else settings.paper_trading
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Price history for momentum cascade
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=100) for asset in self.COMPETITION_ASSETS
        }
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.cycle_count = 0
        self.error_count = 0
        
        app_logger.info(f"Trading Bot initialized | Assets: {self.assets} | Paper: {self.paper_trading}")
    
    async def initialize(self) -> bool:
        """
        Initialize bot connections and data.
        
        Returns:
            True if initialization successful
        """
        app_logger.info("Initializing Trading Bot...")
        
        # Check WEEX API connection
        try:
            weex_connected = await weex_client.ping()
            if weex_connected:
                app_logger.info("✓ WEEX API connected")
            else:
                app_logger.warning("⚠ WEEX API not available (paper trading only)")
        except Exception as e:
            app_logger.warning(f"⚠ WEEX API error: {e}")
            weex_connected = False
        
        # Check CoinGecko API
        try:
            cg_connected = await coingecko_client.ping()
            if cg_connected:
                app_logger.info("✓ CoinGecko API connected")
            else:
                app_logger.warning("⚠ CoinGecko API not available")
        except Exception as e:
            app_logger.warning(f"⚠ CoinGecko API error: {e}")
        
        # Load initial price history
        await self._load_price_history()
        
        app_logger.info("Trading Bot initialization complete")
        return True
    
    async def _load_price_history(self):
        """Load historical prices for momentum analysis."""
        app_logger.info("Loading price history...")
        
        for asset in self.COMPETITION_ASSETS:
            try:
                history = await coingecko_client.get_price_history(asset, days=7)
                if history is not None and not history.empty:
                    prices = history["price"].tolist()
                    self.price_history[asset].extend(prices[-50:])
                    app_logger.debug(f"Loaded {len(prices)} prices for {asset}")
            except Exception as e:
                app_logger.warning(f"Failed to load history for {asset}: {e}")
    
    async def _update_prices(self):
        """Update current prices from WEEX."""
        for asset in self.assets:
            try:
                symbol = weex_client.normalize_symbol(asset)
                ticker = await weex_client.get_ticker(symbol)
                if ticker:
                    price = float(ticker.get("last", ticker.get("close", 0)))
                    if price > 0:
                        self.price_history[asset].append(price)
            except Exception as e:
                app_logger.debug(f"Failed to update price for {asset}: {e}")
    
    def _get_btc_prices(self) -> List[float]:
        """Get BTC price history for cascade analysis."""
        return list(self.price_history.get("bitcoin", []))
    
    def _get_altcoin_prices(self) -> Dict[str, List[float]]:
        """Get altcoin price histories for cascade analysis."""
        return {
            asset: list(prices)
            for asset, prices in self.price_history.items()
            if asset != "bitcoin" and len(prices) > 10
        }
    
    async def _trading_cycle(self):
        """Execute one trading cycle."""
        self.cycle_count += 1
        app_logger.info(f"=== Trading Cycle {self.cycle_count} ===")
        
        try:
            # Update prices
            await self._update_prices()
            
            # Get account balance (or use default for paper)
            if self.paper_trading:
                account_balance = 1000.0  # Paper trading balance
            else:
                try:
                    account = await weex_client.get_account_info()
                    account_balance = float(account.get("available", 1000))
                except Exception:
                    account_balance = 1000.0
            
            # Run trading cycle
            trades = await trade_executor.run_trading_cycle(
                assets=self.assets,
                btc_prices=self._get_btc_prices(),
                altcoin_prices=self._get_altcoin_prices(),
                account_balance=account_balance
            )
            
            if trades:
                app_logger.info(f"Executed {len(trades)} trade(s) in cycle {self.cycle_count}")
                for trade in trades:
                    app_logger.info(
                        f"  → {trade.action.value} {trade.asset} @ {trade.entry_price:.2f}"
                    )
            else:
                app_logger.info("No trades executed this cycle")
            
            # Log status
            summary = trade_executor.get_trade_summary()
            app_logger.info(
                f"Total trades: {summary['total_trades']} | "
                f"Min required: {summary['minimum_required']} | "
                f"Met: {summary['requirement_met']}"
            )
            
        except Exception as e:
            self.error_count += 1
            app_logger.error(f"Error in trading cycle: {e}")
    
    async def run(self, duration_minutes: int = None):
        """
        Run the trading bot.
        
        Args:
            duration_minutes: Run duration in minutes (None = indefinite)
        """
        self._running = True
        self.start_time = datetime.now()
        
        end_time = None
        if duration_minutes:
            end_time = self.start_time + timedelta(minutes=duration_minutes)
            app_logger.info(f"Bot will run until {end_time}")
        
        app_logger.info(f"Starting Trading Bot | Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        app_logger.info(f"Loop interval: {self.loop_interval}s | Assets: {', '.join(self.assets)}")
        
        try:
            while self._running:
                # Check duration
                if end_time and datetime.now() >= end_time:
                    app_logger.info("Duration reached, stopping bot")
                    break
                
                # Run cycle
                await self._trading_cycle()
                
                # Wait for next cycle
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.loop_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop
                    
        except asyncio.CancelledError:
            app_logger.info("Bot cancelled")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the trading bot."""
        app_logger.info("Shutting down Trading Bot...")
        self._running = False
        self._shutdown_event.set()
        
        # Log final stats
        runtime = datetime.now() - self.start_time if self.start_time else timedelta()
        summary = trade_executor.get_trade_summary()
        
        app_logger.info("=" * 50)
        app_logger.info("TRADING BOT FINAL STATS")
        app_logger.info(f"Runtime: {runtime}")
        app_logger.info(f"Cycles completed: {self.cycle_count}")
        app_logger.info(f"Total trades: {summary['total_trades']}")
        app_logger.info(f"Minimum trades met: {summary['requirement_met']}")
        app_logger.info(f"Errors: {self.error_count}")
        app_logger.info("=" * 50)
        
        # Close connections
        await weex_client.close()
    
    def stop(self):
        """Signal the bot to stop."""
        self._running = False
        self._shutdown_event.set()


# Create global bot instance
trading_bot = TradingBot()


async def run_bot(
    mode: str = "paper",
    duration: int = None,
    assets: List[str] = None
):
    """
    Convenience function to run the trading bot.
    
    Args:
        mode: "paper" or "live"
        duration: Duration in minutes
        assets: Assets to trade
    """
    bot = TradingBot(
        assets=assets,
        paper_trading=(mode.lower() == "paper")
    )
    
    await bot.initialize()
    await bot.run(duration_minutes=duration)


def main():
    """Main entry point for running the bot."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Alpha Trading Bot")
    parser.add_argument(
        "--mode", 
        choices=["paper", "live"],
        default="paper",
        help="Trading mode"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in minutes (default: run indefinitely)"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Assets to trade"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Loop interval in seconds"
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        app_logger.info("Interrupt received, stopping...")
        trading_bot.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run bot
    bot = TradingBot(
        assets=args.assets,
        loop_interval=args.interval,
        paper_trading=(args.mode == "paper")
    )
    
    async def run():
        await bot.initialize()
        await bot.run(duration_minutes=args.duration)
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
