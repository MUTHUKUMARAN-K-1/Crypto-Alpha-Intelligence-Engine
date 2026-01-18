"""
WebSocket Handler for Real-Time Data.
Manages WebSocket connections for live market data streaming.
Integrates with CoinGecko WebSocket API for sub-second updates.
"""

from typing import Dict, Any, List, Optional, Callable, Set
import asyncio
from datetime import datetime
import json

from app.utils.logger import app_logger
from app.config import settings


class WebSocketHandler:
    """
    WebSocket connection manager for real-time market data.
    
    Supports:
    - CoinGecko WebSocket API (requires Analyst plan)
    - Price streaming for multiple assets
    - Real-time regime update notifications
    - Client broadcast for frontend updates
    """
    
    def __init__(self):
        """Initialize WebSocket handler."""
        self._ws_connection = None
        self._connected = False
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._latest_prices: Dict[str, Dict[str, Any]] = {}
        self._reconnect_attempts = 0
        self._max_reconnect = 5
    
    async def connect_coingecko(self, api_key: str = None) -> bool:
        """
        Connect to CoinGecko WebSocket API.
        
        Note: CoinGecko WebSocket requires Analyst plan or higher.
        For hackathon, they provide free 2-month access.
        
        Args:
            api_key: CoinGecko API key with WebSocket access
            
        Returns:
            True if connected successfully
        """
        try:
            import websockets
            
            ws_url = "wss://ws.coingecko.com/v1/ws"  # CoinGecko WebSocket endpoint
            
            if api_key:
                ws_url = f"{ws_url}?x_cg_pro_api_key={api_key}"
            
            app_logger.info(f"Connecting to CoinGecko WebSocket...")
            
            self._ws_connection = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10
            )
            
            self._connected = True
            self._reconnect_attempts = 0
            
            app_logger.info("CoinGecko WebSocket connected")
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            return True
            
        except ImportError:
            app_logger.warning("websockets library not installed. WebSocket disabled.")
            return False
        except Exception as e:
            app_logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe_prices(self, asset_ids: List[str], currency: str = "usd"):
        """
        Subscribe to real-time price updates.
        
        Args:
            asset_ids: List of CoinGecko asset IDs
            currency: Currency for prices (default: usd)
        """
        if not self._connected or not self._ws_connection:
            app_logger.warning("WebSocket not connected. Cannot subscribe.")
            return
        
        # CoinGecko WebSocket subscription format
        subscribe_msg = {
            "type": "subscribe",
            "channels": [
                {
                    "name": "ticker",
                    "asset_ids": asset_ids,
                    "currency": currency
                }
            ]
        }
        
        await self._ws_connection.send(json.dumps(subscribe_msg))
        app_logger.info(f"Subscribed to price updates for: {asset_ids}")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        if not self._ws_connection:
            return
        
        try:
            async for message in self._ws_connection:
                data = json.loads(message)
                await self._process_message(data)
                
        except Exception as e:
            app_logger.error(f"WebSocket message handler error: {e}")
            self._connected = False
            await self._attempt_reconnect()
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        msg_type = data.get("type")
        
        if msg_type == "ticker":
            # Price update
            asset_id = data.get("asset_id")
            price_data = {
                "asset_id": asset_id,
                "price": data.get("price"),
                "price_change_24h": data.get("price_change_24h"),
                "volume_24h": data.get("volume_24h"),
                "timestamp": datetime.now().isoformat()
            }
            
            self._latest_prices[asset_id] = price_data
            
            # Notify subscribers
            await self._notify_subscribers("price_update", price_data)
            
        elif msg_type == "error":
            app_logger.error(f"WebSocket error: {data.get('message')}")
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect after connection loss."""
        if self._reconnect_attempts >= self._max_reconnect:
            app_logger.error("Max reconnection attempts reached")
            return
        
        self._reconnect_attempts += 1
        wait_time = min(30, 2 ** self._reconnect_attempts)
        
        app_logger.info(f"Attempting reconnect in {wait_time}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(wait_time)
        
        await self.connect_coingecko(settings.coingecko_api_key)
    
    def add_subscriber(self, event_type: str, callback: Callable):
        """
        Add a subscriber for specific event type.
        
        Args:
            event_type: Type of event (e.g., "price_update")
            callback: Async callback function
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)
    
    def remove_subscriber(self, event_type: str, callback: Callable):
        """Remove a subscriber."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
    
    async def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify all subscribers of an event."""
        if event_type not in self._subscribers:
            return
        
        for callback in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                app_logger.error(f"Subscriber callback error: {e}")
    
    def get_latest_price(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latest cached price for an asset.
        
        Args:
            asset_id: CoinGecko asset ID
            
        Returns:
            Latest price data or None
        """
        return self._latest_prices.get(asset_id)
    
    def get_all_latest_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached latest prices."""
        return self._latest_prices.copy()
    
    async def disconnect(self):
        """Disconnect WebSocket connection."""
        if self._ws_connection:
            await self._ws_connection.close()
            self._ws_connection = None
            self._connected = False
            app_logger.info("WebSocket disconnected")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected


class SimulatedPriceStream:
    """
    Simulated price stream for development/demo when WebSocket unavailable.
    Generates realistic price movements based on current market data.
    """
    
    def __init__(self):
        """Initialize simulated stream."""
        self._running = False
        self._base_prices: Dict[str, float] = {}
        self._callbacks: List[Callable] = []
    
    async def start(
        self,
        assets: List[str],
        base_prices: Dict[str, float],
        update_interval: float = 1.0
    ):
        """
        Start simulated price stream.
        
        Args:
            assets: List of asset IDs
            base_prices: Starting prices for each asset
            update_interval: Seconds between updates
        """
        self._base_prices = base_prices.copy()
        self._running = True
        
        app_logger.info(f"Starting simulated price stream for {assets}")
        
        import random
        
        while self._running:
            for asset in assets:
                if asset in self._base_prices:
                    # Simulate price movement (random walk with momentum)
                    change_pct = random.gauss(0, 0.001)  # 0.1% std dev
                    self._base_prices[asset] *= (1 + change_pct)
                    
                    price_data = {
                        "asset_id": asset,
                        "price": self._base_prices[asset],
                        "price_change_24h": random.uniform(-5, 5),
                        "volume_24h": random.uniform(1e9, 50e9),
                        "timestamp": datetime.now().isoformat(),
                        "simulated": True
                    }
                    
                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(price_data)
                            else:
                                callback(price_data)
                        except Exception as e:
                            app_logger.error(f"Simulated stream callback error: {e}")
            
            await asyncio.sleep(update_interval)
    
    def stop(self):
        """Stop the simulated stream."""
        self._running = False
    
    def add_callback(self, callback: Callable):
        """Add a callback for price updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


# Global instances
websocket_handler = WebSocketHandler()
simulated_stream = SimulatedPriceStream()


async def initialize_realtime_stream(
    assets: List[str] = None,
    use_simulation: bool = False
) -> bool:
    """
    Initialize real-time price streaming.
    
    Args:
        assets: Assets to stream (default: BTC, ETH, SOL)
        use_simulation: Force simulation mode
        
    Returns:
        True if streaming initialized
    """
    if assets is None:
        assets = ["bitcoin", "ethereum", "solana"]
    
    if not use_simulation and settings.coingecko_api_key:
        # Try real WebSocket
        connected = await websocket_handler.connect_coingecko(
            settings.coingecko_api_key
        )
        
        if connected:
            await websocket_handler.subscribe_prices(assets)
            return True
    
    # Fallback to simulation
    app_logger.info("Using simulated price stream")
    
    # Get initial prices from CoinGecko REST API
    from app.data.coingecko_client import coingecko_client
    
    base_prices = {}
    for asset in assets:
        try:
            market_data = await coingecko_client.get_market_data(asset)
            base_prices[asset] = market_data.get("current_price", 0)
        except Exception:
            # Default prices
            defaults = {"bitcoin": 45000, "ethereum": 2500, "solana": 100}
            base_prices[asset] = defaults.get(asset, 100)
    
    asyncio.create_task(
        simulated_stream.start(assets, base_prices, update_interval=2.0)
    )
    
    return True
