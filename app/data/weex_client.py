"""
WEEX API Client for AI Wars Hackathon.
Provides authenticated access to WEEX Futures trading API.
Includes market data, trade execution, and AI log upload.
"""

import asyncio
import hashlib
import hmac
import base64
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

import httpx

from app.config import settings
from app.utils.logger import app_logger


class OrderSide(Enum):
    """Order side enumeration."""
    BUY_OPEN = "1"      # Open long
    BUY_CLOSE = "2"     # Close short
    SELL_OPEN = "3"     # Open short
    SELL_CLOSE = "4"    # Close long


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "0"         # Limit order
    MARKET = "1"        # Market order


class WEEXAPIError(Exception):
    """Custom exception for WEEX API errors."""
    def __init__(self, message: str, code: str = None, status_code: int = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class WEEXClient:
    """
    Async client for WEEX Futures API.
    
    Supports:
    - HMAC-SHA256 authentication
    - Market data (candlesticks, orderbook, funding rates)
    - Trade execution (place/cancel orders, TP/SL)
    - Account management
    - AI log upload for hackathon compliance
    """
    
    # Base URL for WEEX Futures API
    BASE_URL = "https://api-contract.weex.com"
    
    # Competition trading pairs (as per hackathon rules)
    COMPETITION_PAIRS = [
        "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_adausdt",
        "cmt_dogeusdt", "cmt_xrpusdt", "cmt_ltcusdt", "cmt_bnbusdt"
    ]
    
    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        passphrase: str = None
    ):
        """
        Initialize WEEX API client.
        
        Args:
            api_key: WEEX API key
            secret_key: WEEX secret key for signing
            passphrase: WEEX passphrase
        """
        self.api_key = api_key or getattr(settings, 'weex_api_key', None)
        self.secret_key = secret_key or getattr(settings, 'weex_secret_key', None)
        self.passphrase = passphrase or getattr(settings, 'weex_passphrase', None)
        
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True
            )
        return self._client
    
    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        Generate HMAC-SHA256 signature for API authentication.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            method: HTTP method (GET/POST)
            request_path: API endpoint path
            body: Request body (JSON string for POST)
            
        Returns:
            Base64-encoded signature
        """
        if not self.secret_key:
            raise WEEXAPIError("Secret key not configured")
        
        # Message to sign: timestamp + method + requestPath + body
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        
        # HMAC-SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        # Base64 encode
        return base64.b64encode(signature).decode('utf-8')
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        return str(int(time.time() * 1000))
    
    def _get_headers(
        self,
        timestamp: str,
        signature: str
    ) -> Dict[str, str]:
        """
        Get authenticated request headers.
        
        Args:
            timestamp: Request timestamp
            signature: HMAC signature
            
        Returns:
            Headers dictionary
        """
        return {
            "ACCESS-KEY": self.api_key or "",
            "ACCESS-SIGN": signature,
            "ACCESS-PASSPHRASE": self.passphrase or "",
            "ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "locale": "en-US"
        }
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        authenticated: bool = True
    ) -> Dict[str, Any]:
        """
        Make authenticated request to WEEX API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            data: POST body data
            authenticated: Whether to include auth headers
            
        Returns:
            API response data
        """
        await self._rate_limit()
        
        client = await self._get_client()
        url = f"{self.BASE_URL}{endpoint}"
        
        # Build request path with query string
        request_path = endpoint
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            request_path = f"{endpoint}?{query_string}"
        
        # Prepare body
        body = json.dumps(data) if data else ""
        
        # Generate authentication
        headers = {}
        if authenticated:
            if not self.api_key:
                raise WEEXAPIError("API key not configured")
            
            timestamp = self._get_timestamp()
            signature = self._generate_signature(
                timestamp, method, request_path, body
            )
            headers = self._get_headers(timestamp, signature)
        
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params, headers=headers)
            else:
                response = await client.post(
                    url, 
                    content=body,
                    headers=headers
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors
            if result.get("code") and result.get("code") != "00000":
                raise WEEXAPIError(
                    message=result.get("msg", "Unknown error"),
                    code=result.get("code"),
                    status_code=response.status_code
                )
            
            return result
            
        except httpx.HTTPError as e:
            app_logger.error(f"WEEX API request failed: {e}")
            raise WEEXAPIError(f"HTTP error: {e}")
    
    # ==================== MARKET DATA ====================
    
    async def get_server_time(self) -> int:
        """Get WEEX server time."""
        result = await self._request("GET", "/capi/v2/market/time", authenticated=False)
        return result.get("data", {}).get("serverTime", 0)
    
    async def get_contract_info(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get futures contract information.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            List of contract info
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = await self._request(
            "GET", "/capi/v2/market/contracts",
            params=params, authenticated=False
        )
        return result.get("data", [])
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get orderbook depth data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'cmt_btcusdt')
            limit: Number of levels (5, 10, 20, 50, 100)
            
        Returns:
            Orderbook with bids and asks
        """
        result = await self._request(
            "GET", "/capi/v2/market/depth",
            params={"symbol": symbol, "limit": str(limit)},
            authenticated=False
        )
        return result.get("data", {})
    
    async def get_ticker(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get ticker information.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Ticker data
        """
        endpoint = "/capi/v2/market/ticker"
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = await self._request(
            "GET", endpoint, params=params, authenticated=False
        )
        return result.get("data", {})
    
    async def get_candlesticks(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: int = None,
        end_time: int = None,
        limit: int = 100
    ) -> List[List]:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Number of candles (max 500)
            
        Returns:
            List of [timestamp, open, high, low, close, volume]
        """
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": str(min(limit, 500))
        }
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)
        
        result = await self._request(
            "GET", "/capi/v2/market/candles",
            params=params, authenticated=False
        )
        return result.get("data", [])
    
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Funding rate data
        """
        result = await self._request(
            "GET", "/capi/v2/market/current-fund-rate",
            params={"symbol": symbol},
            authenticated=False
        )
        return result.get("data", {})
    
    async def get_funding_history(
        self,
        symbol: str,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get historical funding rates.
        
        Args:
            symbol: Trading pair symbol
            page: Page number
            page_size: Items per page
            
        Returns:
            List of funding rate records
        """
        result = await self._request(
            "GET", "/capi/v2/market/history-fund-rate",
            params={
                "symbol": symbol,
                "pageNo": str(page),
                "pageSize": str(page_size)
            },
            authenticated=False
        )
        return result.get("data", {}).get("resultList", [])
    
    async def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        Get open interest data.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Open interest data
        """
        result = await self._request(
            "GET", "/capi/v2/market/open-interest",
            params={"symbol": symbol},
            authenticated=False
        )
        return result.get("data", {})
    
    # ==================== ACCOUNT ====================
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information and balance.
        
        Returns:
            Account info with balance
        """
        result = await self._request("GET", "/capi/v2/account/account")
        return result.get("data", {})
    
    async def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            List of positions
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = await self._request("GET", "/capi/v2/position/positions", params=params)
        return result.get("data", [])
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair symbol
            leverage: Leverage value (1-20 for hackathon, max 20x)
            
        Returns:
            Response data
        """
        # Cap leverage at 20x per hackathon rules
        leverage = min(leverage, 20)
        
        result = await self._request(
            "POST", "/capi/v2/account/leverage",
            data={"symbol": symbol, "leverage": str(leverage)}
        )
        return result.get("data", {})
    
    # ==================== TRADING ====================
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
        client_oid: str = None,
        take_profit: float = None,
        stop_loss: float = None
    ) -> Dict[str, Any]:
        """
        Place a futures order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'cmt_btcusdt')
            side: Order side (BUY_OPEN, BUY_CLOSE, SELL_OPEN, SELL_CLOSE)
            size: Order size in contracts
            order_type: LIMIT or MARKET
            price: Limit price (required for limit orders)
            client_oid: Client order ID
            take_profit: Take profit price
            stop_loss: Stop loss price
            
        Returns:
            Order response with order_id
        """
        data = {
            "symbol": symbol,
            "size": str(size),
            "type": side.value,
            "order_type": order_type.value,
            "match_price": "1" if order_type == OrderType.MARKET else "0"
        }
        
        if order_type == OrderType.LIMIT and price:
            data["price"] = str(price)
        
        if client_oid:
            data["client_oid"] = client_oid
        
        if take_profit:
            data["presetTakeProfitPrice"] = str(take_profit)
        
        if stop_loss:
            data["presetStopLossPrice"] = str(stop_loss)
        
        app_logger.info(f"Placing order: {data}")
        
        result = await self._request("POST", "/capi/v2/order/placeOrder", data=data)
        return result.get("data", result)
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        client_oid: str = None
    ) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID
            client_oid: Client order ID
            
        Returns:
            Cancel response
        """
        data = {"symbol": symbol}
        if order_id:
            data["orderId"] = order_id
        if client_oid:
            data["clientOid"] = client_oid
        
        result = await self._request("POST", "/capi/v2/order/cancel", data=data)
        return result.get("data", {})
    
    async def cancel_all_orders(self, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Cancel response
        """
        data = {}
        if symbol:
            data["symbol"] = symbol
        
        result = await self._request("POST", "/capi/v2/order/cancelAll", data=data)
        return result.get("data", {})
    
    async def close_all_positions(self, symbol: str = None) -> Dict[str, Any]:
        """
        Close all positions.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Close response
        """
        data = {}
        if symbol:
            data["symbol"] = symbol
        
        result = await self._request("POST", "/capi/v2/order/closeAll", data=data)
        return result.get("data", {})
    
    async def get_order_info(
        self,
        symbol: str,
        order_id: str
    ) -> Dict[str, Any]:
        """
        Get order information.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            
        Returns:
            Order details
        """
        result = await self._request(
            "GET", "/capi/v2/order/detail",
            params={"symbol": symbol, "orderId": order_id}
        )
        return result.get("data", {})
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get current open orders.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        result = await self._request("GET", "/capi/v2/order/current", params=params)
        return result.get("data", [])
    
    async def get_order_history(
        self,
        symbol: str = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            symbol: Specific symbol or None for all
            page: Page number
            page_size: Items per page
            
        Returns:
            List of historical orders
        """
        params = {
            "pageNo": str(page),
            "pageSize": str(page_size)
        }
        if symbol:
            params["symbol"] = symbol
        
        result = await self._request("GET", "/capi/v2/order/history", params=params)
        return result.get("data", {}).get("resultList", [])
    
    async def get_fills(
        self,
        symbol: str = None,
        order_id: str = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get trade fills.
        
        Args:
            symbol: Specific symbol or None for all
            order_id: Specific order ID
            page: Page number
            page_size: Items per page
            
        Returns:
            List of fills
        """
        params = {
            "pageNo": str(page),
            "pageSize": str(page_size)
        }
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        
        result = await self._request("GET", "/capi/v2/order/fills", params=params)
        return result.get("data", {}).get("resultList", [])
    
    # ==================== AI LOG UPLOAD ====================
    
    async def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        explanation: str,
        order_id: str = None
    ) -> Dict[str, Any]:
        """
        Upload AI log for hackathon compliance.
        
        Required by WEEX to verify AI involvement in trading decisions.
        
        Args:
            stage: Decision stage (e.g., "Decision Making", "Strategy Generation")
            model: AI model name (e.g., "RegimeClassifier-v2.0")
            input_data: Model input (prompt, market data, features)
            output_data: Model output (signal, confidence, prediction)
            explanation: Human-readable explanation of the decision
            order_id: Associated order ID (if applicable)
            
        Returns:
            Upload response
        """
        data = {
            "stage": stage,
            "model": model,
            "input": input_data,
            "output": output_data,
            "explanation": explanation
        }
        
        if order_id:
            data["orderId"] = order_id
        
        app_logger.info(f"Uploading AI log: stage={stage}, model={model}")
        
        result = await self._request(
            "POST", "/capi/v2/order/uploadAiLog",
            data=data
        )
        return result
    
    async def generate_and_upload_ai_log(
        self,
        regime: str,
        regime_confidence: float,
        tradability_score: float,
        signal: str,
        signal_confidence: float,
        market_data: Dict[str, Any],
        reasoning: str,
        order_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate and upload a comprehensive AI log.
        
        Convenience method that formats data appropriately.
        
        Args:
            regime: Detected market regime (TREND/RANGE/HIGH-RISK)
            regime_confidence: Confidence in regime detection
            tradability_score: Tradability score (0-100)
            signal: Trade signal (BUY/SELL/HOLD)
            signal_confidence: Confidence in the signal
            market_data: Relevant market data snapshot
            reasoning: AI-generated reasoning explanation
            order_id: Associated order ID
            
        Returns:
            Upload response
        """
        input_data = {
            "prompt": "Analyze market conditions and generate trading signal",
            "data": {
                "market_data": market_data,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        output_data = {
            "regime": regime,
            "regime_confidence": regime_confidence,
            "tradability_score": tradability_score,
            "signal": signal,
            "signal_confidence": signal_confidence
        }
        
        return await self.upload_ai_log(
            stage="Decision Making",
            model="CryptoRegimeEngine-v2.0",
            input_data=input_data,
            output_data=output_data,
            explanation=reasoning,
            order_id=order_id
        )
    
    # ==================== UTILITIES ====================
    
    def is_competition_pair(self, symbol: str) -> bool:
        """Check if symbol is a valid competition trading pair."""
        return symbol.lower() in self.COMPETITION_PAIRS
    
    def normalize_symbol(self, asset: str) -> str:
        """
        Convert asset name to WEEX symbol format.
        
        Args:
            asset: Asset name (e.g., 'BTC', 'bitcoin')
            
        Returns:
            WEEX symbol (e.g., 'cmt_btcusdt')
        """
        asset_map = {
            "bitcoin": "btc", "btc": "btc",
            "ethereum": "eth", "eth": "eth",
            "solana": "sol", "sol": "sol",
            "cardano": "ada", "ada": "ada",
            "dogecoin": "doge", "doge": "doge",
            "ripple": "xrp", "xrp": "xrp",
            "litecoin": "ltc", "ltc": "ltc",
            "binancecoin": "bnb", "bnb": "bnb"
        }
        
        normalized = asset_map.get(asset.lower(), asset.lower())
        return f"cmt_{normalized}usdt"
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def ping(self) -> bool:
        """
        Check API connectivity.
        
        Returns:
            True if API is reachable
        """
        try:
            await self.get_server_time()
            return True
        except Exception as e:
            app_logger.error(f"WEEX API ping failed: {e}")
            return False


# Global client instance
weex_client = WEEXClient()
