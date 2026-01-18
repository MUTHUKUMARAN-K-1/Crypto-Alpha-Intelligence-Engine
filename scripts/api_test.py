"""
WEEX API Testing Scripts for AI Wars Hackathon.
These scripts match the official hackathon requirements exactly.

Run these to pass the API testing phase:
1. test_connection() - Check API connectivity
2. test_account_balance() - Check account balance
3. test_get_price() - Get BTC price
4. test_set_leverage() - Set leverage to 1x
5. test_place_order() - Place a ~10 USDT order on cmt_btcusdt
6. test_get_fills() - Get trade details

After passing all tests, you will receive your competition funds.
"""

import time
import hmac
import hashlib
import base64
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Credentials - Fill these in after receiving from hackathon
API_KEY = os.getenv("WEEX_API_KEY", "")
SECRET_KEY = os.getenv("WEEX_SECRET_KEY", "")
ACCESS_PASSPHRASE = os.getenv("WEEX_PASSPHRASE", "")
BASE_URL = "https://api-contract.weex.com"


def generate_signature_get(secret_key: str, timestamp: str, method: str, request_path: str, query_string: str) -> str:
    """Generate signature for GET requests."""
    message = timestamp + method.upper() + request_path + query_string
    signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def generate_signature_post(secret_key: str, timestamp: str, method: str, request_path: str, query_string: str, body: str) -> str:
    """Generate signature for POST requests."""
    message = timestamp + method.upper() + request_path + query_string + body
    signature = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def send_request_get(request_path: str, query_string: str = "", authenticated: bool = True):
    """Send authenticated GET request."""
    timestamp = str(int(time.time() * 1000))
    
    headers = {
        "Content-Type": "application/json",
        "locale": "en-US"
    }
    
    if authenticated:
        signature = generate_signature_get(SECRET_KEY, timestamp, "GET", request_path, query_string)
        headers.update({
            "ACCESS-KEY": API_KEY,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": ACCESS_PASSPHRASE,
        })
    
    url = BASE_URL + request_path + query_string
    response = requests.get(url, headers=headers)
    return response


def send_request_post(request_path: str, body: dict, query_string: str = ""):
    """Send authenticated POST request."""
    timestamp = str(int(time.time() * 1000))
    body_str = json.dumps(body)
    
    signature = generate_signature_post(SECRET_KEY, timestamp, "POST", request_path, query_string, body_str)
    
    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": ACCESS_PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US"
    }
    
    url = BASE_URL + request_path
    response = requests.post(url, headers=headers, data=body_str)
    return response


# ==================== TEST FUNCTIONS ====================

def test_connection():
    """Test 1: Check API connectivity (no authentication required)."""
    print("\n" + "="*50)
    print("TEST 1: API Connection")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/capi/v2/market/time")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("[OK] Connection successful!")
        return True
    else:
        print("[FAIL] Connection failed!")
        return False


def test_account_balance():
    """Test 2: Check account balance."""
    print("\n" + "="*50)
    print("TEST 2: Account Balance")
    print("="*50)
    
    if not API_KEY:
        print("❌ API_KEY not set in .env file")
        return False
    
    response = send_request_get("/capi/v2/account/assets")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("[OK] Account balance retrieved!")
        return True
    else:
        print("[FAIL] Failed to get balance. Check your credentials.")
        return False


def test_get_price():
    """Test 3: Get BTC price (no authentication required)."""
    print("\n" + "="*50)
    print("TEST 3: Get BTC Price")
    print("="*50)
    
    response = send_request_get("/capi/v2/market/ticker", "?symbol=cmt_btcusdt", authenticated=False)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        price = data.get("last", "N/A")
        print(f"[OK] BTC Price: ${price}")
        return True
    else:
        print("❌ Failed to get price")
        return False


def test_set_leverage():
    """Test 4: Set leverage to 1x for cmt_btcusdt."""
    print("\n" + "="*50)
    print("TEST 4: Set Leverage (1x)")
    print("="*50)
    
    if not API_KEY:
        print("❌ API_KEY not set in .env file")
        return False
    
    body = {
        "symbol": "cmt_btcusdt",
        "marginMode": 1,  # Cross margin
        "longLeverage": "1",
        "shortLeverage": "1"
    }
    
    response = send_request_post("/capi/v2/account/leverage", body)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("[OK] Leverage set to 1x!")
        return True
    else:
        print("[FAIL] Failed to set leverage")
        return False


def test_place_order():
    """
    Test 5: Place a ~10 USDT order on cmt_btcusdt.
    This is the REQUIRED test for hackathon qualification!
    """
    print("\n" + "="*50)
    print("TEST 5: Place Order (~10 USDT on cmt_btcusdt)")
    print("="*50)
    
    if not API_KEY:
        print("❌ API_KEY not set in .env file")
        return False, None
    
    # First get current BTC price
    price_response = send_request_get("/capi/v2/market/ticker", "?symbol=cmt_btcusdt", authenticated=False)
    if price_response.status_code != 200:
        print("❌ Failed to get price")
        return False, None
    
    price_data = price_response.json()
    current_price = float(price_data.get("last", 95000))
    
    # Calculate size for ~10 USDT
    # contract_val = 0.0001 BTC per contract
    # size should be in BTC
    target_value = 10  # USDT
    size = target_value / current_price  # BTC amount
    size = round(size, 4)  # Round to contract precision
    size = max(0.0001, size)  # Minimum order size
    
    # Set limit price slightly below market (so it fills)
    limit_price = round(current_price * 1.01, 1)  # 1% above current
    
    print(f"Current BTC Price: ${current_price}")
    print(f"Order Size: {size} BTC (~${size * current_price:.2f})")
    print(f"Limit Price: ${limit_price}")
    
    body = {
        "symbol": "cmt_btcusdt",
        "client_oid": f"hackathon_test_{int(time.time())}",
        "size": str(size),
        "type": "1",  # 1 = Open long
        "order_type": "0",  # 0 = Limit order
        "match_price": "0",  # 0 = Use specified price
        "price": str(limit_price)
    }
    
    print(f"Order body: {json.dumps(body, indent=2)}")
    
    response = send_request_post("/capi/v2/order/placeOrder", body)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        order_id = result.get("order_id")
        print(f"✅ Order placed! Order ID: {order_id}")
        return True, order_id
    else:
        print("❌ Failed to place order")
        return False, None


def test_get_fills(order_id: str = None):
    """Test 6: Get trade details for completed orders."""
    print("\n" + "="*50)
    print("TEST 6: Get Trade Fills")
    print("="*50)
    
    if not API_KEY:
        print("❌ API_KEY not set in .env file")
        return False
    
    query = "?symbol=cmt_btcusdt"
    if order_id:
        query += f"&orderId={order_id}"
    
    response = send_request_get("/capi/v2/order/fills", query)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Fills retrieved!")
        return True
    else:
        print("❌ Failed to get fills")
        return False


def run_all_tests():
    """Run all hackathon API tests."""
    print("\n" + "="*60)
    print("  WEEX AI WARS HACKATHON - API TESTING")
    print("="*60)
    
    if not API_KEY or not SECRET_KEY or not ACCESS_PASSPHRASE:
        print("\n[!] API credentials not found in .env file!")
        print("Please add your credentials after receiving them from hackathon:")
        print("  WEEX_API_KEY=your_api_key")
        print("  WEEX_SECRET_KEY=your_secret_key")
        print("  WEEX_PASSPHRASE=your_passphrase")
        print("\nRunning connection test only...\n")
        test_connection()
        return
    
    results = []
    
    # Run all tests
    results.append(("Connection", test_connection()))
    results.append(("Account Balance", test_account_balance()))
    results.append(("Get Price", test_get_price()))
    results.append(("Set Leverage", test_set_leverage()))
    
    success, order_id = test_place_order()
    results.append(("Place Order", success))
    
    if order_id:
        results.append(("Get Fills", test_get_fills(order_id)))
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")
    
    if passed_count == len(results):
        print("\n🎉 All tests passed! You're ready for the competition!")
    else:
        print("\n⚠️  Some tests failed. Please check your credentials and try again.")


if __name__ == "__main__":
    run_all_tests()
