import os
import logging
import base64
import hashlib
import json
import time
import uuid
import urllib.request
import urllib.error
from urllib.parse import urlsplit
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config.settings import settings

logger = logging.getLogger("ai_trading_bot.exness_api")


class ExnessApiConnector:
    """
    Direct Cloud REST & WebSocket API Connector for Exness Trader API (v1.0.2).
    Allows 100% platform-independent execution on Linux, Railway, Docker, or Cloud
    without requiring Windows or the MetaTrader 5 desktop client.
    """

    def __init__(self):
        self.api_key = settings.EXNESS_API_KEY
        self.api_secret = settings.EXNESS_API_SECRET
        self.account_id = str(settings.EXNESS_ACCOUNT_ID)
        self.gateway_url = getattr(settings, "EXNESS_GATEWAY_URL", "https://api.exness-api.com").rstrip("/")
        self.access_point: Optional[str] = None
        self._private_key: Optional[Ed25519PrivateKey] = None
        self._is_connected = False
        self._symbol_conditions_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_file = os.path.join(os.path.dirname(__file__), "..", "data", "exness_ap_cache.json")
        self._last_req_time: float = 0.0
        self._min_interval: float = 0.55  # 550ms ensures max ~1.8 req/sec, strictly below Exness 2 req/sec limit
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cached_snapshot_time: float = 0.0
        self._price_cache: Dict[str, Dict[str, Any]] = {}

    def _load_key(self) -> None:
        """Load Ed25519 private key from 128-hex character string or unencrypted PKCS#8 PEM."""
        if self._private_key is not None:
            return

        text = self.api_secret.strip()
        if text.startswith("-----BEGIN PRIVATE KEY-----"):
            key = serialization.load_pem_private_key(text.encode(), password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError("Expected an Ed25519 private key in PEM format")
            self._private_key = key
        else:
            raw = bytes.fromhex(text)
            self._private_key = Ed25519PrivateKey.from_private_bytes(raw[:32])

    @staticmethod
    def _b64url(raw: bytes) -> str:
        """Base64url encode without '=' padding as required by Exness API specification."""
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize symbol format between MT5 (e.g., BTCUSDm, XAUUSDm)
        and Exness Cloud API (e.g., BTCUSD, XAUUSD).
        """
        sym = symbol.upper().strip()
        if sym.endswith("M") and len(sym) in (7, 8):
            sym = sym[:-1]
        return sym

    def _sign_request(
        self,
        method: str,
        path_with_query: str,
        body_bytes: bytes = b"",
        idempotency_key: str = ""
    ) -> Dict[str, str]:
        """
        Generate strict Exness Trader API signed headers (v1).
        Requires: EXN-API-KEY, EXN-IDEMPOTENCY-KEY, EXN-TIMESTAMP, EXN-SIGN-VERSION, EXN-DATA, EXN-SIGN.
        """
        self._load_key()
        timestamp = int(time.time() * 1000)
        body_hash = self._b64url(hashlib.sha256(body_bytes).digest())

        payload_dict = {
            "api_key": self.api_key,
            "idempotency_key": idempotency_key,
            "timestamp": timestamp,
            "sign_version": 1,
            "method": method.upper(),
            "path": path_with_query,
            "body_hash": body_hash
        }

        payload_bytes = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        exn_data = self._b64url(payload_bytes)
        exn_sign = self._b64url(self._private_key.sign(payload_bytes))

        headers = {
            "EXN-API-KEY": self.api_key,
            "EXN-IDEMPOTENCY-KEY": idempotency_key,
            "EXN-TIMESTAMP": str(timestamp),
            "EXN-SIGN-VERSION": "1",
            "EXN-DATA": exn_data,
            "EXN-SIGN": exn_sign,
        }
        if body_bytes:
            headers["Content-Type"] = "application/json"

        return headers

    def _request(
        self,
        method: str,
        path_with_query: str,
        body: Optional[Dict[str, Any]] = None,
        use_access_point: bool = True,
        timeout: int = 15,
        retry: bool = False
    ) -> Tuple[bool, Any]:
        """Send authenticated signed HTTP request to Exness Gateway or Access Point with proactive rate-limiting."""
        # Enforce rate-limit pacing (max 1.8 req/sec)
        elapsed = time.time() - self._last_req_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_req_time = time.time()

        base = self.access_point if use_access_point and self.access_point else self.gateway_url
        url = base + path_with_query

        body_bytes = b""
        idempotency_key = ""
        if body is not None:
            body_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8")
            if method.upper() in ("POST", "PUT", "PATCH"):
                idempotency_key = str(uuid.uuid4())

        headers = self._sign_request(method, path_with_query, body_bytes, idempotency_key)
        req = urllib.request.Request(url, data=body_bytes if body_bytes else None, headers=headers, method=method.upper())

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return True, {}
                return True, json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429 and not retry:
                logger.warning(f"[Exness API] 429 Rate Limit encountered. Cooling down 1.2s and retrying...")
                time.sleep(1.2)
                return self._request(method, path_with_query, body, use_access_point, timeout, retry=True)
            logger.error(f"[Exness API] HTTP {e.code} on {method} {path_with_query}: {err_body}")
            return False, {"error": err_body, "status_code": e.code}
        except Exception as e:
            logger.error(f"[Exness API] Connection error on {method} {path_with_query}: {e}")
            return False, {"error": str(e)}

    def _get_snapshot(self, max_age_seconds: float = 3.0) -> Tuple[bool, Any]:
        """Fetch trading state snapshot with in-memory caching to respect 2 req/sec limit."""
        now = time.time()
        if self._cached_snapshot and (now - self._cached_snapshot_time < max_age_seconds):
            return True, self._cached_snapshot

        path = f"/v1/trading/accounts/{self.account_id}/snapshot"
        ok, data = self._request("GET", path, use_access_point=True)
        if ok and "account_state" in data:
            self._cached_snapshot = data
            self._cached_snapshot_time = now
        return ok, data

    def _load_cached_access_point(self) -> Optional[str]:
        """Load cached Access Point URL to avoid rate-limiting the discovery endpoint."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "r") as f:
                    data = json.load(f)
                    if data.get("account_id") == self.account_id and data.get("access_point"):
                        return data["access_point"]
        except Exception:
            pass
        return None

    def _save_cached_access_point(self, ap: str) -> None:
        """Persist resolved Access Point URL to disk cache."""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump({"account_id": self.account_id, "access_point": ap}, f)
        except Exception:
            pass

    def connect(self) -> bool:
        """
        Resolve the Account Access Point via Gateway Discovery (with disk cache) and verify connection.
        Endpoint: GET /v1/trading/access-point?account_id={account_id}
        """
        if not self.api_key or not self.api_secret or not self.account_id:
            logger.error("[Exness API] Missing EXNESS_API_KEY, EXNESS_API_SECRET, or EXNESS_ACCOUNT_ID.")
            return False

        # 1. Try disk cache first to minimize hits to discovery endpoint
        cached_ap = self._load_cached_access_point()
        if cached_ap:
            self.access_point = cached_ap
            logger.info(f"[Exness API] Using cached Access Point: {self.access_point}")
        else:
            logger.info(f"[Exness API] Resolving Access Point via Exness Gateway ({self.gateway_url})...")
            discovery_path = f"/v1/trading/access-point?account_id={self.account_id}"
            success, data = self._request("GET", discovery_path, use_access_point=False)
            if success and "access_point" in data:
                ap = data["access_point"]
                url_parsed = urlsplit(ap)
                self.access_point = f"https://{url_parsed.netloc}"
                self._save_cached_access_point(self.access_point)
                logger.info(f"[Exness API] Access Point resolved and cached: {self.access_point}")
            else:
                logger.warning(f"[Exness API] Access Point discovery failed ({data}). Trying standard fallback.")
                self.access_point = f"https://ap-e3b37595.exness-api.com"

        # 2. Verify Account Configuration
        acc_path = f"/v1/configuration/accounts/{self.account_id}/account"
        acc_ok, acc_data = self._request("GET", acc_path, use_access_point=True)
        if not acc_ok:
            logger.error(f"[Exness API] Failed to fetch account configuration: {acc_data}")
            return False

        self._is_connected = True
        logger.info(
            f"[Exness API] Connected! Account: {acc_data.get('account_id')} "
            f"('{acc_data.get('account_name', 'Trading')}'), "
            f"Demo: {acc_data.get('is_demo')}, "
            f"Currency: {acc_data.get('settings', {}).get('currency')}, "
            f"Leverage: {acc_data.get('settings', {}).get('leverage')}"
        )
        return True

    def disconnect(self) -> None:
        """Close connection session cleanly."""
        self._is_connected = False
        logger.info("[Exness API] Disconnected.")

    def is_connected(self) -> bool:
        return self._is_connected

    def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch real-time snapshot of account balance, equity, margin, and leverage.
        Endpoint: GET /v1/trading/accounts/{account_id}/snapshot
        """
        if not self._is_connected:
            self.connect()

        ok, data = self._get_snapshot()
        if not ok or "account_state" not in data:
            logger.warning(f"[Exness API] Failed to get account snapshot: {data}")
            return {
                "login": self.account_id,
                "server": "Exness Cloud API",
                "balance": 0.0,
                "equity": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "profit": 0.0,
                "currency": "USD",
                "leverage": 2000
            }

        state = data["account_state"]
        balance = float(state.get("balance", 0.0))
        equity = float(state.get("equity", balance))
        used_margin = float(state.get("used_margin", 0.0))
        leverage = int(state.get("leverage", 2000))

        return {
            "login": self.account_id,
            "server": "Exness Cloud API",
            "balance": balance,
            "equity": equity,
            "margin": used_margin,
            "free_margin": equity - used_margin,
            "profit": equity - balance,
            "currency": "USD",
            "leverage": leverage
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch instrument specs (digits, contract size, point, spread, lot boundaries).
        Compliant with MT5Connector.get_symbol_info return structure.
        """
        inst = self.normalize_symbol(symbol)
        is_crypto = "BTC" in inst
        is_gold = "XAU" in inst or "GOLD" in inst
        is_jpy = "JPY" in inst

        if is_crypto:
            digits = 2
            point = 0.01
            contract_size = 1.0
            default_spread = 2500
        elif is_gold:
            digits = 2
            point = 0.01
            contract_size = 100.0
            default_spread = 35
        elif is_jpy:
            digits = 3
            point = 0.001
            contract_size = 100000.0
            default_spread = 15
        else:
            digits = 5
            point = 0.00001
            contract_size = 100000.0
            default_spread = 15

        return {
            "name": symbol,
            "symbol": symbol,
            "digits": digits,
            "point": point,
            "tick_size": point,
            "tick_value": 1.0 if not is_jpy else 0.65,
            "trade_tick_size": point,
            "trade_tick_value": 1.0 if not is_jpy else 0.65,
            "contract_size": contract_size,
            "volume_min": 0.01,
            "volume_max": 200.0,
            "volume_step": 0.01,
            "spread": default_spread
        }

    def fetch_rates(self, symbol: str, timeframe_str: str = "M15", count: int = 300) -> pd.DataFrame:
        """
        Fetch historical OHLCV candle rates as a Pandas DataFrame.
        Endpoint: GET /v1/market-data/accounts/{account_id}/candles
        """
        inst = self.normalize_symbol(symbol)
        tf = timeframe_str.upper().strip()
        if tf not in ("M15", "H1", "M5", "M30", "H4", "D1"):
            tf = "M15"

        now = datetime.now(timezone.utc)
        days_back = 7 if tf == "M15" else 22
        from_time = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        path = f"/v1/market-data/accounts/{self.account_id}/candles?from={from_time}&instrument={inst}&price_type=bid&timeframe={tf}&to={to_time}"
        ok, data = self._request("GET", path, use_access_point=True)
        if not ok or "candles" not in data:
            logger.error(f"[Exness API] Failed to fetch candles for {inst}: {data}")
            return pd.DataFrame()

        candles = data["candles"]
        if not candles:
            logger.warning(f"[Exness API] No candles returned for {inst}")
            return pd.DataFrame()

        records = []
        for c in candles:
            records.append({
                "time": pd.to_datetime(c["time"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "tick_volume": float(c.get("volume", 0.0)),
                "spread": 15
            })

        df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
        return df.tail(count).reset_index(drop=True)

    def get_rates(self, symbol: str, timeframe_str: str, count: int = 200) -> Optional[pd.DataFrame]:
        """Convenience alias for fetch_rates."""
        df = self.fetch_rates(symbol, timeframe_str, count)
        return df if not df.empty else None

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        comment: str = "AI-Trading-Bot",
        magic: int = 888123
    ) -> Dict[str, Any]:
        """
        Execute market order compliant with MT5Connector.execute_order signature.
        """
        return self.open_position(
            symbol=symbol,
            action=order_type,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=comment
        )

    def open_position(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI Trading"
    ) -> Dict[str, Any]:
        """
        Open immediate market position via Exness Trader API.
        Endpoint: POST /v1/trading/accounts/{account_id}/positions
        """
        inst = self.normalize_symbol(symbol)
        side = "buy" if action.upper() == "BUY" else "sell"
        norm_vol = f"{volume:.2f}"

        body = {
            "instrument": inst,
            "side": side,
            "volume": norm_vol,
            "comment": comment[:31]  # Exness max comment length
        }
        if sl is not None and sl > 0:
            body["stop_loss_price"] = f"{sl:.5f}" if "EUR" in inst else f"{sl:.2f}"
        if tp is not None and tp > 0:
            body["take_profit_price"] = f"{tp:.5f}" if "EUR" in inst else f"{tp:.2f}"

        path = f"/v1/trading/accounts/{self.account_id}/positions"
        logger.info(f"[Exness API] Executing Market {action} {norm_vol} lot on {inst} (SL: {sl}, TP: {tp})...")

        ok, data = self._request("POST", path, body=body, use_access_point=True)
        if not ok:
            err_msg = data.get("error", "Unknown error")
            logger.error(f"[Exness API] Failed to open position: {err_msg}")
            return {
                "success": False,
                "error": str(err_msg),
                "retcode": 500,
                "comment": str(err_msg)
            }

        pos_id = data.get("position_id")
        exec_price = float(data.get("open_price", data.get("price", 0.0)))
        logger.info(f"[Exness API] Position opened successfully! Position #{pos_id} @ {exec_price}")

        return {
            "success": True,
            "ticket": int(pos_id),
            "price": exec_price,
            "volume": float(norm_vol),
            "retcode": 0,
            "comment": "Position opened via Exness API"
        }

    def get_symbol_price(self, symbol: str) -> float:
        """
        Fetch current market bid price using latest candle with 5-second caching.
        """
        inst = self.normalize_symbol(symbol)
        now = time.time()
        cached = getattr(self, "_price_cache", {}).get(inst)
        if cached and (now - cached["time"]) < 5.0:
            return cached["price"]

        try:
            df = self.fetch_rates(inst, "M15", 2)
            if not df.empty and len(df) > 0:
                p = float(df.iloc[-1]["close"])
                if not hasattr(self, "_price_cache"):
                    self._price_cache = {}
                self._price_cache[inst] = {"price": p, "time": now}
                return p
        except Exception as e:
            logger.warning(f"[Exness API] Error fetching symbol price for {inst}: {e}")

        return 0.0

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active open positions mapped to standard MT5 dictionary format.
        Calculates live floating PnL and real-time market prices dynamically.
        Endpoint: GET /v1/trading/accounts/{account_id}/snapshot
        """
        ok, data = self._get_snapshot()
        if not ok or "positions" not in data or not data["positions"]:
            return []

        orders_by_pos = {}
        for o in data.get("orders", []):
            pos_ref = o.get("position_id")
            if pos_ref:
                orders_by_pos[str(pos_ref)] = o

        state = data.get("account_state", {})
        bal = float(state.get("balance", 0.0))
        eq = float(state.get("equity", bal))
        total_pnl = round(eq - bal, 2)

        raw_positions = data["positions"]
        num_positions = len(raw_positions)

        open_list = []
        for p in raw_positions:
            pos_id = str(p.get("position_id"))
            inst = p.get("instrument", "")
            direction = p.get("direction", "buy").upper()
            vol = float(p.get("volume", 0.01))
            open_p = float(p.get("open_price", 0.0))

            spec = self.get_symbol_info(inst)
            c_size = spec.get("contract_size", 1.0) if spec else 1.0

            # Dynamic floating PnL and current price calculation
            if num_positions == 1:
                # With 1 position, total unrealized PnL from Exness account equity is exact
                profit_val = total_pnl
                if (vol * c_size) > 0:
                    if direction == "BUY":
                        current_p = open_p + (profit_val / (vol * c_size))
                    else:
                        current_p = open_p - (profit_val / (vol * c_size))
                else:
                    current_p = open_p
            else:
                # With multiple positions, evaluate market price for each symbol
                market_p = self.get_symbol_price(inst)
                if market_p > 0:
                    current_p = market_p
                    mult = 1.0 if direction == "BUY" else -1.0
                    raw_diff = mult * (current_p - open_p)
                    if "JPY" in inst:
                        profit_val = (raw_diff * vol * c_size) / current_p
                    else:
                        profit_val = raw_diff * vol * c_size
                    profit_val = round(profit_val, 2)
                else:
                    current_p = open_p
                    profit_val = 0.0

            # Retrieve SL/TP from attached order if present
            att_order = orders_by_pos.get(pos_id, {})
            sl_val = float(att_order.get("stop_loss_price") or 0.0)
            tp_val = float(att_order.get("take_profit_price") or 0.0)

            # Match symbol format with or without 'm' suffix
            mapped_sym = inst + "m" if not inst.endswith("m") else inst

            open_list.append({
                "ticket": int(pos_id),
                "symbol": mapped_sym,
                "type": direction,
                "volume": vol,
                "price_open": open_p,
                "sl": sl_val,
                "tp": tp_val,
                "price_current": round(current_p, 5 if "EUR" in inst else 2),
                "profit": profit_val,
                "time": p.get("create_time", "")
            })

        if symbol:
            sym_clean = self.normalize_symbol(symbol)
            open_list = [p for p in open_list if self.normalize_symbol(p["symbol"]) == sym_clean]

        return open_list

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """
        Update Stop Loss price for Break-Even trailing.
        Endpoint: PATCH /v1/trading/accounts/{account_id}/orders/{order_id}
        """
        path = f"/v1/trading/accounts/{self.account_id}/snapshot"
        ok, data = self._request("GET", path, use_access_point=True)
        if not ok:
            return False

        order_id = None
        for o in data.get("orders", []):
            if str(o.get("position_id")) == str(ticket):
                order_id = o.get("order_id")
                break

        if not order_id:
            logger.warning(f"[Exness API] No active order found for position #{ticket} to modify SL.")
            return False

        patch_path = f"/v1/trading/accounts/{self.account_id}/orders/{order_id}"
        body = {
            "stop_loss_price": f"{new_sl:.5f}" if new_sl < 50 else f"{new_sl:.2f}"
        }
        patch_ok, patch_data = self._request("PATCH", patch_path, body=body, use_access_point=True)
        if patch_ok:
            logger.info(f"[Exness API] SL modified successfully for Position #{ticket} -> {new_sl}")
            return True

        logger.error(f"[Exness API] Failed to modify SL for Position #{ticket}: {patch_data}")
        return False

    def close_partial_position(self, ticket: int, volume: float) -> Dict[str, Any]:
        """
        Partially close position (Partial TP 50%).
        Endpoint: DELETE /v1/trading/accounts/{account_id}/positions/{position_id}?volume={volume}
        """
        norm_vol = f"{volume:.2f}"
        path = f"/v1/trading/accounts/{self.account_id}/positions/{ticket}?volume={norm_vol}"
        logger.info(f"[Exness API] Closing partial position #{ticket}: {norm_vol} lot...")

        ok, data = self._request("DELETE", path, use_access_point=True)
        if ok:
            logger.info(f"[Exness API] Partial close succeeded for #{ticket} ({norm_vol} lot closed).")
            return {
                "success": True,
                "closed_volume": volume,
                "price": float(data.get("price", 0.0))
            }

        logger.error(f"[Exness API] Partial close failed for #{ticket}: {data}")
        return {"success": False, "error": str(data)}

    def close_position(self, ticket: int) -> bool:
        """
        Fully close position at current market price.
        Endpoint: DELETE /v1/trading/accounts/{account_id}/positions/{position_id}
        """
        path = f"/v1/trading/accounts/{self.account_id}/positions/{ticket}"
        logger.info(f"[Exness API] Fully closing position #{ticket}...")

        ok, data = self._request("DELETE", path, use_access_point=True)
        if ok:
            logger.info(f"[Exness API] Position #{ticket} fully closed.")
            return True

        logger.error(f"[Exness API] Failed to close position #{ticket}: {data}")
        return False

    def close_all_positions(self) -> int:
        """Emergency Panic Button: close all open positions immediately."""
        positions = self.get_open_positions()
        closed_count = 0
        for p in positions:
            if self.close_position(p["ticket"]):
                closed_count += 1
        logger.warning(f"[Exness API] Panic close completed: {closed_count}/{len(positions)} closed.")
        return closed_count

    def get_closed_deal_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """
        Query deal history to determine exit price, realized profit, and exit reason.
        Endpoint: GET /v1/history/accounts/{account_id}/deals
        """
        path = f"/v1/history/accounts/{self.account_id}/deals?position_id={ticket}"
        ok, data = self._request("GET", path, use_access_point=True)
        if not ok or "deals" not in data:
            return None

        deals = data["deals"]
        if not deals:
            return None

        out_deals = [d for d in deals if d.get("entry") in ("out", "close", "DEAL_ENTRY_OUT")]
        if not out_deals:
            out_deals = deals[-1:]

        total_profit = sum(float(d.get("profit", 0.0)) for d in out_deals)
        final_deal = out_deals[-1]
        exit_price = float(final_deal.get("price", 0.0))

        if abs(total_profit) <= 0.05:
            reason_str = "BREAK_EVEN"
        elif total_profit > 0.05:
            reason_str = "TAKE_PROFIT"
        else:
            reason_str = "STOP_LOSS"

        return {
            "price": exit_price,
            "profit": round(total_profit, 2),
            "reason": reason_str,
            "time": final_deal.get("time", "")
        }
