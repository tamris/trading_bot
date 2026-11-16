import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger("ai_trading_bot.mt5")


class MT5Connector:
    """Handles communication, data extraction, and trade execution with MetaTrader 5."""

    TIMEFRAME_MAP = {}
    if MT5_AVAILABLE:
        TIMEFRAME_MAP = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

    def __init__(self, mock_mode: Optional[bool] = None):
        self.mock_mode = settings.MT5_MOCK_MODE if mock_mode is None else mock_mode
        self._connected = False

        # In-memory mock storage for testing and dry-run
        self._mock_balance = 10000.0
        self._mock_equity = 10000.0
        self._mock_positions: List[Dict[str, Any]] = []
        self._mock_ticket_counter = 100000

    def connect(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal or start mock mode."""
        if self.mock_mode or not MT5_AVAILABLE:
            logger.info("MT5 Connector running in SIMULATION / MOCK MODE.")
            self._connected = True
            return True

        logger.info(f"Connecting to MT5 terminal at {settings.MT5_PATH}...")
        init_ok = mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN if settings.MT5_LOGIN > 0 else None,
            password=settings.MT5_PASSWORD if settings.MT5_PASSWORD else None,
            server=settings.MT5_SERVER if settings.MT5_SERVER else None,
            timeout=30000
        )

        if not init_ok:
            error = mt5.last_error()
            logger.warning(f"MT5 initialize failed: {error}. Falling back to MOCK MODE.")
            self.mock_mode = True
            self._connected = True
            return True

        # Perform login if credentials supplied
        if settings.MT5_LOGIN > 0 and settings.MT5_PASSWORD:
            authorized = mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            if not authorized:
                error = mt5.last_error()
                logger.error(f"MT5 login failed for account {settings.MT5_LOGIN}: {error}. Switching to Mock Mode.")
                self.mock_mode = True
                self._connected = True
                return True

        account_info = mt5.account_info()
        if account_info is not None:
            logger.info(
                f"Successfully connected to MT5. Account: {account_info.login}, "
                f"Server: {account_info.server}, Balance: {account_info.balance}, Equity: {account_info.equity}"
            )
        else:
            logger.info("Connected to MT5 terminal (demo/unauthorized mode).")

        self._connected = True
        return True

    def disconnect(self):
        """Shutdown MT5 terminal connection."""
        if not self.mock_mode and MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 Connector disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> Dict[str, Any]:
        """Fetch current balance, equity, margin, and profit."""
        if self.mock_mode or not MT5_AVAILABLE:
            floating_pnl = sum(p.get("profit", 0.0) for p in self._mock_positions)
            return {
                "login": 99999999,
                "server": "Mock-Server",
                "balance": self._mock_balance,
                "equity": self._mock_balance + floating_pnl,
                "margin": 0.0,
                "free_margin": self._mock_balance + floating_pnl,
                "profit": floating_pnl,
                "currency": "USD",
                "leverage": 100,
            }

        acc = mt5.account_info()
        if acc is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return {}

        return {
            "login": acc.login,
            "server": acc.server,
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "profit": acc.profit,
            "currency": acc.currency,
            "leverage": acc.leverage,
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve symbol trading specifications (tick value, tick size, lot limits, etc.)."""
        if self.mock_mode or not MT5_AVAILABLE:
            is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
            return {
                "symbol": symbol,
                "point": 0.01 if is_gold else 0.00001,
                "digits": 2 if is_gold else 5,
                "tick_size": 0.01 if is_gold else 0.00001,
                "tick_value": 1.0 if is_gold else 1.0,
                "contract_size": 100.0 if is_gold else 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "bid": 2650.50 if is_gold else 1.08500,
                "ask": 2650.80 if is_gold else 1.08515,
                "spread": 30 if is_gold else 15,
            }

        # Ensure symbol is visible in MarketWatch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol} in MarketWatch: {mt5.last_error()}")

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found in MT5: {mt5.last_error()}")
            return None

        tick = mt5.symbol_info_tick(symbol)
        bid = tick.bid if tick else 0.0
        ask = tick.ask if tick else 0.0

        return {
            "symbol": info.name,
            "point": info.point,
            "digits": info.digits,
            "tick_size": info.trade_tick_size or info.point,
            "tick_value": info.trade_tick_value or 1.0,
            "contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "bid": bid,
            "ask": ask,
            "spread": info.spread,
        }

    def fetch_rates(self, symbol: str, timeframe_str: str = "M15", count: int = 300) -> pd.DataFrame:
        """Fetch historical OHLCV candle rates as a Pandas DataFrame."""
        if self.mock_mode or not MT5_AVAILABLE:
            return self._generate_mock_rates(symbol, timeframe_str, count)

        tf = self.TIMEFRAME_MAP.get(timeframe_str.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"Empty rates returned for {symbol} ({timeframe_str}): {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all currently open positions."""
        if self.mock_mode or not MT5_AVAILABLE:
            if symbol:
                return [p for p in self._mock_positions if p["symbol"].upper() == symbol.upper()]
            return list(self._mock_positions)

        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []

        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "price_current": p.price_current,
                "profit": p.profit,
                "magic": p.magic,
                "comment": p.comment,
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc),
            })
        return result

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI-Trading-Bot",
        magic: int = 888123
    ) -> Dict[str, Any]:
        """Send a market order to MT5 with dynamic filling type fallback."""
        order_type_upper = order_type.upper()
        if order_type_upper not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid order type: {order_type}")

        if self.mock_mode or not MT5_AVAILABLE:
            self._mock_ticket_counter += 1
            info = self.get_symbol_info(symbol)
            exec_price = price or (info["ask"] if order_type_upper == "BUY" else info["bid"])
            mock_order = {
                "ticket": self._mock_ticket_counter,
                "symbol": symbol,
                "type": order_type_upper,
                "volume": volume,
                "price_open": exec_price,
                "sl": sl or 0.0,
                "tp": tp or 0.0,
                "price_current": exec_price,
                "profit": 0.0,
                "magic": magic,
                "comment": comment,
                "time": datetime.now(timezone.utc),
            }
            self._mock_positions.append(mock_order)
            logger.info(f"[MOCK ORDER EXECUTED] #{mock_order['ticket']} {order_type_upper} {volume} {symbol} @ {exec_price} SL:{sl} TP:{tp}")
            return {
                "success": True,
                "ticket": mock_order["ticket"],
                "price": exec_price,
                "volume": volume,
                "comment": "Mock execution succeeded"
            }

        sym_info = self.get_symbol_info(symbol)
        if not sym_info:
            return {"success": False, "error": f"Symbol info unavailable for {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "error": f"Failed to get tick for {symbol}"}

        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type_upper == "BUY" else mt5.ORDER_TYPE_SELL
        exec_price = tick.ask if order_type_upper == "BUY" else tick.bid

        # Filling type fallback sequence (IOC -> FOK -> RETURN)
        filling_types = [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN
        ]

        last_error = None
        for filling in filling_types:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": exec_price,
                "sl": sl if sl else 0.0,
                "tp": tp if tp else 0.0,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[ORDER EXECUTED] Ticket #{result.order} {order_type_upper} {volume} {symbol} @ {result.price}")
                return {
                    "success": True,
                    "ticket": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "retcode": result.retcode,
                    "comment": result.comment
                }
            else:
                last_error = result.comment if result else str(mt5.last_error())
                logger.warning(f"order_send failed with filling {filling}: {last_error}")
                if "Market closed" in str(last_error):
                    break

        return {"success": False, "error": f"All filling modes failed: {last_error}"}

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Update Stop Loss of an open position (e.g. Break-Even Point)."""
        if self.mock_mode or not MT5_AVAILABLE:
            for p in self._mock_positions:
                if p["ticket"] == ticket:
                    p["sl"] = new_sl
                    logger.info(f"[MOCK SL MODIFIED] Ticket #{ticket} new SL: {new_sl}")
                    return True
            return False

        position = None
        for p in mt5.positions_get():
            if p.ticket == ticket:
                position = p
                break

        if not position:
            logger.error(f"Position #{ticket} not found for SL modification.")
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": position.tp,
        }
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position #{ticket} SL updated to {new_sl}")
            return True

        logger.error(f"Failed to modify SL for #{ticket}: {res.comment if res else mt5.last_error()}")
        return False

    def close_position(self, ticket: int) -> bool:
        """Close an individual position by ticket."""
        if self.mock_mode or not MT5_AVAILABLE:
            self._mock_positions = [p for p in self._mock_positions if p["ticket"] != ticket]
            logger.info(f"[MOCK POSITION CLOSED] Ticket #{ticket}")
            return True

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position #{ticket} not found to close.")
            return False

        p = positions[0]
        sym_info = self.get_symbol_info(p.symbol)
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            return False

        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_type,
            "position": p.ticket,
            "price": price,
            "deviation": 20,
            "magic": p.magic,
            "comment": "Panic Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)

    def close_partial_position(self, ticket: int, volume: float) -> Dict[str, Any]:
        """Close partial volume of an active position (Partial Take Profit)."""
        if self.mock_mode or not MT5_AVAILABLE:
            for p in self._mock_positions:
                if p["ticket"] == ticket:
                    old_vol = p["volume"]
                    p["volume"] = max(0.01, round(old_vol - volume, 2))
                    logger.info(f"[MOCK PARTIAL CLOSE] Ticket #{ticket}: {volume} closed, {p['volume']} remaining")
                    return {"success": True, "closed_volume": volume, "remaining_volume": p["volume"], "price": p["price_current"]}
            return {"success": False, "error": f"Mock ticket #{ticket} not found"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position #{ticket} not found for partial close.")
            return {"success": False, "error": f"Position #{ticket} not found"}

        p = positions[0]
        sym_info = self.get_symbol_info(p.symbol)
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick or not sym_info:
            return {"success": False, "error": "Symbol tick unavailable"}

        vol_min = float(sym_info.get("volume_min") or 0.01)
        vol_step = float(sym_info.get("volume_step") or 0.01)

        import math
        steps = math.floor(round(volume / vol_step, 6))
        norm_vol = round(steps * vol_step, 2)

        if norm_vol < vol_min:
            return {"success": False, "error": f"Partial volume {norm_vol} < minimum {vol_min}"}

        remaining_vol = round(p.volume - norm_vol, 2)
        if remaining_vol < vol_min:
            return {"success": False, "error": f"Remaining volume {remaining_vol} < minimum {vol_min}"}

        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": norm_vol,
            "type": close_type,
            "position": p.ticket,
            "price": price,
            "deviation": 20,
            "magic": p.magic,
            "comment": "Partial TP (50%)",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[PARTIAL TP CLOSED] Ticket #{ticket}: {norm_vol} lot @ {price}. Remaining: {remaining_vol} lot.")
            return {
                "success": True,
                "closed_volume": norm_vol,
                "remaining_volume": remaining_vol,
                "price": price,
                "ticket": ticket
            }
        else:
            err = res.comment if res else str(mt5.last_error())
            logger.warning(f"Partial close failed for #{ticket}: {err}")
            return {"success": False, "error": err}

    def close_all_positions(self) -> int:
        """Emergency Panic Button: Close all active positions."""
        positions = self.get_open_positions()
        closed_count = 0
        for p in positions:
            if self.close_position(p["ticket"]):
                closed_count += 1
        logger.warning(f"Panic close completed: {closed_count}/{len(positions)} positions closed.")
        return closed_count

    def get_closed_deal_info(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Fetch close deal history from MT5 to determine exit price, profit, and exit reason."""
        if self.mock_mode or not MT5_AVAILABLE:
            return None

        try:
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            from_date = now - timedelta(days=7)
            deals = mt5.history_deals_get(position=ticket)
            if not deals:
                deals = mt5.history_deals_get(from_date, now, position=ticket)
            if deals:
                out_deals = [d for d in deals if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT]
                if out_deals:
                    final_deal = out_deals[-1]
                    total_pnl = sum(
                        getattr(d, "profit", 0.0) + getattr(d, "swap", 0.0) + getattr(d, "fee", 0.0)
                        for d in out_deals
                    )
                    
                    last_reason_code = getattr(final_deal, "reason", 0)
                    reason_map = {
                        4: "STOP_LOSS",
                        5: "TAKE_PROFIT",
                        3: "MANUAL_CLOSE",
                        6: "STOP_OUT"
                    }
                    raw_reason = reason_map.get(last_reason_code, "CLOSED")

                    # Categorize accurately based on aggregate deal outcome
                    if abs(total_pnl) <= 0.05:
                        reason_str = "BREAK_EVEN"
                    elif total_pnl > 0.05:
                        reason_str = "TAKE_PROFIT"
                    else:
                        reason_str = raw_reason if raw_reason in ("STOP_LOSS", "STOP_OUT") else "STOP_LOSS"

                    return {
                        "price": float(final_deal.price),
                        "profit": round(float(total_pnl), 2),
                        "reason": reason_str,
                        "time": final_deal.time
                    }
        except Exception as e:
            logger.warning(f"Error fetching closed deal for ticket #{ticket}: {e}")
        return None

    def _generate_mock_rates(self, symbol: str, timeframe_str: str, count: int) -> pd.DataFrame:
        """Generate synthetic OHLCV candle data for realistic offline testing."""
        np.random.seed(42)
        base_price = 2650.0 if "XAU" in symbol.upper() else 1.0850
        step_scale = 0.5 if "XAU" in symbol.upper() else 0.0003

        # Generate realistic random walk
        steps = np.random.normal(0, step_scale, count)
        prices = base_price + np.cumsum(steps)

        times = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="15min" if "15" in timeframe_str else "1h")
        opens = prices
        highs = prices + np.abs(np.random.normal(0, step_scale * 0.8, count))
        lows = prices - np.abs(np.random.normal(0, step_scale * 0.8, count))
        closes = prices + np.random.normal(0, step_scale * 0.5, count)

        df = pd.DataFrame({
            "time": times,
            "open": opens,
            "high": np.maximum.reduce([opens, highs, closes]),
            "low": np.minimum.reduce([opens, lows, closes]),
            "close": closes,
            "tick_volume": np.random.randint(50, 1000, count),
            "spread": np.random.randint(10, 30, count),
            "real_volume": np.zeros(count, dtype=int)
        })
        return df
