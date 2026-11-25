import logging
import time
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, List, Optional, Tuple, Set
import math

from config.settings import settings
from core.mt5_connector import MT5Connector

logger = logging.getLogger("ai_trading_bot.risk")


class RiskManager:
    """Enforces position sizing, daily drawdown controls, session killzones, and partial TP trailing."""

    MAX_SPREADS = {
        "XAUUSDM": 45,
        "EURUSDM": 25,
        "USDJPYM": 25,
        "BTCUSDM": 3500,
        "ETHUSDM": 600,
        "XAUUSD": 45,
        "EURUSD": 25,
        "USDJPY": 25,
        "BTCUSD": 3500,
        "ETHUSD": 600
    }

    def __init__(self, mt5_connector: MT5Connector, db: Optional[Any] = None):
        self.mt5 = mt5_connector
        self.db = db
        self.risk_percent = settings.RISK_PERCENT
        self.max_daily_drawdown_percent = settings.MAX_DAILY_DRAWDOWN_PERCENT
        self.max_positions_per_symbol = settings.MAX_OPEN_POSITIONS_PER_SYMBOL
        self.max_crypto_active_positions = getattr(settings, "MAX_OPEN_POSITIONS_CRYPTO_ACTIVE_SESSION", 1)
        self.max_crypto_weekend_positions = getattr(settings, "MAX_OPEN_POSITIONS_CRYPTO_WEEKEND", 2)
        self.max_total_positions = settings.MAX_TOTAL_OPEN_POSITIONS
        self.bep_multiplier = settings.BEP_TRIGGER_MULTIPLIER
        self.symbol_cooldown_minutes = getattr(settings, "SYMBOL_COOLDOWN_MINUTES", 45)
        self._last_exit_times: Dict[str, datetime] = {}

        # Daily Drawdown tracking (WIB)
        self._current_date: Optional[date] = None
        self._daily_starting_equity: float = 0.0
        self._dd_pnl_offset: float = 0.0
        self._broker_today_pnl_cache: Optional[Tuple[float, str, float]] = None

        # Track tickets that have already taken Partial TP
        self._partial_closed_tickets: Set[int] = set()
        # Track tickets that have already had Break-Even applied
        self._bep_moved_tickets: Set[int] = set()

    def _get_today_closed_pnl_wib(self) -> Optional[float]:
        """
        Compute today's realized closed PnL (WIB date) prioritizing broker deal history
        when available so daily drawdown stays accurate even after Railway restarts.
        Returns None only when neither broker history nor DB is attached.
        """
        wib = timezone(timedelta(hours=7))
        today_str = datetime.now(wib).strftime("%Y-%m-%d")

        broker_pnl: Optional[float] = None
        if hasattr(self.mt5, "get_recent_closed_positions"):
            now_ts = time.time()
            if (
                self._broker_today_pnl_cache is not None
                and self._broker_today_pnl_cache[1] == today_str
                and (now_ts - self._broker_today_pnl_cache[0]) < 30.0
            ):
                broker_pnl = self._broker_today_pnl_cache[2]
            else:
                try:
                    recent = self.mt5.get_recent_closed_positions(days_back=2) or []
                    total_broker = 0.0
                    found_today = False
                    for rc in recent:
                        raw_t = str(rc.get("close_time") or "").strip()
                        if not raw_t:
                            continue
                        try:
                            dt = datetime.fromisoformat(raw_t.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            dt_wib = dt.astimezone(wib)
                            if dt_wib.strftime("%Y-%m-%d") == today_str:
                                total_broker += float(rc.get("profit", 0.0))
                                found_today = True
                        except Exception:
                            continue
                    if found_today:
                        broker_pnl = round(total_broker, 2)
                        self._broker_today_pnl_cache = (now_ts, today_str, broker_pnl)
                except Exception as e:
                    logger.warning(f"[Risk] Could not fetch broker closed deals for drawdown check: {e}")

        db_pnl: Optional[float] = None
        if self.db and hasattr(self.db, "get_today_pnl"):
            db_pnl = float(self.db.get_today_pnl())

        if broker_pnl is not None:
            return broker_pnl
        return db_pnl

    def initialize_daily_equity(self, current_equity: Optional[float] = None, force_reset: bool = False):
        """Record the initial realized balance at the start of a trading day (00:00 WIB)."""
        wib = timezone(timedelta(hours=7))
        today_wib = datetime.now(wib).date()
        is_new_day = self._current_date != today_wib
        if is_new_day:
            self._dd_pnl_offset = 0.0
            self._broker_today_pnl_cache = None

        if is_new_day or self._daily_starting_equity <= 0 or force_reset:
            if current_equity is None:
                acc = self.mt5.get_account_info()
                # Prefer realized balance so container restarts during floating profits never inflate baseline!
                base_bal = float(acc.get("balance", acc.get("equity", 10000.0)))
                if not force_reset:
                    today_closed = self._get_today_closed_pnl_wib()
                    if today_closed is not None:
                        base_bal = max(base_bal - today_closed, 10.0)
                current_equity = base_bal

            self._daily_starting_equity = float(current_equity)
            self._current_date = today_wib
            logger.info(f"Daily starting balance initialized: ${self._daily_starting_equity:.2f} for date {today_wib} (WIB)")

    def reset_drawdown(self, new_equity: Optional[float] = None) -> float:
        """Manually reset daily drawdown baseline to current balance/equity (clears any halt)."""
        self._broker_today_pnl_cache = None
        acc = self.mt5.get_account_info()
        if new_equity is None:
            new_equity = float(acc.get("balance", acc.get("equity", 100.0)))
        today_closed = self._get_today_closed_pnl_wib()
        if today_closed is not None:
            self._dd_pnl_offset = float(today_closed)
        self.initialize_daily_equity(current_equity=new_equity, force_reset=True)
        return self._daily_starting_equity

    def check_daily_drawdown(self) -> Tuple[bool, float, str]:
        """
        Check if today's actual trading drawdown exceeds the maximum allowed percentage.
        Prioritizes broker deal history + SQLite realized PnL + live floating PnL,
        making it immune to withdrawals, deposits, and Railway container restarts.
        Returns (is_allowed, current_drawdown_pct, message).
        """
        self.initialize_daily_equity()
        acc = self.mt5.get_account_info()
        current_balance = float(acc.get("balance", self._daily_starting_equity))
        current_equity = float(acc.get("equity", current_balance))

        if self._daily_starting_equity <= 0:
            return True, 0.0, "Starting equity not set"

        raw_today_closed = self._get_today_closed_pnl_wib()
        if raw_today_closed is not None:
            today_closed_pnl = raw_today_closed - self._dd_pnl_offset
            floating_pnl = float(acc.get("profit", current_equity - current_balance))
            today_trading_pnl = today_closed_pnl + floating_pnl

            # Keep baseline synced to realized start-of-day balance
            base_balance = max(current_balance - today_closed_pnl, current_balance, 10.0)
            self._daily_starting_equity = base_balance

            if today_trading_pnl >= 0.0:
                return True, 0.0, f"Daily Drawdown OK: 0.00% (Profit Hari Ini: +${today_trading_pnl:.2f})"

            drawdown_dollars = abs(today_trading_pnl)
            drawdown_pct = (drawdown_dollars / base_balance) * 100.0

            if drawdown_pct >= self.max_daily_drawdown_percent:
                msg = (
                    f"DAILY DRAWDOWN BREACHED: {drawdown_pct:.2f}% >= {self.max_daily_drawdown_percent}%. "
                    f"Loss Hari Ini: -${drawdown_dollars:.2f} (Modal Acuan: ${base_balance:.2f}, Equity: ${current_equity:.2f}). "
                    "Trading dihentikan sementara hingga 00:00 WIB (Ketik /resetdd untuk reset manual)."
                )
                logger.error(msg)
                return False, drawdown_pct, msg

            return True, max(0.0, drawdown_pct), f"Daily Drawdown OK: {drawdown_pct:.2f}%"

        # Fallback mode (no DB attached, e.g. unit tests)
        drawdown_dollars = self._daily_starting_equity - current_equity
        drawdown_pct = (drawdown_dollars / self._daily_starting_equity) * 100.0 if self._daily_starting_equity > 0 else 0.0

        if drawdown_pct >= self.max_daily_drawdown_percent:
            msg = (
                f"DAILY DRAWDOWN BREACHED: {drawdown_pct:.2f}% >= {self.max_daily_drawdown_percent}%. "
                f"Starting: ${self._daily_starting_equity:.2f}, Current: ${current_equity:.2f}. "
                "Trading dihentikan sementara hingga 00:00 WIB (Ketik /resetdd jika saldo berubah akibat WD/Deposit)."
            )
            logger.error(msg)
            return False, drawdown_pct, msg

        return True, max(0.0, drawdown_pct), f"Daily Drawdown OK: {drawdown_pct:.2f}%"

    def is_session_allowed(self, symbol: str) -> Tuple[bool, str]:
        """Verify trading session rules (WIB) and weekend status."""
        wib = timezone(timedelta(hours=7))
        now_wib = datetime.now(wib)
        weekday = now_wib.weekday()  # 0=Monday ... 5=Saturday, 6=Sunday
        hour = now_wib.hour
        minute = now_wib.minute
        sym_upper = symbol.upper()

        # 1. Weekend Rule: Only Crypto (BTC & ETH) trades on Saturday & Sunday
        is_crypto = self._is_crypto(sym_upper)
        if weekday in (5, 6):
            if not is_crypto:
                return False, "Market closed for weekend (Forex/Metals reopen Monday 05:00 WIB)"

        # 1b. Friday Cutoff Rule (21:00 WIB onwards): Stop new Forex/Gold entries to avoid Weekend Gap
        if weekday == 4 and hour >= 21 and not is_crypto:
            return False, f"Friday Cutoff (21:00 WIB): Stop entry baru {symbol} jelang akhir pekan (Anti-Weekend Gap)"

        # 2. Daily Rollover Blackout (04:00 - 05:59 WIB) - Spreads widen heavily
        if 4 <= hour < 6:
            return False, "Daily market rollover window (04:00-06:00 WIB) - Spread protection"

        # 3. Session Killzones if session filtering enabled
        if getattr(settings, "SESSION_FILTER_ENABLED", True):
            # Crypto (BTC & ETH): 24/7 liquid (outside daily rollover 04:00-06:00 WIB)
            if is_crypto:
                return True, "Crypto 24/7 active"

            # USDJPY: Active ONLY in Asian (06:00-12:59) and New York / Overlap (19:00-02:59 WIB)
            # Blocked during London-only (14:00-18:59) and transition hours (13:00-13:59, 03:00-03:59)
            if "JPY" in sym_upper:
                if 6 <= hour < 13:
                    return True, "USDJPY Asian session active"
                elif hour >= 19 or hour < 3:
                    return True, "USDJPY New York session active"
                elif 14 <= hour < 19:
                    return False, f"USDJPY standby di Sesi London (Aktif kembali Sesi NY 19:00 WIB, kini: {hour:02d}:{minute:02d} WIB)"
                else:
                    return False, f"USDJPY di luar jam sesi aktif Asia/NY (kini: {hour:02d}:{minute:02d} WIB)"

            # EURUSD & XAUUSD: Active ONLY in London (14:00-18:59) and New York / Overlap (19:00-02:59 WIB)
            if "EUR" in sym_upper or "XAU" in sym_upper or "GOLD" in sym_upper:
                if hour >= 14 or hour < 3:
                    return True, f"{symbol} London/NY session active"
                elif 6 <= hour < 14:
                    return False, f"{symbol} standby di Sesi Asia/Pra-Eropa (Tunggu London Open 14:00 WIB)"
                else:
                    return False, f"{symbol} di luar jam sesi aktif London/NY (kini: {hour:02d}:{minute:02d} WIB)"

        return True, "Session OK"

    def check_spread(self, symbol: str) -> Tuple[bool, str]:
        """Check if current broker spread is within acceptable limits."""
        sym_info = self.mt5.get_symbol_info(symbol)
        if not sym_info:
            return False, f"Cannot fetch symbol info for {symbol}"

        spread = sym_info.get("spread", 0)
        max_allowed = self.MAX_SPREADS.get(symbol.upper(), 50)

        if spread > max_allowed:
            return False, f"Spread too high ({spread} > {max_allowed} pts)"

        return True, f"Spread OK ({spread} pts)"

    @staticmethod
    def _clean_symbol(sym: str) -> str:
        """Strip trailing 'm' suffix if present (e.g., BTCUSDm -> BTCUSD) for clean comparison."""
        s = str(sym or "").upper().strip()
        if s.endswith("M") and len(s) in (7, 8):
            return s[:-1]
        return s

    @classmethod
    def _is_crypto(cls, sym: str) -> bool:
        s = cls._clean_symbol(sym)
        return "BTC" in s or "ETH" in s or "CRYPTO" in s

    def get_max_positions_for_symbol(self, symbol: str) -> Tuple[int, str]:
        """
        Dynamically determine maximum allowed open positions for a symbol based on market regime:
        - Crypto (BTC & ETH) during Weekday: 1 position per coin & max 1 total Crypto combined (saving slots for FX/Gold)
        - Crypto (BTC & ETH) during Weekend (Sat-Sun): 1 position per coin (max 1 BTC + 1 ETH = 2 total Crypto)
        - Forex / Metals (XAUUSD, EURUSD, USDJPY): standard max_positions_per_symbol (e.g. 2)
        """
        wib = timezone(timedelta(hours=7))
        now_wib = datetime.now(wib)
        weekday = now_wib.weekday()  # 0=Monday ... 5=Saturday, 6=Sunday

        if self._is_crypto(symbol):
            # Weekend (Saturday & Sunday): Max 1 per coin (1 BTC + 1 ETH = max 2 Crypto total)
            if weekday in (5, 6):
                return 1, f"Mode Akhir Pekan (Maks 1 Posisi/Koin, Total Crypto Maks {self.max_crypto_weekend_positions})"

            # Weekdays (Monday - Friday): Forex/Gold active -> Cap total Crypto combined to 1
            return self.max_crypto_active_positions, f"Hari Kerja (Total Crypto Gabungan Maks {self.max_crypto_active_positions} - Slot FX/Gold Dijaga)"

        return self.max_positions_per_symbol, f"Standar (Maks {self.max_positions_per_symbol} Posisi)"

    def record_symbol_exit(self, symbol: str, exit_time: Optional[datetime] = None):
        """Record the exit timestamp of a trade to enforce the re-entry cooldown period."""
        self._broker_today_pnl_cache = None
        wib = timezone(timedelta(hours=7))
        now_wib = exit_time or datetime.now(wib)
        sym_clean = self._clean_symbol(symbol)
        self._last_exit_times[sym_clean] = now_wib
        logger.info(f"Cooldown active for {sym_clean} until {now_wib + timedelta(minutes=self.symbol_cooldown_minutes)} WIB.")

    def check_symbol_cooldown(self, symbol: str) -> Tuple[bool, str]:
        """Check if a symbol is in the post-exit cooldown window to prevent rapid overtrading/churn."""
        if self.symbol_cooldown_minutes <= 0:
            return True, "Cooldown disabled"

        sym_clean = self._clean_symbol(symbol)
        last_exit = self._last_exit_times.get(sym_clean)
        if not last_exit:
            return True, "No recent exit recorded"

        wib = timezone(timedelta(hours=7))
        now_wib = datetime.now(wib)
        time_since_exit = now_wib - last_exit
        cooldown_delta = timedelta(minutes=self.symbol_cooldown_minutes)

        if time_since_exit < cooldown_delta:
            remaining_mins = (cooldown_delta - time_since_exit).total_seconds() / 60.0
            msg = f"{symbol} in re-entry cooldown ({remaining_mins:.0f}m remaining after recent trade exit). Anti-churn active."
            logger.info(msg)
            return False, msg

        return True, "Cooldown expired"

    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        """
        Verify if a new position can be opened considering sessions, drawdown, spread, cooldown, and limits.
        """
        # 1. Session and Weekend check
        sess_ok, sess_msg = self.is_session_allowed(symbol)
        if not sess_ok:
            return False, sess_msg

        # 2. Symbol re-entry cooldown check (Anti-churn in sideways markets)
        cooldown_ok, cooldown_msg = self.check_symbol_cooldown(symbol)
        if not cooldown_ok:
            return False, cooldown_msg

        # 3. Daily drawdown check
        dd_ok, dd_pct, dd_msg = self.check_daily_drawdown()
        if not dd_ok:
            return False, dd_msg

        # 4. Spread check
        spread_ok, spread_msg = self.check_spread(symbol)
        if not spread_ok:
            return False, spread_msg

        # 5. Total active positions check
        all_positions = self.mt5.get_open_positions()
        if len(all_positions) >= self.max_total_positions:
            msg = f"Max total positions reached ({len(all_positions)}/{self.max_total_positions})."
            logger.warning(msg)
            return False, msg

        # 6. Positions per symbol check (Dynamic Slot Allocation with symbol normalization)
        sym_clean = self._clean_symbol(symbol)
        symbol_positions = [
            p for p in all_positions 
            if self._clean_symbol(p.get("symbol", "")) == sym_clean
        ]
        allowed_max, regime_desc = self.get_max_positions_for_symbol(symbol)
        if len(symbol_positions) >= allowed_max:
            msg = f"Max positions for {symbol} reached ({len(symbol_positions)}/{allowed_max}) [{regime_desc}]."
            logger.warning(msg)
            return False, msg

        # 7. Aggregate Crypto correlation guard (BTC + ETH combined)
        # Weekday: max 1 total Crypto position across BTC & ETH (preserves FX/Gold slots & prevents double correlated risk)
        # Weekend: max 2 total Crypto positions (combined with per-symbol cap 1 -> strictly 1 BTC + 1 ETH)
        if self._is_crypto(symbol):
            wib = timezone(timedelta(hours=7))
            weekday = datetime.now(wib).weekday()
            crypto_cap = self.max_crypto_weekend_positions if weekday in (5, 6) else self.max_crypto_active_positions
            crypto_positions = [
                p for p in all_positions
                if self._is_crypto(p.get("symbol", ""))
            ]
            if len(crypto_positions) >= crypto_cap:
                msg = (
                    f"Max aggregate Crypto positions reached ({len(crypto_positions)}/{crypto_cap}) "
                    f"untuk cegah double risk BTC/ETH [{regime_desc}]."
                )
                logger.warning(msg)
                return False, msg

        return True, "Risk checks passed"

    def calculate_lot_size(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        account_balance: Optional[float] = None
    ) -> float:
        """
        Calculate dynamically sized lot based on exact 1.0% dollar risk, tick value, and tick size.

        Formula:
        Max Dollar Risk = Account Balance * (Risk Percent / 100)
        Dollar Risk per 1.0 Lot = (|Entry - SL| / Tick Size) * Tick Value
        Raw Lot = Max Dollar Risk / Dollar Risk per 1.0 Lot
        """
        if account_balance is None or account_balance <= 0:
            acc = self.mt5.get_account_info()
            account_balance = float(acc.get("balance", 10000.0))

        dollar_risk = account_balance * (self.risk_percent / 100.0)
        sl_distance = abs(entry_price - sl_price)

        if sl_distance <= 0:
            logger.error(f"Invalid SL distance ({sl_distance}) for {symbol}. Returning min lot.")
            return 0.01

        sym_info = self.mt5.get_symbol_info(symbol)
        if not sym_info:
            logger.error(f"Cannot obtain symbol info for {symbol}. Returning default 0.01 lot.")
            return 0.01

        tick_size = float(sym_info.get("tick_size") or 0.01)
        tick_value = float(sym_info.get("tick_value") or 1.0)
        vol_min = float(sym_info.get("volume_min") or 0.01)
        vol_max = float(sym_info.get("volume_max") or 100.0)
        vol_step = float(sym_info.get("volume_step") or 0.01)

        # Dollar risk for 1 standard lot
        risk_per_standard_lot = (sl_distance / tick_size) * tick_value

        if risk_per_standard_lot <= 0:
            logger.error(f"Calculated 0 risk per lot for {symbol}. Returning min volume.")
            return vol_min

        raw_lot = dollar_risk / risk_per_standard_lot

        # Step normalization with epsilon protection against IEEE-754 floating point imprecision
        steps = math.floor(round(raw_lot / vol_step, 6))
        normalized_lot = steps * vol_step
        normalized_lot = round(normalized_lot, 2)

        # Clamp between min and max volume limits
        final_lot = max(vol_min, min(normalized_lot, vol_max))

        logger.info(
            f"[LOT CALC] Symbol: {symbol} | Balance: ${account_balance:.2f} | Risk(1%): ${dollar_risk:.2f} | "
            f"SL Dist: {sl_distance:.5f} | TickVal: {tick_value} | Calculated Lot: {final_lot}"
        )
        return final_lot

    def check_and_apply_breakeven(self, current_atr: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check active positions and apply Partial Take Profit (50%) + Move SL to Break-Even
        once profit reaches 1:1 Risk-to-Reward (1.5 * ATR).
        Returns list of execution result dictionaries.
        Guarantees strictly at most one BEP execution and alert per ticket.
        """
        positions = self.mt5.get_open_positions()
        results = []

        active_tickets = {p["ticket"] for p in positions}
        self._bep_moved_tickets = {t for t in self._bep_moved_tickets if t in active_tickets}
        self._partial_closed_tickets = {t for t in self._partial_closed_tickets if t in active_tickets}

        for p in positions:
            ticket = p["ticket"]
            symbol = p["symbol"]
            pos_type = p["type"]
            price_open = float(p["price_open"])
            current_sl = float(p.get("sl") or 0.0)
            price_current = float(p.get("price_current") or price_open)
            current_vol = float(p.get("volume") or 0.01)

            # 1. Skip if already tracked as BEP moved in memory
            if ticket in self._bep_moved_tickets:
                continue

            # 2. Check if persistent DB already marked BEP alert sent
            if self.db and hasattr(self.db, "is_bep_alert_sent") and self.db.is_bep_alert_sent(ticket):
                self._bep_moved_tickets.add(ticket)
                continue

            # 3. Determine price tolerance to handle broker rounding (e.g. 3733.98 vs 3733.978)
            sym_spec = self.mt5.get_symbol_info(symbol) if hasattr(self.mt5, "get_symbol_info") else None
            point = float(sym_spec.get("point", 0.01)) if sym_spec else 0.01
            tolerance = max(point * 5.0, 0.05)

            # If current SL is already at or past entry (e.g. broker rounded or trailing), treat as already at BEP
            if current_sl > 0.0:
                if pos_type == "BUY" and current_sl >= (price_open - tolerance):
                    self._bep_moved_tickets.add(ticket)
                    if self.db and hasattr(self.db, "mark_bep_alert_sent"):
                        self.db.mark_bep_alert_sent(ticket)
                    continue
                elif pos_type == "SELL" and current_sl <= (price_open + tolerance):
                    self._bep_moved_tickets.add(ticket)
                    if self.db and hasattr(self.db, "mark_bep_alert_sent"):
                        self.db.mark_bep_alert_sent(ticket)
                    continue

            # Case-insensitive ATR lookup with symbol normalization
            sym_clean = self._clean_symbol(symbol)
            atr = 0.0
            for k, v in current_atr.items():
                if self._clean_symbol(k) == sym_clean:
                    atr = float(v)
                    break

            if atr <= 0:
                continue

            target_profit_dist = self.bep_multiplier * atr
            has_reached_target = False

            if pos_type == "BUY":
                is_sl_below_entry = (current_sl == 0.0 or current_sl < (price_open - tolerance))
                has_reached_target = (price_current - price_open >= target_profit_dist) and is_sl_below_entry
            elif pos_type == "SELL":
                is_sl_above_entry = (current_sl == 0.0 or current_sl > (price_open + tolerance))
                has_reached_target = (price_open - price_current >= target_profit_dist) and is_sl_above_entry

            if has_reached_target:
                # 1. Attempt 50% Partial Take Profit if enabled and volume allows (>= 0.02 lot)
                partial_tp_done = False
                closed_vol = 0.0
                remaining_vol = current_vol

                if getattr(settings, "PARTIAL_TP_ENABLED", True) and ticket not in self._partial_closed_tickets:
                    if current_vol >= 0.02:
                        half_lot = round(current_vol / 2.0, 2)
                        close_res = self.mt5.close_partial_position(ticket, half_lot)
                        if close_res.get("success"):
                            self._partial_closed_tickets.add(ticket)
                            partial_tp_done = True
                            closed_vol = half_lot
                            remaining_vol = close_res.get("remaining_volume", round(current_vol - half_lot, 2))
                            logger.info(f"Partial TP executed for #{ticket} ({symbol}): Closed {closed_vol} lot.")

                # 2. Always move SL to Break-Even (Entry Price)
                if self.mt5.modify_position_sl(ticket, price_open):
                    self._bep_moved_tickets.add(ticket)
                    if self.db and hasattr(self.db, "mark_bep_alert_sent"):
                        self.db.mark_bep_alert_sent(ticket)
                    logger.info(f"Moved SL to Break-Even for #{ticket} ({symbol}) at {price_open}")
                    if partial_tp_done:
                        results.append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "action": "PARTIAL_TP",
                            "closed_lot": closed_vol,
                            "remaining_lot": remaining_vol,
                            "entry_price": price_open,
                            "close_price": price_current,
                            "pnl": float(p.get("profit", 0.0)) / 2.0
                        })
                    else:
                        results.append({
                            "ticket": ticket,
                            "symbol": symbol,
                            "action": "BEP_MOVED",
                            "entry_price": price_open
                        })

        return results

    def check_and_close_weekend_positions(self) -> List[Dict[str, Any]]:
        """
        Weekend Flattener (Anti-Weekend Gap):
        Every Saturday between 03:00 and 03:59 WIB (1-2 hours before Forex/Gold market close
        and before 04:00 rollover spread widening), automatically close any remaining open
        Forex/Gold positions. Crypto (BTC & ETH) positions are NEVER touched.
        """
        wib = timezone(timedelta(hours=7))
        now_wib = datetime.now(wib)

        if not (now_wib.weekday() == 5 and 3 <= now_wib.hour < 4):
            return []

        positions = self.mt5.get_open_positions()
        if not positions:
            return []

        closed_list: List[Dict[str, Any]] = []
        for p in positions:
            symbol = str(p.get("symbol") or "")
            if self._is_crypto(symbol):
                continue

            ticket = p["ticket"]
            if self.mt5.close_position(ticket):
                logger.info(
                    f"[WEEKEND AUTO-CLOSE] Closed {symbol} #{ticket} before market close "
                    f"(PnL: {float(p.get('profit', 0.0)):+.2f})"
                )
                closed_list.append({
                    "ticket": ticket,
                    "symbol": symbol,
                    "action": p.get("type", ""),
                    "volume": float(p.get("volume", 0.01)),
                    "entry_price": float(p.get("price_open", 0.0)),
                    "close_price": float(p.get("price_current", p.get("price_open", 0.0))),
                    "pnl": float(p.get("profit", 0.0)),
                })
            else:
                logger.error(f"[WEEKEND AUTO-CLOSE] Failed to close {symbol} #{ticket}")

        return closed_list
