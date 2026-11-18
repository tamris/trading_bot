"""
SQLite Database layer to persist trade lifecycle (entry, modification, take profit, stop loss, close).
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("ai_trading_bot.database")

# Jakarta / Western Indonesia Timezone (UTC+7)
WIB = timezone(timedelta(hours=7))


def get_wib_now() -> datetime:
    return datetime.now(WIB)


def get_wib_str(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = get_wib_now()
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(WIB)
    else:
        dt = dt.astimezone(WIB)
    return dt.strftime("%Y-%m-%d %H:%M:%S WIB")


class TradeDatabase:
    """Manages SQLite storage for trade orders, execution records, and close events."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = base_dir / "data" / "trading_history.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    lot REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    confluence_score INTEGER DEFAULT 0,
                    ai_reasoning TEXT DEFAULT '',
                    close_price REAL,
                    close_time TEXT,
                    close_reason TEXT,
                    profit_loss REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'OPEN',
                    entry_alert_sent INTEGER DEFAULT 0,
                    close_alert_sent INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")

            # Auto-migrate table if created with previous schema
            cursor.execute("PRAGMA table_info(trades)")
            cols = [col["name"] for col in cursor.fetchall()]
            if "entry_alert_sent" not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN entry_alert_sent INTEGER DEFAULT 0")
            if "close_alert_sent" not in cols:
                cursor.execute("ALTER TABLE trades ADD COLUMN close_alert_sent INTEGER DEFAULT 0")

            # Table for persistent Telegram subscribers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_subscribers (
                    chat_id TEXT PRIMARY KEY,
                    username TEXT DEFAULT '',
                    first_name TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL
                )
            """)

            # Table for persistent Gemini API Key Pool
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_value TEXT UNIQUE NOT NULL,
                    label TEXT DEFAULT '',
                    status TEXT DEFAULT 'ACTIVE',
                    requests_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    rate_limit_count INTEGER DEFAULT 0,
                    cooldown_until REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                )
            """)
            # Table for persistent bot metadata (e.g. reset timestamps)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key_status ON api_keys(status)")
            conn.commit()

    def add_subscriber(self, chat_id: str, username: str = "", first_name: str = ""):
        """Persist or update an active Telegram subscriber."""
        now_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO telegram_subscribers (chat_id, username, first_name, created_at, last_active_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        username = excluded.username,
                        first_name = excluded.first_name,
                        last_active_at = excluded.last_active_at
                """, (str(chat_id), str(username or ""), str(first_name or ""), now_str, now_str))
                conn.commit()
        except Exception as e:
            logger.warning(f"[DB] Error saving subscriber {chat_id}: {e}")

    def get_subscribers(self) -> List[str]:
        """Fetch all unique subscriber chat IDs."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chat_id FROM telegram_subscribers")
                return [str(row["chat_id"]) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"[DB] Error fetching subscribers: {e}")
            return []

    def mark_entry_alert_sent(self, ticket: int) -> bool:
        """Mark that the entry alert was successfully delivered to Telegram."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE trades SET entry_alert_sent = 1 WHERE ticket = ?", (ticket,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"[DB] Error marking entry alert for #{ticket}: {e}")
            return False

    def mark_close_alert_sent(self, ticket: int) -> bool:
        """Mark that the close alert (TP/SL) was successfully delivered to Telegram."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE trades SET close_alert_sent = 1 WHERE ticket = ?", (ticket,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"[DB] Error marking close alert for #{ticket}: {e}")
            return False

    def is_entry_alert_sent(self, ticket: int) -> bool:
        """Check if an entry alert has already been sent for this trade."""
        trade = self.get_trade_by_ticket(ticket)
        return bool(trade and trade.get("entry_alert_sent", 0) == 1)

    def is_close_alert_sent(self, ticket: int) -> bool:
        """Check if a close alert has already been sent for this trade."""
        trade = self.get_trade_by_ticket(ticket)
        return bool(trade and trade.get("close_alert_sent", 0) == 1)

    def record_entry(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        confluence_score: int = 0,
        ai_reasoning: str = "",
        entry_alert_sent: int = 0
    ) -> bool:
        """Insert a newly executed trade with OPEN status."""
        entry_time_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        ticket, symbol, action, lot, entry_price, sl_price, tp_price,
                        entry_time, confluence_score, ai_reasoning, status, entry_alert_sent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """, (
                    ticket, symbol, action.upper(), float(lot), float(entry_price),
                    float(sl_price or 0.0), float(tp_price or 0.0),
                    entry_time_str, int(confluence_score or 0), str(ai_reasoning or ""),
                    int(entry_alert_sent)
                ))
                conn.commit()
                logger.info(f"[DB] Trade #{ticket} recorded as OPEN in SQLite (WIB: {entry_time_str}).")
                return True
        except Exception as e:
            logger.error(f"[DB] Error recording trade entry #{ticket}: {e}")
            return False

    def record_close(
        self,
        ticket: int,
        close_price: float,
        profit_loss: float,
        close_reason: str = "MANUAL_CLOSE",
        close_alert_sent: Optional[int] = None
    ) -> bool:
        """Update an existing trade with exit details and set status to CLOSED."""
        close_time_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if close_alert_sent is not None:
                    cursor.execute("""
                        UPDATE trades
                        SET close_price = ?,
                            close_time = ?,
                            profit_loss = ?,
                            close_reason = ?,
                            status = 'CLOSED',
                            close_alert_sent = ?
                        WHERE ticket = ?
                    """, (float(close_price), close_time_str, float(profit_loss), str(close_reason), int(close_alert_sent), ticket))
                else:
                    cursor.execute("""
                        UPDATE trades
                        SET close_price = ?,
                            close_time = ?,
                            profit_loss = ?,
                            close_reason = ?,
                            status = 'CLOSED'
                        WHERE ticket = ?
                    """, (float(close_price), close_time_str, float(profit_loss), str(close_reason), ticket))
                conn.commit()
                logger.info(f"[DB] Trade #{ticket} marked as CLOSED (Reason: {close_reason}, PnL: ${profit_loss:.2f}).")
                return True
        except Exception as e:
            logger.error(f"[DB] Error recording trade close #{ticket}: {e}")
            return False

    def update_sl(self, ticket: int, new_sl: float) -> bool:
        """Update Stop Loss of a trade in database (e.g., Trailing Break-Even)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE trades SET sl_price = ? WHERE ticket = ?", (new_sl, ticket))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"[DB] Error updating SL for #{ticket}: {e}")
            return False

    def update_lot_and_pnl(self, ticket: int, remaining_lot: float, partial_pnl: float) -> bool:
        """Update lot size and accumulate realized profit after a Partial TP."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades 
                    SET lot = ?, profit_loss = profit_loss + ? 
                    WHERE ticket = ?
                """, (remaining_lot, partial_pnl, ticket))
                conn.commit()
                logger.info(f"[DB] Trade #{ticket} updated: lot={remaining_lot}, partial_pnl=+${partial_pnl:.2f}.")
                return True
        except Exception as e:
            logger.error(f"[DB] Error updating partial trade #{ticket}: {e}")
            return False

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Return list of all currently open trades."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY ticket ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_trades(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch the most recent trades (open and closed)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY ticket DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trade_by_ticket(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Fetch single trade by ticket number."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def is_trade_closed(self, ticket: int) -> bool:
        """Check if a trade exists in database and is already marked CLOSED."""
        trade = self.get_trade_by_ticket(ticket)
        return bool(trade and trade.get("status") == "CLOSED")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregated trading statistics distinguishing Wins, BEP (Zero Loss), and Losses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM trades WHERE status = 'CLOSED'")
            total_closed = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as wins FROM trades WHERE status = 'CLOSED' AND profit_loss > 0.05")
            wins = cursor.fetchone()["wins"]

            cursor.execute("SELECT COUNT(*) as beps FROM trades WHERE status = 'CLOSED' AND ABS(profit_loss) <= 0.05")
            beps = cursor.fetchone()["beps"]

            cursor.execute("SELECT COUNT(*) as losses FROM trades WHERE status = 'CLOSED' AND profit_loss < -0.05")
            losses = cursor.fetchone()["losses"]

            cursor.execute("SELECT SUM(profit_loss) as total_pnl FROM trades WHERE status = 'CLOSED'")
            total_pnl = cursor.fetchone()["total_pnl"] or 0.0

            cursor.execute("SELECT COUNT(*) as open_cnt FROM trades WHERE status = 'OPEN'")
            open_cnt = cursor.fetchone()["open_cnt"]

            win_rate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0
            adjusted_win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
            capital_preserved_rate = ((wins + beps) / total_closed * 100.0) if total_closed > 0 else 0.0

            return {
                "total_closed": total_closed,
                "wins": wins,
                "beps": beps,
                "losses": losses,
                "win_rate": win_rate,
                "adjusted_win_rate": adjusted_win_rate,
                "capital_preserved_rate": capital_preserved_rate,
                "total_pnl": total_pnl,
                "open_trades": open_cnt
            }

    def set_metadata(self, key: str, value: str):
        """Store key-value metadata in SQLite."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO bot_metadata (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, (str(key), str(value)))
                conn.commit()
        except Exception as e:
            logger.error(f"[DB] Error setting metadata {key}: {e}")

    def get_metadata(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetch metadata value by key."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM bot_metadata WHERE key = ?", (str(key),))
                row = cursor.fetchone()
                return str(row["value"]) if row else default
        except Exception as e:
            logger.error(f"[DB] Error fetching metadata {key}: {e}")
            return default

    def get_last_reset_time(self) -> Optional[str]:
        """Get the ISO timestamp of the last history reset."""
        return self.get_metadata("last_reset_time", None)

    def reset_closed_trades(self) -> int:
        """
        Delete all CLOSED/historical trades from the database,
        preserving all currently OPEN positions intact, and mark reset timestamp.
        """
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades WHERE status != 'OPEN'")
                deleted_count = cursor.rowcount
                cursor.execute("""
                    INSERT INTO bot_metadata (key, value) VALUES ('last_reset_time', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, (now_iso,))
                conn.commit()
            
            # Run VACUUM outside transaction
            conn = self._get_connection()
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.close()

            logger.info(f"[DB] Reset complete: deleted {deleted_count} closed trades at {now_iso}. Preserved OPEN positions.")
            return deleted_count
        except Exception as e:
            logger.error(f"[DB] Error resetting closed trades: {e}")
            return 0

    # ==========================================
    # Gemini API Key Pool Management (SQLite)
    # ==========================================

    def add_api_key(self, key_value: str, label: str = "") -> Optional[int]:
        """Insert a new Gemini API key into the database pool. Returns row id if successful."""
        key_value = key_value.strip()
        if not key_value or key_value == "your_gemini_api_key_here":
            return None

        created_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO api_keys (key_value, label, status, created_at)
                    VALUES (?, ?, 'ACTIVE', ?)
                    ON CONFLICT(key_value) DO UPDATE SET status = 'ACTIVE'
                """, (key_value, label, created_str))
                conn.commit()
                row_id = cursor.lastrowid
                logger.info(f"[DB] Gemini API key added/updated in DB (ID: {row_id}).")
                return row_id
        except Exception as e:
            logger.error(f"[DB] Error adding API key to DB: {e}")
            return None

    def get_all_api_keys(self) -> List[Dict[str, Any]]:
        """Fetch all registered API keys from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_api_key(self, key_id: int) -> bool:
        """Delete an API key by its database ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                if deleted:
                    logger.info(f"[DB] Gemini API key #{key_id} deleted from DB.")
                return deleted
        except Exception as e:
            logger.error(f"[DB] Error deleting API key #{key_id}: {e}")
            return False

    def update_key_usage(
        self,
        key_value: str,
        is_success: bool = True,
        is_rate_limit: bool = False,
        cooldown_until_ts: float = 0.0
    ) -> bool:
        """Update request stats, success count, rate limits, and cooldown status."""
        now_str = get_wib_str()
        status_val = "COOLDOWN" if is_rate_limit else "ACTIVE"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE api_keys
                    SET requests_count = requests_count + 1,
                        success_count = success_count + ?,
                        rate_limit_count = rate_limit_count + ?,
                        status = ?,
                        cooldown_until = ?,
                        last_used_at = ?
                    WHERE key_value = ?
                """, (
                    1 if is_success else 0,
                    1 if is_rate_limit else 0,
                    status_val,
                    cooldown_until_ts,
                    now_str,
                    key_value.strip()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"[DB] Error updating API key usage: {e}")
            return False

    def seed_api_keys(self, keys: List[str]) -> int:
        """Populate table with initial keys from config/env if not already present."""
        seeded = 0
        for k in keys:
            k = k.strip()
            if not k or k == "your_gemini_api_key_here":
                continue
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM api_keys WHERE key_value = ?", (k,))
                if not cursor.fetchone():
                    self.add_api_key(k, label="Default Key")
                    seeded += 1
        return seeded
