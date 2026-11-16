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
                    status TEXT DEFAULT 'OPEN'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")

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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key_status ON api_keys(status)")
            conn.commit()

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
        ai_reasoning: str = ""
    ) -> bool:
        """Insert a newly executed trade with OPEN status."""
        entry_time_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        ticket, symbol, action, lot, entry_price, sl_price, tp_price,
                        entry_time, confluence_score, ai_reasoning, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    ticket, symbol, action.upper(), lot, entry_price, sl_price, tp_price,
                    entry_time_str, confluence_score, ai_reasoning
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
        close_reason: str = "MANUAL_CLOSE"
    ) -> bool:
        """Update an existing trade with exit details and set status to CLOSED."""
        close_time_str = get_wib_str()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades
                    SET close_price = ?,
                        close_time = ?,
                        profit_loss = ?,
                        close_reason = ?,
                        status = 'CLOSED'
                    WHERE ticket = ?
                """, (close_price, close_time_str, profit_loss, close_reason, ticket))
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

    def reset_closed_trades(self) -> int:
        """
        Delete all CLOSED/historical trades from the database,
        preserving all currently OPEN positions intact.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades WHERE status != 'OPEN'")
                deleted_count = cursor.rowcount
                conn.commit()
            
            # Run VACUUM outside transaction
            conn = self._get_connection()
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.close()

            logger.info(f"[DB] Reset complete: deleted {deleted_count} closed trades. Preserved all OPEN positions.")
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
