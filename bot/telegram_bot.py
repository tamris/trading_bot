import sys
import socket
from pathlib import Path

# Force IPv4 for api.telegram.org to prevent IPv6 blackhole timeouts on Indonesian ISPs
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_telegram_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host and "telegram.org" in str(host):
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_telegram_getaddrinfo

# Ensure project root is in sys.path when executed directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import asyncio
from typing import Optional, Dict, Any, List
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from config.settings import settings
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from core.database import TradeDatabase, get_wib_str, get_wib_now

logger = logging.getLogger("ai_trading_bot.telegram")

# Keyboard persistent menu under the chat input
REPLY_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("📊 Status Akun"), KeyboardButton("📜 Riwayat Trade")],
        [KeyboardButton("🔍 Radar Pasar"), KeyboardButton("🕒 Sesi Pasar")],
        [KeyboardButton("⏸️ Pause Bot"), KeyboardButton("▶️ Resume Bot")],
        [KeyboardButton("🚨 Tutup Semua (Panic)")]
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)


class TelegramBotManager:
    """Manages Telegram bot alerts, interactive keyboard menus, and SQLite trade history."""

    def __init__(
        self,
        mt5_connector: MT5Connector,
        risk_manager: RiskManager,
        trading_state: Dict[str, Any],
        db: Optional[TradeDatabase] = None,
        gemini_analyst: Optional[Any] = None
    ):
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager
        self.trading_state = trading_state
        self.db = db or TradeDatabase()
        self.gemini = gemini_analyst
        self.radar_provider = None
        self.scan_report_mode = getattr(settings, "SCAN_REPORT_MODE", "1h").lower().strip()
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self._configured_chat_ids: List[str] = [
            cid.strip() for cid in str(self.chat_id or "").split(",") if cid.strip()
        ]
        self._active_chat_ids: Set[str] = set()
        self.app = None
        self._is_running = False

    def is_configured(self) -> bool:
        return bool(self.token and (self._configured_chat_ids or self.chat_id))

    def _register_active_chat(self, update: Optional[Update]):
        """Dynamically register chat ID from user interaction so all active users receive alerts."""
        if update and update.effective_chat:
            self._active_chat_ids.add(str(update.effective_chat.id))

    @property
    def target_chat_ids(self) -> List[str]:
        """All unique target chat IDs (configured + actively communicating users)."""
        all_ids = set(self._configured_chat_ids)
        all_ids.update(self._active_chat_ids)
        if not all_ids and self.chat_id:
            all_ids.add(str(self.chat_id).strip())
        return [cid for cid in all_ids if cid]

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle unexpected telegram polling/network errors gracefully without crashing."""
        err = context.error
        logger.warning(f"Telegram polling/network warning: {type(err).__name__}: {err}")

    async def start_bot(self):
        """Start Telegram bot polling in background with resilience against network timeouts."""
        if not self.is_configured():
            logger.info("Telegram Bot Token or Chat ID not configured. Running without Telegram bot.")
            return

        for attempt in range(1, 4):
            try:
                self.app = (
                    ApplicationBuilder()
                    .token(self.token)
                    .connect_timeout(20.0)
                    .read_timeout(30.0)
                    .write_timeout(20.0)
                    .pool_timeout(20.0)
                    .get_updates_read_timeout(35.0)
                    .get_updates_connection_pool_size(8)
                    .connection_pool_size(8)
                    .build()
                )

                # Global error handler to catch network hiccups safely
                self.app.add_error_handler(self._error_handler)

                # Register command handlers
                self.app.add_handler(CommandHandler("start", self._cmd_start))
                self.app.add_handler(CommandHandler("status", self._cmd_status))
                self.app.add_handler(CommandHandler("history", self._cmd_history))
                self.app.add_handler(CommandHandler("radar", self._cmd_radar))
                self.app.add_handler(CommandHandler("report", self._cmd_report))
                self.app.add_handler(CommandHandler("sesi", self._cmd_session))
                self.app.add_handler(CommandHandler("session", self._cmd_session))
                self.app.add_handler(CommandHandler("keys", self._cmd_keys))
                self.app.add_handler(CommandHandler("addkey", self._cmd_addkey))
                self.app.add_handler(CommandHandler("delkey", self._cmd_delkey))
                self.app.add_handler(CommandHandler("pause", self._cmd_pause))
                self.app.add_handler(CommandHandler("resume", self._cmd_resume))
                self.app.add_handler(CommandHandler("closeall", self._cmd_closeall))
                self.app.add_handler(CommandHandler("resetdb", self._cmd_resetdb))
                self.app.add_handler(CommandHandler("help", self._cmd_help))

                # Register quick keyboard menu button clicks
                self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_menu_button))

                await self.app.initialize()
                await self.app.start()

                # Register official Telegram bottom-left Menu commands
                try:
                    await self.app.bot.set_my_commands([
                        BotCommand("status", "📊 Status Saldo & Posisi Aktif"),
                        BotCommand("radar", "🔍 Radar Pasar (Alasan Belum Entry)"),
                        BotCommand("report", "⏱️ Atur Notif Radar: /report 1h, 15m, off"),
                        BotCommand("history", "📜 Riwayat Trade (Entry -> Close)"),
                        BotCommand("sesi", "🕒 Status Sesi Pasar & Pair Aktif"),
                        BotCommand("keys", "🔑 Cek Pool API Key (SQLite DB)"),
                        BotCommand("pause", "⏸️ Jeda Auto-Trade"),
                        BotCommand("resume", "▶️ Lanjutkan Auto-Trade"),
                        BotCommand("closeall", "🚨 Tutup Darurat Semua Posisi"),
                        BotCommand("resetdb", "🧹 Bersihkan Riwayat (Simpan Open)"),
                        BotCommand("help", "ℹ️ Bantuan & Panduan Menu")
                    ])
                except Exception as e:
                    logger.warning(f"Could not set Telegram menu commands: {e}")

                # drop_pending_updates=False ensures user commands sent while offline are processed!
                await self.app.updater.start_polling(
                    drop_pending_updates=False,
                    poll_interval=0.5,
                    timeout=10,
                    bootstrap_retries=5
                )
                self._is_running = True
                logger.info("Telegram Bot successfully started and listening for commands.")

                # Send startup greeting
                wib_now = get_wib_str()
                await self.send_alert(
                    "🚀 *AI Trading Bot - ONLINE!*\n\n"
                    f"• *Waktu Aktif:* `{wib_now}`\n"
                    f"• *Pasang Pantau:* `{', '.join(settings.SYMBOLS)}`\n"
                    "• *Menu Cepat:* Gunakan tombol di bawah keyboard atau ketik /status\n",
                    with_keyboard=True
                )
                break
            except Exception as e:
                logger.warning(f"Telegram Bot start attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    await asyncio.sleep(3)
                else:
                    logger.error(f"Failed to start Telegram Bot after 3 attempts: {e}")

    async def stop_bot(self):
        """Stop Telegram bot cleanly."""
        if self.app and self._is_running:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                self._is_running = False
                logger.info("Telegram Bot stopped.")
            except Exception as e:
                logger.warning(f"Error stopping Telegram Bot: {e}")

    async def send_alert(self, message: str, with_keyboard: bool = False):
        """Send notification message to all target chat IDs with retry on timeout and plain-text fallback."""
        if not self.is_configured() or not self.app:
            logger.info(f"[TELEGRAM LOG ONLY]:\n{message}")
            return

        targets = self.target_chat_ids
        for target_id in targets:
            for attempt in range(1, 4):
                try:
                    await self.app.bot.send_message(
                        chat_id=target_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=REPLY_KEYBOARD if with_keyboard else None,
                        read_timeout=20,
                        write_timeout=20
                    )
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(f"Telegram send_alert to {target_id} attempt {attempt}/3: {e}")

                    # Fallback: if Telegram rejects due to markdown entity parsing, send cleanly without formatting
                    if "can't parse" in err_str or "entity" in err_str or "parse" in err_str or "badrequest" in err_str:
                        try:
                            clean_text = (
                                message.replace("*", "")
                                .replace("`", "")
                                .replace("_", "")
                            )
                            await self.app.bot.send_message(
                                chat_id=target_id,
                                text=clean_text,
                                parse_mode=None,
                                reply_markup=REPLY_KEYBOARD if with_keyboard else None,
                                read_timeout=20,
                                write_timeout=20
                            )
                            logger.info(f"Successfully delivered alert to {target_id} via plain text fallback.")
                            break
                        except Exception as fb_err:
                            logger.error(f"Plain text fallback to {target_id} also failed: {fb_err}")

                    if attempt < 3:
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"Failed to send Telegram alert to {target_id} after 3 attempts: {e}")

    async def send_trade_signal_alert(
        self,
        symbol: str,
        action: str,
        lot_size: float,
        entry_price: float,
        sl: float,
        tp: float,
        confluence_score: int,
        reasoning: str,
        order_result: Dict[str, Any]
    ):
        """Format and broadcast detailed trade execution report in WIB."""
        status_icon = "🟢" if action == "BUY" else "🔴"
        order_status = f"✅ Sukses (Tiket #{order_result.get('ticket')})" if order_result.get("success") else f"⚠️ Gagal: {order_result.get('error')}"
        wib_time = get_wib_str()

        # Sanitize reasoning to avoid breaking Telegram Markdown parser
        clean_reasoning = (
            str(reasoning or "")
            .replace("_", " ")
            .replace("*", "")
            .replace("`", "'")
            .strip()
        )

        dec = 2 if ("BTC" in symbol or "XAU" in symbol) else 5

        msg = (
            f"{status_icon} *ORDER EKSEKUSI BARU (ENTRY)*\n\n"
            f"• *Instrumen:* `{symbol}`\n"
            f"• *Aksi:* `{action}`\n"
            f"• *Volume:* `{lot_size}` Lot (Resiko 1.0%)\n"
            f"• *Harga Entry:* `{entry_price:.{dec}f}`\n"
            f"• *Stop Loss:* `{sl:.{dec}f}`\n"
            f"• *Take Profit:* `{tp:.{dec}f}`\n"
            f"• *Skor Konfluensi:* `{confluence_score}/100`\n"
            f"• *Waktu Entry:* `{wib_time}`\n"
            f"• *Status Broker:* {order_status}\n\n"
            f"🧠 *Analisa AI:*\n_{clean_reasoning}_\n"
        )
        await self.send_alert(msg, with_keyboard=True)

    async def send_trade_close_alert(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot_size: float,
        entry_price: float,
        close_price: float,
        pnl: float,
        reason: str,
        entry_time: str,
        close_time: str
    ):
        """Broadcast real-time close report (TP, SL, BEP, or Manual Close)."""
        if reason == "TAKE_PROFIT":
            title = "🎯 *TARGET TERCAPAI (TAKE PROFIT)!*"
            icon = "💰"
        elif reason == "STOP_LOSS":
            title = "🛑 *TERKENA STOP LOSS*"
            icon = "⚠️"
        elif "BREAK_EVEN" in reason or "BEP" in reason:
            title = "🛡️ *EXIT AT BREAK-EVEN*"
            icon = "🔒"
        else:
            title = "🏁 *POSISI DITUTUP (CLOSED)*"
            icon = "📊"

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        msg = (
            f"{title}\n\n"
            f"• *Tiket:* `#{ticket}`\n"
            f"• *Instrumen:* `{symbol}` ({action})\n"
            f"• *Volume:* `{lot_size}` Lot\n"
            f"• *Harga Entry:* `{entry_price:.5f}` ({entry_time})\n"
            f"• *Harga Exit:* `{close_price:.5f}` ({close_time})\n"
            f"• *Alasan Exit:* `{reason}`\n"
            f"• *Hasil PnL:* `{pnl_str}` {icon}\n"
        )
        await self.send_alert(msg, with_keyboard=True)

    async def send_partial_tp_alert(
        self,
        ticket: int,
        symbol: str,
        closed_lot: float,
        remaining_lot: float,
        partial_pnl: float,
        entry_price: float,
        close_price: float
    ):
        """Broadcast alert when 50% partial take profit is locked in and SL is moved to BEP."""
        pnl_str = f"+${partial_pnl:.2f}" if partial_pnl >= 0 else f"-${abs(partial_pnl):.2f}"
        msg = (
            "🎯 *PARTIAL TAKE PROFIT TERCAPAI (TP1 - 50%)!*\n\n"
            f"• *Tiket:* `#{ticket}`\n"
            f"• *Instrumen:* `{symbol}`\n"
            f"• *Volume Ditutup:* `{closed_lot}` Lot (Amankan Profit)\n"
            f"• *Volume Berjalan:* `{remaining_lot}` Lot (Risk-Free)\n"
            f"• *Profit Diamankan:* `{pnl_str}` 💰\n"
            f"• *Harga Exit TP1:* `{close_price:.5f}`\n"
            f"• *Proteksi:* 🛡️ Stop Loss otomatis digeser ke harga Entry (`{entry_price:.5f}`).\n\n"
            "Posisi sisa sekarang *100% BEBAS RESIKO RUGI* mengejar target TP maksimal!"
        )
        await self.send_alert(msg, with_keyboard=True)

    # ==========================================
    # Interactive Button Router
    # ==========================================

    async def _handle_menu_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle quick button taps from the persistent ReplyKeyboardMarkup."""
        self._register_active_chat(update)
        text = update.message.text.strip()
        if "Status" in text:
            await self._cmd_status(update, context)
        elif "Riwayat" in text or "History" in text:
            await self._cmd_history(update, context)
        elif "Radar" in text:
            await self._cmd_radar(update, context)
        elif "Sesi" in text or "Session" in text:
            await self._cmd_session(update, context)
        elif "Pool" in text or "Key" in text:
            await self._cmd_keys(update, context)
        elif "Pause" in text:
            await self._cmd_pause(update, context)
        elif "Resume" in text:
            await self._cmd_resume(update, context)
        elif "Tutup Semua" in text or "Panic" in text:
            await self._cmd_closeall(update, context)
        else:
            await update.message.reply_text(
                "Silakan pilih menu di bawah keyboard:",
                reply_markup=REPLY_KEYBOARD
            )

    # ==========================================
    # Command Handlers
    # ==========================================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._register_active_chat(update)
        wib_time = get_wib_str()
        acc = self.mt5.get_account_info()
        mode_str = "⏸️ Jeda (Paused)" if self.trading_state.get("is_paused") else "🟢 Aktif (Active)"

        text = (
            "🤖 *Selamat Datang di Bot Trading AI MT5!*\n\n"
            f"• *Waktu Server:* `{wib_time}`\n"
            f"• *Akun Trading:* `{acc.get('login', 'N/A')}` ({acc.get('server', 'N/A')})\n"
            f"• *Saldo Akun:* `${acc.get('balance', 0.0):,.2f}`\n"
            f"• *Status Auto-Trade:* {mode_str}\n"
            f"• *Pair Dipantau:* `{', '.join(settings.SYMBOLS)}`\n\n"
            "💡 *Menu Cepat:* Kamu bisa langsung klik tombol menu di bawah kolom chat untuk cek status dan riwayat trade!"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "ℹ️ *Panduan Perintah Bot Trading:*\n\n"
            "• `/status` atau tombol *📊 Status Akun* - Cek saldo, equity, posisi yang sedang terbuka & drawdown.\n"
            "• `/history` atau tombol *📜 Riwayat Trade* - Lihat daftar riwayat trade dari open sampai TP/SL (SQLite).\n"
            "• `/sesi` atau tombol *🕒 Sesi Pasar* - Cek sesi pasar aktif saat ini & status izin tiap pair (WIB).\n"
            "• `/keys` atau tombol *🔑 Cek Pool API* - Cek status multi-key Gemini API (SQLite DB).\n"
            "• `/addkey <key>` - Tambahkan Gemini API key baru ke dalam pool database.\n"
            "• `/delkey <id>` - Hapus Gemini API key berdasarkan ID dari database.\n"
            "• `/pause` atau tombol *⏸️ Pause Bot* - Jeda order baru (tetap memantau chart & kirim sinyal).\n"
            "• `/resume` atau tombol *▶️ Resume Bot* - Mengaktifkan kembali eksekusi order otomatis.\n"
            "• `/resetdb` - Bersihkan riwayat lama yang sudah CLOSED (posisi OPEN tetap aman).\n"
            "• `/closeall` atau tombol *🚨 Tutup Semua* - Tombol darurat untuk menutup semua posisi di MT5 seketika."
        )
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._register_active_chat(update)
        wib_time = get_wib_str()
        dd_ok, dd_pct, dd_msg = self.risk_manager.check_daily_drawdown()
        acc = self.mt5.get_account_info()
        positions = self.mt5.get_open_positions()

        pos_summary = ""
        if positions:
            for p in positions:
                profit_val = p.get("profit", 0.0)
                pnl_symbol = "+" if profit_val >= 0 else ""
                cur_p = p.get("price_current", p.get("price_open", 0.0))
                pos_summary += (
                    f"  - `#{p['ticket']}` {p['type']} {p['volume']} {p['symbol']} "
                    f"@ `{p['price_open']}` (Kini: `{cur_p}`) "
                    f"PnL: `{pnl_symbol}${profit_val:.2f}`\n"
                )
        else:
            pos_summary = "  _Tidak ada posisi yang sedang terbuka._\n"

        status_text = (
            "📊 *Status Sistem Trading (WIB)*\n\n"
            f"• *Waktu Saat Ini:* `{wib_time}`\n"
            f"• *Akun:* `{acc.get('login', 'N/A')}` ({acc.get('server', 'N/A')})\n"
            f"• *Balance:* `${acc.get('balance', 0.0):,.2f}`\n"
            f"• *Equity:* `${acc.get('equity', 0.0):,.2f}`\n"
            f"• *Daily Drawdown:* `{dd_pct:.2f}%` (Batas Max: {settings.MAX_DAILY_DRAWDOWN_PERCENT}%)\n"
            f"• *Status Resiko:* {'🟢 Aman (Safe)' if dd_ok else '🔴 Terkunci (Blocked)'}\n"
            f"• *Auto-Trade:* {'⏸️ Jeda (Paused)' if self.trading_state.get('is_paused') else '🟢 Aktif'}\n\n"
            f"*Posisi Aktif ({len(positions)}/{settings.MAX_TOTAL_OPEN_POSITIONS}):*\n{pos_summary}\n"
            f"• *Pengecekan M15 Terakhir:* {self.trading_state.get('last_check_time', 'Menunggu candle')}"
        )
        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display current trading session, allowed active symbols, and global timetable in WIB."""
        wib_time = get_wib_str()
        now_dt = get_wib_now()
        current_hour = now_dt.hour
        weekday = now_dt.weekday()

        if weekday in (5, 6):
            session_name = "🏖️ Akhir Pekan (Forex & Gold Tutup)"
            session_desc = "Pasar Forex dan Logam libur. Hanya Crypto (BTC) yang aktif 24/7."
        elif 6 <= current_hour < 13:
            session_name = "🌏 Sesi Asia (Tokyo & Sydney)"
            session_desc = "Fokus utama: USDJPY & BTC. Emas & Euro standby menunggu Sesi London (14:00 WIB)."
        elif 13 <= current_hour < 14:
            session_name = "⏳ Transisi Pra-Eropa (Menuju London Open)"
            session_desc = "Pasar Asia bersiap tutup, bersiap menyambut pembukaan bursa London."
        elif 14 <= current_hour < 19:
            session_name = "🇬🇧 Sesi London (Eropa)"
            session_desc = "Volatilitas tinggi! Fokus utama: XAUUSD (Gold), EURUSD, & BTC."
        elif 19 <= current_hour < 21:
            session_name = "🔥 Overlap London & New York (Golden Hours)"
            session_desc = "Likuiditas tertinggi dunia! Semua pair aktif dengan momentum kencang."
        elif current_hour >= 21 or current_hour < 4:
            session_name = "🇺🇸 Sesi New York (Amerika)"
            session_desc = "Pergerakan kencang berita AS. Semua pair aktif."
        else:
            session_name = "🛡️ Rollover Interbank (Proteksi Spread Melebar)"
            session_desc = "Semua entry di-freeze otomatis demi menghindari spread malam yang tinggi."

        msg = (
            "🕒 *STATUS SESI PASAR GLOBAL (WIB)*\n\n"
            f"• *Waktu Saat Ini:* `{wib_time}`\n"
            f"• *Sesi Sedang Berjalan:* *{session_name}*\n"
            f"• *Karakteristik:* _{session_desc}_\n\n"
            "📋 *Status Izin Entry Pair (The 4 Kings):*\n"
        )

        for sym in settings.SYMBOLS:
            allowed, reason = self.risk_manager.is_session_allowed(sym)
            max_pos, quota_desc = self.risk_manager.get_max_positions_for_symbol(sym)
            icon = "🟢" if allowed else "⏸️"
            status_text = "*AKTIF (Bisa Entry)*" if allowed else "*STANDBY*"
            msg += f"{icon} *{sym}:* {status_text} (Kuota: `{max_pos}` Posisi)\n   └ _{reason}_\n   └ 🏷️ _{quota_desc}_\n\n"

        msg += (
            "📌 *Jadwal Jam Sesi Utama (WIB):*\n"
            "• `06:00 - 13:00` 🌏 Sesi Asia (Fokus: USDJPY & BTC)\n"
            "• `14:00 - 21:00` 🇬🇧 Sesi London (Fokus: XAUUSD, EURUSD & BTC)\n"
            "• `19:00 - 03:00` 🇺🇸 Sesi New York (Semua Pair Aktif Agresif)\n"
            "• `04:00 - 06:00` 🛡️ Rollover Break (Freeze Anti-Spread)\n\n"
            "💡 *Alokasi Kuota Slot Dinamis:*\n"
            "• *Hari Kerja (Senin–Jumat):* BTC dibatasi maks *1 posisi* agar minimal 2 slot tersisa aman untuk Forex/Gold.\n"
            "• *Akhir Pekan (Sabtu–Minggu):* Forex/Gold libur, kuota BTC otomatis naik ke maks *2 posisi*."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display recent trade history and win rate from SQLite database."""
        stats = self.db.get_summary_stats()
        trades = self.db.get_recent_trades(limit=5)

        total_pnl = stats["total_pnl"]
        pnl_prefix = "+" if total_pnl >= 0 else ""
        wins = stats.get("wins", 0)
        beps = stats.get("beps", 0)
        losses = stats.get("losses", 0)
        total_closed = stats.get("total_closed", 0)
        adjusted_wr = stats.get("adjusted_win_rate", 0.0)
        preserved_rate = stats.get("capital_preserved_rate", 0.0)

        header = (
            "📜 *Riwayat Transaksi (SQLite DB)*\n\n"
            f"• *Total Trade Selesai:* `{total_closed}`\n"
            f"• 🎯 *Menang (TP):* `{wins}`\n"
            f"• 🛡️ *BEP (Modal Aman $0):* `{beps}`\n"
            f"• 🛑 *Kalah (SL):* `{losses}`\n"
            f"• 📊 *Real Win Rate (Win vs Loss):* `{adjusted_wr:.1f}%`\n"
            f"• 🛡️ *Capital Preserved (Win + BEP):* `{preserved_rate:.1f}%`\n"
            f"• *Total PnL Akumulasi:* `{pnl_prefix}${total_pnl:.2f}`\n"
            f"• *Posisi Masih Terbuka:* `{stats['open_trades']}`\n"
            "------------------------------------------\n"
        )

        if not trades:
            msg = header + "\n_Belum ada riwayat transaksi yang tercatat di database._"
            await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)
            return

        body = "*5 Transaksi Terakhir:*\n\n"
        for idx, t in enumerate(trades, 1):
            status = t["status"]
            if status == "OPEN":
                icon = "🟢 [OPEN]"
                res_str = f"Entry: `{t['entry_price']}` | SL: `{t['sl_price']}` | TP: `{t['tp_price']}`"
            else:
                pnl = t.get("profit_loss", 0.0)
                reason = t.get("close_reason", "CLOSED")
                if "BREAK_EVEN" in reason or "BEP" in reason or abs(pnl) <= 0.05:
                    icon = "🛡️ [BEP: $0.00]"
                elif pnl > 0.05 or "TP" in reason or "PROFIT" in reason:
                    icon = f"🎯 [TP: +${pnl:.2f}]"
                else:
                    icon = f"🛑 [SL: -${abs(pnl):.2f}]"
                res_str = f"Exit: `{t.get('close_price')}` ({t.get('close_time', '')})"

            body += (
                f"{idx}. {icon} `#{t['ticket']}` *{t['symbol']}* ({t['action']} {t['lot']} Lot)\n"
                f"   • Waktu Entry: `{t['entry_time']}`\n"
                f"   • {res_str}\n"
            )
            if t.get("ai_reasoning"):
                # Short snippet
                snippet = t["ai_reasoning"][:65] + "..." if len(t["ai_reasoning"]) > 65 else t["ai_reasoning"]
                body += f"   • _Catatan: {snippet}_\n"
            body += "\n"

        await update.message.reply_text(header + body, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_keys(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Display status of all Gemini API keys in the rotation pool."""
        if not self.gemini:
            await update.message.reply_text("Gemini Analyst tidak tersedia.")
            return

        status_list = self.gemini.get_pool_status()
        if not status_list:
            await update.message.reply_text(
                "⚠️ *Belum ada Gemini API Key yang terpasang!*\n\n"
                "Tambahkan API key baru dengan cara ketik:\n"
                "`/addkey <api_key_kamu>`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )
            return

        text = (
            f"🔑 *Status Pool Gemini API Key ({len(status_list)} Key Aktif)*\n\n"
            f"• *Penyimpanan:* 🗄️ SQLite Database (`api_keys`)\n"
            f"• *Model AI:* `{self.gemini.model}`\n"
            f"• *Metode:* Round-Robin (Rotasi bergantian per request)\n"
            f"• *Auto-Failover:* Otomatis pindah key jika terkena 429 Limit\n"
            "------------------------------------------\n"
        )

        for s in status_list:
            key_id = s.get("id", s["index"])
            text += (
                f"*[ID: #{key_id}]* Key: `{s['masked']}`\n"
                f"   • *Status:* {s['status']}\n"
                f"   • *Request:* `{s['requests']}` (Sukses: `{s['success']}` | Limit: `{s['rate_limits']}`)\n\n"
            )

        text += (
            "💡 *Manajemen Key (SQLite):*\n"
            "• Tambah: `/addkey <api_key>`\n"
            "• Hapus: `/delkey <id>` (contoh: `/delkey 3`)"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_addkey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add a new Gemini API key dynamically to SQLite database pool."""
        if not context.args or not context.args[0].strip():
            await update.message.reply_text(
                "ℹ️ *Format Perintah /addkey:*\n\n"
                "Ketik perintah diikuti API key kamu, contoh:\n"
                "`/addkey AIzaSyDXXXXX...`",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )
            return

        new_key = context.args[0].strip()
        if not self.gemini:
            await update.message.reply_text("Gemini Analyst tidak tersedia.")
            return

        added = self.gemini.add_key(new_key)
        if not added:
            await update.message.reply_text("⚠️ Key gagal ditambahkan (mungkin key tidak valid atau sudah ada di dalam pool).")
            return

        # Backup sync to .env as well
        try:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
                has_pool = False
                new_lines = []
                for line in lines:
                    if line.startswith("GEMINI_API_KEYS="):
                        has_pool = True
                        existing = line.split("=", 1)[1].strip()
                        if existing:
                            keys_list = [k.strip() for k in existing.split(",") if k.strip()]
                            if new_key not in keys_list:
                                keys_list.append(new_key)
                            new_lines.append(f"GEMINI_API_KEYS={','.join(keys_list)}")
                        else:
                            new_lines.append(f"GEMINI_API_KEYS={new_key}")
                    else:
                        new_lines.append(line)
                if not has_pool:
                    new_lines.append(f"GEMINI_API_KEYS={new_key}")
                env_path.write_text("\n".join(new_lines), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not sync new key to .env backup: {e}")

        masked = f"{new_key[:6]}...{new_key[-4:]}" if len(new_key) > 10 else new_key
        await update.message.reply_text(
            f"✅ *API Key Baru Berhasil Disimpan ke Database!*\n\n"
            f"• *Key:* `{masked}`\n"
            f"• *Penyimpanan:* 🗄️ SQLite Database (Tersimpan Permanen)\n"
            f"• *Total Key Aktif di Pool:* `{len(self.gemini._pool)}` Key\n\n"
            "Key ini sudah langsung aktif dalam rotasi Round-Robin & auto-failover!",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=REPLY_KEYBOARD
        )

    async def _cmd_delkey(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Delete a Gemini API key from SQLite database by its ID."""
        if not context.args or not context.args[0].strip():
            await update.message.reply_text(
                "ℹ️ *Format Perintah /delkey:*\n\n"
                "Ketik perintah diikuti nomor ID key yang mau dihapus, contoh:\n"
                "`/delkey 3`\n\n"
                "Cek nomor ID key dengan klik tombol *🔑 Cek Pool API*.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )
            return

        try:
            key_id = int(context.args[0].strip())
        except ValueError:
            await update.message.reply_text("⚠️ ID key harus berupa angka (contoh: `/delkey 2`).")
            return

        if not self.gemini:
            await update.message.reply_text("Gemini Analyst tidak tersedia.")
            return

        deleted = self.gemini.delete_key(key_id)
        if deleted:
            await update.message.reply_text(
                f"🗑️ *API Key ID #{key_id} Berhasil Dihapus!*\n\n"
                f"• *Sisa Key di Pool:* `{len(self.gemini._pool)}` Key aktif di database.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )
        else:
            await update.message.reply_text(
                f"⚠️ Key dengan ID #{key_id} tidak ditemukan di database pool.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.trading_state["is_paused"] = True
        logger.info("Auto-trade paused via Telegram /pause command.")
        await update.message.reply_text(
            "⏸️ *Auto-trading DIJEDA (PAUSED).*\n"
            "Bot tetap memantau pasar & mengirim analisa, tapi TIDAK akan memasang order baru.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=REPLY_KEYBOARD
        )

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.trading_state["is_paused"] = False
        logger.info("Auto-trade resumed via Telegram /resume command.")
        await update.message.reply_text(
            "▶️ *Auto-trading DILANJUTKAN (ACTIVE).*\n"
            "Order baru akan otomatis dieksekusi saat muncul setup teknikal & konfluensi valid.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=REPLY_KEYBOARD
        )

    async def _cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚨 *Menjalankan TUTUP DARURAT (PANIC CLOSE ALL)...*", parse_mode=ParseMode.MARKDOWN)
        closed_count = self.mt5.close_all_positions()
        await update.message.reply_text(
            f"🚨 *PANIC CLOSE SELESAI:*\nBerhasil menutup `{closed_count}` posisi terbuka di MT5.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=REPLY_KEYBOARD
        )

    async def _cmd_resetdb(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        deleted = self.db.reset_closed_trades()
        open_trades = self.db.get_open_trades()
        open_info = ""
        if open_trades:
            for t in open_trades:
                open_info += f"  • `#{t['ticket']}` *{t['symbol']}* ({t['action']} {t['lot']} Lot) @ `{t['entry_price']}`\n"
        else:
            open_info = "  _Tidak ada posisi terbuka._\n"

        msg = (
            "🧹 *DATABASE RIWAYAT BERHASIL DIBERSIHKAN!*\n\n"
            f"• *Data Closed Dihapus:* `{deleted}` riwayat\n"
            f"• *Posisi Aktif (OPEN) Tetap Disimpan:* `{len(open_trades)}` posisi\n\n"
            f"*Posisi Berjalan Saat Ini:*\n{open_info}\n"
            "Status database kini bersih dan siap mencatat transaksi berikutnya."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    def set_radar_provider(self, provider_func):
        """Set callback to query live radar summary from the main trading engine."""
        self.radar_provider = provider_func

    async def _cmd_radar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Query real-time market radar status for all monitored instruments."""
        self._register_active_chat(update)
        if self.radar_provider:
            if asyncio.iscoroutinefunction(self.radar_provider):
                msg = await self.radar_provider()
            else:
                msg = self.radar_provider()
        else:
            msg = "📡 *Radar Pasar:* Mesin analisa sedang bersiap. Silakan coba beberapa saat lagi."
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Configure periodic scan report frequency: /report 1h, /report 15m, /report off."""
        args = context.args
        if not args:
            await update.message.reply_text(
                f"⏱️ *Pengaturan Notifikasi Radar Pasar:*\n\n"
                f"• *Status Saat Ini:* `{self.scan_report_mode.upper()}`\n\n"
                "Gunakan perintah berikut untuk mengubah frekuensi:\n"
                "• `/report 1h` -> Laporan ringkas tiap 1 jam (Menit :00)\n"
                "• `/report 15m` -> Laporan setiap candle close M15\n"
                "• `/report off` -> Matikan laporan berkala (hanya jika ada sinyal)",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=REPLY_KEYBOARD
            )
            return

        choice = args[0].lower().strip()
        if choice in ("1h", "1jam", "hourly"):
            self.scan_report_mode = "1h"
            msg = "✅ *Frekuensi Laporan Radar:* Diubah ke `1 JAM` (Tiap menit :00)."
        elif choice in ("15m", "15min", "m15"):
            self.scan_report_mode = "15m"
            msg = "✅ *Frekuensi Laporan Radar:* Diubah ke `15 MENIT` (Tiap penutupan candle M15)."
        elif choice in ("off", "mati", "disable"):
            self.scan_report_mode = "off"
            msg = "⏸️ *Laporan Berkala Dimatikan.* Notifikasi hanya akan dikirim saat ada sinyal/eksekusi."
        else:
            msg = "⚠️ Pilihan tidak valid. Gunakan `/report 1h`, `/report 15m`, atau `/report off`."

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=REPLY_KEYBOARD)

    async def send_ai_rejection_alert(
        self,
        symbol: str,
        tech_signal: Any,
        ai_decision: Any,
        min_score: int = 85
    ):
        """Send transparent notification when Gemini AI rejects an otherwise valid technical setup."""
        if not self.is_configured():
            return

        now_wib = get_wib_str()
        score = getattr(ai_decision, "confluence_score", 0)
        sentiment = getattr(ai_decision, "news_sentiment", "NEUTRAL")
        raw_reasoning = getattr(ai_decision, "reasoning", "Konfluensi berita tidak mencukupi")
        clean_reasoning = (
            str(raw_reasoning or "")
            .replace("_", " ")
            .replace("*", "")
            .replace("`", "'")
            .strip()
        )
        action = getattr(tech_signal, "action", "SETUP")

        msg = (
            f"🤖 *[Analisis Gemini AI - {symbol} {action}]*\n\n"
            f"• *Status Eksekusi:* DIBATALKAN ❌\n"
            f"• *Skor Keyakinan AI:* `{score} / 100` *(Syarat minimal {min_score})*\n"
            f"• *Sentimen Berita:* `{sentiment}`\n"
            f"• *Alasan Penolakan AI:*\n_{clean_reasoning}_\n\n"
            f"• *Waktu Analisa:* `{now_wib}`\n"
            f"💡 _Bot mencegah entry karena konfluensi berita & sentimen tidak mendukung setup teknikal._"
        )
        await self.send_alert(msg)


if __name__ == "__main__":
    from check_telegram_bot import main as run_diag
    asyncio.run(run_diag())
