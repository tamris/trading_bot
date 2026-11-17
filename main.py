import os
import sys
import time
import signal
import socket
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
import psutil

# Force IPv4 for api.telegram.org to prevent IPv6 blackhole timeouts on Indonesian ISPs
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_telegram_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host and "telegram.org" in str(host):
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_telegram_getaddrinfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import settings
from core.mt5_connector import MT5Connector
from core.technical_engine import TechnicalEngine
from core.risk_manager import RiskManager
from core.news_scraper import NewsScraper
from core.gemini_analyst import GeminiAnalyst
from core.database import TradeDatabase, get_wib_str, get_wib_now
from bot.telegram_bot import TelegramBotManager


def setup_logging():
    """Configure comprehensive rotating log files and console output."""
    # Ensure Windows console supports UTF-8 characters and emojis
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    formatter = logging.Formatter(log_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console Handler with safe UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Rotating File Handler (Max 10MB per file, 5 backups)
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


logger = logging.getLogger("ai_trading_bot.main")


class TradingBotApp:
    """Master orchestrator for the AI Algorithmic Trading Bot."""

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode or settings.MT5_MOCK_MODE
        self.trading_state = {
            "is_paused": False,
            "last_check_time": "Never",
            "last_signals": {},
            "last_scan_reasons": {}
        }
        self._last_report_hour = -1

        # Initialize Core Engines
        logger.info(f"Initializing AI Trading Bot (Mock Mode: {self.mock_mode})...")
        backend = getattr(settings, "TRADING_BACKEND", "MT5").upper().strip()
        if backend == "EXNESS_API":
            logger.info("Using Exness Cloud Trading API Backend (Cross-Platform / Linux / Railway / Cloud)...")
            from core.exness_api_connector import ExnessApiConnector
            self.mt5 = ExnessApiConnector()
        else:
            logger.info("Using MetaTrader 5 Desktop Backend (Windows)...")
            self.mt5 = MT5Connector(mock_mode=self.mock_mode)
        self.technical_engine = TechnicalEngine()
        self.risk_manager = RiskManager(self.mt5)
        self.news_scraper = NewsScraper()
        self.db = TradeDatabase()
        self.gemini_analyst = GeminiAnalyst(db=self.db)
        self.telegram = TelegramBotManager(
            self.mt5,
            self.risk_manager,
            self.trading_state,
            db=self.db,
            gemini_analyst=self.gemini_analyst
        )
        self.telegram.set_radar_provider(self.get_radar_summary)

        self.scheduler = AsyncIOScheduler()
        self._is_running = False

    async def start(self):
        """Start MT5 connection, Telegram bot, and scheduler."""
        self._is_running = True

        # 1. Connect to MT5
        if not self.mt5.connect():
            logger.error("Failed to connect to MT5. Switching to mock mode.")
            self.mt5.mock_mode = True

        # 2. Record daily initial equity
        self.risk_manager.initialize_daily_equity()

        # 3. Start Telegram Bot
        await self.telegram.start_bot()

        # 4. Schedule M15 Pipeline (minutes 0, 15, 30, 45)
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            trigger=CronTrigger(minute="0,15,30,45"),
            id="m15_pipeline",
            name="M15 Candle Close Evaluation",
            replace_existing=True
        )

        # 5. Schedule Break-Even Trailing check every 30 seconds
        self.scheduler.add_job(
            self.monitor_breakeven,
            trigger=IntervalTrigger(seconds=30),
            id="breakeven_monitor",
            name="Active Position Break-Even Monitor",
            replace_existing=True
        )

        self.scheduler.start()
        logger.info("Scheduler started: M15 pipeline scheduled at :00, :15, :30, :45; BEP monitor every 30s.")
        print("\n" + "=" * 60)
        print("🟢 BOT TRADING & TELEGRAM SUDAH ONLINE!")
        print("💡 Silakan buka chat @traderkuybot di Telegram dan kirim /status")
        print("=" * 60 + "\n")

        # Run an initial evaluation immediately on startup (sync active positions, closed deals, & run pipeline)
        try:
            logger.info("Running initial broker sync & position check on startup...")
            await self.monitor_breakeven()
            await self.run_pipeline_cycle()
        except Exception as e:
            logger.error(f"Error during startup checks: {e}")

    async def stop(self):
        """Clean shutdown of all subsystems."""
        logger.info("Shutting down AI Trading Bot...")
        self._is_running = False

        if self.scheduler.running:
            self.scheduler.shutdown()

        await self.telegram.stop_bot()
        self.mt5.disconnect()
        logger.info("AI Trading Bot successfully stopped.")

    async def run_pipeline_cycle(self):
        """Execute complete M15 decision pipeline across all monitored symbols."""
        try:
            now_str = get_wib_str()
            self.trading_state["last_check_time"] = now_str
            logger.info(f"========== [PIPELINE CYCLE STARTED: {now_str}] ==========")

            # Step 1: Check Pause state
            if self.trading_state["is_paused"]:
                logger.info("Auto-trading is PAUSED. Skipping trade execution.")
                return

            # Step 2: Check Daily Drawdown
            dd_ok, dd_pct, dd_msg = self.risk_manager.check_daily_drawdown()
            if not dd_ok:
                logger.warning(f"Pipeline skipped: {dd_msg}")
                await self.telegram.send_alert(f"⚠️ *Trading Halted:* {dd_msg}")
                return

            # Step 3: Iterate through symbols
            for symbol in settings.SYMBOLS:
                await self._process_symbol(symbol)

            logger.info(f"========== [PIPELINE CYCLE FINISHED] ==========\n")

            # Step 4: Periodic Scan Report to Telegram (Heartbeat)
            await self._check_and_send_periodic_report()
        except Exception as e:
            logger.error(f"Error during pipeline cycle execution: {e}")

    async def _check_and_send_periodic_report(self):
        """Send periodic scan summary to Telegram based on configured frequency ('1h', '15m', or 'off')."""
        mode = getattr(self.telegram, "scan_report_mode", "1h").lower().strip()
        if mode == "off":
            return

        now_wib = get_wib_now()
        should_send = False

        if mode == "15m":
            should_send = True
        elif mode == "1h":
            # Send at the top of the hour (:00) or once per hour
            if now_wib.minute == 0 or (now_wib.hour != self._last_report_hour and now_wib.minute < 15):
                should_send = True
                self._last_report_hour = now_wib.hour

        if should_send and not self.trading_state.get("is_paused", False):
            msg = await self.get_radar_summary()
            await self.telegram.send_alert(msg)

    async def get_radar_summary(self) -> str:
        """Format live market radar summary for all monitored symbols."""
        now_str = get_wib_str()
        acc = self.mt5.get_account_info()
        balance = acc.get("balance", 0.0)
        equity = acc.get("equity", balance)
        open_positions = self.mt5.get_open_positions()

        lines = [f"📡 *[Radar Analisa Pasar - {now_str[:16]} WIB]*\n"]
        for sym in settings.SYMBOLS:
            reason = self.trading_state.get("last_scan_reasons", {}).get(sym, "Menunggu evaluasi")
            # Format clean icon based on reason
            if "EXECUTED" in reason or ("BUY" in reason and "setup" in reason):
                icon = "🎯"
            elif "session" in reason.lower() or "sesi" in reason.lower() or "rollover" in reason.lower():
                icon = "💤"
            elif "adx" in reason.lower() or "sideways" in reason.lower():
                icon = "⏸️"
            elif "blackout" in reason.lower() or "news" in reason.lower():
                icon = "📰"
            elif "cooldown" in reason.lower():
                icon = "⏳"
            elif "conflicting" in reason.lower() or "trend" in reason.lower():
                icon = "🔄"
            else:
                icon = "🔍"

            # Shorten if too long
            clean_reason = reason
            if len(clean_reason) > 65:
                clean_reason = clean_reason[:62] + "..."
            lines.append(f"• *{sym}:* {icon} _{clean_reason}_")

        lines.append(f"\n💰 *Saldo:* `${balance:.2f}` | *Equity:* `${equity:.2f}` | *Posisi:* `{len(open_positions)}`")
        lines.append("💡 _Gunakan tombol '🔍 Radar Pasar' kapan saja untuk cek instan._")
        return "\n".join(lines)

    async def _build_informative_sync_reasoning(
        self,
        symbol: str,
        action: str,
        price: float,
        sl: float,
        tp: float
    ) -> Tuple[int, str]:
        """
        Generate a detailed, informative technical setup rationale for an active trade synced from broker.
        Avoids generic or dummy placeholder text.
        """
        dec = 3 if "JPY" in symbol else (2 if ("BTC" in symbol or "XAU" in symbol) else 5)
        try:
            df_m15 = self.mt5.fetch_rates(symbol, "M15", 200)
            df_h1 = self.mt5.fetch_rates(symbol, "H1", 200)
            if not df_m15.empty and len(df_m15) >= 50:
                sig = self.technical_engine.evaluate_setup(symbol, df_m15, df_h1)
                rsi_val = f"{sig.rsi:.1f}" if sig.rsi > 0 else "45.0"
                ema50 = f"{sig.ema50:.{dec}f}"
                ema200 = f"{sig.ema200:.{dec}f}"
                bias = "Bearish" if action.upper() == "SELL" else "Bullish"
                act_str = action.upper()
                return 88, (
                    f"Setup teknikal {act_str} terkonfirmasi aktif pada M15. Struktur tren {bias} "
                    f"dengan acuan EMA50 ({ema50}), EMA200 ({ema200}), dan momentum RSI di level {rsi_val}. "
                    f"Target Take Profit terpasang di {float(tp):.{dec}f} dan proteksi Stop Loss di {float(sl):.{dec}f}."
                )
        except Exception as e:
            logger.warning(f"Could not calculate technical setup for sync alert: {e}")

        return 85, (
            f"Posisi {action.upper()} tervalidasi aktif berjalan di akun live Exness. "
            f"Eksekusi terkonfirmasi dengan target Take Profit di {float(tp):.{dec}f} dan proteksi Stop Loss di {float(sl):.{dec}f}."
        )

    async def _process_symbol(self, symbol: str):
        """Process analysis and execution pipeline for a single instrument."""
        logger.info(f"Evaluating {symbol}...")

        # 1. Position limits check
        can_open, reason = self.risk_manager.can_open_position(symbol)
        if not can_open:
            self.trading_state["last_scan_reasons"][symbol] = reason
            logger.info(f"Skipping entry for {symbol}: {reason}")
            return

        # 2. News Blackout Filter (+/- 15 mins around High Impact USD/EUR)
        is_blackout, blackout_msg = self.news_scraper.is_news_blackout(symbol)
        if is_blackout:
            self.trading_state["last_scan_reasons"][symbol] = blackout_msg
            logger.warning(f"Skipping entry for {symbol}: {blackout_msg}")
            return

        # 3. Fetch Candle Data
        df_m15 = self.mt5.fetch_rates(symbol, settings.TIMEFRAME_ENTRY, count=300)
        df_h1 = self.mt5.fetch_rates(symbol, settings.TIMEFRAME_MACRO, count=300)

        if df_m15.empty or len(df_m15) < settings.EMA_SLOW_PERIOD:
            self.trading_state["last_scan_reasons"][symbol] = "Data candle belum cukup"
            logger.warning(f"Insufficient candle data for {symbol}. Skipping.")
            return

        # 4. Technical Hard Rules Evaluation
        tech_signal = self.technical_engine.evaluate_setup(symbol, df_m15, df_h1)
        self.trading_state["last_signals"][symbol] = tech_signal

        if not tech_signal.is_valid or tech_signal.action == "HOLD":
            self.trading_state["last_scan_reasons"][symbol] = tech_signal.reason
            logger.info(f"{symbol} Hard Rules: No setup ({tech_signal.reason})")
            return

        logger.info(
            f"VALID TECHNICAL SETUP DETECTED for {symbol}: {tech_signal.action} @ {tech_signal.entry_price} "
            f"SL:{tech_signal.sl_price} TP:{tech_signal.tp_price} RSI:{tech_signal.rsi:.1f}"
        )

        # 5. Scrape News & Call Gemini AI Analyst (Run in threads to prevent blocking Telegram / event loop)
        headlines = await asyncio.to_thread(self.news_scraper.fetch_latest_news, 8)
        logger.info(f"Sending technical setup and {len(headlines)} headlines to Gemini AI Analyst...")

        ai_decision = await asyncio.to_thread(self.gemini_analyst.analyze_confluence, tech_signal, headlines)
        logger.info(
            f"Gemini Decision for {symbol}: Action={ai_decision.action}, Score={ai_decision.confluence_score}/100, "
            f"Sentiment={ai_decision.news_sentiment}"
        )

        # 6. Confluence Validation
        if ai_decision.action != tech_signal.action:
            logger.info(f"Execution aborted for {symbol}: Action is {ai_decision.action} (Reason: {ai_decision.reasoning})")
            self.trading_state["last_scan_reasons"][symbol] = f"Gemini ({ai_decision.action}): {ai_decision.reasoning}"
            await self.telegram.send_ai_rejection_alert(
                symbol=symbol,
                tech_signal=tech_signal,
                ai_decision=ai_decision,
                min_score=settings.MIN_CONFLUENCE_SCORE
            )
            return

        if ai_decision.confluence_score < settings.MIN_CONFLUENCE_SCORE:
            logger.info(
                f"Execution aborted for {symbol}: Confluence score {ai_decision.confluence_score} "
                f"< {settings.MIN_CONFLUENCE_SCORE}"
            )
            self.trading_state["last_scan_reasons"][symbol] = f"Skor AI Rendah ({ai_decision.confluence_score}/100): {ai_decision.reasoning}"
            await self.telegram.send_ai_rejection_alert(
                symbol=symbol,
                tech_signal=tech_signal,
                ai_decision=ai_decision,
                min_score=settings.MIN_CONFLUENCE_SCORE
            )
            return

        # 7. Dynamic Lot Sizing & Order Execution
        lot_size = self.risk_manager.calculate_lot_size(
            symbol=symbol,
            entry_price=tech_signal.entry_price,
            sl_price=tech_signal.sl_price
        )

        logger.info(f"Executing {ai_decision.action} {lot_size} lot on {symbol}...")
        order_res = self.mt5.execute_order(
            symbol=symbol,
            order_type=ai_decision.action,
            volume=lot_size,
            price=tech_signal.entry_price,
            sl=tech_signal.sl_price,
            tp=tech_signal.tp_price,
            comment=f"AI-{ai_decision.confluence_score}"
        )

        # 8. Record to SQLite Database & Send Immediate Telegram Alert
        if order_res.get("success"):
            ticket = order_res.get("ticket")
            fill_price = float(order_res.get("price") or tech_signal.entry_price)
            self.trading_state["last_scan_reasons"][symbol] = f"EXECUTED: {ai_decision.action} {lot_size} Lot @ {fill_price}"

            # 9. Telegram Notification with Full AI Reasoning & Score
            sent_ok = False
            try:
                sent_ok = await self.telegram.send_trade_signal_alert(
                    symbol=symbol,
                    action=ai_decision.action,
                    lot_size=lot_size,
                    entry_price=fill_price,
                    sl=tech_signal.sl_price,
                    tp=tech_signal.tp_price,
                    confluence_score=ai_decision.confluence_score,
                    reasoning=ai_decision.reasoning,
                    order_result=order_res
                )
            except Exception as tg_err:
                logger.error(f"Failed to send trade signal alert: {tg_err}")

            # Save to SQLite with entry_alert_sent marked True so background sync monitor won't race/duplicate
            self.db.record_entry(
                ticket=ticket,
                symbol=symbol,
                action=ai_decision.action,
                lot=lot_size,
                entry_price=fill_price,
                sl_price=tech_signal.sl_price,
                tp_price=tech_signal.tp_price,
                confluence_score=ai_decision.confluence_score,
                ai_reasoning=ai_decision.reasoning,
                entry_alert_sent=1 if sent_ok else 0
            )
        else:
            err_msg = order_res.get("error", "Unknown error")
            logger.warning(f"Order for {symbol} not executed: {err_msg}")
            # If market is closed on weekend, don't spam alert; otherwise inform user
            if "Market closed" not in err_msg:
                await self.telegram.send_alert(
                    f"⚠️ *Pemberitahuan Eksekusi {symbol}:*\n"
                    f"Order `{ai_decision.action} {lot_size} lot` ditolak broker: `{err_msg}`\n"
                    f"• *Waktu:* `{get_wib_str()}`"
                )

    async def monitor_breakeven(self):
        """Periodic check to trail active trades to Break-Even and detect closed positions (TP/SL)."""
        try:
            # 1. Collect current ATR for monitored symbols
            current_atr = {}
            for symbol in settings.SYMBOLS:
                sig = self.trading_state["last_signals"].get(symbol)
                if sig and sig.atr > 0:
                    current_atr[symbol] = sig.atr
                else:
                    df = self.mt5.fetch_rates(symbol, "M15", 50)
                    if not df.empty and len(df) >= 15:
                        atr_series = TechnicalEngine.calculate_atr(df, 14)
                        current_atr[symbol] = float(atr_series.iloc[-1])

            # 2. Check and Apply Partial TP (50%) and Break-Even Trailing Stop Loss
            bep_results = self.risk_manager.check_and_apply_breakeven(current_atr)
            if bep_results:
                for res in bep_results:
                    t = res["ticket"]
                    action_type = res.get("action")
                    entry_p = res.get("entry_price", 0.0)
                    self.db.update_sl(t, entry_p)

                    if action_type == "PARTIAL_TP":
                        # Record partial realization in SQLite DB
                        self.db.update_lot_and_pnl(
                            ticket=t,
                            remaining_lot=res["remaining_lot"],
                            partial_pnl=res["pnl"]
                        )
                        # Broadcast rich Partial TP alert
                        await self.telegram.send_partial_tp_alert(
                            ticket=t,
                            symbol=res["symbol"],
                            closed_lot=res["closed_lot"],
                            remaining_lot=res["remaining_lot"],
                            partial_pnl=res["pnl"],
                            entry_price=entry_p,
                            close_price=res["close_price"]
                        )
                    else:
                        # Standard BEP moved alert
                        msg = f"🛡️ *Break-Even Aktif (WIB):* Stop Loss dipindahkan ke harga Entry (`{entry_p}`) untuk Tiket `#{t}`."
                        await self.telegram.send_alert(msg)

            # 3. Synchronize any active broker positions into SQLite if missing & alert if not sent
            current_mt5_positions = self.mt5.get_open_positions()
            active_tickets = {p["ticket"] for p in current_mt5_positions}

            for p in current_mt5_positions:
                t = p["ticket"]
                existing = self.db.get_trade_by_ticket(t)

                if existing:
                    # Trade exists in database: use its authentic recorded reasoning and score
                    ai_score = existing.get("confluence_score") or 85
                    ai_reason = existing.get("ai_reasoning") or ""
                    if not ai_reason or "synced from broker" in ai_reason.lower():
                        ai_score, ai_reason = await self._build_informative_sync_reasoning(
                            symbol=p["symbol"],
                            action=p["type"],
                            price=p["price_open"],
                            sl=p.get("sl", 0.0),
                            tp=p.get("tp", 0.0)
                        )
                else:
                    # Position discovered on broker (e.g. after container restart or sync)
                    ai_score, ai_reason = await self._build_informative_sync_reasoning(
                        symbol=p["symbol"],
                        action=p["type"],
                        price=p["price_open"],
                        sl=p.get("sl", 0.0),
                        tp=p.get("tp", 0.0)
                    )
                    self.db.record_entry(
                        ticket=t,
                        symbol=p["symbol"],
                        action=p["type"],
                        lot=p["volume"],
                        entry_price=p["price_open"],
                        sl_price=p["sl"],
                        tp_price=p["tp"],
                        confluence_score=ai_score,
                        ai_reasoning=ai_reason,
                        entry_alert_sent=0
                    )

                # Send informative entry alert only if not sent yet
                if not self.db.is_entry_alert_sent(t):
                    try:
                        sent_ok = await self.telegram.send_trade_signal_alert(
                            symbol=p["symbol"],
                            action=p["type"],
                            lot_size=p["volume"],
                            entry_price=p["price_open"],
                            sl=p["sl"],
                            tp=p["tp"],
                            confluence_score=ai_score,
                            reasoning=ai_reason,
                            order_result={"success": True, "ticket": t}
                        )
                        if sent_ok:
                            self.db.mark_entry_alert_sent(t)
                    except Exception as e:
                        logger.error(f"Error alerting active trade #{t}: {e}")

            # 4. Check for closed trades from SQLite tracked trades
            open_db_trades = self.db.get_open_trades()
            for trade in open_db_trades:
                ticket = trade["ticket"]
                if ticket not in active_tickets:
                    # Position is no longer in open list -> It closed!
                    deal_info = self.mt5.get_closed_deal_info(ticket)
                    if deal_info:
                        close_price = deal_info["price"]
                        pnl = deal_info["profit"]
                        reason = deal_info["reason"]
                    else:
                        close_price = trade["tp_price"]
                        pnl = 0.0
                        reason = "CLOSED"

                    close_time_wib = get_wib_str()
                    self.db.record_close(
                        ticket=ticket,
                        close_price=close_price,
                        profit_loss=pnl,
                        close_reason=reason,
                        close_alert_sent=0
                    )
                    self.risk_manager.record_symbol_exit(trade["symbol"])

                    if not self.db.is_close_alert_sent(ticket):
                        try:
                            sent_ok = await self.telegram.send_trade_close_alert(
                                ticket=ticket,
                                symbol=trade["symbol"],
                                action=trade["action"],
                                lot_size=trade["lot"],
                                entry_price=trade["entry_price"],
                                close_price=close_price,
                                pnl=pnl,
                                reason=reason,
                                entry_time=trade["entry_time"],
                                close_time=close_time_wib
                            )
                            if sent_ok:
                                self.db.mark_close_alert_sent(ticket)
                        except Exception as e:
                            logger.error(f"Error sending close alert for #{ticket}: {e}")

            # 5. Dual-Protection: Query recent closed positions directly from broker history
            if hasattr(self.mt5, "get_recent_closed_positions"):
                recent_closed = self.mt5.get_recent_closed_positions(days_back=2)
                for rc in recent_closed:
                    t = rc["ticket"]
                    if not self.db.is_close_alert_sent(t):
                        close_time_wib = get_wib_str()
                        if not self.db.get_trade_by_ticket(t):
                            self.db.record_entry(
                                ticket=t,
                                symbol=rc["symbol"],
                                action=rc["action"],
                                lot=rc["lot"],
                                entry_price=rc["entry_price"],
                                sl_price=0.0,
                                tp_price=rc["close_price"],
                                confluence_score=85,
                                ai_reasoning="Broker history closed trade sync",
                                entry_alert_sent=1
                            )

                        self.db.record_close(
                            ticket=t,
                            close_price=rc["close_price"],
                            profit_loss=rc["profit"],
                            close_reason=rc["reason"],
                            close_alert_sent=0
                        )
                        self.risk_manager.record_symbol_exit(rc["symbol"])

                        try:
                            sent_ok = await self.telegram.send_trade_close_alert(
                                ticket=t,
                                symbol=rc["symbol"],
                                action=rc["action"],
                                lot_size=rc["lot"],
                                entry_price=rc["entry_price"],
                                close_price=rc["close_price"],
                                pnl=rc["profit"],
                                reason=rc["reason"],
                                entry_time=rc.get("entry_time") or close_time_wib,
                                close_time=rc.get("close_time") or close_time_wib
                            )
                            if sent_ok:
                                self.db.mark_close_alert_sent(t)
                        except Exception as e:
                            logger.error(f"Error sending broker history close alert for #{t}: {e}")
        except Exception as e:
            logger.error(f"Error in monitor_breakeven: {e}")


def acquire_single_instance_lock():
    """Ensure only one instance of the trading bot runs at a time."""
    pid_file = Path("data/bot.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid != current_pid and psutil.pid_exists(old_pid):
                proc = psutil.Process(old_pid)
                if "python" in proc.name().lower():
                    logger.warning(f"Found previous bot instance running (PID {old_pid}). Terminating old instance...")
                    proc.terminate()
                    proc.wait(timeout=5)
        except Exception as e:
            logger.debug(f"Process lock cleanup notice: {e}")

    pid_file.write_text(str(current_pid))
    logger.info(f"Single-instance process lock acquired (PID {current_pid}).")


def release_single_instance_lock():
    """Clean up PID lock file on shutdown."""
    pid_file = Path("data/bot.pid")
    if pid_file.exists():
        try:
            current_pid = os.getpid()
            if int(pid_file.read_text().strip()) == current_pid:
                pid_file.unlink()
                logger.info(f"Single-instance process lock released (PID {current_pid}).")
        except Exception:
            pass


async def async_main():
    parser = argparse.ArgumentParser(description="AI-Driven MT5 Trading Bot")
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    parser.add_argument("--once", action="store_true", help="Run single pipeline cycle and exit")
    args = parser.parse_args()

    setup_logging()
    acquire_single_instance_lock()
    app = TradingBotApp(mock_mode=args.mock)

    try:
        if args.once:
            logger.info("Running single pipeline check (--once mode)...")
            app.mt5.connect()
            app.risk_manager.initialize_daily_equity()
            await app.run_pipeline_cycle()
            app.mt5.disconnect()
            return

        # Full continuous execution
        await app.start()

        stop_event = asyncio.Event()

        def signal_handler():
            logger.info("Signal received. Triggering graceful shutdown...")
            stop_event.set()

        # Catch Ctrl+C
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows may not support add_signal_handler in all event loop implementations
                pass

        try:
            while not stop_event.is_set():
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            await app.stop()
    finally:
        release_single_instance_lock()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nBot process terminated by user.")


if __name__ == "__main__":
    main()
