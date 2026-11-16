"""
Diagnostic utility to verify Telegram Bot configuration and test interactive commands.
Usage:
    python check_telegram_bot.py
"""
import sys
import asyncio
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from bot.telegram_bot import TelegramBotManager


async def main():
    print("=" * 60)
    print("🤖 Telegram Bot Connection Diagnostic Tool")
    print("=" * 60)
    print(f"Bot Token:  {settings.TELEGRAM_BOT_TOKEN[:10]}...{settings.TELEGRAM_BOT_TOKEN[-5:] if settings.TELEGRAM_BOT_TOKEN else 'NOT SET'}")
    print(f"Chat ID:    {settings.TELEGRAM_CHAT_ID or 'NOT SET'}")
    print("-" * 60)

    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in .env file.")
        sys.exit(1)

    print("1. Initializing MT5 & Risk Manager for Telegram Context...")
    mt5 = MT5Connector()
    mt5.connect()
    risk_manager = RiskManager(mt5)
    trading_state = {
        "is_paused": False,
        "last_check_time": "Diagnostic Mode",
        "last_signals": {}
    }

    print("2. Starting Telegram Bot Manager...")
    bot_manager = TelegramBotManager(mt5, risk_manager, trading_state)

    if not bot_manager.is_configured():
        print("❌ Telegram Bot Manager reports not configured.")
        sys.exit(1)

    await bot_manager.start_bot()
    print("✅ Telegram Bot connected & polling active!")

    print("3. Sending test message to Telegram...")
    try:
        await bot_manager.send_alert(
            "🚀 *Bot Trading Connected!*\n\n"
            "Halo bro! Bot Telegram trading kamu sudah aktif dan terhubung ke MT5.\n"
            "Coba ketik perintah berikut di chat ini:\n"
            "• `/status` - Cek saldo, equity & posisi aktif\n"
            "• `/pause` - Jeda auto-trade\n"
            "• `/resume` - Lanjutkan auto-trade\n"
            "• `/closeall` - Tutup darurat semua posisi"
        )
        print("✅ Test message successfully sent to your Telegram chat!")
    except Exception as e:
        print(f"⚠️ Failed to send test notification: {e}")

    print("\n" + "=" * 60)
    print("🎉 Bot is currently listening for commands on Telegram!")
    print("💡 Buka aplikasi Telegram kamu, cari bot @traderkuybot dan coba kirim /status")
    print("Tekan Ctrl+C di terminal ini untuk berhenti.")
    print("=" * 60)

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping bot...")
    finally:
        await bot_manager.stop_bot()
        mt5.disconnect()
        print("Bot stopped successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")
