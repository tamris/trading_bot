"""
Diagnostic utility to verify live connection to MetaTrader 5 terminal and broker demo account.
Usage:
    python check_mt5_connection.py
"""
import sys
from config.settings import settings

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 package is not installed. Run: pip install MetaTrader5")
    sys.exit(1)


def main():
    print("=" * 60)
    print("🔍 MetaTrader 5 Connection Diagnostic Tool")
    print("=" * 60)
    print(f"Terminal Path: {settings.MT5_PATH}")
    print(f"Account Login: {settings.MT5_LOGIN}")
    print(f"Server Name:   {settings.MT5_SERVER}")
    print("-" * 60)

    # 1. Initialize Terminal
    print("1. Initializing terminal...")
    initialized = mt5.initialize(
        path=settings.MT5_PATH,
        login=settings.MT5_LOGIN if settings.MT5_LOGIN > 0 else None,
        password=settings.MT5_PASSWORD if settings.MT5_PASSWORD else None,
        server=settings.MT5_SERVER if settings.MT5_SERVER else None,
        timeout=30000
    )

    if not initialized:
        error = mt5.last_error()
        print(f"❌ Failed to initialize MT5: {error}")
        print("\n💡 Troubleshooting Tips:")
        print("  - Make sure MetaTrader 5 is installed at the specified MT5_PATH.")
        print("  - Launch MetaTrader 5 manually once to complete initial broker setup.")
        sys.exit(1)

    print("✅ Terminal initialized successfully.")

    # 2. Account Login
    if settings.MT5_LOGIN > 0 and settings.MT5_PASSWORD and settings.MT5_PASSWORD != "your_demo_password_here":
        print(f"2. Logging into account {settings.MT5_LOGIN} on {settings.MT5_SERVER}...")
        authorized = mt5.login(
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER
        )
        if not authorized:
            print(f"❌ Login failed: {mt5.last_error()}")
            print("💡 Please double-check MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file.")
            mt5.shutdown()
            sys.exit(1)
        print("✅ Account authorized successfully.")
    else:
        print("⚠️ MT5_LOGIN or MT5_PASSWORD not set or using placeholders. Skipping broker authentication.")

    # 3. Account Information
    acc = mt5.account_info()
    if acc:
        print("\n📊 Account Overview:")
        print(f"  • Login:       {acc.login}")
        print(f"  • Server:      {acc.server}")
        print(f"  • Currency:    {acc.currency}")
        print(f"  • Balance:     ${acc.balance:,.2f}")
        print(f"  • Equity:      ${acc.equity:,.2f}")
        print(f"  • Leverage:    1:{acc.leverage}")
        print(f"  • Trade Mode:  {'Demo' if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'Real'}")

    # 4. Symbol Availability & Quotes
    print("\n📈 Symbol Verification:")
    for symbol in settings.SYMBOLS:
        selected = mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info and tick:
            print(f"  • {symbol:8} Bid: {tick.bid:<10} Ask: {tick.ask:<10} Spread: {info.spread} pts [OK]")
        else:
            print(f"  • {symbol:8} ⚠️ Could not fetch quotes or symbol not found in MarketWatch.")

    mt5.shutdown()
    print("\n" + "=" * 60)
    print("🎉 All MT5 diagnostic checks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
