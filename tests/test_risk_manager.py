import pytest
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager


@pytest.fixture
def risk_env():
    connector = MT5Connector(mock_mode=True)
    connector.connect()
    rm = RiskManager(connector)
    rm.initialize_daily_equity(current_equity=10000.0)
    return connector, rm


def test_lot_size_xauusd(risk_env):
    connector, rm = risk_env
    # Balance: $10,000 | 1% Risk = $100
    # XAUUSD: tick_size = 0.01, tick_value = 1.0 ($1 per 0.01 move per lot)
    # Entry = 2650.00, SL = 2648.00 -> SL Dist = 2.00 (200 points / ticks)
    # Risk per 1.0 lot = (2.00 / 0.01) * 1.0 = $200
    # Raw lot = 100 / 200 = 0.50 lot
    lot = rm.calculate_lot_size(
        symbol="XAUUSD",
        entry_price=2650.00,
        sl_price=2648.00,
        account_balance=10000.0
    )
    assert lot == 0.50


def test_lot_size_eurusd(risk_env):
    connector, rm = risk_env
    # Balance: $10,000 | 1% Risk = $100
    # EURUSD: tick_size = 0.00001, tick_value = 1.0 ($1 per 0.00001 move per lot)
    # Entry = 1.08500, SL = 1.08300 -> SL Dist = 0.00200 (200 points)
    # Risk per 1.0 lot = (0.00200 / 0.00001) * 1.0 = $200
    # Raw lot = 100 / 200 = 0.50 lot
    lot = rm.calculate_lot_size(
        symbol="EURUSD",
        entry_price=1.08500,
        sl_price=1.08300,
        account_balance=10000.0
    )
    assert lot == 0.50


def test_lot_size_clamping(risk_env):
    connector, rm = risk_env
    # Tiny balance with wide SL -> should clamp to volume_min (0.01)
    lot_min = rm.calculate_lot_size(
        symbol="XAUUSD",
        entry_price=2650.00,
        sl_price=2500.00,
        account_balance=50.0
    )
    assert lot_min == 0.01


def test_daily_drawdown_limit(risk_env):
    connector, rm = risk_env
    # Starting equity is $10,000

    # SMR 1: Equity drops to $9,800 -> 2% drawdown (allowed)
    connector._mock_balance = 9800.0
    is_ok, dd_pct, msg = rm.check_daily_drawdown()
    assert is_ok is True
    assert pytest.approx(dd_pct, rel=1e-2) == 2.0

    # SMR 2: Equity drops to $9,650 -> 3.5% drawdown (exceeds 3.0% limit -> blocked)
    connector._mock_balance = 9650.0
    is_ok, dd_pct, msg = rm.check_daily_drawdown()
    assert is_ok is False
    assert dd_pct >= 3.0
    assert "DAILY DRAWDOWN BREACHED" in msg


def test_breakeven_trailing(risk_env):
    connector, rm = risk_env
    # Open a mock BUY position on XAUUSD
    order = connector.execute_order(
        symbol="XAUUSD",
        order_type="BUY",
        volume=0.1,
        price=2650.00,
        sl=2640.00,
        tp=2670.00
    )
    ticket = order["ticket"]

    # ATR is 10.0 -> BEP target is 1.5 * ATR = 15.0 points above entry (2665.00)
    # Current price reaches 2666.00 (profit > 1.5 * ATR)
    for p in connector._mock_positions:
        if p["ticket"] == ticket:
            p["price_current"] = 2666.00

    # Trigger BEP check
    modified = rm.check_and_apply_breakeven(current_atr={"XAUUSD": 10.0})
    assert any(m["ticket"] == ticket for m in modified)

    # Verify that position's SL was moved to Entry Price (2650.00)
    pos = [p for p in connector._mock_positions if p["ticket"] == ticket][0]
    assert pos["sl"] == 2650.00
