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

    # Second BEP check must NOT trigger again (in-memory deduplication)
    modified_again = rm.check_and_apply_breakeven(current_atr={"XAUUSD": 10.0})
    assert len(modified_again) == 0


def test_breakeven_float_tolerance_sell(risk_env):
    connector, rm = risk_env
    # SELL order on XAUUSD with 3 decimals entry (e.g. 3733.978)
    order = connector.execute_order(
        symbol="XAUUSD",
        order_type="SELL",
        volume=0.01,
        price=3733.978,
        sl=3745.00,
        tp=3700.00
    )
    ticket = order["ticket"]
    for p in connector._mock_positions:
        if p["ticket"] == ticket:
            p["price_open"] = 3733.978
            # In profit (price dropped to 3715.00)
            p["price_current"] = 3715.00

    # ATR = 5.0 -> target profit dist = 7.5 points
    res = rm.check_and_apply_breakeven(current_atr={"XAUUSD": 5.0})
    assert len(res) == 1
    assert res[0]["ticket"] == ticket

    # Broker rounded SL to 2 decimals (3733.98)
    for p in connector._mock_positions:
        if p["ticket"] == ticket:
            p["sl"] = 3733.98

    # Second check: despite 3733.98 > 3733.978, float tolerance and _bep_moved_tickets prevent re-triggering
    res2 = rm.check_and_apply_breakeven(current_atr={"XAUUSD": 5.0})
    assert len(res2) == 0

    # Simulate bot restart (new RiskManager instance) where broker position still has sl=3733.98
    rm_restarted = RiskManager(connector)
    res_restarted = rm_restarted.check_and_apply_breakeven(current_atr={"XAUUSD": 5.0})
    # Due to tolerance check (3733.98 <= 3733.978 + 0.05), it must NOT re-trigger!
    assert len(res_restarted) == 0


def test_daily_drawdown_immune_to_floating_restart_and_wd():
    class FakeMT5:
        def __init__(self, balance, equity):
            self.balance = balance
            self.equity = equity

        def get_account_info(self):
            return {
                "balance": self.balance,
                "equity": self.equity,
                "profit": self.equity - self.balance
            }

    class FakeDB:
        def __init__(self, today_pnl=0.0):
            self.today_pnl = today_pnl

        def get_today_pnl(self):
            return self.today_pnl

    # Case 1: Bot starts while a trade is floating +$3.61 (Balance=$100, Equity=$103.61)
    mt5 = FakeMT5(balance=100.0, equity=103.61)
    db = FakeDB(today_pnl=0.0)
    rm = RiskManager(mt5, db=db)
    rm.initialize_daily_equity()
    assert rm._daily_starting_equity == pytest.approx(100.0)

    # Trade closes at BEP ($0), next trade floats -$0.11 (Balance=$100, Equity=$99.89)
    mt5.balance = 100.0
    mt5.equity = 99.89
    is_ok, dd_pct, _ = rm.check_daily_drawdown()
    assert is_ok is True
    assert dd_pct == pytest.approx(0.11, abs=1e-2)

    # Case 2: Actual realized trading loss of -$3.20 on $100 account -> must breach 3.0%
    mt5.balance = 96.80
    mt5.equity = 96.80
    db.today_pnl = -3.20
    is_ok, dd_pct, msg = rm.check_daily_drawdown()
    assert is_ok is False
    assert dd_pct == pytest.approx(3.20, abs=1e-2)
    assert "DAILY DRAWDOWN BREACHED" in msg

    # Case 3: Manual /resetdd immediately clears halt
    rm.reset_drawdown()
    is_ok, dd_pct, _ = rm.check_daily_drawdown()
    assert is_ok is True
    assert dd_pct == pytest.approx(0.0)


def test_bep_database_persistence(tmp_path):
    from core.database import TradeDatabase
    db_file = tmp_path / "test_trades.db"
    db = TradeDatabase(str(db_file))

    # Record entry
    db.record_entry(
        ticket=999999,
        symbol="XAUUSD",
        action="SELL",
        lot=0.01,
        entry_price=3733.978,
        sl_price=3745.00,
        tp_price=3700.00
    )

    assert db.is_bep_alert_sent(999999) is False

    # Mark BEP alert sent
    db.mark_bep_alert_sent(999999)
    assert db.is_bep_alert_sent(999999) is True

    # Re-open DB instance (simulating app restart)
    db2 = TradeDatabase(str(db_file))
    assert db2.is_bep_alert_sent(999999) is True


def test_journal_metrics_and_csv_export(tmp_path):
    from core.database import TradeDatabase
    db_file = tmp_path / "journal_test.db"
    db = TradeDatabase(str(db_file))

    # 1. Add Win trade
    db.record_entry(ticket=1, symbol="XAUUSD", action="BUY", lot=0.01, entry_price=2650.0, sl_price=2640.0, tp_price=2670.0, ai_reasoning="Gold trend")
    db.record_close(ticket=1, close_price=2670.0, profit_loss=20.0, close_reason="TAKE_PROFIT")

    # 2. Add Loss trade
    db.record_entry(ticket=2, symbol="BTCUSD", action="SELL", lot=0.01, entry_price=64000.0, sl_price=64500.0, tp_price=63000.0, ai_reasoning="Crypto short")
    db.record_close(ticket=2, close_price=64500.0, profit_loss=-5.0, close_reason="STOP_LOSS")

    # 3. Add BEP trade
    db.record_entry(ticket=3, symbol="EURUSD", action="BUY", lot=0.01, entry_price=1.0850, sl_price=1.0800, tp_price=1.0950, ai_reasoning="Euro long")
    db.record_close(ticket=3, close_price=1.0850, profit_loss=0.0, close_reason="BREAK_EVEN")

    # 4. Add Open trade
    db.record_entry(ticket=4, symbol="USDJPY", action="BUY", lot=0.01, entry_price=155.0, sl_price=154.5, tp_price=156.0, ai_reasoning="JPY breakout")

    metrics = db.get_detailed_journal_metrics()
    assert metrics["total_closed"] == 3
    assert metrics["open_trades"] == 1
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["beps"] == 1
    assert metrics["adjusted_win_rate"] == 50.0
    assert metrics["capital_preserved_rate"] == pytest.approx(66.7, abs=1e-1)
    assert metrics["net_pnl"] == 15.0
    assert metrics["gross_profit"] == 20.0
    assert metrics["gross_loss"] == 5.0
    assert metrics["profit_factor"] == 4.0
    assert "XAUUSD" in metrics["by_symbol"]
    assert metrics["by_symbol"]["XAUUSD"]["wins"] == 1

    # Check CSV export
    csv_str = db.export_trades_to_csv()
    assert "Tiket,Waktu Entry (WIB)" in csv_str
    assert "XAUUSD" in csv_str
    assert "TAKE_PROFIT" in csv_str
    assert "Gold trend" in csv_str


def test_session_filter_strict_windows(risk_env, monkeypatch):
    from datetime import datetime, timezone, timedelta
    import core.risk_manager as rm_mod

    wib = timezone(timedelta(hours=7))
    _, rm = risk_env

    def _set_time(hour: int, minute: int = 15, weekday_day: int = 20):
        # 2026-11-20 is a Friday (weekday=4)
        fixed = datetime(2026, 11, weekday_day, hour, minute, 0, tzinfo=wib)

        class _FakeDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is not None else fixed.replace(tzinfo=None)

        monkeypatch.setattr(rm_mod, "datetime", _FakeDateTime)

    # Sesi Asia (09:15 WIB): USDJPY & BTC aktif; XAUUSD & EURUSD standby
    _set_time(9, 15)
    assert rm.is_session_allowed("USDJPY")[0] is True
    assert rm.is_session_allowed("BTCUSD")[0] is True
    assert rm.is_session_allowed("XAUUSD")[0] is False
    assert rm.is_session_allowed("EURUSD")[0] is False

    # Transisi Pra-Eropa (13:15 WIB): hanya BTC aktif; Forex & Gold standby
    _set_time(13, 15)
    assert rm.is_session_allowed("USDJPY")[0] is False
    assert rm.is_session_allowed("EURUSD")[0] is False
    assert rm.is_session_allowed("XAUUSD")[0] is False
    assert rm.is_session_allowed("BTCUSD")[0] is True

    # Sesi London (15:15 WIB): XAUUSD, EURUSD & BTC aktif; USDJPY STANDBY!
    _set_time(15, 15)
    assert rm.is_session_allowed("USDJPY")[0] is False
    assert rm.is_session_allowed("XAUUSD")[0] is True
    assert rm.is_session_allowed("EURUSD")[0] is True
    assert rm.is_session_allowed("BTCUSD")[0] is True

    # Sesi New York / Overlap (20:15 WIB) & NY malam (01:15 WIB): Semua pair aktif
    for hr in (20, 1):
        _set_time(hr, 15)
        assert rm.is_session_allowed("USDJPY")[0] is True
        assert rm.is_session_allowed("XAUUSD")[0] is True
        assert rm.is_session_allowed("EURUSD")[0] is True
        assert rm.is_session_allowed("BTCUSD")[0] is True

    # Rollover (04:30 WIB): Semua pair freeze
    _set_time(4, 30)
    assert rm.is_session_allowed("USDJPY")[0] is False
    assert rm.is_session_allowed("XAUUSD")[0] is False
    assert rm.is_session_allowed("EURUSD")[0] is False
    assert rm.is_session_allowed("BTCUSD")[0] is False
