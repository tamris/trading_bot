import pytest
from datetime import datetime, timedelta, timezone
from core.database import WIB
from core.journal import JournalTrade, TradingJournal, export_journal_excel, shift_month
from core.journal_render import render_dashboard, render_calendar, render_trade_log


@pytest.fixture
def sample_trades():
    now = datetime(2026, 11, 19, 14, 0, 0, tzinfo=WIB)
    return [
        JournalTrade(
            ticket=1001,
            symbol="XAUUSD",
            action="SELL",
            lot=0.01,
            entry_price=3733.98,
            exit_price=3720.00,
            sl_price=3733.98,
            tp_price=3703.85,
            pnl=13.98,
            entry_dt=now - timedelta(hours=4),
            close_dt=now - timedelta(hours=1),
            status="CLOSED",
            close_reason="TAKE_PROFIT",
            confluence_score=88,
            ai_reasoning="Strong bearish trend with USD strength"
        ),
        JournalTrade(
            ticket=1002,
            symbol="BTCUSD",
            action="SELL",
            lot=0.01,
            entry_price=63500.0,
            exit_price=63800.0,
            sl_price=63800.0,
            tp_price=62000.0,
            pnl=-3.00,
            entry_dt=now - timedelta(hours=6),
            close_dt=now - timedelta(hours=3),
            status="CLOSED",
            close_reason="STOP_LOSS",
            confluence_score=85,
            ai_reasoning="BTC rejection"
        ),
        JournalTrade(
            ticket=1003,
            symbol="EURUSD",
            action="BUY",
            lot=0.01,
            entry_price=1.0850,
            exit_price=1.0850,
            sl_price=1.0850,
            tp_price=1.0920,
            pnl=0.0,
            entry_dt=now - timedelta(hours=8),
            close_dt=now - timedelta(hours=5),
            status="CLOSED",
            close_reason="BREAK_EVEN",
            confluence_score=85,
            ai_reasoning="Euro bounce"
        ),
        JournalTrade(
            ticket=1004,
            symbol="XAUUSD",
            action="SELL",
            lot=0.01,
            entry_price=3740.0,
            exit_price=3732.0,
            sl_price=3740.0,
            tp_price=3710.0,
            pnl=8.0,
            entry_dt=now - timedelta(hours=2),
            close_dt=None,
            status="OPEN",
            confluence_score=85,
            ai_reasoning="Gold second leg"
        )
    ]


def test_trading_journal_metrics(sample_trades):
    stats = TradingJournal.compute_stats(sample_trades)
    assert stats["total_closed"] == 3
    assert stats["open_trades"] == 1
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["beps"] == 1
    assert stats["win_rate"] == 50.0
    assert stats["capital_preserved_rate"] == pytest.approx(66.7, abs=1e-1)
    assert stats["net_pnl"] == 10.98
    assert stats["gross_profit"] == 13.98
    assert stats["gross_loss"] == 3.00
    assert stats["profit_factor"] == pytest.approx(4.66, abs=1e-2)
    assert "XAUUSD" in stats["by_symbol"]
    assert stats["by_symbol"]["XAUUSD"]["wins"] == 1


def test_render_dashboard_png(sample_trades):
    stats = TradingJournal.compute_stats(sample_trades)
    account = {"balance": 100.0, "equity": 108.0}
    png_bytes = render_dashboard(stats, account=account, title="TEST JOURNAL", period_label="Testing")
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_calendar_png(sample_trades):
    stats = TradingJournal.compute_stats(sample_trades)
    cal_bytes = render_calendar(stats["daily"], year=2026, month=11)
    assert isinstance(cal_bytes, bytes)
    assert len(cal_bytes) > 1000
    assert cal_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_trade_log_png(sample_trades):
    log_bytes = render_trade_log(sample_trades, limit=10)
    assert isinstance(log_bytes, bytes)
    assert len(log_bytes) > 1000
    assert log_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_export_excel_workbook(sample_trades):
    stats = TradingJournal.compute_stats(sample_trades)
    xlsx_bytes = export_journal_excel(sample_trades, stats)
    assert isinstance(xlsx_bytes, bytes)
    assert len(xlsx_bytes) > 1000
    # ZIP/XLSX signature: PK\x03\x04
    assert xlsx_bytes.startswith(b"PK\x03\x04")


def test_shift_month():
    assert shift_month(2026, 11, -1) == (2026, 10)
    assert shift_month(2026, 11, 1) == (2026, 12)
    assert shift_month(2026, 1, -1) == (2025, 12)
    assert shift_month(2026, 12, 1) == (2027, 1)
