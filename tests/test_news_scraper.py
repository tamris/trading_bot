import pytest
from datetime import datetime, timezone, timedelta
from core.news_scraper import NewsScraper


def test_news_blackout_detection():
    ns = NewsScraper()
    event_time = datetime(2026, 11, 14, 15, 0, 0, tzinfo=timezone.utc)
    ns._cached_events = [
        {
            "title": "US Non-Farm Payrolls",
            "country": "USD",
            "impact": "High",
            "date": event_time.isoformat()
        }
    ]

    # Test 1: Exactly 5 minutes before release -> inside 15 min window -> Blackout
    check_t1 = event_time - timedelta(minutes=5)
    is_blackout, msg = ns.is_news_blackout("XAUUSD", check_time=check_t1)
    assert is_blackout is True
    assert "Non-Farm Payrolls" in msg

    # Test 2: Exactly 20 minutes before release -> outside 15 min window -> Safe
    check_t2 = event_time - timedelta(minutes=20)
    is_blackout, msg = ns.is_news_blackout("XAUUSD", check_time=check_t2)
    assert is_blackout is False

    # Test 3: Currency independence (EUR High Impact should not trigger blackout on XAUUSD, but should on EURUSD)
    ns._cached_events = [
        {
            "title": "ECB Interest Rate Decision",
            "country": "EUR",
            "impact": "High",
            "date": event_time.isoformat()
        }
    ]
    # For XAUUSD -> Safe (currency is USD only)
    is_blackout_gold, _ = ns.is_news_blackout("XAUUSD", check_time=check_t1)
    assert is_blackout_gold is False

    # For EURUSD -> Blackout (currency involves EUR)
    is_blackout_eur, _ = ns.is_news_blackout("EURUSD", check_time=check_t1)
    assert is_blackout_eur is True


def test_fetch_latest_news_structure():
    ns = NewsScraper()
    news = ns.fetch_latest_news(limit=5)
    assert isinstance(news, list)
    assert len(news) > 0
    first = news[0]
    assert "title" in first
    assert "source" in first
