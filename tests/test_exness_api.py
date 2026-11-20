import pytest
from core.exness_api_connector import ExnessApiConnector


def test_normalize_symbol():
    assert ExnessApiConnector.normalize_symbol("BTCUSDm") == "BTCUSD"
    assert ExnessApiConnector.normalize_symbol("EURUSDm") == "EURUSD"
    assert ExnessApiConnector.normalize_symbol("XAUUSDm") == "XAUUSD"
    assert ExnessApiConnector.normalize_symbol("BTCUSD") == "BTCUSD"
    assert ExnessApiConnector.normalize_symbol("btcusdm") == "BTCUSD"


def test_symbol_info_specifications():
    conn = ExnessApiConnector()
    
    # Forex EURUSD
    eur = conn.get_symbol_info("EURUSD")
    assert eur["digits"] == 5
    assert eur["point"] == 0.00001
    assert eur["contract_size"] == 100000.0
    assert eur["volume_min"] == 0.01

    # Gold XAUUSD
    gold = conn.get_symbol_info("XAUUSDm")
    assert gold["digits"] == 2
    assert gold["point"] == 0.01
    assert gold["contract_size"] == 100.0

    # Crypto BTCUSD
    btc = conn.get_symbol_info("BTCUSD")
    assert btc["digits"] == 2
    assert btc["point"] == 0.01
    assert btc["contract_size"] == 1.0

    # Crypto ETHUSD
    eth = conn.get_symbol_info("ETHUSD")
    assert eth["digits"] == 2
    assert eth["point"] == 0.01
    assert eth["contract_size"] == 1.0
    assert eth["tick_value"] == 0.01


def test_b64url():
    raw = b"hello_world"
    encoded = ExnessApiConnector._b64url(raw)
    assert "=" not in encoded
    assert isinstance(encoded, str)


def test_format_price_digits():
    conn = ExnessApiConnector()
    assert conn._format_price("EURUSD", 1.097351) == "1.09735"
    assert conn._format_price("USDJPY", 159.5891) == "159.589"
    assert conn._format_price("BTCUSD", 63097.7) == "63097.70"
    assert conn._format_price("XAUUSD", 2650.5) == "2650.50"


def test_open_position_async_ack_resolves_ticket(monkeypatch):
    conn = ExnessApiConnector()
    conn.account_id = "31000217262"
    conn._is_connected = True

    snapshot_calls = {"count": 0}

    def fake_request(method, path, body=None, use_access_point=True, timeout=15, retry=False):
        if method == "GET" and "/snapshot" in path:
            snapshot_calls["count"] += 1
            if snapshot_calls["count"] == 1:
                # Pre-order snapshot already has ticket 17269633
                return True, {
                    "account_state": {"balance": "100.00", "equity": "100.00"},
                    "positions": [
                        {"position_id": "17269633", "instrument": "USDJPY", "direction": "buy", "open_price": "159.570"}
                    ]
                }
            # Post-order snapshot contains the newly created position 17274388
            return True, {
                "account_state": {"balance": "100.00", "equity": "100.00"},
                "positions": [
                    {"position_id": "17269633", "instrument": "USDJPY", "direction": "buy", "open_price": "159.570"},
                    {"position_id": "17274388", "instrument": "USDJPY", "direction": "buy", "open_price": "159.589"}
                ]
            }
        if method == "POST" and "/positions" in path:
            assert body["stop_loss_price"] == "159.490"
            assert body["take_profit_price"] == "159.760"
            # Exness REST API 202 AckResponse (NO position_id!)
            return True, {"operation_id": "99887766", "status": "accepted"}
        if method == "GET" and "/operations/99887766" in path:
            return True, {"operation_id": "99887766", "status": "confirmed"}
        return False, {"error": f"Unexpected call: {method} {path}"}

    monkeypatch.setattr(conn, "_request", fake_request)
    monkeypatch.setattr("core.exness_api_connector.time.sleep", lambda _: None)

    res = conn.open_position(
        symbol="USDJPY",
        action="BUY",
        volume=0.01,
        sl=159.49,
        tp=159.76,
        comment="AI-88"
    )
    assert res["success"] is True
    assert res["ticket"] == 17274388
    assert res["price"] == pytest.approx(159.589)


def test_open_position_async_rejected(monkeypatch):
    conn = ExnessApiConnector()
    conn.account_id = "31000217262"
    conn._is_connected = True

    def fake_request(method, path, body=None, use_access_point=True, timeout=15, retry=False):
        if method == "GET" and "/snapshot" in path:
            return True, {"account_state": {"balance": "100.00", "equity": "100.00"}, "positions": []}
        if method == "POST" and "/positions" in path:
            return True, {"operation_id": "55555", "status": "accepted"}
        if method == "GET" and "/operations/55555" in path:
            return True, {"operation_id": "55555", "status": "rejected", "error_message": "TRADING_RULE_INSUFFICIENT_MARGIN"}
        return False, {"error": "Unexpected"}

    monkeypatch.setattr(conn, "_request", fake_request)
    monkeypatch.setattr("core.exness_api_connector.time.sleep", lambda _: None)

    res = conn.open_position(symbol="EURUSD", action="SELL", volume=0.01, sl=1.099, tp=1.095)
    assert res["success"] is False
    assert "TRADING_RULE_INSUFFICIENT_MARGIN" in res["error"]
