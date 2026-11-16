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


def test_b64url():
    raw = b"hello_world"
    encoded = ExnessApiConnector._b64url(raw)
    assert "=" not in encoded
    assert isinstance(encoded, str)
