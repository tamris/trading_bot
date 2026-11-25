import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from core.technical_engine import TechnicalEngine, TechnicalSignal


@pytest.fixture
def sample_candles():
    """Generate 250 predictable candles for indicator testing."""
    np.random.seed(42)
    count = 250
    times = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="15min")
    # Linear uptrend
    base = np.linspace(2600.0, 2700.0, count)
    noise = np.random.normal(0, 0.5, count)
    close = base + noise
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.2

    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": 100,
        "spread": 20,
        "real_volume": 0
    })
    return df


def test_ema_calculation(sample_candles):
    engine = TechnicalEngine()
    ema50 = engine.calculate_ema(sample_candles["close"], 50)
    ema200 = engine.calculate_ema(sample_candles["close"], 200)

    assert len(ema50) == len(sample_candles)
    assert len(ema200) == len(sample_candles)
    # In uptrend, EMA50 should be higher than EMA200 towards the end
    assert ema50.iloc[-1] > ema200.iloc[-1]


def test_rsi_calculation(sample_candles):
    engine = TechnicalEngine()
    rsi = engine.calculate_rsi(sample_candles["close"], 14)

    assert len(rsi) == len(sample_candles)
    # RSI must be strictly bounded between 0 and 100
    assert (rsi >= 0).all() and (rsi <= 100).all()
    # In steady uptrend, RSI should be above 50
    assert rsi.iloc[-1] > 50.0


def test_atr_calculation(sample_candles):
    engine = TechnicalEngine()
    atr = engine.calculate_atr(sample_candles, 14)

    assert len(atr) == len(sample_candles)
    # ATR must always be strictly positive
    assert (atr.dropna() > 0).all()


def test_evaluate_setup_buy(sample_candles):
    engine = TechnicalEngine()
    # Create scenario where last candle touches EMA 50 from above, close > EMA 200, RSI in [40, 65]
    df = engine.compute_indicators(sample_candles)
    last_ema50 = df["ema50"].iloc[-1]
    last_ema200 = df["ema200"].iloc[-1]
    last_atr = df["atr"].iloc[-1]

    # Craft last candle for valid BUY
    sample_candles.loc[sample_candles.index[-1], "close"] = last_ema50 + 0.1
    sample_candles.loc[sample_candles.index[-1], "low"] = last_ema50 - 0.2
    sample_candles.loc[sample_candles.index[-1], "high"] = last_ema50 + 1.0

    sig = engine.evaluate_setup("XAUUSD", sample_candles)
    assert isinstance(sig, TechnicalSignal)
    assert sig.symbol == "XAUUSD"
    if sig.is_valid:
        assert sig.action == "BUY"
        assert sig.bias == "BULLISH"
        assert sig.sl_price < sig.entry_price
        assert sig.tp_price > sig.entry_price
        # Verify 1:2 Risk to Reward
        sl_dist = sig.entry_price - sig.sl_price
        tp_dist = sig.tp_price - sig.entry_price
        assert pytest.approx(tp_dist / sl_dist, rel=1e-2) == 2.0


def test_eurusd_rejects_reversal_above_ema50_and_overextended_sell():
    """Verify EURUSD cannot trigger SELL when price broke ABOVE EMA50 or is overextended >1x ATR below EMA50."""
    engine = TechnicalEngine()
    count = 250
    times = pd.date_range(end=datetime.now(timezone.utc), periods=count, freq="15min")
    # Clean downtrend on EURUSD (~1.1000 down to 1.0900)
    close = np.linspace(1.10000, 1.09000, count)
    df = pd.DataFrame({
        "time": times,
        "open": close + 0.00010,
        "high": close + 0.00030,
        "low": close - 0.00030,
        "close": close,
        "tick_volume": 100,
        "spread": 10,
        "real_volume": 0
    })
    ind = engine.compute_indicators(df)
    ema50 = float(ind["ema50"].iloc[-1])
    ema200 = float(ind["ema200"].iloc[-1])
    atr = float(ind["atr"].iloc[-1])
    assert ema50 < ema200

    # Case 1: Bullish reversal breakout ABOVE EMA50 (still below EMA200) -> MUST BE REJECTED!
    df_rev = df.copy()
    df_rev.loc[df_rev.index[-1], "open"] = ema50 + 0.00040
    df_rev.loc[df_rev.index[-1], "high"] = ema50 + 0.00050
    df_rev.loc[df_rev.index[-1], "low"] = ema50 + 0.00010
    df_rev.loc[df_rev.index[-1], "close"] = ema50 + 0.00025
    sig_rev = engine.evaluate_setup("EURUSD", df_rev)
    assert sig_rev.is_valid is False
    assert sig_rev.action == "HOLD"

    # Case 2: Overextended 15 pips below EMA50 (was wrongly accepted by 0.2% = 22 pips before!) -> MUST BE REJECTED!
    df_ext = df.copy()
    far_below = ema50 - max(0.00150, 2.5 * atr)
    df_ext.loc[df_ext.index[-1], "open"] = far_below + 0.00010
    df_ext.loc[df_ext.index[-1], "high"] = far_below + 0.00020
    df_ext.loc[df_ext.index[-1], "low"] = far_below - 0.00020
    df_ext.loc[df_ext.index[-1], "close"] = far_below
    sig_ext = engine.evaluate_setup("EURUSD", df_ext)
    assert sig_ext.is_valid is False
    assert sig_ext.action == "HOLD"
