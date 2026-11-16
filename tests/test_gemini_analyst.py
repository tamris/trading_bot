import pytest
from core.technical_engine import TechnicalSignal
from core.gemini_analyst import GeminiAnalyst, GeminiTradingDecision


@pytest.fixture
def dummy_tech_buy_signal():
    return TechnicalSignal(
        symbol="XAUUSD",
        is_valid=True,
        bias="BULLISH",
        action="BUY",
        entry_price=2650.00,
        sl_price=2635.00,
        tp_price=2680.00,
        atr=10.0,
        rsi=55.0,
        ema50=2645.0,
        ema200=2620.0,
        reason="Test BUY setup"
    )


def test_gemini_decision_schema_validation():
    valid_data = {
        "symbol": "XAUUSD",
        "technical_bias": "BULLISH",
        "news_sentiment": "BEARISH_USD",
        "confluence_score": 85,
        "action": "BUY",
        "reasoning": "Strong confluence between weak USD headlines and bullish trend."
    }
    decision = GeminiTradingDecision(**valid_data)
    assert decision.symbol == "XAUUSD"
    assert decision.confluence_score == 85
    assert decision.action == "BUY"


def test_confluence_filter_score_below_80(dummy_tech_buy_signal):
    analyst = GeminiAnalyst()
    analyst.min_confluence_score = 80

    decision = GeminiTradingDecision(
        symbol="XAUUSD",
        technical_bias="BULLISH",
        news_sentiment="BEARISH_USD",
        confluence_score=75,  # Below threshold
        action="BUY",
        reasoning="Moderate setup"
    )

    validated = analyst._validate_and_filter(dummy_tech_buy_signal, decision)
    # Must be forced to HOLD
    assert validated.action == "HOLD"
    assert "Score Filtered" in validated.reasoning


def test_confluence_filter_action_mismatch(dummy_tech_buy_signal):
    analyst = GeminiAnalyst()
    analyst.min_confluence_score = 80

    decision = GeminiTradingDecision(
        symbol="XAUUSD",
        technical_bias="BEARISH",
        news_sentiment="BULLISH_USD",
        confluence_score=90,
        action="SELL",  # Tech signal was BUY
        reasoning="Macro pressure indicates sell"
    )

    validated = analyst._validate_and_filter(dummy_tech_buy_signal, decision)
    # Must be forced to HOLD due to conflict
    assert validated.action == "HOLD"
    assert "Bias Mismatch" in validated.reasoning


def test_confluence_filter_approved(dummy_tech_buy_signal):
    analyst = GeminiAnalyst()
    analyst.min_confluence_score = 80

    decision = GeminiTradingDecision(
        symbol="XAUUSD",
        technical_bias="BULLISH",
        news_sentiment="BEARISH_USD",
        confluence_score=88,  # >= 80
        action="BUY",         # Matches Tech BUY
        reasoning="High conviction confluence."
    )

    validated = analyst._validate_and_filter(dummy_tech_buy_signal, decision)
    assert validated.action == "BUY"
    assert validated.confluence_score == 88
