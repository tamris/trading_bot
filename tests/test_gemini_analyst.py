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


def test_evaluate_journal():
    analyst = GeminiAnalyst()
    metrics = {
        "total_closed": 3,
        "wins": 2,
        "losses": 1,
        "beps": 0,
        "adjusted_win_rate": 66.7,
        "capital_preserved_rate": 66.7,
        "net_pnl": 15.50,
        "gross_profit": 20.00,
        "gross_loss": 4.50,
        "profit_factor": 4.44,
        "risk_reward_ratio": 2.22,
        "avg_win": 10.00,
        "avg_loss": 4.50,
        "best_trade": 12.00,
        "worst_trade": -4.50,
        "by_symbol": {
            "XAUUSD": {"total": 2, "wins": 2, "losses": 0, "beps": 0, "win_rate": 100.0, "pnl": 20.00},
            "BTCUSD": {"total": 1, "wins": 0, "losses": 1, "beps": 0, "win_rate": 0.0, "pnl": -4.50}
        }
    }
    recent = [
        {"ticket": 101, "symbol": "XAUUSD", "action": "BUY", "profit_loss": 12.00, "close_reason": "TAKE_PROFIT", "ai_reasoning": "Gold trend"},
        {"ticket": 102, "symbol": "XAUUSD", "action": "BUY", "profit_loss": 8.00, "close_reason": "TAKE_PROFIT", "ai_reasoning": "Gold trend"},
        {"ticket": 103, "symbol": "BTCUSD", "action": "SELL", "profit_loss": -4.50, "close_reason": "STOP_LOSS", "ai_reasoning": "BTC sell"}
    ]
    evaluation = analyst.evaluate_journal(metrics, recent)
    assert isinstance(evaluation, str)
    assert len(evaluation) > 50
    assert "Poin Keunggulan" in evaluation or "Keunggulan" in evaluation


def test_gemini_fail_safe_hold_when_all_keys_fail(dummy_tech_buy_signal):
    from core.gemini_analyst import GeminiKeyInfo

    analyst = GeminiAnalyst()

    class _BrokenInteractions:
        def create(self, **kwargs):
            raise RuntimeError("503 Service Unavailable")

    class _BrokenClient:
        interactions = _BrokenInteractions()

    k1 = GeminiKeyInfo(key="FAKE_KEY_1_1234567890")
    k1.client = _BrokenClient()
    analyst._pool = [k1]

    decision = analyst.analyze_confluence(dummy_tech_buy_signal, [])
    # CRITICAL: Must NEVER auto-approve with fake score 85! Must enforce Fail-Safe HOLD (score 0)
    assert decision.action == "HOLD"
    assert decision.confluence_score == 0
    assert "Fail-Safe HOLD" in decision.reasoning
    assert k1.error_count == 1
    assert "503" in k1.last_error


def test_gemini_jpy_sentiment_and_markdown_json_cleanup():
    raw = '```json\n{"symbol":"USDJPY","technical_bias":"BEARISH","news_sentiment":"BULLISH_JPY","confluence_score":87,"action":"SELL","reasoning":"BOJ hawkish"}\n```'
    cleaned = GeminiAnalyst._clean_json_text(raw)
    decision = GeminiTradingDecision.model_validate_json(cleaned)
    assert decision.news_sentiment == "BULLISH_JPY"
    assert decision.confluence_score == 87
