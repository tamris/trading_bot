import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # MT5 Settings
    MT5_LOGIN: int = Field(default=0, description="MT5 Account Login ID")
    MT5_PASSWORD: str = Field(default="", description="MT5 Account Password")
    MT5_SERVER: str = Field(default="Exness-MT5Trial", description="MT5 Server Name")
    MT5_PATH: str = Field(
        default=r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
        description="Path to MT5 terminal64.exe"
    )
    MT5_MOCK_MODE: bool = Field(default=True, description="Enable mock mode for testing without active broker")

    # Exness Direct Cloud API Settings (Linux / Railway / Cloud)
    TRADING_BACKEND: str = Field(default="MT5", description="Execution backend: 'MT5' (Windows terminal) or 'EXNESS_API' (Direct cloud REST API)")
    EXNESS_API_KEY: str = Field(default="", description="Exness API Key (exnsk-...)")
    EXNESS_API_SECRET: str = Field(default="", description="Exness Ed25519 Private Key Hex (128 chars)")
    EXNESS_ACCOUNT_ID: str = Field(default="", description="Exness Trading Account ID")
    EXNESS_GATEWAY_URL: str = Field(default="https://api.exness-api.com", description="Exness Gateway discovery URL")

    # Gemini AI Settings
    GEMINI_API_KEY: str = Field(default="", description="Primary Google Gemini API Key")
    GEMINI_API_KEYS_RAW: str = Field(default="", alias="GEMINI_API_KEYS", description="Comma-separated pool of Gemini API Keys")
    GEMINI_MODEL: str = Field(default="gemini-3.5-flash-lite", description="Gemini model identifier")

    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram Bot Token from BotFather")
    TELEGRAM_CHAT_ID: str = Field(default="5443625348,7030527954", description="Telegram Chat ID for alerts and monitoring")

    # Trading Symbol & Timeframes
    SYMBOLS_RAW: str = Field(default="BTCUSD,ETHUSD,XAUUSD,EURUSD,USDJPY", alias="SYMBOLS", description="Comma-separated symbols to trade")
    TIMEFRAME_MACRO: str = Field(default="H1", description="Macro trend timeframe")
    TIMEFRAME_ENTRY: str = Field(default="M15", description="Entry execution timeframe")

    # Risk Management Parameters
    RISK_PERCENT: float = Field(default=1.0, ge=0.1, le=5.0, description="Max risk percentage per trade")
    MAX_DAILY_DRAWDOWN_PERCENT: float = Field(default=3.0, ge=1.0, le=10.0, description="Max daily drawdown before pausing")
    MAX_OPEN_POSITIONS_PER_SYMBOL: int = Field(default=1, ge=1, le=5, description="Max open positions per symbol (strict 1 per pair prevents double-SL)")
    MAX_OPEN_POSITIONS_CRYPTO_ACTIVE_SESSION: int = Field(default=1, ge=1, le=5, description="Max open positions for crypto during active forex market sessions")
    MAX_OPEN_POSITIONS_CRYPTO_WEEKEND: int = Field(default=2, ge=1, le=5, description="Max open positions for crypto during weekends (when forex/gold are closed)")
    MAX_TOTAL_OPEN_POSITIONS: int = Field(default=3, ge=1, le=10, description="Max total open positions across all symbols")
    NEWS_PROTECTION_MINUTES: int = Field(default=15, description="Minutes before and after high-impact news to freeze entry")
    MIN_CONFLUENCE_SCORE: int = Field(default=85, ge=50, le=100, description="Minimum confluence score to execute trade")
    SYMBOL_COOLDOWN_MINUTES: int = Field(default=45, ge=0, le=180, description="Minutes to wait before re-entering the same symbol after a trade exit")

    # Indicator Parameters (Hard Rules)
    EMA_FAST_PERIOD: int = Field(default=50, description="Fast EMA period for pullback")
    EMA_SLOW_PERIOD: int = Field(default=200, description="Slow EMA period for macro trend")
    RSI_PERIOD: int = Field(default=14, description="RSI period")
    RSI_BUY_MIN: float = Field(default=40.0, description="RSI minimum for BUY setup")
    RSI_BUY_MAX: float = Field(default=65.0, description="RSI maximum for BUY setup")
    RSI_SELL_MIN: float = Field(default=35.0, description="RSI minimum for SELL setup")
    RSI_SELL_MAX: float = Field(default=60.0, description="RSI maximum for SELL setup")
    ATR_PERIOD: int = Field(default=14, description="ATR period for volatility-based SL/TP")
    ATR_SL_MULTIPLIER: float = Field(default=1.5, description="SL multiplier based on ATR")
    ATR_TP_MULTIPLIER: float = Field(default=3.0, description="TP multiplier based on ATR (1:2 RR)")
    BEP_TRIGGER_MULTIPLIER: float = Field(default=1.5, description="Profit ATR multiplier to trigger Break-Even")
    ADX_PERIOD: int = Field(default=14, description="ADX trend strength period")
    ADX_MIN_THRESHOLD: float = Field(default=20.0, ge=10.0, le=40.0, description="Minimum ADX required to validate trending market and reject sideways chop")

    # Professional Strategy Features
    PARTIAL_TP_ENABLED: bool = Field(default=True, description="Enable 50% partial take profit at 1:1 RR")
    SESSION_FILTER_ENABLED: bool = Field(default=True, description="Enable session killzones and rollover filter")
    SCAN_REPORT_MODE: str = Field(default="1h", description="Periodic market radar scan reporting to Telegram ('1h', '15m', or 'off')")

    # Project Directories
    LOGS_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = BASE_DIR / "logs" / "trading_bot.log"

    @property
    def SYMBOLS(self) -> List[str]:
        return [s.strip() for s in self.SYMBOLS_RAW.split(",") if s.strip()]

    @property
    def GEMINI_API_KEYS(self) -> List[str]:
        keys = []
        if self.GEMINI_API_KEYS_RAW:
            keys.extend([k.strip() for k in self.GEMINI_API_KEYS_RAW.split(",") if k.strip()])
        if self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip() not in keys:
            keys.append(self.GEMINI_API_KEY.strip())
        return keys


# Instantiate singleton settings
settings = Settings()

# Ensure logs directory exists
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
