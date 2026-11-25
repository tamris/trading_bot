import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger("ai_trading_bot.technical")


@dataclass
class TechnicalSignal:
    symbol: str
    is_valid: bool
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    action: str  # "BUY", "SELL", "HOLD"
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    rsi: float
    ema50: float
    ema200: float
    adx: float = 0.0
    h1_ema200: Optional[float] = None
    reason: str = ""


class TechnicalEngine:
    """Calculates technical indicators and verifies strict hard trading rules."""

    def __init__(self):
        self.ema_fast_period = settings.EMA_FAST_PERIOD
        self.ema_slow_period = settings.EMA_SLOW_PERIOD
        self.rsi_period = settings.RSI_PERIOD
        self.atr_period = settings.ATR_PERIOD
        self.sl_multiplier = settings.ATR_SL_MULTIPLIER
        self.tp_multiplier = settings.ATR_TP_MULTIPLIER

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Wilder's Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing using alpha = 1 / period
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Wilder's Average True Range (ATR)."""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return atr

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) to measure trend strength."""
        high = df['high']
        low = df['low']
        close = df['close']
        close_prev = close.shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
        minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        return adx.fillna(25.0)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach EMA 50, EMA 200, RSI 14, ATR 14, and ADX 14 to candles DataFrame."""
        if len(df) < self.ema_slow_period:
            logger.warning(f"Insufficient candle data ({len(df)} candles). At least {self.ema_slow_period} required.")

        df = df.copy()
        df['ema50'] = self.calculate_ema(df['close'], self.ema_fast_period)
        df['ema200'] = self.calculate_ema(df['close'], self.ema_slow_period)
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = self.calculate_atr(df, self.atr_period)
        df['adx'] = self.calculate_adx(df, getattr(settings, "ADX_PERIOD", 14))
        return df

    def evaluate_setup(
        self,
        symbol: str,
        df_m15: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None
    ) -> TechnicalSignal:
        """
        Evaluate Hard Rules on M15 with optional H1 macro confirmation.

        BUY Rules:
        1. Close > EMA 200 (Bullish Trend).
        2. Pullback near EMA 50 (Low <= EMA 50 * 1.002 or distance to EMA50 <= 0.8 * ATR).
        3. RSI between 40 and 65.

        SELL Rules:
        1. Close < EMA 200 (Bearish Trend).
        2. Pullback near EMA 50 (High >= EMA 50 * 0.998 or distance to EMA50 <= 0.8 * ATR).
        3. RSI between 35 and 60.
        """
        if df_m15 is None or len(df_m15) < self.ema_slow_period:
            return TechnicalSignal(
                symbol=symbol,
                is_valid=False,
                bias="NEUTRAL",
                action="HOLD",
                entry_price=0.0,
                sl_price=0.0,
                tp_price=0.0,
                atr=0.0,
                rsi=50.0,
                ema50=0.0,
                ema200=0.0,
                reason="Insufficient M15 candle data for indicator calculation"
            )

        df_m15_ind = self.compute_indicators(df_m15)
        last_candle = df_m15_ind.iloc[-1]
        prev_candle = df_m15_ind.iloc[-2]

        open_price = float(last_candle['open'])
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        prev_high = float(prev_candle['high'])
        prev_low = float(prev_candle['low'])
        ema50 = float(last_candle['ema50'])
        ema200 = float(last_candle['ema200'])
        rsi = float(last_candle['rsi'])
        atr = float(last_candle['atr'])
        adx = float(last_candle['adx']) if 'adx' in last_candle else 25.0

        sym_upper = symbol.upper()
        price_dec = 3 if "JPY" in sym_upper else (2 if ("XAU" in sym_upper or "GOLD" in sym_upper or "BTC" in sym_upper or "ETH" in sym_upper) else 5)

        h1_ema200 = None
        h1_macro_bullish = True
        h1_macro_bearish = True

        if df_h1 is not None and len(df_h1) >= self.ema_slow_period:
            df_h1_ind = self.compute_indicators(df_h1)
            h1_last = df_h1_ind.iloc[-1]
            h1_close = float(h1_last['close'])
            h1_ema200 = float(h1_last['ema200'])
            h1_macro_bullish = h1_close >= h1_ema200
            h1_macro_bearish = h1_close <= h1_ema200

        # Pure ATR-based Pullback & Rejection rules (universal across Forex, Gold, and Crypto):
        # 1. Wick of current or previous candle touched/entered EMA50 value zone (within 0.5 * ATR)
        # 2. Close rejected back to the trend side of EMA50 (prevents entering against reversals!)
        # 3. Distance from EMA50 <= 1.0 * ATR (prevents chasing overextended moves at the top/bottom!)
        touched_ema50_buy = (low <= ema50 + 0.5 * atr) or (prev_low <= ema50 + 0.5 * atr)
        near_ema50_buy = touched_ema50_buy and (close > ema50) and ((close - ema50) <= 1.0 * atr)

        touched_ema50_sell = (high >= ema50 - 0.5 * atr) or (prev_high >= ema50 - 0.5 * atr)
        near_ema50_sell = touched_ema50_sell and (close < ema50) and ((ema50 - close) <= 1.0 * atr)

        # ADX trend strength filter (reject flat, choppy, sideways ranges)
        min_adx = getattr(settings, "ADX_MIN_THRESHOLD", 20.0)
        is_trending = adx >= min_adx

        # Candlestick confirmation (avoid entering on counter-momentum candles)
        is_bullish_candle = (close >= open_price) or ((close - low) > (high - close))
        is_bearish_candle = (close <= open_price) or ((high - close) > (close - low))

        # ----------------- Check BUY Setup -----------------
        # Require full EMA alignment (EMA50 > EMA200) + price > EMA200 + H1 macro bullish
        is_buy_trend = (close > ema200) and (ema50 > ema200) and h1_macro_bullish
        is_buy_rsi = settings.RSI_BUY_MIN <= rsi <= settings.RSI_BUY_MAX

        if is_buy_trend and near_ema50_buy and is_buy_rsi:
            if not is_trending:
                return TechnicalSignal(
                    symbol=symbol,
                    is_valid=False,
                    bias="BULLISH",
                    action="HOLD",
                    entry_price=close,
                    sl_price=0.0,
                    tp_price=0.0,
                    atr=atr,
                    rsi=rsi,
                    ema50=ema50,
                    ema200=ema200,
                    adx=adx,
                    h1_ema200=h1_ema200,
                    reason=f"Sideways Market Rejected: ADX ({adx:.1f}) < {min_adx} (Trending volume insufficient)"
                )
            if not is_bullish_candle:
                return TechnicalSignal(
                    symbol=symbol,
                    is_valid=False,
                    bias="BULLISH",
                    action="HOLD",
                    entry_price=close,
                    sl_price=0.0,
                    tp_price=0.0,
                    atr=atr,
                    rsi=rsi,
                    ema50=ema50,
                    ema200=ema200,
                    adx=adx,
                    h1_ema200=h1_ema200,
                    reason="Candle confirmation missing (waiting for bullish bounce from EMA50)"
                )

            sl = round(close - (self.sl_multiplier * atr), price_dec)
            tp = round(close + (self.tp_multiplier * atr), price_dec)
            return TechnicalSignal(
                symbol=symbol,
                is_valid=True,
                bias="BULLISH",
                action="BUY",
                entry_price=close,
                sl_price=sl,
                tp_price=tp,
                atr=atr,
                rsi=rsi,
                ema50=ema50,
                ema200=ema200,
                adx=adx,
                h1_ema200=h1_ema200,
                reason=f"BUY setup: Close ({close:.{price_dec}f}) > EMA50 ({ema50:.{price_dec}f}) > EMA200 ({ema200:.{price_dec}f}), ATR pullback & bounce valid, RSI ({rsi:.1f}), ADX ({adx:.1f})"
            )

        # ----------------- Check SELL Setup -----------------
        # Require full EMA alignment (EMA50 < EMA200) + price < EMA200 + H1 macro bearish
        is_sell_trend = (close < ema200) and (ema50 < ema200) and h1_macro_bearish
        is_sell_rsi = settings.RSI_SELL_MIN <= rsi <= settings.RSI_SELL_MAX

        if is_sell_trend and near_ema50_sell and is_sell_rsi:
            if not is_trending:
                return TechnicalSignal(
                    symbol=symbol,
                    is_valid=False,
                    bias="BEARISH",
                    action="HOLD",
                    entry_price=close,
                    sl_price=0.0,
                    tp_price=0.0,
                    atr=atr,
                    rsi=rsi,
                    ema50=ema50,
                    ema200=ema200,
                    adx=adx,
                    h1_ema200=h1_ema200,
                    reason=f"Sideways Market Rejected: ADX ({adx:.1f}) < {min_adx} (Trending volume insufficient)"
                )
            if not is_bearish_candle:
                return TechnicalSignal(
                    symbol=symbol,
                    is_valid=False,
                    bias="BEARISH",
                    action="HOLD",
                    entry_price=close,
                    sl_price=0.0,
                    tp_price=0.0,
                    atr=atr,
                    rsi=rsi,
                    ema50=ema50,
                    ema200=ema200,
                    adx=adx,
                    h1_ema200=h1_ema200,
                    reason="Candle confirmation missing (waiting for bearish rejection from EMA50)"
                )

            sl = round(close + (self.sl_multiplier * atr), price_dec)
            tp = round(close - (self.tp_multiplier * atr), price_dec)
            return TechnicalSignal(
                symbol=symbol,
                is_valid=True,
                bias="BEARISH",
                action="SELL",
                entry_price=close,
                sl_price=sl,
                tp_price=tp,
                atr=atr,
                rsi=rsi,
                ema50=ema50,
                ema200=ema200,
                adx=adx,
                h1_ema200=h1_ema200,
                reason=f"SELL setup: Close ({close:.{price_dec}f}) < EMA50 ({ema50:.{price_dec}f}) < EMA200 ({ema200:.{price_dec}f}), ATR pullback & rejection valid, RSI ({rsi:.1f}), ADX ({adx:.1f})"
            )

        # ----------------- No Setup (HOLD) -----------------
        bias = "BULLISH" if close > ema200 else ("BEARISH" if close < ema200 else "NEUTRAL")
        reasons = []
        if bias == "BULLISH" and ema50 <= ema200:
            reasons.append("EMA50 <= EMA200 (struktur uptrend belum terbentuk)")
        elif bias == "BEARISH" and ema50 >= ema200:
            reasons.append("EMA50 >= EMA200 (struktur downtrend belum terbentuk)")
        if not (near_ema50_buy or near_ema50_sell):
            reasons.append("Price not in valid ATR pullback/rejection zone at EMA50")
        if not (is_buy_rsi or is_sell_rsi):
            reasons.append(f"RSI ({rsi:.1f}) out of trigger boundaries")
        if not (h1_macro_bullish if bias == "BULLISH" else h1_macro_bearish):
            reasons.append("H1 macro trend conflicting")

        return TechnicalSignal(
            symbol=symbol,
            is_valid=False,
            bias=bias,
            action="HOLD",
            entry_price=close,
            sl_price=0.0,
            tp_price=0.0,
            atr=atr,
            rsi=rsi,
            ema50=ema50,
            ema200=ema200,
            h1_ema200=h1_ema200,
            reason="; ".join(reasons) if reasons else "Conditions not met"
        )
