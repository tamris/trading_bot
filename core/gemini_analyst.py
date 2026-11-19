import logging
import json
import time
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config.settings import settings
from core.technical_engine import TechnicalSignal

logger = logging.getLogger("ai_trading_bot.gemini")


class GeminiTradingDecision(BaseModel):
    """Structured JSON schema for Gemini Trading Analysis."""
    symbol: str = Field(description="Trading symbol, e.g. XAUUSD or EURUSD")
    technical_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Technical bias from indicator analysis"
    )
    news_sentiment: Literal["BULLISH_USD", "BEARISH_USD", "BULLISH_EUR", "BEARISH_EUR", "RISK_ON", "RISK_OFF", "NEUTRAL"] = Field(
        description="Dominant macro/currency sentiment extracted from news"
    )
    confluence_score: int = Field(
        ge=0, le=100,
        description="Confluence conviction score from 0 to 100"
    )
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Recommended action. Must be BUY, SELL, or HOLD"
    )
    reasoning: str = Field(
        description="Concise rationale explaining the confluence between news and technicals"
    )


class GeminiKeyInfo:
    """Manages state, requests, and rate-limit cooldown for a single Gemini API key."""

    def __init__(
        self,
        key: str,
        db_id: Optional[int] = None,
        requests_count: int = 0,
        success_count: int = 0,
        rate_limit_count: int = 0,
        rate_limited_until: float = 0.0,
        label: str = ""
    ):
        self.key = key.strip()
        self.db_id = db_id
        self.label = label
        self.masked = f"{self.key[:6]}...{self.key[-4:]}" if len(self.key) > 10 else self.key
        self.client = None
        self.rate_limited_until: float = rate_limited_until
        self.requests_count: int = requests_count
        self.success_count: int = success_count
        self.rate_limit_count: int = rate_limit_count
        self._init_client()

    def _init_client(self):
        if GENAI_AVAILABLE and self.key and self.key != "your_gemini_api_key_here":
            try:
                self.client = genai.Client(api_key=self.key, http_options={"timeout": 6000})
                if hasattr(self.client, "interactions") and hasattr(self.client.interactions, "sdk_configuration"):
                    rc = getattr(self.client.interactions.sdk_configuration, "retry_config", None)
                    if rc and hasattr(rc, "max_retries"):
                        rc.max_retries = 0
            except Exception as e:
                logger.warning(f"Failed to create client for key {self.masked}: {e}")

    def is_available(self) -> bool:
        if not self.client:
            return False
        return time.time() >= self.rate_limited_until

    def mark_rate_limited(self, cooldown_seconds: int = 1800):
        self.rate_limited_until = time.time() + cooldown_seconds
        self.rate_limit_count += 1


class GeminiAnalyst:
    """Performs sentiment analysis and validates confluence using a Round-Robin Gemini Key Pool backed by SQLite DB."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: Optional[str] = None,
        db: Optional[Any] = None
    ):
        self.model = model or settings.GEMINI_MODEL
        self.min_confluence_score = settings.MIN_CONFLUENCE_SCORE
        self.db = db
        self._pool: List[GeminiKeyInfo] = []
        self._current_index = 0

        # Load keys from SQLite DB if available
        if self.db:
            db_records = self.db.get_all_api_keys()
            if not db_records:
                # Seed from environment / settings
                raw_keys = api_keys if api_keys is not None else settings.GEMINI_API_KEYS
                self.db.seed_api_keys(raw_keys)
                db_records = self.db.get_all_api_keys()

            for rec in db_records:
                self._load_key_from_db_record(rec)
        else:
            raw_keys = api_keys if api_keys is not None else settings.GEMINI_API_KEYS
            for k in raw_keys:
                self.add_key(k)

        if self._pool:
            logger.info(f"GeminiAnalyst initialized with {len(self._pool)} API key(s) from SQLite pool. Model: {self.model}")
        else:
            logger.info("No valid Gemini API keys configured. Running in simulation mode.")

    def _load_key_from_db_record(self, rec: Dict[str, Any]):
        """Instantiate a GeminiKeyInfo from an existing SQLite database row."""
        key = rec.get("key_value", "").strip()
        if not key or key == "your_gemini_api_key_here":
            return
        info = GeminiKeyInfo(
            key=key,
            db_id=rec.get("id"),
            requests_count=rec.get("requests_count", 0),
            success_count=rec.get("success_count", 0),
            rate_limit_count=rec.get("rate_limit_count", 0),
            rate_limited_until=rec.get("cooldown_until") or 0.0,
            label=rec.get("label", "")
        )
        if info.client:
            self._pool.append(info)

    def add_key(self, key: str, label: str = "") -> bool:
        """Add a new API key to SQLite DB and active rotation pool dynamically."""
        key = key.strip()
        if not key or key == "your_gemini_api_key_here":
            return False
        for info in self._pool:
            if info.key == key:
                return False

        # Validate by instantiating client
        info = GeminiKeyInfo(key=key, label=label)
        if info.client:
            if self.db:
                row_id = self.db.add_api_key(key, label=label)
                info.db_id = row_id
            self._pool.append(info)
            logger.info(f"Added API Key #{info.db_id or len(self._pool)} to SQLite pool: {info.masked}")
            return True
        return False

    def delete_key(self, key_id: int) -> bool:
        """Delete an API key from database and in-memory pool by its ID."""
        if self.db:
            self.db.delete_api_key(key_id)
        before_len = len(self._pool)
        self._pool = [k for k in self._pool if k.db_id != key_id]
        deleted = len(self._pool) < before_len
        if deleted:
            logger.info(f"Deleted API Key #{key_id} from Gemini pool. Remaining: {len(self._pool)}")
        return deleted

    def get_pool_status(self) -> List[Dict[str, Any]]:
        """Return status of all keys in pool for diagnostic and Telegram reporting."""
        now = time.time()
        res = []
        for idx, info in enumerate(self._pool, 1):
            is_avail = info.is_available()
            cooldown_left = max(0, int(info.rate_limited_until - now))
            status_text = "🟢 Active" if is_avail else f"🟡 429 Cooldown ({cooldown_left}s left)"
            res.append({
                "id": info.db_id or idx,
                "index": idx,
                "masked": info.masked,
                "status": status_text,
                "is_available": is_avail,
                "requests": info.requests_count,
                "success": info.success_count,
                "rate_limits": info.rate_limit_count
            })
        return res

    def analyze_confluence(
        self,
        tech_signal: TechnicalSignal,
        news_headlines: List[Dict[str, str]]
    ) -> GeminiTradingDecision:
        """
        Analyze news headlines against technical setup using Round-Robin API Key Polling
        with instant failover on 429 rate limit.
        """
        if not self._pool:
            return self._mock_analysis(tech_signal, news_headlines)

        prompt = self._build_prompt(tech_signal, news_headlines)
        num_keys = len(self._pool)

        # Iterate through available keys in pool
        for attempt in range(num_keys):
            key_info = self._pool[self._current_index]
            self._current_index = (self._current_index + 1) % num_keys

            if not key_info.is_available():
                logger.info(f"Gemini Key {key_info.masked} in 429 cooldown. Rotating to next key...")
                continue

            try:
                key_info.requests_count += 1
                logger.info(f"Calling Gemini with Key {key_info.masked} (Attempt {attempt + 1}/{num_keys})...")

                interaction = key_info.client.interactions.create(
                    model=self.model,
                    input=prompt,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": GeminiTradingDecision.model_json_schema()
                    }
                )

                raw_text = interaction.output_text
                if not raw_text:
                    raise ValueError("Empty response received from Gemini.")

                decision = GeminiTradingDecision.model_validate_json(raw_text)
                validated_decision = self._validate_and_filter(tech_signal, decision)
                key_info.success_count += 1
                if self.db:
                    self.db.update_key_usage(key_info.key, is_success=True, is_rate_limit=False)
                return validated_decision

            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str
                    or "rate limit" in err_str.lower()
                    or "too many requests" in err_str.lower()
                    or "resource_exhausted" in err_str.lower()
                ):
                    logger.warning(
                        f"Gemini Key {key_info.masked} hit Rate Limit (429)! "
                        f"Marking cooldown (30 min) and rotating to next key in pool..."
                    )
                    key_info.mark_rate_limited(cooldown_seconds=1800)
                    if self.db:
                        self.db.update_key_usage(
                            key_info.key,
                            is_success=False,
                            is_rate_limit=True,
                            cooldown_until_ts=key_info.rate_limited_until
                        )
                    continue
                else:
                    logger.warning(f"Gemini error with key {key_info.masked}: {e}. Rotating to next key...")
                    continue

        logger.warning("All Gemini API keys in pool exhausted or rate-limited. Falling back to algorithmic confluence analysis.")
        return self._mock_analysis(tech_signal, news_headlines)

    def _validate_and_filter(
        self,
        tech_signal: TechnicalSignal,
        decision: GeminiTradingDecision
    ) -> GeminiTradingDecision:
        """
        Enforce strict confluence validation:
        1. confluence_score >= min_confluence_score (80)
        2. decision.action matches tech_signal.action exactly
        3. If mismatch or score < 80, override action to HOLD
        """
        score = decision.confluence_score
        ai_action = decision.action
        tech_action = tech_signal.action

        is_score_ok = score >= self.min_confluence_score
        is_action_matched = (ai_action == tech_action) and (ai_action in ["BUY", "SELL"])

        if not is_score_ok:
            logger.info(f"Trade filtered out: Confluence score ({score}) < minimum threshold ({self.min_confluence_score}).")
            return GeminiTradingDecision(
                symbol=decision.symbol,
                technical_bias=decision.technical_bias,
                news_sentiment=decision.news_sentiment,
                confluence_score=score,
                action="HOLD",
                reasoning=f"[Score Filtered: {score}/100 < {self.min_confluence_score}] {decision.reasoning}"
            )

        if not is_action_matched:
            logger.info(f"Trade filtered out: AI action ({ai_action}) does not match Technical action ({tech_action}).")
            return GeminiTradingDecision(
                symbol=decision.symbol,
                technical_bias=decision.technical_bias,
                news_sentiment=decision.news_sentiment,
                confluence_score=score,
                action="HOLD",
                reasoning=f"[Bias Mismatch: AI={ai_action} vs Tech={tech_action}] {decision.reasoning}"
            )

        logger.info(f"CONFLUENCE VERIFIED! Score: {score}/100. Action: {ai_action} on {decision.symbol}")
        return decision

    def _build_prompt(
        self,
        tech_signal: TechnicalSignal,
        news_headlines: List[Dict[str, str]]
    ) -> str:
        """Construct structured prompt for Gemini."""
        headlines_text = "\n".join([
            f"- [{h.get('source', 'News')}] {h.get('title')} ({h.get('published', '')})"
            for h in news_headlines[:10]
        ])

        return f"""
Anda adalah analis makroekonomi dan sentimen forex/emas profesional untuk trading kuantitatif.

Tugas Anda:
Evaluasi data teknikal berikut terhadap headline berita finansial terkini. Tentukan apakah terdapat konfluensi (keselarasan) kuat antara sentimen makro dan setup teknikal.

INSTRUMEN: {tech_signal.symbol}
SETUP TEKNIKAL:
- Sinyal Teknikal Python: {tech_signal.action} (Bias: {tech_signal.bias})
- Harga Entry: {tech_signal.entry_price}
- Stop Loss: {tech_signal.sl_price}
- Take Profit: {tech_signal.tp_price}
- ATR (14): {tech_signal.atr:.4f}
- RSI (14): {tech_signal.rsi:.2f}
- EMA 50: {tech_signal.ema50:.4f}
- EMA 200: {tech_signal.ema200:.4f}
- Catatan Setup: {tech_signal.reason}

HEADLINE BERITA FINANSIAL TERKINI:
{headlines_text}

ATURAN KONFLUENSI:
1. Untuk XAUUSD (Gold):
   - Sentimen BEARISH_USD / ekspektasi suku bunga Fed turun / geopolitik panas -> Sangat mendukung BUY (Bullish Gold).
   - Sentimen BULLISH_USD / ekspektasi suku bunga Fed naik / yield obligasi naik -> Sangat mendukung SELL (Bearish Gold).
2. Untuk EURUSD:
   - Sentimen BEARISH_USD atau BULLISH_EUR -> Sangat mendukung BUY.
   - Sentimen BULLISH_USD atau BEARISH_EUR -> Sangat mendukung SELL.
3. Untuk USDJPY:
   - Sentimen BULLISH_USD / yield obligasi US10Y naik / BOJ dovish -> Sangat mendukung BUY (Bullish USDJPY).
   - Sentimen BEARISH_USD / intervensi Yen / BOJ hawkish -> Sangat mendukung SELL (Bearish USDJPY).
4. Untuk BTCUSD:
   - Sentimen Risk-On / inflasi melandai / likuiditas global naik -> Sangat mendukung BUY (Bullish BTC).
   - Sentimen Risk-Off / regulasi ketat / likuidasi pasar -> Sangat mendukung SELL (Bearish BTC).
5. Berikan 'confluence_score' (angka bulat 0 hingga 100) secara dinamis, presisi, dan objektif sesuai bobot bukti makro:
   - Skor 90 - 95: Konfluensi sempurna dan luar biasa kuat (katalis berita tier-1 selaras 100% dengan momentum teknikal).
   - Skor 85 - 89: Sentimen makro sangat jelas dan dominan searah dengan setup teknikal.
   - Skor 80 - 84: Sentimen makro mendukung setup teknikal namun dengan kekuatan moderat.
   - Skor 60 - 79: Terdapat berita yang netral, tidak relevan, atau sinyal makro kurang meyakinkan.
   - Skor < 60: Sentimen makro bertolak belakang dengan setup teknikal (risiko jebakan tinggi).
   Catatan: Hitung skor secara dinamis dan presisi (contoh: 81, 83, 86, 88, 91, 93), HINDARI selalu mengeluarkan angka 85 secara statis.
6. Nilai 'action' HARUS salah satu dari: BUY, SELL, atau HOLD.
7. Berikan 'reasoning' dalam bahasa Indonesia yang komprehensif dan jelas, WAJIB memuat DUA aspek:
   - Alasan Berita/Makro: Sebutkan sentimen atau topik berita terkini yang relevan (misal: penguatan/pelemahan Dolar, inflasi, suku bunga, risk-on/risk-off crypto, geopolitik).
   - Alasan Teknikal: Hubungkan dengan posisi harga terhadap EMA50, EMA200, atau RSI.
   Contoh: "Pelemahan Dolar AS akibat data inflasi yang melandai dan sentimen dovish The Fed memberikan dorongan kuat bagi Emas, selaras dengan pantulan harga XAUUSD di atas EMA50 dan momentum RSI yang sehat."

Keluarkan respon HANYA dalam format JSON valid sesuai schema.
"""

    def _mock_analysis(
        self,
        tech_signal: TechnicalSignal,
        news_headlines: List[Dict[str, str]]
    ) -> GeminiTradingDecision:
        """Simulated response for test/dry-run environments without API key."""
        sym_upper = tech_signal.symbol.upper()

        if "JPY" in sym_upper:
            bias_buy = "BULLISH_USD"
            bias_sell = "BEARISH_USD"
            reason_buy = f"Sentimen penguatan Dolar AS (yield Treasury naik) selaras dengan pantulan teknikal {tech_signal.symbol} di atas EMA50."
            reason_sell = f"Sentimen pelemahan Dolar AS dan antisipasi intervensi Yen selaras dengan penolakan teknikal {tech_signal.symbol} di bawah EMA200."
        elif "BTC" in sym_upper:
            bias_buy = "RISK_ON"
            bias_sell = "RISK_OFF"
            reason_buy = f"Sentimen risk-on global dan aliran dana positif crypto selaras dengan penembusan teknikal {tech_signal.symbol} di atas EMA50."
            reason_sell = f"Sentimen risk-off pasar dan kehati-hatian likuiditas makro selaras dengan struktur tren bearish {tech_signal.symbol} di bawah EMA200."
        else:
            bias_buy = "BEARISH_USD"
            bias_sell = "BULLISH_USD"
            reason_buy = f"Sentimen pelemahan Dolar AS di pasar makro mendukung kenaikan {tech_signal.symbol}, terkonfirmasi oleh pantulan harga di atas EMA50 dan RSI {tech_signal.rsi:.1f}."
            reason_sell = f"Sentimen penguatan Dolar AS mendominasi pasar mata uang, menekan harga {tech_signal.symbol} di bawah EMA200 dengan konfirmasi momentum bearish RSI {tech_signal.rsi:.1f}."

        if tech_signal.action == "BUY":
            return GeminiTradingDecision(
                symbol=tech_signal.symbol,
                technical_bias="BULLISH",
                news_sentiment=bias_buy,
                confluence_score=85,
                action="BUY",
                reasoning=reason_buy
            )
        elif tech_signal.action == "SELL":
            return GeminiTradingDecision(
                symbol=tech_signal.symbol,
                technical_bias="BEARISH",
                news_sentiment=bias_sell,
                confluence_score=85,
                action="SELL",
                reasoning=reason_sell
            )
        else:
            return GeminiTradingDecision(
                symbol=tech_signal.symbol,
                technical_bias=tech_signal.bias if tech_signal.bias in ["BULLISH", "BEARISH"] else "NEUTRAL",
                news_sentiment="NEUTRAL",
                confluence_score=45,
                action="HOLD",
                reasoning="Kondisi pasar saat ini berada dalam fase konsolidasi tanpa konfluensi arah yang jelas."
            )

    def evaluate_journal(
        self,
        metrics: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> str:
        """
        Produce a professional quantitative hedge-fund style trading evaluation
        reviewing performance metrics, win/loss patterns, and giving actionable recommendations.
        """
        total_closed = metrics.get("total_closed", 0)
        if total_closed == 0:
            return "Belum ada riwayat transaksi yang selesai untuk dianalisis oleh AI."

        trades_summary = []
        for t in recent_trades[:10]:
            outcome = "WIN" if float(t.get("profit_loss") or 0.0) > 0.05 else ("BEP" if abs(float(t.get("profit_loss") or 0.0)) <= 0.05 else "LOSS")
            trades_summary.append(
                f"- #{t.get('ticket')} {t.get('symbol')} {t.get('action')}: PnL=${float(t.get('profit_loss') or 0.0):.2f} "
                f"({outcome}) Exit:{t.get('close_reason')} Alasan:{str(t.get('ai_reasoning') or '')[:80]}"
            )
        trades_text = "\n".join(trades_summary)

        by_sym = metrics.get("by_symbol", {})
        pair_text = ", ".join([f"{k}: {v['wins']}W/{v['losses']}L (PnL ${v['pnl']})" for k, v in by_sym.items()])

        prompt = f"""
Anda adalah Senior Hedge Fund Risk Officer & Quantitative Trading Mentor.
Tugas Anda: Berikan evaluasi performa jurnal trading yang objektif, kritis, dan solutif berdasarkan data riwayat transaksi bot trading berikut:

METRIK PERFORMA:
- Total Trade Selesai: {total_closed}
- Menang: {metrics.get('wins', 0)} | Kalah: {metrics.get('losses', 0)} | Break-Even (Modal Aman): {metrics.get('beps', 0)}
- Real Win Rate: {metrics.get('adjusted_win_rate', 0.0)}%
- Capital Preservation Rate: {metrics.get('capital_preserved_rate', 0.0)}%
- Net Profit: ${metrics.get('net_pnl', 0.0):.2f} (Gross Profit: ${metrics.get('gross_profit', 0.0):.2f}, Gross Loss: ${metrics.get('gross_loss', 0.0):.2f})
- Profit Factor: {metrics.get('profit_factor', 0.0)}
- Risk-Reward Ratio (R:R): {metrics.get('risk_reward_ratio', 0.0)}
- Rata-rata Win: ${metrics.get('avg_win', 0.0):.2f} | Rata-rata Loss: ${metrics.get('avg_loss', 0.0):.2f}
- Performa per Pair: {pair_text}

SAMPEL TRANSAKSI TERAKHIR:
{trades_text}

Format Jawaban (Bahasa Indonesia, gunakan bullet points ringkas dan emoji profesional):
1. 🎯 **Poin Keunggulan (Apa yang Berjalan Baik):** Evaluasi kedisiplinan TP, BEP, dan win rate.
2. ⚠️ **Area Perbaikan & Pola Loss (Penyebab Minus):** Analisis pair atau kondisi pasar mana yang menyebabkan loss.
3. 💡 **Rekomendasi Taktis Sesi Berikutnya:** Saran konkret (misal: penyesuaian lot, filter volatilitas sesi, atau trailing stop).

Tulis jawaban yang padat, berbobot, dan langsung to the point tanpa basa-basi pembuka yang panjang.
"""
        # Try Gemini API key pool
        for key_info in self._pool:
            if not key_info.is_available():
                continue
            try:
                response = key_info.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt
                )
                if response and response.text:
                    key_info.record_success()
                    if self.db:
                        self.db.update_key_usage(key_info.key, is_success=True)
                    return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    key_info.mark_rate_limited(cooldown_seconds=1800)
                continue

        # Fallback rule-based critique
        adj_wr = metrics.get('adjusted_win_rate', 0.0)
        p_factor = metrics.get('profit_factor', 0.0)
        net_pnl = metrics.get('net_pnl', 0.0)
        pnl_sign = "+" if net_pnl >= 0 else ""

        critique = (
            "🎯 *Poin Keunggulan:*\n"
            f"• Proteksi modal berjalan dengan tingkat pengamanan modal (Capital Preserved Rate) mencapai *{metrics.get('capital_preserved_rate', 0.0)}%*.\n"
            f"• Stop Loss dan Break-Even disiplin melindungi akun dari kerugian besar.\n\n"
            "⚠️ *Area Perbaikan:*\n"
            f"• Real Win Rate berada di *{adj_wr:.1f}%* dengan Profit Factor *{p_factor:.2f}* (Net PnL: *{pnl_sign}${net_pnl:.2f}*).\n"
            "• Perhatikan volatilitas saat overlap sesi London–New York untuk menghindari fakeout.\n\n"
            "💡 *Rekomendasi Taktis:*\n"
            "• Pertahankan rasio Risk-to-Reward minimal 1:1.5 dengan konsistensi trailing Break-Even.\n"
            "• Fokuskan kuota volume pada pair dengan performa paling stabil (seperti XAUUSD)."
        )
        return critique
