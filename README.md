# AI-Driven News Sentiment & Technical Confluence Trading Bot (MT5)

Sistem bot *algorithmic trading* berbasis Python yang dirancang untuk memantau, menganalisis, dan mengeksekusi posisi otomatis pada instrumen **XAUUSD (Gold)** dan **EURUSD (Forex)** di platform **MetaTrader 5 (MT5)** dengan integrasi **Hard Technical Rules**, **News Blackout Protection**, **Google Gemini AI Confluence Engine**, dan antarmuka kontrol **Telegram Bot**.

---

## 🌟 Fitur Utama

1. **Python Hard Technical Rules:**
   - **Timeframe:** Makro H1 & Eksekusi M15.
   - **Indikator:** EMA 200 (Filter Tren), EMA 50 (Dynamic Pullback), RSI 14 (Momentum), ATR 14 (Volatilitas Dinamis).
   - **Risk-to-Reward:** 1 : 2 (Stop Loss = 1.5 * ATR, Take Profit = 3.0 * ATR).

2. **Economic Calendar & News Blackout Protection:**
   - Mengambil data kalender ekonomi High Impact untuk mata uang USD & EUR.
   - Memblokir eksekusi order baru dalam jendela **+/- 15 menit** di sekitar rilis berita berdampak tinggi.

3. **Google Gemini AI Confluence Engine:**
   - Model: **`gemini-3.8-flash`** via SDK modern `google-genai`.
   - Mengambil 5–10 headline berita pasar terkini via RSS feed (ForexLive, FXStreet).
   - Memvalidasi keselarasan sentimen berita makro terhadap bias teknikal dengan skema **Structured JSON Output (Pydantic)**.
   - Transaksi **HANYA** dieksekusi jika `confluence_score >= 80` dan arah rekomendasi AI (`BUY`/`SELL`) selaras 100% dengan sinyal teknikal.

4. **Dynamic Risk Management:**
   - **1.0% Risk per Trade:** Ukuran lot dihitung dinamis berdasarkan jarak Stop Loss dalam pips/points dan nilai tick masing-masing pair.
   - **Max Open Positions:** Maksimal 1 posisi per pair, maksimal 2 posisi total bersamaan.
   - **Daily Drawdown Limit:** Jika floating + realized loss hari ini mencapai **3.0%**, bot otomatis mengunci trading baru hingga pergantian hari (00:00 UTC).
   - **Break-Even Point (BEP) Trailing:** Begitu profit mencapai 1 : 1 (1.5 * ATR), Stop Loss otomatis digeser ke harga Entry.

5. **Telegram Bot Control & Alerts:**
   - Notifikasi real-time setiap kali order dieksekusi lengkap dengan alasan AI dan detail SL/TP.
   - Perintah interaktif: `/start`, `/status`, `/pause`, `/resume`, `/closeall` (tombol darurat penutupan seluruh posisi).

---

## 📁 Struktur Direktori

```text
bot_trading/
├── config/
│   ├── __init__.py
│   └── settings.py              # Konfigurasi Pydantic Settings
├── core/
│   ├── __init__.py
│   ├── mt5_connector.py         # MT5 Bridge (Live + Mock Simulation Mode)
│   ├── technical_engine.py      # Kalkulasi EMA, RSI, ATR & Hard Rules
│   ├── news_scraper.py          # RSS News Scraper & Economic Calendar Filter
│   ├── gemini_analyst.py        # Gemini AI SDK & Structured Output
│   └── risk_manager.py          # 1% Lot Sizing, 3% Daily DD, BEP Trailing
├── bot/
│   ├── __init__.py
│   └── telegram_bot.py          # Telegram Bot command handlers & alerts
├── data/
│   └── calendar_cache.json      # Cache lokal kalender ekonomi
├── logs/
│   └── trading_bot.log          # File log rotasi otomatis
├── tests/
│   ├── test_technical_engine.py # Unit test indikator teknikal
│   ├── test_risk_manager.py     # Unit test lot sizing & risk engine
│   ├── test_news_scraper.py     # Unit test filter berita & blackout
│   └── test_gemini_analyst.py   # Unit test schema & filter konfluensi
├── .env.example                 # Template environment variables
├── .env                         # Konfigurasi aktif
├── requirements.txt             # Dependensi Python
├── check_mt5_connection.py     # Tool diagnostik koneksi MT5
├── pytest.ini                   # Konfigurasi test runner
└── main.py                      # Master orchestrator & scheduler APScheduler
```

---

## 🚀 Panduan Memulai

### 1. Konfigurasi Lingkungan (`.env`)
Salin file template `.env.example` menjadi `.env` dan lengkapi kredensial Anda:

```env
# MetaTrader 5 Configuration
MT5_LOGIN=12345678
MT5_PASSWORD=demo_password_anda
MT5_SERVER=Exness-MT5Trial
MT5_PATH=C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe
MT5_MOCK_MODE=False

# Google Gemini API
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-3.8-flash

# Telegram Bot
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=987654321
```

> **Catatan:** Jika `MT5_MOCK_MODE=True` atau kredensial akun belum diisi, bot akan otomatis berjalan dalam **Simulation / Mock Mode** yang aman untuk pengujian logika.

### 2. Uji Diagnostik Koneksi MT5
Sebelum menjalankan bot penuh, jalankan alat diagnostik untuk memastikan koneksi ke terminal dan akun MT5 Anda:
```powershell
python check_mt5_connection.py
```

### 3. Menjalankan Unit Test
Verifikasi seluruh formula matematika dan logika filter:
```powershell
python -m pytest tests/ -v
```

### 4. Menjalankan Bot Trading
- **Mode Uji Coba Sekali Jalan (Single Cycle / Dry Run):**
  ```powershell
  python main.py --mock --once
  ```
- **Mode Produksi 24/7 (Scheduler Otomatis M15):**
  ```powershell
  python main.py
  ```

---

## 📱 Perintah Telegram Bot

| Command | Fungsi |
| :--- | :--- |
| `/start` | Menampilkan ringkasan status bot, saldo/ekuitas akun, dan mode trading aktif. |
| `/status` | Mengecek kondisi market, daily drawdown hari ini, dan posisi terbuka. |
| `/pause` | Menghentikan eksekusi order otomatis (bot hanya memantau tanpa buka posisi). |
| `/resume` | Mengaktifkan kembali pembukaan posisi otomatis. |
| `/closeall` | **Panic Button:** Menutup seluruh posisi yang sedang terbuka di MT5 seketika. |
