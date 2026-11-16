# Project Blueprint: AI-Driven News Sentiment & Technical Confluence Trading Bot

## 1. Ringkasan Sistem (System Overview)
Sistem ini adalah bot *algorithmic trading* berbasis Python yang berjalan 24/7 untuk memantau, menganalisis, dan mengeksekusi posisi otomatis pada instrumen **XAUUSD (Gold)** dan **EURUSD (Forex)** di platform **MetaTrader 5 (MT5)** (diawali dengan lingkungan **Akun Demo**).

Sistem menggabungkan **Hard Technical Rules** (indikator teknikal pasti di sisi Python), **News & Economic Calendar Protection**, serta **Google Gemini API** sebagai mesin analisis sentimen dan validasi konfluensi sebelum mengeksekusi order.

---

## 2. Arsitektur Teknologi (Tech Stack)
- **Core Language:** Python 3.10+
- **Trading Bridge:** `MetaTrader5` (Official Python Library)
- **Technical Analysis:** `pandas`, `numpy`, `pandas-ta`
- **AI Engine:** `google-genai` (Model: `gemini-2.5-flash` / `gemini-2.5-flash-lite` dengan *Structured JSON Output*)
- **Data & News Feed:** RSS Feeds (ForexLive, FXStreet, Reuters) + Economic Calendar Scraper (Forex Factory / Investing)
- **Scheduler:** `APScheduler` (Eksekusi pengecekan setiap penutupan *candle* M15)
- **Monitoring & Control Interface:** `python-telegram-bot` (Notifikasi sinyal, rasional AI, laporan eksekusi, dan *kill-switch* manual)
- **Environment Awal:** Akun Demo MT5 (`Exness-MT5Trial` atau broker sejenis) di PC Lokal / Windows VPS.

---

## 3. Alur Kerja Sistem (Execution Pipeline)

```text
[Scheduler: Setiap 15 Menit (M15 Close)]
       │
       ▼
[1. Risk & Kill-Switch Check] ──► Cek Daily Drawdown (< 3%) & Posisi Aktif
       │ (Lolos)
       ▼
[2. News Protection Filter]   ──► Cek Kalender Ekonomi (USD/EUR High Impact +/- 15 Menit)
       │ (Aman)
       ▼
[3. Technical Hard Filter]    ──► Hitung EMA 200, EMA 50, RSI 14, & ATR 14 di Python
       │ (Setup Valid)
       ▼
[4. News Scraper & Gemini AI] ──► Kirim Data Teknikal + Headline Berita ke Gemini API
       │ (Output JSON: Confluence Score >= 80)
       ▼
[5. Position Sizing & Order]  ──► Hitung Lot Otomatis (Max Risk 1%) -> MT5.order_send()
       │
       ▼
[6. Telegram Reporting]       ──► Kirim Detail Entry, SL/TP, & Rasional Analisa ke Telegram
```

---

## 4. Aturan Indikator Teknikal Pasti (Hard Rules)
Python wajib menyaring kondisi pasar sebelum memanggil Gemini API (untuk menghemat *token* dan menghindari sinyal palsu):

### Timeframe
- **Tren Makro:** H1
- **Eksekusi Entry:** M15

### Syarat Sinyal BUY
1. Harga penutupan (Close) M15 berada **di atas EMA 200** (Tren Bullish).
2. Harga berada di area *pullback* dekat **EMA 50** atau *support* dinamis.
3. **RSI (14)** berada di rentang `40 - 65` (tidak *overbought* di atas 70).

### Syarat Sinyal SELL
1. Harga penutupan (Close) M15 berada **di bawah EMA 200** (Tren Bearish).
2. Harga berada di area *pullback* dekat **EMA 50** atau *resistance* dinamis.
3. **RSI (14)** berada di rentang `35 - 60` (tidak *oversold* di bawah 30).

### Perhitungan Stop Loss & Take Profit Dinamis
- Menggunakan indikator **ATR (14)** pada timeframe M15 agar adaptif terhadap volatilitas pasar:
  - **Stop Loss (SL):** `1.5 * ATR(14)` dari harga *entry*.
  - **Take Profit (TP):** `3.0 * ATR(14)` dari harga *entry* (Minimal *Risk-to-Reward Ratio* `1 : 2`).

---

## 5. Integrasi Gemini API & Prompt Specification
Jika *Technical Hard Rules* terpenuhi, sistem mengambil 5–10 *headline* berita finansial terbaru terkait **USD, EUR, The Fed, ECB, dan Geopolitik**, lalu mengirimkannya ke Gemini API dengan skema **JSON Output** wajib.

### Skema JSON Output Wajib dari Gemini
```json
{
  "symbol": "XAUUSD",
  "technical_bias": "BULLISH",
  "news_sentiment": "BEARISH_USD",
  "confluence_score": 85,
  "action": "BUY",
  "reasoning": "Sentimen pelemahan USD pasca pernyataan dovish pejabat The Fed selaras dengan pantulan harga emas di atas EMA 200 H1 dan dukungan momentum RSI."
}
```

### Validasi Keputusan AI
- Sistem **HANYA** mengeksekusi order jika:
  1. `confluence_score >= 80`
  2. `action` (`BUY` atau `SELL`) **sama persis** dengan `technical_bias` dari filter indikator Python.
  3. Jika tidak selaras, `action` otomatis dianggap `HOLD` dan dicatat di log saja.

---

## 6. Parameter Manajemen Risiko (Risk Management Engine)
Modul pengaman wajib dieksekusi tepat sebelum `MT5.order_send()` dipanggil:

| Aturan Risiko | Spesifikasi |
| :--- | :--- |
| **Mode Akun Tahap 1** | **MT5 Demo Account** (Wajib lulus uji coba minimal 2–4 minggu) |
| **Maksimal Risiko per Trade** | **1.0%** dari total *Balance/Equity* akun |
| **Kalkulasi Ukuran Lot** | Dihitung otomatis berdasarkan jarak SL (dalam pips/points) dan *tick value* masing-masing pair (`XAUUSD` vs `EURUSD`) agar kerugian maksimal tetap 1% |
| **Maksimal Posisi Aktif** | Maksimal **1 posisi terbuka** per *pair* (Total maksimal 2 posisi bersamaan) |
| **Daily Drawdown Limit** | Jika total kerugian terealisasi + *floating* dalam 1 hari mencapai **3.0%**, bot otomatis berhenti membuka posisi baru hingga pergantian hari (00:00 Server Time) |
| **News Protection Window** | Blokir semua eksekusi baru **15 menit sebelum hingga 15 menit sesudah** rilis berita berlabel *High Impact* (Folder Merah) untuk USD & EUR |
| **Trailing Stop / Break-Even** | Jika profit sudah mencapai `1 : 1` (1.5 * ATR), geser SL otomatis ke titik *Entry* (Break-Even Point / BEP) |

---

## 7. Struktur Direktori Proyek (Project Structure)
Gunakan struktur modular berikut saat men-generate kode bersama *AI Agent*:

```text
ai-trading-bot/
├── config/
│   └── settings.py          # Konfigurasi pair, risk %, API keys, timeframe, parameter indikator
├── core/
│   ├── mt5_connector.py     # Koneksi MT5, fetch candle OHLCV, cek saldo, eksekusi order_send
│   ├── technical_engine.py  # Kalkulasi EMA, RSI, ATR & validasi Hard Rules
│   ├── news_scraper.py      # Fetch RSS berita finansial & jadwal kalender ekonomi High Impact
│   ├── gemini_analyst.py    # Integrasi Gemini API, prompt engineering, & parsing JSON
│   └── risk_manager.py      # Hitung lot dinamis 1% risk, cek Daily Drawdown 3%, & BEP trailing
├── bot/
│   └── telegram_bot.py      # Notifikasi sinyal, rasional AI, status akun, & command (/status, /stop, /closeall)
├── logs/
│   └── trading_bot.log      # Pencatatan seluruh keputusan AI, skor konfluensi, & riwayat order
├── .env                     # Simpan MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, GEMINI_API_KEY, TELEGRAM_TOKEN
├── requirements.txt         # Daftar library Python
└── main.py                  # Entry point utama & scheduler 15 menit
```

---

## 8. Perintah Telegram Bot (Kontrol & Monitoring)
Bot Telegram berfungsi sebagai antarmuka pemantauan real-time:
- `/start` — Menampilkan status sistem, informasi akun demo, dan saldo/ekuitas terkini.
- `/status` — Mengecek kondisi market terakhir (harga XAUUSD & EURUSD, status indikator, dan *daily drawdown* hari ini).
- `/pause` — Menghentikan sementara eksekusi *auto-trade* (bot hanya mengirim analisis tanpa buka posisi).
- `/resume` — Mengaktifkan kembali mode *auto-trade*.
- `/closeall` — Tombol darurat (*panic button*) untuk menutup seluruh posisi yang sedang terbuka di MT5.

---

## 9. Instruksi Eksekusi untuk AI Coding Agent
Jika dokumen ini digunakan sebagai konteks untuk *AI Coding Agent*, ikuti urutan pembuatan kode berikut:
1. Buat file konfigurasi `.env` dan `config/settings.py` terlebih dahulu dengan parameter default untuk **Akun Demo**.
2. Bangun `core/mt5_connector.py` dan pastikan fungsi koneksi serta pengambilan data historis `XAUUSD` dan `EURUSD` berjalan tanpa error.
3. Implementasikan `core/technical_engine.py` dan `core/risk_manager.py` lengkap dengan *unit test* sederhana untuk memastikan rumus perhitungan *lot size* 1% risiko akurat untuk *tick size* Emas dan Forex.
4. Bangun `core/news_scraper.py` dan `core/gemini_analyst.py` menggunakan *Structured Outputs* JSON.
5. Integrasikan seluruh modul ke `main.py` dan hubungkan notifikasinya ke `bot/telegram_bot.py`.