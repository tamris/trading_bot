"""
Professional Trading Journal engine.

Builds a normalized list of trades using the broker deal history as the source of truth
(exact entry/exit timestamps and realized PnL, including partial closes), enriched with the
AI reasoning / confluence score stored in SQLite. Computes professional journal metrics:
equity curve, max drawdown, profit factor, expectancy, streaks, daily calendar PnL,
and breakdowns per pair, session, side and weekday.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.database import WIB, parse_any_time

logger = logging.getLogger("ai_trading_bot.journal")

BEP_THRESHOLD = 0.05  # |PnL| <= $0.05 is treated as Break-Even

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]
WEEKDAYS_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def session_of(dt: Optional[datetime]) -> str:
    """Map an entry time (WIB) to the trading session used by the bot."""
    if dt is None:
        return "Unknown"
    h = dt.astimezone(WIB).hour
    if 6 <= h < 14:
        return "Asia"
    if 14 <= h < 19:
        return "London"
    if 19 <= h < 21:
        return "London-NY Overlap"
    if h >= 21 or h < 4:
        return "New York"
    return "Rollover"


def outcome_of(pnl: float) -> str:
    if pnl > BEP_THRESHOLD:
        return "WIN"
    if pnl < -BEP_THRESHOLD:
        return "LOSS"
    return "BEP"


@dataclass
class JournalTrade:
    ticket: int
    symbol: str
    action: str
    lot: float
    entry_price: float
    exit_price: Optional[float]
    sl_price: float
    tp_price: float
    pnl: float
    entry_dt: Optional[datetime]
    close_dt: Optional[datetime]
    status: str = "CLOSED"
    close_reason: str = ""
    confluence_score: int = 0
    ai_reasoning: str = ""
    outcome: str = field(default="")

    def __post_init__(self):
        if not self.outcome:
            self.outcome = "OPEN" if self.status == "OPEN" else outcome_of(self.pnl)

    @property
    def session(self) -> str:
        return session_of(self.entry_dt)

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.entry_dt and self.close_dt:
            return max((self.close_dt - self.entry_dt).total_seconds() / 60.0, 0.0)
        return None

    @property
    def close_date(self) -> Optional[date]:
        return self.close_dt.astimezone(WIB).date() if self.close_dt else None

    @property
    def price_decimals(self) -> int:
        s = self.symbol.upper()
        if "JPY" in s:
            return 3
        if "BTC" in s or "XAU" in s:
            return 2
        return 5


def format_duration(minutes: Optional[float]) -> str:
    if minutes is None:
        return "-"
    m = int(round(minutes))
    if m < 60:
        return f"{m}m"
    h, mm = divmod(m, 60)
    if h < 24:
        return f"{h}j {mm}m"
    d, hh = divmod(h, 24)
    return f"{d}h {hh}j"


class TradingJournal:
    """Collects trades and computes professional journal analytics."""

    def __init__(self, db: Any, broker: Optional[Any] = None):
        self.db = db
        self.broker = broker

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_trades(self, days_back: int = 90, include_open: bool = True) -> List[JournalTrade]:
        db_rows: List[Dict[str, Any]] = []
        try:
            db_rows = self.db.get_all_journal_trades(limit=5000)
        except Exception as e:
            logger.warning(f"[Journal] Could not read DB trades: {e}")
        db_map = {int(r["ticket"]): r for r in db_rows if r.get("ticket") is not None}

        reset_dt = None
        try:
            reset_dt = parse_any_time(self.db.get_last_reset_time())
        except Exception:
            reset_dt = None

        trades: Dict[int, JournalTrade] = {}

        # 1. Broker history = source of truth for closed trades
        broker_closed: List[Dict[str, Any]] = []
        if self.broker is not None and hasattr(self.broker, "get_recent_closed_positions"):
            try:
                broker_closed = self.broker.get_recent_closed_positions(days_back=days_back) or []
            except Exception as e:
                logger.warning(f"[Journal] Broker history unavailable, falling back to DB: {e}")

        for rc in broker_closed:
            try:
                t = int(rc["ticket"])
            except Exception:
                continue
            row = db_map.get(t, {})
            close_dt = parse_any_time(rc.get("close_time")) or parse_any_time(row.get("close_time"))
            entry_dt = parse_any_time(rc.get("entry_time")) or parse_any_time(row.get("entry_time"))
            pnl = round(float(rc.get("profit") or 0.0), 2)
            trades[t] = JournalTrade(
                ticket=t,
                symbol=str(rc.get("symbol") or row.get("symbol") or ""),
                action=str(rc.get("action") or row.get("action") or "").upper(),
                lot=float(rc.get("lot") or row.get("lot") or 0.01),
                entry_price=float(rc.get("entry_price") or row.get("entry_price") or 0.0),
                exit_price=float(rc.get("close_price") or 0.0),
                sl_price=float(row.get("sl_price") or 0.0),
                tp_price=float(row.get("tp_price") or 0.0),
                pnl=pnl,
                entry_dt=entry_dt,
                close_dt=close_dt,
                status="CLOSED",
                close_reason=outcome_reason(pnl),
                confluence_score=int(row.get("confluence_score") or 0),
                ai_reasoning=str(row.get("ai_reasoning") or ""),
            )

        # 2. DB closed trades not present in broker window (older history / MT5 backend)
        for t, row in db_map.items():
            if t in trades or str(row.get("status", "")).upper() != "CLOSED":
                continue
            pnl = round(float(row.get("profit_loss") or 0.0), 2)
            trades[t] = JournalTrade(
                ticket=t,
                symbol=str(row.get("symbol") or ""),
                action=str(row.get("action") or "").upper(),
                lot=float(row.get("lot") or 0.01),
                entry_price=float(row.get("entry_price") or 0.0),
                exit_price=float(row.get("close_price") or 0.0),
                sl_price=float(row.get("sl_price") or 0.0),
                tp_price=float(row.get("tp_price") or 0.0),
                pnl=pnl,
                entry_dt=parse_any_time(row.get("entry_time")),
                close_dt=parse_any_time(row.get("close_time")),
                status="CLOSED",
                close_reason=str(row.get("close_reason") or outcome_reason(pnl)),
                confluence_score=int(row.get("confluence_score") or 0),
                ai_reasoning=str(row.get("ai_reasoning") or ""),
            )

        # Respect /resetdb: hide closed trades that ended before the last reset
        if reset_dt:
            trades = {
                k: v for k, v in trades.items()
                if v.close_dt is None or v.close_dt > reset_dt
            }

        closed_list = sorted(
            trades.values(),
            key=lambda x: (x.close_dt or x.entry_dt or datetime.min.replace(tzinfo=WIB))
        )

        if not include_open:
            return closed_list

        # 3. Open positions (live floating PnL from broker when available)
        open_list: List[JournalTrade] = []
        live_positions: List[Dict[str, Any]] = []
        if self.broker is not None and hasattr(self.broker, "get_open_positions"):
            try:
                live_positions = self.broker.get_open_positions() or []
            except Exception:
                live_positions = []

        if live_positions:
            for p in live_positions:
                t = int(p["ticket"])
                if t in trades:
                    continue
                row = db_map.get(t, {})
                open_list.append(JournalTrade(
                    ticket=t,
                    symbol=str(p.get("symbol") or ""),
                    action=str(p.get("type") or "").upper(),
                    lot=float(p.get("volume") or 0.01),
                    entry_price=float(p.get("price_open") or 0.0),
                    exit_price=float(p.get("price_current") or 0.0),
                    sl_price=float(p.get("sl") or 0.0),
                    tp_price=float(p.get("tp") or 0.0),
                    pnl=round(float(p.get("profit") or 0.0), 2),
                    entry_dt=parse_any_time(p.get("time")) or parse_any_time(row.get("entry_time")),
                    close_dt=None,
                    status="OPEN",
                    confluence_score=int(row.get("confluence_score") or 0),
                    ai_reasoning=str(row.get("ai_reasoning") or ""),
                ))
        else:
            for t, row in db_map.items():
                if t in trades or str(row.get("status", "")).upper() != "OPEN":
                    continue
                open_list.append(JournalTrade(
                    ticket=t,
                    symbol=str(row.get("symbol") or ""),
                    action=str(row.get("action") or "").upper(),
                    lot=float(row.get("lot") or 0.01),
                    entry_price=float(row.get("entry_price") or 0.0),
                    exit_price=None,
                    sl_price=float(row.get("sl_price") or 0.0),
                    tp_price=float(row.get("tp_price") or 0.0),
                    pnl=0.0,
                    entry_dt=parse_any_time(row.get("entry_time")),
                    close_dt=None,
                    status="OPEN",
                    confluence_score=int(row.get("confluence_score") or 0),
                    ai_reasoning=str(row.get("ai_reasoning") or ""),
                ))

        return closed_list + sorted(open_list, key=lambda x: x.ticket)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_stats(trades: List[JournalTrade]) -> Dict[str, Any]:
        closed = [t for t in trades if t.status == "CLOSED"]
        open_trades = [t for t in trades if t.status == "OPEN"]
        closed_sorted = sorted(
            closed, key=lambda x: (x.close_dt or x.entry_dt or datetime.min.replace(tzinfo=WIB))
        )

        wins = [t for t in closed if t.outcome == "WIN"]
        losses = [t for t in closed if t.outcome == "LOSS"]
        beps = [t for t in closed if t.outcome == "BEP"]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        net_pnl = sum(t.pnl for t in closed)
        n_closed = len(closed)
        decided = len(wins) + len(losses)

        win_rate = (len(wins) / decided * 100.0) if decided else 0.0
        preserved = ((len(wins) + len(beps)) / n_closed * 100.0) if n_closed else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = net_pnl / n_closed if n_closed else 0.0

        # Equity curve & max drawdown (closed-trade based)
        equity_points = []
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in closed_sorted:
            cum += t.pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            equity_points.append((t.close_dt, round(cum, 2), t))

        # Streaks (BEP neither extends nor breaks a streak)
        max_win_streak = max_loss_streak = cur_w = cur_l = 0
        for t in closed_sorted:
            if t.outcome == "WIN":
                cur_w += 1
                cur_l = 0
            elif t.outcome == "LOSS":
                cur_l += 1
                cur_w = 0
            max_win_streak = max(max_win_streak, cur_w)
            max_loss_streak = max(max_loss_streak, cur_l)

        current_streak = ""
        for t in reversed(closed_sorted):
            if t.outcome == "BEP":
                continue
            kind = t.outcome
            count = 0
            for u in reversed(closed_sorted):
                if u.outcome == "BEP":
                    continue
                if u.outcome == kind:
                    count += 1
                else:
                    break
            current_streak = f"{count}{'W' if kind == 'WIN' else 'L'}"
            break

        daily = TradingJournal.daily_pnl(closed)
        day_values = [d["pnl"] for d in daily.values()]
        green_days = sum(1 for v in day_values if v > BEP_THRESHOLD)
        red_days = sum(1 for v in day_values if v < -BEP_THRESHOLD)
        best_day = max(daily.items(), key=lambda kv: kv[1]["pnl"]) if daily else None
        worst_day = min(daily.items(), key=lambda kv: kv[1]["pnl"]) if daily else None

        durations = [t.duration_minutes for t in closed if t.duration_minutes is not None]
        avg_duration = sum(durations) / len(durations) if durations else None

        def _group(key_fn) -> Dict[str, Dict[str, Any]]:
            groups: Dict[str, Dict[str, Any]] = {}
            for t in closed:
                k = key_fn(t)
                g = groups.setdefault(k, {"total": 0, "wins": 0, "losses": 0, "beps": 0, "pnl": 0.0})
                g["total"] += 1
                g["pnl"] += t.pnl
                if t.outcome == "WIN":
                    g["wins"] += 1
                elif t.outcome == "LOSS":
                    g["losses"] += 1
                else:
                    g["beps"] += 1
            for g in groups.values():
                d = g["wins"] + g["losses"]
                g["win_rate"] = (g["wins"] / d * 100.0) if d else 0.0
                g["pnl"] = round(g["pnl"], 2)
            return dict(sorted(groups.items(), key=lambda kv: kv[1]["pnl"], reverse=True))

        pnl_values = [t.pnl for t in closed]
        return {
            "total_closed": n_closed,
            "open_trades": len(open_trades),
            "floating_pnl": round(sum(t.pnl for t in open_trades), 2),
            "wins": len(wins),
            "losses": len(losses),
            "beps": len(beps),
            "win_rate": round(win_rate, 1),
            "adjusted_win_rate": round(win_rate, 1),
            "capital_preserved_rate": round(preserved, 1),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": profit_factor if profit_factor == float("inf") else round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward_ratio": round(rr, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(max_dd, 2),
            "best_trade": round(max(pnl_values), 2) if pnl_values else 0.0,
            "worst_trade": round(min(pnl_values), 2) if pnl_values else 0.0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_streak": current_streak or "-",
            "trading_days": len(daily),
            "green_days": green_days,
            "red_days": red_days,
            "best_day": (best_day[0], round(best_day[1]["pnl"], 2)) if best_day else None,
            "worst_day": (worst_day[0], round(worst_day[1]["pnl"], 2)) if worst_day else None,
            "avg_duration_min": avg_duration,
            "equity_points": equity_points,
            "daily": daily,
            "by_symbol": _group(lambda t: t.symbol or "UNKNOWN"),
            "by_session": _group(lambda t: t.session),
            "by_side": _group(lambda t: t.action or "-"),
            "by_weekday": _group(lambda t: WEEKDAYS_ID[t.close_date.weekday()] if t.close_date else "-"),
        }

    @staticmethod
    def daily_pnl(trades: List[JournalTrade]) -> Dict[date, Dict[str, Any]]:
        daily: Dict[date, Dict[str, Any]] = {}
        for t in trades:
            if t.status != "CLOSED" or t.close_date is None:
                continue
            d = daily.setdefault(t.close_date, {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "beps": 0})
            d["pnl"] += t.pnl
            d["trades"] += 1
            if t.outcome == "WIN":
                d["wins"] += 1
            elif t.outcome == "LOSS":
                d["losses"] += 1
            else:
                d["beps"] += 1
        for d in daily.values():
            d["pnl"] = round(d["pnl"], 2)
        return dict(sorted(daily.items()))

    @staticmethod
    def month_stats(daily: Dict[date, Dict[str, Any]], year: int, month: int) -> Dict[str, Any]:
        days = {d: v for d, v in daily.items() if d.year == year and d.month == month}
        vals = [v["pnl"] for v in days.values()]
        return {
            "net": round(sum(vals), 2),
            "trades": sum(v["trades"] for v in days.values()),
            "wins": sum(v["wins"] for v in days.values()),
            "losses": sum(v["losses"] for v in days.values()),
            "trading_days": len(days),
            "green_days": sum(1 for v in vals if v > BEP_THRESHOLD),
            "red_days": sum(1 for v in vals if v < -BEP_THRESHOLD),
            "best": max(days.items(), key=lambda kv: kv[1]["pnl"]) if days else None,
            "worst": min(days.items(), key=lambda kv: kv[1]["pnl"]) if days else None,
        }


def outcome_reason(pnl: float) -> str:
    o = outcome_of(pnl)
    return {"WIN": "TAKE_PROFIT", "LOSS": "STOP_LOSS", "BEP": "BREAK_EVEN"}[o]


def shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def export_journal_excel(trades: List[JournalTrade], stats: Dict[str, Any]) -> bytes:
    """Generate a fully-styled, professional multi-sheet Excel (.xlsx) workbook."""
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default blank sheet

    font_title = Font(name="Calibri", size=15, bold=True, color="0F172A")
    font_sub = Font(name="Calibri", size=10, italic=True, color="64748B")
    font_hdr = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True)
    font_regular = Font(name="Calibri", size=11)

    fill_hdr = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    fill_hdr_accent = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
    fill_green = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    fill_red = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
    fill_gray = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    fill_blue = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")

    color_green = Font(name="Calibri", size=11, bold=True, color="166534")
    color_red = Font(name="Calibri", size=11, bold=True, color="991B1B")

    thin_border = Border(
        left=Side(style='thin', color="CBD5E1"),
        right=Side(style='thin', color="CBD5E1"),
        top=Side(style='thin', color="CBD5E1"),
        bottom=Side(style='thin', color="CBD5E1")
    )

    # ==========================================
    # SHEET 1: DASHBOARD & METRIK
    # ==========================================
    ws1 = wb.create_sheet(title="Dashboard & KPI")
    ws1.views.sheetView[0].showGridLines = True

    ws1["A1"] = "AI TRADING BOT — JURNAL TRADING & EVALUASI PERFORMA"
    ws1["A1"].font = font_title
    ws1["A2"] = f"Dibuat: {datetime.now(WIB).strftime('%d %B %Y %H:%M:%S WIB')}"
    ws1["A2"].font = font_sub

    ws1["A4"] = "METRIK UTAMA"
    ws1["B4"] = "NILAI"
    ws1["A4"].font = font_hdr
    ws1["B4"].font = font_hdr
    ws1["A4"].fill = fill_hdr
    ws1["B4"].fill = fill_hdr

    pf_val = "∞" if stats.get('profit_factor') == float('inf') else f"{float(stats.get('profit_factor') or 0.0):.2f}"
    kpi_rows = [
        ("Net PnL Akumulasi ($)", f"${stats.get('net_pnl', 0.0):,.2f}"),
        ("Gross Profit ($)", f"${stats.get('gross_profit', 0.0):,.2f}"),
        ("Gross Loss ($)", f"${stats.get('gross_loss', 0.0):,.2f}"),
        ("Profit Factor", pf_val),
        ("Real Win Rate (%)", f"{stats.get('win_rate', 0.0):.1f}%"),
        ("Capital Preserved Rate (%)", f"{stats.get('capital_preserved_rate', 0.0):.1f}%"),
        ("Total Trade Selesai", stats.get('total_closed', 0)),
        ("Trade Menang (TP)", stats.get('wins', 0)),
        ("Trade Kalah (SL)", stats.get('losses', 0)),
        ("Trade Break-Even (BEP)", stats.get('beps', 0)),
        ("Posisi Masih Terbuka", stats.get('open_trades', 0)),
        ("Average Win ($)", f"${stats.get('avg_win', 0.0):,.2f}"),
        ("Average Loss ($)", f"${stats.get('avg_loss', 0.0):,.2f}"),
        ("Risk-to-Reward Ratio (R:R)", f"{stats.get('risk_reward_ratio', 0.0):.2f}R"),
        ("Expectancy per Trade ($)", f"${stats.get('expectancy', 0.0):,.2f}"),
        ("Max Drawdown ($)", f"${stats.get('max_drawdown', 0.0):,.2f}"),
        ("Best Trade ($)", f"${stats.get('best_trade', 0.0):,.2f}"),
        ("Worst Trade ($)", f"${stats.get('worst_trade', 0.0):,.2f}"),
        ("Max Win Streak", f"{stats.get('max_win_streak', 0)}W"),
        ("Max Loss Streak", f"{stats.get('max_loss_streak', 0)}L"),
    ]

    for idx, (label, val) in enumerate(kpi_rows, start=5):
        ws1[f"A{idx}"] = label
        ws1[f"B{idx}"] = val
        ws1[f"A{idx}"].border = thin_border
        ws1[f"B{idx}"].border = thin_border
        ws1[f"A{idx}"].font = font_bold if "Net" in label or "Win Rate" in label else font_regular
        ws1[f"B{idx}"].font = font_bold if "Net" in label or "Win Rate" in label else font_regular
        ws1[f"B{idx}"].alignment = Alignment(horizontal="right")

    # Pair breakdown on column D-J
    ws1["D4"] = "PAIR / SYMBOL"
    ws1["E4"] = "TOTAL"
    ws1["F4"] = "WIN"
    ws1["G4"] = "LOSS"
    ws1["H4"] = "BEP"
    ws1["I4"] = "WIN RATE"
    ws1["J4"] = "NET PNL ($)"
    for col in ["D", "E", "F", "G", "H", "I", "J"]:
        ws1[f"{col}4"].font = font_hdr
        ws1[f"{col}4"].fill = fill_hdr_accent
        ws1[f"{col}4"].alignment = Alignment(horizontal="center")

    by_sym = stats.get("by_symbol") or {}
    r = 5
    for sym, sdata in by_sym.items():
        ws1[f"D{r}"] = sym
        ws1[f"E{r}"] = sdata["total"]
        ws1[f"F{r}"] = sdata["wins"]
        ws1[f"G{r}"] = sdata["losses"]
        ws1[f"H{r}"] = sdata["beps"]
        ws1[f"I{r}"] = f"{sdata['win_rate']:.1f}%"
        ws1[f"J{r}"] = f"${sdata['pnl']:,.2f}"
        for col in ["D", "E", "F", "G", "H", "I", "J"]:
            ws1[f"{col}{r}"].border = thin_border
            ws1[f"{col}{r}"].font = font_regular
            if col != "D":
                ws1[f"{col}{r}"].alignment = Alignment(horizontal="center" if col != "J" else "right")
        r += 1

    # ==========================================
    # SHEET 2: KALENDER HARIAN (DAILY PNL)
    # ==========================================
    ws2 = wb.create_sheet(title="Kalender PnL Harian")
    ws2.views.sheetView[0].showGridLines = True

    ws2["A1"] = "KALENDER REKAP PNL HARIAN"
    ws2["A1"].font = font_title
    ws2["A2"] = "Rekapitulasi keuntungan/kerugian per hari trading beserta jumlah trade."
    ws2["A2"].font = font_sub

    headers_d = ["Tanggal (WIB)", "Hari", "Total Trade", "Win", "Loss", "BEP", "Net PnL ($)", "Status"]
    for col_idx, h in enumerate(headers_d, start=1):
        c = ws2.cell(row=4, column=col_idx, value=h)
        c.font = font_hdr
        c.fill = fill_hdr
        c.alignment = Alignment(horizontal="center")

    daily = stats.get("daily") or {}
    r = 5
    for d, dinfo in sorted(daily.items(), reverse=True):
        pnl = dinfo["pnl"]
        day_name = WEEKDAYS_ID[d.weekday()]
        status = "HIJAU (PROFIT)" if pnl > BEP_THRESHOLD else ("MERAH (LOSS)" if pnl < -BEP_THRESHOLD else "BEP / NETRAL")
        ws2.cell(row=r, column=1, value=d.strftime("%Y-%m-%d")).alignment = Alignment(horizontal="center")
        ws2.cell(row=r, column=2, value=day_name).alignment = Alignment(horizontal="center")
        ws2.cell(row=r, column=3, value=dinfo["trades"]).alignment = Alignment(horizontal="center")
        ws2.cell(row=r, column=4, value=dinfo["wins"]).alignment = Alignment(horizontal="center")
        ws2.cell(row=r, column=5, value=dinfo["losses"]).alignment = Alignment(horizontal="center")
        ws2.cell(row=r, column=6, value=dinfo["beps"]).alignment = Alignment(horizontal="center")
        pnl_cell = ws2.cell(row=r, column=7, value=f"${pnl:,.2f}")
        pnl_cell.alignment = Alignment(horizontal="right")
        pnl_cell.font = color_green if pnl > BEP_THRESHOLD else (color_red if pnl < -BEP_THRESHOLD else font_bold)
        st_cell = ws2.cell(row=r, column=8, value=status)
        st_cell.alignment = Alignment(horizontal="center")
        st_cell.fill = fill_green if pnl > BEP_THRESHOLD else (fill_red if pnl < -BEP_THRESHOLD else fill_gray)
        for c in range(1, 9):
            ws2.cell(row=r, column=c).border = thin_border
        r += 1

    # ==========================================
    # SHEET 3: RIWAYAT TRANSAKSI LENGKAP (ALL TRADES)
    # ==========================================
    ws3 = wb.create_sheet(title="Log Semua Trade")
    ws3.views.sheetView[0].showGridLines = True

    ws3["A1"] = "JURNAL TRANSAKSI LENGKAP"
    ws3["A1"].font = font_title
    ws3["A2"] = "Daftar detail seluruh transaksi yang tercatat di akun trading."
    ws3["A2"].font = font_sub

    headers_t = [
        "Tiket", "Waktu Entry (WIB)", "Waktu Exit (WIB)", "Durasi", "Sesi",
        "Pair", "Posisi", "Lot", "Harga Entry", "Harga Exit", "SL", "TP",
        "PnL ($)", "Hasil", "Status", "Alasan Exit", "Skor AI", "Analisa AI & Catatan"
    ]
    for col_idx, h in enumerate(headers_t, start=1):
        c = ws3.cell(row=4, column=col_idx, value=h)
        c.font = font_hdr
        c.fill = fill_hdr
        c.alignment = Alignment(horizontal="center")

    sorted_trades = sorted(
        trades,
        key=lambda x: (x.status == "OPEN", x.close_dt or x.entry_dt or datetime.min.replace(tzinfo=WIB)),
        reverse=True
    )

    r = 5
    for t in sorted_trades:
        dec = t.price_decimals
        ep_str = f"{t.entry_price:.{dec}f}"
        xp_str = f"{t.exit_price:.{dec}f}" if t.exit_price else "-"
        sl_str = f"{t.sl_price:.{dec}f}" if t.sl_price else "-"
        tp_str = f"{t.tp_price:.{dec}f}" if t.tp_price else "-"
        dur_str = format_duration(t.duration_minutes) if t.close_dt else "-"
        en_str = t.entry_dt.strftime("%Y-%m-%d %H:%M:%S") if t.entry_dt else "-"
        ex_str = t.close_dt.strftime("%Y-%m-%d %H:%M:%S") if t.close_dt else "Berjalan"

        row_vals = [
            t.ticket, en_str, ex_str, dur_str, t.session,
            t.symbol, t.action, f"{t.lot:.2f}", ep_str, xp_str, sl_str, tp_str,
            f"${t.pnl:,.2f}", t.outcome, t.status, t.close_reason or "-", t.confluence_score, t.ai_reasoning
        ]

        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws3.cell(row=r, column=col_idx, value=val)
            cell.font = font_regular
            cell.border = thin_border
            if col_idx in [1, 4, 5, 6, 7, 14, 15, 17]:
                cell.alignment = Alignment(horizontal="center")
            elif col_idx in [8, 9, 10, 11, 12, 13]:
                cell.alignment = Alignment(horizontal="right")

        # Outcome badge coloring
        out_cell = ws3.cell(row=r, column=14)
        if t.status == "OPEN":
            out_cell.fill = fill_blue
        elif t.outcome == "WIN":
            out_cell.fill = fill_green
            out_cell.font = color_green
        elif t.outcome == "LOSS":
            out_cell.fill = fill_red
            out_cell.font = color_red
        else:
            out_cell.fill = fill_gray

        r += 1

    # Auto-fit column widths for all sheets
    for ws in [ws1, ws2, ws3]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or "")
                if cell.row in [1, 2]:
                    continue
                max_len = max(max_len, len(val_str))
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
