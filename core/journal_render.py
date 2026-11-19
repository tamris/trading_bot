"""
Professional trading-journal visuals rendered with matplotlib (headless / Agg).

Produces PNG bytes for:
  1. Performance dashboard (KPI cards, equity curve + drawdown, daily PnL bars,
     per-pair performance, outcome donut, session performance)
  2. Monthly PnL calendar (daily PnL per cell + weekly totals)
  3. Trade log table (latest trades with entry/exit, PnL, duration, session)
"""
import calendar as pycal
import io
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from core.database import WIB  # noqa: E402
from core.journal import (  # noqa: E402
    BEP_THRESHOLD,
    MONTHS_ID,
    JournalTrade,
    TradingJournal,
    format_duration,
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG = "#0d1117"
CARD = "#161b22"
CARD_2 = "#1c2330"
BORDER = "#2a3140"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#22c55e"
GREEN_DARK = "#0f3d24"
RED = "#ef4444"
RED_DARK = "#4a1717"
ACCENT = "#7c83ff"
AMBER = "#f59e0b"
GRID = "#222936"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BORDER,
})
# Dollar signs must render literally (never as mathtext)
try:
    plt.rcParams["text.parse_math"] = False
except KeyError:  # very old matplotlib
    pass


def money(v: float, signed: bool = True) -> str:
    if signed:
        sign = "+" if v > 0 else ("-" if v < 0 else "")
        return f"{sign}${abs(v):,.2f}"
    return f"${v:,.2f}"


def pnl_color(v: float) -> str:
    if v > BEP_THRESHOLD:
        return GREEN
    if v < -BEP_THRESHOLD:
        return RED
    return MUTED


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _card(ax, title: Optional[str] = None):
    """Style an axes as a dark card with an optional title."""
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(BORDER)
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=10)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _empty_note(ax, msg: str = "Belum ada data"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)


def _pf_str(pf: Any) -> str:
    if pf == float("inf"):
        return "âˆž"
    return f"{float(pf):.2f}"


# ---------------------------------------------------------------------------
# 1. Dashboard
# ---------------------------------------------------------------------------
def render_dashboard(
    stats: Dict[str, Any],
    account: Optional[Dict[str, Any]] = None,
    title: str = "AI TRADE  \u2022  TRADING JOURNAL",
    period_label: str = "",
) -> bytes:
    fig = plt.figure(figsize=(16, 11.2), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(
        4, 12, left=0.035, right=0.975, top=0.90, bottom=0.05,
        hspace=0.55, wspace=0.6, height_ratios=[0.78, 2.1, 1.75, 0.0001]
    )

    now = datetime.now(WIB)
    fig.text(0.035, 0.955, title, fontsize=21, fontweight="bold", color=TEXT, va="center")
    sub = f"{period_label}   \u2022   Dibuat {now.strftime('%d %b %Y %H:%M')} WIB"
    fig.text(0.035, 0.925, sub, fontsize=10.5, color=MUTED, va="center")
    if account:
        bal = float(account.get("balance") or 0.0)
        eq = float(account.get("equity") or bal)
        fig.text(
            0.975, 0.955, f"Balance {money(bal, False)}   Equity {money(eq, False)}",
            fontsize=12.5, color=TEXT, ha="right", va="center", fontweight="bold"
        )
        fl = float(stats.get("floating_pnl") or 0.0)
        fig.text(
            0.975, 0.925,
            f"Posisi terbuka: {stats.get('open_trades', 0)}   Floating: {money(fl)}",
            fontsize=10.5, color=pnl_color(fl) if stats.get("open_trades") else MUTED, ha="right", va="center"
        )

    # ---- KPI cards -------------------------------------------------------
    net = stats["net_pnl"]
    kpis = [
        ("Net P&L", money(net), pnl_color(net),
         f"Gross +${stats['gross_profit']:,.2f} / -${stats['gross_loss']:,.2f}"),
        ("Win Rate", f"{stats['win_rate']:.1f}%", GREEN if stats["win_rate"] >= 50 else (AMBER if stats["win_rate"] > 0 else MUTED),
         f"{stats['wins']}W  {stats['losses']}L  {stats['beps']}BE"),
        ("Profit Factor", _pf_str(stats["profit_factor"]),
         GREEN if (stats["profit_factor"] == float("inf") or stats["profit_factor"] >= 1.5) else (AMBER if stats["profit_factor"] >= 1 else RED if stats["total_closed"] else MUTED),
         "Gross profit / gross loss"),
        ("Avg Win / Avg Loss", f"{stats['risk_reward_ratio']:.2f}R" if stats["risk_reward_ratio"] else "-", ACCENT,
         f"+${stats['avg_win']:,.2f} / -${stats['avg_loss']:,.2f}"),
        ("Expectancy", money(stats["expectancy"]), pnl_color(stats["expectancy"]), "Rata-rata P&L per trade"),
        ("Max Drawdown", f"-${stats['max_drawdown']:,.2f}", RED if stats["max_drawdown"] > 0 else MUTED,
         f"Streak terbaik {stats['max_win_streak']}W  \u2022  terburuk {stats['max_loss_streak']}L"),
    ]
    for i, (label, value, color, foot) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        _card(ax)
        _blank(ax)
        ax.add_patch(FancyBboxPatch((0, 0.0), 0.012, 1.0, boxstyle="square,pad=0", color=color, transform=ax.transAxes, clip_on=False))
        ax.text(0.07, 0.80, label.upper(), fontsize=9, color=MUTED, fontweight="bold", transform=ax.transAxes)
        ax.text(0.07, 0.45, value, fontsize=20, color=color, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.07, 0.13, foot, fontsize=8.2, color=MUTED, transform=ax.transAxes)

    # ---- Equity curve ----------------------------------------------------
    ax_eq = fig.add_subplot(gs[1, 0:8])
    _card(ax_eq, "Equity Curve (Cumulative P&L)")
    pts = stats.get("equity_points") or []
    if pts:
        xs = [p[0] or datetime.now(WIB) for p in pts]
        ys = [p[1] for p in pts]
        # Anchor the curve at 0 right before the first trade
        x0 = xs[0] - timedelta(minutes=30)
        xs = [x0] + xs
        ys = [0.0] + ys
        idx = list(range(len(xs)))
        ax_eq.plot(idx, ys, color=ACCENT, linewidth=2.4, zorder=3)
        ax_eq.fill_between(idx, ys, 0, where=[y >= 0 for y in ys], color=GREEN, alpha=0.18, interpolate=True, zorder=2)
        ax_eq.fill_between(idx, ys, 0, where=[y < 0 for y in ys], color=RED, alpha=0.18, interpolate=True, zorder=2)
        # Trade markers
        for i, (_, y, t) in enumerate(pts, start=1):
            ax_eq.scatter(i, y, s=34, color=pnl_color(t.pnl), edgecolor=CARD, linewidth=1.2, zorder=4)
        # Drawdown shading (peak to current)
        peak = 0.0
        peaks = []
        for y in ys:
            peak = max(peak, y)
            peaks.append(peak)
        ax_eq.fill_between(idx, ys, peaks, color=RED, alpha=0.07, zorder=1)
        ax_eq.axhline(0, color=BORDER, linewidth=1)
        ax_eq.annotate(
            money(ys[-1]), xy=(idx[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
            color=pnl_color(ys[-1]), fontsize=11, fontweight="bold", va="center"
        )
        # X labels: dates
        step = max(1, len(xs) // 8)
        ticks = idx[1::step]
        ax_eq.set_xticks(ticks)
        ax_eq.set_xticklabels([xs[i].strftime("%d/%m\n%H:%M") for i in ticks], fontsize=8)
        ax_eq.set_xlim(-0.3, len(idx) - 0.4 + max(1, len(idx) * 0.06))
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if abs(v) >= 10 else f"${v:,.1f}"))
        ax_eq.grid(axis="y", color=GRID, linewidth=0.8)
        ax_eq.set_axisbelow(True)
        ax_eq.set_xlabel("Urutan trade (waktu close WIB)", fontsize=9)
    else:
        _blank(ax_eq)
        _empty_note(ax_eq, "Belum ada trade yang selesai")

    # ---- Daily P&L bars --------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 8:12])
    _card(ax_d, "Daily P&L (14 hari terakhir)")
    daily = stats.get("daily") or {}
    if daily:
        today = datetime.now(WIB).date()
        days = [today - timedelta(days=i) for i in range(13, -1, -1)]
        vals = [daily.get(d, {}).get("pnl", 0.0) for d in days]
        colors = [pnl_color(v) if v != 0 else BORDER for v in vals]
        ax_d.bar(range(len(days)), vals, color=colors, width=0.68, zorder=3)
        ax_d.axhline(0, color=BORDER, linewidth=1)
        ax_d.set_xticks(range(len(days)))
        ax_d.set_xticklabels([d.strftime("%d") for d in days], fontsize=8)
        ax_d.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if abs(v) >= 10 else f"${v:,.1f}"))
        ax_d.grid(axis="y", color=GRID, linewidth=0.8)
        ax_d.set_axisbelow(True)
        vmax = max([abs(v) for v in vals] + [1.0])
        for i, v in enumerate(vals):
            if abs(v) > BEP_THRESHOLD:
                ax_d.text(i, v + (vmax * 0.05 if v > 0 else -vmax * 0.05), f"{v:+.1f}", ha="center",
                          va="bottom" if v > 0 else "top", fontsize=7, color=pnl_color(v))
        ax_d.set_ylim(-vmax * 1.3, vmax * 1.3)
        ax_d.set_xlabel(f"Tanggal ({MONTHS_ID[today.month - 1][:3]} {today.year})", fontsize=9)
    else:
        _blank(ax_d)
        _empty_note(ax_d)

    # ---- Performance by pair --------------------------------------------
    ax_p = fig.add_subplot(gs[2, 0:5])
    _card(ax_p, "Performa per Pair")
    by_sym = stats.get("by_symbol") or {}
    if by_sym:
        names = list(by_sym.keys())[::-1]
        vals = [by_sym[n]["pnl"] for n in names]
        vmax = max([abs(v) for v in vals] + [1.0])
        ax_p.barh(range(len(names)), vals, color=[pnl_color(v) for v in vals], height=0.55, zorder=3)
        ax_p.axvline(0, color=BORDER, linewidth=1)
        ax_p.set_xlim(-vmax * 2.6, vmax * 2.2)
        ax_p.set_ylim(-0.7, len(names) - 0.3)
        for i, n in enumerate(names):
            g = by_sym[n]
            v = g["pnl"]
            ax_p.text(-vmax * 2.5, i, n, va="center", ha="left", fontsize=10.5, color=TEXT, fontweight="bold")
            ax_p.text(v + (vmax * 0.05 if v >= 0 else -vmax * 0.05), i,
                      f"{money(v)}  WR {g['win_rate']:.0f}%  {g['wins']}W/{g['losses']}L/{g['beps']}BE",
                      va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=TEXT)
        ax_p.set_yticks([])
        ax_p.set_xticks([])
        ax_p.grid(False)
    else:
        _blank(ax_p)
        _empty_note(ax_p)

    # ---- Outcome donut ---------------------------------------------------
    ax_o = fig.add_subplot(gs[2, 5:8])
    _card(ax_o, "Komposisi Hasil")
    w, l, b = stats["wins"], stats["losses"], stats["beps"]
    if w + l + b > 0:
        sizes, cols, labels = [], [], []
        for n, c, lab in ((w, GREEN, "Win"), (l, RED, "Loss"), (b, MUTED, "BE")):
            if n:
                sizes.append(n)
                cols.append(c)
                labels.append(f"{lab} {n}")
        ax_o.pie(sizes, colors=cols, startangle=90, counterclock=False,
                 wedgeprops=dict(width=0.32, edgecolor=CARD, linewidth=2))
        ax_o.text(0, 0.08, f"{stats['win_rate']:.0f}%", ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT)
        ax_o.text(0, -0.22, "win rate", ha="center", va="center", fontsize=9, color=MUTED)
        ax_o.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
                    fontsize=8.5, labelcolor=TEXT, handlelength=1)
        ax_o.set_aspect("equal")
    else:
        _blank(ax_o)
        _empty_note(ax_o)

    # ---- Session performance --------------------------------------------
    ax_s = fig.add_subplot(gs[2, 8:12])
    _card(ax_s, "Performa per Sesi (WIB)")
    by_sess = stats.get("by_session") or {}
    if by_sess:
        order = ["Asia", "London", "London-NY Overlap", "New York", "Rollover", "Unknown"]
        names = [s for s in order if s in by_sess]
        vals = [by_sess[s]["pnl"] for s in names]
        short = {"London-NY Overlap": "LDN-NY", "New York": "NY"}
        ax_s.bar([short.get(n, n) for n in names], vals, color=[pnl_color(v) for v in vals], width=0.55, zorder=3)
        ax_s.axhline(0, color=BORDER, linewidth=1)
        vmax = max([abs(v) for v in vals] + [1.0])
        ax_s.set_ylim(-vmax * 1.45, vmax * 1.45)
        for i, n in enumerate(names):
            g = by_sess[n]
            v = g["pnl"]
            ax_s.text(i, v + (vmax * 0.06 if v >= 0 else -vmax * 0.06),
                      f"{money(v)}\n{g['total']} trade", ha="center", va="bottom" if v >= 0 else "top",
                      fontsize=8, color=TEXT)
        ax_s.tick_params(axis="x", labelsize=9, colors=TEXT)
        ax_s.set_yticks([])
    else:
        _blank(ax_s)
        _empty_note(ax_s)

    # Footer strip
    best = stats.get("best_day")
    worst = stats.get("worst_day")
    foot = (
        f"Hari trading: {stats['trading_days']}  (hijau {stats['green_days']} / merah {stats['red_days']})"
        f"     Hari terbaik: {best[0].strftime('%d %b')} {money(best[1])}" if best else
        f"Hari trading: {stats['trading_days']}"
    )
    if worst:
        foot += f"     Hari terburuk: {worst[0].strftime('%d %b')} {money(worst[1])}"
    foot += f"     Best trade: {money(stats['best_trade'])}     Worst trade: {money(stats['worst_trade'])}"
    foot += f"     Rata-rata durasi: {format_duration(stats.get('avg_duration_min'))}"
    fig.text(0.035, 0.018, foot, fontsize=9.2, color=MUTED, va="center")

    return _fig_to_png(fig)


# ---------------------------------------------------------------------------
# 2. Monthly calendar
# ---------------------------------------------------------------------------
def render_calendar(daily: Dict[date, Dict[str, Any]], year: int, month: int) -> bytes:
    ms = TradingJournal.month_stats(daily, year, month)
    weeks = pycal.Calendar(firstweekday=0).monthdatescalendar(year, month)
    n_weeks = len(weeks)

    fig = plt.figure(figsize=(16, 3.1 + n_weeks * 1.55), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.80])
    ax.set_facecolor(BG)
    ax.axis("off")

    cols = 8  # 7 days + weekly total
    gap = 0.12
    cw, ch = 1.0, 1.0
    header_h = 0.45
    ax.set_xlim(0, cols)
    ax.set_ylim(-(n_weeks * ch) - 0.05, header_h + 0.05)

    # Title + month stats
    fig.text(0.03, 0.95, f"Kalender P&L  \u2022  {MONTHS_ID[month - 1]} {year}", fontsize=21, fontweight="bold", color=TEXT, va="center")
    wr = (ms["wins"] / (ms["wins"] + ms["losses"]) * 100.0) if (ms["wins"] + ms["losses"]) else 0.0
    summary = [
        ("NET BULAN INI", money(ms["net"]), pnl_color(ms["net"])),
        ("TRADE", f"{ms['trades']}", TEXT),
        ("WIN RATE", f"{wr:.0f}%", GREEN if wr >= 50 else (AMBER if wr > 0 else MUTED)),
        ("HARI TRADING", f"{ms['trading_days']}  ({ms['green_days']}H / {ms['red_days']}M)", TEXT),
        ("HARI TERBAIK", f"{ms['best'][0].day} {MONTHS_ID[month - 1][:3]}  {money(ms['best'][1]['pnl'])}" if ms["best"] else "-", GREEN if ms["best"] and ms["best"][1]["pnl"] > 0 else MUTED),
        ("HARI TERBURUK", f"{ms['worst'][0].day} {MONTHS_ID[month - 1][:3]}  {money(ms['worst'][1]['pnl'])}" if ms["worst"] else "-", RED if ms["worst"] and ms["worst"][1]["pnl"] < 0 else MUTED),
    ]
    x = 0.03
    widths = [0.13, 0.07, 0.09, 0.17, 0.2, 0.2]
    for (lab, val, colr), wdt in zip(summary, widths):
        fig.text(x, 0.895, lab, fontsize=8.5, color=MUTED, fontweight="bold", va="center")
        fig.text(x, 0.862, val, fontsize=13, color=colr, fontweight="bold", va="center")
        x += wdt

    # Header row
    day_names = ["SEN", "SEL", "RAB", "KAM", "JUM", "SAB", "MIN", "MINGGUAN"]
    for c, name in enumerate(day_names):
        ax.text(c + 0.5, header_h / 2, name, ha="center", va="center", fontsize=10, fontweight="bold",
                color=ACCENT if c == 7 else MUTED)

    month_vals = [v["pnl"] for d, v in daily.items() if d.year == year and d.month == month]
    vmax = max([abs(v) for v in month_vals] + [1.0])
    today = datetime.now(WIB).date()

    for r, week in enumerate(weeks):
        y0 = -(r + 1) * ch
        week_pnl = 0.0
        week_trades = 0
        for c, d in enumerate(week):
            x0 = c * cw
            in_month = d.month == month
            info = daily.get(d) if in_month else None
            face = CARD if in_month else BG
            edge = BORDER if in_month else "#161b22"
            if info:
                v = info["pnl"]
                week_pnl += v
                week_trades += info["trades"]
                inten = 0.35 + 0.65 * min(abs(v) / vmax, 1.0)
                if v > BEP_THRESHOLD:
                    face = _blend(CARD, GREEN, 0.18 + 0.32 * inten)
                    edge = GREEN
                elif v < -BEP_THRESHOLD:
                    face = _blend(CARD, RED, 0.18 + 0.32 * inten)
                    edge = RED
                else:
                    face = CARD_2
                    edge = MUTED
            box = FancyBboxPatch(
                (x0 + gap / 2, y0 + gap / 2), cw - gap, ch - gap,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=1.3 if info else 0.8
            )
            ax.add_patch(box)
            if d == today and in_month:
                ax.add_patch(FancyBboxPatch(
                    (x0 + gap / 2 - 0.02, y0 + gap / 2 - 0.02), cw - gap + 0.04, ch - gap + 0.04,
                    boxstyle="round,pad=0,rounding_size=0.07", facecolor="none", edgecolor=ACCENT, linewidth=2.2
                ))
            ax.text(x0 + 0.13, y0 + ch - 0.2, f"{d.day}", fontsize=10.5,
                    color=(TEXT if in_month else "#30363d"), fontweight="bold", va="center")
            if info:
                v = info["pnl"]
                ax.text(x0 + 0.5, y0 + 0.5, money(v), ha="center", va="center", fontsize=14.5,
                        fontweight="bold", color=(TEXT if abs(v) > BEP_THRESHOLD else MUTED))
                detail = f"{info['trades']} trade  \u2022  {info['wins']}W {info['losses']}L"
                if info["beps"]:
                    detail += f" {info['beps']}BE"
                ax.text(x0 + 0.5, y0 + 0.2, detail, ha="center", va="center", fontsize=8, color=TEXT, alpha=0.85)
            elif in_month and d.weekday() >= 5:
                ax.text(x0 + 0.5, y0 + 0.45, "weekend", ha="center", va="center", fontsize=8, color="#3b4252")

        # Weekly total cell
        x0 = 7 * cw
        has_week = week_trades > 0
        face = CARD_2
        edge = BORDER
        if has_week:
            edge = pnl_color(week_pnl) if abs(week_pnl) > BEP_THRESHOLD else MUTED
        ax.add_patch(FancyBboxPatch(
            (x0 + gap / 2, y0 + gap / 2), cw - gap, ch - gap,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor=face, edgecolor=edge, linewidth=1.3
        ))
        ax.text(x0 + 0.13, y0 + ch - 0.2, f"Minggu {r + 1}", fontsize=9, color=MUTED, fontweight="bold", va="center")
        if has_week:
            ax.text(x0 + 0.5, y0 + 0.5, money(week_pnl), ha="center", va="center", fontsize=14.5,
                    fontweight="bold", color=pnl_color(week_pnl))
            ax.text(x0 + 0.5, y0 + 0.2, f"{week_trades} trade", ha="center", va="center", fontsize=8, color=MUTED)
        else:
            ax.text(x0 + 0.5, y0 + 0.45, "-", ha="center", va="center", fontsize=12, color="#3b4252")

    return _fig_to_png(fig)


def _blend(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    m = [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in m)


# ---------------------------------------------------------------------------
# 3. Trade log
# ---------------------------------------------------------------------------
def render_trade_log(trades: List[JournalTrade], limit: int = 15) -> bytes:
    rows = sorted(
        trades,
        key=lambda t: (t.status == "OPEN", t.close_dt or t.entry_dt or datetime.min.replace(tzinfo=WIB)),
        reverse=True,
    )[:limit]
    n = max(len(rows), 1)
    fig = plt.figure(figsize=(16, 1.55 + 0.46 * (n + 1)), dpi=100, facecolor=BG)
    fig.text(0.03, 1 - 0.5 / fig.get_figheight(), f"Trade Log  \u2022  {len(rows)} transaksi terakhir",
             fontsize=19, fontweight="bold", color=TEXT, va="center")
    ax = fig.add_axes([0.03, 0.02, 0.94, 1 - 1.05 / fig.get_figheight()])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-n, 1)

    cols = [
        ("TIKET", 0.0, "left"), ("PAIR", 0.085, "left"), ("SIDE", 0.165, "left"), ("LOT", 0.215, "left"),
        ("ENTRY (WIB)", 0.26, "left"), ("EXIT (WIB)", 0.385, "left"), ("HARGA ENTRY", 0.51, "left"),
        ("HARGA EXIT", 0.6, "left"), ("DURASI", 0.69, "left"), ("SESI", 0.755, "left"),
        ("HASIL", 0.85, "left"), ("P&L", 1.0, "right"),
    ]
    ax.add_patch(FancyBboxPatch((-0.005, 0.08), 1.01, 0.84, boxstyle="round,pad=0,rounding_size=0.004",
                                facecolor=CARD_2, edgecolor=BORDER))
    for name, x, ha in cols:
        ax.text(x, 0.5, name, fontsize=8.8, color=MUTED, fontweight="bold", ha=ha, va="center")

    if not rows:
        ax.text(0.5, -0.5, "Belum ada transaksi", ha="center", va="center", color=MUTED, fontsize=11)
        return _fig_to_png(fig)

    badge_col = {"WIN": GREEN, "LOSS": RED, "BEP": MUTED, "OPEN": ACCENT}
    badge_txt = {"WIN": "WIN", "LOSS": "LOSS", "BEP": "BREAK-EVEN", "OPEN": "OPEN"}
    for i, t in enumerate(rows):
        y = -i - 0.5
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((-0.005, y - 0.42), 1.01, 0.84, boxstyle="round,pad=0,rounding_size=0.004",
                                        facecolor=CARD, edgecolor="none"))
        dec = t.price_decimals
        side_col = GREEN if t.action == "BUY" else RED
        vals = [
            (f"#{t.ticket}", TEXT), (t.symbol, TEXT), (t.action, side_col), (f"{t.lot:.2f}", TEXT),
            (t.entry_dt.strftime("%d/%m %H:%M") if t.entry_dt else "-", TEXT),
            (t.close_dt.strftime("%d/%m %H:%M") if t.close_dt else "berjalan", TEXT if t.close_dt else ACCENT),
            (f"{t.entry_price:.{dec}f}", TEXT),
            (f"{t.exit_price:.{dec}f}" if t.exit_price else "-", TEXT),
            (format_duration(t.duration_minutes) if t.close_dt else "-", MUTED),
            (t.session, MUTED),
        ]
        for (txt, colr), (_, x, ha) in zip(vals, cols[:10]):
            ax.text(x, y, txt, fontsize=9.3, color=colr, ha=ha, va="center",
                    fontweight="bold" if colr == side_col else "normal")
        bc = badge_col.get(t.outcome, MUTED)
        ax.add_patch(FancyBboxPatch((0.85, y - 0.24), 0.075, 0.48, boxstyle="round,pad=0,rounding_size=0.01",
                                    facecolor=_blend(CARD, bc, 0.25), edgecolor=bc, linewidth=1))
        ax.text(0.8875, y, badge_txt.get(t.outcome, t.outcome), fontsize=7.8, color=bc, ha="center",
                va="center", fontweight="bold")
        pnl_txt = money(t.pnl) + ("  (floating)" if t.status == "OPEN" else "")
        ax.text(1.0, y, pnl_txt, fontsize=10, color=pnl_color(t.pnl), ha="right", va="center", fontweight="bold")

    return _fig_to_png(fig)
