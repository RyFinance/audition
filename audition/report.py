"""Self-contained HTML report card plus a JSON dump of every check."""

from __future__ import annotations

import html
import io
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats  # noqa: E402
from .audit import grade, survivor_universe, top10  # noqa: E402
from .catalog import ADDRESSED, CATEGORIES, FAILED, LABEL, NOT, PARTIAL, TOP10_IDS  # noqa: E402

BG, PANEL, FG, MUTED, LINE = "#f2f2f2", "#f2f2f2", "#0a0a0a", "#6e6e6e", "#d6d6d6"
ACCENT, GREEN, AMBER, RED = "#0a0a0a", "#2e7d57", "#b5760f", "#c23b2a"
STATUS_COLOR = {ADDRESSED: GREEN, PARTIAL: AMBER, FAILED: RED, NOT: MUTED}

plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "text.color": FG,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": LINE, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9, "svg.fonttype": "none",
})


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _pct_axis(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fig_equity(a):
    rr = a.home
    r = rr.sim.returns
    bopen = getattr(rr, "bench_open", rr.bench)
    br = (bopen.shift(-1) / bopen - 1).reindex(r.index).fillna(0.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    ax1.plot(stats.equity(r), color=ACCENT, lw=1.4, label=a.strategy.name)
    ax1.plot(stats.equity(br), color=MUTED, lw=1.0, label=f"{rr.cfg['bench']} buy & hold")
    ax1.set_yscale("log")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylabel("growth of 1 (log)")
    dd = stats.drawdown(r)
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.35, lw=0)
    ax2.plot(dd, color=RED, lw=0.8)
    _pct_axis(ax2)
    ax2.set_ylabel("drawdown")
    return _svg(fig)


def fig_waterfall(steps):
    fig, ax = plt.subplots(figsize=(9, 2.9))
    labels = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    colors = [MUTED if l.startswith(("You reported", "Audition, same", "Today's members")) else ACCENT for l in labels]
    colors[-1] = GREEN if vals[-1] > 0.5 else AMBER if vals[-1] > 0 else RED
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:.2f}", va="center",
                ha="left" if v >= 0 else "right", color=FG, fontsize=9)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Sharpe ratio")
    ax.grid(axis="y", visible=False)
    lo = min(0, min(vals)) - 0.3
    hi = max(vals) + 0.6
    ax.set_xlim(lo, hi)
    return _svg(fig)


def fig_yearly(a):
    y = stats.yearly(a.home.sim.returns)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(y.index.astype(str), y.values, color=[GREEN if v >= 0 else RED for v in y.values], width=0.7)
    _pct_axis(ax)
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("Net return by year", loc="left", color=FG)
    ax.grid(axis="x", visible=False)
    return _svg(fig)


def fig_costs(curve):
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    xs = list(curve.keys())
    ys = list(curve.values())
    ax.plot(xs, ys, color=ACCENT, marker="o", lw=1.4)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(1, color=LINE, lw=0.8, ls="--")
    ax.set_xlabel("cost multiple (1 = Audition's estimate)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Cost stress", loc="left", color=FG)
    return _svg(fig)


def fig_capacity(ev):
    curve = ev["curve"]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    caps = [float(k) for k in curve]
    ax.plot(caps, list(curve.values()), color=ACCENT, marker="o", lw=1.4)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("capital ($)")
    ax.set_ylabel("Sharpe")
    ax.set_title("Capacity", loc="left", color=FG)
    return _svg(fig)


def fig_sensitivity(a):
    sens = a.extras["sens"]
    if not sens:
        return None
    base = a.home.summary["sharpe"]
    vals = sorted(s[2] for s in sens)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.hist(vals, bins=min(20, max(5, len(vals) // 2)), color=ACCENT, alpha=0.8)
    ax.axvline(base, color=AMBER, lw=1.5, label=f"chosen {base:.2f}")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False)
    ax.set_xlabel("Sharpe of nearby parameter sets")
    ax.set_title("Parameter neighbourhood", loc="left", color=FG)
    return _svg(fig)


def fig_placebo(a):
    pl = a.extras["placebo"]
    if not pl:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.hist(pl, bins=min(15, max(5, len(pl) // 2)), color=MUTED, alpha=0.9)
    ax.axvline(a.home.summary["sharpe"], color=ACCENT, lw=1.6, label="real")
    ax.legend(frameon=False)
    ax.set_xlabel("Sharpe on shuffled-calendar data")
    ax.set_title("Placebo test", loc="left", color=FG)
    return _svg(fig)


def fig_mc_hist(a):
    mcf = a.extras["mc_full"]
    fan = mcf["fan3"]
    final = fan[:, -1]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    h = mcf["horizons"][3]["block"]
    ax.bar(["5th", "25th", "50th", "75th", "95th"], h["cagr_p"], color=[RED, AMBER, FG, GREEN, GREEN], width=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    _pct_axis(ax)
    ax.set_title("3-year CAGR by percentile", loc="left", color=FG)
    ax.grid(axis="x", visible=False)
    del final
    return _svg(fig)


def fig_mc_dd(a):
    mcf = a.extras["mc_full"]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    labels = ["1y", "3y", "5y"]
    for k, (name, col) in enumerate((("block", FG), ("iid", MUTED))):
        vals = [mcf["horizons"][y][name]["mdd_p"][0] for y in (1, 3, 5)]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=f"{name} 5th pct")
    ax.set_xticks(range(3), labels)
    _pct_axis(ax)
    ax.legend(frameon=False)
    ax.set_title("Bad-case max drawdown", loc="left", color=FG)
    ax.grid(axis="x", visible=False)
    return _svg(fig)


def fig_mc(a):
    mc = a.extras["mc"]
    fan = mc["block"]["fan"]
    x = np.arange(fan.shape[1]) / a.home.cfg["ppy"]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.fill_between(x, fan[0], fan[4], color=ACCENT, alpha=0.15, lw=0)
    ax.fill_between(x, fan[1], fan[3], color=ACCENT, alpha=0.3, lw=0)
    ax.plot(x, fan[2], color=ACCENT, lw=1.4)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xlabel("years")
    ax.set_ylabel("growth of 1")
    ax.set_title("Block-bootstrap paths (5/25/50/75/95)", loc="left", color=FG)
    return _svg(fig)


def fig_events(rows):
    rows = [r for r in rows if np.isfinite(r["share_pnl"])][:7][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.42 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r["share_expo"] for r in rows], height=0.36, color=LINE, label="share of exposure")
    ax.barh(y - 0.19, [r["share_pnl"] for r in rows], height=0.36,
            color=[GREEN if r["share_pnl"] >= 0 else RED for r in rows], label="share of gross P&L")
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.axvline(0, color=MUTED, lw=0.8)
    _pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _svg(fig)


def fig_twins(rows):
    rows = rows[:10][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.36 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y, [r["corr"] for r in rows], height=0.6,
            color=[RED if abs(r["corr"]) >= 0.8 else AMBER if abs(r["corr"]) >= 0.5 else MUTED for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r["corr"] + (0.02 if r["corr"] >= 0 else -0.02), yi, f"{r['corr']:.2f}", va="center",
                ha="left" if r["corr"] >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlim(min(-0.1, min(r["corr"] for r in rows) - 0.12), 1.05)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("weekly correlation with your strategy")
    ax.grid(axis="y", visible=False)
    return _svg(fig)


REPORT_CSS = """
.mono,code,.k,td.id{font-family:var(--mono)}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;padding-bottom:28px;border-bottom:1px solid var(--line)}
.brand{font:500 12px/1 var(--sans);letter-spacing:.32em;text-transform:uppercase}
h1{font:400 64px/1 var(--serif);margin:18px 0 10px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.grade{text-align:right}
.grade .L{font:400 128px/.85 var(--serif)}
.grade .s{color:var(--muted);font-size:12px;margin-top:8px}
.verdict{margin:32px 0 8px;font:400 30px/1.2 var(--serif);max-width:860px}
h2{font:500 11px/1 var(--sans);letter-spacing:.24em;text-transform:uppercase;color:var(--fg);margin:64px 0 16px;
  display:flex;align-items:center;gap:12px}
h2::after{content:"";flex:1;height:1px;background:var(--line)}
.vt{font:400 64px/1 var(--serif);margin:14px 0 6px;letter-spacing:-.01em}
.panel{border-top:1px solid var(--line);padding:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid var(--line)}
.kpi{padding:16px 14px 18px 0;border-bottom:1px solid var(--line)}
.kpi .k{font:500 10px/1 var(--sans);color:var(--muted);text-transform:uppercase;letter-spacing:.14em}
.kpi .v{font:400 38px/1.05 var(--serif);margin-top:10px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));column-gap:28px}
.panel svg{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse}
th{font:500 10px/1 var(--sans);color:var(--muted);text-align:left;text-transform:uppercase;letter-spacing:.14em;padding:12px 10px 10px 0;border-bottom:1px solid var(--line)}
td{padding:14px 12px 14px 0;border-bottom:1px solid var(--line);vertical-align:top}
td.id{color:var(--muted);width:40px;font-size:12px}
td.t{width:230px;font-weight:500}
td.t .q{display:block;color:var(--muted);font-weight:400;font-size:12px;margin-top:3px}
td.st{width:130px}
.pill{display:inline-block;font:500 10px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;padding:5px 9px;border-radius:99px;border:1px solid currentColor;white-space:nowrap}
.addressed{color:var(--green)}.partial{color:var(--amber)}.failed{color:var(--red)}.not_addressed{color:var(--muted)}
.star{font-size:11px;margin-left:4px}
.scope{color:var(--muted);font-size:11px}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-bottom:12px;align-items:center}
pre{border-top:1px solid var(--line);padding:14px 0;overflow:auto;font:12px/1.5 var(--mono);color:var(--muted);margin:0}
.foot{color:var(--muted);font-size:12px;margin-top:48px}
.lead2{font:400 22px/1.3 var(--serif);max-width:860px;margin:0 0 18px}
.mode{display:inline-block;font:500 10px/1 var(--sans);letter-spacing:.2em;text-transform:uppercase;padding:6px 10px;
  border:1px solid currentColor;margin-left:10px;vertical-align:middle}
.note{color:var(--muted);font-size:13px;border-left:2px solid var(--line);padding:4px 0 4px 12px;margin:8px 0}
.seal{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));border-top:1px solid var(--line)}
.btn{font:500 12px/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;background:#000;color:#fff;border:0;padding:15px 20px;cursor:pointer}
.btn:disabled{opacity:.35;cursor:not-allowed}
@media (max-width:640px){header{flex-direction:column;align-items:flex-start}.grade{text-align:left}h1,.vt{font-size:44px}
  .verdict{font-size:24px}td.t,td.st{width:auto}}
"""


def _pill(status):
    return f'<span class="pill {status}">{LABEL[status]}</span>'


def _p(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _n(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _verdict(a, letter):
    c = a.checks
    if c[1].status == FAILED or c[2].status == FAILED:
        return "The strategy uses future information. None of the performance numbers can be trusted until that is fixed."
    net = a.home.summary["sharpe"]
    if net <= 0:
        gross = next((v for k, v in a.extras["waterfall"] if k.startswith("Next-open fill")), None)
        if gross is not None and gross > 0:
            return (f"It loses money after costs. Before costs the Sharpe is {gross:.2f}; fees, spread and impact "
                    f"take it to {net:.2f}. The edge is smaller than what it costs to trade.")
        return (f"It loses money even before costs (Sharpe {gross:.2f} with free trading, {net:.2f} after). "
                "There is no edge here to audit.")
    failed = [x for x in c.values() if x.status == FAILED]
    top_failed = [x for x in failed if x.id in TOP10_IDS]
    wf = a.extras["waterfall"]
    audited = wf[-1][1]
    head = next((v for k, v in wf if k.startswith("You reported")), None)
    lead = (f"Reported Sharpe {head:.2f}; audited {audited:.2f}. " if head is not None
            else f"Audited Sharpe {audited:.2f} after costs, autocorrelation and selection. ")
    ev3 = c[3].evidence
    if ev3.get("pit") and "bias_cagr" in ev3:
        lead += (f"Survivorship was worth {ev3['bias_cagr'] * 100:+.1f} CAGR points a year: today's members show "
                 f"{ev3['survivor_cagr'] * 100:.1f}%, the index as it really was {ev3['pit_cagr'] * 100:.1f}%. ")
    if top_failed:
        names = ", ".join(x.title.lower() for x in top_failed[:3])
        return lead + f"Fails on {names}."
    if survivor_universe(a):
        return lead + ("Every stock in the universe survived to today, so the grade is capped at C: a selection "
                       "edge on a hindsight list can be entirely survivorship. Re-run on point-in-time members.")
    if failed:
        names = ", ".join(x.title.lower() for x in failed[:3])
        return lead + f"No top-10 failures; weak on {names}."
    return lead + "No failed checks; the remaining gaps need forward trading or process evidence."


def _check_rows(checks):
    rows = []
    for c in checks:
        star = '<span class="star" title="top-10 priority">★</span>' if c.id in TOP10_IDS else ""
        scope = "" if c.scope == "backtest" else f'<span class="scope"> · {c.scope}</span>'
        rows.append(
            f'<tr id="c{c.id}"><td class="id">{c.id}</td>'
            f'<td class="t">{html.escape(c.title)}{star}<span class="q">{html.escape(c.question)}</span></td>'
            f'<td class="st">{_pill(c.status)}{scope}</td><td>{html.escape(c.finding)}</td></tr>')
    return "\n".join(rows)


TAPES = [  # body, edge, label
    ("#111111", "#000000", "#efe9dc"),
    ("#d9d4c7", "#b3ad9f", "#ffffff"),
    ("#2b2b2b", "#151515", "#f3d9b1"),
    ("#33445f", "#1f2b3e", "#efe9dc"),
    ("#6b6a3a", "#4c4b27", "#efe9dc"),
    ("#8b2f22", "#62200f", "#efe9dc"),
    ("#c9a227", "#9c7c14", "#111111"),
    ("#2f5d50", "#1d3d34", "#efe9dc"),
    ("#e8e3d6", "#c4bdab", "#111111"),
    ("rgba(40,40,40,.55)", "rgba(20,20,20,.5)", "#efe9dc"),
]


def _jump(j):
    return (f'<a class="mono" href="#evidence" onclick="setTimeout(function(){{document.getElementById(\'c{j}\')'
            f'.scrollIntoView()}},60)">#{j}</a>')


def write_report(a, out: Path):
    from . import ui

    out = Path(out)
    score, letter = grade(a)
    s = a.home.summary
    ev = {i: c.evidence for i, c in a.checks.items()}
    color = GREEN if letter in "AB" else AMBER if letter == "C" else RED
    counts = {k: sum(1 for c in a.checks.values() if c.status == k) for k in (ADDRESSED, PARTIAL, FAILED, NOT)}
    wf = a.extras["waterfall"]
    m = a.extras["manifest"]
    when = datetime.fromtimestamp(a.finished).strftime("%Y-%m-%d %H:%M")
    verdict = _verdict(a, letter)

    kpis = [
        ("CAGR", _p(s["cagr"])), ("Sharpe", _n(s["sharpe"])), ("Sortino", _n(s["sortino"])),
        ("Max drawdown", _p(s["max_dd"])), ("Calmar", _n(s["calmar"])), ("Volatility", _p(s["vol"])),
        ("Worst day", _p(s["worst_day"])), ("Worst month", _p(s["worst_month"])),
        ("ES 95% / day", _p(s["es95"], 2)), ("Years", _n(s["years"], 1)),
        ("Trades", str(a.checks[9].evidence.get("trades", "n/a"))),
        ("Cost / turnover", f"{a.home.sim.cost_bps_per_turnover:.1f} bp"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in kpis)

    t10_rows = top10(a)
    t10 = "".join(
        f'<tr><td class="id">{i + 1}</td><td class="t">{html.escape(label)}</td><td class="st">{_pill(st)}</td>'
        f'<td>{" ".join(html.escape(a.checks[j].finding) for j in ids if a.checks[j].status == st)[:600]}'
        f' <span class="scope">→ {", ".join(_jump(j) for j in ids)}</span></td></tr>'
        for i, (label, ids, st) in enumerate(t10_rows))
    t10_ok = sum(1 for _, _, st in t10_rows if st == ADDRESSED)
    t10_fail = sum(1 for _, _, st in t10_rows if st == FAILED)

    reps = a.extras["replication"]
    rep_rows = "".join(
        f'<tr><td class="t">{html.escape(r["name"])}<span class="q">{r["tickers"]} tickers · {html.escape(r["source"])} · from {r["start"]}</span></td>'
        + (f'<td colspan=5 class="failed">strategy error: {html.escape(r["error"])}</td>' if r["error"] else
           f'<td class="mono">{_n(r["sharpe"])}</td><td class="mono">{_p(r["cagr"])}</td>'
           f'<td class="mono">{_p(r["max_dd"])}</td><td class="mono">{_p(r["vol"])}</td>'
           f'<td class="mono">{r["cost_bps"]:.1f} bp</td>') + "</tr>"
        for r in reps)
    pos = sum(1 for r in reps if not r["error"] and r["sharpe"] > 0)

    charts = [fig_yearly(a), fig_costs(a.extras["cost_curve"]), fig_capacity(ev[28]),
              fig_sensitivity(a), fig_placebo(a), fig_mc(a)]
    chart_html = "".join(f'<div class="panel">{c}</div>' for c in charts if c)

    bench = a.home.cfg["bench"]
    metric_html = "".join(
        f'<h2>{html.escape(g)}</h2><div class="panel" style="padding:0"><table><tr><th>Metric</th><th>Strategy</th>'
        f'<th>{html.escape(bench)}</th></tr>'
        + "".join(f'<tr><td class="t">{html.escape(k)}</td><td class="mono">{v}</td><td class="mono" style="color:var(--muted)">{b}</td></tr>'
                  for k, v, b in rows) + "</table></div>"
        for g, rows in a.extras["metrics"])
    grid = a.extras["monthly"]
    yr = (1 + a.home.sim.returns).groupby(a.home.sim.returns.index.year).prod() - 1

    def cell(v):
        if v is None or not np.isfinite(v):
            return '<td></td>'
        alpha = min(abs(v) / 0.1, 1) * 0.55
        bg = f"rgba(46,125,87,{alpha:.2f})" if v >= 0 else f"rgba(194,59,42,{alpha:.2f})"
        return f'<td class="mono" style="background:{bg};text-align:right;padding:8px 8px">{v * 100:.1f}</td>'
    months = "JFMAMJJASOND"
    grid_html = ('<div class="panel" style="padding:0;overflow:auto"><table><tr><th>Year</th>'
                 + "".join(f'<th style="text-align:right">{m}</th>' for m in months) + '<th style="text-align:right">Year</th></tr>'
                 + "".join(f'<tr><td class="mono">{y}</td>' + "".join(cell(grid.loc[y].get(mm)) for mm in range(1, 13))
                           + cell(float(yr.get(y, float("nan")))) + "</tr>" for y in grid.index)
                 + "</table></div>")

    mcf = a.extras["mc_full"]

    def mc_rows(name):
        out = []
        for yrs in (1, 3, 5):
            h = mcf["horizons"][yrs][name]
            out.append(f'<tr><td class="t">{yrs} year{"s" if yrs > 1 else ""}</td>'
                       + "".join(f'<td class="mono">{_p(v)}</td>' for v in (h["cagr_p"][0], h["cagr_p"][2], h["cagr_p"][4]))
                       + "".join(f'<td class="mono">{_p(v)}</td>' for v in (h["mdd_p"][0], h["mdd_p"][2]))
                       + f'<td class="mono">{_p(h["p_loss"])}</td><td class="mono">{_p(h["p_dd25"])}</td>'
                       f'<td class="mono">{_p(h["p_dd50"])}</td></tr>')
        return "".join(out)
    mc_head = ('<tr><th>Horizon</th><th>CAGR 5th</th><th>CAGR median</th><th>CAGR 95th</th><th>Max DD 5th</th>'
               '<th>Max DD median</th><th>P(loss)</th><th>P(DD ≥ 25%)</th><th>P(DD ≥ 50%)</th></tr>')
    st2 = mcf["stress_2x"]
    pm = mcf["param"]
    extra_rows = (f'<tr><td class="t">3 years, 2x costs</td><td class="mono">{_p(st2["cagr_p"][0])}</td>'
                  f'<td class="mono">{_p(st2["cagr_p"][2])}</td><td class="mono">{_p(st2["cagr_p"][4])}</td>'
                  f'<td class="mono">{_p(st2["mdd_p"][0])}</td><td class="mono">{_p(st2["mdd_p"][2])}</td>'
                  f'<td class="mono">{_p(st2["p_loss"])}</td><td class="mono">{_p(st2["p_dd25"])}</td><td class="mono">{_p(st2["p_dd50"])}</td></tr>')
    if pm:
        extra_rows += (f'<tr><td class="t">3 years, parameters uncertain</td><td class="mono">{_p(pm["cagr_p"][0])}</td>'
                       f'<td class="mono">{_p(pm["cagr_p"][2])}</td><td class="mono">{_p(pm["cagr_p"][4])}</td>'
                       f'<td class="mono">{_p(pm["mdd_p"][0])}</td><td class="mono">{_p(pm["mdd_p"][2])}</td>'
                       f'<td class="mono">{_p(pm["p_loss"])}</td><td></td><td class="mono">{_p(pm["p_dd50"])}</td></tr>')
    mc_html = (f'<p class="verdict">{mcf["paths"]:,} simulated paths per horizon. Stationary block bootstrap '
               '(20-day mean blocks) keeps volatility clustering; the iid rows shuffle days independently and usually look safer than they should.</p>'
               f'<h2>Block bootstrap</h2><div class="panel" style="padding:0;overflow:auto"><table>{mc_head}{mc_rows("block")}{extra_rows}</table></div>'
               f'<h2>Independent shuffle, for comparison</h2><div class="panel" style="padding:0;overflow:auto"><table>{mc_head}{mc_rows("iid")}</table></div>'
               f'<div class="charts" style="margin-top:24px"><div class="panel">{fig_mc(a)}</div><div class="panel">{fig_mc_hist(a)}</div>'
               f'<div class="panel">{fig_mc_dd(a)}</div><div class="panel">{fig_costs(a.extras["cost_curve"])}</div></div>')

    sections = []
    for code, title in CATEGORIES:
        cs = [c for c in sorted(a.checks.values(), key=lambda c: c.id) if c.category == code]
        cc = {k: sum(1 for c in cs if c.status == k) for k in (ADDRESSED, PARTIAL, FAILED, NOT)}
        summ = " · ".join(f'<span class="{k}">{v} {LABEL[k].lower()}</span>' for k, v in cc.items() if v)
        sections.append(f'<h2>{html.escape(title)}</h2><div class="sub" style="margin:-6px 0 10px">{summ}</div>'
                        f'<div class="panel" style="padding:0"><table><tr><th>#</th><th>Concern</th><th>Status</th>'
                        f'<th>What Audition found</th></tr>{_check_rows(cs)}</table></div>')

    head = (f'<header><div><h1>{html.escape(a.strategy.name)}</h1>'
            f'<div class="sub mono">{html.escape(a.strategy.path.name)} · {html.escape(json.dumps(a.strategy.params))} · '
            f'{a.home.cfg["name"]} · {m["data_start"]} → {m["data_end"]} · ${a.options.capital:,.0f}</div></div>'
            f'<div class="grade"><div class="L" style="color:{color}">{letter}</div>'
            f'<div class="s">{score:.0%} of testable checks</div></div></header>')
    legend = (f'<div class="legend" style="margin-top:24px"><span>{_pill(ADDRESSED)} measured and holds</span>'
              f'<span>{_pill(PARTIAL)} measured with caveats, or self-declared</span><span>{_pill(FAILED)} measured and fails</span>'
              f'<span>{_pill(NOT)} not tested</span><span><span class="star">★</span> top-10 priority</span></div>')

    probes = _probe_views(a)
    vault_ex = a.extras.get("vault", {})
    mode = vault_ex.get("mode", "open")
    if mode == "rehearsal":
        head = head.replace("</h1>", f'<span class="mode" style="color:var(--amber)">rehearsal · data to {html.escape(a.options.end or "")}</span></h1>', 1)
    elif mode == "final":
        fin = vault_ex.get("final", {})
        head = head.replace("</h1>", f'<span class="mode" style="color:{GREEN if fin.get("passed") else RED}">final audition</span></h1>', 1)

    def sub(title, body):
        """A probe body as a section of a combined page: its lead sentence drops to body size."""
        return f'<h2>{title}</h2>' + body.replace('<p class="verdict">', '<p class="lead2">', 1)

    manifest_html = (f'<pre>{html.escape(json.dumps(m, indent=2, default=str))}</pre>'
                     f'<p class="foot">Generated {when} in {a.finished - a.started:.0f}s by Audition 0.2. Prices from Yahoo Finance; '
                     "T-bill rate from ^IRX. A backtest audit can't prove an edge, only find reasons to distrust one. Not investment advice.</p>")
    markets_html = (f'<p class="lead2">Same file, same parameters, other markets. Positive Sharpe in {pos} of {len(reps)}.</p>'
                    '<div class="panel" style="padding:0"><table><tr><th>Market</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th>'
                    f'<th>Vol</th><th>Cost/turn</th></tr>{rep_rows}</table></div>')
    views = [
        ("verdict", "Verdict",
         f'{head}<p class="verdict">{html.escape(verdict)}</p>'
         f'<h2>From headline to audited Sharpe</h2><div class="panel">{fig_waterfall(wf)}</div>'
         f'<h2>The 10 that matter · {t10_ok} addressed, {t10_fail} failed</h2>'
         f'<div class="panel" style="padding:0"><table><tr><th>#</th><th>Priority</th><th>Status</th><th>Finding</th></tr>{t10}</table></div>'),
        ("returns", "Returns",
         '<p class="verdict">Net of fees, spread, tax, impact and financing, with idle cash in T-bills.</p>'
         f'<div class="grid">{kpi_html}</div><div class="panel" style="margin-top:24px">{fig_equity(a)}</div>'
         f'<div class="charts" style="margin-top:24px">{chart_html}</div>'
         f'<h2>Monthly returns (%)</h2>{grid_html}{metric_html}'
         + sub("What could happen next · Monte Carlo", mc_html)),
        ("source", "Where the edge comes from",
         sub("Look-alikes · is it a known factor or ETF?", probes["twins"][0])
         + sub("Events · did scheduled news drive it?", probes["events"][0])),
        ("oos", "Out of sample",
         sub("Holdout · untouched years", probes["vault"][0]) + sub("Other markets", markets_html)),
        ("evidence", f"All {len(a.checks)} checks", legend + "".join(sections) + "<h2>Reproduce this run</h2>" + manifest_html),
    ]

    def view(vid, title, body):
        return (f'<main class="view" id="{vid}"><div class="wrap"><div class="brand">Audition · '
                f'{html.escape(a.strategy.name)}</div><h1 class="vt">{title}</h1>{body}</div></main>')

    name = html.escape(a.strategy.name)
    tw_it, ev_it, vt_it = probes["twins"][1], probes["events"][1], probes["vault"][1]
    items = [
        dict(view="verdict", head="Verdict", title="Verdict", num=letter, numLabel=f"grade · {score:.0%}",
             sub=f"grade {letter}", meta=[("Audited Sharpe", _n(wf[-1][1])), ("Top 10 addressed", f"{t10_ok} of 10"),
                                         ("Failed checks", str(counts[FAILED])), ("Market", a.home.cfg["name"])]),
        dict(view="returns", head="Returns", title="Returns", num=_n(s["sharpe"]), numLabel="net sharpe",
             sub=f"CAGR {_p(s['cagr'])}", meta=[("CAGR", _p(s["cagr"])), ("Max drawdown", _p(s["max_dd"])),
                                                ("Median 3y CAGR (MC)", _p(mcf["horizons"][3]["block"]["cagr_p"][2])),
                                                ("3y odds of −50%", _p(mcf["horizons"][3]["block"]["p_dd50"]))]),
        dict(view="source", head="Where the edge comes from", title="Edge source", num=tw_it["num"],
             numLabel=tw_it["numLabel"], sub="look-alikes · events",
             meta=tw_it["meta"][:2] + ev_it["meta"][:2]),
        dict(view="oos", head="Out of sample", title="Out of sample", num=vt_it["num"], numLabel=vt_it["numLabel"],
             sub=f"holdout · {pos}/{len(reps)} markets",
             meta=vt_it["meta"][:2] + [("Markets positive", f"{pos} of {len(reps)}")]
             + [(html.escape(r["name"]), "error" if r["error"] else _n(r["sharpe"])) for r in reps[1:2]]),
        dict(view="evidence", head=f"All {len(a.checks)} checks", title="Evidence", num=str(counts[ADDRESSED]),
             numLabel=f"of {len(a.checks)} addressed", sub="checks · reproduce",
             meta=[(LABEL[k], str(v)) for k, v in counts.items()]),
    ]
    for it, (body, edge, label) in zip(items, TAPES):
        it.update(body=body, edge=edge, label=label)
    objects = [ui.tape_html(it, i) for i, it in enumerate(items)]
    views_html = "".join(view(v, t, b) for v, t, b in views)
    dock = ui.dock([
        ("Read in order", [("verdict", "1 · Verdict", "The grade, why, and the 10 checks that matter"),
                           ("returns", "2 · Returns", "Performance now, and Monte Carlo futures"),
                           ("source", "3 · Where the edge comes from", "Known look-alikes and scheduled events"),
                           ("oos", "4 · Out of sample", "Untouched years and other markets"),
                           ("evidence", "5 · Evidence", f"All {len(a.checks)} checks, and how to reproduce"),
                           ("", "Tape picker", "Browse the five sections as cassettes"),
                           ("!tour", "Tutorial", "A 30-second tour of this page")]),
    ], "Tapes", home_href="index.html", badge=f'<span style="color:{color}">{letter}</span>')
    tour = [
        dict(title="This is an audit", text="Five sections, meant to be read in order: the verdict, the returns, where the edge "
             "comes from, how it does out of sample, and the evidence behind every check."),
        dict(target="#sel-num", title="The grade", text="A to F. The worst problems cap it, so a strategy can't hide one fatal flaw "
             "behind seventy passes."),
        dict(target="#holder", title="Each tape is a section", text="Scroll to change tapes, click to play one."),
        dict(target=".dk-mid", title="Step through it", text="‹ and › (or the arrow keys) move through the sections in order. "
             "The middle opens the full list."),
        dict(target=".dk-home", title="Back to the shelf", text="Every strategy you've audited lives there. Restart this tour from "
             "the menu any time."),
    ]
    doc = ui.page(f"Audition · {a.strategy.name}", items, objects, views_html, dock,
                  "Scroll to choose a tape · click or press Enter to play it", extra_css=REPORT_CSS,
                  tour=tour, tour_key="aud-tour-report")
    out.write_text(doc)

    payload = dict(
        strategy=a.strategy.name, file=a.strategy.path.name, report=out.name, grade=letter, score=score,
        verdict=verdict, generated=when, counts=counts,
        summary={k: v for k, v in s.items() if k != "rho"},
        checks={i: dict(title=c.title, category=c.category, scope=c.scope, status=c.status,
                        finding=c.finding, evidence=c.evidence) for i, c in sorted(a.checks.items())},
        top10=[dict(label=l, checks=ids, status=st) for l, ids, st in t10_rows],
        waterfall=wf, replication=reps, manifest=m, mode=mode,
        vault={k: v for k, v in vault_ex.items() if k != "state"},
        events={k: v for k, v in a.extras.get("events", {}).items() if k != "rows"},
        twins={k: v for k, v in a.extras.get("twins", {}).items() if k in ("verdict", "mix", "error")},
        daily_returns=[float(x) for x in a.home.sim.returns.to_numpy()],
    )
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, default=_json_default))


def _probe_views(a):
    """(body html, selector item fields) for the events, twins and vault tapes."""
    out = {}
    ev = a.extras.get("events", {})
    if ev.get("error") or not ev.get("rows"):
        out["events"] = (f'<p class="verdict">Event map unavailable: {html.escape(str(ev.get("error", "no data")))}</p>',
                         dict(num="n/a", numLabel="event map", sub="unavailable", meta=[("Status", "unavailable")]))
    else:
        rows = ev["rows"]
        trs = "".join(
            f'<tr><td class="t">{html.escape(r["label"])}<span class="q">{r["scope"]}</span></td>'
            f'<td class="mono">{r["events"]:,}</td><td class="mono">{_p(r["share_expo"])}</td>'
            f'<td class="mono">{_p(r["share_pnl"])}</td><td class="mono">{_n(r["sharpe_without"])}</td></tr>' for r in rows)
        body = (f'<p class="verdict">{html.escape(ev["verdict"])}</p>'
                f'<div class="panel">{fig_events(rows)}</div>'
                '<h2>Every event type</h2><div class="panel" style="padding:0"><table><tr><th>Event</th><th>Hits</th>'
                '<th>Share of exposure</th><th>Share of gross P&amp;L</th><th>Sharpe without it</th></tr>'
                f'{trs}</table></div>'
                f'<p class="sub" style="margin-top:14px">Sharpe with every event period removed: {_n(ev["clean_sharpe"])} '
                f'(all periods: {_n(ev["base_sharpe"])}). '
                + " · ".join(html.escape(n) for n in ev.get("notes", [])) + '</p>'
                '<p class="foot">FOMC dates from federalreserve.gov, CPI release dates from ALFRED (CPIAUCSL vintages), '
                'earnings dates from Yahoo, ex-dividend days from the gap between raw and adjusted closes. '
                'Each event is assigned to the open-to-open holding period that absorbs it.</p>')
        top = next((r for r in rows if np.isfinite(r["share_pnl"])), rows[0])
        out["events"] = (body, dict(num=_p(top["share_pnl"], 0), numLabel=f"of P&L from {top['label'].lower()}s",
                                    sub=f"{len(rows)} event types",
                                    meta=[(html.escape(r["label"]), _p(r["share_pnl"])) for r in rows[:4]]))
    tw = a.extras.get("twins", {})
    if tw.get("error") or not tw.get("rows"):
        out["twins"] = (f'<p class="verdict">Twin finder unavailable: {html.escape(str(tw.get("error", "no data")))}</p>',
                        dict(num="n/a", numLabel="twin finder", sub="unavailable", meta=[("Status", "unavailable")]))
    else:
        rows = tw["rows"]
        trs = "".join(
            f'<tr><td class="t">{html.escape(r["name"])}<span class="q">{html.escape(r["family"])} · {html.escape(r["kind"])}</span></td>'
            f'<td class="mono">{r["corr"]:.2f}</td><td class="mono">{_n(r["beta"])}</td>'
            f'<td class="mono">{_p(r["same_sign"], 0)}</td><td class="mono">{_n(r["twin_sharpe"])}</td>'
            f'<td class="mono">{r["weeks"] / 52:.1f}y</td></tr>' for r in rows)
        mix = tw.get("mix")
        mix_html = ""
        if mix:
            mix_html = ('<h2>The closest mix</h2><div class="grid">'
                        + "".join(f'<div class="kpi"><div class="k">{html.escape(t["name"])}</div><div class="v">{t["weight"]:+.2f}</div></div>'
                                  for t in mix["twins"])
                        + f'<div class="kpi"><div class="k">R² explained</div><div class="v">{mix["r2"]:.0%}</div></div>'
                        f'<div class="kpi"><div class="k">Alpha left / yr</div><div class="v">{mix["alpha_ann"]:+.1%}</div></div>'
                        f'<div class="kpi"><div class="k">Alpha t-stat</div><div class="v">{mix["alpha_t"]:.1f}</div></div></div>')
        body = (f'<p class="verdict">{html.escape(tw["verdict"])}</p><div class="panel">{fig_twins(rows)}</div>{mix_html}'
                '<h2>Closest twins</h2><div class="panel" style="padding:0"><table><tr><th>Twin</th><th>Corr</th><th>Beta</th>'
                f'<th>Same direction</th><th>Twin Sharpe</th><th>Overlap</th></tr>{trs}</table></div>'
                f'<p class="foot">Compared with {tw.get("library_size", 0)} twins: Fama-French factors and reversal factors, '
                'buy-and-hold ETFs (markets, bonds, commodities, style ETFs, bitcoin) and rule-based classics built from them '
                '(trend, time-series momentum, risk parity, 60/40, sector momentum, overnight drift, turn of the month, '
                'sell in May). Weekly excess returns; the mix is a forward-stepwise regression of up to three twins with '
                'Newey-West t-stats.</p>')
        t1 = rows[0]
        out["twins"] = (body, dict(num=f"{t1['corr']:.2f}", numLabel=f"correlation with {t1['name']}",
                                   sub=f"closest: {t1['key']}",
                                   meta=[("Closest", html.escape(t1["name"])),
                                         ("Mix explains", f"{mix['r2']:.0%}" if mix else "n/a"),
                                         ("Alpha left", f"{mix['alpha_ann']:+.1%}" if mix else "n/a"),
                                         ("Library", f"{tw.get('library_size', 0)} twins")]))
    vx = a.extras.get("vault", {})
    st = vx.get("state", {}) or {}
    mode = vx.get("mode", "open")
    notes = "".join(f'<div class="note">{html.escape(n)}</div>' for n in vx.get("notes", []))
    if st.get("state", "none") == "none":
        body = ('<p class="verdict">This run saw the full history. Nothing is locked away yet, so none of it is out of sample.</p>'
                '<p class="sub" style="max-width:720px">Rehearse instead: Audition locks the last two years the first time, every '
                'later rehearsal stops before them, and each new version you try is counted. When you\'re done, take the '
                'final audition once on the locked years. Retakes and full-history peeks are recorded and deflate the result.</p>'
                f'<pre>audition run {html.escape(a.strategy.path.name)} --rehearsal\naudition final {html.escape(a.strategy.path.name)}</pre>'
                + notes)
        item = dict(num="open", numLabel="no vault yet", sub="full history",
                    meta=[("Mode", "open (full history)"), ("Locked", "nothing"), ("Next", "rehearse"), ("Final", "not taken")])
    else:
        fin = vx.get("final")
        finals = st.get("finals", [])
        cards = [("Locked since", st["sealed_from"]), ("Rehearsals", str(len(st.get("rehearsals", [])))),
                 ("Versions tried", str(len(st.get("versions", [])))), ("Peeks", str(len(st.get("peeks", [])))),
                 ("Finals taken", str(len(finals)))]
        grid = '<div class="seal">' + "".join(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards) + "</div>"
        if mode == "final" and fin:
            ok = fin.get("passed")
            lead = (f'Final audition {"passed" if ok else "failed"}: on the locked period ({fin["sealed_from"]} on, '
                    f'{fin["periods"]} days) Sharpe {_n(fin["sharpe_locked"])} against {_n(fin["sharpe_before"])} before it, '
                    f'CAGR {_p(fin["cagr_locked"])}, max drawdown {_p(fin["max_dd_locked"])}.')
            lead += (" Consistent with the rehearsals." if fin["z"] > -2 else
                     f" That's {abs(fin['z']):.1f} standard errors below the rehearsal Sharpe: the rehearsals overfit.")
            if len(finals) > 1:
                lead += f" This is retake #{len(finals)}; the locked years are no longer unseen."
            item = dict(num=_n(fin["sharpe_locked"]), numLabel="sharpe on the locked years", sub="final, sealed",
                        meta=[("Locked Sharpe", _n(fin["sharpe_locked"])), ("Before", _n(fin["sharpe_before"])),
                              ("Locked CAGR", _p(fin["cagr_locked"])), ("Finals taken", str(len(finals)))])
        else:
            lead = (f"Rehearsal. Data stops at {html.escape(a.options.end or '')}; {st['sealed_from']} onward is sealed. "
                    + (f"Final audition already taken {len(finals)} time(s)." if finals else
                       "The final audition hasn't been taken."))
            item = dict(num=str(len(st.get("rehearsals", []))), numLabel="rehearsals so far", sub="rehearsing" if not finals else "final taken",
                        meta=[("Locked since", st["sealed_from"]), ("Versions", str(len(st.get("versions", [])))),
                              ("Peeks", str(len(st.get("peeks", [])))), ("Final", "taken" if finals else "not yet")])
        hist = "".join(f'<tr><td class="mono">{f["time"]}</td><td class="mono">{f["sha"]}</td><td class="mono">{_n(f.get("sharpe_locked"))}</td>'
                       f'<td class="mono">{_p(f.get("cagr_locked"))}</td><td>{"passed" if f.get("passed") else "failed"}</td></tr>' for f in finals)
        btn = ("" if finals and mode == "final" else
               f'<div style="margin-top:28px"><button class="btn" id="finalbtn" data-file="{html.escape(str(a.strategy.path))}">'
               f'{"Retake" if finals else "Take"} the final audition</button> '
               f'<span class="sub" id="finalmsg">Runs once on the locked years and seals the result.</span></div>'
               '<script>(function(){var b=document.getElementById("finalbtn");if(!b)return;'
               'if(location.protocol==="file:"){b.disabled=true;document.getElementById("finalmsg").textContent='
               '"Open this report through audition serve to take the final, or run: audition final <file>";return;}'
               'b.onclick=function(){if(!confirm("Take the final audition now? The locked years are revealed and the result is sealed.'
               + (' This is a retake and will be recorded as one.' if finals else '') + '"))return;b.disabled=true;'
               'var m=document.getElementById("finalmsg");m.textContent="Running…";'
               'fetch("/api/final",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({file:b.dataset.file})})'
               '.then(function(r){return r.json()}).then(function(j){if(j.error){m.textContent=j.error;b.disabled=false;return;}'
               '(function poll(){fetch("/api/job/"+j.job).then(function(r){return r.json()}).then(function(s){'
               'm.textContent=(s.log||[]).slice(-1)[0]||"Running…";if(s.status==="running")return setTimeout(poll,1500);'
               'if(s.status==="done")location.href=s.report+"#oos";else{m.textContent="Failed: "+(s.log||[]).slice(-3).join(" ");b.disabled=false;}})})();});};})();</script>')
        body = (f'<p class="verdict">{lead}</p>{grid}{notes}'
                + (f'<h2>Final auditions</h2><div class="panel" style="padding:0"><table><tr><th>Sealed</th><th>Version</th>'
                   f'<th>Locked Sharpe</th><th>Locked CAGR</th><th>Result</th></tr>{hist}</table></div>' if finals else "")
                + btn)
    out["vault"] = (body, item)
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)
