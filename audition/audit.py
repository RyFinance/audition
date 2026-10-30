"""Runs a strategy through every check and returns a structured result for the report."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as data_mod
from . import engine, montecarlo, stats
from . import strategy as strat_mod
from .catalog import ADDRESSED, CHECKS, FAILED, NOT, PARTIAL, RANK, TOP10, TOP10_IDS, worst
from . import universes
from .regions import REGIONS, infer_region


@dataclass
class Check:
    id: int
    status: str = NOT
    finding: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def category(self):
        return CHECKS[self.id][0]

    @property
    def title(self):
        return CHECKS[self.id][1]

    @property
    def question(self):
        return CHECKS[self.id][2]

    @property
    def scope(self):
        return CHECKS[self.id][3]


@dataclass
class RegionRun:
    code: str
    cfg: dict
    universe_source: str
    bars: data_mod.Bars
    bench: pd.Series          # benchmark close
    rf: pd.Series
    cm: engine.CostModel
    w: pd.DataFrame
    sim: engine.SimResult
    summary: dict
    error: str | None = None


@dataclass
class Options:
    regions: list
    capital: float = 1_000_000.0
    start: str | None = None
    quick: bool = False
    log: object = print
    end: str | None = None          # last date the audit may see (rehearsals stop at the vault seal)
    mode: str = "open"              # "open" (full history), "rehearsal" or "final"


@dataclass
class Audit:
    strategy: strat_mod.Strategy
    options: Options
    home: RegionRun
    regions: dict
    checks: dict
    extras: dict
    started: float
    finished: float = 0.0


# --------------------------------------------------------------------------- setup

def _region_plan(strat, opts):
    """[(code, cfg, universe, source)] with the home region first."""
    plan = []
    uni = strat.universe
    if uni == "sp500" or (isinstance(uni, (list, tuple)) and universes.looks_like_sp500(uni)):
        # point-in-time S&P 500 is the honest home; the static list becomes the survivor arm
        cfg = dict(REGIONS["us"], name="S&P 500, point-in-time members")
        plan.append(("sp500", cfg, "sp500", "point-in-time S&P 500 (members on each date)"))
        surv = list(uni) if isinstance(uni, (list, tuple)) else None
        plan.append(("survivors", dict(REGIONS["us"], name="Today's S&P 500 members only"), surv,
                      "survivor arm: today's members for the whole history"))
        for code in opts.regions:
            plan.append((code, REGIONS[code], REGIONS[code]["universe"], "Audition default large caps"))
    elif isinstance(uni, str) and uni.startswith("byo:"):
        from .sources import byo_symbols
        syms = byo_symbols(uni[4:])
        base = infer_region(syms)
        cfg = dict(REGIONS[base], name=f"Your data: {uni[4:]} ({REGIONS[base]['name']} costs)")
        plan.append(("home", cfg, syms, f"your uploaded dataset {uni[4:]}"))
        for code in opts.regions:
            plan.append((code, REGIONS[code], REGIONS[code]["universe"], "Audition default large caps"))
    elif isinstance(uni, (list, tuple)):
        base = infer_region(uni)
        cfg = dict(REGIONS[base], name=f"Your universe ({REGIONS[base]['name']} costs)")
        plan.append(("home", cfg, list(uni), "your UNIVERSE"))
        for code in opts.regions:
            plan.append((code, REGIONS[code], REGIONS[code]["universe"], "Audition default large caps"))
    else:
        uni = uni or {}
        codes = list(uni.keys()) + [c for c in opts.regions if c not in uni]
        for code in codes:
            if code not in REGIONS:
                raise ValueError(f"unknown region {code!r}; choose from {sorted(REGIONS)}")
            if code in uni:
                plan.append((code, REGIONS[code], list(uni[code]), "your UNIVERSE"))
            else:
                plan.append((code, REGIONS[code], REGIONS[code]["universe"], "Audition default large caps"))
    return plan


def _prepare_region(strat, code, cfg, universe, source, opts) -> RegionRun:
    start = opts.start or cfg["start"]
    pit = None
    if universe == "sp500":
        pit = universes.load_sp500(start)
        bars = pit.bars
    else:
        if universe is None:  # survivor arm for UNIVERSE = "sp500": today's members
            universe = sorted(universes.sp500().current())
        own = source.startswith("your") or source.startswith("survivor")
        bars = data_mod.load(universe, start, source=strat.data_source if own else "yahoo")
    if opts.end:
        bars = bars.until(opts.end)
        if pit is not None:
            pit.bars = bars
    bench_bars = data_mod.load([cfg["bench"]], start)
    bench = bench_bars.close.iloc[:, 0].reindex(bars.index).ffill()
    bench_open = bench_bars.open.iloc[:, 0].reindex(bars.index).ffill()
    rf = data_mod.risk_free(bars.index, cfg["ppy"])
    fx = data_mod.fx_rate(cfg["fx"], bars.index) if cfg.get("fx") else None
    cm = engine.cost_model(bars, cfg, opts.capital, fx, overrides=strat.declare.get("costs"))
    try:
        w = strat_mod.run(strat, bars, bench, code)
    except Exception as e:  # report, don't crash the whole audit
        empty = pd.DataFrame(0.0, index=bars.index, columns=bars.tickers)
        sim = engine.simulate(empty, bars, rf, cm, cfg["ppy"])
        return RegionRun(code, cfg, source, bars, bench, rf, cm, empty, sim,
                         stats.summary(sim.returns, sim.rf, cfg["ppy"]), error=f"{type(e).__name__}: {e}")
    active = w.abs().sum(axis=1) > 1e-12
    eval_start = active.idxmax() if active.any() else None
    sim = engine.simulate(w, bars, rf, cm, cfg["ppy"], eval_start=eval_start)
    run = RegionRun(code, cfg, source, bars, bench, rf, cm, w, sim, stats.summary(sim.returns, sim.rf, cfg["ppy"]))
    run.eval_start = eval_start
    run.bench_open = bench_open
    run.pit = pit
    run.ineligible = dict(w.attrs)
    return run


def _sim(rr: RegionRun, w=None, cm=None, **kw):
    return engine.simulate(rr.w if w is None else w, rr.bars, rr.rf, cm or rr.cm, rr.cfg["ppy"],
                           eval_start=getattr(rr, "eval_start", None), **kw)


def _sharpe(sim, ppy):
    return stats.sharpe(sim.returns - sim.rf, ppy)


def _try_run(strat, rr, bars=None, params=None, bench=None):
    try:
        return strat_mod.run(strat, bars or rr.bars, rr.bench if bench is None else bench, rr.code, params)
    except Exception:
        return None


def _fmt_pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _fmt(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# --------------------------------------------------------------------------- research

def check_lookahead(strat, rr, k):
    n = len(rr.bars.index)
    cuts = sorted(set(np.linspace(int(n * 0.3), n - 2, k).astype(int)))
    mismatches, errors = [], []
    for pos in cuts:
        try:
            wt = strat_mod.run(strat, rr.bars.slice(pos), rr.bench.iloc[:pos + 1], rr.code)
        except Exception as e:
            errors.append(f"{rr.bars.index[pos].date()}: {type(e).__name__}")
            continue
        # a rule that only uses the past gives the same weight on every day up to the cut,
        # not just on the cut day itself (centred windows and full-sample fits differ earlier)
        full = rr.w.iloc[:pos + 1].reindex(columns=wt.columns).fillna(0.0)
        tr = wt.reindex(full.index).fillna(0.0)
        diff = (tr - full).abs()
        tol = 1e-8 * max(1.0, float(full.abs().max().max()))
        bad = diff.max(axis=1) > tol
        if bad.any():
            day = bad[bad].index[-1]
            t = diff.loc[day].idxmax()
            mismatches.append(dict(date=str(day.date()), cut=str(rr.bars.index[pos].date()), ticker=t,
                                   truncated=float(tr.at[day, t]), full=float(full.at[day, t]), days=int(bad.sum())))
    c = Check(1, evidence=dict(cuts=len(cuts), mismatches=mismatches, errors=errors))
    if mismatches:
        m = mismatches[0]
        c.status = FAILED
        c.finding = (f"Weights change when future data is removed. On {m['date']} the {m['ticker']} weight is "
                     f"{m['full']:.3f} with the full history but {m['truncated']:.3f} when history stops at {m['cut']} "
                     f"({len(mismatches)} of {len(cuts)} cut dates show it). The strategy is reading the future.")
    elif errors and len(errors) == len(cuts):
        c.status = PARTIAL
        c.finding = "The strategy failed on truncated history, so the lookahead test could not run."
    else:
        c.status = ADDRESSED
        c.finding = (f"Re-ran the strategy on history cut off at {len(cuts)} dates; every decision matched the "
                     "full-history run. No decision depends on later data.")
    return c


def check_leakage(strat, rr, seed=3):
    rng = np.random.default_rng(seed)
    n = len(rr.bars.index)
    bad = []
    for frac in (0.5, 0.75):
        pos = int(n * frac)
        # independent noise per asset, so cross-sectional rankings after the cut change too
        k = len(rr.bars.tickers)
        noise = np.exp(np.cumsum(rng.normal(0, 0.02, (n - pos - 1, k)), axis=0))
        bnoise = np.exp(np.cumsum(rng.normal(0, 0.02, n - pos - 1)))
        def warp(df):
            out = df.copy()
            out.iloc[pos + 1:] = out.iloc[pos + 1:].to_numpy() * noise
            return out
        b = rr.bars.replace(open=warp(rr.bars.open), high=warp(rr.bars.high), low=warp(rr.bars.low),
                            close=warp(rr.bars.close))
        bench = rr.bench.copy()
        bench.iloc[pos + 1:] = bench.iloc[pos + 1:].to_numpy() * bnoise
        w2 = _try_run(strat, rr, bars=b, bench=bench)
        if w2 is None:
            continue
        d = (w2.iloc[:pos + 1] - rr.w.iloc[:pos + 1]).abs().max(axis=1)
        if d.max() > 1e-8:
            first = d[d > 1e-8].index[0]
            bad.append(dict(cut=str(rr.bars.index[pos].date()), first_changed=str(first.date()),
                            rows_changed=int((d > 1e-8).sum())))
    c = Check(2, evidence=dict(tests=2, leaks=bad))
    if bad:
        c.status = FAILED
        c.finding = (f"Scrambling prices after {bad[0]['cut']} changed {bad[0]['rows_changed']} earlier decisions "
                     f"(first on {bad[0]['first_changed']}). Something like a full-sample normalization, a "
                     "negative shift or a fitted model is leaking future data into past signals.")
    else:
        c.status = ADDRESSED
        c.finding = ("Replaced every price after two cut dates with random noise; no decision before the cut "
                     "changed. Features, normalization and rankings only use past data.")
    return c


def check_survivorship(rr, declare, survivors=None):
    if getattr(rr, "pit", None) is not None:
        return _measured_survivorship(rr, survivors)
    bars = rr.bars
    end = bars.index[-1]
    last = bars.close.apply(lambda s: s.last_valid_index())
    dead = [t for t, d in last.items() if d is None or (end - d).days > 10]
    n = len(bars.tickers)
    alive_all = not dead
    ev = dict(tickers=n, delisted_in_universe=dead, source=rr.universe_source)
    c3, c12 = Check(3, evidence=ev), Check(12, evidence=ev)
    if declare.get("survivorship_free"):
        c3.status = PARTIAL
        c3.finding = ("You declared a survivorship-free universe. Audition's Yahoo data only lists securities "
                      "that still trade, so it can't confirm it.")
    elif alive_all and n >= 5:
        c3.status = PARTIAL
        c3.finding = (f"All {n} names in this universe still trade today. Yahoo carries no delisted companies, "
                      "so the universe was effectively chosen with hindsight. A list of today's winners "
                      "can add several points of CAGR a year, mostly to momentum and trend books. For US large "
                      "caps, set UNIVERSE = \"sp500\" and Audition rebuilds the index as it was on each date.")
    else:
        c3.status = PARTIAL
        c3.finding = (f"{len(dead)} of {n} names stopped trading during the sample, so some failures are "
                      "included. Yahoo still has no fully delisted names, so this remains incomplete.")
    if declare.get("point_in_time_universe"):
        c12.status = PARTIAL
        c12.finding = "Declared point-in-time constituents; Audition used the fixed list it was given and can't verify it."
    else:
        c12.status = NOT
        c12.finding = ("The universe is a fixed list for the whole history. Index membership at each date isn't "
                       "reconstructed. For US large caps use UNIVERSE = \"sp500\".")
    return c3, c12


def _measured_survivorship(rr, survivors):
    pit = rr.pit
    cov = pit.coverage
    in_sample = cov.loc[cov.index >= rr.sim.returns.index[0].year] if len(rr.sim.returns) else cov
    worst_unc = float(in_sample["uncovered"].max()) if len(in_sample) else float("nan")
    avg_unc = float(in_sample["uncovered"].mean()) if len(in_sample) else float("nan")
    worst_year = int(in_sample["uncovered"].idxmax()) if len(in_sample) else None
    s_p = rr.summary
    ev = dict(tickers=len(rr.bars.tickers), delisted_in_universe=["(point-in-time)"], pit=True,
              coverage=cov.round(4).reset_index().to_dict("records"), worst_uncovered=worst_unc,
              avg_uncovered=avg_unc, rejected_reused=pit.rejected_reused, unpriced=len(pit.unpriced),
              ineligible=getattr(rr, "ineligible", {}))
    c3, c12 = Check(3, evidence=ev), Check(12, evidence=ev)
    if survivors is not None and not survivors.error:
        # compare on the same dates
        idx = rr.sim.returns.index.intersection(survivors.sim.returns.index)
        sv = stats.summary(survivors.sim.returns.loc[idx], survivors.sim.rf.loc[idx], rr.cfg["ppy"])
        pt = stats.summary(rr.sim.returns.loc[idx], rr.sim.rf.loc[idx], rr.cfg["ppy"])
        d_cagr, d_sr = sv["cagr"] - pt["cagr"], sv["sharpe"] - pt["sharpe"]
        ev.update(survivor_cagr=sv["cagr"], survivor_sharpe=sv["sharpe"], pit_cagr=pt["cagr"], pit_sharpe=pt["sharpe"],
                  bias_cagr=d_cagr, bias_sharpe=d_sr)
        msg = (f"Run on today's members for the whole history, the rule shows CAGR {_fmt_pct(sv['cagr'])} and Sharpe "
               f"{sv['sharpe']:.2f}. Restricted to the stocks actually in the index on each date: CAGR "
               f"{_fmt_pct(pt['cagr'])}, Sharpe {pt['sharpe']:.2f}. Survivorship was worth {d_cagr * 100:+.1f} points "
               f"of CAGR a year and {d_sr:+.2f} of Sharpe.")
    else:
        msg = "Point-in-time membership applied (no survivor arm to compare against)."
    msg += (f" {avg_unc:.0%} of index members a year (worst {worst_unc:.0%}, {worst_year}) have no price on Yahoo, "
            "mostly bankruptcies and buyouts, so the point-in-time result is still an upper bound.")
    c3.finding = msg
    c3.status = ADDRESSED if worst_unc <= 0.05 else PARTIAL
    n_rej = len(pit.rejected_reused)
    ine = getattr(rr, "ineligible", {}) or {}
    c12.finding = (f"Index membership rebuilt for every date from Wikipedia's dated change list "
                   f"({len(pit.membership.snapshots)} snapshots back to {pit.membership.first_change.year}); "
                   f"{len(rr.bars.tickers)} current and former members priced, {n_rej} recycled ticker symbol(s) rejected, "
                   f"{len(pit.unpriced)} members unpriceable.")
    if ine.get("ineligible_openings"):
        c12.status = PARTIAL
        c12.finding += (f" Your rule tried to open {ine['ineligible_openings']} positions in stocks that weren't in "
                        "the index yet; Audition dropped them, leaving those slots in cash. Rank only where "
                        "data.eligible is True for a faithful test.")
    else:
        c12.status = ADDRESSED if worst_unc <= 0.05 else PARTIAL
        c12.finding += " The rule only opened positions in stocks that were members at the time."
    return c3, c12


def sensitivity_runs(strat, rr, quick, seed=5):
    """Re-run the strategy with nearby parameter values. Returns list of dicts."""
    base = strat.numeric_params()
    if not base:
        return []
    rng = np.random.default_rng(seed)
    configs = []
    mults = [0.75, 1.25] if quick else [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
    for k, v in base.items():
        for m in mults:
            nv = v * m
            if isinstance(v, int):
                nv = max(1, int(round(nv)))
                if nv == v:
                    continue
            configs.append(({k: nv}, f"{k}={nv:g}"))
    for _ in range(8 if quick else 24):
        p = {}
        for k, v in base.items():
            nv = v * rng.uniform(0.7, 1.3)
            p[k] = max(1, int(round(nv))) if isinstance(v, int) else nv
        configs.append((p, "joint"))
    out = []
    ppy = rr.cfg["ppy"]
    seen = set()
    for p, label in configs:
        key = tuple(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        w = _try_run(strat, rr, params=p)
        if w is None:
            continue
        sim = _sim(rr, w=w)
        ex = sim.returns - sim.rf
        out.append(dict(params=p, label=label, sharpe=stats.sharpe(ex, ppy),
                        cagr=stats.cagr(sim.returns, ppy), sr_period=float(ex.mean() / ex.std()) if ex.std() > 0 else 0.0,
                        returns=sim.returns))
    return out


def check_sensitivity(strat, rr, sens):
    c = Check(8)
    if not strat.numeric_params():
        c.status = NOT
        c.finding = "PARAMS has no numeric values, so there is nothing to vary. Put tunable numbers in PARAMS."
        return c
    base = rr.summary["sharpe"]
    srs = np.array([s["sharpe"] for s in sens]) if sens else np.array([])
    if len(srs) == 0:
        c.status = PARTIAL
        c.finding = "Every parameter variation raised an error, so sensitivity couldn't be measured."
        return c
    frac_pos = float((srs > 0).mean())
    frac_half = float((srs >= 0.5 * base).mean()) if base > 0 else float("nan")
    med = float(np.median(srs))
    pct_rank = float((srs < base).mean())
    one_at_a_time = [s for s in sens if s["label"] != "joint"]
    c.evidence = dict(base=base, runs=len(srs), median=med, frac_positive=frac_pos, frac_half=frac_half,
                      base_percentile=pct_rank, worst=float(srs.min()), best=float(srs.max()),
                      table=[(s["label"], s["sharpe"]) for s in one_at_a_time])
    if base > 0 and (med < 0.5 * base or frac_pos < 0.5):
        c.status = FAILED
        c.finding = (f"The chosen parameters sit on a peak: Sharpe {base:.2f}, but the median nearby setting "
                     f"gets {med:.2f} and only {frac_pos:.0%} of {len(srs)} neighbours are profitable.")
    elif base > 0 and frac_half < 0.7:
        c.status = PARTIAL
        c.finding = (f"{frac_half:.0%} of {len(srs)} nearby settings keep at least half the Sharpe "
                     f"(median {med:.2f} vs {base:.2f}). The edge is somewhat parameter-specific.")
    elif base <= 0:
        c.status = FAILED
        c.finding = f"The base parameters lose money after costs (Sharpe {base:.2f}); sensitivity is moot."
    else:
        c.status = ADDRESSED
        c.finding = (f"{frac_half:.0%} of {len(srs)} nearby settings keep at least half the Sharpe; median "
                     f"{med:.2f} vs {base:.2f} for the chosen values. The result sits on a plateau, not a spike.")
    return c


def check_multiple_testing(rr, declare, sens):
    ppy = rr.cfg["ppy"]
    ex = rr.sim.returns - rr.sim.rf
    declared = declare.get("variants_tried")
    srs = [s["sr_period"] for s in sens]
    var = float(np.var(srs, ddof=1)) if len(srs) >= 3 else (float(ex.mean() / ex.std()) ** 2 if ex.std() > 0 else 0.0)
    n_trials = int(declared) if declared else max(len(sens) + 1, 2)
    sr0 = stats.expected_max_sr(n_trials, var)
    dsr = stats.psr(ex, sr0)
    sr_p = float(ex.mean() / ex.std()) if ex.std() > 0 else 0.0
    adj = (sr_p - sr0) * math.sqrt(ppy)
    c = Check(4, evidence=dict(n_trials=n_trials, declared=bool(declared), sr0_annual=sr0 * math.sqrt(ppy),
                               deflated_sharpe_prob=dsr, selection_adjusted_sharpe=adj))
    src = f"{n_trials} variants you declared" if declared else \
        f"{n_trials} variants (the sensitivity grid; declare variants_tried for the real number)"
    base = (f"Against {src}, a skill-less search would still find a Sharpe of about "
            f"{sr0 * math.sqrt(ppy):.2f}. Deflated Sharpe probability {dsr:.0%}; selection-adjusted Sharpe {adj:.2f}.")
    if dsr >= 0.95:
        c.status = ADDRESSED if declared else PARTIAL
    elif dsr >= 0.5:
        c.status = PARTIAL
    else:
        c.status = FAILED
    c.finding = base + ("" if declared else " Not verified until you declare how many variants you actually tried.")
    return c


def check_overfitting(strat, rr, sens_check):
    ppy = rr.cfg["ppy"]
    r = rr.sim.returns
    ex = r - rr.sim.rf
    cut = int(len(r) * 0.7)
    s_in, s_out = stats.sharpe(ex.iloc[:cut], ppy), stats.sharpe(ex.iloc[cut:], ppy)
    n_par = len(strat.numeric_params())
    trades = engine.count_trades(rr.sim.held)
    c = Check(5, evidence=dict(params=n_par, trades=trades, sharpe_first70=s_in, sharpe_last30=s_out,
                               split=str(r.index[cut].date())))
    ratio = s_out / s_in if s_in > 0 else float("nan")
    msg = (f"{n_par} tunable parameter(s), {trades} trades. Sharpe {s_in:.2f} before "
           f"{r.index[cut].date()} and {s_out:.2f} after.")
    if sens_check.status == FAILED or (s_in > 0.5 and s_out <= 0):
        c.status = FAILED
        c.finding = msg + " The later period doesn't hold up and/or the parameters sit on a narrow peak."
    elif np.isfinite(ratio) and ratio >= 0.5 and sens_check.status == ADDRESSED:
        c.status = ADDRESSED
        c.finding = msg + " The later period keeps most of the edge and nearby parameters work too."
    else:
        c.status = PARTIAL
        c.finding = msg + " Some decay or parameter dependence; treat the headline as optimistic."
    return c


def check_oos(rr, declare):
    ppy = rr.cfg["ppy"]
    r, rf = rr.sim.returns, rr.sim.rf
    ex = r - rf
    start = declare.get("oos_start")
    looks = declare.get("oos_looks")
    c = Check(6)
    if not start:
        cut = int(len(r) * 0.7)
        c.status = NOT
        c.evidence = dict(audition_split=str(r.index[cut].date()),
                          sharpe_in=stats.sharpe(ex.iloc[:cut], ppy), sharpe_out=stats.sharpe(ex.iloc[cut:], ppy))
        c.finding = ("No out-of-sample start declared. Audition's own 70/30 split gives Sharpe "
                     f"{c.evidence['sharpe_in']:.2f} then {c.evidence['sharpe_out']:.2f}, but if you saw that "
                     "period while building the strategy it isn't out of sample. Declare oos_start and oos_looks.")
        return c
    ts = pd.Timestamp(start)
    s_in, s_out = stats.sharpe(ex[ex.index < ts], ppy), stats.sharpe(ex[ex.index >= ts], ppy)
    n_out = int((ex.index >= ts).sum())
    c.evidence = dict(oos_start=start, looks=looks, sharpe_in=s_in, sharpe_out=s_out, oos_periods=n_out)
    msg = f"Declared OOS from {start} ({n_out} periods): Sharpe {s_in:.2f} in-sample, {s_out:.2f} out-of-sample."
    if s_out <= 0:
        c.status = FAILED
        c.finding = msg + " The edge doesn't survive the out-of-sample period."
    elif looks is None:
        c.status = PARTIAL
        c.finding = msg + " How many times the OOS period was looked at is undeclared; each look turns it into in-sample."
    elif int(looks) <= 1:
        c.status = ADDRESSED
        c.finding = msg + " Declared as viewed once."
    else:
        c.status = PARTIAL
        c.finding = msg + f" Viewed {looks} times; repeated looks make it partly in-sample."
    return c


def check_regimes(rr):
    ppy = rr.cfg["ppy"]
    r = rr.sim.returns
    ex = r - rr.sim.rf
    by_year = ex.groupby(ex.index.year).sum()
    total = float(by_year.sum())
    best_year = int(by_year.idxmax()) if len(by_year) else None
    without_best = total - float(by_year.max()) if len(by_year) else 0.0
    b = rr.bench.reindex(r.index).ffill()
    sma = b.rolling(200, min_periods=150).mean()
    bull = (b > sma).shift(1).fillna(False).astype(bool)
    bvol = b.pct_change(fill_method=None).rolling(20).std().shift(1)
    hi_vol = bvol > bvol.median()
    seg = {
        "bull (benchmark > 200d avg)": stats.sharpe(ex[bull], ppy),
        "bear (benchmark < 200d avg)": stats.sharpe(ex[~bull], ppy),
        "high volatility": stats.sharpe(ex[hi_vol], ppy),
        "low volatility": stats.sharpe(ex[~hi_vol], ppy),
    }
    share = float(by_year.max() / total) if total > 0 else float("nan")
    c = Check(7, evidence=dict(segments=seg, best_year=best_year, best_year_share=share,
                               yearly=stats.yearly(r).to_dict(), excess_without_best=without_best))
    weak = [k for k, v in seg.items() if v < 0]
    if total > 0 and without_best <= 0:
        c.status = FAILED
        c.finding = (f"Remove {best_year} and the strategy has no excess return left. "
                     "The edge comes from one year.")
    elif total <= 0:
        c.status = FAILED
        c.finding = "No positive excess return overall, so no regime carries it."
    elif share > 0.5 or len(weak) >= 2:
        c.status = PARTIAL
        c.finding = (f"{best_year} supplies {share:.0%} of excess return"
                     + (f"; negative Sharpe in {', '.join(weak)}" if weak else "") + ".")
    else:
        c.status = ADDRESSED
        c.finding = ("Excess return is spread across years (best year " f"{share:.0%} of the total) and "
                     + ("is positive in every regime split." if not weak else f"only negative in {weak[0]}."))
    return c


def check_sample(rr):
    s = rr.summary
    trades = engine.count_trades(rr.sim.held)
    ex = rr.sim.returns - rr.sim.rf
    neff = montecarlo.effective_n(ex)
    c = Check(9, evidence=dict(periods=s["n"], years=s["years"], trades=trades, effective_n=neff,
                               min_trl_years=s["min_trl_years"]))
    mtrl = s["min_trl_years"]
    msg = (f"{s['n']} periods ({s['years']:.1f} years), {trades} position openings, about {neff:.0f} "
           f"independent observations. A Sharpe this size needs {mtrl:.1f} years to be 95% sure it beats zero.")
    if trades < 30 or not np.isfinite(mtrl) or mtrl > s["years"]:
        c.status = FAILED
        c.finding = msg + " The sample is too small for the claimed statistics."
    elif trades < 100 or mtrl > 0.5 * s["years"]:
        c.status = PARTIAL
        c.finding = msg
    else:
        c.status = ADDRESSED
        c.finding = msg
    return c


def placebo_runs(strat, rr, n, seed=9):
    """Shuffle the order of days (same shuffle for every asset and field) and re-run."""
    rng = np.random.default_rng(seed)
    bars = rr.bars
    close = bars.close
    rets = (close / close.shift(1)).iloc[1:]
    rel = {f: (getattr(bars, f) / close).iloc[1:] for f in ("open", "high", "low")}
    vol = bars.volume.iloc[1:]
    brets = (rr.bench / rr.bench.shift(1)).iloc[1:]
    out = []
    for _ in range(n):
        perm = rng.permutation(len(rets))
        r = rets.iloc[perm].set_axis(rets.index)
        # an asset that wasn't listed yet stays unlisted: keep the original NaN mask
        mask = close.iloc[1:].isna()
        r = r.where(~mask)
        first = close.bfill().iloc[0]
        path = r.fillna(1.0).cumprod() * first
        path = path.where(~mask)
        new_close = pd.concat([close.iloc[:1], path])
        frames = {f: pd.concat([getattr(bars, f).iloc[:1], rel[f].iloc[perm].set_axis(rets.index) * path])
                  for f in ("open", "high", "low")}
        b = bars.replace(close=new_close, volume=pd.concat([bars.volume.iloc[:1], vol.iloc[perm].set_axis(vol.index)]),
                         **frames)
        bench = pd.concat([rr.bench.iloc[:1], brets.iloc[perm].set_axis(brets.index).fillna(1.0).cumprod() * rr.bench.bfill().iloc[0]])
        w = _try_run(strat, rr, bars=b, bench=bench)
        if w is None:
            continue
        sim = engine.simulate(w, b, rr.rf, rr.cm, rr.cfg["ppy"], eval_start=getattr(rr, "eval_start", None))
        out.append(stats.sharpe(sim.returns - sim.rf, rr.cfg["ppy"]))
    return out


def check_significance(rr, placebo):
    ppy = rr.cfg["ppy"]
    ex = rr.sim.returns - rr.sim.rf
    s = rr.summary
    ci, p_boot = montecarlo.block_sharpe_ci(ex, ppy)
    real = s["sharpe"]
    p_perm = (1 + sum(x >= real for x in placebo)) / (len(placebo) + 1) if placebo else float("nan")
    c = Check(10, evidence=dict(t_stat=s["t_stat"], psr=s["psr"], sharpe_ci95=ci, p_bootstrap=p_boot,
                                placebo=placebo, p_permutation=p_perm))
    msg = (f"t-stat {s['t_stat']:.2f}, probabilistic Sharpe {s['psr']:.0%}, block-bootstrap 95% interval "
           f"for Sharpe {ci[0]:.2f} to {ci[2]:.2f}.")
    if placebo:
        msg += (f" On {len(placebo)} shuffled-calendar placebos (time structure destroyed) the strategy beat "
                f"its real Sharpe {sum(x >= real for x in placebo)} times, p = {p_perm:.2f}.")
    if ci[0] > 0 and s["psr"] >= 0.95 and (not placebo or p_perm <= 0.05):
        c.status = ADDRESSED
    elif ci[0] <= 0 and s["psr"] < 0.9:
        c.status = FAILED
        msg += " The result is consistent with no edge."
    else:
        c.status = PARTIAL
    c.finding = msg
    return c


# --------------------------------------------------------------------------- data

SPLIT_RATIOS = [1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2, 3, 4, 5, 10]


def check_adjustments(rr, strat=None):
    adj = rr.bars.close
    r = adj / adj.shift(1)
    big = r[(r > 1.4) | (r < 0.6)].stack().dropna()
    vol = rr.bars.volume
    vol_ratio = vol / vol.rolling(20, min_periods=5).median().shift(1)
    split_like = []
    for (d, t), v in big.items():
        # a missed split moves the price by an exact ratio on ordinary volume; real news comes with a volume spike
        spike = vol_ratio.at[d, t] if (d in vol_ratio.index and t in vol_ratio.columns) else float("nan")
        if any(abs(v / s - 1) < 0.03 for s in SPLIT_RATIOS) and not (np.isfinite(spike) and spike > 2):
            split_like.append(dict(date=str(d.date()), ticker=t, move=float(v - 1)))
    factor = (rr.bars.close / rr.bars.raw_close)
    div_adjusted = bool((factor.std() > 1e-6).any())
    c = Check(11, evidence=dict(large_moves=len(big), split_like=split_like[:20], dividend_adjusted=div_adjusted))
    if split_like:
        x = split_like[0]
        # measure: neutralize those bars (hold the prior close) and re-run
        s2 = float("nan")
        if strat is not None:
            fixed = {}
            mask = pd.DataFrame(False, index=adj.index, columns=adj.columns)
            for e in split_like:
                mask.at[pd.Timestamp(e["date"]), e["ticker"]] = True
            # from the jump onward the series is on a different scale: rescale it back
            for f in ("open", "high", "low", "close"):
                df = getattr(rr.bars, f).copy()
                for e in split_like:
                    d, t = pd.Timestamp(e["date"]), e["ticker"]
                    df.loc[df.index >= d, t] = df.loc[df.index >= d, t] / (1 + e["move"])
                fixed[f] = df
            b = rr.bars.replace(**fixed)
            w2 = _try_run(strat, rr, bars=b)
            if w2 is not None:
                s2 = _sharpe(engine.simulate(w2, b, rr.rf, rr.cm, rr.cfg["ppy"],
                                             eval_start=getattr(rr, "eval_start", None)), rr.cfg["ppy"])
        base = rr.summary["sharpe"]
        c.evidence.update(sharpe_repaired=s2)
        moved = np.isfinite(s2) and abs(s2 - base) > max(0.1, 0.15 * abs(base))
        c.status = FAILED if moved or not np.isfinite(s2) else PARTIAL
        c.finding = (f"{len(split_like)} day(s) look like unadjusted splits or bad prints, e.g. {x['ticker']} "
                     f"{x['move']:+.0%} on {x['date']}. "
                     + (f"Repairing them moves Sharpe from {base:.2f} to {s2:.2f}" if np.isfinite(s2) else
                        "Couldn't re-run with them repaired")
                     + (", so they matter." if moved else "; they don't drive the result."))
    elif not div_adjusted:
        c.status = PARTIAL
        c.finding = "Prices don't appear dividend-adjusted; total return is understated for payers."
    else:
        c.status = ADDRESSED
        c.finding = (f"Split- and dividend-adjusted prices; no split-sized jumps found "
                     f"({len(big)} moves over 40% checked).")
    return c


def check_alignment(rr):
    close = rr.bars.close
    first = close.apply(lambda s: s.first_valid_index())
    last = close.apply(lambda s: s.last_valid_index())
    live = pd.DataFrame({t: (close.index >= first[t]) & (close.index <= last[t]) for t in close.columns},
                        index=close.index)
    gaps = close.isna() & live
    frac_rows = float(gaps.any(axis=1).mean())
    suffixes = sorted({("." + t.split(".")[-1]) if "." in t else ("-USD" if t.endswith("-USD") else "US")
                       for t in close.columns})
    c = Check(13, evidence=dict(venues=suffixes, rows_with_gaps=frac_rows))
    if len(suffixes) > 1:
        c.status = PARTIAL
        c.finding = (f"The universe trades on {len(suffixes)} venues ({', '.join(suffixes)}) with different "
                     f"hours and holidays; {frac_rows:.1%} of days have some assets closed. Audition fills each "
                     "asset at its own open, but close-to-close comparisons across venues mix timestamps.")
    elif frac_rows > 0.01:
        c.status = PARTIAL
        c.finding = f"{frac_rows:.1%} of days have some assets missing on the same calendar."
    else:
        c.status = ADDRESSED
        c.finding = "Single venue, one daily calendar; signals and fills share timestamps."
    return c


def check_missing(rr):
    close = rr.bars.close
    first = close.apply(lambda s: s.first_valid_index())
    last = close.apply(lambda s: s.last_valid_index())
    live = pd.DataFrame({t: (close.index >= first[t]) & (close.index <= last[t]) for t in close.columns},
                        index=close.index)
    gaps = close.isna() & live
    held_on_gap = int(((rr.w.abs() > 1e-9) & gaps).to_numpy().sum())
    w_before_listing = int(((rr.w.abs() > 1e-9) & ~live).to_numpy().sum())
    gap_frac = float(gaps.to_numpy().sum() / max(live.to_numpy().sum(), 1))
    c = Check(14, evidence=dict(gap_fraction=gap_frac, weights_on_missing=held_on_gap,
                                weights_outside_listing=w_before_listing))
    if w_before_listing:
        c.status = FAILED
        c.finding = (f"{w_before_listing} position-days are held in assets before they listed or after they "
                     "stopped trading. Missing prices are being treated as tradable.")
    elif held_on_gap or gap_frac > 0.005:
        c.status = PARTIAL
        c.finding = (f"{gap_frac:.2%} of live asset-days are missing; {held_on_gap} position-days sit on missing "
                     "bars (marked flat by Audition).")
    else:
        c.status = ADDRESSED
        c.finding = f"Missing bars are rare ({gap_frac:.2%}) and the strategy never holds through one."
    return c


def check_revisions(strat, rr):
    f = rr.bars.raw_close / rr.bars.close
    b = rr.bars.replace(close=rr.bars.raw_close, open=rr.bars.open * f, high=rr.bars.high * f, low=rr.bars.low * f)
    w2 = _try_run(strat, rr, bars=b)
    c = Check(15)
    if w2 is None:
        c.status = PARTIAL
        c.finding = "Couldn't re-run on unadjusted prices."
        return c
    s2 = _sharpe(_sim(rr, w=w2), rr.cfg["ppy"])
    base = rr.summary["sharpe"]
    c.evidence = dict(sharpe_adjusted=base, sharpe_unadjusted_signals=s2)
    if abs(s2 - base) > max(0.3, 0.25 * abs(base)):
        c.status = PARTIAL
        c.finding = (f"Signals computed on prices as they were quoted at the time give Sharpe {s2:.2f} vs {base:.2f} "
                     "on back-adjusted prices. Back-adjustment uses dividends paid later, so the signal partly "
                     "depends on revised history.")
    else:
        c.status = ADDRESSED
        c.finding = (f"Signals on as-quoted prices give Sharpe {s2:.2f} vs {base:.2f}; back-adjustment doesn't "
                     "drive the result. (Fundamental or estimate data, if any, isn't covered.)")
    return c


def check_calendars(rr):
    idx = rr.bars.index
    weekend = int((idx.dayofweek >= 5).sum())
    dup = int(idx.duplicated().sum())
    gaps = pd.Series(idx[1:] - idx[:-1]).dt.days
    long_gaps = int((gaps > 5).sum())
    ghost = rr.bars.volume.eq(0) & rr.bars.high.eq(rr.bars.low)
    ghost_rows = int(ghost.all(axis=1).sum())
    crypto = rr.cfg.get("ppy") == 365
    c = Check(16, evidence=dict(weekend_rows=weekend, duplicate_dates=dup, gaps_over_5d=long_gaps, ghost_rows=ghost_rows))
    issues = []
    if weekend and not crypto:
        issues.append(f"{weekend} weekend rows")
    if dup:
        issues.append(f"{dup} duplicate dates")
    if ghost_rows:
        issues.append(f"{ghost_rows} zero-volume flat days (holiday filler)")
    if long_gaps:
        issues.append(f"{long_gaps} gaps over 5 days")
    if issues:
        c.status = PARTIAL
        c.finding = "Calendar oddities: " + ", ".join(issues) + ". Audition holds positions flat across them."
    else:
        c.status = ADDRESSED
        c.finding = "Exchange calendar is clean: no weekend rows, duplicates, holiday filler bars or long gaps."
    return c


def check_bad_ticks(strat, rr):
    close = rr.bars.close
    r = close.pct_change(fill_method=None)
    sig = r.rolling(60, min_periods=20).std().shift(1)
    spike = (r.abs() > 8 * sig) & ((r.shift(-1) * r) < 0) & (r.shift(-1).abs() > 0.5 * r.abs())
    n = int(spike.to_numpy().sum())
    c = Check(17, evidence=dict(spikes=n))
    if n == 0:
        c.status = ADDRESSED
        c.finding = "No one-day spike-and-revert prints over 8 sigma."
        return c
    cleaned = {}
    for f in ("open", "high", "low", "close"):
        df = getattr(rr.bars, f).copy()
        df = df.mask(spike, close.shift(1))
        cleaned[f] = df
    b = rr.bars.replace(**cleaned)
    w2 = _try_run(strat, rr, bars=b)
    s2 = _sharpe(engine.simulate(w2, b, rr.rf, rr.cm, rr.cfg["ppy"], eval_start=getattr(rr, "eval_start", None)), rr.cfg["ppy"]) if w2 is not None else float("nan")
    base = rr.summary["sharpe"]
    c.evidence.update(sharpe_cleaned=s2)
    if np.isfinite(s2) and base > 0 and s2 < 0.7 * base:
        c.status = FAILED
        c.finding = f"{n} suspect spikes. Removing them cuts Sharpe from {base:.2f} to {s2:.2f}; bad prints drive the result."
    else:
        c.status = PARTIAL if n > 5 else ADDRESSED
        c.finding = f"{n} suspect spike(s). With them removed Sharpe is {_fmt(s2)} vs {base:.2f}; they don't drive the result."
    return c


def _alpaca_closes(tickers, start):
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        return None
    out = {}
    for t in tickers:
        url = (f"https://data.alpaca.markets/v2/stocks/{t}/bars?timeframe=1Day&adjustment=all"
               f"&start={start}&limit=10000&feed=sip")
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        try:
            rows = []
            tok = None
            while True:
                u = url + (f"&page_token={tok}" if tok else "")
                req = urllib.request.Request(u, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
                js = json.loads(urllib.request.urlopen(req, timeout=20).read())
                rows += js.get("bars") or []
                tok = js.get("next_page_token")
                if not tok:
                    break
            s = pd.Series({pd.Timestamp(b["t"]).tz_localize(None).normalize(): b["c"] for b in rows})
            if len(s):
                out[t] = s
        except Exception:
            continue
    return out


def check_source(rr):
    c = Check(18)
    us = [t for t in rr.bars.tickers if "." not in t and not t.endswith("-USD")]
    alp = _alpaca_closes(us[:10], str(rr.bars.index[0].date())) if us else None
    if not alp:
        c.status = NOT
        c.finding = ("Only one data source (Yahoo) was used. Set APCA_API_KEY_ID / APCA_API_SECRET_KEY to have "
                     "Audition cross-check US prices against Alpaca SIP data.")
        return c
    rows = []
    for t, s in alp.items():
        y = rr.bars.close[t].dropna()
        a = s.reindex(y.index).dropna()
        ry, ra = y.reindex(a.index).pct_change(), a.pct_change()
        d = (ry - ra).abs()
        rows.append(dict(ticker=t, days=len(a), corr=float(ry.corr(ra)), max_diff=float(d.max()),
                         days_over_1pct=int((d > 0.01).sum())))
    c.evidence = dict(compared=rows)
    worst_corr = min(r["corr"] for r in rows)
    bad = sum(r["days_over_1pct"] for r in rows)
    if worst_corr > 0.99 and bad < 5 * len(rows):
        c.status = ADDRESSED
        c.finding = f"Daily returns for {len(rows)} names agree with Alpaca SIP (worst correlation {worst_corr:.4f})."
    else:
        c.status = PARTIAL
        c.finding = (f"Yahoo and Alpaca disagree: worst return correlation {worst_corr:.3f}, {bad} days differ by "
                     "more than 1%. Check which source is right before trusting the backtest.")
    return c


# --------------------------------------------------------------------------- execution

def check_fills(rr):
    trades = rr.sim.held.diff().fillna(rr.sim.held).abs() > 1e-9
    o = rr.bars.open.reindex(trades.index)
    v = rr.bars.volume.reindex(trades.index)
    impossible = trades & (o.isna() | (v <= 0))
    n_t = int(trades.to_numpy().sum())
    n_bad = int(impossible.to_numpy().sum())
    c = Check(19, evidence=dict(trades=n_t, impossible=n_bad))
    if n_t and n_bad / n_t > 0.01:
        c.status = FAILED
        c.finding = f"{n_bad} of {n_t} fills land on bars with no open price or zero volume; those fills couldn't happen."
    elif n_bad:
        c.status = PARTIAL
        c.finding = f"{n_bad} of {n_t} fills land on bars with no open price or zero volume."
    else:
        c.status = ADDRESSED
        c.finding = (f"All {n_t} fills happen at the next bar's open after the signal close, on bars that "
                     "actually traded. No same-bar fills.")
    return c


def check_slippage_impact(rr, ppy):
    no_imp = _sim(rr, cm=rr.cm.scaled(impact=False))
    s_no = _sharpe(no_imp, ppy)
    base = rr.summary["sharpe"]
    part = rr.sim.participation.where(rr.sim.held.diff().abs() > 1e-9).stack()
    med_p = float(part.median()) if len(part) else 0.0
    p95 = float(part.quantile(0.95)) if len(part) else 0.0
    c20 = Check(20, status=ADDRESSED, evidence=dict(cost_bps_per_turnover=rr.sim.cost_bps_per_turnover))
    c20.finding = (f"Slippage scales with each order's size and the asset's volatility (square-root impact "
                   f"model, coefficient {rr.cm.impact_coef:g}), not a fixed number. Average all-in cost {rr.sim.cost_bps_per_turnover:.1f} bps per "
                   f"unit traded at ${rr.cm.capital:,.0f}.")
    c23 = Check(23, evidence=dict(sharpe_no_impact=s_no, sharpe=base, median_participation=med_p, p95_participation=p95))
    msg = (f"Median order is {med_p:.3%} of 20-day dollar volume (95th pct {p95:.2%}). Impact moves Sharpe from "
           f"{s_no:.2f} to {base:.2f} at ${rr.cm.capital:,.0f}.")
    if s_no > 0 and base < 0.5 * s_no:
        c23.status = PARTIAL
        c23.finding = msg + " Impact eats more than half the edge at this size."
    else:
        c23.status = ADDRESSED
        c23.finding = msg
    c22 = Check(22, evidence=dict(median_participation=med_p, p95_participation=p95))
    if p95 > 0.2:
        c22.status = FAILED
        c22.finding = f"5% of orders exceed {p95:.0%} of daily volume; they would not fill at the open in one go."
    elif p95 > 0.02:
        c22.status = PARTIAL
        c22.finding = (f"The largest orders reach {p95:.1%} of daily volume, more than an opening auction "
                       "usually absorbs; expect partial fills or spreading the order out.")
    else:
        c22.status = ADDRESSED
        c22.finding = f"Orders stay under {p95:.2%} of daily volume (95th pct); full fills at the open are plausible."
    return c20, c22, c23


def check_spread(rr):
    sp = rr.cm.spread_bps
    det = rr.cm.spread_detail or {}
    traded = [t for t in rr.sim.held.columns if rr.sim.held[t].abs().sum() > 0]
    c = Check(21, status=ADDRESSED, evidence=dict(spread_bps=sp.round(1).to_dict(), detail=det))
    if det and any(d.get("declared") for d in det.values()):
        c.finding = f"Using the spreads you declared (median {sp.median():.1f} bps); half is paid on every trade."
        return c
    show = sorted(traded, key=lambda t: -rr.sim.held[t].abs().sum())[:4]
    parts = []
    for t in show:
        d = det.get(t, {})
        dv = d.get("dollar_volume")
        parts.append(f"{t} {d.get('used', float('nan')):.1f} bps"
                     + (f" (${dv / 1e6:,.0f}M/day)" if dv and np.isfinite(dv) else ""))
    c.finding = (f"Half the spread is paid on every trade. Each asset's spread is the typical quoted spread "
                 f"for its dollar volume over the last two years, or a high/low estimate (Abdi-Ranaldo) for names under $2M a day, "
                 f"floor {rr.cfg['spread_floor_bps']:.0f} bps"
                 + (": " + ", ".join(parts) + "." if parts else ".")
                 + " Declare DECLARE['costs'] = {'spread_bps': ..} to use your broker's numbers.")
    return c


def check_latency(rr, ppy):
    inst = _sim(rr, fill="close", delay=0)
    d2 = _sim(rr, delay=2)
    s_i, s_2, base = _sharpe(inst, ppy), _sharpe(d2, ppy), rr.summary["sharpe"]
    c = Check(24, evidence=dict(instant_close=s_i, next_open=base, open_plus_1=s_2))
    msg = (f"Sharpe {s_i:.2f} if filled at the signal's own close (impossible), {base:.2f} at the next open "
           f"(Audition default), {s_2:.2f} one day later.")
    if base > 0 and s_2 < 0.5 * base:
        c.status = PARTIAL
        c.finding = msg + " The edge decays fast; any execution delay costs most of it."
    else:
        c.status = ADDRESSED
        c.finding = msg
    return c


def check_fees(rr):
    taxed = {t: v for t, v in rr.cm.buy_tax_bps.items() if v > 0}
    c = Check(25, status=ADDRESSED, evidence=dict(fee_bps=rr.cm.fee_bps, buy_taxes=taxed))
    c.finding = (f"{rr.cm.fee_bps:g} bps commission per side" +
                 (f", plus buy-side transaction tax on {len(taxed)} name(s) (UK stamp duty / FTT)." if taxed else "."))
    return c


def check_shorting(rr, ppy):
    short = (-rr.sim.held.clip(upper=0)).sum(axis=1)
    c = Check(26)
    if short.max() < 1e-9:
        c.status = ADDRESSED
        c.finding = "Long-only; no borrow needed."
        return c
    stress = _sim(rr, cm=rr.cm.scaled(borrow_bps_yr=500.0))
    s5 = _sharpe(stress, ppy)
    c.evidence = dict(avg_short=float(short.mean()), sharpe_borrow_50bp=rr.summary["sharpe"], sharpe_borrow_500bp=s5)
    c.status = PARTIAL
    c.finding = (f"Average short exposure {short.mean():.0%}. Borrow charged at 0.5%/yr (Sharpe "
                 f"{rr.summary['sharpe']:.2f}); at 5%/yr hard-to-borrow rates Sharpe is {s5:.2f}. Locate "
                 "availability and recalls aren't modelled.")
    return c


def check_orders(declare):
    c = Check(27)
    if declare.get("order_type", "market").lower() != "market":
        c.status = PARTIAL
        c.finding = ("You declared non-market orders. Audition only models market-on-open fills, so limit-order "
                     "fill rates and adverse selection aren't tested.")
    else:
        c.status = ADDRESSED
        c.finding = ("Market-on-open orders, filled at the actual open including overnight gaps; no intrabar "
                     "assumptions.")
    return c


def check_capacity(rr, ppy, capital):
    caps = [1e5, 1e6, 1e7, 1e8, 1e9]
    res = {cap: _sharpe(_sim(rr, cm=rr.cm.scaled(capital=cap)), ppy) for cap in caps}
    small = res[1e5]
    ok = [cap for cap in caps if small > 0 and res[cap] >= 0.5 * small]
    cap_max = max(ok) if ok else 0
    c = Check(28, evidence=dict(curve={f"{k:.0e}": v for k, v in res.items()}, capacity=cap_max))
    at = rr.summary["sharpe"]
    msg = ("Sharpe by capital: " + ", ".join(f"${k/1e6:g}M {v:.2f}" for k, v in res.items()) +
           f". Keeps half its small-account Sharpe up to about ${cap_max/1e6:g}M.")
    if at <= 0:
        c.status = FAILED
    elif small > 0 and at < 0.75 * small:
        c.status = PARTIAL
    else:
        c.status = ADDRESSED
    c.finding = msg
    return c


# --------------------------------------------------------------------------- metrics

def _cmp(name, audit_val, reported, tol_abs, tol_rel, pct=True):
    if reported is None:
        return None
    diff = abs(audit_val - reported)
    ok = diff <= max(tol_abs, tol_rel * abs(reported))
    f = _fmt_pct if pct else _fmt
    return ok, f"You reported {name} {f(reported)}; Audition measures {f(audit_val)} over the same window."


def _reported_window(rr, rep, ppy):
    """Audition's summary over the window the reported numbers cover (whole sample if none given).

    Reported numbers describe the file as written, so when Audition swapped a static S&P list for
    point-in-time members, they are compared with the run on the list as given (the survivor arm)."""
    ref = getattr(rr, "reported_ref", None) or rr
    arm = " on your list as given" if ref is not rr else ""
    if not rep.get("start") and not rep.get("end"):
        return ref.summary, arm
    r = ref.sim.returns.loc[rep.get("start"):rep.get("end")]
    return (stats.summary(r, ref.sim.rf, ppy),
            f" (compared over {r.index[0].date()} to {r.index[-1].date()}{arm})")


def check_metrics(rr, declare, ppy):
    s = rr.summary
    rep = declare.get("reported", {}) or {}
    ws, wtxt = _reported_window(rr, rep, ppy)
    out = []

    zero_cash = rr.sim.returns - rr.sim.cash_return.clip(lower=0)
    c29 = Check(29, evidence=dict(cagr=s["cagr"], cagr_idle_cash_zero=stats.cagr(zero_cash, ppy)))
    cmp = _cmp("CAGR", ws["cagr"], rep.get("cagr"), 0.03, 0.25)
    c29.finding = (f"CAGR {_fmt_pct(s['cagr'])}, compounded daily, net of all costs, idle cash earning T-bills "
                   f"({_fmt_pct(c29.evidence['cagr_idle_cash_zero'])} if idle cash earned nothing).")
    if cmp:
        c29.status = ADDRESSED if cmp[0] else FAILED
        c29.finding = cmp[1] + " " + c29.finding + ("" if cmp[0] else " The reported number doesn't reproduce.")
    else:
        c29.status = ADDRESSED
    out.append(c29)

    raw_sr = stats.sharpe(rr.sim.returns, ppy)
    c30 = Check(30, evidence=dict(sharpe=s["sharpe"], sharpe_no_rf=raw_sr))
    c30.finding = (f"Sharpe {s['sharpe']:.2f} on daily excess returns over 13-week T-bills, annualized by "
                   f"sqrt({ppy}). Ignoring the risk-free rate would show {raw_sr:.2f}.")
    cmp = _cmp("Sharpe", ws["sharpe"], rep.get("sharpe"), 0.2, 0.2, pct=False)
    if cmp:
        c30.status = ADDRESSED if cmp[0] else FAILED
        c30.finding = cmp[1] + " " + c30.finding
    else:
        c30.status = ADDRESSED
    out.append(c30)

    rho1 = s["rho"][0] if s["rho"] else 0.0
    m = (1 + rr.sim.returns).resample("ME").prod() - 1
    mrf = (1 + rr.sim.rf).resample("ME").prod() - 1
    sr_m = stats.sharpe(m - mrf, 12)
    c31 = Check(31, evidence=dict(naive=s["sharpe"], lo_adjusted=s["lo_sharpe"], rho1=rho1, monthly=sr_m))
    msg = (f"Lag-1 autocorrelation {rho1:+.3f}. Autocorrelation-adjusted Sharpe (Lo 2002) {s['lo_sharpe']:.2f} "
           f"vs {s['sharpe']:.2f} naive; from monthly returns {sr_m:.2f}.")
    if s["sharpe"] > 0 and s["lo_sharpe"] < 0.5 * s["sharpe"]:
        c31.status = FAILED
        c31.finding = msg + " Smoothed returns inflate the daily Sharpe a lot."
    elif s["sharpe"] > 0 and s["lo_sharpe"] < 0.8 * s["sharpe"]:
        c31.status = PARTIAL
        c31.finding = msg + " Positive autocorrelation inflates the Sharpe somewhat."
    else:
        c31.status = ADDRESSED
        c31.finding = msg
    out.append(c31)

    c32 = Check(32, status=ADDRESSED, evidence=dict(max_dd=s["max_dd"], calmar=s["calmar"]))
    c32.finding = f"Max drawdown {_fmt_pct(s['max_dd'])} on daily net equity; Calmar {_fmt(s['calmar'])}."
    cmp = _cmp("max drawdown", ws["max_dd"], rep.get("max_dd"), 0.03, 0.25)
    if cmp:
        c32.status = ADDRESSED if cmp[0] else FAILED
        c32.finding = cmp[1] + " " + c32.finding
    out.append(c32)

    dur_y = s["dd_duration"] / ppy
    c33 = Check(33, evidence=dict(periods=s["dd_duration"], years=dur_y))
    c33.status = PARTIAL if dur_y > 2 else ADDRESSED
    c33.finding = f"Longest time under water: {s['dd_duration']} periods ({dur_y:.1f} years)."
    out.append(c33)

    c34 = Check(34, evidence={k: s[k] for k in ("worst_day", "worst_week", "worst_month", "es95", "es99", "skew", "kurt")})
    c34.finding = (f"Worst day {_fmt_pct(s['worst_day'])}, worst month {_fmt_pct(s['worst_month'])}, expected "
                   f"shortfall {_fmt_pct(s['es95'], 2)} (95%) / {_fmt_pct(s['es99'], 2)} (99%) per day, skew "
                   f"{s['skew']:.2f}, excess kurtosis {s['kurt']:.1f}.")
    c34.status = PARTIAL if (s["skew"] < -1 or s["kurt"] > 10) else ADDRESSED
    if c34.status == PARTIAL:
        c34.finding += " Fat left tail: the average hides occasional large losses."
    out.append(c34)

    gross = rr.sim.held.abs().sum(axis=1)
    c35 = Check(35, evidence=dict(avg_gross=float(gross.mean()), max_gross=float(gross.max())))
    c35.finding = (f"Gross exposure averages {gross.mean():.2f}x, peaks at {gross.max():.2f}x. Leverage is "
                   "financed at T-bills + 0.5% and included in every return and risk figure.")
    c35.status = PARTIAL if gross.max() > 3 else ADDRESSED
    out.append(c35)

    contrib = rr.sim.contrib.sum()
    tot = float(contrib.sum())
    top = contrib.idxmax() if len(contrib) else None
    share = float(contrib.max() / tot) if tot > 0 else float("nan")
    without = tot - float(contrib.max()) if len(contrib) else 0.0
    c36 = Check(36, evidence=dict(top=top, top_share=share, by_asset=contrib.sort_values(ascending=False).head(10).to_dict()))
    if len(contrib) > 1 and tot > 0 and without <= 0:
        c36.status = FAILED
        c36.finding = f"{top} supplies {share:.0%} of gross P&L; without it the strategy makes nothing."
    elif len(contrib) > 1 and share > 0.5:
        c36.status = PARTIAL
        c36.finding = f"{top} supplies {share:.0%} of gross P&L."
    else:
        c36.status = ADDRESSED
        c36.finding = (f"Largest contributor {top} supplies {_fmt_pct(share, 0)} of gross P&L." if len(contrib) > 1
                       else "Single-asset strategy; concentration is by design.")
    out.append(c36)

    rets = rr.bars.close.pct_change(fill_method=None).iloc[-min(len(rr.bars.index), 756):]
    corr = rets.corr()
    held_names = [t for t in rr.sim.held.columns if rr.sim.held[t].abs().mean() > 0.01]
    sub = corr.loc[held_names, held_names] if len(held_names) > 1 else None
    avg_c = float(sub.where(~np.eye(len(sub), dtype=bool)).stack().mean()) if sub is not None else float("nan")
    w = rr.sim.held.abs()
    eff = ((w.sum(axis=1) ** 2) / (w ** 2).sum(axis=1)).replace([np.inf, -np.inf], np.nan).mean()
    c37 = Check(37, evidence=dict(avg_pair_corr=avg_c, effective_positions=float(eff) if np.isfinite(eff) else None))
    if np.isfinite(avg_c) and avg_c > 0.6:
        c37.status = PARTIAL
        c37.finding = (f"Held assets have an average pairwise correlation of {avg_c:.2f} (last 3 years); "
                       f"{eff:.1f} effective positions behave close to one bet.")
    else:
        c37.status = ADDRESSED
        c37.finding = (f"Average pairwise correlation of held assets {_fmt(avg_c)}; about {_fmt(eff, 1)} "
                       "effective positions at a time.")
    out.append(c37)

    bopen = getattr(rr, "bench_open", rr.bench)
    bret = (bopen.shift(-1) / bopen - 1).reindex(rr.sim.returns.index).fillna(0.0)
    bsum = stats.summary(bret, rr.sim.rf, ppy)
    ba = stats.beta_alpha(rr.sim.returns, bret, rr.sim.rf, ppy)
    c38 = Check(38, evidence=dict(bench=rr.cfg["bench"], bench_sharpe=bsum["sharpe"], bench_cagr=bsum["cagr"],
                                  bench_max_dd=bsum["max_dd"], **ba))
    msg = (f"Against buy-and-hold {rr.cfg['bench']} (Sharpe {bsum['sharpe']:.2f}, CAGR {_fmt_pct(bsum['cagr'])}): "
           f"beta {_fmt(ba['beta'])}, alpha {_fmt_pct(ba['alpha'])}/yr, alpha t-stat {_fmt(ba['alpha_t'])}.")
    if s["sharpe"] < bsum["sharpe"] and not (ba["alpha_t"] > 1):
        c38.status = FAILED
        c38.finding = msg + " Holding the index did better on a risk-adjusted basis."
    elif ba["alpha_t"] > 2:
        c38.status = ADDRESSED
        c38.finding = msg
    else:
        c38.status = PARTIAL
        c38.finding = msg + " Alpha isn't statistically distinct from market exposure."
    out.append(c38)
    return out


# --------------------------------------------------------------------------- monte carlo

def check_montecarlo(rr, ppy, sens, quick):
    r = rr.sim.returns
    mc = montecarlo.run(r, ppy, years=3.0, paths=1000 if quick else 3000)
    b, i = mc["block"], mc["iid"]
    out = []
    c39 = Check(39, status=ADDRESSED, evidence=dict(block=b["mdd_p"], iid=i["mdd_p"]))
    c39.finding = (f"Stationary block bootstrap (mean block {mc['mean_block']:.0f} periods) keeps volatility "
                   f"clustering and autocorrelation. Median 3-year max drawdown {_fmt_pct(b['mdd_p'][2])} with "
                   f"blocks vs {_fmt_pct(i['mdd_p'][2])} if days were shuffled independently.")
    out.append(c39)

    worst = float(r.min())
    c40 = Check(40, status=PARTIAL, evidence=dict(hist_worst_day=worst, p5_mdd=b["mdd_p"][0]))
    c40.finding = (f"Resampling can't produce a day worse than the worst one in the sample ({_fmt_pct(worst)}). "
                   f"5th-percentile 3-year drawdown is {_fmt_pct(b['mdd_p'][0])}; a crash larger than anything "
                   "in the data isn't in these numbers.")
    out.append(c40)

    c41 = Check(41)
    if sens:
        cg = np.array([s["cagr"] for s in sens])
        srs = np.array([s["sharpe"] for s in sens])
        c41.evidence = dict(cagr_p=np.percentile(cg, [5, 50, 95]).tolist(), sharpe_p=np.percentile(srs, [5, 50, 95]).tolist())
        c41.status = ADDRESSED if np.percentile(srs, 5) > 0 else PARTIAL
        c41.finding = (f"Across {len(sens)} plausible parameter sets, Sharpe ranges {np.percentile(srs, 5):.2f} "
                       f"to {np.percentile(srs, 95):.2f} (5th-95th pct), CAGR {_fmt_pct(np.percentile(cg, 5))} to "
                       f"{_fmt_pct(np.percentile(cg, 95))}.")
    else:
        c41.status = NOT
        c41.finding = "No numeric PARAMS to draw from."
    out.append(c41)

    mults = [0, 1, 2, 3, 5]
    curve = {m: _sharpe(_sim(rr, cm=rr.cm.scaled(cost_mult=m)), ppy) for m in mults}
    be = None
    for a, b_ in zip(mults, mults[1:]):
        if curve[a] > 0 >= curve[b_]:
            be = a + (b_ - a) * curve[a] / (curve[a] - curve[b_])
            break
    c42 = Check(42, evidence=dict(curve=curve, breakeven_multiple=be))
    msg = "Sharpe at " + ", ".join(f"{m}x costs {v:.2f}" for m, v in curve.items()) + "."
    if curve[2] <= 0:
        c42.status = FAILED
        c42.finding = msg + " Doubling costs wipes out the edge."
    elif curve[3] <= 0:
        c42.status = PARTIAL
        c42.finding = msg + f" Breaks even at about {be:.1f}x costs."
    else:
        c42.status = ADDRESSED
        c42.finding = msg
    out.append(c42)

    b_ = rr.bench.reindex(r.index).ffill()
    bear = (b_ < b_.rolling(200, min_periods=150).mean()).shift(1).fillna(False).astype(bool)
    c43 = Check(43)
    rb = r[bear]
    if len(rb) >= 60:
        mcb = montecarlo.run(rb, ppy, years=1.0, paths=1000)
        c43.status = ADDRESSED
        c43.evidence = dict(bear_days=len(rb), p_loss_1y=mcb["block"]["p_loss"], mdd_med=mcb["block"]["mdd_p"][2])
        c43.finding = (f"Resampling only bear-market periods ({len(rb)} of them): {mcb['block']['p_loss']:.0%} "
                       f"chance of losing money over a year, median drawdown {_fmt_pct(mcb['block']['mdd_p'][2])}.")
    else:
        c43.status = PARTIAL
        c43.finding = f"Only {len(rb)} bear-market periods in the sample; too few to simulate a bear regime."
    out.append(c43)

    neff = montecarlo.effective_n(r - rr.sim.rf)
    c44 = Check(44, status=ADDRESSED, evidence=dict(n=len(r), effective_n=neff))
    c44.finding = (f"Paths are resampled from {len(r)} periods ({neff:.0f} effectively independent); "
                   "confidence intervals use the real sample length, not the number of paths.")
    out.append(c44)

    c45 = Check(45, evidence=dict(p_dd25=b["p_dd25"], p_dd50=b["p_dd50"], p_loss=b["p_loss"]))
    c45.finding = (f"Over 3 years: {b['p_dd25']:.1%} chance of a 25% drawdown, {b['p_dd50']:.1%} chance of 50%, "
                   f"{b['p_loss']:.1%} chance of ending down.")
    c45.status = FAILED if b["p_dd50"] > 0.10 else PARTIAL if b["p_dd50"] > 0.02 else ADDRESSED
    out.append(c45)
    return out, mc, curve


# --------------------------------------------------------------------------- independence / forward

def check_independence(rr, declare, src_check, ppy):
    rep = declare.get("reported", {}) or {}
    out = []
    c46 = Check(46)
    if src_check.status == ADDRESSED:
        c46.status = ADDRESSED
        c46.finding = "Audition is a separate engine, and its prices were cross-checked against a second data vendor."
    else:
        c46.status = PARTIAL
        c46.finding = ("Audition is a separate engine from whatever produced your backtest, but it uses Yahoo "
                       "data, so a shared data error isn't ruled out.")
    out.append(c46)

    n_impl = declare.get("independent_implementations")
    c47 = Check(47)
    if n_impl:
        c47.status = PARTIAL
        c47.finding = (f"You declared {n_impl} implementations. If they were built from the same written spec, "
                       "a wrong assumption in the spec survives all of them. Audition only tests the file it was given.")
    else:
        c47.status = NOT
        c47.finding = "Process question: have someone write the rules from the idea, not from your spec or code."
    out.append(c47)

    c48 = Check(48, status=ADDRESSED)
    c48.finding = ("Every return, cost and statistic here was recomputed by Audition from the strategy's raw "
                   "target weights; none came from the strategy's own code or tool.")
    out.append(c48)

    c49 = Check(49)
    diffs = []
    s, wtxt = _reported_window(rr, rep, ppy)
    for k, tol_a, tol_r in (("cagr", 0.03, 0.25), ("sharpe", 0.2, 0.2), ("max_dd", 0.03, 0.25)):
        if rep.get(k) is not None:
            ok = abs(s[k] - rep[k]) <= max(tol_a, tol_r * abs(rep[k]))
            diffs.append(dict(metric=k, reported=rep[k], audition=s[k], ok=ok))
    c49.evidence = dict(compared=diffs)
    if not diffs:
        c49.status = NOT
        c49.finding = "Add DECLARE['reported'] = {'cagr': .., 'sharpe': .., 'max_dd': ..} to compare your numbers."
    elif all(d["ok"] for d in diffs):
        c49.status = ADDRESSED
        c49.finding = ("Your reported numbers reproduce within tolerance in Audition's independent implementation" + wtxt + ": "
                       + "; ".join(f"{d['metric']} {d['reported']:.3g} vs {d['audition']:.3g}" for d in diffs) + ".")
    else:
        bad = [d for d in diffs if not d["ok"]]
        c49.status = FAILED
        c49.finding = ("Your numbers don't reproduce: " + "; ".join(
            f"{d['metric']} reported {d['reported']:.3g} vs Audition {d['audition']:.3g}" for d in bad) + wtxt +
            ". The original implementation differs from the rules in this file, or its accounting is off.")
    out.append(c49)

    c50 = Check(50, status=ADDRESSED)
    c50.finding = "A manifest with the strategy hash, data fingerprint, parameters, seeds and versions is saved with this report."
    out.append(c50)

    f32 = _sim(rr, dtype=np.float32)
    d = abs(_sharpe(f32, ppy) - s["sharpe"])
    c51 = Check(51, evidence=dict(sharpe_float32=_sharpe(f32, ppy), diff=d))
    c51.status = ADDRESSED if d < 0.01 else PARTIAL
    c51.finding = f"Re-running in 32-bit floats moves Sharpe by {d:.4f}; numeric precision doesn't matter here."
    out.append(c51)

    c52 = Check(52, status=ADDRESSED)
    c52.finding = ("This run is an independent verification: Audition's data, fills, costs and statistics share "
                   "no code with the tool the strategy was developed in.")
    out.append(c52)
    return out


def check_forward(strat, rr, declare, ppy):
    out = []
    paper = declare.get("paper_trading_days") or 0
    live = declare.get("live_trading_days") or 0
    c53 = Check(53)
    if paper or live:
        c53.status = PARTIAL
        c53.finding = (f"Declared {paper} paper and {live} live trading days. Audition can't see those results; "
                       f"at this Sharpe you need about {rr.summary['min_trl_years']:.1f} years of them to be 95% sure.")
    else:
        c53.status = NOT
        c53.finding = (f"No forward results. At this Sharpe, about {rr.summary['min_trl_years']:.1f} years of "
                       "live or paper trading are needed to confirm it with 95% confidence.")
    out.append(c53)
    for i, msg in ((54, "Compare paper fills to real fills once trading."),
                   (55, "Needs a live log of signals vs orders vs positions."),
                   (60, "Needs restart, API-failure and rejected-order drills on the live system.")):
        out.append(Check(i, status=NOT, finding="Outside a backtest. " + msg))

    rc = declare.get("risk_controls") or []
    c56 = Check(56, status=PARTIAL if rc else NOT,
                finding=(f"Declared controls: {', '.join(rc)}. Test each by tripping it on paper." if rc
                         else "No risk controls declared (position limit, daily loss limit, kill switch)."))
    out.append(c56)

    # outage drill: blank five days of data in the middle and confirm the strategy stays sane
    n = len(rr.bars.index)
    lo, hi = n // 2, n // 2 + 5
    frames = {f: getattr(rr.bars, f).copy() for f in ("open", "high", "low", "close", "volume")}
    for f in frames.values():
        f.iloc[lo:hi] = np.nan
    c57 = Check(57)
    try:
        w = strat_mod.run(strat, rr.bars.replace(**frames), rr.bench, rr.code)
        g_before = rr.w.abs().sum(axis=1).max()
        g = w.iloc[lo:hi + 10].abs().sum(axis=1).max()
        if g > max(1.5 * g_before, 1e-9) + 1e-9:
            c57.status = FAILED
            c57.finding = f"After a 5-day data blackout, gross exposure jumps to {g:.2f}x (normal max {g_before:.2f}x)."
        else:
            c57.status = PARTIAL
            c57.finding = ("Survived a simulated 5-day data blackout without errors or exposure spikes. Live "
                           "feeds fail in other ways too (stale prices, bad ticks), so test the live system.")
    except Exception as e:
        c57.status = FAILED
        c57.finding = f"The strategy crashed on a 5-day data blackout: {type(e).__name__}: {e}"
    out.append(c57)

    ex = rr.sim.returns - rr.sim.rf
    rng = np.random.default_rng(13)
    idx = montecarlo.stationary_indices(len(ex), 126, 2000, 20.0, rng)
    x = ex.to_numpy()[idx]
    sr6 = x.mean(axis=1) / x.std(axis=1) * math.sqrt(ppy)
    alarm = float(np.percentile(sr6, 5))
    last = stats.sharpe(ex.iloc[-126:], ppy)
    c58 = Check(58, status=PARTIAL, evidence=dict(alarm_6m_sharpe=alarm, last_6m=last))
    c58.finding = (f"Suggested tripwire: a live 6-month Sharpe below {alarm:.2f} happens in only 5% of 6-month "
                   f"stretches of the backtest itself, so treat it as a sign the edge has decayed. Six months is "
                   f"noisy: the backtest's own range is that wide. Last 6 months in sample: {last:.2f}.")
    out.append(c58)

    c59 = Check(59, status=NOT)
    c59.finding = (f"Your call. For sizing: annual vol {_fmt_pct(rr.summary['vol'])}, max drawdown "
                   f"{_fmt_pct(rr.summary['max_dd'])}, 3-year 50% drawdown odds in the Monte Carlo section.")
    out.append(c59)
    return out


# --------------------------------------------------------------------------- orchestration

def rr_extra_ev(rr):
    return getattr(rr, "_surv_ev", {})


def _waterfall(rr, mt_check, ppy, declare):
    steps = []
    reported = declare.get("reported") or {}
    rep = reported.get("sharpe")
    if rep is not None:
        if reported.get("start") or reported.get("end"):
            ws, _ = _reported_window(rr, reported, ppy)
            span = f"{str(reported.get('start', ''))[:4]}-{str(reported.get('end', ''))[:4]}"
            steps.append((f"You reported ({span})", rep))
            label = "Audition, same window" + (" (your list)" if getattr(rr, "reported_ref", None) else "")
            steps.append((label, ws["sharpe"]))
        else:
            steps.append(("You reported", rep))
    if getattr(rr, "pit", None) is not None and "survivor_sharpe" in rr_extra_ev(rr):
        steps.append(("Today's members, full history", rr_extra_ev(rr)["survivor_sharpe"]))
    gross_inst = _sim(rr, cm=rr.cm.scaled(cost_mult=0.0, borrow_bps_yr=0.0), fill="close", delay=0)
    steps.append(("Same-close fill, no costs" + (" (point-in-time)" if getattr(rr, "pit", None) else ""),
                  _sharpe(gross_inst, ppy)))
    gross = _sim(rr, cm=rr.cm.scaled(cost_mult=0.0, borrow_bps_yr=0.0))
    steps.append(("Next-open fill", _sharpe(gross, ppy)))
    nominal = _sim(rr, cm=rr.cm.scaled(impact=False))
    steps.append(("+ fees, spread, tax, borrow", _sharpe(nominal, ppy)))
    steps.append((f"+ impact at ${rr.cm.capital/1e6:g}M", rr.summary["sharpe"]))
    lo = rr.summary["lo_sharpe"]
    steps.append(("Autocorrelation-adjusted", lo))
    # never let a negative autocorrelation raise the audited number
    steps.append(("Selection-adjusted", min(lo, rr.summary["sharpe"]) - mt_check.evidence["sr0_annual"]))
    return steps


def _data_fingerprint(bars):
    import hashlib
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(bars.close.fillna(-1), index=True).values.tobytes())
    return h.hexdigest()[:16]


def run_audit(path, opts: Options) -> Audit:
    t0 = time.time()
    log = opts.log
    strat = strat_mod.load(path)
    log(f"strategy  {strat.name}  ({strat.path.name}, params {strat.params})")
    from . import vault
    vault_info = dict(mode=opts.mode, notes=[])
    if opts.mode in ("rehearsal", "final"):
        v = vault.open_vault(strat.name)
        if opts.mode == "rehearsal":
            opts.end = (pd.Timestamp(v["sealed_from"]) - pd.Timedelta(days=1)).date().isoformat()
            log(f"vault     rehearsal: data stops at {opts.end}; {v['sealed_from']} onward is locked")
        else:
            log(f"vault     FINAL AUDITION: unlocking {v['sealed_from']} onward"
                + (f" (retake #{len(v['finals']) + 1})" if v["finals"] else ""))
        strat.declare, vault_info["notes"] = vault.injected_declare(v, strat.declare, opts.mode == "final", strat.sha256)
        vault_info["sealed_from"] = v["sealed_from"]
    elif vault.load(strat.name):
        v = vault.record_peek(strat.name, strat.sha256)
        vault_info["notes"].append("full-history run while a vault exists: recorded as a peek at the locked period")

    plan = _region_plan(strat, opts)
    runs = {}
    for code, cfg, uni, src in plan:
        n = "all historical" if uni == "sp500" else ("today's" if uni is None else len(uni))
        log(f"region    {code:<9} {n} tickers  [{src}]")
        try:
            runs[code] = _prepare_region(strat, code, cfg, uni, src, opts)
        except Exception as e:
            log(f"          skipped: {e}")
    home_code = plan[0][0]
    if home_code not in runs:
        raise RuntimeError("couldn't load data for the home region")
    rr = runs[home_code]
    if rr.error:
        raise RuntimeError(f"strategy failed on the home region: {rr.error}")
    ppy = rr.cfg["ppy"]
    decl = strat.declare
    q = opts.quick
    if isinstance(strat.universe, (list, tuple)) and "survivors" in runs and not runs["survivors"].error:
        rr.reported_ref = runs["survivors"]

    checks = {}

    def add(*cs):
        for c in cs:
            checks[c.id] = c

    log("research  lookahead, leakage, sensitivity, placebo ...")
    add(check_lookahead(strat, rr, 6 if q else 12))
    add(check_leakage(strat, rr))
    add(*check_survivorship(rr, decl, runs.get("survivors")))
    rr._surv_ev = checks[3].evidence
    sens = sensitivity_runs(strat, rr, q)
    sc = check_sensitivity(strat, rr, sens)
    add(sc)
    mt = check_multiple_testing(rr, decl, sens)
    add(mt, check_overfitting(strat, rr, sc), check_oos(rr, decl), check_regimes(rr), check_sample(rr))
    placebo = placebo_runs(strat, rr, 10 if q else 30)
    add(check_significance(rr, placebo))

    log("data      adjustments, calendars, bad ticks, sources ...")
    src = check_source(rr)
    add(check_adjustments(rr, strat), check_alignment(rr), check_missing(rr), check_revisions(strat, rr),
        check_calendars(rr), check_bad_ticks(strat, rr), src)

    log("execution fills, impact, latency, capacity ...")
    add(check_fills(rr), *check_slippage_impact(rr, ppy), check_spread(rr), check_latency(rr, ppy),
        check_fees(rr), check_shorting(rr, ppy), check_orders(decl), check_capacity(rr, ppy, opts.capital))

    log("metrics   and monte carlo ...")
    add(*check_metrics(rr, decl, ppy))
    mc_checks, mc, cost_curve = check_montecarlo(rr, ppy, sens, q)
    add(*mc_checks)
    add(*check_independence(rr, decl, src, ppy))
    add(*check_forward(strat, rr, decl, ppy))

    log("extended  checks 61-70, full metrics, monte carlo study ...")
    from . import extended
    survivors = not checks[3].evidence.get("delisted_in_universe") and len(rr.bars.tickers) >= 5
    ext, ew = extended.run_checks(strat, rr, runs, ppy, q, Check, _sim, _sharpe, _try_run, survivors=survivors)
    add(*ext)
    bopen = getattr(rr, "bench_open", rr.bench)
    bench_r = (bopen.shift(-1) / bopen - 1).reindex(rr.sim.returns.index).fillna(0.0)
    metrics_rows, monthly_grid, yearly_r = extended.full_metrics(rr, bench_r, ew, ppy)
    mc_full = extended.mc_study(rr, ppy, sens, q)


    missing = set(CHECKS) - set(checks)
    assert not missing, f"checks not produced: {missing}"

    replication = []
    for code, r in runs.items():
        replication.append(dict(code=code, name=r.cfg["name"], source=r.universe_source, tickers=len(r.bars.tickers),
                                start=str(r.bars.index[0].date()), error=r.error, **{k: r.summary[k] for k in (
                                    "sharpe", "cagr", "max_dd", "vol")}, cost_bps=r.sim.cost_bps_per_turnover))

    extras = dict(
        waterfall=_waterfall(rr, mt, ppy, decl),
        mc=mc, cost_curve=cost_curve, metrics=metrics_rows, monthly=monthly_grid, mc_full=mc_full,
        bench_r=bench_r, ew=ew, sens=[(s["label"], s["params"], s["sharpe"]) for s in sens],
        placebo=placebo, replication=replication,
        manifest=dict(
            strategy_file=str(strat.path), strategy_sha256=strat.sha256, params=strat.params,
            declare=decl, regions=[p[0] for p in plan], capital=opts.capital, quick=q,
            data_fingerprint=_data_fingerprint(rr.bars), data_start=str(rr.bars.index[0].date()),
            data_end=str(rr.bars.index[-1].date()), missing_tickers=rr.bars.missing,
            python=sys.version.split()[0], platform=platform.platform(),
            numpy=np.__version__, pandas=pd.__version__, seeds=dict(mc=7, ci=11, placebo=9, sens=5, leak=3),
        ),
    )
    log("probes    event collision map, twin finder ...")
    from . import events, twins
    try:
        extras["events"] = events.collision_map(rr)
    except Exception as e:
        extras["events"] = dict(error=f"{type(e).__name__}: {e}")
    try:
        extras["twins"] = twins.find(rr)
    except Exception as e:
        extras["twins"] = dict(error=f"{type(e).__name__}: {e}")

    if opts.mode == "rehearsal":
        vault.record_rehearsal(strat.name, strat.sha256, rr.summary.get("sharpe"))
    elif opts.mode == "final":
        vault_info["final"] = _locked_result(rr, vault_info["sealed_from"], ppy)
        vault.record_final(strat.name, strat.sha256, vault_info["final"])
    vault_info["state"] = vault.status(strat.name)
    extras["vault"] = vault_info
    extras["manifest"]["mode"] = opts.mode
    extras["manifest"]["data_source"] = strat.data_source
    a = Audit(strategy=strat, options=opts, home=rr, regions=runs, checks=checks, extras=extras, started=t0)
    a.finished = time.time()
    return a


def _locked_result(rr, sealed_from, ppy) -> dict:
    ex = rr.sim.returns - rr.sim.rf
    cut = pd.Timestamp(sealed_from)
    pre, post = ex[ex.index < cut], ex[ex.index >= cut]
    r_post = rr.sim.returns[rr.sim.returns.index >= cut]
    sr_pre, sr_post = stats.sharpe(pre, ppy), stats.sharpe(post, ppy)
    y_pre, y_post = len(pre) / ppy, len(post) / ppy
    se = math.sqrt(1 / max(y_pre, 0.1) + 1 / max(y_post, 0.1))   # Sharpe standard error ~ 1/sqrt(years)
    z = (sr_post - sr_pre) / se if se > 0 else float("nan")
    eq = (1 + r_post).cumprod()
    return dict(sealed_from=sealed_from, periods=int(len(post)), sharpe_before=sr_pre, sharpe_locked=sr_post,
                cagr_locked=float(eq.iloc[-1] ** (ppy / max(len(r_post), 1)) - 1) if len(r_post) else float("nan"),
                max_dd_locked=float((eq / eq.cummax() - 1).min()) if len(r_post) else float("nan"),
                total_locked=float(eq.iloc[-1] - 1) if len(r_post) else float("nan"), z=z,
                passed=bool(sr_post > 0 and z > -2))


def survivor_universe(a: Audit) -> bool:
    """20+ names, none of which stopped trading: a hindsight-picked list that can manufacture a selection edge."""
    ev = a.checks[3].evidence
    if ev.get("pit"):
        return False
    return (ev.get("tickers", 0) >= 20 and not ev.get("delisted_in_universe")
            and not a.strategy.declare.get("survivorship_free"))


def grade(a: Audit):
    pts = {ADDRESSED: 1.0, PARTIAL: 0.5, FAILED: 0.0, NOT: 0.0}
    num = den = 0.0
    for c in a.checks.values():
        if c.scope == "live":
            continue
        w = 3.0 if c.id in TOP10_IDS else 1.0
        num += w * pts[c.status]
        den += w
    score = num / den if den else 0.0
    letter = "A" if score >= 0.85 else "B" if score >= 0.7 else "C" if score >= 0.55 else "D" if score >= 0.4 else "F"
    failed = [c for c in a.checks.values() if c.status == FAILED]
    top_failed = [c for c in failed if c.id in TOP10_IDS]
    if failed and letter == "A":
        letter = "B"
    if top_failed and letter in "ABC":
        letter = "D"
    if survivor_universe(a) and letter in "AB":
        letter = "C"
    # the grade is about how much the headline deserves trust: cap it by the haircut
    wf = a.extras.get("waterfall") or []
    head = next((v for k, v in wf if k.startswith("You reported")), None)
    if head is None:
        head = next((v for k, v in wf if k.startswith("Today's members")), None)
    audited = wf[-1][1] if wf else None
    if head and head > 0 and audited is not None:
        kept = audited / head
        if kept < 0.25 and letter in "ABC":
            letter = "D"
        elif kept < 0.5 and letter in "AB":
            letter = "C"
    if a.checks[1].status == FAILED or a.checks[2].status == FAILED:
        letter = "F"
    if a.home.summary["sharpe"] <= 0:  # loses to T-bills after costs: nothing to trust
        letter = "F"
    return score, letter


def top10(a: Audit):
    rows = []
    for label, ids in TOP10:
        st = worst([a.checks[i].status for i in ids])
        rows.append((label, ids, st))
    return rows
